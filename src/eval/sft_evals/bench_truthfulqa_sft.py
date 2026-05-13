"""TruthfulQA SFT eval — matches OLMES `truthfulqa::tulu` byte-perfectly.

6-shot, chat-format, single-turn fewshot (default `fewshot_as_multiturn=False`),
short_prefix=True (Q:/A:). Loglikelihood scoring per choice using the existing
make_score_fn JIT path. Primary metric: MC2 (sum_softmax over true mc2 entries).

Reference: allenai/OLMo-2-0425-1B-SFT model card → TruthQA = 42.1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, jnp, load_easydel, make_score_fn, write_summary
from sft_evals._sft_common import build_messages, format_prompt

import numpy as np
from datasets import load_dataset


_NUM_SHOTS = 6


def _data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sft")


def _load_fewshot() -> list[dict]:
    with open(os.path.join(_data_dir(), "truthfulqa_6shot.json"),
              encoding="utf-8", errors="surrogatepass") as f:
        return json.load(f)


def _doc_to_text(question: str) -> str:
    """short_prefix=True per truthfulqa::tulu config."""
    return f"Q: {question}\nA:"


def _doc_to_target_first(choices: list[str]) -> str:
    """OLMES doc_to_target: " " + choices[0] (correct mc1 first)."""
    return " " + choices[0]


def _process_doc(doc: dict, idx: int) -> dict:
    """Mirror TruthfulQA._process_doc."""
    if "choices" in doc and "labels" in doc and "mc1_indices" in doc:
        return doc
    choices = list(doc["mc1_targets"]["choices"])
    labels = list(doc["mc1_targets"]["labels"])
    mc1_indices = list(range(len(choices)))
    mc2_indices: list[int] = []
    for choice, label in zip(doc["mc2_targets"]["choices"], doc["mc2_targets"]["labels"]):
        if choice not in choices:
            choices.append(choice)
            labels.append(label)
            mc2_indices.append(len(choices) - 1)
        else:
            i = choices.index(choice)
            assert labels[i] == label, "Mismatching mc1/mc2 labels"
            mc2_indices.append(i)
    return {
        "index": idx,
        "question": doc["question"],
        "choices": choices,
        "labels": labels,
        "mc1_indices": mc1_indices,
        "mc2_indices": mc2_indices,
    }


def _build_fewshot_pairs(fewshot_raw: list[dict]) -> list[tuple[str, str]]:
    out = []
    for ex in fewshot_raw:
        u = _doc_to_text(ex["question"])
        a = _doc_to_target_first(ex["choices"])  # all 6 fewshot have one true choice
        out.append((u, a))
    return out


def _mc1_mc2_score(sum_logits: list[float], labels: list[int],
                    mc1_indices: list[int], mc2_indices: list[int]) -> tuple[int, float]:
    """Port of MC1MC2Accuracy.process_one_doc (oe_eval/metrics/metric.py:352-378)."""
    max_mc1 = -math.inf
    acc_mc1 = None
    probs_true = 0.0
    probs_false = 0.0
    for idx, sl in enumerate(sum_logits):
        if idx in mc1_indices:
            if sl > max_mc1:
                max_mc1 = sl
                acc_mc1 = labels[idx]
        if idx in mc2_indices:
            if labels[idx]:
                probs_true += math.exp(sl)
            else:
                probs_false += math.exp(sl)
    score_mc2 = probs_true / (probs_true + probs_false) if (probs_true + probs_false) > 0 else 0.0
    return int(bool(acc_mc1)), score_mc2


def truthfulqa_sft_eval(model_path: str, batch_size: int, seq_len: int):
    model, tokenizer, mesh = load_easydel(model_path, max_pos=seq_len)
    score_fn, params = make_score_fn(model, mesh)
    pad_id = tokenizer.pad_token_id

    fewshot_raw = _load_fewshot()[:_NUM_SHOTS]
    fewshot = _build_fewshot_pairs(fewshot_raw)

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                      split="validation")
    docs = [_process_doc(d, i) for i, d in enumerate(ds)]
    print(f"  TruthfulQA SFT: {len(docs)} docs, {_NUM_SHOTS}-shot, "
          f"loglikelihood scoring; total choices = "
          f"{sum(len(d['choices']) for d in docs)}")

    # Build all (doc_idx, choice_idx, prompt_str, continuation_str) triples.
    triples: list[tuple[int, int, str, str]] = []
    for di, doc in enumerate(docs):
        msgs = build_messages(
            system=None, fewshot=fewshot,
            user=_doc_to_text(doc["question"]),
            multiturn=False, assistant_prefix=None,
        )
        ctx = format_prompt(tokenizer, msgs, None)
        for ci, choice in enumerate(doc["choices"]):
            triples.append((di, ci, ctx, " " + choice))

    n_seq = len(triples)
    scores = np.zeros(n_seq, dtype=np.float32)
    t0 = time.time()

    for bi in range(0, n_seq, batch_size):
        be = min(bi + batch_size, n_seq)
        bids = np.full((batch_size, seq_len), pad_id, dtype=np.int32)
        battn = np.zeros((batch_size, seq_len), dtype=np.int32)
        btgt = np.zeros((batch_size, seq_len), dtype=np.int32)
        for j, idx in enumerate(range(bi, be)):
            _, _, ctx, cont = triples[idx]
            full = tokenizer(ctx + cont, add_special_tokens=False)["input_ids"]
            ctx_ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
            if len(full) > seq_len:
                full = full[-seq_len:]
                ctx_ids = ctx_ids[-seq_len:]
            slen = len(full)
            tstart = min(len(ctx_ids), slen)
            bids[j, :slen] = full
            battn[j, :slen] = 1
            btgt[j, tstart:slen] = 1
        with mesh:
            lp = score_fn(params, jnp.array(bids), jnp.array(battn), jnp.array(btgt))
        scores[bi:be] = np.asarray(lp)[:be - bi]
        if bi == 0 or be == n_seq:
            print(f"    batch {bi // batch_size + 1}/"
                  f"{(n_seq + batch_size - 1) // batch_size} | {time.time() - t0:.0f}s")

    elapsed = time.time() - t0

    # Group by doc, compute MC1/MC2 per doc.
    by_doc: dict[int, list[float]] = {}
    for (di, ci, _, _), s in zip(triples, scores):
        by_doc.setdefault(di, [0.0] * len(docs[di]["choices"]))[ci] = float(s)

    mc1_correct = 0
    mc2_sum = 0.0
    for di, doc in enumerate(docs):
        m1, m2 = _mc1_mc2_score(by_doc[di], doc["labels"],
                                  doc["mc1_indices"], doc["mc2_indices"])
        mc1_correct += m1
        mc2_sum += m2
    mc1 = mc1_correct / len(docs)
    mc2 = mc2_sum / len(docs)
    return {
        "mc1": mc1,
        "mc2": mc2,
        "n": len(docs),
        "wall_seconds": int(elapsed),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--label", required=True)
    args = p.parse_args()

    import os, socket
    if os.environ.get("SINGLE_HOST"):
        os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
        os.environ["TPU_VISIBLE_CHIPS"] = "0,1,2,3"
        os.environ["CLOUD_TPU_TASK_ID"] = "0"
    jax.distributed.initialize()
    print(f"[{args.label}] devices={jax.device_count()} processes={jax.process_count()}")

    r = truthfulqa_sft_eval(args.model_path, args.batch_size, args.seq_len)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['n']}, {_NUM_SHOTS}-shot SFT, "
              f"wall={r['wall_seconds']}s) ===")
        print(f"  TruthfulQA-SFT mc2: {r['mc2']:.4f}  mc1: {r['mc1']:.4f}")
        print(f"  paper allenai/OLMo-2-0425-1B-SFT (Table 16, mc2): 0.421")
        write_summary("truthfulqa-sft", args.label, args.model_path,
                      r["n"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
