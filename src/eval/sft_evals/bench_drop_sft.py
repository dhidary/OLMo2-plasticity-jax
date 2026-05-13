"""DROP SFT eval — matches OLMES `drop::tulu` setup.

3-shot CoT, chat-format with fewshot_as_multiturn=True, greedy decode,
F1 score against any of the gold answer spans (DROP standard metric).

Reference: allenai/OLMo-2-0425-1B-SFT model card → DROP = 33.8 (F1).
"""

from __future__ import annotations

import argparse
import os
import re
import string
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, load_esurge, write_summary
from sft_evals._sft_common import build_messages, format_prompt

from datasets import load_dataset


_NUM_SHOTS = 3
_STOP_SEQS = ["Question:", "</s>", "<|im_end|>"]


def _doc_to_text(passage: str, question: str) -> str:
    """Mirror OLMES drop format: passage + question."""
    return f"Passage: {passage}\nQuestion: {question}\nAnswer:"


def _gold_answer_strings(answers_spans: dict) -> list[str]:
    """Pick all valid gold answer strings from DROP's answer span structure."""
    spans = answers_spans.get("spans", [])
    if not spans:
        return []
    # Each "spans" entry is a list of strings (one full answer can have multiple spans).
    # OLMES treats each as an alternate gold answer for max-F1.
    out = []
    for s in spans:
        if isinstance(s, list):
            out.append(" ".join(s))
        elif isinstance(s, str):
            out.append(s)
    return [a for a in out if a]


# ---- DROP F1 (ports allenai/drop_eval normalization + bag-of-numbers metric) ----

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def _normalize(s: str) -> str:
    """OLMES drop normalize: lowercase, remove articles/punct, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(_PUNCT_TABLE)
    s = " ".join(s.split())
    return s


def _split_to_bag(s: str) -> set[str]:
    """OLMES drop: tokens form a bag; if a token is a number, normalize numerically."""
    tokens = _normalize(s).split()
    bag = set()
    for t in tokens:
        try:
            f = float(t)
            bag.add(str(int(f)) if f == int(f) else str(f))
        except ValueError:
            bag.add(t)
    return bag


def _f1(pred: str, gold: str) -> float:
    """DROP F1: bag-of-tokens precision/recall with numeric normalization."""
    p_bag = _split_to_bag(pred)
    g_bag = _split_to_bag(gold)
    if not p_bag or not g_bag:
        return float(p_bag == g_bag)
    intersection = p_bag & g_bag
    if not intersection:
        return 0.0
    precision = len(intersection) / len(p_bag)
    recall = len(intersection) / len(g_bag)
    return 2 * precision * recall / (precision + recall)


def _trim(text: str) -> str:
    cut = len(text)
    for s in _STOP_SEQS:
        j = text.find(s)
        if 0 <= j < cut:
            cut = j
    return text[:cut].strip()


def _extract_answer(text: str) -> str:
    """Extract DROP answer. SFT model usually outputs answer directly or in CoT."""
    text = _trim(text)
    if not text:
        return ""
    # 1) "The answer is X." pattern (CoT-style)
    m = re.search(r"answer is[:\s]*([^.\n]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 2) "Final answer: X" or "Therefore, X"
    m = re.search(r"(?:final answer|therefore)[:,\s]+([^.\n]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 3) First line typically holds the direct answer for non-CoT
    first = text.split("\n")[0].strip()
    if first:
        return first
    return text.strip()


def drop_sft_eval(model_path: str, max_gen_toks: int, max_model_len: int,
                  page_size: int = 32, max_num_seqs: int = 8):
    from easydel.inference.sampling_params import SamplingParams
    engine, tokenizer = load_esurge(model_path, max_model_len, page_size, max_num_seqs)

    train_ds = load_dataset("ucinlp/drop", split="train")
    test_ds = load_dataset("ucinlp/drop", split="validation")

    # Pick 3 fewshot examples deterministically (first 3 with valid answers)
    fewshot_raw = []
    for ex in train_ds:
        golds = _gold_answer_strings(ex["answers_spans"])
        if golds:
            fewshot_raw.append((_doc_to_text(ex["passage"], ex["question"]), " " + golds[0]))
        if len(fewshot_raw) >= _NUM_SHOTS:
            break
    fewshot = fewshot_raw

    print(f"  DROP SFT: {len(test_ds)} test items, {_NUM_SHOTS}-shot, "
          f"chat+multiturn, max_gen_toks={max_gen_toks}")

    prompts = []
    all_golds = []
    for ex in test_ds:
        msgs = build_messages(
            system=None, fewshot=fewshot,
            user=_doc_to_text(ex["passage"], ex["question"]),
            multiturn=True, assistant_prefix=None,
        )
        prompts.append(format_prompt(tokenizer, msgs, None))
        all_golds.append(_gold_answer_strings(ex["answers_spans"]))
    req_ids = [f"req-{i:06d}" for i in range(len(prompts))]

    sp = SamplingParams(
        max_tokens=max_gen_toks, temperature=0.0,
        stop=_STOP_SEQS, skip_special_tokens=True,
    )

    # Chunk to avoid eSurge multi-host hang on large prompt counts.
    print(f"  generating {len(prompts)} prompts (chunks of 256)...")
    by_id: dict = {}
    CHUNK = 256
    t0 = time.time()
    for ci in range(0, len(prompts), CHUNK):
        ce = min(ci + CHUNK, len(prompts))
        outs = engine.generate(prompts[ci:ce], sp, request_id=req_ids[ci:ce], use_tqdm=False)
        for o in outs:
            by_id[getattr(o, "request_id", None)] = o
        print(f"    chunk {ci}-{ce} done (cum={len(by_id)})")
    elapsed = time.time() - t0

    # Score: max F1 over all gold answers per question.
    f1_scores = []
    em_scores = []
    for i, (rid, golds) in enumerate(zip(req_ids, all_golds)):
        out = by_id.get(rid)
        if out is None or not golds:
            f1_scores.append(0.0)
            em_scores.append(0.0)
            continue
        text = out.outputs[0].text if hasattr(out, "outputs") else out.get_text()
        pred = _extract_answer(text)
        best_f1 = max(_f1(pred, g) for g in golds)
        best_em = max(int(_normalize(pred) == _normalize(g)) for g in golds)
        f1_scores.append(best_f1)
        em_scores.append(best_em)
        if i < 5:
            print(f"    [{i}] gold={golds[0]!r} pred={pred!r} f1={best_f1:.3f}")

    f1 = sum(f1_scores) / max(len(f1_scores), 1)
    em = sum(em_scores) / max(len(em_scores), 1)
    return {
        "exact_match": f1,  # primary metric for DROP is F1; alias for aggregator compat
        "f1": f1,
        "em": em,
        "total": len(prompts),
        "wall_seconds": int(elapsed),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--max_gen_toks", type=int, default=512)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--max_num_seqs", type=int, default=8)
    p.add_argument("--label", required=True)
    args = p.parse_args()

    import os, socket
    if os.environ.get("SINGLE_HOST"):
        os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
        os.environ["TPU_VISIBLE_CHIPS"] = "0,1,2,3"
        os.environ["CLOUD_TPU_TASK_ID"] = "0"
    jax.distributed.initialize()
    print(f"[{args.label}] devices={jax.device_count()} processes={jax.process_count()}")

    r = drop_sft_eval(args.model_path, args.max_gen_toks, args.max_length,
                      args.page_size, args.max_num_seqs)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total']}, {_NUM_SHOTS}-shot SFT, "
              f"wall={r['wall_seconds']}s) ===")
        print(f"  DROP-SFT F1: {r['f1']:.4f} EM: {r['em']:.4f}")
        print(f"  paper allenai/OLMo-2-0425-1B-SFT (Table 16): 0.338")
        write_summary("drop-sft", args.label, args.model_path, r["total"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
