"""MMLU SFT eval — closest reproduction of allenai/OLMo-2-0425-1B-SFT MMLU=0.364.

Empirically explored OLMES `mmlu:mc::tulu` (5-shot MC scoring) and
`mmlu:0shot_cot::tulu3` (0-shot CoT generative). Best fit: MC scoring with
v6 settings — 5-shot val fewshot, no "Answer:" suffix, no description, score
" A"/" B"/" C"/" D" continuations via loglikelihood. Lands at +1.11pt;
fully OLMES-literal MC overshoots to +6pt and CoT undershoots to -1.79pt.
The +1.11pt residual is irreducible without a deeper investigation into
eSurge/EasyDeL's chat-template tokenization vs HF/vLLM. See
results/sft/mmlu.md for the variants table.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, jnp, load_easydel, make_score_fn, write_summary
from sft_evals._sft_common import build_messages, format_prompt

import numpy as np
from datasets import load_dataset


MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]
_LETTERS = "ABCD"
_NUM_SHOTS = 5
_LABEL_FORMAT = " A."


def _make_mcq_prompt(question: str, choices: list[str]) -> str:
    """v6 prompt: question + choices, no trailing 'Answer:'."""
    lines = [f"{_LABEL_FORMAT.replace('A', _LETTERS[i])} {c}"
             for i, c in enumerate(choices)]
    return f"Question: {question}\n" + "\n".join(lines) + "\n"


def _process_doc(doc: dict) -> dict:
    return {
        "question": doc["question"],
        "choices_text": doc["choices"],
        "answer_idx": doc["answer"],
        "answer_text": _LETTERS[doc["answer"]],
    }


def mmlu_sft_eval(model_path: str, batch_size: int, seq_len: int, num_shots: int = _NUM_SHOTS):
    model, tokenizer, mesh = load_easydel(model_path, max_pos=seq_len)
    score_fn, params = make_score_fn(model, mesh)
    pad_id = tokenizer.pad_token_id

    per_subject: dict[str, float] = {}
    total_correct = 0
    total_items = 0
    t0 = time.time()

    for sub in MMLU_SUBJECTS:
        val_ds = load_dataset("cais/mmlu", sub, split="validation")
        test_ds = load_dataset("cais/mmlu", sub, split="test")
        fewshot_raw = [_process_doc(d) for d in list(val_ds)[:num_shots]]
        fewshot = [
            (_make_mcq_prompt(ex["question"], ex["choices_text"]),
             " " + ex["answer_text"])
            for ex in fewshot_raw
        ]

        triples: list[tuple[int, int, str, str]] = []
        golds: list[int] = []
        for di, doc_raw in enumerate(test_ds):
            doc = _process_doc(doc_raw)
            user_text = _make_mcq_prompt(doc["question"], doc["choices_text"])
            msgs = build_messages(
                system=None, fewshot=fewshot, user=user_text,
                multiturn=True, assistant_prefix=None,
            )
            ctx = format_prompt(tokenizer, msgs, None)
            for ci, letter in enumerate(_LETTERS):
                triples.append((di, ci, ctx, " " + letter))
            golds.append(doc["answer_idx"])

        n_seq = len(triples)
        scores = np.zeros(n_seq, dtype=np.float32)
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

        n_docs = len(test_ds)
        score_matrix = scores.reshape(n_docs, 4)
        preds = np.argmax(score_matrix, axis=1)
        sub_correct = int((preds == np.asarray(golds)).sum())
        sub_acc = sub_correct / max(n_docs, 1)
        per_subject[sub] = sub_acc
        total_correct += sub_correct
        total_items += n_docs
        print(f"  [{sub}] n={n_docs} acc={sub_acc:.4f}")

    elapsed = time.time() - t0
    macro = sum(per_subject.values()) / len(per_subject)
    micro = total_correct / total_items
    return {
        "macro_acc": macro,
        "micro_acc": micro,
        "n_subjects": len(per_subject),
        "total_items": total_items,
        "total_correct": total_correct,
        "per_subject": per_subject,
        "wall_seconds": int(elapsed),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=4096)  # OLMo 2 1B model max_position_embeddings cap
    # Generative-style args (ignored; kept for launch_bench.sh compatibility).
    p.add_argument("--max_gen_toks", type=int, default=0)
    p.add_argument("--max_length", type=int, default=0)
    p.add_argument("--page_size", type=int, default=0)
    p.add_argument("--max_num_seqs", type=int, default=0)
    p.add_argument("--label", required=True)
    p.add_argument("--num_shots", type=int, default=_NUM_SHOTS)
    args = p.parse_args()

    import os, socket
    if os.environ.get("SINGLE_HOST"):
        os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
        os.environ["TPU_VISIBLE_CHIPS"] = "0,1,2,3"
        os.environ["CLOUD_TPU_TASK_ID"] = "0"
    jax.distributed.initialize()
    print(f"[{args.label}] devices={jax.device_count()} processes={jax.process_count()}")

    r = mmlu_sft_eval(args.model_path, args.batch_size, args.seq_len, args.num_shots)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total_items']}, "
              f"{args.num_shots}-shot MC SFT, wall={r['wall_seconds']}s) ===")
        print(f"  MMLU-SFT macro_acc: {r['macro_acc']:.4f} micro: {r['micro_acc']:.4f}")
        print(f"  paper allenai/OLMo-2-0425-1B-SFT (Table 16): 0.364")
        write_summary("mmlu-cot-sft", args.label, args.model_path,
                      r["total_items"], args.num_shots, r)


if __name__ == "__main__":
    main()
