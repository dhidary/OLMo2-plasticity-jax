"""MMLU MC: 5-shot per subject, single-letter scoring, macro-average over 57 subjects.

Matches OLMES `mmlu:mc::olmes`. Each subject gets its own preamble and 5 dev shots
(fixed order). Paper Table 22 reports `mc_or_rc` suite max — for OLMo 2 1B post-
midtrain that's ~0.443; pretrain-only is ~0.269 (near random).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, jnp, load_easydel, make_letter_score_fn, write_summary

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
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]
assert len(MMLU_SUBJECTS) == 57

_NUM_SHOTS = 5
_LETTERS = "ABCD"


def _make_mcq_prompt(question: str, choices) -> str:
    """OLMES make_mcq_prompt with default ' A.' label format."""
    lines = "\n".join(f" {_LETTERS[i]}. {ch}" for i, ch in enumerate(choices))
    return f"Question: {question}\n{lines}\nAnswer:"


def _make_subject_prefix(subject: str, dev_examples) -> str:
    """Per-subject preamble + 5 dev shots in fixed order."""
    pretty = subject.replace("_", " ")
    preamble = f"The following are multiple choice questions (with answers) about {pretty}.\n\n"
    parts = []
    for ex in list(dev_examples)[:_NUM_SHOTS]:
        prompt = _make_mcq_prompt(ex["question"], ex["choices"])
        gold_letter = _LETTERS[int(ex["answer"])]
        parts.append(f"{prompt} {gold_letter}")
    return preamble + "\n\n".join(parts) + "\n\n"


def mmlu_eval(model, tokenizer, mesh, batch_size: int, seq_len: int):
    pad_id = tokenizer.pad_token_id

    # Single-forward MC scoring: one forward per item, read 4 letter logits.
    letter_ids = []
    for L in _LETTERS:
        ids = tokenizer(" " + L, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"' {L}' tokenizes to {ids}, expected 1 token"
        letter_ids.append(ids[0])
    score_fn, params = make_letter_score_fn(model, mesh, letter_ids)

    # Load all 57 subjects at once via the "all" config — avoids HF rate limits
    # (one download per host vs 57 × 2 splits × 4 workers = 456 API calls).
    print("  loading cais/mmlu (all 57 subjects in one dataset)...")
    all_dev = load_dataset("cais/mmlu", "all", split="dev")
    all_test = load_dataset("cais/mmlu", "all", split="test")
    by_subject_dev = {s: [] for s in MMLU_SUBJECTS}
    by_subject_test = {s: [] for s in MMLU_SUBJECTS}
    for ex in all_dev:
        by_subject_dev[ex["subject"]].append(ex)
    for ex in all_test:
        by_subject_test[ex["subject"]].append(ex)

    per_subject = {}
    total_correct = 0
    total_items = 0
    t0 = time.time()

    for subj_idx, subject in enumerate(MMLU_SUBJECTS):
        dev = by_subject_dev[subject]
        test = by_subject_test[subject]
        prefix = _make_subject_prefix(subject, dev)

        items = []  # (input_ids, last_idx, gold)
        for ex in test:
            ctx = prefix + _make_mcq_prompt(ex["question"], ex["choices"])
            ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
            if len(ids) > seq_len:
                ids = ids[-seq_len:]
            items.append((ids, len(ids) - 1, int(ex["answer"])))

        n_items = len(items)
        correct = 0
        for bi in range(0, n_items, batch_size):
            be = min(bi + batch_size, n_items)
            bids = np.full((batch_size, seq_len), pad_id, dtype=np.int32)
            battn = np.zeros((batch_size, seq_len), dtype=np.int32)
            bidx = np.zeros(batch_size, dtype=np.int32)
            for j, idx in enumerate(range(bi, be)):
                ids, last_idx, _ = items[idx]
                bids[j, :len(ids)] = ids
                battn[j, :len(ids)] = 1
                bidx[j] = last_idx
            with mesh:
                letter_logits = np.asarray(score_fn(
                    params, jnp.array(bids), jnp.array(battn), jnp.array(bidx)
                ))
            for k, idx in enumerate(range(bi, be)):
                _, _, gold = items[idx]
                if int(np.argmax(letter_logits[k])) == gold:
                    correct += 1

        acc = correct / max(n_items, 1)
        per_subject[subject] = acc
        total_correct += correct
        total_items += n_items
        print(f"  [{subj_idx+1:>2}/57] {subject:<38} acc={acc:.3f} ({correct}/{n_items}) | elapsed={time.time()-t0:.0f}s")

    macro = sum(per_subject.values()) / len(per_subject)
    micro = total_correct / max(total_items, 1)
    return {
        "macro_acc": macro,
        "micro_acc": micro,
        "n_subjects": len(per_subject),
        "total_items": total_items,
        "total_correct": total_correct,
        "per_subject": per_subject,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--label", required=True)
    args = p.parse_args()

    jax.distributed.initialize()
    print(f"[{args.label}] devices={jax.device_count()} processes={jax.process_count()}")

    model, tokenizer, mesh = load_easydel(args.model_path)
    if jax.process_index() == 0:
        print(f"[{args.label}] model loaded")

    r = mmlu_eval(model, tokenizer, mesh, args.batch_size, args.seq_len)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total_items']}, {_NUM_SHOTS}-shot, MC) ===")
        print(f"  MMLU macro_acc: {r['macro_acc']:.4f}  (micro: {r['micro_acc']:.4f})")
        print(f"  paper OLMo 2 1B post-midtrain (Table 22): 0.443")
        print(f"  paper OLMo 2 1B stage1 only baseline:     0.269")
        write_summary("mmlu-mc", args.label, args.model_path, r["total_items"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
