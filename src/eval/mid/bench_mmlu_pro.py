"""MMLU Pro MC: 5-shot per category, single-letter scoring (10 letters A-J),
micro-average over 14 categories.

Matches OLMES `mmlu_pro:mc::none` suite. Each category gets fixed 5 fewshot
examples drawn from validation_docs (filtered by category). Paper Table 22
OLMo 2 1B = 0.161.
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


MMLU_PRO_CATEGORIES = [
    "math", "health", "physics", "business", "biology", "chemistry",
    "computer science", "economics", "engineering", "philosophy", "other",
    "history", "psychology", "law",
]
assert len(MMLU_PRO_CATEGORIES) == 14

_NUM_SHOTS = 5
_LETTERS = "ABCDEFGHIJ"  # MMLU Pro is 10-way MC


def _make_mcq_prompt(question: str, options) -> str:
    n = len(options)
    lines = "\n".join(f" {_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return f"Question: {question}\n{lines}\nAnswer:"


def _make_category_prefix(dev_examples) -> str:
    parts = []
    for ex in list(dev_examples)[:_NUM_SHOTS]:
        prompt = _make_mcq_prompt(ex["question"], ex["options"])
        gold_letter = _LETTERS[int(ex["answer_index"])]
        parts.append(f"{prompt} {gold_letter}")
    return "\n\n".join(parts) + "\n\n"


def mmlu_pro_eval(model, tokenizer, mesh, batch_size: int, seq_len: int):
    pad_id = tokenizer.pad_token_id

    letter_ids = []
    for L in _LETTERS:
        ids = tokenizer(" " + L, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"' {L}' tokenizes to {ids}, expected 1 token"
        letter_ids.append(ids[0])
    score_fn, params = make_letter_score_fn(model, mesh, letter_ids)

    print("  loading TIGER-Lab/MMLU-Pro...")
    val = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    test = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    by_cat_val: dict = {c: [] for c in MMLU_PRO_CATEGORIES}
    by_cat_test: dict = {c: [] for c in MMLU_PRO_CATEGORIES}
    for ex in val:
        if ex["category"] in by_cat_val:
            by_cat_val[ex["category"]].append(ex)
    for ex in test:
        if ex["category"] in by_cat_test:
            by_cat_test[ex["category"]].append(ex)

    per_cat = {}
    total_correct = 0
    total_items = 0
    t0 = time.time()

    for ci, cat in enumerate(MMLU_PRO_CATEGORIES):
        dev = by_cat_val[cat]
        tst = by_cat_test[cat]
        prefix = _make_category_prefix(dev)

        items = []
        for ex in tst:
            ctx = prefix + _make_mcq_prompt(ex["question"], ex["options"])
            ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
            if len(ids) > seq_len:
                ids = ids[-seq_len:]
            n_opts = len(ex["options"])
            items.append((ids, len(ids) - 1, int(ex["answer_index"]), n_opts))

        n_items = len(items)
        correct = 0
        for bi in range(0, n_items, batch_size):
            be = min(bi + batch_size, n_items)
            bids = np.full((batch_size, seq_len), pad_id, dtype=np.int32)
            battn = np.zeros((batch_size, seq_len), dtype=np.int32)
            bidx = np.zeros(batch_size, dtype=np.int32)
            for j, idx in enumerate(range(bi, be)):
                ids, last_idx, _, _ = items[idx]
                bids[j, :len(ids)] = ids
                battn[j, :len(ids)] = 1
                bidx[j] = last_idx
            with mesh:
                letter_logits = np.asarray(score_fn(
                    params, jnp.array(bids), jnp.array(battn), jnp.array(bidx)
                ))
            for k, idx in enumerate(range(bi, be)):
                _, _, gold, n_opts = items[idx]
                # Mask out logits for letters beyond this question's option count.
                row = letter_logits[k].copy()
                row[n_opts:] = -1e30
                if int(np.argmax(row)) == gold:
                    correct += 1

        acc = correct / max(n_items, 1)
        per_cat[cat] = acc
        total_correct += correct
        total_items += n_items
        print(f"  [{ci+1:>2}/14] {cat:<20} acc={acc:.3f} ({correct}/{n_items}) | elapsed={time.time()-t0:.0f}s")

    macro = sum(per_cat.values()) / len(per_cat)
    micro = total_correct / max(total_items, 1)
    return {
        "macro_acc": macro,
        "micro_acc": micro,
        "n_categories": len(per_cat),
        "total_items": total_items,
        "total_correct": total_correct,
        "per_category": per_cat,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=3072)
    p.add_argument("--label", required=True)
    args = p.parse_args()

    jax.distributed.initialize()
    print(f"[{args.label}] devices={jax.device_count()} processes={jax.process_count()}")

    model, tokenizer, mesh = load_easydel(args.model_path)
    if jax.process_index() == 0:
        print(f"[{args.label}] model loaded")

    r = mmlu_pro_eval(model, tokenizer, mesh, args.batch_size, args.seq_len)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total_items']}, {_NUM_SHOTS}-shot, MC) ===")
        print(f"  MMLU Pro micro_acc: {r['micro_acc']:.4f}  (macro: {r['macro_acc']:.4f})")
        print(f"  paper OLMo 2 1B post-midtrain (Table 22): 0.161")
        write_summary("mmlu-pro-mc", args.label, args.model_path, r["total_items"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
