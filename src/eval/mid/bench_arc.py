"""ARC-Challenge MC: 5-shot, single-letter scoring, acc_raw.

Matches OLMES `arc_challenge:mc::olmes`. Paper Table 22 reports `mc_or_rc` suite
max; for stage2 checkpoints MC dominates.
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


# Verbatim from allenai/olmes:oe_eval/tasks/fewshot_sources.py FEWSHOT_SOURCES["OLMES:ARC-Challenge"].
OLMES_ARC_CHALLENGE_FEWSHOT = [
    {
        "question": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
        "choices": ["dry palms", "wet palms", "palms covered with oil", "palms covered with lotion"],
        "answerKey": "A",
    },
    {
        "question": "Which of the following statements best explains why magnets usually stick to a refrigerator door?",
        "choices": [
            "The refrigerator door is smooth.",
            "The refrigerator door contains iron.",
            "The refrigerator door is a good conductor.",
            "The refrigerator door has electric wires in it.",
        ],
        "answerKey": "B",
    },
    {
        "question": "A fold observed in layers of sedimentary rock most likely resulted from the",
        "choices": [
            "cooling of flowing magma.",
            "converging of crustal plates.",
            "deposition of river sediments.",
            "solution of carbonate minerals.",
        ],
        "answerKey": "B",
    },
    {
        "question": "Which of these do scientists offer as the most recent explanation as to why many plants and animals died out at the end of the Mesozoic era?",
        "choices": [
            "worldwide disease",
            "global mountain building",
            "rise of mammals that preyed upon plants and animals",
            "impact of an asteroid created dust that blocked the sunlight",
        ],
        "answerKey": "D",
    },
    {
        "question": "Which of the following is a trait that a dog does NOT inherit from its parents?",
        "choices": [
            "the length of its fur",
            "the shape of its nose",
            "the size of its appetite",
            "the color of its fur",
        ],
        "answerKey": "C",
    },
]

_NUM_SHOTS = 5
_LETTERS = "ABCDE"


def _gold_index(answer_key: str) -> int:
    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    return _LETTERS.index(num_to_letter.get(answer_key, answer_key))


def _make_mcq_prompt(question: str, choices: list[str]) -> str:
    """OLMES make_mcq_prompt: leading space before each letter."""
    lines = "\n".join(f" {_LETTERS[i]}. {ch}" for i, ch in enumerate(choices))
    return f"Question: {question}\n{lines}\nAnswer:"


def _make_fewshot_prefix() -> str:
    parts = []
    for ex in OLMES_ARC_CHALLENGE_FEWSHOT[:_NUM_SHOTS]:
        prompt = _make_mcq_prompt(ex["question"], ex["choices"])
        parts.append(f"{prompt} {_LETTERS[_gold_index(ex['answerKey'])]}")
    return "\n\n".join(parts) + "\n\n"


def arc_challenge_mc(model, tokenizer, mesh, batch_size: int, seq_len: int):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    prefix_text = _make_fewshot_prefix()
    pad_id = tokenizer.pad_token_id
    print(f"  ARC-C MC: {len(ds)} test items, {_NUM_SHOTS}-shot")

    # Single-forward MC: one forward per item, read logits at the last content
    # position for the K letter tokens (one for each choice), argmax.
    letter_ids = []
    for L in _LETTERS:
        ids = tokenizer(" " + L, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"' {L}' tokenizes to {ids}, expected 1 token"
        letter_ids.append(ids[0])

    items = []  # (input_ids, last_idx, gold, n_ch)
    for ex in ds:
        choices = ex["choices"]["text"]
        ctx = prefix_text + _make_mcq_prompt(ex["question"], choices)
        ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
        if len(ids) > seq_len:
            ids = ids[-seq_len:]
        items.append((ids, len(ids) - 1, _gold_index(ex["answerKey"]), len(choices)))

    score_fn, params = make_letter_score_fn(model, mesh, letter_ids)
    n_items = len(items)
    correct = 0
    t0 = time.time()
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
            _, _, gold, n_ch = items[idx]
            if int(np.argmax(letter_logits[k, :n_ch])) == gold:
                correct += 1
        if bi == 0 or be == n_items:
            print(f"    batch {bi // batch_size + 1}/{(n_items + batch_size - 1) // batch_size} | {time.time() - t0:.0f}s")

    return {"acc_raw": correct / max(n_items, 1), "correct_raw": correct, "total": n_items}


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

    r = arc_challenge_mc(model, tokenizer, mesh, args.batch_size, args.seq_len)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total']}, {_NUM_SHOTS}-shot, MC) ===")
        print(f"  ARC-C acc_raw: {r['acc_raw']:.4f}  ({r['correct_raw']}/{r['total']})")
        print(f"  paper OLMo 2 1B post-midtrain (Table 22): 0.513")
        print(f"  paper OLMo 2 1B stage1 only baseline:     0.261")
        write_summary("arc-c-mc", args.label, args.model_path, r["total"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
