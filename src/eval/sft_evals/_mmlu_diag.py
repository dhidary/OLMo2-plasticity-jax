"""Diagnostic for MMLU MC chat-template + tokenization.

Builds the v6 prompt for one MMLU example and dumps:
  - rendered chat-template string
  - token IDs for ctx and ctx+cont (each of 4 letters)
  - sum_logits for each letter
  - argmax + gold

Run on TPU: python src/eval/sft_evals/_mmlu_diag.py --model_path <path>
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, jnp, load_easydel, make_score_fn
from sft_evals._sft_common import build_messages, format_prompt
from sft_evals.bench_mmlu_sft import (
    _LETTERS, _make_mcq_prompt, _process_doc, _description_for,
)

import numpy as np
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--subject", type=str, default="abstract_algebra")
    p.add_argument("--with_answer", action="store_true",
                   help="Append 'Answer:' to user msg (test v4-style prompt)")
    p.add_argument("--with_desc", action="store_true",
                   help="Add per-subject description (test v7-style prompt)")
    p.add_argument("--use_dev_fewshot", action="store_true",
                   help="Use dev split for fewshot (test v7-style)")
    args = p.parse_args()

    jax.distributed.initialize()
    is_lead = jax.process_index() == 0

    def log(*a):
        if is_lead:
            print(*a, flush=True)

    log(f"Loading model from {args.model_path}...")
    model, tokenizer, mesh = load_easydel(args.model_path, max_pos=args.seq_len)
    score_fn, params = make_score_fn(model, mesh)
    pad_id = tokenizer.pad_token_id

    log("Chat template (first 200 chars):")
    log(repr((tokenizer.chat_template or "")[:200]))
    log()

    # Pull fewshot + first test doc
    sub = args.subject
    fewshot_split = "dev" if args.use_dev_fewshot else "validation"
    fs_ds = load_dataset("cais/mmlu", sub, split=fewshot_split)
    test_ds = load_dataset("cais/mmlu", sub, split="test")
    fewshot_raw = [_process_doc(d) for d in list(fs_ds)[:5]]
    test_doc = _process_doc(test_ds[0])

    def _make(question, choices):
        lines = [f" {_LETTERS[i]}. {c}" for i, c in enumerate(choices)]
        suffix = "\nAnswer:" if args.with_answer else "\n"
        return f"Question: {question}\n" + "\n".join(lines) + suffix

    fewshot = [(_make(ex["question"], ex["choices_text"]),
                " " + ex["answer_text"])
               for ex in fewshot_raw]
    user_text = _make(test_doc["question"], test_doc["choices_text"])
    description = _description_for(sub) if args.with_desc else None

    msgs = build_messages(
        system=None, fewshot=fewshot, user=user_text,
        multiturn=True, description=description, assistant_prefix=None,
    )
    ctx = format_prompt(tokenizer, msgs, None)

    log(f"=== CTX (len={len(ctx)} chars) ===")
    log(ctx[-400:])
    log()

    ctx_ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
    log(f"CTX tokens: {len(ctx_ids)} | last 10 IDs: {ctx_ids[-10:]}")
    log(f"Last 10 decoded: {[tokenizer.decode([t]) for t in ctx_ids[-10:]]}")
    log()

    # Score 4 continuations in a single batched call (pad to 16 for fsdp).
    log("Scoring 4 letter continuations:")
    BS = 16
    bids = np.full((BS, args.seq_len), pad_id, dtype=np.int32)
    battn = np.zeros((BS, args.seq_len), dtype=np.int32)
    btgt = np.zeros((BS, args.seq_len), dtype=np.int32)
    decoded = []
    for li, letter in enumerate(_LETTERS):
        cont = " " + letter
        full = tokenizer(ctx + cont, add_special_tokens=False)["input_ids"]
        added = full[len(ctx_ids):]
        decoded.append((letter, added))
        slen = len(full)
        tstart = min(len(ctx_ids), slen)
        bids[li, :slen] = full
        battn[li, :slen] = 1
        btgt[li, tstart:slen] = 1
    with mesh:
        lp = score_fn(params, jnp.array(bids), jnp.array(battn), jnp.array(btgt))
    scores = [float(x) for x in np.asarray(lp)[:4]]
    for (letter, added), s in zip(decoded, scores):
        log(f"  ' {letter}' → +{len(added)} tokens: ids={added} "
            f"decoded={[tokenizer.decode([t]) for t in added]}  sum_logits={s:.4f}")

    pred_idx = int(np.argmax(scores))
    gold_idx = test_doc["answer_idx"]
    log(f"\nPred: {_LETTERS[pred_idx]} (idx {pred_idx}, score {scores[pred_idx]:.4f})")
    log(f"Gold: {_LETTERS[gold_idx]} (idx {gold_idx}, score {scores[gold_idx]:.4f})")
    log(f"Match: {pred_idx == gold_idx}")


if __name__ == "__main__":
    main()
