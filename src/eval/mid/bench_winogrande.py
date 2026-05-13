"""WinoGrande RC: 5-shot, score continuation after the blank, acc_raw.

Matches OLMES `winogrande:rc::olmes`. Paper Table 22 0.665 is the suite max
(mc_or_rc); for OLMo 2 1B post-midtrain RC is the higher of the two and what
the paper conventionally reports for WG.

Per item: split sentence at `_`. For each option, prompt = sentence[:blank] +
option, continuation = " " + sentence[blank+1:].strip(). Sum log-probs over
continuation tokens; argmax.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, jnp, load_easydel, make_score_fn, write_summary

import numpy as np
from datasets import load_dataset


# Verbatim from allenai/olmes:oe_eval/tasks/fewshot_sources.py FEWSHOT_SOURCES["OLMES:winogrande"].
OLMES_WG_FEWSHOT = [
    {"sentence": "John moved the couch from the garage to the backyard to create space. The _ is small.",
     "option1": "garage", "option2": "backyard", "answer": "1"},
    {"sentence": "Dennis drew up a business proposal to present to Logan because _ wants his investment.",
     "option1": "Dennis", "option2": "Logan", "answer": "1"},
    {"sentence": "Felicia unexpectedly made fried eggs for breakfast in the morning for Katrina and now _ owes a favor.",
     "option1": "Felicia", "option2": "Katrina", "answer": "2"},
    {"sentence": "The circuit failed to power the television but kept the radio going, as the _ had a weak connection.",
     "option1": "television", "option2": "radio", "answer": "1"},
    {"sentence": "Neil told Craig that he has to take care of the child for the day because _ promised to do so.",
     "option1": "Neil", "option2": "Craig", "answer": "2"},
]

_NUM_SHOTS = 5


def _split_at_blank(sentence: str):
    blank = sentence.index("_")
    pre = sentence[:blank]
    post = sentence[blank + 1:]  # may start with space
    return pre, post


def _shot_text(ex) -> str:
    """Few-shot example: full gold-substituted sentence."""
    pre, post = _split_at_blank(ex["sentence"])
    gold = ex["option1"] if ex["answer"] == "1" else ex["option2"]
    return (pre + gold + post).rstrip()


def _make_fewshot_prefix() -> str:
    parts = [_shot_text(ex) for ex in OLMES_WG_FEWSHOT[:_NUM_SHOTS]]
    return "\n\n".join(parts) + "\n\n"


def winogrande_rc(model, tokenizer, mesh, batch_size: int, seq_len: int):
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    prefix_text = _make_fewshot_prefix()
    pad_id = tokenizer.pad_token_id
    print(f"  WinoGrande RC: {len(ds)} val items, {_NUM_SHOTS}-shot")

    def _build(ctx, end):
        full = tokenizer(ctx + end, add_special_tokens=False)["input_ids"]
        ctx_ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
        if len(full) > seq_len:
            full = full[-seq_len:]
            ctx_ids = ctx_ids[-seq_len:]
        return full, min(len(ctx_ids), len(full)), len(full)

    seqs = []
    items = []  # (start, stop, gold)
    for ex in ds:
        pre, post = _split_at_blank(ex["sentence"])
        # OLMES: prompt = pre + option, continuation = " " + post.strip()
        cont = " " + post.strip()
        cs = len(seqs)
        for opt in (ex["option1"], ex["option2"]):
            seqs.append(_build(prefix_text + pre + opt, cont))
        items.append((cs, len(seqs), int(ex["answer"]) - 1))

    score_fn, params = make_score_fn(model, mesh)
    n_seq = len(seqs)
    scores = np.zeros(n_seq, dtype=np.float32)
    t0 = time.time()
    for bi in range(0, n_seq, batch_size):
        be = min(bi + batch_size, n_seq)
        bids = np.full((batch_size, seq_len), pad_id, dtype=np.int32)
        battn = np.zeros((batch_size, seq_len), dtype=np.int32)
        btgt = np.zeros((batch_size, seq_len), dtype=np.int32)
        for j, idx in enumerate(range(bi, be)):
            seq, tstart, slen = seqs[idx]
            bids[j, :slen] = seq
            battn[j, :slen] = 1
            btgt[j, tstart:slen] = 1
        with mesh:
            lp = score_fn(params, jnp.array(bids), jnp.array(battn), jnp.array(btgt))
        scores[bi:be] = np.asarray(lp)[:be - bi]
        if bi == 0 or be == n_seq:
            print(f"    batch {bi // batch_size + 1}/{(n_seq + batch_size - 1) // batch_size} | {time.time() - t0:.0f}s")

    correct = sum(int(np.argmax(scores[cs:ce]) == gold) for cs, ce, gold in items)
    return {"acc_raw": correct / max(len(items), 1), "correct_raw": correct, "total": len(items)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--label", required=True)
    args = p.parse_args()

    jax.distributed.initialize()
    print(f"[{args.label}] devices={jax.device_count()} processes={jax.process_count()}")

    model, tokenizer, mesh = load_easydel(args.model_path)
    if jax.process_index() == 0:
        print(f"[{args.label}] model loaded")

    r = winogrande_rc(model, tokenizer, mesh, args.batch_size, args.seq_len)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total']}, {_NUM_SHOTS}-shot, RC) ===")
        print(f"  WinoGrande acc_raw: {r['acc_raw']:.4f}  ({r['correct_raw']}/{r['total']})")
        print(f"  paper OLMo 2 1B post-midtrain (Table 22): 0.665")
        write_summary("winogrande-rc", args.label, args.model_path, r["total"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
