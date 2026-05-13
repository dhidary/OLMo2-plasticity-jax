"""HellaSwag RC: 5-shot, full-text continuations, char-length-normalized acc.

Matches OLMES `hellaswag:rc::olmes`. Subset is 1000 sampled with
random.Random(1234) — the OLMES default — not first-1000.
"""

from __future__ import annotations

import argparse
import os
import random as _rnd
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, jnp, load_easydel, make_score_fn, write_summary

import numpy as np
from datasets import load_dataset


# Verbatim from allenai/olmes:oe_eval/tasks/fewshot_sources.py FEWSHOT_SOURCES["OLMES:hellaswag"].
OLMES_HELLASWAG_FEWSHOT = [
    {
        "activity_label": "Health",
        "ctx_a": "[header] How to cope with suicidal thoughts [title] Put off any plans. [step] Promise yourself that you'll wait 48 hours before doing anything. Remember, thoughts don't have the power to force you to act.",
        "ctx_b": "",
        "endings": [
            "Even when you do, there may be a small image of the future still lurking around your brain. [substeps] For instance, don't tell yourself that you can't make it.",
            "You're doing something, and no one can force you to act. It's completely natural to feel negative thoughts before you act.",
            "Do not panic if people talk to you (even if it's about quitting smoking). Have a plan for how you're going to react to a group of people who bring on suicidal thoughts.",
            "Sometimes extreme pain can distort our perception. Waiting before taking action will give your mind time to clear.",
        ],
        "label": 3,
    },
    {
        "activity_label": "Education and Communications",
        "ctx_a": "[header] How to make a liquid into a solid [title] Place a small open container of water in the freezer compartment of a class or home refrigerator. [title] Leave the water there for several hours or overnight. [title] Remove from the freezer and note what has occurred.",
        "ctx_b": "",
        "endings": [
            "[step] Water changes state from liquid to solid when it reaches a temperature of 0 degrees celsius, or 32 degrees fahrenheit. This is a simple example of changing from liquid to solid, or freezing.",
            "[substeps] Check that the container is completely dry, but no ice has formed. You should get a sample before disposing of it.",
            "[step] Don't drink and continue making liquid. [title] Separate the ice water if you're not used to using water.",
            "[title] Set a timer to check on the reaction. [step] The liquid should be safe to use again once the water has frozen completely and the food appears firm.",
        ],
        "label": 0,
    },
    {
        "activity_label": "Baking cookies",
        "ctx_a": "A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them. The pans are filled with pastries and loaded into the oven.",
        "ctx_b": "a knife",
        "endings": [
            "is seen moving on a board and cutting out its contents.",
            "hits the peeled cheesecake, followed by sliced custard and still cooked ice cream.",
            "etches a shape into the inside of the baked pans.",
            "is used to cut cylinder shaped dough into rounds.",
        ],
        "label": 3,
    },
    {
        "activity_label": "Starting a campfire",
        "ctx_a": "He takes his lighter and lights the newspaper in several places to start the fire. The bonfire starts burning and continues to burn.",
        "ctx_b": "he",
        "endings": [
            "plays with the dog and makes two cookies.",
            "adds a few more twigs to keep the flames burning.",
            "gets up and attempts to put a flag on it, fails and makes a complete ass out of himself.",
            "puts on equipment and stools.",
        ],
        "label": 1,
    },
    {
        "activity_label": "Finance and Business",
        "ctx_a": "[header] How to write a method statement [title] Prepare to write by conducting a risk assessment--an in-depth examination of the task or process. [substeps] Identify the work hazards (those that could potentially cause poor health or personal harm) that are inherent in the task. Analyze what has been done about these hazards and if these measures are enough to reduce the harm potential to an acceptable level.",
        "ctx_b": "",
        "endings": [
            "Determine if there are further steps you would like to take. For example, if you want to write about looking as though you've truly experienced the problem in practice, doing a risk assessment may help you so further in mental illness.",
            "Review the information presented to the project and get an understanding of the hazards. [title] Organize and plan a rest period that will help the sanitation industry and forest service team manage the task more effectively.",
            "Decide what additional measures need to be taken to reduce harm if an acceptable level has not been met. [title] Begin to write your method statement, starting at the header.",
            "[title] Write the search code (cnet) heading. [step] To write an article or report, simply write the following code (cnet: alternative sources and outcomes.",
        ],
        "label": 2,
    },
]

_NUM_SHOTS = 5
_LIMIT = 1000


def _olmes_preprocess(text: str) -> str:
    """Strip [title]/[bracket] markup the dataset inherited from WikiHow."""
    text = text.strip()
    text = re.sub(r"\.? \[title\]", ". ", text)
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _olmes_query(ex) -> str:
    ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
    return _olmes_preprocess(ex["activity_label"] + ": " + ctx)


def _make_fewshot_prefix(tokenizer):
    parts = []
    for ex in OLMES_HELLASWAG_FEWSHOT[:_NUM_SHOTS]:
        gold = _olmes_preprocess(ex["endings"][int(ex["label"])])
        parts.append(f"{_olmes_query(ex)} {gold}")
    prefix = "\n\n".join(parts) + "\n\n"
    return prefix, tokenizer(prefix, add_special_tokens=False)["input_ids"]


def hellaswag(model, tokenizer, mesh, batch_size: int, seq_len: int):
    ds = load_dataset("allenai/hellaswag", split="validation")
    # OLMES `hellaswag:rc::olmes`: limit=1000 with default random_subsample_seed=1234.
    ds = ds.select(_rnd.Random(1234).sample(range(len(ds)), _LIMIT))

    prefix_text, _ = _make_fewshot_prefix(tokenizer)
    pad_id = tokenizer.pad_token_id

    seqs = []
    starts = []
    lens = []
    end_chars = []
    items = []  # (start, stop, gold)

    for ex in ds:
        full_ctx = prefix_text + _olmes_query(ex)
        ctx_ids = tokenizer(full_ctx, add_special_tokens=False)["input_ids"]
        for end in ex["endings"]:
            end_text = " " + _olmes_preprocess(end)
            full = tokenizer(full_ctx + end_text, add_special_tokens=False)["input_ids"]
            tstart = max(0, len(ctx_ids) - max(0, len(full) - seq_len))
            seq = full[-seq_len:]
            seqs.append(seq)
            starts.append(tstart)
            lens.append(len(seq))
            end_chars.append(len(end_text))
        items.append((len(seqs) - 4, len(seqs), int(ex["label"])))

    n_seq = len(seqs)
    print(f"  HellaSwag RC: {len(ds)} val items -> {n_seq} forward passes; batches of {batch_size}")
    score_fn, params = make_score_fn(model, mesh)

    scores = np.zeros(n_seq, dtype=np.float32)
    t0 = time.time()
    for bi in range(0, n_seq, batch_size):
        be = min(bi + batch_size, n_seq)
        bids = np.full((batch_size, seq_len), pad_id, dtype=np.int32)
        battn = np.zeros((batch_size, seq_len), dtype=np.int32)
        btgt = np.zeros((batch_size, seq_len), dtype=np.int32)
        for j, idx in enumerate(range(bi, be)):
            slen = lens[idx]
            bids[j, :slen] = seqs[idx]
            battn[j, :slen] = 1
            btgt[j, starts[idx]:slen] = 1
        with mesh:
            lp = score_fn(params, jnp.array(bids), jnp.array(battn), jnp.array(btgt))
        scores[bi:be] = np.asarray(lp)[:be - bi]
        if bi == 0 or be == n_seq:
            print(f"    batch {bi // batch_size + 1}/{(n_seq + batch_size - 1) // batch_size} | {time.time() - t0:.0f}s")

    norm_scores = scores / np.array(end_chars, dtype=np.float32)
    correct = correct_norm = 0
    for start, stop, gold in items:
        if int(np.argmax(scores[start:stop])) == gold:
            correct += 1
        if int(np.argmax(norm_scores[start:stop])) == gold:
            correct_norm += 1
    total = len(items)
    return {
        "acc": correct / max(total, 1),
        "acc_norm": correct_norm / max(total, 1),
        "correct": correct,
        "correct_norm": correct_norm,
        "total": total,
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

    r = hellaswag(model, tokenizer, mesh, args.batch_size, args.seq_len)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={_LIMIT}, {_NUM_SHOTS}-shot, RC) ===")
        print(f"  HellaSwag acc:          {r['acc']:.4f}  ({r['correct']}/{r['total']})")
        print(f"  HellaSwag acc_per_char: {r['acc_norm']:.4f}  ({r['correct_norm']}/{r['total']})")
        print(f"  paper OLMo 2 1B post-midtrain (Table 22): 0.695")
        print(f"  paper OLMo 2 1B stage1 only baseline:     0.675")
        write_summary("hellaswag-rc", args.label, args.model_path, _LIMIT, _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
