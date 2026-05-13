"""AGIEval English: 8 tasks, MC variant, single-letter scoring, macro-averaged.

Matches OLMES `agi_eval_english::olmes` suite. Suite excludes sat-en-without-passage.
Per-task num_shots: 3 for most, 5 for sat-math/aqua-rat. Paper Table 22 OLMo 2 1B = 0.363.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, jnp, load_easydel, make_letter_score_fn, write_summary

import numpy as np


AGI_EVAL_TASKS = [
    "lsat-ar", "lsat-lr", "lsat-rc", "logiqa-en",
    "sat-math", "sat-en", "aqua-rat", "gaokao-english",
]

_NUM_SHOTS = {"sat-math": 5, "aqua-rat": 5}  # default 3 for the rest
_LETTERS = "ABCDE"


def _data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "agi_eval")


def _load_jsonl(path: str):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _strip_choice_prefix(choice: str) -> str:
    return re.sub(r"^\s*\([A-E]\)\s*|^\s*[A-E][.?]?\s*", "", choice)


def _clean_text(text: str) -> str:
    text = re.sub(r"(\([A-G]\))(\w)", r"\1 \2", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def _build_query(item) -> tuple[str, int, int]:
    """Return (query_text, gold_idx, n_options)."""
    options = [_strip_choice_prefix(c) for c in item["options"]]
    n = len(options)
    passage = item.get("passage")
    qp = f"{passage}\nQuestion: " if passage else "Question: "
    lines = "\n".join(f" {_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    query = f"{qp}{item['question']}\n{lines}\nAnswer:"
    query = _clean_text(query)
    gold = _LETTERS.index(item["label"])
    return query, gold, n


def _build_fewshot_prefix(fewshot_items, k: int) -> str:
    parts = []
    for ex in fewshot_items[:k]:
        q, gold, _ = _build_query(ex)
        parts.append(f"{q} {_LETTERS[gold]}")
    return "\n\n".join(parts) + "\n\n"


def agieval_eval(model, tokenizer, mesh, batch_size: int, seq_len: int):
    pad_id = tokenizer.pad_token_id

    letter_ids = []
    for L in _LETTERS:
        ids = tokenizer(" " + L, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"' {L}' tokenizes to {ids}"
        letter_ids.append(ids[0])
    score_fn, params = make_letter_score_fn(model, mesh, letter_ids)

    fewshot_path = os.path.join(_data_dir(), "fewshot.json")
    with open(fewshot_path) as f:
        fewshot_all = json.load(f)

    per_task = {}
    total_correct = 0
    total_items = 0
    t0 = time.time()

    for ti, task in enumerate(AGI_EVAL_TASKS):
        k = _NUM_SHOTS.get(task, 3)
        prefix = _build_fewshot_prefix(fewshot_all[task], k)
        items_raw = _load_jsonl(os.path.join(_data_dir(), f"{task}.jsonl"))

        items = []
        for ex in items_raw:
            ctx, gold, n_opts = _build_query(ex)
            full = prefix + ctx
            ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            if len(ids) > seq_len:
                ids = ids[-seq_len:]
            items.append((ids, len(ids) - 1, gold, n_opts))

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
            for k_, idx in enumerate(range(bi, be)):
                _, _, gold, n_opts = items[idx]
                row = letter_logits[k_].copy()
                row[n_opts:] = -1e30
                if int(np.argmax(row)) == gold:
                    correct += 1

        acc = correct / max(n_items, 1)
        per_task[task] = acc
        total_correct += correct
        total_items += n_items
        print(f"  [{ti+1}/8] {task:<18} acc={acc:.3f} ({correct}/{n_items}) | n_shots={k} | elapsed={time.time()-t0:.0f}s")

    macro = sum(per_task.values()) / len(per_task)
    micro = total_correct / max(total_items, 1)
    return {
        "macro_acc": macro,
        "micro_acc": micro,
        "n_tasks": len(per_task),
        "total_items": total_items,
        "total_correct": total_correct,
        "per_task": per_task,
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

    r = agieval_eval(model, tokenizer, mesh, args.batch_size, args.seq_len)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total_items']}, MC) ===")
        print(f"  AGIEval macro_acc: {r['macro_acc']:.4f}  (micro: {r['micro_acc']:.4f})")
        print(f"  paper OLMo 2 1B post-midtrain (Table 22): 0.363")
        write_summary("agieval-mc", args.label, args.model_path, r["total_items"], 3, r)


if __name__ == "__main__":
    main()
