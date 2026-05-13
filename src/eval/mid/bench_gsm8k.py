"""GSM8K via eSurge: 8-shot CoT, greedy, evaluated on the OLMES 1119 held-out subset."""

from __future__ import annotations

import argparse
import os
import random as _rnd
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, load_esurge, write_summary

import numpy as np
from datasets import load_dataset


# Verbatim from allenai/olmes:oe_eval/tasks/fewshot_sources.py FEWSHOT_SOURCES["STD:GSM8k"].
OLMES_GSM8K_FEWSHOT = [
    {"question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6."},
    {"question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5."},
    {"question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39."},
    {"question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8."},
    {"question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9."},
    {"question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29."},
    {"question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33."},
    {"question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8."},
]

_NUM_SHOTS = 8
_NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")
# eSurge ignores string stops; we trim post-hoc at the first occurrence.
_STOP_SEQS = ["\n\nQuestion:", "Question:", "</s>", "<|im_end|>", "\n\n"]


def _build_prompt(question: str) -> str:
    parts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in OLMES_GSM8K_FEWSHOT[:_NUM_SHOTS]]
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def _extract_pred(text: str) -> str:
    s = text.replace(",", "")
    s = re.sub(r"\$", "", s)
    s = re.sub(r"(?s).*#### ", "", s)
    s = re.sub(r"\.$", "", s.strip())
    nums = _NUM_RE.findall(s)
    return nums[-1] if nums else s


def _extract_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def _trim(text: str) -> str:
    cut = len(text)
    for s in _STOP_SEQS:
        j = text.find(s)
        if 0 <= j < cut:
            cut = j
    return text[:cut]


def gsm8k_eval(model_path: str, max_gen_toks: int, max_model_len: int,
               page_size: int = 32, max_num_seqs: int = 8):
    from easydel.inference.sampling_params import SamplingParams
    engine, tokenizer = load_esurge(model_path, max_model_len, page_size, max_num_seqs)

    # OLMES "1119 held-out": drop the 200 GSM* dev items.
    ds = load_dataset("openai/gsm8k", "main", split="test")
    drop = set(_rnd.Random(42).sample(range(len(ds)), 200))
    ds = ds.select([i for i in range(len(ds)) if i not in drop])
    print(f"  GSM8K: {len(ds)} test items, {_NUM_SHOTS}-shot, max_gen_toks={max_gen_toks}")

    prompts = [_build_prompt(ex["question"]) for ex in ds]
    golds = [_extract_gold(ex["answer"]) for ex in ds]
    req_ids = [f"req-{i:06d}" for i in range(len(prompts))]

    sp = SamplingParams(
        max_tokens=max_gen_toks,
        temperature=0.0,
        stop=["Question:", "</s>", "<|im_end|>", "\n\n"],
        skip_special_tokens=True,
    )

    t0 = time.time()
    outs_unordered = engine.generate(prompts, sp, request_id=req_ids, use_tqdm=False)
    elapsed = time.time() - t0
    print(f"  generation done in {elapsed:.0f}s")

    by_id = {getattr(o, "request_id", None): o for o in outs_unordered}
    outputs = [by_id[rid] for rid in req_ids]

    correct = 0
    for i, (out, gold) in enumerate(zip(outputs, golds)):
        text = out.outputs[0].text if hasattr(out, "outputs") else out.get_text()
        trimmed = _trim(text)
        pred = _extract_pred(trimmed)
        ok = pred.strip().lower() == gold.strip().lower()
        correct += int(ok)
        if i < 5:
            finish = getattr(out.outputs[0], "finish_reason", "?") if hasattr(out, "outputs") else "?"
            print(f"    [{i}] gold={gold} pred={pred} ok={ok} | finish={finish} | trimmed: {trimmed!r}")

    return {
        "exact_match": correct / max(len(prompts), 1),
        "correct": correct,
        "total": len(prompts),
        "wall_seconds": int(elapsed),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--max_gen_toks", type=int, default=512)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--max_num_seqs", type=int, default=8)
    p.add_argument("--label", required=True)
    args = p.parse_args()

    jax.distributed.initialize()
    print(f"[{args.label}] devices={jax.device_count()} processes={jax.process_count()}")

    r = gsm8k_eval(args.model_path, args.max_gen_toks, args.max_length, args.page_size, args.max_num_seqs)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total']}, {_NUM_SHOTS}-shot, wall={r['wall_seconds']}s) ===")
        print(f"  GSM8K exact_match: {r['exact_match']:.4f}  ({r['correct']}/{r['total']})")
        print(f"  paper OLMo 2 1B post-midtrain (Table 22, 1119 held-out): 0.438")
        print(f"  paper OLMo 2 1B stage1 only baseline (1119 held-out):    0.033")
        write_summary("gsm8k", args.label, args.model_path, r["total"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
