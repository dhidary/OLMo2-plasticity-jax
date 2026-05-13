"""GSM8K SFT eval — matches OLMES `gsm8k::tulu` byte-perfectly.

8-shot CoT, chat-format with fewshot_as_multiturn=True, assistant_prefix="Answer:",
greedy decoding, stop=["Question:", "</s>", "<|im_end|>"], full 1319 test set.

Reference: allenai/OLMo-2-0425-1B-SFT model card → GSM8K = 52.1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, load_esurge, write_summary
from sft_evals._sft_common import (
    build_messages, format_prompt, exact_match_hf,
    extract_gsm8k_number, GSM8K_REGEXES_TO_IGNORE,
)

from datasets import load_dataset


_NUM_SHOTS = 8
# OLMES gsm8k::tulu inherits class TASK_CONFIG_DEFAULTS chat_overrides:
#   stop_sequences=["Question:", "</s>", "<|im_end|>"]  (no "\n\n")
#   assistant_prefix="Answer:"
#   fewshot_as_multiturn=True
_STOP_SEQS = ["Question:", "</s>", "<|im_end|>"]
_ASSISTANT_PREFIX = "Answer:"


def _data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sft")


def _load_fewshot() -> list[dict]:
    with open(os.path.join(_data_dir(), "gsm8k_8shot.json")) as f:
        return json.load(f)


def _doc_to_text(question: str) -> str:
    """Matches OLMES gsm8k.py:171 query format."""
    return f"Question: {question}\nAnswer:"


def _doc_to_target(answer: str) -> str:
    """Matches OLMES doc_to_target with use_cot=True (default): " " + full answer."""
    return " " + answer


def _trim(text: str) -> str:
    cut = len(text)
    for s in _STOP_SEQS:
        j = text.find(s)
        if 0 <= j < cut:
            cut = j
    return text[:cut]


def _extract_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def gsm8k_sft_eval(model_path: str, max_gen_toks: int, max_model_len: int,
                   page_size: int = 32, max_num_seqs: int = 8):
    from easydel.inference.sampling_params import SamplingParams
    engine, tokenizer = load_esurge(model_path, max_model_len, page_size, max_num_seqs)

    fewshot_raw = _load_fewshot()[:_NUM_SHOTS]
    fewshot = [(_doc_to_text(ex["question"]), _doc_to_target(ex["answer"]))
               for ex in fewshot_raw]

    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  GSM8K SFT: {len(ds)} test items, {_NUM_SHOTS}-shot, "
          f"chat+multiturn, max_gen_toks={max_gen_toks}")

    prompts = []
    for ex in ds:
        msgs = build_messages(
            system=None, fewshot=fewshot,
            user=_doc_to_text(ex["question"]),
            multiturn=True, assistant_prefix=_ASSISTANT_PREFIX,
        )
        prompts.append(format_prompt(tokenizer, msgs, _ASSISTANT_PREFIX))
    golds = [_extract_gold(ex["answer"]) for ex in ds]
    req_ids = [f"req-{i:06d}" for i in range(len(prompts))]

    sp = SamplingParams(
        max_tokens=max_gen_toks, temperature=0.0,
        stop=_STOP_SEQS, skip_special_tokens=True,
    )

    t0 = time.time()
    outs_unordered = engine.generate(prompts, sp, request_id=req_ids, use_tqdm=False)
    elapsed = time.time() - t0
    print(f"  generation done in {elapsed:.0f}s")

    by_id = {getattr(o, "request_id", None): o for o in outs_unordered}
    outputs = [by_id[rid] for rid in req_ids]

    preds, refs = [], []
    for i, (out, gold) in enumerate(zip(outputs, golds)):
        text = out.outputs[0].text if hasattr(out, "outputs") else out.get_text()
        trimmed = _trim(text)
        preds.append(extract_gsm8k_number(trimmed))
        refs.append(gold)
        if i < 5:
            print(f"    [{i}] gold={gold} pred={preds[-1]} | trimmed: {trimmed!r}")

    em = exact_match_hf(preds, refs, regexes_to_ignore=GSM8K_REGEXES_TO_IGNORE,
                        ignore_case=True)
    correct = sum(int(p == r) for p, r in zip(preds, refs))  # raw display only
    return {
        "exact_match": em,
        "correct_strict": correct,
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

    r = gsm8k_sft_eval(args.model_path, args.max_gen_toks, args.max_length,
                       args.page_size, args.max_num_seqs)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total']}, {_NUM_SHOTS}-shot SFT, "
              f"wall={r['wall_seconds']}s) ===")
        print(f"  GSM8K-SFT exact_match: {r['exact_match']:.4f}")
        print(f"  paper allenai/OLMo-2-0425-1B-SFT (Table 16): 0.521")
        write_summary("gsm8k-sft", args.label, args.model_path, r["total"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
