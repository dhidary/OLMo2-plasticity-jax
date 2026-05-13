"""MATH SFT eval — matches OLMES `minerva_math::tulu` byte-perfectly.

4-shot CoT, chat-format, fewshot_as_multiturn=True, assistant_prefix=None
(class default override), greedy max_gen_toks=1024, stop=[]. Minerva-style
boxed answer extraction. exact_match via `is_equiv` (sympy-backed; falls
back to normalized-string compare without sympy).

Reference: allenai/OLMo-2-0425-1B-SFT model card → MATH = 13.2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, load_esurge, write_summary
from sft_evals._sft_common import build_messages, format_prompt
from sft_evals._math_utils import (
    extract_math_answers, is_equiv, last_boxed_only_string,
    normalize_final_answer, remove_boxed,
)

from datasets import load_dataset


MATH_TASK_TYPES = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]
_NUM_SHOTS = 4
_STOP_SEQS: list[str] = []


def _data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sft")


def _load_fewshot() -> list[dict]:
    with open(os.path.join(_data_dir(), "math_minerva_4shot.json"),
              encoding="utf-8", errors="surrogatepass") as f:
        return json.load(f)


def _doc_to_text(problem: str) -> str:
    """Mirror OLMES minerva_math.py:163 (cot_style="minerva")."""
    return f"Problem:\n{problem}\n\nSolution:"


def _doc_to_target(solution: str) -> str:
    """Mirror OLMES doc_to_target with use_cot=True: " " + solution."""
    return " " + solution


def _gold_answer(solution: str) -> str:
    """Extract gold final answer from solution (mirrors _process_doc:177)."""
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return ""
    try:
        return normalize_final_answer(remove_boxed(boxed))
    except AssertionError:
        return ""


def math_sft_eval(model_path: str, max_gen_toks: int, max_model_len: int,
                  page_size: int = 32, max_num_seqs: int = 8):
    from easydel.inference.sampling_params import SamplingParams
    engine, tokenizer = load_esurge(model_path, max_model_len, page_size, max_num_seqs)

    fewshot_raw = _load_fewshot()[:_NUM_SHOTS]
    fewshot = [(_doc_to_text(ex["problem"]), _doc_to_target(ex["solution"]))
               for ex in fewshot_raw]

    sp = SamplingParams(max_tokens=max_gen_toks, temperature=0.0,
                        stop=_STOP_SEQS, skip_special_tokens=True)

    # Build all prompts up front; one engine.generate avoids the multi-host
    # eSurge scheduler sync error seen on per-iteration generate calls.
    all_prompts: list[str] = []
    all_golds: list[str] = []
    all_req_ids: list[str] = []
    type_ranges: dict[str, tuple[int, int]] = {}

    t0 = time.time()
    for ttype in MATH_TASK_TYPES:
        ds = load_dataset("EleutherAI/hendrycks_math", ttype, split="test")
        start = len(all_prompts)
        for i, ex in enumerate(ds):
            user_text = _doc_to_text(ex["problem"])
            msgs = build_messages(
                system=None, fewshot=fewshot, user=user_text,
                multiturn=True, assistant_prefix=None,
            )
            all_prompts.append(format_prompt(tokenizer, msgs, None))
            all_golds.append(_gold_answer(ex["solution"]))
            all_req_ids.append(f"{ttype}-{i}")
        type_ranges[ttype] = (start, len(all_prompts))
        print(f"  built {ttype}: n={len(all_prompts)-start} (cum={len(all_prompts)})")

    # Chunk into smaller engine.generate calls to avoid eSurge multi-host
    # scheduler hangs at high prompt counts (~5000).
    print(f"  generating {len(all_prompts)} prompts across {len(MATH_TASK_TYPES)} types (chunks of 256)...")
    by_id: dict = {}
    CHUNK = 256
    for ci in range(0, len(all_prompts), CHUNK):
        ce = min(ci + CHUNK, len(all_prompts))
        outs = engine.generate(all_prompts[ci:ce], sp, request_id=all_req_ids[ci:ce], use_tqdm=False)
        for o in outs:
            by_id[getattr(o, "request_id", None)] = o
        print(f"    chunk {ci}-{ce} done (cum={len(by_id)})")

    per_type: dict[str, float] = {}
    total_correct = 0
    total_items = 0
    for ttype, (s, e) in type_ranges.items():
        sub_correct = 0
        for rid, gold in zip(all_req_ids[s:e], all_golds[s:e]):
            out = by_id[rid]
            text = out.outputs[0].text if hasattr(out, "outputs") else out.get_text()
            answers = extract_math_answers(text)
            if any(is_equiv(a, gold) for a in answers):
                sub_correct += 1
        sub_acc = sub_correct / max(e - s, 1)
        per_type[ttype] = sub_acc
        total_correct += sub_correct
        total_items += (e - s)
        print(f"  [{ttype}] n={e-s} acc={sub_acc:.4f}")

    elapsed = time.time() - t0
    macro = sum(per_type.values()) / len(per_type)
    micro = total_correct / total_items
    return {
        "exact_match": micro,
        "macro_acc": macro,
        "n_types": len(per_type),
        "total_items": total_items,
        "total_correct": total_correct,
        "per_type": per_type,
        "wall_seconds": int(elapsed),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--max_gen_toks", type=int, default=1024)
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

    r = math_sft_eval(args.model_path, args.max_gen_toks, args.max_length,
                      args.page_size, args.max_num_seqs)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total_items']}, "
              f"4-shot Minerva CoT SFT, wall={r['wall_seconds']}s) ===")
        print(f"  MATH-SFT exact_match (micro): {r['exact_match']:.4f} macro: {r['macro_acc']:.4f}")
        print(f"  paper allenai/OLMo-2-0425-1B-SFT (Table 16): 0.132")
        write_summary("math-sft", args.label, args.model_path,
                      r["total_items"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
