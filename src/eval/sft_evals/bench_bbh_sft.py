"""BBH SFT eval — matches OLMES `bbh:cot-v1::tulu` byte-perfectly.

3-shot CoT, chat-format, fewshot_as_multiturn=True, assistant_prefix=None,
greedy max_gen_toks=512, stop=[]. Per-subtask answer regex from
BBH_ANSWER_REGEX (port of oe_eval/tasks/oe_eval_tasks/bbh.py:31-59).
ignore_case=True; ignore_punctuation=True except for `dyck_languages`.

Reference: allenai/OLMo-2-0425-1B-SFT model card → BBH = 32.8.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, load_esurge, write_summary
from sft_evals._sft_common import build_messages, format_prompt, bbh_extract_v1, exact_match_hf

from datasets import load_dataset


BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]

_NUM_SHOTS = 3
# OLMES bbh:cot-v1::tulu chat_overrides: stop_sequences=[]
_STOP_SEQS: list[str] = []


def _data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sft")


def _load_descriptions() -> dict[str, str]:
    with open(os.path.join(_data_dir(), "bbh", "_descriptions.json")) as f:
        return json.load(f)


def _load_fewshot(sub: str) -> list[dict]:
    with open(os.path.join(_data_dir(), "bbh", f"{sub}.json")) as f:
        return json.load(f)


# Mirrors GenericBBH._process_doc with short_prefix=True, use_cot=True (defaults).
def _doc_to_text(question: str) -> str:
    return f"Q: {question}\nA: Let's think step by step."


def _doc_to_target(answer_text: str) -> str:
    """`use_cot=True`: returns " " + solution, with leading 'Let's think step by step.' stripped."""
    solution = re.sub(r"^Let's think step by step\.", "", answer_text)
    return " " + solution


def _extract_gold(target: str) -> str:
    """Mirrors GenericBBH._process_doc: regex `(?<=answer is )(.*)(?=.)` on the gold target."""
    m = re.search(r"(?<=answer is )(.*)(?=.)", target)
    return m[0] if m else target


def bbh_sft_eval(model_path: str, max_gen_toks: int, max_model_len: int,
                 page_size: int = 32, max_num_seqs: int = 8):
    from easydel.inference.sampling_params import SamplingParams
    engine, tokenizer = load_esurge(model_path, max_model_len, page_size, max_num_seqs)

    descriptions = _load_descriptions()
    sp = SamplingParams(max_tokens=max_gen_toks, temperature=0.0,
                        stop=_STOP_SEQS, skip_special_tokens=True)

    # Build all prompts across all subtasks first; one engine.generate call
    # avoids the multi-host eSurge scheduler sync error that appears when
    # sub-batches differ across processes between iterations.
    all_prompts: list[str] = []
    all_golds: list[str] = []
    all_req_ids: list[str] = []
    sub_ranges: dict[str, tuple[int, int]] = {}  # sub -> (start, end)

    t0 = time.time()
    for sub in BBH_TASKS:
        ds = load_dataset("lukaemon/bbh", sub, split="test")
        fewshot_raw = _load_fewshot(sub)[:_NUM_SHOTS]
        fewshot = [(_doc_to_text(ex["input"]), _doc_to_target(ex["target"]))
                   for ex in fewshot_raw]
        start = len(all_prompts)
        for i, ex in enumerate(ds):
            user_text = _doc_to_text(ex["input"])
            msgs = build_messages(
                system=None, fewshot=fewshot, user=user_text,
                multiturn=True, description=descriptions[sub] + "\n\n",
                assistant_prefix=None,
            )
            all_prompts.append(format_prompt(tokenizer, msgs, None))
            all_golds.append(_extract_gold(ex["target"]))
            all_req_ids.append(f"{sub}-{i}")
        sub_ranges[sub] = (start, len(all_prompts))
        print(f"  built {sub}: n={len(all_prompts)-start} (cum={len(all_prompts)})")

    # Chunk into smaller engine.generate calls to avoid eSurge multi-host
    # scheduler hangs at high prompt counts (~6500). Each chunk is processed
    # independently; results merged via request_id.
    print(f"  generating {len(all_prompts)} prompts across {len(BBH_TASKS)} subtasks (chunks of 256)...")
    by_id: dict = {}
    CHUNK = 256
    for ci in range(0, len(all_prompts), CHUNK):
        ce = min(ci + CHUNK, len(all_prompts))
        outs = engine.generate(all_prompts[ci:ce], sp, request_id=all_req_ids[ci:ce], use_tqdm=False)
        for o in outs:
            by_id[getattr(o, "request_id", None)] = o
        print(f"    chunk {ci}-{ce} done (cum={len(by_id)})")

    per_task: dict[str, float] = {}
    total_correct = 0
    total_items = 0
    for sub, (s, e) in sub_ranges.items():
        preds = []
        for rid in all_req_ids[s:e]:
            out = by_id[rid]
            text = out.outputs[0].text if hasattr(out, "outputs") else out.get_text()
            preds.append(bbh_extract_v1(text, sub))
        sub_golds = all_golds[s:e]
        ignore_punct = sub != "dyck_languages"
        sub_acc = exact_match_hf(preds, sub_golds, ignore_case=True,
                                  ignore_punctuation=ignore_punct)
        per_task[sub] = sub_acc
        sub_correct = round(sub_acc * (e - s))
        total_correct += sub_correct
        total_items += (e - s)
        print(f"  [{sub}] n={e-s} acc={sub_acc:.4f}")

    elapsed = time.time() - t0
    macro = sum(per_task.values()) / len(per_task)
    micro = total_correct / total_items
    return {
        "macro_acc": macro,
        "micro_acc": micro,
        "n_tasks": len(per_task),
        "total_items": total_items,
        "total_correct": total_correct,
        "per_task": per_task,
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

    r = bbh_sft_eval(args.model_path, args.max_gen_toks, args.max_length,
                     args.page_size, args.max_num_seqs)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total_items']}, "
              f"3-shot CoT-v1 SFT, wall={r['wall_seconds']}s) ===")
        print(f"  BBH-SFT macro_acc: {r['macro_acc']:.4f} micro: {r['micro_acc']:.4f}")
        print(f"  paper allenai/OLMo-2-0425-1B-SFT (Table 16): 0.328")
        write_summary("bbh-cot-sft", args.label, args.model_path,
                      r["total_items"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
