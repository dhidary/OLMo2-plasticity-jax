"""PopQA SFT eval — matches OLMES `popqa::tulu` byte-perfectly.

15-shot, chat-format, fewshot_as_multiturn=True, assistant_prefix=None,
greedy max_gen_toks=15, stop=["\\n\\n"]. For each test doc, the 15 fewshot
shots are the 16 source entries minus the one with matching prop_id (one
per template). Alias-substring scoring per OLMES popqa.py:170-180.

Reference: allenai/OLMo-2-0425-1B-SFT model card → PopQA = 12.7.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, load_esurge, write_summary
from sft_evals._sft_common import build_messages, format_prompt, popqa_alias_match

from datasets import load_dataset


_NUM_SHOTS = 15
_STOP_SEQS = ["\n\n"]


def _data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sft")


def _load_fewshot() -> list[dict]:
    with open(os.path.join(_data_dir(), "popqa_fewshot.json"),
              encoding="utf-8", errors="surrogatepass") as f:
        return json.load(f)


def _doc_to_text(question: str) -> str:
    """Mirror OLMES popqa.py:113."""
    return f"Q: {question} A:"


def _doc_to_target(gold: str) -> str:
    """Mirror OLMES popqa.py:115-116."""
    return " " + gold


def _trim(text: str) -> str:
    cut = len(text)
    for s in _STOP_SEQS:
        j = text.find(s)
        if 0 <= j < cut:
            cut = j
    return text[:cut]


def popqa_sft_eval(model_path: str, max_gen_toks: int, max_model_len: int,
                   page_size: int = 32, max_num_seqs: int = 8):
    from easydel.inference.sampling_params import SamplingParams
    engine, tokenizer = load_esurge(model_path, max_model_len, page_size, max_num_seqs)

    fewshot_raw = _load_fewshot()
    # Map prop_id -> (user_text, assistant_text) for the fewshot pool.
    pool = {ex["prop_id"]: (_doc_to_text(ex["question"]),
                            _doc_to_target(ex["obj"]))
            for ex in fewshot_raw}

    ds = load_dataset("akariasai/PopQA", split="test")
    print(f"  PopQA SFT: {len(ds)} test items, {_NUM_SHOTS}-shot, "
          f"chat+multiturn, max_gen_toks={max_gen_toks}")

    sp = SamplingParams(max_tokens=max_gen_toks, temperature=0.0,
                        stop=_STOP_SEQS, skip_special_tokens=True)

    prompts = []
    aliases_per_doc = []
    for ex in ds:
        prop_id = ex["prop_id"]
        # Take all 16 except the matching template.
        fewshot = [pool[pid] for pid in pool if pid != prop_id]
        # OLMES iterates POPQA_TEMPLATES.keys() order, which is insertion order
        # of the source dict — preserved here by the fewshot json's order.
        msgs = build_messages(
            system=None, fewshot=fewshot,
            user=_doc_to_text(ex["question"]),
            multiturn=True, assistant_prefix=None,
        )
        prompts.append(format_prompt(tokenizer, msgs, None))
        aliases_per_doc.append(json.loads(ex["possible_answers"]))

    req_ids = [f"req-{i:06d}" for i in range(len(prompts))]
    t0 = time.time()
    outs = engine.generate(prompts, sp, request_id=req_ids, use_tqdm=False)
    elapsed = time.time() - t0

    by_id = {getattr(o, "request_id", None): o for o in outs}
    outputs = [by_id[rid] for rid in req_ids]

    correct = 0
    for i, (out, aliases) in enumerate(zip(outputs, aliases_per_doc)):
        text = out.outputs[0].text if hasattr(out, "outputs") else out.get_text()
        trimmed = _trim(text)
        ok = popqa_alias_match(trimmed, aliases)
        correct += int(ok)
        if i < 5:
            print(f"    [{i}] aliases={aliases[:3]} pred={trimmed!r} ok={ok}")
    return {
        "exact_match": correct / max(len(prompts), 1),
        "correct": correct,
        "total": len(prompts),
        "wall_seconds": int(elapsed),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--max_gen_toks", type=int, default=15)
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

    r = popqa_sft_eval(args.model_path, args.max_gen_toks, args.max_length,
                       args.page_size, args.max_num_seqs)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total']}, {_NUM_SHOTS}-shot SFT, "
              f"wall={r['wall_seconds']}s) ===")
        print(f"  PopQA-SFT exact_match: {r['exact_match']:.4f}")
        print(f"  paper allenai/OLMo-2-0425-1B-SFT (Table 16): 0.127")
        write_summary("popqa-sft", args.label, args.model_path, r["total"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
