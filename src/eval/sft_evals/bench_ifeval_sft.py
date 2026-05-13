"""IFEval SFT eval — matches OLMES `ifeval::tulu` byte-perfectly.

0-shot, chat-format, greedy max_gen_toks=2048 (per ifeval::tulu config), stop=[].
The user message is just doc['prompt'] (no fewshot, no description).
Strict + loose verification via re-vendored Google instruction_following_eval.

Reference: allenai/OLMo-2-0425-1B-SFT model card → IFEval = 50.5
(prompt_level_loose_acc, the OLMES primary_metric for ifeval::tulu).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, load_esurge, write_summary
from sft_evals._sft_common import build_messages, format_prompt
from ifeval_lib.utils import (  # type: ignore
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)

from datasets import load_dataset


_NUM_SHOTS = 0
_STOP_SEQS: list[str] = []


def ifeval_sft_eval(model_path: str, max_gen_toks: int, max_model_len: int,
                    page_size: int = 32, max_num_seqs: int = 8):
    from easydel.inference.sampling_params import SamplingParams
    engine, tokenizer = load_esurge(model_path, max_model_len, page_size, max_num_seqs)

    ds = load_dataset("HuggingFaceH4/ifeval", split="train")
    print(f"  IFEval SFT: {len(ds)} prompts, 0-shot, chat, "
          f"max_gen_toks={max_gen_toks}")

    sp = SamplingParams(max_tokens=max_gen_toks, temperature=0.0,
                        stop=_STOP_SEQS, skip_special_tokens=True)

    prompts = []
    inputs = []
    for ex in ds:
        msgs = build_messages(
            system=None, fewshot=[], user=ex["prompt"],
            multiturn=False, assistant_prefix=None,
        )
        prompts.append(format_prompt(tokenizer, msgs, None))
        inputs.append(InputExample(
            key=ex["key"],
            instruction_id_list=ex["instruction_id_list"],
            prompt=ex["prompt"],
            kwargs=ex["kwargs"],
        ))
    req_ids = [f"req-{i:06d}" for i in range(len(prompts))]

    t0 = time.time()
    outs = engine.generate(prompts, sp, request_id=req_ids, use_tqdm=False)
    elapsed = time.time() - t0

    by_id = {getattr(o, "request_id", None): o for o in outs}
    outputs = [by_id[rid] for rid in req_ids]

    # Aggregate strict/loose at prompt and instruction levels.
    n_prompts = len(inputs)
    prompt_strict, prompt_loose = 0, 0
    inst_strict_correct, inst_loose_correct, inst_total = 0, 0, 0

    for inp, out in zip(inputs, outputs):
        text = out.outputs[0].text if hasattr(out, "outputs") else out.get_text()
        # Verifier signature is (inp, response: str), not a dict.
        s = test_instruction_following_strict(inp, text)
        l = test_instruction_following_loose(inp, text)
        if s.follow_all_instructions:
            prompt_strict += 1
        if l.follow_all_instructions:
            prompt_loose += 1
        inst_strict_correct += sum(s.follow_instruction_list)
        inst_loose_correct += sum(l.follow_instruction_list)
        inst_total += len(s.follow_instruction_list)

    return {
        "prompt_level_strict_acc": prompt_strict / max(n_prompts, 1),
        "prompt_level_loose_acc": prompt_loose / max(n_prompts, 1),
        "inst_level_strict_acc": inst_strict_correct / max(inst_total, 1),
        "inst_level_loose_acc": inst_loose_correct / max(inst_total, 1),
        "n_prompts": n_prompts,
        "n_instructions": inst_total,
        "wall_seconds": int(elapsed),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    # OLMES ifeval::tulu: max_gen_toks=2048 (config L5459), but original ifeval
    # default is 1280; tulu config overrides to 2048.
    p.add_argument("--max_gen_toks", type=int, default=2048)
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

    r = ifeval_sft_eval(args.model_path, args.max_gen_toks, args.max_length,
                        args.page_size, args.max_num_seqs)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['n_prompts']} prompts, "
              f"{r['n_instructions']} instructions, 0-shot SFT, "
              f"wall={r['wall_seconds']}s) ===")
        print(f"  IFEval-SFT prompt_level_loose_acc: {r['prompt_level_loose_acc']:.4f}")
        print(f"     prompt_level_strict_acc: {r['prompt_level_strict_acc']:.4f}")
        print(f"     inst_level_loose_acc:    {r['inst_level_loose_acc']:.4f}")
        print(f"     inst_level_strict_acc:   {r['inst_level_strict_acc']:.4f}")
        print(f"  paper allenai/OLMo-2-0425-1B-SFT (Table 16, prompt_loose): 0.505")
        write_summary("ifeval-sft", args.label, args.model_path,
                      r["n_prompts"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
