"""TriviaQA: 5-shot generative QA, SQuAD-style F1/EM, eSurge greedy.

Matches OLMES `triviaqa::olmes`. Paper Table 22 OLMo 2 1B = 0.547 (F1).
"""

from __future__ import annotations

import argparse
import os
import re
import string
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, load_esurge, write_summary

from datasets import load_dataset


# Verbatim from allenai/olmes:oe_eval/tasks/fewshot_sources.py FEWSHOT_SOURCES["OLMES:triviaqa"].
# (Q, answer_value) — doc_to_target uses " " + answer_value.
OLMES_TRIVIAQA_FEWSHOT = [
    ("From which country did Angola achieve independence in 1975?", "Portugal"),
    ("In architecture, what is a lancet?", "A window"),
    ("In which ocean is the island group the Maldives?", "The Indian Ocean"),
    ("In the UK these books are called ‘Where’s Wally?’ What is his name in the USA?", "Waldo"),
    ("The conflict between rival political factions the Girondins and the Jacobins, during the French Revolution, was known as The Reign of what?", "Terror"),
    ("Only one actor has played the part of Sherlock Holmes in every field of entertainment, that is stage, screen, radio, TV and records. What is his name?", "BASIL RATHBONE"),
    ("What city built the first underground train network?", "London"),
    ("Who was killed when his private plane crashed in Monterey Bay, California in 1997?", "John Denver"),
    ("Which TV series featured the characters, Tinker, Eric Catchpole and Lady Jane Felsham?", "LOVEJOY"),
    ("Who is the person of religious history whose pardon was the subject of a 1950 novel by Pär Lagerkvist as well as a 1961 portrayal by Anthony Quinn?", "Barabbas"),
]

_NUM_SHOTS = 5
_STOP_SEQS = ["\n\n", "Question:", "</s>", "<|im_end|>"]


def _build_prompt(question: str) -> str:
    parts = [f"Question: {q}\nAnswer: {a}" for q, a in OLMES_TRIVIAQA_FEWSHOT[:_NUM_SHOTS]]
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def _trim(text: str) -> str:
    cut = len(text)
    for s in _STOP_SEQS:
        j = text.find(s)
        if 0 <= j < cut:
            cut = j
    return text[:cut].strip()


# SQuAD V1 EM/F1 — verbatim from olmes:oe_eval/dependencies/squad/squad_emf1.py.
def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    pt = _normalize(pred).split()
    gt = _normalize(gold).split()
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pt) if pt else 0.0
    r = n / len(gt) if gt else 0.0
    return (2 * p * r) / (p + r) if (p + r) else 0.0


def _em(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def _max_over_refs(metric_fn, pred, refs):
    return max((metric_fn(pred, r) for r in refs), default=0.0)


def triviaqa_eval(model_path: str, max_gen_toks: int, max_model_len: int,
                  page_size: int = 32, max_num_seqs: int = 8):
    from easydel.inference.sampling_params import SamplingParams
    engine, tokenizer = load_esurge(model_path, max_model_len, page_size, max_num_seqs)

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split="validation")
    print(f"  TriviaQA: {len(ds)} val items, {_NUM_SHOTS}-shot, max_gen_toks={max_gen_toks}")

    prompts = [_build_prompt(ex["question"]) for ex in ds]
    refs_list = [list(set(ex["answer"]["aliases"] + ex["answer"]["normalized_aliases"])) for ex in ds]
    req_ids = [f"req-{i:06d}" for i in range(len(prompts))]

    sp = SamplingParams(
        max_tokens=max_gen_toks,
        temperature=0.0,
        stop=_STOP_SEQS,
        skip_special_tokens=True,
    )

    t0 = time.time()
    outs_unordered = engine.generate(prompts, sp, request_id=req_ids, use_tqdm=False)
    elapsed = time.time() - t0
    print(f"  generation done in {elapsed:.0f}s")

    by_id = {getattr(o, "request_id", None): o for o in outs_unordered}
    outputs = [by_id[rid] for rid in req_ids]

    em_sum = 0.0
    f1_sum = 0.0
    for i, (out, refs) in enumerate(zip(outputs, refs_list)):
        text = out.outputs[0].text if hasattr(out, "outputs") else out.get_text()
        pred = _trim(text)
        em = _max_over_refs(_em, pred, refs)
        f1 = _max_over_refs(_f1, pred, refs)
        em_sum += em
        f1_sum += f1
        if i < 5:
            print(f"    [{i}] gold[0]={refs[0]!r} pred={pred!r} em={em:.0f} f1={f1:.2f}")

    n = len(prompts)
    return {
        "exact_match": em_sum / max(n, 1),
        "f1": f1_sum / max(n, 1),
        "total": n,
        "wall_seconds": int(elapsed),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--max_gen_toks", type=int, default=50)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--page_size", type=int, default=32)
    p.add_argument("--max_num_seqs", type=int, default=8)
    p.add_argument("--label", required=True)
    args = p.parse_args()

    jax.distributed.initialize()
    print(f"[{args.label}] devices={jax.device_count()} processes={jax.process_count()}")

    r = triviaqa_eval(args.model_path, args.max_gen_toks, args.max_length, args.page_size, args.max_num_seqs)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total']}, {_NUM_SHOTS}-shot, wall={r['wall_seconds']}s) ===")
        print(f"  TriviaQA exact_match: {r['exact_match']:.4f}")
        print(f"  TriviaQA f1:          {r['f1']:.4f}")
        print(f"  paper OLMo 2 1B post-midtrain (Table 22, F1): 0.547")
        write_summary("triviaqa", args.label, args.model_path, r["total"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
