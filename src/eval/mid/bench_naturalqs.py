"""NaturalQuestions Open: 5-shot generative QA, DROP-style F1/EM, eSurge greedy.

Matches OLMES `naturalqs::olmes` (limit=1000). Paper Table 22 OLMo 2 1B = 0.208 (F1).
"""

from __future__ import annotations

import argparse
import os
import re
import string
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import jax, load_esurge, write_summary

from datasets import load_dataset


# Verbatim from allenai/olmes:oe_eval/tasks/fewshot_sources.py FEWSHOT_SOURCES["OLMES:naturalqs"].
# (Q, [answers]). doc_to_target = " " + ", ".join(answers).
OLMES_NQ_FEWSHOT = [
    ("which side of the white house is the front", ["North"]),
    ("who's hosting the super bowl in 2019", ["Atlanta, Georgia"]),
    ("what is the origin of the name cynthia", ["Greek"]),
    ("who is the guy who voiced disney channel", ['"Buzz" Brainard', "Cameron"]),
    ("what is the size of the angles of an equilateral triangle", ["60°"]),
    ("who plays mavis in the movie hotel transylvania", ["Sadie Sandler", "Selena Gomez"]),
    ("west ham players in the 1966 world cup", ["Martin Peters", "Geoff Hurst", "Bobby Moore"]),
    ("who sings the theme song for miami vice", ["Jan Hammer"]),
    ("ice sheets and tundra are typical of which koppen climate category", ["polar"]),
    ("what's the legal marriage age in new york", ["18"]),
]

_NUM_SHOTS = 5
_LIMIT = 1000  # OLMES naturalqs::olmes limit
_STOP_SEQS = ["Question:", "Q:", "\n\n"]


def _build_prompt(question: str) -> str:
    parts = [f"Question: {q}\nAnswer: {', '.join(a)}" for q, a in OLMES_NQ_FEWSHOT[:_NUM_SHOTS]]
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def _trim(text: str) -> str:
    cut = len(text)
    for s in _STOP_SEQS:
        j = text.find(s)
        if 0 <= j < cut:
            cut = j
    return text[:cut].strip()


# DROP-style normalization (used by OLMES naturalqs metric).
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _norm_token(tok: str) -> str:
    tok = tok.lower()
    if not _is_number(tok):
        tok = "".join(ch for ch in tok if ch not in set(string.punctuation))
    tok = _ARTICLES.sub(" ", tok)
    if _is_number(tok):
        tok = str(float(tok))
    return " ".join(tok.split())


def _normalize_drop(answer: str) -> str:
    tokens = [_norm_token(t) for t in re.split(r" |-", answer)]
    tokens = [t for t in tokens if t.strip()]
    return " ".join(tokens).strip()


def _drop_em_f1(pred: str, gold: str):
    pn = _normalize_drop(pred)
    gn = _normalize_drop(gold)
    em = 1.0 if pn == gn else 0.0
    pset = set(pn.split())
    gset = set(gn.split())
    common = pset & gset
    if not pset or not gset:
        f1 = 0.0
    elif not common:
        f1 = 0.0
    else:
        p = len(common) / len(pset)
        r = len(common) / len(gset)
        f1 = (2 * p * r) / (p + r)
    return em, f1


def _max_em_f1(pred: str, golds):
    em = 0.0
    f1 = 0.0
    for g in golds:
        if not g.strip():
            continue
        e, f = _drop_em_f1(pred, g)
        if e > em:
            em = e
        if f > f1:
            f1 = f
    return em, f1


def naturalqs_eval(model_path: str, max_gen_toks: int, max_model_len: int,
                   page_size: int = 32, max_num_seqs: int = 8):
    from easydel.inference.sampling_params import SamplingParams
    engine, tokenizer = load_esurge(model_path, max_model_len, page_size, max_num_seqs)

    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    ds = ds.select(range(min(_LIMIT, len(ds))))
    print(f"  NaturalQs: {len(ds)} val items (limit {_LIMIT}), {_NUM_SHOTS}-shot, max_gen_toks={max_gen_toks}")

    prompts = [_build_prompt(ex["question"]) for ex in ds]
    refs_list = [list(ex["answer"]) for ex in ds]
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

    em_sum = f1_sum = 0.0
    for i, (out, refs) in enumerate(zip(outputs, refs_list)):
        text = out.outputs[0].text if hasattr(out, "outputs") else out.get_text()
        pred = _trim(text)
        em, f1 = _max_em_f1(pred, refs)
        em_sum += em
        f1_sum += f1
        if i < 5:
            print(f"    [{i}] golds={refs} pred={pred!r} em={em:.0f} f1={f1:.2f}")

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

    r = naturalqs_eval(args.model_path, args.max_gen_toks, args.max_length, args.page_size, args.max_num_seqs)
    if jax.process_index() == 0:
        print(f"\n=== RESULT [{args.label}] (n={r['total']}, {_NUM_SHOTS}-shot, wall={r['wall_seconds']}s) ===")
        print(f"  NaturalQs exact_match: {r['exact_match']:.4f}")
        print(f"  NaturalQs f1:          {r['f1']:.4f}")
        print(f"  paper OLMo 2 1B post-midtrain (Table 22, F1): 0.208")
        write_summary("naturalqs", args.label, args.model_path, r["total"], _NUM_SHOTS, r)


if __name__ == "__main__":
    main()
