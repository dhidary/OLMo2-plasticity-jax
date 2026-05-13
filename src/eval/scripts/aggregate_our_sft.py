#!/usr/bin/env python3
"""Aggregate our SFT'd ckpts × SFT benches into results/sft/our_ckpts.md.

Pulls all `ourSFT_*` JSONs from gs://$GCS_BUCKET/eval/results/, keeps the
latest per (label, task), writes a comparative table next to the released
SFT card numbers.
"""
import json
import os
import re
import subprocess
import sys

_bucket = os.environ.get("GCS_BUCKET")
if not _bucket:
    sys.exit("GCS_BUCKET env var is required")
BUCKET = f"gs://{_bucket}/eval/results/"
LABELS = [
    "ourSFT_240k", "ourSFT_500k_v6", "ourSFT_1000k_v2", "ourSFT_1200k",
    "ourSFT_1400k_v2", "ourSFT_1600k", "ourSFT_1800k", "ourSFT_stage2_ingr3_v3",
    "ourSFT_1907359",
]
# (task_name_in_json, header, primary_metric, target_value)
BENCHES = [
    ("gsm8k-sft",     "GSM8K",      "exact_match",            0.521),
    ("mmlu-cot-sft",  "MMLU",       "macro_acc",              0.364),
    ("bbh-cot-sft",   "BBH",        "exact_match",            0.328),
    ("popqa-sft",     "PopQA",      "exact_match",            0.127),
    ("math-sft",      "MATH",       "exact_match",            0.132),
    ("ifeval-sft",    "IFEval",     "prompt_level_loose_acc", 0.505),
    ("truthfulqa-sft","TruthfulQA", "mc2",                    0.421),
    ("drop-sft",      "DROP",       "exact_match",            0.338),  # not yet implemented
]

def list_jsons():
    r = subprocess.run(["gcloud", "storage", "ls", BUCKET],
                       capture_output=True, text=True, timeout=60)
    return [l for l in r.stdout.split() if l.endswith(".json")
            and ("ourSFT" in l or "released_sft" in l)]

def read_json(uri):
    r = subprocess.run(["gcloud", "storage", "cat", uri],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except Exception:
        return None

def main():
    files = list_jsons()
    print(f"found {len(files)} ourSFT json files", file=sys.stderr)
    # latest_per[(label, task)] = (timestamp, metric_dict)
    latest = {}
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"^(\d{8}T\d{6}Z)_(.+)\.json$", base)
        if not m:
            continue
        ts = m.group(1)
        d = read_json(f)
        if not d:
            continue
        label = d.get("label")
        task = d.get("task")
        if not label or not task:
            continue
        key = (label, task)
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, d.get("metrics", {}))

    # Build table
    lines = []
    lines.append("# Our SFT checkpoints — comparative results")
    lines.append("")
    lines.append(f"Live aggregation of `ourSFT_*` JSONs in `{BUCKET}`.")
    lines.append("Latest result per (ckpt, bench) is shown. Targets are from the")
    lines.append("[`allenai/OLMo-2-0425-1B-SFT`](https://huggingface.co/allenai/OLMo-2-0425-1B-SFT)")
    lines.append("model card (Table 16, post-SFT, before DPO/RLVR).")
    lines.append("")
    header = ["ckpt"] + [b[1] for b in BENCHES]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    target_row = ["**target (released card)**"]
    for _, _, _, tgt in BENCHES:
        target_row.append(f"**{tgt:.3f}**")
    lines.append("| " + " | ".join(target_row) + " |")
    # "Our repro of released SFT" row — pulls best released_sft_* result per bench
    repro_row = ["*our repro of released SFT*"]
    for task, _, primary, _ in BENCHES:
        # find latest released_sft_* result for this task
        best = None
        for (lbl, tk), (ts, m) in latest.items():
            if tk == task and lbl.startswith("released_sft"):
                if best is None or ts > best[0]:
                    best = (ts, m)
        if not best:
            repro_row.append("—")
        else:
            v = best[1].get(primary)
            if v is None:
                for alt in ("exact_match_flex", "micro_acc"):
                    if alt in best[1]: v = best[1][alt]; break
            repro_row.append(f"*{v:.4f}*" if v is not None else "—")
    lines.append("| " + " | ".join(repro_row) + " |")
    for label in LABELS:
        row = [label.replace("ourSFT_", "")]
        for task, _, primary, _ in BENCHES:
            entry = latest.get((label, task))
            if not entry:
                row.append("—")
            else:
                m = entry[1]
                v = m.get(primary)
                if v is None:
                    # try alternates
                    for alt in ("exact_match_flex", "micro_acc"):
                        if alt in m:
                            v = m[alt]; break
                row.append(f"{v:.4f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    n_done = sum(1 for label in LABELS for task, _, _, _ in BENCHES if (label, task) in latest)
    n_total = len(LABELS) * len(BENCHES)
    lines.append(f"{n_done}/{n_total} cells filled.")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "..", "results", "sft", "our_ckpts.md")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({n_done}/{n_total} cells)", file=sys.stderr)

if __name__ == "__main__":
    main()
