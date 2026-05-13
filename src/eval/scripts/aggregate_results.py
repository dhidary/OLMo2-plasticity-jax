#!/usr/bin/env python3
"""Pull bench summaries from GCS and update per-task results/*.md tables.

GCS layout: gs://$GCS_BUCKET/eval/results/{ts}_{task}_{label}.json
Local md targets: results/{task_short}.md

For each task we render two tables between markers:
    <!-- BENCH-RESULTS-START -->
    ... auto-generated tables ...
    <!-- BENCH-RESULTS-END -->

Tables:
- "Pretrain → midtrain pairs" with stage1_X / midtrained_X side-by-side, Δ.
- "All results (sorted)" with model_path.

Run: python src/eval/scripts/aggregate_results.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict


_bucket = os.environ.get("GCS_BUCKET")
if not _bucket:
    sys.exit("GCS_BUCKET env var is required")
GCS_BUCKETS = [b.strip() for b in _bucket.split(",") if b.strip()]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results")
RESULTS_DIR = os.path.abspath(RESULTS_DIR)

# Map summary "task" field → (results/<file>.md, primary_metric, [extra_metric, …]).
# arc-c-mc / hellaswag-rc / mmlu-mc / gsm8k are intentionally omitted: those
# tables are hand-curated (long history of label variants pre-dates our
# canonical naming scheme), so auto-gen would only add noise.
TASK_TO_MD = {
    "winogrande-rc": ("winogrande.md", "acc_raw", []),
    "naturalqs": ("naturalqs.md", "f1", ["exact_match"]),
    "triviaqa": ("triviaqa.md", "f1", ["exact_match"]),
    "agieval-mc": ("agieval.md", "macro_acc", ["micro_acc"]),
    "mmlu-pro-mc": ("mmlu_pro.md", "micro_acc", ["macro_acc"]),
    # SFT-suite (chat-format, OLMES `::tulu`).
    "gsm8k-sft": ("sft/gsm8k.md", "exact_match", []),
    "mmlu-cot-sft": ("sft/mmlu.md", "macro_acc", ["micro_acc"]),
    "bbh-cot-sft": ("sft/bbh.md", "macro_acc", ["micro_acc"]),
    "popqa-sft": ("sft/popqa.md", "exact_match", []),
    "math-sft": ("sft/math.md", "exact_match", ["macro_acc"]),
    "ifeval-sft": ("sft/ifeval.md", "prompt_level_loose_acc",
                   ["prompt_level_strict_acc", "inst_level_loose_acc", "inst_level_strict_acc"]),
    "truthfulqa-sft": ("sft/truthfulqa.md", "mc2", ["mc1"]),
}

# Order for "Pretrain → midtrain pairs" rows.
PAIR_STEPS = ["240k", "1000k", "1200k", "1400k", "1600k", "1800k"]

# Labels we consider "canonical" — everything else (cache_test, mc_validate,
# *_fp32, *_v2, *_arc_sf, …) is treated as exploratory and filtered out.
_CANONICAL_RE = re.compile(
    r"^(?:"
    r"stage1_(?:240|500|1000|1200|1400|1600|1800|1907|final)k?"
    r"|midtrained_(?:240|500|1000|1200|1400|1600|1800|1907|final)k"
    r"|stage2_ingr[123]"
    r"|main"
    r")$"
)

START_MARKER = "<!-- BENCH-RESULTS-START -->"
END_MARKER = "<!-- BENCH-RESULTS-END -->"


def list_summaries(bucket: str) -> list[str]:
    out = subprocess.run(
        ["gcloud", "storage", "ls", f"gs://{bucket}/eval/results/"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip().endswith(".json")]


def fetch_summary(uri: str) -> dict | None:
    out = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def parse_uri(uri: str) -> tuple[str, str, str] | None:
    """Returns (ts, task, label) or None.
    URI tail is e.g. '20260430T035003Z_bbh_stage2_ingr3.json'.
    Task names can contain underscore (e.g. mmlu_pro), so we match by known task list.
    """
    name = uri.split("/")[-1].removesuffix(".json")
    m = re.match(r"^(\d{8}T\d{6}Z)_(.+)$", name)
    if not m:
        return None
    ts, rest = m.group(1), m.group(2)
    # Try every known task as a prefix. SFT names listed FIRST so they win
    # over the bare-name prefix (e.g. "gsm8k_sft_..." matches "gsm8k_sft"
    # before "gsm8k").
    known_tasks = [
        "gsm8k_sft", "mmlu_sft", "bbh_sft", "popqa_sft", "math_sft",
        "ifeval_sft", "truthfulqa_sft",
        "arc-c-mc", "arc", "hellaswag-rc", "hellaswag", "mmlu-mc", "mmlu-pro-mc",
        "mmlu_pro", "mmlu", "winogrande-rc", "winogrande", "naturalqs", "triviaqa",
        "agieval-mc", "agieval", "popqa", "math-500", "math", "bbh-cot", "bbh",
        "ifeval", "gsm8k",
    ]
    for t in known_tasks:
        prefix = t + "_"
        if rest.startswith(prefix):
            label = rest[len(prefix):]
            # Map shell task name → summary task name where they differ.
            task_map = {
                "arc": "arc-c-mc", "hellaswag": "hellaswag-rc", "mmlu": "mmlu-mc",
                "winogrande": "winogrande-rc", "agieval": "agieval-mc",
                "mmlu_pro": "mmlu-pro-mc", "math": "math-500", "bbh": "bbh-cot",
                "gsm8k_sft": "gsm8k-sft", "mmlu_sft": "mmlu-cot-sft",
                "bbh_sft": "bbh-cot-sft", "popqa_sft": "popqa-sft",
                "math_sft": "math-sft", "ifeval_sft": "ifeval-sft",
                "truthfulqa_sft": "truthfulqa-sft",
            }
            return (ts, task_map.get(t, t), label)
    return None


def collect_latest(only_canonical: bool = True) -> dict:
    """Returns {(task, label): summary_dict} with latest by ts.
    If only_canonical, drop labels that don't match the canonical regex
    (filters out cache_test, *_fp32, *_v2, *_arc_sf, etc.).
    """
    latest: dict[tuple[str, str], tuple[str, dict]] = {}
    for bucket in GCS_BUCKETS:
        for uri in list_summaries(bucket):
            parsed = parse_uri(uri)
            if not parsed:
                continue
            ts, task, label = parsed
            if only_canonical and not _CANONICAL_RE.match(label):
                continue
            key = (task, label)
            if key in latest and latest[key][0] >= ts:
                continue
            summary = fetch_summary(uri)
            if summary is None:
                continue
            latest[key] = (ts, summary)
    return {k: v[1] for k, v in latest.items()}


def fmt(x: float | int | None, prec: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{prec}f}"


def render_pairs_table(rows: list[tuple[str, float | None, float | None]]) -> str:
    """rows = [(step_label, stage1_val, midtrained_val), ...]"""
    lines = ["| step | stage1 (pretrain only) | midtrained (50B Dolmino) | Δ |",
             "|---|---|---|---|"]
    for step, s1, mt in rows:
        delta = "—"
        if s1 is not None and mt is not None:
            delta = f"{(mt - s1) * 100:+.1f}"
        lines.append(f"| {step} | {fmt(s1)} | {fmt(mt)} | {delta} |")
    return "\n".join(lines)


def render_all_results_table(primary: str, extras: list[str], by_label: dict[str, dict]) -> str:
    """All labels sorted by primary metric desc, with extras + model_path."""
    rows = []
    for label, summary in by_label.items():
        m = summary.get("metrics", {})
        v = m.get(primary)
        if v is None:
            continue
        extras_vals = [m.get(e) for e in extras]
        path = summary.get("model_path", "—")
        rows.append((v, label, extras_vals, path))
    rows.sort(reverse=True, key=lambda r: r[0])
    headers = ["label", primary, *extras, "model_path"]
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for v, label, extras_vals, path in rows:
        cells = [label, fmt(v), *(fmt(x) for x in extras_vals), f"`{path}`"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_block(task: str, primary: str, extras: list[str], by_label: dict[str, dict]) -> str:
    """Build the full markdown block between START/END markers."""
    pair_rows = []
    for step in PAIR_STEPS:
        s1 = by_label.get(f"stage1_{step}", {}).get("metrics", {}).get(primary)
        mt = by_label.get(f"midtrained_{step}", {}).get("metrics", {}).get(primary)
        pair_rows.append((step, s1, mt))

    parts = ["", f"### Pretrain → midtrain pairs (primary: `{primary}`)",
             "", render_pairs_table(pair_rows),
             "", "### All results (sorted by primary metric)",
             "", render_all_results_table(primary, extras, by_label),
             ""]
    return "\n".join(parts)


def update_md(md_path: str, block: str) -> bool:
    if not os.path.exists(md_path):
        return False
    with open(md_path) as f:
        text = f.read()
    if START_MARKER in text and END_MARKER in text:
        new = re.sub(
            re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
            START_MARKER + "\n" + block + "\n" + END_MARKER,
            text, flags=re.DOTALL,
        )
    else:
        # Append markers at the end of the file.
        new = text.rstrip() + "\n\n" + START_MARKER + "\n" + block + "\n" + END_MARKER + "\n"
    if new != text:
        with open(md_path, "w") as f:
            f.write(new)
        return True
    return False


def main():
    print("Listing GCS summaries...", file=sys.stderr)
    latest = collect_latest()
    print(f"  collected {len(latest)} (task, label) entries", file=sys.stderr)

    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for (task, label), summary in latest.items():
        by_task[task][label] = summary

    for task, by_label in sorted(by_task.items()):
        if task not in TASK_TO_MD:
            print(f"  skip unknown task '{task}' ({len(by_label)} entries)", file=sys.stderr)
            continue
        md_file, primary, extras = TASK_TO_MD[task]
        md_path = os.path.join(RESULTS_DIR, md_file)
        block = render_block(task, primary, extras, by_label)
        changed = update_md(md_path, block)
        flag = "updated" if changed else "no change"
        print(f"  {task:<14} -> {md_file:<16}  ({len(by_label)} labels, {flag})", file=sys.stderr)


if __name__ == "__main__":
    main()
