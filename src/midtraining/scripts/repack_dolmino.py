"""Repack OLMo2's official dolmino50 mix into fixed-length shards and upload to GCS.

Shards are written in OLMo-exact order: dolmino50.txt file order preserved
(matching OLMo's olmo/data/__init__.py:27-38), each file packed independently
via floor(tokens / max_length) with the per-file tail dropped (matching
OLMo's MemMapDataset._get_file_length). Sequences are emitted into shards of
`shard_size` rows in pure file-stream order — no shuffling here.

All shuffling happens at load time via
`src/midtraining/data.py:load_dolmino_global_index`, which builds a PCG64
permutation of [0, total_sequences) with the same seed OLMo uses (the
`seed:` field in the stage-2 config, default 42). With that loader, the
iteration order is byte-identical to what OLMo itself would produce given
the same dolmino50 files and seed.

Usage:
    python scripts/repack_dolmino.py --bucket $GCS_BUCKET
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

MANIFEST_URL = (
    "https://raw.githubusercontent.com/allenai/olmo-cookbook/"
    "main/src/cookbook/data/mixes/dolmino50.txt"
)
DATA_BASE = "https://olmo-data.org"  # CDN mirror of s3://ai2-llm/


def load_manifest():
    """Fetch the dolmino50.txt file manifest. Returns list of (source, url) tuples preserving order."""
    r = requests.get(MANIFEST_URL, timeout=60)
    r.raise_for_status()
    current_source = None
    entries = []
    for line in r.text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#SOURCE:"):
            current_source = line.split(":", 1)[1].strip().split(" ")[0].rstrip("/")
            continue
        if line.startswith("s3://ai2-llm/"):
            path = line.replace("s3://ai2-llm/", "")
            entries.append((current_source or "unknown", f"{DATA_BASE}/{path}"))
    return entries


def repack(
    bucket: str,
    max_length: int = 4096,
    shard_size: int = 10_000,
    max_tokens: int = 50_000_000_000,
    start_shard: int = 0,
    upload_workers: int = 4,
    max_in_flight: int = 4,
    prefix: str = "dolmino-tokenized-official",
):
    gcs_prefix = f"gs://{bucket}/{prefix}/seq{max_length}"

    print("Loading dolmino50 manifest...")
    entries = load_manifest()
    print(f"  {len(entries)} files across {len({s for s, _ in entries})} sources")
    print("  file order: dolmino50.txt verbatim (OLMo-exact)")

    max_sequences = max_tokens // max_length
    print(f"Target: {max_tokens:,} tokens = {max_sequences:,} sequences of {max_length}")
    print(f"Shard size: {shard_size} sequences (~{shard_size * max_length * 4 / 1e6:.0f} MB)")
    print(f"GCS destination: {gcs_prefix}/")
    if start_shard > 0:
        print(
            f"Resume mode: only uploading shards >= {start_shard} "
            f"(shards 0..{start_shard-1} already in GCS). Per-file packing is stateless, "
            f"so shards are byte-identical whether resumed or fresh."
        )
    print(f"Upload workers: {upload_workers}, max in-flight shards: {max_in_flight}")

    upload_pool = ThreadPoolExecutor(max_workers=upload_workers)
    in_flight: deque = deque()

    # Pending sequences [N, max_length] waiting to fill a shard.
    # OLMo's MemMapDataset slices each file at floor(len/max_length) and drops
    # the tail remainder per file — no cross-file concatenation. We match that.
    pending = np.empty((0, max_length), dtype=np.int32)

    shard_idx = 0
    total_sequences = 0
    total_bytes = 0
    t0 = time.time()

    for i, (source, url) in enumerate(entries):
        if total_sequences >= max_sequences:
            break

        t_dl = time.time()
        arr = None
        last_err = None
        for attempt in range(3):
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
                tmp = f.name
            try:
                resp = requests.get(url, stream=True, timeout=600)
                resp.raise_for_status()
                file_bytes = 0
                with open(tmp, "wb") as out:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        out.write(chunk)
                        file_bytes += len(chunk)
                arr = np.fromfile(tmp, dtype=np.uint32).astype(np.int32)
                total_bytes += file_bytes
                break
            except Exception as e:
                last_err = e
                print(f"  [retry {attempt+1}/3] {url}: {e}", flush=True)
                time.sleep(2 ** attempt)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        if arr is None:
            raise RuntimeError(f"Failed to download {url} after 3 attempts: {last_err}")
        dl_time = time.time() - t_dl

        n_full = len(arr) // max_length
        if n_full > 0:
            new_seqs = arr[: n_full * max_length].reshape(n_full, max_length)
            room = max_sequences - total_sequences
            if n_full > room:
                new_seqs = new_seqs[:room]
                n_full = room
            pending = np.concatenate([pending, new_seqs], axis=0)
            total_sequences += n_full

        # Emit full shards in pure file-stream order. The global PCG64 shuffle
        # that matches OLMo is applied at load time, not here.
        while len(pending) >= shard_size:
            shard = pending[:shard_size].copy()
            if shard_idx >= start_shard:
                while len(in_flight) >= max_in_flight:
                    in_flight.popleft().result()
                fut = upload_pool.submit(_save_shard, shard, shard_idx, gcs_prefix)
                in_flight.append(fut)
            pending = pending[shard_size:]
            shard_idx += 1

        elapsed = time.time() - t0
        tps = total_sequences * max_length / elapsed if elapsed > 0 else 0
        pct = total_sequences / max_sequences * 100
        eta_h = (max_sequences - total_sequences) * max_length / tps / 3600 if tps > 0 else 0
        src_name = source.rsplit("/", 1)[-1] if source else "?"
        fname = url.rsplit("/", 1)[-1]
        print(
            f"  [{i+1}/{len(entries)}] {src_name}/{fname} "
            f"({arr.nbytes/1e6:.0f}MB, dl {dl_time:.1f}s) | "
            f"{total_sequences:,}/{max_sequences:,} seqs ({pct:.1f}%) | "
            f"{tps:,.0f} tok/s | {elapsed:.0f}s | ETA {eta_h:.1f}h",
            flush=True,
        )

    # Final partial shard (fewer than shard_size rows).
    if len(pending) > 0:
        if shard_idx >= start_shard:
            while len(in_flight) >= max_in_flight:
                in_flight.popleft().result()
            fut = upload_pool.submit(_save_shard, pending.copy(), shard_idx, gcs_prefix)
            in_flight.append(fut)
        shard_idx += 1

    print("Waiting for uploads to finish...")
    while in_flight:
        in_flight.popleft().result()
    upload_pool.shutdown()

    manifest = {
        "max_length": max_length,
        "shard_size": shard_size,
        "num_shards": shard_idx,
        "total_sequences": total_sequences,
        "total_tokens": total_sequences * max_length,
        "source": "allenai/olmo-cookbook/src/cookbook/data/mixes/dolmino50.txt",
        "manifest_url": MANIFEST_URL,
        "file_order": "dolmino50.txt verbatim (OLMo-exact)",
        "packing": "per-file floor(len/max_length); tail remainder dropped per file",
        "shuffle_policy": (
            "none at repack; load-time PCG64(seed) global index reproduces OLMo's "
            "iteration order byte-for-byte"
        ),
        "num_source_files": len(entries),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        subprocess.run(
            ["gcloud", "storage", "cp", f.name, f"{gcs_prefix}/manifest.json"],
            check=True,
        )
        os.unlink(f.name)

    elapsed = time.time() - t0
    print(
        f"\nDone: {total_sequences:,} sequences ({total_sequences * max_length:,} tokens) "
        f"in {shard_idx} shards, {elapsed:.0f}s"
    )
    print(f"Downloaded: {total_bytes / 1e9:.1f} GB from olmo-data.org")
    print(f"Saved to {gcs_prefix}/")


def _save_shard(sequences, shard_idx, gcs_prefix):
    """Save a shard as numpy array and upload to GCS. Thread-safe."""
    if not isinstance(sequences, np.ndarray):
        sequences = np.array(sequences, dtype=np.int32)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, sequences.astype(np.int32))
        tmp_path = f.name
    gcs_path = f"{gcs_prefix}/shard_{shard_idx:05d}.npy"
    try:
        last_err = None
        for attempt in range(3):
            try:
                subprocess.run(
                    ["gcloud", "storage", "cp", tmp_path, gcs_path],
                    check=True, capture_output=True,
                )
                return
            except subprocess.CalledProcessError as e:
                last_err = e
                print(f"  [upload retry {attempt+1}/3] {gcs_path}: {e.stderr[:200] if e.stderr else e}", flush=True)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to upload shard {shard_idx} after 3 attempts: {last_err}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--max_tokens", type=int, default=50_000_000_000)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--shard_size", type=int, default=10_000)
    parser.add_argument(
        "--start_shard",
        type=int,
        default=0,
        help="Skip uploading shards < this index. Safe to resume freely — per-file "
             "packing is stateless, so shards are byte-identical whether resumed or fresh.",
    )
    parser.add_argument("--upload_workers", type=int, default=4)
    parser.add_argument("--max_in_flight", type=int, default=4)
    parser.add_argument(
        "--prefix",
        type=str,
        default="dolmino-tokenized-official",
        help="GCS sub-prefix to write to (bucket/<prefix>/seq<max_length>/...)",
    )
    args = parser.parse_args()
    repack(**vars(args))
