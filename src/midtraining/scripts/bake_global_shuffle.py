"""Physically apply OLMo's PCG64 global shuffle to pretokenized dolmino shards.

Reads in-order shards produced by `repack_dolmino.py` and writes a new set of
shards where each row has been placed according to `PCG64(seed).shuffle` over
the global index space. After this runs, training can stream the output shards
sequentially from GCS with no on-the-fly shuffling — sequential reads of the
shuffled shards are mathematically identical to random reads of the unshuffled
shards in PCG64 order.

Disk-free streaming algorithm:
  * Never writes source shards to local disk.
  * Processes output shards in batches of `batch_output_shards` (default 100).
  * For each batch:
      - Allocate an in-memory output buffer per shard (shard_size × max_length).
      - Stream every source shard from GCS (parallel prefetch), and for each row
        decide — using the pre-computed permutation — whether it belongs in any
        output shard in the current batch. If so, scatter the row into the
        right position. Source shards are freed immediately after processing.
      - Once all source shards have been streamed, upload the finished output
        shards for this batch in parallel.
  * Re-reads source shards once per batch, so total ingress ≈
        num_passes × total_source_bytes
    (≈ 13 passes × 186 GB = 2.4 TB at batch_output_shards=100 for a 50B run).
    GCS intra-region egress/ingress within the same project is free.

Memory: roughly `batch_output_shards × shard_size × max_length × 4` bytes +
~300 MB per concurrent prefetch. Default (batch=100, prefetch=4) ≈ 17 GB.

Usage:
    python scripts/bake_global_shuffle.py \\
        --bucket $GCS_BUCKET \\
        --seed 42
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


def bake(
    bucket: str,
    source_prefix: str = "dolmino-tokenized-official",
    output_prefix: str = "dolmino-tokenized-shuffled",
    max_length: int = 4096,
    seed: int = 42,
    batch_output_shards: int = 100,
    download_prefetch: int = 4,
    upload_workers: int = 4,
    max_in_flight_uploads: int = 4,
    start_batch: int = 0,
):
    source_gcs = f"gs://{bucket}/{source_prefix}/seq{max_length}"
    output_gcs = f"gs://{bucket}/{output_prefix}/seq{max_length}"

    manifest = _read_gcs_json(f"{source_gcs}/manifest.json")
    if "shuffle_policy" not in manifest or "none at repack" not in manifest["shuffle_policy"]:
        raise RuntimeError(
            f"Source at {source_gcs} is not in pure file-stream order — this "
            f"bake script assumes source shards are in OLMo-exact dolmino50 order "
            f"with per-file floor packing. Re-run repack_dolmino.py first.\n"
            f"Current manifest shuffle_policy: {manifest.get('shuffle_policy')!r}"
        )

    num_shards = manifest["num_shards"]
    shard_size = manifest["shard_size"]
    total_seqs = manifest["total_sequences"]
    total_tokens = manifest["total_tokens"]
    print(
        f"Source: {total_seqs:,} sequences, {total_tokens:,} tokens, "
        f"{num_shards} shards @ {source_gcs}"
    )
    print(f"Output: {output_gcs}")
    print(f"Seed: {seed} (PCG64)")

    # Build global permutation. indices[k] = source position of output position k.
    # inv_indices[p] = output position where source position p lands.
    print(f"Building PCG64(seed={seed}) permutation of {total_seqs:,} indices...")
    t0 = time.time()
    rng = np.random.Generator(np.random.PCG64(seed=seed))
    indices = np.arange(total_seqs, dtype=np.uint32)
    rng.shuffle(indices)
    inv_indices = np.empty_like(indices)
    inv_indices[indices] = np.arange(total_seqs, dtype=np.uint32)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Precompute, for each source position p:
    #   out_shard(p)  = inv_indices[p] // shard_size
    #   out_row(p)    = inv_indices[p]  % shard_size
    # We keep these as uint32 arrays indexed by source position. ~48 MB each for 12M.
    out_shard_of_src = (inv_indices // shard_size).astype(np.uint32)
    out_row_of_src = (inv_indices % shard_size).astype(np.uint32)
    del inv_indices

    num_output_shards = (total_seqs + shard_size - 1) // shard_size
    # Size of each output shard (last one may be partial)
    out_sizes = [
        min(shard_size, total_seqs - k * shard_size)
        for k in range(num_output_shards)
    ]

    num_batches = (num_output_shards + batch_output_shards - 1) // batch_output_shards
    print(
        f"Output: {num_output_shards} shards in {num_batches} batches of "
        f"{batch_output_shards} (memory ≈ "
        f"{batch_output_shards * shard_size * max_length * 4 / 1e9:.1f} GB per batch)"
    )

    # Persist the index file so the shuffled manifest has audit trail.
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp_idx = f.name
    try:
        np.save(tmp_idx, indices)
        subprocess.run(
            ["gcloud", "storage", "cp", tmp_idx,
             f"{output_gcs}/global_indices_seed{seed}.npy"],
            check=True,
        )
        print(f"  Uploaded global_indices_seed{seed}.npy → {output_gcs}/")
    finally:
        os.unlink(tmp_idx)
    # We only needed indices for the upload; free it now.
    del indices

    for batch_idx in range(start_batch, num_batches):
        pass_start = batch_idx * batch_output_shards
        pass_end = min(pass_start + batch_output_shards, num_output_shards)
        _bake_one_batch(
            bucket=bucket,
            source_gcs=source_gcs,
            output_gcs=output_gcs,
            batch_idx=batch_idx,
            num_batches=num_batches,
            pass_start=pass_start,
            pass_end=pass_end,
            num_source_shards=num_shards,
            shard_size=shard_size,
            max_length=max_length,
            out_shard_of_src=out_shard_of_src,
            out_row_of_src=out_row_of_src,
            out_sizes=out_sizes,
            download_prefetch=download_prefetch,
            upload_workers=upload_workers,
            max_in_flight_uploads=max_in_flight_uploads,
        )

    # Write final manifest.
    new_manifest = {
        **manifest,
        "globally_preshuffled": True,
        "shuffle_seed": seed,
        "shuffle_algorithm": "numpy.random.PCG64.shuffle(np.arange(total_sequences, dtype=uint32))",
        "shuffled_from": source_prefix,
        "parity_note": (
            "Reading these shards sequentially is byte-identical to iterating the "
            "source shards in PCG64(seed).shuffle(np.arange(N)) order — matching "
            "olmo/data/iterable_dataset.py:91-98."
        ),
    }
    # Strip any legacy / stale fields that no longer describe this prefix.
    new_manifest.pop("file_shuffle_seed", None)
    new_manifest.pop("file_order", None)
    new_manifest.pop("packing", None)
    new_manifest["file_order"] = "PCG64(seed).shuffle of source file-order"
    new_manifest["packing"] = (
        "source shards float their rows into shuffled shards via PCG64 permutation; "
        "rows are still 4096-token int32 sequences"
    )
    _upload_manifest(new_manifest, f"{output_gcs}/manifest.json")
    print(f"\nAll batches done. Shuffled output at {output_gcs}/")


def _bake_one_batch(
    bucket,
    source_gcs,
    output_gcs,
    batch_idx,
    num_batches,
    pass_start,
    pass_end,
    num_source_shards,
    shard_size,
    max_length,
    out_shard_of_src,
    out_row_of_src,
    out_sizes,
    download_prefetch,
    upload_workers,
    max_in_flight_uploads,
):
    """Fill output shards [pass_start, pass_end) from a streaming source pass."""
    batch_size = pass_end - pass_start
    print(
        f"\n[batch {batch_idx+1}/{num_batches}] output shards "
        f"[{pass_start}, {pass_end})"
    )
    t_batch = time.time()

    # Allocate output buffers. Each is (out_size_k, max_length).
    buffers = [
        np.empty((out_sizes[pass_start + i], max_length), dtype=np.int32)
        for i in range(batch_size)
    ]
    # Track fill count per output shard so we can assert completeness.
    filled = np.zeros(batch_size, dtype=np.int64)

    dl_pool = ThreadPoolExecutor(max_workers=download_prefetch)

    def _fetch(i):
        t0 = time.time()
        arr = _download_shard_to_memory(f"{source_gcs}/shard_{i:05d}.npy")
        return i, arr, time.time() - t0

    # Maintain a bounded prefetch queue.
    in_flight = []
    next_to_submit = 0
    for _ in range(download_prefetch):
        if next_to_submit >= num_source_shards:
            break
        in_flight.append(dl_pool.submit(_fetch, next_to_submit))
        next_to_submit += 1

    total_src_rows = 0
    total_kept = 0
    try:
        while in_flight:
            src_idx, src_arr, dl_dt = in_flight.pop(0).result()
            if next_to_submit < num_source_shards:
                in_flight.append(dl_pool.submit(_fetch, next_to_submit))
                next_to_submit += 1

            # Slice of the per-row lookup tables that covers this source shard.
            global_src_start = src_idx * shard_size
            global_src_end = global_src_start + src_arr.shape[0]
            tgt_shard = out_shard_of_src[global_src_start:global_src_end]
            tgt_row = out_row_of_src[global_src_start:global_src_end]

            # Rows that land in any output shard in the current batch.
            in_batch = (tgt_shard >= pass_start) & (tgt_shard < pass_end)
            selected_src_rows = np.where(in_batch)[0]
            total_src_rows += src_arr.shape[0]

            if len(selected_src_rows) > 0:
                selected_tgt_shard = tgt_shard[selected_src_rows] - pass_start
                selected_tgt_row = tgt_row[selected_src_rows]
                # Scatter. No duplicate (tgt_shard, tgt_row) pairs since the
                # permutation is a bijection — safe to index in one go per shard.
                for batch_local_shard in range(batch_size):
                    mask = selected_tgt_shard == batch_local_shard
                    if not mask.any():
                        continue
                    rows_here_src = selected_src_rows[mask]
                    rows_here_tgt = selected_tgt_row[mask]
                    buffers[batch_local_shard][rows_here_tgt] = src_arr[rows_here_src]
                    filled[batch_local_shard] += len(rows_here_src)
                total_kept += len(selected_src_rows)

            if src_idx % 100 == 0 or src_idx == num_source_shards - 1:
                elapsed = time.time() - t_batch
                print(
                    f"  src {src_idx+1}/{num_source_shards} scanned "
                    f"(dl {dl_dt:.2f}s, kept {total_kept:,}/{total_src_rows:,} "
                    f"rows so far, {elapsed:.0f}s elapsed)",
                    flush=True,
                )
    finally:
        dl_pool.shutdown(wait=False, cancel_futures=True)

    # Completeness check — each output buffer must be fully filled.
    for i in range(batch_size):
        if filled[i] != out_sizes[pass_start + i]:
            raise RuntimeError(
                f"Output shard {pass_start + i} under-filled: "
                f"{filled[i]}/{out_sizes[pass_start + i]} rows"
            )

    # Upload this batch's output shards in parallel.
    print(f"  uploading {batch_size} output shards...")
    up_pool = ThreadPoolExecutor(max_workers=upload_workers)
    up_flight = []

    def _drain():
        while len(up_flight) >= max_in_flight_uploads:
            up_flight.pop(0).result()

    try:
        for i in range(batch_size):
            _drain()
            up_flight.append(
                up_pool.submit(
                    _save_and_upload, buffers[i], pass_start + i, output_gcs
                )
            )
        for fut in up_flight:
            fut.result()
    finally:
        up_pool.shutdown()

    # Free the batch's memory before the next pass.
    for i in range(batch_size):
        buffers[i] = None
    del buffers

    print(
        f"[batch {batch_idx+1}/{num_batches}] done in {time.time() - t_batch:.0f}s"
    )


def _download_shard_to_memory(gcs_path):
    """Download a shard to a temp file, load into memory, delete the temp file."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp = f.name
    try:
        last_err = None
        for attempt in range(3):
            try:
                subprocess.run(
                    ["gcloud", "storage", "cp", gcs_path, tmp],
                    check=True, capture_output=True,
                )
                # Load fully into RAM (no memmap) so we can delete the temp file.
                arr = np.load(tmp)
                return np.ascontiguousarray(arr)
            except subprocess.CalledProcessError as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to download {gcs_path}: {last_err}")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _save_and_upload(data, shard_idx, output_gcs):
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, data.astype(np.int32))
        tmp_path = f.name
    gcs_path = f"{output_gcs}/shard_{shard_idx:05d}.npy"
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
                time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Failed to upload shard {shard_idx} after 3 attempts: {last_err}"
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _read_gcs_json(gcs_path):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", gcs_path, tmp],
            check=True, capture_output=True,
        )
        with open(tmp) as f:
            return json.load(f)
    finally:
        os.unlink(tmp)


def _upload_manifest(data, gcs_path):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, indent=2)
        f.flush()
        tmp = f.name
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", tmp, gcs_path],
            check=True,
        )
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--source_prefix", default="dolmino-tokenized-official")
    parser.add_argument("--output_prefix", default="dolmino-tokenized-shuffled")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="PCG64 seed — must match the OLMo training config's seed field for "
             "byte-exact parity (stage-2 default is 42).",
    )
    parser.add_argument(
        "--batch_output_shards", type=int, default=100,
        help="Number of output shards to assemble per source-stream pass. "
             "Memory scales linearly (~160 MB × batch_output_shards @ shard_size=10000). "
             "Smaller = less memory, more passes over source data.",
    )
    parser.add_argument(
        "--download_prefetch", type=int, default=4,
        help="Number of concurrent in-flight source-shard downloads per pass.",
    )
    parser.add_argument("--upload_workers", type=int, default=4)
    parser.add_argument("--max_in_flight_uploads", type=int, default=4)
    parser.add_argument(
        "--start_batch", type=int, default=0,
        help="Resume: skip the first N batches. Deterministic from seed so "
             "partial runs can resume safely.",
    )
    args = parser.parse_args()
    bake(**vars(args))
