"""Pretokenized dolmino-mix-1124 loader for midtraining.

Shards are pre-shuffled offline by `scripts/bake_global_shuffle.py` so reading
them sequentially matches OLMo's PCG64 shuffled-index order byte-for-byte.
The repetition filter (`find_periodic_sequences`) is ported verbatim from
`olmo/data/util.py`; matched sequences are dropped (vs OLMo's loss-mask).
"""

import json
import os
import subprocess
import tempfile
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def load_preshuffled_dolmino(
    gcs_bucket,
    max_length=4096,
    start_index=0,
    instance_filter=True,
    repetition_min_period=1,
    repetition_max_period=13,
    repetition_max_count=32,
    prefix="dolmino-tokenized-shuffled",
    prefetch=2,
):
    """Stream pre-shuffled dolmino shards from `gs://{gcs_bucket}/{prefix}/seq{max_length}/`.

    start_index is a global sequence offset for resuming mid-stream.
    """
    gcs_prefix = f"gs://{gcs_bucket}/{prefix}/seq{max_length}"

    manifest = _read_gcs_json(f"{gcs_prefix}/manifest.json")
    if not manifest.get("globally_preshuffled"):
        raise RuntimeError(
            f"{gcs_prefix}/manifest.json is not marked globally_preshuffled. "
            f"Run `python scripts/bake_global_shuffle.py --bucket {gcs_bucket}` first."
        )
    num_shards = manifest["num_shards"]
    shard_size = manifest["shard_size"]
    total_seqs = manifest["total_sequences"]
    total_tokens = manifest["total_tokens"]
    seed = manifest.get("shuffle_seed")
    print(
        f"  Pre-shuffled dolmino: {total_seqs:,} sequences, "
        f"{total_tokens:,} tokens, {num_shards} shards (PCG64 seed={seed})"
    )

    shard_paths = [f"{gcs_prefix}/shard_{i:05d}.npy" for i in range(num_shards)]

    start_shard = start_index // shard_size
    start_row = start_index - start_shard * shard_size
    if start_shard >= num_shards:
        raise ValueError(
            f"start_index={start_index:,} is past the end of the dataset "
            f"({total_seqs:,} sequences)"
        )

    return _PreshuffledIterator(
        shard_paths=shard_paths,
        total_sequences=total_seqs,
        shard_size=shard_size,
        start_shard=start_shard,
        start_row=start_row,
        instance_filter=instance_filter,
        repetition_min_period=repetition_min_period,
        repetition_max_period=repetition_max_period,
        repetition_max_count=repetition_max_count,
        prefetch=prefetch,
    )


class _PreshuffledIterator:
    """Sequential iterator over pre-shuffled shards with single-shard prefetch."""

    def __init__(
        self,
        shard_paths,
        total_sequences,
        shard_size,
        start_shard=0,
        start_row=0,
        instance_filter=True,
        repetition_min_period=1,
        repetition_max_period=13,
        repetition_max_count=32,
        prefetch=2,
    ):
        self.shard_paths = shard_paths
        self.total_sequences = total_sequences
        self.shard_size = shard_size
        self.start_shard = start_shard
        self.start_row = start_row
        self.instance_filter = instance_filter
        self.rep_min = repetition_min_period
        self.rep_max = repetition_max_period
        self.rep_count = repetition_max_count
        self.prefetch = max(1, prefetch)
        # Raw-sequence position of the NEXT row to read, in global units
        # (shard_idx * shard_size + row_idx). Counts every row the iterator
        # advances past — including the <1% dropped by _validate_instance —
        # so it is the exact value to pass as `start_index` on resume. Saved
        # in checkpoint meta.json to avoid the filter-drift rewind that
        # `step * effective_batch` suffers from.
        self.raw_index = start_shard * shard_size + start_row

    def __iter__(self):
        n = len(self.shard_paths)
        pool = ThreadPoolExecutor(max_workers=self.prefetch)

        def _fetch(i):
            t0 = time.time()
            arr = _read_gcs_npy(self.shard_paths[i])
            return i, arr, time.time() - t0

        # Kick off a small prefetch queue.
        in_flight = []
        next_to_submit = self.start_shard
        for _ in range(self.prefetch):
            if next_to_submit >= n:
                break
            in_flight.append(pool.submit(_fetch, next_to_submit))
            next_to_submit += 1

        n_dropped = 0
        n_yielded = 0
        try:
            while in_flight:
                fut = in_flight.pop(0)
                i, arr, dt = fut.result()
                fname = self.shard_paths[i].rsplit("/", 1)[-1]
                print(
                    f"  [shard {i+1}/{n}] {fname} ({dt:.1f}s, "
                    f"{len(arr)} seqs)",
                    flush=True,
                )
                if next_to_submit < n:
                    in_flight.append(pool.submit(_fetch, next_to_submit))
                    next_to_submit += 1

                first_row = self.start_row if i == self.start_shard else 0
                for row_idx in range(first_row, len(arr)):
                    seq = arr[row_idx]
                    # Advance before potentially dropping/yielding so that
                    # a consumer reading `self.raw_index` between yields
                    # sees the "next to read" position.
                    self.raw_index = i * self.shard_size + row_idx + 1
                    if self.instance_filter and not _validate_instance(
                        seq, self.rep_min, self.rep_max, self.rep_count
                    ):
                        n_dropped += 1
                        continue
                    n_yielded += 1
                    yield seq
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        if n_dropped > 0:
            total = n_yielded + n_dropped
            print(
                f"  instance_filter dropped {n_dropped}/{total} "
                f"({100 * n_dropped / total:.2f}%)",
                flush=True,
            )

    def __len__(self):
        consumed = self.start_shard * self.shard_size + self.start_row
        return self.total_sequences - consumed


def pretokenized_batch_iterator(dataset, batch_size, cycle=False):
    """Yield numpy batches from a pretokenized dataset.

    Each item from dataset is a 1D array of token IDs (length max_length).
    Yields dicts with input_ids [batch_size, max_length] and
    attention_mask [batch_size, max_length] (all ones).
    """
    while True:
        batch = []
        for seq in dataset:
            batch.append(seq)
            if len(batch) == batch_size:
                input_ids = np.stack(batch, axis=0).astype(np.int32)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": np.ones_like(input_ids),
                }
                batch = []
        if not cycle:
            break


# OLMo instance_filter — verbatim port of olmo/data/util.py.

RepetitionTuple = namedtuple("RepetitionTuple", ["start", "end", "period", "times"])


def _find_end_first_consecutive_true(arr):
    """End position of the first consecutive run of True in `arr`."""
    if not arr[0]:
        return 0
    prog = np.cumsum(arr)
    if prog[-1] == len(arr):
        return len(arr)
    true_locs = np.where(prog[:-1] == prog[1:])[0]
    return true_locs[0] + 1


def _find_start_last_consecutive_true(arr):
    """Start position of the last consecutive run of True in `arr`."""
    reverse = _find_end_first_consecutive_true(arr[::-1])
    return len(arr) - reverse if reverse > 0 else -1


def _group_consecutive_values(arr, stepsize=1):
    return np.split(arr, np.where(np.diff(arr) != stepsize)[0] + 1)


def _find_periodic_sequences(arr, max_period, min_period=1, mask_value=-1):
    """Yield RepetitionTuples describing any periodic runs in `arr`."""
    if (arr == mask_value).sum() > 0:
        raise ValueError("`mask_value` is in the array")

    max_period = min(max_period, len(arr) // 3)

    for period in range(min_period, max_period + 1):
        padded_arr = np.pad(
            arr, (0, period - (len(arr) % period)), constant_values=mask_value
        )
        shaped_arr = padded_arr.reshape(-1, period)

        is_equal_to_prev_row = shaped_arr == np.roll(shaped_arr, shift=1, axis=0)
        rows_with_period, *_ = np.where(is_equal_to_prev_row.all(axis=1))

        if len(rows_with_period) == 0:
            continue

        where_true_consecutive = _group_consecutive_values(rows_with_period)

        for sequence in where_true_consecutive:
            start_row = sequence[0]
            end_row = sequence[-1]

            start_offset = _find_start_last_consecutive_true(
                is_equal_to_prev_row[start_row - 1]
            )
            start_offset = period - start_offset if start_offset > 0 else 0

            end_offset = _find_end_first_consecutive_true(
                is_equal_to_prev_row[end_row + 1]
            )

            start_pos = (start_row - 1) * period - start_offset
            end_pos = ((end_row + 1) * period) + end_offset

            out = RepetitionTuple(
                start=start_pos,
                end=end_pos,
                period=period,
                times=(end_pos - start_pos) // period,
            )
            if out.times > 2:
                yield out


def _validate_instance(input_ids, min_period, max_period, max_count):
    """True if the instance passes OLMo's repetition filter, else False.

    Mirrors MemMapDataset._validate_instance: reject iff any detected periodic
    sequence has `times >= max_count`.
    """
    try:
        for m in _find_periodic_sequences(
            np.asarray(input_ids),
            max_period=max_period,
            min_period=min_period,
        ):
            if m.times >= max_count:
                return False
    except Exception:
        return True
    return True


# ==========================================================================
# GCS helpers
# ==========================================================================


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


def _read_gcs_npy(gcs_path):
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", gcs_path, tmp],
            check=True, capture_output=True,
        )
        return np.load(tmp)
    finally:
        os.unlink(tmp)
