"""Midtraining (continued pretraining) on dolmino-mix-1124.

Custom training loop using EasyDeL only for model loading.
All training logic uses plain JAX/optax — no ejit, no grain threads,
no SFTTrainer. This avoids the multi-host TPU desync issues in EasyDeL.
"""

import argparse
import builtins
import functools
import gc
import math
import os
import re
import shutil
import subprocess
import tempfile
import time

import jax

# Must run before any easydel/sft.model import — those touch XLA on import
# in newer easydel versions and would forbid distributed.initialize() afterwards.
# Guard for re-import: this module is also imported by sft.train, which calls
# initialize() itself.
if not jax._src.distributed.global_state.client:
    jax.distributed.initialize()

from jax.experimental.multihost_utils import sync_global_devices

import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from etils import epath
from jax.sharding import NamedSharding, PartitionSpec
from typing import Any, NamedTuple

import wandb

from midtraining.data import (
    load_preshuffled_dolmino,
    pretokenized_batch_iterator,
)
from sft.model import load_model
from sft.registry import get_model_spec


TOTAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 4096

OLMO2_1B_SCHEDULE = {
    "peak_lr": 4e-4,
    "warmup_steps": 2000,
    "total_steps": 2_384_186,
    "alpha_f": 0.1,
    "weight_decay": 0.1,
}


# --- Utilities ---

def compute_cosine_lr(step, peak_lr, warmup_steps, total_steps, alpha_f):
    """Compute LR at a given step on the original OLMo2 cosine schedule."""
    if step <= warmup_steps:
        return peak_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return alpha_f * peak_lr + 0.5 * (1 - alpha_f) * peak_lr * (1 + math.cos(math.pi * progress))


def parse_step_from_checkpoint(name):
    match = re.search(r"step(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse step from checkpoint name: {name}")
    return int(match.group(1))


def verify_fsdp(graphstate, opt_state, mesh):
    """Verify FSDP sharding and print diagnostics. Call on all hosts."""
    param_leaves = jax.tree.leaves(graphstate)
    opt_leaves = jax.tree.leaves(opt_state)

    # Check param sharding
    total_param_bytes = sum(l.size * l.dtype.itemsize for l in param_leaves)
    local_param_bytes = sum(
        sum(s.data.nbytes for s in l.addressable_shards)
        for l in param_leaves
    )
    num_devices = jax.device_count()
    num_local = jax.local_device_count()
    fsdp_ratio = total_param_bytes / local_param_bytes if local_param_bytes else 0

    # Count sharded vs replicated
    sharded = sum(
        1 for l in param_leaves
        if any("fsdp" in str(s) for s in l.sharding.spec if s is not None)
    )

    print(f"  FSDP verification:")
    print(f"    Params: {total_param_bytes / 1e9:.2f} GB total, "
          f"{local_param_bytes / 1e6:.0f} MB local ({num_local} devices)")
    print(f"    Sharding ratio: {fsdp_ratio:.1f}x "
          f"(expect {num_devices / num_local:.0f}x for {num_devices}-way FSDP)")
    print(f"    Sharded params: {sharded}/{len(param_leaves)}")
    print(f"    Optimizer leaves: {len(opt_leaves)}")

    if fsdp_ratio < num_devices / num_local * 0.5:
        print(f"    WARNING: FSDP ratio too low — params may be replicated!")


def _grad_path_label(path):
    """Stringify a jax pytree keypath into a stable W&B metric suffix."""
    parts = []
    for k in path:
        if hasattr(k, "key"):
            parts.append(str(k.key))
        elif hasattr(k, "idx"):
            parts.append(str(k.idx))
        elif hasattr(k, "name"):
            parts.append(str(k.name))
        else:
            parts.append(str(k))
    return ".".join(parts)


def _per_leaf_norms(tree, prefix):
    """Per-leaf L2 norms + a global L2 norm for a pytree.

    Emits one metric per leaf (~179 for this model) plus a global norm.
    Sums done in float32 to avoid bf16/fp16 underflow for small grads.
    Only called from the norm-producing train_step variant (log steps), so
    the per-leaf collective overhead only fires on log steps.
    """
    sq_tree = jax.tree.map(
        lambda x: jnp.sum(jnp.square(x.astype(jnp.float32))), tree,
    )
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(sq_tree)
    out = {}
    global_sq = jnp.float32(0.0)
    for path, sq in leaves_with_path:
        label = _grad_path_label(path) or "root"
        out[f"{prefix}.{label}"] = jnp.sqrt(sq)
        global_sq = global_sq + sq
    out[f"{prefix}.global"] = jnp.sqrt(global_sq)
    return out


def get_hbm_stats():
    """Get HBM stats for local devices. Returns dict for wandb logging."""
    stats = {}
    for d in jax.local_devices():
        mem = d.memory_stats()
        if mem:
            stats[f"hbm/device_{d.id}_used_gb"] = mem.get("bytes_in_use", 0) / 1e9
            stats[f"hbm/device_{d.id}_peak_gb"] = mem.get("peak_bytes_in_use", 0) / 1e9
    if stats:
        used_vals = [v for k, v in stats.items() if "used_gb" in k]
        peak_vals = [v for k, v in stats.items() if "peak_gb" in k]
        stats["hbm/avg_used_gb"] = sum(used_vals) / len(used_vals)
        stats["hbm/max_peak_gb"] = max(peak_vals)
    return stats


def _tree_local_bytes(tree):
    """Sum per-host HBM bytes of all arrays in a pytree (handles sharding)."""
    total = 0
    for x in jax.tree.leaves(tree):
        if hasattr(x, "addressable_shards"):
            for shard in x.addressable_shards:
                total += shard.data.nbytes
        elif hasattr(x, "nbytes"):
            total += x.nbytes
    return total


def _tree_global_bytes(tree):
    """Sum logical (all-host) bytes of all arrays in a pytree."""
    return sum(x.nbytes for x in jax.tree.leaves(tree) if hasattr(x, "nbytes"))


def print_memory_breakdown(graphstate, opt_state, microbatches, mesh=None):
    """Print per-chip HBM breakdown. Call after first training step.

    Explains where the ~10GB/chip actually goes under FSDP:
      - Persistent sharded state (weights + optimizer, sharded across fsdp axis)
      - Ephemeral all-gather buffers (full unsharded weights + gradients during fwd/bwd)
      - Activations (dots-with-no-batch-dims saved; rest recomputed in backward)
      - XLA runtime pool + fragmentation
    """
    num_devices = jax.device_count()
    num_local = jax.local_device_count()

    weights_local = _tree_local_bytes(graphstate)
    weights_global = _tree_global_bytes(graphstate)
    opt_local = _tree_local_bytes(opt_state)
    opt_global = _tree_global_bytes(opt_state)
    batch_local = _tree_local_bytes(microbatches)

    # Per-chip figures
    weights_per_chip = weights_local / num_local
    opt_per_chip = opt_local / num_local
    batch_per_chip = batch_local / num_local

    # Expected all-gather buffer size: under pure FSDP each chip must temporarily
    # materialize the full unsharded weight tensor for fwd and the full gradient
    # tensor for bwd before reduce-scatter.
    gathered_weights_per_chip = weights_global  # full replica on each chip at peak
    gathered_grads_per_chip = weights_global    # same for grads pre-RS

    # Count sharded params / FSDP ratio
    param_leaves = jax.tree.leaves(graphstate)
    fsdp_sharded = sum(
        1 for l in param_leaves
        if any("fsdp" in str(s) for s in l.sharding.spec if s is not None)
    )
    tp_sharded = sum(
        1 for l in param_leaves
        if any("tp" in str(s) for s in l.sharding.spec if s is not None)
    )
    fsdp_ratio = (weights_global / weights_local) if weights_local else 0
    expected_fsdp_ratio = num_devices / num_local

    # Real device memory — collect everything memory_stats() gives us.
    # Keys (all per-chip, bytes):
    #   bytes_in_use         — JAX arrays currently live in XLA's pool
    #   peak_bytes_in_use    — high water mark of the above
    #   bytes_reserved       — size of XLA's preallocated pool (the real HBM claim)
    #   peak_bytes_reserved  — high water mark of the above
    #   bytes_limit          — total HBM on the chip
    #   largest_free_block_bytes — biggest contiguous free chunk (fragmentation signal)
    #   num_allocs           — live allocations in the pool
    stats = []
    for d in jax.local_devices():
        m = d.memory_stats() or {}
        stats.append({
            "in_use":  m.get("bytes_in_use", 0),
            "peak":    m.get("peak_bytes_in_use", 0),
            "reserved": m.get("bytes_reserved", 0),
            "peak_reserved": m.get("peak_bytes_reserved", 0),
            "limit":   m.get("bytes_limit", 0),
            "largest_free": m.get("largest_free_block_bytes", 0),
            "num_allocs": m.get("num_allocs", 0),
        })

    def _avg(key): return sum(s[key] for s in stats) / len(stats) if stats else 0
    def _max(key): return max((s[key] for s in stats), default=0)

    in_use_avg   = _avg("in_use")
    peak_max     = _max("peak")
    reserved_max = _max("peak_reserved") or _max("reserved")
    limit_avg    = _avg("limit")
    largest_free_min = min((s["largest_free"] for s in stats), default=0)

    # What's left *outside* XLA's pool — this is what determines whether the
    # compiled program can even be loaded onto the chip. Pool-relative OOMs
    # look different from outside-pool OOMs; see bottom of printout.
    free_outside_pool = max(0, limit_avg - reserved_max)

    tracked_per_chip = weights_per_chip + opt_per_chip + batch_per_chip
    unaccounted_peak = max(0, peak_max - tracked_per_chip)

    def _gb(x): return f"{x/1e9:>6.2f} GB"
    def _mb(x): return f"{x/1e6:>7.1f} MB"

    bar = "=" * 72
    print(bar)
    print(f"Memory breakdown — host {jax.process_index()}, "
          f"{num_local} local / {num_devices} total chips")
    if mesh is not None:
        print(f"  mesh axes: {mesh.axis_names}  shape: {tuple(mesh.shape[a] for a in mesh.axis_names)}")
    print(f"  params: {len(param_leaves)} leaves  "
          f"fsdp-sharded={fsdp_sharded}  tp-sharded={tp_sharded}  "
          f"fsdp-ratio={fsdp_ratio:.1f}× (expected {expected_fsdp_ratio:.0f}×)")
    if fsdp_ratio < expected_fsdp_ratio * 0.9:
        print(f"  WARNING: FSDP ratio below expected — some params may be replicated!")
    print(f"  {'-'*66}")
    print(f"  PERSISTENT (sharded, at rest) per-chip:")
    print(f"    weights (local shard):      {_gb(weights_per_chip)}  "
          f"(global {weights_global/1e9:.2f} GB)")
    print(f"    optimizer (local shard):    {_gb(opt_per_chip)}  "
          f"(global {opt_global/1e9:.2f} GB)")
    print(f"    batch (live):               {_gb(batch_per_chip)}")
    print(f"    subtotal persistent:        {_gb(tracked_per_chip)}")
    print(f"  {'-'*66}")
    print(f"  EPHEMERAL (during step) expected per-chip:")
    print(f"    all-gathered weights:       {_gb(gathered_weights_per_chip)}  "
          f"(alloc'd during fwd)")
    print(f"    all-gathered gradients:     {_gb(gathered_grads_per_chip)}  "
          f"(alloc'd during bwd)")
    expected_peak_min = tracked_per_chip + gathered_weights_per_chip + gathered_grads_per_chip
    print(f"    expected min step-peak:     {_gb(expected_peak_min)}  "
          f"(persistent + gathered w/g; activations add on top)")
    print(f"  {'-'*66}")
    print(f"  OBSERVED per-chip (from jax memory_stats):")
    print(f"    HBM limit (chip total):     {_gb(limit_avg)}")
    print(f"    XLA pool reserved (peak):   {_gb(reserved_max)}  "
          f"← the real HBM claim from the compiler")
    print(f"    free outside pool:          {_gb(free_outside_pool)}  "
          f"← room for exec load & TPU runtime")
    print(f"    JAX bytes_in_use (avg):     {_gb(in_use_avg)}  "
          f"({int(_avg('num_allocs'))} allocs)")
    print(f"    JAX peak_bytes_in_use:      {_gb(peak_max)}")
    print(f"    largest free block (min):   {_mb(largest_free_min)}  "
          f"(fragmentation indicator)")
    print(f"    unaccounted @ in-use peak:  {_gb(unaccounted_peak)}  "
          f"(activations + XLA scratch)")
    pool_util    = reserved_max / limit_avg if limit_avg else 0
    in_use_util  = peak_max / reserved_max if reserved_max else 0
    print(f"    pool vs HBM limit:          {pool_util*100:>5.1f}%  "
          f"(how much of the chip XLA claimed)")
    print(f"    in-use vs pool:             {in_use_util*100:>5.1f}%  "
          f"(how full the pool actually gets)")
    # Actionable hints based on the numbers
    print(f"  {'-'*66}")
    if pool_util > 0.95:
        headroom_mb = free_outside_pool / 1e6
        print(f"  WARNING: pool is using {pool_util*100:.1f}% of HBM — only "
              f"{headroom_mb:.0f} MB free outside pool. Program load or rare "
              f"runtime allocs can OOM. If you see 'largest contiguous region "
              f"of free memory is X MB' errors, drop TOTAL_BATCH_SIZE one step.")
    elif pool_util < 0.50:
        print(f"  HINT: pool is only {pool_util*100:.0f}% of HBM — lots of "
              f"headroom. TOTAL_BATCH_SIZE could likely increase.")
    else:
        print(f"  OK: pool at {pool_util*100:.0f}% of HBM — comfortable margin.")
    print(bar, flush=True)


def save_to_gcs(local_dir, gcs_path):
    print(f"Uploading {local_dir} to {gcs_path}...", flush=True)
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", f"{local_dir}/*", gcs_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"Saved to {gcs_path}", flush=True)


def _opt_state_gcs_path(gcs_step_path):
    return epath.Path(gcs_step_path.rstrip("/")) / "opt_state"


def _save_opt_state_to_gcs(opt_state, gcs_step_path):
    """Write FSDP-sharded optimizer state directly to GCS via orbax.

    Must be called on ALL hosts in lockstep — orbax coordinates per-host
    shard writes so each chip persists only the slice of m/v it owns.
    """
    path = _opt_state_gcs_path(gcs_step_path)
    if jax.process_index() == 0 and path.exists():
        path.rmtree()
    sync_global_devices(f"opt_state_clear_{gcs_step_path}")
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(path, opt_state)
    ckptr.wait_until_finished()


def _restore_opt_state_from_gcs(opt_state_template, gcs_step_path, graphstate, mesh):
    """Restore opt_state from GCS, forcing each leaf onto its correct sharding.

    m/v shardings are read from `graphstate` (the source of truth — m/v are
    tree-isomorphic to params). `count` is replicated across the mesh. We don't
    trust `opt_state_template`'s shardings because old checkpoints were saved
    with single-device scalars/arrays and orbax doesn't always reshard them on
    restore.

    Returns a freshly-initialised opt_state if no orbax payload is present at
    the path (legacy checkpoints from before opt-state checkpointing landed).
    """
    path = _opt_state_gcs_path(gcs_step_path)
    if not path.exists():
        if jax.process_index() == 0:
            print(
                f"WARNING: no opt_state at {path} — falling back to fresh "
                f"optimizer init. Expect a loss spike (m/v reset, count=0)."
            )
        return opt_state_template
    # Build the abstract with shardings taken from `graphstate` (params) rather
    # than from `opt_state_template`. optimizer.init runs under jit but NNX
    # VariableState wrappers don't reliably propagate the params' sharding to
    # the zeros_like m/v, so the template's leaves can be committed to device 0.
    # orbax's StandardCheckpointer honors each ShapeDtypeStruct.sharding and
    # materialises leaves directly onto those devices via tensorstore, so
    # building the abstract with the correct target shardings is sufficient.
    replicated = NamedSharding(mesh, PartitionSpec())
    param_abstract = jax.tree.map(
        lambda p: jax.ShapeDtypeStruct(p.shape, p.dtype, sharding=p.sharding),
        graphstate,
    )
    count_abstract = jax.ShapeDtypeStruct((), jnp.int32, sharding=replicated)
    abstract = _FusedClippedAdamWState(
        m=param_abstract,
        v=param_abstract,
        count=count_abstract,
    )
    ckptr = ocp.StandardCheckpointer()
    return ckptr.restore(path, abstract)


def _save_checkpoint_step(state, graphstate, opt_state, tokenizer, mesh, step, total_tokens, raw_index, save_gcs_path):
    """Save an intermediate training checkpoint to GCS. Resumable.

    Layout written to {save_gcs_path}/step_{step:07d}/:
      - model/                      (tensorstore FSDP-sharded weights, via save_pretrained)
      - tokenizer.json, config.json, etc. (identical on every host)
      - opt_state/                  (orbax FSDP-sharded m/v/count)
      - meta.json                   ({step, total_tokens, raw_index})
      - _COMPLETE                   (written LAST, from host 0, after every host has finished)

    `raw_index` is the dataset iterator's next-to-read raw-sequence position
    at the moment of save. It counts every row the iterator has stepped past,
    including those dropped by OLMo's `_validate_instance` filter, so passing
    it back as `start_index` on resume is bit-exact — no ~1% rewind into data
    the model already trained on.

    `find_latest_checkpoint` only accepts dirs with _COMPLETE so a crashed
    mid-save leaves a junk dir that we skip on resume.
    """
    import json as _json
    local_dir = f"/tmp/midtrain_ckpt_step/{step}"
    shutil.rmtree(local_dir, ignore_errors=True)
    os.makedirs(local_dir, exist_ok=True)

    # 1. Every host writes its FSDP shard files to its local_dir.
    with mesh:
        model = state.merge(graphstate)
        model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    if jax.process_index() == 0:
        with open(os.path.join(local_dir, "meta.json"), "w") as f:
            _json.dump(
                {"step": step, "total_tokens": total_tokens, "raw_index": raw_index},
                f,
            )

    gcs_step_path = f"{save_gcs_path.rstrip('/')}/step_{step:07d}/"

    # 2. Every host uploads its local_dir. save_pretrained writes each chip's
    #    tensorstore shard (model/<param>/{fsdp_idx}.0) to the host that owns
    #    that chip, so host-0-only upload would drop 3/4 of the shards.
    #    Common files (config.json, tokenizer, meta.json on host 0) race
    #    harmlessly — content is identical or present on only one host.
    sync_global_devices(f"save_ckpt_pre_upload_{step}")
    save_to_gcs(local_dir, gcs_step_path)

    # 3. Cooperatively write opt_state (orbax handles multi-host shard writes).
    _save_opt_state_to_gcs(opt_state, gcs_step_path)

    # 4. Barrier so host 0 only stamps _COMPLETE after every host's upload
    #    is durable on GCS. External readers (and find_latest_checkpoint on
    #    resume) use this marker as the atomic "all shards present" signal.
    sync_global_devices(f"save_ckpt_post_upload_{step}")
    if jax.process_index() == 0:
        # Sibling marker (no trailing slash on step path), so find_latest
        # can spot it in a single `gcloud storage ls` of the parent dir.
        marker_path = f"{gcs_step_path.rstrip('/')}_COMPLETE"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(f"step={step}\ntotal_tokens={total_tokens}\n")
            tmp = f.name
        subprocess.run(
            ["gcloud", "storage", "cp", tmp, marker_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.unlink(tmp)

    sync_global_devices(f"save_ckpt_done_{step}")
    shutil.rmtree(local_dir, ignore_errors=True)


def find_latest_checkpoint(save_gcs_path):
    """Find the latest COMPLETED step_XXXXXXX checkpoint.

    Returns (step, total_tokens, raw_index, gcs_path) or (0, 0, None, None).
    `raw_index` is the dataset's next-to-read raw-sequence offset at save
    time (see `_save_checkpoint_step`). It's `None` for pre-raw-index
    checkpoints — the caller must then fall back to `step * effective_batch`.

    A checkpoint is considered complete only if its `_COMPLETE` sibling marker
    exists — the marker is stamped LAST by `_save_checkpoint_step` after every
    host has uploaded its FSDP shards, so its presence is an atomic "all shards
    durable" signal. Partial saves from a crashed/preempted host (directory
    exists but marker absent) are skipped.
    """
    import json as _json
    result = subprocess.run(
        ["gcloud", "storage", "ls", save_gcs_path.rstrip('/') + "/"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return 0, 0, None, None

    # Collect completed step numbers by looking for `step_XXXXXXX_COMPLETE`
    # marker objects that sit next to their `step_XXXXXXX/` directories.
    completed_steps = set()
    step_dirs = {}
    for line in result.stdout.strip().split("\n"):
        entry = line.strip().rstrip("/")
        name = entry.rsplit("/", 1)[-1]
        if name.startswith("step_") and name.endswith("_COMPLETE"):
            try:
                completed_steps.add(int(name[len("step_"):-len("_COMPLETE")]))
            except ValueError:
                pass
        elif name.startswith("step_"):
            try:
                step_dirs[int(name[len("step_"):])] = entry + "/"
            except ValueError:
                pass

    # Pick the largest step that is both a directory AND has a _COMPLETE marker.
    candidates = sorted(completed_steps & step_dirs.keys(), reverse=True)
    if not candidates:
        return 0, 0, None, None
    max_step = candidates[0]
    max_dir = step_dirs[max_step]

    # Read meta.json for total_tokens and raw_index (step is already known
    # from the marker). `raw_index` may be missing on pre-raw-index checkpoints.
    meta_gcs = max_dir + "meta.json"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        r = subprocess.run(["gcloud", "storage", "cp", meta_gcs, tmp], capture_output=True, check=False)
        if r.returncode != 0:
            return 0, 0, None, None
        with open(tmp) as f:
            meta = _json.load(f)
        return meta["step"], meta["total_tokens"], meta.get("raw_index"), max_dir
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _legacy_resume_raw_index(gcs_bucket, max_length, target_valid):
    """Recover the exact raw-sequence offset for a legacy checkpoint that
    lacks `raw_index` in meta.json.

    Streams shards from index 0 through `_validate_instance`, counting
    post-filter yields, and returns the iterator's `raw_index` the moment
    it reaches `target_valid` (== step * effective_batch). This is the
    only way to resume bit-exactly from a checkpoint saved before this
    field was recorded — `step * effective_batch` alone under-counts the
    ~<1% dropped by the filter and causes a rewind into already-trained
    data.

    Cost: one streaming pass over all shards from 0 up to the resume
    point (~410 shards / ~20 min for a step-8000 checkpoint). The
    returned raw_index is then used to build a fresh training iterator,
    which re-downloads those same shards — at one-epoch training this is
    just a ~2× download of the resume prefix, paid once.

    Runs independently on every host (deterministic, so all arrive at
    the same raw_index) — no cross-host broadcast required.
    """
    scout = load_preshuffled_dolmino(
        gcs_bucket, max_length=max_length, start_index=0,
    )
    valid = 0
    for _ in scout:
        valid += 1
        if valid >= target_valid:
            return scout.raw_index
    raise RuntimeError(
        f"dataset exhausted after {valid:,} valid sequences while "
        f"searching for target_valid={target_valid:,}. Is the checkpoint "
        f"from a different dataset than the current manifest?"
    )


# --- Training step — plain jax.jit ---

# --- Fused clipped AdamW — optax-compatible GradientTransformation ---

class _FusedClippedAdamWState(NamedTuple):
    m: Any
    v: Any
    count: jnp.ndarray


def fused_clipped_adamw(learning_rate, b1=0.9, b2=0.95, eps=1e-8,
                        weight_decay=0.0, mask=None, clip_norm=1.0):
    """Optax GradientTransformation that fuses `clip_by_global_norm + adamw`
    into a single pass over the gradient/param trees.

    optax.chain(clip_by_global_norm, adamw) traverses the tree ~3 times
    (clip-scale pass, adamw m/v/update pass, apply pass). This version
    folds clip + m/v update + weight-decay + lr scaling into one tree_map
    that returns the update delta directly, cutting per-leaf kernel count
    and letting XLA fuse across the whole AdamW formula per param.

    Interface matches optax: returns `(updates, new_state)`; pair with
    `optax.apply_updates(params, updates)` as usual.
    """
    def init_fn(params):
        return _FusedClippedAdamWState(
            m=jax.tree.map(lambda p: jnp.zeros_like(p, dtype=jnp.float32), params),
            v=jax.tree.map(lambda p: jnp.zeros_like(p, dtype=jnp.float32), params),
            count=jnp.zeros([], jnp.int32),
        )

    def update_fn(grads, state, params=None):
        assert params is not None, "fused_clipped_adamw needs params (for decay)"
        new_count = state.count + 1
        lr = learning_rate(new_count) if callable(learning_rate) else learning_rate
        lr = jnp.asarray(lr, dtype=jnp.float32)

        # Global grad L2 norm in ONE tree reduce (fp32 for stability under FSDP).
        # clip_norm=None skips the reduce entirely (saves a tree pass per step).
        if clip_norm is None:
            clip_factor = jnp.float32(1.0)
        else:
            g_sq_sums = jax.tree.map(
                lambda g: jnp.sum(jnp.square(g.astype(jnp.float32))), grads,
            )
            total_sq = jax.tree.reduce(
                lambda a, b: a + b, g_sq_sums, jnp.float32(0.0),
            )
            g_norm = jnp.sqrt(total_sq + jnp.float32(1e-30))
            clip_factor = jnp.minimum(jnp.float32(1.0), clip_norm / (g_norm + jnp.float32(1e-6)))

        # Weight-decay mask tree (bools, one per leaf). Computed once per step.
        if mask is None:
            decay_mask = jax.tree.map(lambda _: True, params)
        elif callable(mask):
            decay_mask = mask(params)
        else:
            decay_mask = mask

        bias_1 = jnp.float32(1.0) - jnp.asarray(b1, jnp.float32) ** new_count.astype(jnp.float32)
        bias_2 = jnp.float32(1.0) - jnp.asarray(b2, jnp.float32) ** new_count.astype(jnp.float32)
        b1_c = jnp.asarray(b1, jnp.float32)
        b2_c = jnp.asarray(b2, jnp.float32)

        def leaf_update(g, p, m_, v_, do_decay):
            # Fused: clip-scale grad, update m/v, compute step, add weight decay,
            # apply lr — all in one function, one kernel per leaf under jit.
            g32 = g.astype(jnp.float32) * clip_factor
            new_m = b1_c * m_ + (jnp.float32(1.0) - b1_c) * g32
            new_v = b2_c * v_ + (jnp.float32(1.0) - b2_c) * jnp.square(g32)
            step = (new_m / bias_1) / (jnp.sqrt(new_v / bias_2) + eps)
            step = step + jnp.where(
                do_decay, weight_decay * p.astype(jnp.float32), jnp.float32(0.0),
            )
            # Return the delta (optax convention: updates are added to params).
            update = (-lr * step).astype(p.dtype)
            return update, new_m, new_v

        # tree_map across 5 trees; each leaf returns a (update, m, v) tuple,
        # which becomes a tree where each leaf is a 3-tuple.
        per_leaf = jax.tree.map(
            leaf_update, grads, params, state.m, state.v, decay_mask,
        )

        is_tuple3 = lambda x: isinstance(x, tuple) and len(x) == 3
        updates = jax.tree.map(lambda t: t[0], per_leaf, is_leaf=is_tuple3)
        new_m = jax.tree.map(lambda t: t[1], per_leaf, is_leaf=is_tuple3)
        new_v = jax.tree.map(lambda t: t[2], per_leaf, is_leaf=is_tuple3)

        new_state = _FusedClippedAdamWState(m=new_m, v=new_v, count=new_count)
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def make_train_step(state, optimizer, gradient_accumulation_steps=1,
                    compute_norms=False):
    """Create a jitted training step with gradient accumulation.

    When gradient_accumulation_steps > 1, uses jax.lax.scan to process
    microbatches sequentially and accumulate gradients before the
    optimizer step. Memory cost is O(1 microbatch) for activations.

    `microbatches` should have shape [GA, micro_batch_size, seq_len].

    If `compute_norms=True`, the step also returns a dict of per-layer
    grad/update/param L2 norms. Those cost ~54 small scalar all-reduces
    per step, so keep this off on non-log steps.
    """
    # jax.checkpoint does double duty: (1) envelope remat around compute_loss,
    # (2) trace-scope isolation that flax nnx's UpdateContextManager needs to
    # avoid UnexpectedTracerError on step 2.
    # Policy `dots_with_no_batch_dims_saveable` saves only matmul outputs with
    # no batch dim (small, cheap to store, expensive to recompute) — a middle
    # ground between full remat and no remat.
    _remat_policy = jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    @functools.partial(jax.checkpoint, policy=_remat_policy)
    def compute_loss(params, batch):
        module = state.merge(params)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        logits = module(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
        # Avoid slicing logits (the large [bs, seq, vocab] tensor) —
        # pad the small labels/mask instead to keep seq_len aligned.
        shift_labels = jnp.pad(input_ids[:, 1:], ((0, 0), (0, 1)))
        shift_mask = jnp.pad(attention_mask[:, 1:], ((0, 0), (0, 1)))
        # Fused CE + OLMo2 z-loss: one pass over the [B, T, V] logits tensor
        # (vs two if we called softmax_cross_entropy_with_integer_labels and
        # then logsumexp separately — the former computes logsumexp internally).
        # OLMo2 z-loss multiplier 1e-5 per AI2's OLMo2-1B-stage2-seed42.yaml.
        log_z = jax.scipy.special.logsumexp(logits, axis=-1)
        logit_at_label = jnp.take_along_axis(
            logits, shift_labels[..., None], axis=-1,
        ).squeeze(-1)
        token_losses = log_z - logit_at_label  # CE per token
        z_loss_per_token = 1e-5 * (log_z ** 2)
        mask_sum = shift_mask.sum()
        ce_loss = (token_losses * shift_mask).sum() / mask_sum
        z_loss = (z_loss_per_token * shift_mask).sum() / mask_sum
        return ce_loss + z_loss

    # Mixed-precision helper — cast sharded fp32 master weights to bf16 for
    # compute. Done on the sharded arrays so each chip only casts its local
    # shard; XLA can then schedule the bf16 all-gather (cheaper bandwidth and
    # ~half the transient all-gather buffer vs fp32). Matches OLMo2's
    # MixedPrecision(param_dtype=bf16, ...). Integer params (none expected,
    # but just in case) pass through unchanged.
    def _to_bf16(x):
        if jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.bfloat16)
        return x

    # donate_argnums lets XLA reuse the input graphstate/opt_state buffers
    # for the output in-place. Saves an alloc+free per step, reduces transient
    # peak HBM, and opens up a few more fusion opportunities.
    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def train_step(graphstate, opt_state, microbatches):
        # Cast fp32 master weights → bf16 once per step (shared across all GA
        # microbatches). Grads from value_and_grad will come out in bf16 (same
        # dtype as the input); we upcast them to fp32 immediately for the
        # accumulator, optimizer, and apply_updates steps.
        bf16_graphstate = jax.tree.map(_to_bf16, graphstate)

        if gradient_accumulation_steps == 1:
            batch = jax.tree.map(lambda x: x[0], microbatches)
            loss, bf16_grads = jax.value_and_grad(compute_loss)(bf16_graphstate, batch)
            grads = jax.tree.map(lambda g: g.astype(jnp.float32), bf16_grads)
        else:
            def accum_step(carry, microbatch):
                loss, bf16_grads = jax.value_and_grad(compute_loss)(
                    bf16_graphstate, microbatch,
                )
                # Upcast to fp32 before accumulation for stability across GA.
                fp32_grads = jax.tree.map(lambda g: g.astype(jnp.float32), bf16_grads)
                new_grads = jax.tree.map(jnp.add, carry[0], fp32_grads)
                new_loss = carry[1] + loss
                return (new_grads, new_loss), None

            # fp32 accumulator — same dtype as the master weights.
            zero_grads = jax.tree.map(jnp.zeros_like, graphstate)
            (total_grads, total_loss), _ = jax.lax.scan(
                accum_step,
                (zero_grads, jnp.float32(0.0)),
                microbatches,
            )
            grads = jax.tree.map(
                lambda g: g / gradient_accumulation_steps, total_grads,
            )
            loss = total_loss / gradient_accumulation_steps

        if compute_norms:
            # Grads are post-accumulation (averaged over microbatches) and
            # pre-clip, so `grad_norm.global` is the value compared against
            # the clip threshold (1.0) inside optax.clip_by_global_norm.
            grad_metrics = _per_leaf_norms(grads, "grad_norm")

        updates, new_opt_state = optimizer.update(grads, opt_state, graphstate)
        new_graphstate = optax.apply_updates(graphstate, updates)

        if compute_norms:
            update_metrics = _per_leaf_norms(updates, "update_norm")
            param_metrics = _per_leaf_norms(new_graphstate, "param_norm")
            norm_metrics = {**grad_metrics, **update_metrics, **param_metrics}
        else:
            norm_metrics = {}

        return new_graphstate, new_opt_state, loss, norm_metrics

    return train_step


# --- Main training function ---

def midtrain(
    model, tokenizer, dataset, mesh,
    lr_start, lr_end, max_steps,
    total_batch_size=TOTAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    max_length=MAX_LENGTH,
    weight_decay=0.1,
    start_step=0,
    start_tokens=0,
    save_interval=1000,
    save_gcs_path=None,
    resume_gcs_dir=None,
):
    """Run continued pretraining with a custom JAX training loop."""
    is_main = jax.process_index() == 0
    effective_batch = total_batch_size * gradient_accumulation_steps

    # --- Optimizer (OLMo2 settings) ---
    # - betas (0.9, 0.95) per OLMo2 tech report
    # - gradient clipping 1.0
    # - weight decay: AI2's stage-2 uses decay_norm_and_bias=True, decay_embeddings=False.
    #   So we decay everything EXCEPT the embedding table (shape[0] == vocab_size),
    #   including 1-D norms and biases.
    schedule = optax.linear_schedule(lr_start, lr_end, max_steps)

    # --- Functional state ---
    state = model.to_state()
    graphstate = state.graphstate

    # Weight-decay mask: decay all params except the embedding matrix.
    vocab_size = model.config.vocab_size
    def _decay_mask(params):
        return jax.tree.map(
            lambda p: p.shape[0] != vocab_size,
            params,
        )

    # Fused clipped AdamW — same math as
    #   optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(...))
    # but folded into one tree_map to reduce per-leaf kernel count and let
    # XLA fuse the full AdamW formula across grad/m/v/param/decay_mask per
    # leaf. Swap to `optax.chain(...)` to A/B test.
    optimizer = fused_clipped_adamw(
        learning_rate=schedule,
        b1=0.9,
        b2=0.95,
        weight_decay=weight_decay,
        mask=_decay_mask,
        clip_norm=1.0,
    )
    # jit init with EXPLICIT out_shardings: without a hint, XLA's auto-sharding
    # doesn't propagate the params' FSDP sharding through `jnp.zeros_like` and
    # defaults to replicating m/v on every chip — which balloons per-chip HBM
    # by ~12GB (the global m+v) and OOMs the train step compile. We pin m/v to
    # the params' shardings and count to replicated-across-mesh.
    replicated = NamedSharding(mesh, PartitionSpec())
    param_shardings = jax.tree.map(lambda p: p.sharding, graphstate)
    opt_out_shardings = _FusedClippedAdamWState(
        m=param_shardings,
        v=param_shardings,
        count=replicated,
    )
    with mesh:
        opt_state = jax.jit(optimizer.init, out_shardings=opt_out_shardings)(graphstate)
    if resume_gcs_dir is not None:
        if is_main:
            print(f"Restoring optimizer state (m, v, count) from {resume_gcs_dir}")
        opt_state = _restore_opt_state_from_gcs(opt_state, resume_gcs_dir, graphstate, mesh)

    # --- Batch sharding: [GA, micro_bs, seq_len] -> shard micro_bs axis ---
    batch_sharding = NamedSharding(mesh, PartitionSpec(None, ("dp", "fsdp")))

    # --- Jitted train step ---
    # Two variants: the fast path skips per-layer norm all-reduces (~54 tiny
    # collectives that meaningfully add to step time). The norm-producing
    # variant is only used on the step before a log event. Both compile once,
    # so the upfront cost is doubled step-1 compile; steady-state is faster.
    train_step_fast = make_train_step(
        state, optimizer, gradient_accumulation_steps, compute_norms=False,
    )
    train_step_with_norms = make_train_step(
        state, optimizer, gradient_accumulation_steps, compute_norms=True,
    )

    # --- FSDP diagnostics ---
    if is_main:
        param_leaves = jax.tree.leaves(graphstate)
        print(f"  Params: {len(param_leaves)} leaves")
        print(f"  First param shape={param_leaves[0].shape} sharding={param_leaves[0].sharding}")
        verify_fsdp(graphstate, opt_state, mesh)

    # Log every 5 for the first 20 steps (so we can eyeball that gradients,
    # norms, loss, and tok/s look sane early), then every 20 thereafter.
    def should_log(s):
        if s == 1:
            return True
        if s <= 20:
            return s % 5 == 0
        return s % 20 == 0

    # --- Training loop ---
    data_iter = pretokenized_batch_iterator(dataset, effective_batch, cycle=True)
    step = start_step
    total_tokens = start_tokens

    if is_main and start_step > 0:
        print(f"Resuming from step {start_step} ({start_tokens:,} tokens)")

    with mesh:
        for batch_np in data_iter:
            if step >= max_steps:
                break

            # Reshape to microbatches: [GA, micro_bs, seq_len]
            microbatches_np = {
                k: v.reshape(gradient_accumulation_steps, total_batch_size, v.shape[-1])
                for k, v in batch_np.items()
            }
            microbatches = {
                k: jax.device_put(v, batch_sharding)
                for k, v in microbatches_np.items()
            }

            # All workers must pick the SAME step function (SPMD), so base
            # this purely on the step counter — not `is_main`. Logging itself
            # stays is_main-gated below.
            step_fn = train_step_with_norms if should_log(step + 1) else train_step_fast

            t0 = time.time()
            graphstate, opt_state, loss, norm_metrics = step_fn(
                graphstate, opt_state, microbatches,
            )

            step += 1
            total_tokens += effective_batch * max_length

            if step == 1:
                # Block so HBM peak reflects the completed fwd+bwd+update.
                jax.block_until_ready((graphstate, opt_state, loss))
                if is_main:
                    print_memory_breakdown(graphstate, opt_state, microbatches, mesh=mesh)

            # Periodic checkpoint save — also fires at step == max_steps so
            # the final state gets persisted via the same proven path as the
            # mid-training saves. No separate post-loop save flow.
            #
            # `dataset.raw_index` counts every row the iterator has stepped
            # past (including filter-dropped rows), so it's the exact offset
            # to resume at — `step * effective_batch` would under-count by
            # the ~1% dropped by `_validate_instance` and cause a rewind.
            if save_gcs_path and (step % save_interval == 0 or step == max_steps):
                _save_checkpoint_step(
                    state, graphstate, opt_state, tokenizer, mesh,
                    step, total_tokens, dataset.raw_index, save_gcs_path,
                )

            if is_main and should_log(step):
                loss_val = float(loss)
                dt = time.time() - t0
                lr_now = float(schedule(step))
                tps = effective_batch * max_length / dt
                norm_metrics_host = {k: float(v) for k, v in norm_metrics.items()}
                g_global = norm_metrics_host.get("grad_norm.global", float("nan"))
                print(
                    f"step {step}/{max_steps}  "
                    f"loss={loss_val:.4f}  "
                    f"lr={lr_now:.2e}  "
                    f"|g|={g_global:.3f}  "
                    f"dt={dt:.2f}s  "
                    f"tok/s={tps:,.0f}  "
                    f"tokens={total_tokens:,}"
                )
                if wandb.run is not None:
                    wandb.log({
                        "loss": loss_val,
                        "lr": lr_now,
                        "dt": dt,
                        "tok/s": tps,
                        "tokens": total_tokens,
                        **get_hbm_stats(),
                        **norm_metrics_host,
                    }, step=step)

    if is_main:
        print(f"Training complete: {step} steps, {total_tokens:,} tokens")


# --- Run midtraining for a checkpoint ---

def run_midtrain(
    model_name, checkpoint_name,
    run_name=None,
    total_tokens=50_000_000_000,
    total_batch_size=TOTAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    max_length=MAX_LENGTH,
):
    gcs_bucket = os.environ.get("GCS_BUCKET")
    if not gcs_bucket:
        raise RuntimeError("GCS_BUCKET environment variable is required")

    # Skip if already completed
    save_gcs_path = f"gs://{gcs_bucket}/midtrained/{model_name}/{checkpoint_name}"
    done_marker = f"{save_gcs_path}/COMPLETE"
    check = subprocess.run(
        ["gcloud", "storage", "ls", done_marker],
        capture_output=True, check=False,
    )
    if check.returncode == 0:
        print(f"Checkpoint {checkpoint_name} already COMPLETE — skipping.")
        return

    sched = OLMO2_1B_SCHEDULE
    model_spec = get_model_spec(model_name)

    step = parse_step_from_checkpoint(checkpoint_name)
    lr_start = compute_cosine_lr(
        step, sched["peak_lr"], sched["warmup_steps"],
        sched["total_steps"], sched["alpha_f"],
    )
    lr_end = 0.0  # Midtraining decays LR to zero (per OLMo2 tech report)

    effective_batch = total_batch_size * gradient_accumulation_steps
    tokens_per_step = effective_batch * max_length
    max_steps = total_tokens // tokens_per_step

    print(f"Midtraining: {checkpoint_name}")
    print(f"  Pretraining step: {step}")
    print(f"  LR: {lr_start:.6e} -> {lr_end:.6e} (linear)")
    print(f"  Tokens: {total_tokens:,} ({max_steps} steps @ {tokens_per_step:,} tok/step)")
    print(f"  Batch: {total_batch_size} x {gradient_accumulation_steps} GA = {effective_batch} effective")
    print(f"  Weight decay: {sched['weight_decay']}")
    print(f"  JAX: {jax.device_count()} devices, process {jax.process_index()}/{jax.process_count()}")

    if jax.process_index() == 0:
        wandb_project = run_name if run_name else "plasticity-midtrain"
        wandb.init(
            project=wandb_project,
            name=f"{model_name}/{checkpoint_name}",
            config={
                "run_name": run_name,
                "model": model_name,
                "checkpoint": checkpoint_name,
                "pretraining_step": step,
                "lr_start": lr_start,
                "lr_end": lr_end,
                "total_tokens": total_tokens,
                "max_steps": max_steps,
                "total_batch_size": total_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch": effective_batch,
                "max_length": max_length,
                "weight_decay": sched["weight_decay"],
                "devices": jax.device_count(),
            },
        )

    # Check for existing intermediate checkpoint to resume from
    start_step, start_tokens, start_raw_index, resume_gcs_dir = find_latest_checkpoint(save_gcs_path)

    # fp32 master weights + bf16 compute to match OLMo2's amp_bf16 setup.
    if resume_gcs_dir:
        print(f"Resuming from {resume_gcs_dir} (step {start_step}, {start_tokens:,} tokens)")
        resume_path = resume_gcs_dir.replace(f"gs://{gcs_bucket}/", "")
        model, tokenizer, mesh = load_model(
            spec=model_spec, checkpoint_path=resume_path, max_length=max_length,
            param_dtype="float32",
        )
    else:
        model, tokenizer, mesh = load_model(
            spec=model_spec, revision=checkpoint_name, max_length=max_length,
            param_dtype="float32",
        )

    # Resume offset into the data stream, in raw-sequence units (pre-filter).
    # New checkpoints write the iterator's exact raw position into meta.json;
    # for legacy checkpoints that don't, we recover it by replaying the
    # instance_filter from index 0 until we've yielded `step * effective_batch`
    # valid sequences. Slow (~one download pass over the resume prefix) but
    # bit-exact — the alternative step-based offset under-counts filter drops
    # by ~1% and rewinds into already-trained data.
    if start_raw_index is not None:
        data_start_index = start_raw_index
    elif resume_gcs_dir:
        target_valid = start_step * total_batch_size * gradient_accumulation_steps
        print(
            f"Legacy checkpoint (no raw_index in meta.json). Scanning "
            f"instance_filter from shard 0 to recover exact raw_index "
            f"(target_valid={target_valid:,})..."
        )
        data_start_index = _legacy_resume_raw_index(
            gcs_bucket, max_length, target_valid,
        )
        filter_drops = data_start_index - target_valid
        print(
            f"  → raw_index = {data_start_index:,} "
            f"(filter drops over resume prefix: {filter_drops:,}, "
            f"{100 * filter_drops / data_start_index:.2f}%)"
        )
    else:
        data_start_index = 0

    print("Loading pre-shuffled dolmino-mix-1124 from GCS (streaming)...")
    dataset = load_preshuffled_dolmino(
        gcs_bucket,
        max_length=max_length,
        start_index=data_start_index,
    )

    midtrain(
        model, tokenizer, dataset, mesh,
        lr_start=lr_start, lr_end=lr_end,
        max_steps=max_steps,
        total_batch_size=total_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        weight_decay=sched["weight_decay"],
        start_step=start_step,
        start_tokens=start_tokens,
        save_gcs_path=save_gcs_path,
        resume_gcs_dir=resume_gcs_dir,
    )

    # The final model checkpoint was written inside the training loop at
    # step == max_steps via _save_checkpoint_step (same path as mid-training
    # saves). All that's left is to stamp the run-level COMPLETE marker so
    # subsequent runs of this script skip this checkpoint. Downstream
    # consumers read the trained model from {save_gcs_path}/step_{max_steps:07d}/.
    if jax.process_index() == 0:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(f"step={max_steps}\n")
            tmp = f.name
        subprocess.run(
            ["gcloud", "storage", "cp", tmp, f"{save_gcs_path}/COMPLETE"],
            check=True,
        )
        os.unlink(tmp)

    sync_global_devices(f"finalize_{checkpoint_name}")
    gc.collect()

    if wandb.run is not None:
        wandb.finish()

    print(f"Midtraining complete for {checkpoint_name}")


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(description="Midtraining on dolmino-mix-1124")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoints", required=True, nargs="+")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--total_tokens", type=int, default=50_000_000_000)
    parser.add_argument("--total_batch_size", type=int, default=TOTAL_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    return parser.parse_args()


def main():
    if jax.process_index() != 0:
        builtins.print = lambda *a, **kw: None

    args = parse_args()
    for ckpt in args.checkpoints:
        print(f"\n{'='*60}")
        print(f"Starting checkpoint: {ckpt}")
        print(f"{'='*60}")
        try:
            run_midtrain(
                model_name=args.model,
                checkpoint_name=ckpt,
                run_name=args.run_name,
                total_tokens=args.total_tokens,
                total_batch_size=args.total_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_length=args.max_length,
            )
        except Exception as e:
            import traceback
            print(f"FAILED: {ckpt}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
