"""SFT pipeline: checkpoint-outer, datasets-inner. Saves to GCS in EasyDeL format."""

from __future__ import annotations

import argparse
import builtins
import functools
import gc
import json as _json
import math
import os
import shutil
import subprocess
import tempfile
import time

import jax

# Per-host mode: each worker runs an independent process on its 4 local chips.
# Default is multi-host. sft.model imports easydel which initialises the XLA
# backend, so distributed.initialize() must run before any other import.
_SFT_SINGLE_HOST = os.environ.get("SFT_SINGLE_HOST", "").lower() in ("1", "true", "yes")
if not _SFT_SINGLE_HOST:
    jax.distributed.initialize()

# Persistent XLA compile cache. GCS-backed when GCS_BUCKET is set so peer
# workers + future TPUs in the same region hit a shared cache (per-region,
# no cross-region transfer); falls back to local disk otherwise.
_GCS_BUCKET = os.environ.get("GCS_BUCKET")
if _GCS_BUCKET:
    _JAX_CACHE_DIR = f"gs://{_GCS_BUCKET}/jax-compile-cache"
else:
    _JAX_CACHE_DIR = os.path.expanduser("~/.cache/jax-compile-cache")
    os.makedirs(_JAX_CACHE_DIR, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _JAX_CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.multihost_utils import sync_global_devices as _sync_global_devices
from jax.sharding import NamedSharding, PartitionSpec
import wandb

from midtraining.train import (
    _FusedClippedAdamWState,
    _per_leaf_norms,
    fused_clipped_adamw,
    get_hbm_stats,
    save_to_gcs,
)

from sft.data import load_sft_dataset
from sft.model import load_model
from sft.registry import get_model_spec


def sync_global_devices(name):
    if not _SFT_SINGLE_HOST:
        _sync_global_devices(name)


_OLMO2_ASST_HEADER = "<|assistant|>\n"

# Chat template from `allenai/OLMo-2-0425-1B-SFT` tokenizer (the SFT artifact has
# it; the base ckpts don't). We attach it manually before tokenizing chat data.
# Verbatim from the SFT tokenizer's `chat_template` field; `bos_token` /
# `eos_token` are filled in by Jinja at render time.
_OLMO2_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{% if not loop.last %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
    "{% else %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
    "{% endfor %}"
)


def _ensure_chat_template(tokenizer):
    # Base/midtrained OLMo-2 tokenizers lack chat_template; the SFT artifact has one.
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = _OLMO2_CHAT_TEMPLATE
    return tokenizer


def _process_one_chat(msgs, tokenizer, eos_str, bos_id, max_length: int, add_bos: bool):
    """Tokenize a single chat → (ids, loss_mask, truncated, dropped). Returns
    (None, None, False, True) if the chat has no assistant content."""
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
    )
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    mask = [0] * len(ids)
    cursor = 0
    while True:
        h = text.find(_OLMO2_ASST_HEADER, cursor)
        if h == -1:
            break
        content_start = h + len(_OLMO2_ASST_HEADER)
        e = text.find(eos_str, content_start)
        if e == -1:
            break
        content_end = e + len(eos_str)
        for ti, (s, _) in enumerate(offsets):
            if content_start <= s < content_end:
                mask[ti] = 1
        cursor = content_end
    if add_bos and (len(ids) == 0 or ids[0] != bos_id):
        ids = [bos_id] + list(ids)
        mask = [0] + mask
    truncated = False
    if len(ids) > max_length:
        ids = ids[:max_length]
        mask = mask[:max_length]
        truncated = True
    if sum(mask) == 0:
        return None, None, False, True
    return list(ids), mask, truncated, False


def _tokenize_chats_to_rows(dataset, tokenizer, max_length: int, add_bos: bool):
    # Loss-mask covers assistant content + closing eos. Serial because Pool
    # children re-init libtpu and break the JAX-multihost cluster.
    _ensure_chat_template(tokenizer)
    eos_str = tokenizer.eos_token
    bos_id = tokenizer.bos_token_id
    rows = []
    n_truncated = 0
    n_dropped = 0
    for row in dataset:
        ids, mask, trunc, dropped = _process_one_chat(
            row["messages"], tokenizer, eos_str, bos_id, max_length, add_bos,
        )
        if dropped:
            n_dropped += 1
            continue
        if trunc:
            n_truncated += 1
        rows.append((ids, mask))
    if n_dropped:
        print(f"  dropped {n_dropped} chats with empty assistant mask")
    return rows, n_truncated


def _pad_rows(rows, max_length: int):
    """One chat per row, right-padded to max_length."""
    n = len(rows)
    input_ids = np.zeros((n, max_length), dtype=np.int32)
    attn_mask = np.zeros((n, max_length), dtype=np.int32)
    loss_mask = np.zeros((n, max_length), dtype=np.int32)
    for i, (ids, m) in enumerate(rows):
        input_ids[i, :len(ids)] = ids
        attn_mask[i, :len(ids)] = 1
        loss_mask[i, :len(m)] = m
    return input_ids, attn_mask, loss_mask


def _pack_rows(rows, max_length: int):
    """First-fit-decreasing packing of variable-length chats into ML rows.

    Returns (input_ids, segment_ids, loss_mask) shape [N_packed, max_length].
    segment_ids: -1 for padding, 0..K-1 for the K-th doc within a row.
    Packing is mathematically identical to unpacked when paired with a
    block-diagonal attention mask + per-doc reset position ids — both are
    derived downstream from segment_ids via EasyDeL MaskInfo.from_segments.
    """
    import bisect
    rows_sorted = sorted(rows, key=lambda r: -len(r[0]))
    bins_ids: list[list[int]] = []
    bins_mask: list[list[int]] = []
    bins_segs: list[list[int]] = []
    remainings: list[tuple[int, int]] = []

    for ids, m in rows_sorted:
        L = len(ids)
        idx = bisect.bisect_left(remainings, (L, -1))
        if idx < len(remainings):
            rem, bi = remainings.pop(idx)
            bins_ids[bi].extend(ids)
            bins_mask[bi].extend(m)
            bins_segs[bi].append(L)
            new_rem = rem - L
        else:
            bi = len(bins_ids)
            bins_ids.append(list(ids))
            bins_mask.append(list(m))
            bins_segs.append([L])
            new_rem = max_length - L
        if new_rem > 0:
            bisect.insort(remainings, (new_rem, bi))

    n = len(bins_ids)
    input_ids = np.zeros((n, max_length), dtype=np.int32)
    loss_mask = np.zeros((n, max_length), dtype=np.uint8)
    segment_ids = np.full((n, max_length), -1, dtype=np.int16)
    for i, (b_ids, b_m, b_segs) in enumerate(zip(bins_ids, bins_mask, bins_segs)):
        used = sum(b_segs)
        input_ids[i, :used] = b_ids
        loss_mask[i, :used] = b_m
        off = 0
        for sid, L in enumerate(b_segs):
            segment_ids[i, off:off + L] = sid
            off += L
    avg_docs = sum(len(s) for s in bins_segs) / max(n, 1)
    util = sum(sum(s) for s in bins_segs) / (n * max_length) if n else 0.0
    print(f"  packed {sum(len(s) for s in bins_segs):,} chats → {n:,} rows "
          f"(avg {avg_docs:.1f} docs/row, util {util*100:.1f}%)")
    return input_ids, segment_ids, loss_mask


def _gcs_cp(src, dst):
    subprocess.run(["gcloud", "storage", "cp", src, dst],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _gcs_exists(gcs_path):
    return subprocess.run(["gcloud", "storage", "ls", gcs_path],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _load_npz(path):
    """Returns (input_ids, mask_arr, loss_mask, n_trunc, packed) where mask_arr
    is segment_ids[N,T] int16 (-1=pad) when packed, else attn_mask[N,T] int32."""
    z = np.load(path)
    ids = z["input_ids"].astype(np.int32, copy=False)
    lm = z["loss_mask"].astype(np.int32, copy=False)
    if "segment_ids" in z.files:
        seg = z["segment_ids"].astype(np.int16, copy=False)
        return ids, seg, lm, int(z["n_trunc"]), True
    lengths = z["lengths"].astype(np.int64)
    attn = (np.arange(ids.shape[1])[None, :] < lengths[:, None]).astype(np.int32)
    return ids, attn, lm, int(z["n_trunc"]), False


def _tokenize_or_load_cache(dataset_name, dataset, tokenizer, max_length, add_bos, pack: bool, max_samples=None):
    # Two-tier cache: local disk → ${GCS_BUCKET}/sft_tokenized → tokenize fresh.
    # `lengths` is required (unpacked) because OLMo-2's tokenizer maps id=0 to
    # "!", so `ids != 0` would mis-mask "!" as padding.
    cache_dir = os.path.expanduser("~/.cache/sft_tokenized")
    os.makedirs(cache_dir, exist_ok=True)
    vocab = tokenizer.vocab_size
    suffix = "_pack" if pack else ""
    samples_suffix = f"_n{max_samples}" if max_samples is not None else ""
    key = f"{dataset_name}_ml{max_length}_bos{int(add_bos)}_v{vocab}{suffix}{samples_suffix}.npz"
    path = os.path.join(cache_dir, key)
    bucket = os.environ.get("GCS_BUCKET")
    gcs_path = f"gs://{bucket}/sft_tokenized/{key}" if bucket else None

    if os.path.exists(path):
        print(f"  [cache] local hit: {path}")
        return _load_npz(path)

    if gcs_path and _gcs_exists(gcs_path):
        print(f"  [cache] GCS hit: {gcs_path} → downloading")
        t0 = time.time()
        _gcs_cp(gcs_path, path)
        print(f"  [cache] downloaded {os.path.getsize(path)/1e9:.1f} GB in {time.time()-t0:.0f}s")
        return _load_npz(path)

    # Cold cache. In multihost mode only rank 0 tokenizes + uploads; the
    # others wait at the barrier and download from GCS afterwards. In
    # per-host mode each worker is its own JAX cluster (process_index==0),
    # so all 4 tokenize independently — same as before.
    is_main = jax.process_index() == 0
    if is_main:
        print(f"  [cache] miss; tokenizing {dataset_name} (pack={pack})...")
        t0 = time.time()
        rows, n_trunc = _tokenize_chats_to_rows(dataset, tokenizer, max_length, add_bos)
        print(f"  [cache] tokenized {len(rows):,} chats in {time.time()-t0:.0f}s")
    else:
        print(f"  [cache] miss; waiting for rank 0 to tokenize {dataset_name}")

    hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
    if is_main and os.path.exists(hf_cache):
        shutil.rmtree(hf_cache, ignore_errors=True)
        print(f"  [cache] freed {hf_cache}")

    save_stem = path[:-4] if path.endswith(".npz") else path
    if is_main:
        if pack:
            input_ids, segment_ids, loss_mask = _pack_rows(rows, max_length)
            np.savez(save_stem,
                     input_ids=input_ids,
                     loss_mask=loss_mask,
                     segment_ids=segment_ids,
                     n_trunc=np.int32(n_trunc))
        else:
            input_ids, attn_mask, loss_mask = _pad_rows(rows, max_length)
            lengths = attn_mask.sum(axis=1).astype(np.uint16)
            np.savez(save_stem,
                     input_ids=input_ids,
                     loss_mask=loss_mask.astype(np.uint8, copy=False),
                     lengths=lengths,
                     n_trunc=np.int32(n_trunc))
        print(f"  [cache] wrote {path} ({os.path.getsize(path)/1e9:.1f} GB)")
        if gcs_path:
            try:
                t0 = time.time()
                _gcs_cp(path, gcs_path)
                print(f"  [cache] uploaded to {gcs_path} in {time.time()-t0:.0f}s")
            except Exception as e:
                print(f"  [cache] GCS upload failed (non-fatal): {e}")

    # Multihost barrier: non-main workers waited above, now sync + download.
    if not _SFT_SINGLE_HOST and jax.process_count() > 1:
        _sync_global_devices(f"sft_tokenize_done_{key}")
    if not is_main:
        if not (gcs_path and _gcs_exists(gcs_path)):
            raise RuntimeError(
                f"rank 0 finished tokenizing but cache not found at {gcs_path}"
            )
        t0 = time.time()
        _gcs_cp(gcs_path, path)
        print(f"  [cache] downloaded {os.path.getsize(path)/1e9:.1f} GB in {time.time()-t0:.0f}s")

    return _load_npz(path)


def make_sft_train_step(state, optimizer, gradient_accumulation_steps=1,
                        remat=False, loss_reduction="mean", packed=False,
                        loss_chunk_size=0):
    """Build a jit'd train step.

    remat=True: forward recompute, trades +33% compute for activation memory.
    packed=True: batches use segment_ids → block-diagonal attention + per-doc
        position resets; mathematically identical to one doc per row.
    loss_chunk_size>0: scan the lm_head over T-chunks (peak logits B*chunk*V
        instead of B*T*V). T must divide cleanly.
    """
    if remat:
        _remat_policy = jax.checkpoint_policies.dots_with_no_batch_dims_saveable
        _maybe_checkpoint = functools.partial(jax.checkpoint, policy=_remat_policy)
    else:
        _maybe_checkpoint = lambda f: f  # no-op decorator

    if packed:
        from ejkernel.types import MaskInfo

    @_maybe_checkpoint
    def compute_loss_sum(params, batch):
        """Returns (sum_of_token_losses, num_completion_tokens). Caller decides
        whether to divide by n_toks (mean reduction) or not (sum reduction)."""
        module = state.merge(params)
        fwd_kwargs = {"input_ids": batch["input_ids"]}
        if packed:
            fwd_kwargs["mask_info"] = MaskInfo.from_segments(batch["segment_ids"])
        else:
            fwd_kwargs["attention_mask"] = batch["attention_mask"]

        shift_labels = jnp.pad(batch["input_ids"][:, 1:], ((0, 0), (0, 1)))
        shift_loss = jnp.pad(batch["loss_mask"][:, 1:], ((0, 0), (0, 1)))
        if packed:
            seg = batch["segment_ids"]
            shift_seg = jnp.pad(seg[:, 1:], ((0, 0), (0, 1)), constant_values=-1)
            shift_loss = shift_loss * (shift_seg == seg).astype(shift_loss.dtype)

        if loss_chunk_size > 0:
            # Chunked path: never materialize [B, T, V] logits. Forward stops
            # at last_hidden_state, then scan over T-chunks of lm_head + CE.
            out = module(apply_lm_head=False, **fwd_kwargs)
            hidden = out.last_hidden_state  # [B, T, H]
            B, T, H = hidden.shape
            TC = loss_chunk_size
            assert T % TC == 0, f"max_length {T} must be divisible by loss_chunk_size {TC}"
            n_chunks = T // TC
            # Reshape so scan's leading axis is the chunk axis.
            h_xs = hidden.reshape(B, n_chunks, TC, H).transpose(1, 0, 2, 3)
            lbl_xs = shift_labels.reshape(B, n_chunks, TC).transpose(1, 0, 2)
            m_xs = shift_loss.reshape(B, n_chunks, TC).transpose(1, 0, 2)

            # CRITICAL: wrap body in jax.checkpoint so scan does NOT save the
            # per-iter residuals (logits_c et al). Without this, scan's default
            # behavior is to save residuals from every iteration, which means
            # 8 chunks × [B, TC, V] = [B, T, V] live simultaneously during
            # backward — defeating the purpose of chunking. With the
            # checkpoint, each step's intermediates are recomputed during the
            # backward of that step, peak = 1 chunk's logits.
            def step_inner(carry, h_c, lbl_c, m_c):
                logits_c = module.apply_lm_head(h_c)  # [B, TC, V]
                log_z = jax.scipy.special.logsumexp(logits_c, axis=-1)
                logit_at = jnp.take_along_axis(
                    logits_c, lbl_c[..., None], axis=-1,
                ).squeeze(-1)
                token_losses = log_z - logit_at
                sum_loss, n_toks = carry
                m_f = m_c.astype(jnp.float32)
                return (sum_loss + (token_losses * m_f).sum().astype(jnp.float32),
                        n_toks + m_f.sum())

            step_inner_ckpt = jax.checkpoint(
                step_inner, prevent_cse=False,
                policy=jax.checkpoint_policies.nothing_saveable,
            )

            def step(carry, x):
                h_c, lbl_c, m_c = x
                return step_inner_ckpt(carry, h_c, lbl_c, m_c), None

            (sum_loss, n_toks), _ = jax.lax.scan(
                step, (jnp.float32(0.0), jnp.float32(0.0)),
                (h_xs, lbl_xs, m_xs),
            )
        else:
            logits = module(**fwd_kwargs).logits
            log_z = jax.scipy.special.logsumexp(logits, axis=-1)
            logit_at_label = jnp.take_along_axis(
                logits, shift_labels[..., None], axis=-1,
            ).squeeze(-1)
            token_losses = log_z - logit_at_label
            sum_loss = (token_losses * shift_loss).sum().astype(jnp.float32)
            n_toks = shift_loss.sum().astype(jnp.float32)
        return sum_loss, n_toks

    def _to_bf16(x):
        if jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.bfloat16)
        return x

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def train_step(graphstate, opt_state, microbatches):
        bf16_graphstate = jax.tree.map(_to_bf16, graphstate)
        if gradient_accumulation_steps == 1:
            batch = jax.tree.map(lambda x: x[0], microbatches)
            (sum_loss, n_toks), bf16_grads = jax.value_and_grad(
                compute_loss_sum, has_aux=True,
            )(bf16_graphstate, batch)
            if loss_reduction == "sum":
                grads = jax.tree.map(lambda g: g.astype(jnp.float32), bf16_grads)
                loss = sum_loss
            else:
                scale = 1.0 / jnp.maximum(n_toks, 1.0)
                grads = jax.tree.map(
                    lambda g: g.astype(jnp.float32) * scale, bf16_grads,
                )
                loss = sum_loss / jnp.maximum(n_toks, 1.0)
            loss_per_tok = sum_loss / jnp.maximum(n_toks, 1.0)
        else:
            def accum_step(carry, microbatch):
                (sum_loss, n_toks), bf16_grads = jax.value_and_grad(
                    compute_loss_sum, has_aux=True,
                )(bf16_graphstate, microbatch)
                fp32_grads = jax.tree.map(
                    lambda g: g.astype(jnp.float32), bf16_grads,
                )
                acc_grads, acc_sum_loss, acc_n_toks = carry
                new_grads = jax.tree.map(jnp.add, acc_grads, fp32_grads)
                return (new_grads, acc_sum_loss + sum_loss, acc_n_toks + n_toks), None

            zero_grads = jax.tree.map(
                lambda p: jnp.zeros(p.shape, dtype=jnp.float32), graphstate,
            )
            (total_grads, total_sum_loss, total_n_toks), _ = jax.lax.scan(
                accum_step,
                (zero_grads, jnp.float32(0.0), jnp.float32(0.0)),
                microbatches,
            )
            if loss_reduction == "sum":
                grads = total_grads
                loss = total_sum_loss
            else:
                scale = 1.0 / jnp.maximum(total_n_toks, 1.0)
                grads = jax.tree.map(lambda g: g * scale, total_grads)
                loss = total_sum_loss * scale
            loss_per_tok = total_sum_loss / jnp.maximum(total_n_toks, 1.0)

        norm_metrics = _per_leaf_norms(grads, "grad_norm")
        updates, new_opt_state = optimizer.update(grads, opt_state, graphstate)
        new_graphstate = optax.apply_updates(graphstate, updates)
        norm_metrics.update(_per_leaf_norms(new_graphstate, "param_norm"))
        return new_graphstate, new_opt_state, loss, loss_per_tok, norm_metrics

    return train_step


def _build_schedule(peak_lr, total_steps, schedule_type, warmup_frac):
    """Optax schedule with optional linear warmup. schedule_type: linear | cosine."""
    warmup_steps = max(int(warmup_frac * total_steps), 0)
    decay_steps = max(total_steps - warmup_steps, 1)
    if schedule_type == "linear":
        decay = optax.linear_schedule(peak_lr, 0.0, decay_steps)
    elif schedule_type == "cosine":
        decay = optax.cosine_decay_schedule(peak_lr, decay_steps, alpha=0.1)
    else:
        raise ValueError(f"Unknown schedule: {schedule_type!r} (linear|cosine)")
    if warmup_steps > 0:
        warmup = optax.linear_schedule(0.0, peak_lr, warmup_steps)
        return optax.join_schedules([warmup, decay], [warmup_steps])
    return decay


def save_sft_model(state, graphstate, tokenizer, mesh, gcs_path, meta):
    """Save fine-tuned model to GCS in EasyDeL format. No optimizer state.

    Layout: {gcs_path}/model/ + tokenizer/config json + meta.json + sibling
    {gcs_path}_COMPLETE marker (host 0, written last after sync barrier).
    """
    is_main = jax.process_index() == 0
    local_dir = f"/tmp/sft_save/{int(time.time())}_{os.getpid()}"
    shutil.rmtree(local_dir, ignore_errors=True)
    os.makedirs(local_dir, exist_ok=True)

    with mesh:
        state.merge(graphstate).save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    if is_main:
        with open(os.path.join(local_dir, "meta.json"), "w") as f:
            _json.dump(meta, f, indent=2)

    sync_global_devices(f"sft_save_pre_upload_{gcs_path}")
    save_to_gcs(local_dir, gcs_path)

    sync_global_devices(f"sft_save_post_upload_{gcs_path}")
    if is_main:
        marker_path = f"{gcs_path.rstrip('/')}_COMPLETE"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(_json.dumps(meta) + "\n")
            tmp = f.name
        subprocess.run(
            ["gcloud", "storage", "cp", tmp, marker_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.unlink(tmp)

    sync_global_devices(f"sft_save_done_{gcs_path}")
    shutil.rmtree(local_dir, ignore_errors=True)


def sft_one(
    model, tokenizer, mesh,
    *,
    dataset_name, run_name, base_ckpt, model_name,
    learning_rate, total_batch_size, gradient_accumulation_steps,
    max_steps, max_length, weight_decay, log_every, save_gcs_path,
    save_model=False, epochs=None,
    schedule_type="linear", warmup_frac=0.03,
    add_bos=True, seed=1, loss_reduction="sum",
    adam_b2=0.999, clip_norm=None, pack=False, loss_chunk_size=0,
    max_samples=None,
):
    is_main = jax.process_index() == 0
    effective_batch = total_batch_size * gradient_accumulation_steps

    wandb_project = os.environ.get("WANDB_PROJECT", "sft-plasticity")
    if is_main:
        wandb.init(
            project=wandb_project,
            name=f"{run_name}_{base_ckpt}_{dataset_name}",
            reinit=True,
            config={
                "model": model_name, "base_ckpt": base_ckpt,
                "dataset": dataset_name, "learning_rate": learning_rate,
                "total_batch_size": total_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch": effective_batch,
                "max_length": max_length, "weight_decay": weight_decay,
                "epochs": epochs, "schedule": schedule_type,
                "warmup_frac": warmup_frac, "pack": pack,
                "loss_chunk_size": loss_chunk_size,
            },
        )

    print(f"Loading SFT dataset {dataset_name}... (max_samples={max_samples})")
    raw = load_sft_dataset(dataset_name, max_samples=max_samples)
    input_ids, mask_arr, loss_mask, n_trunc, packed = _tokenize_or_load_cache(
        dataset_name, raw, tokenizer, max_length, add_bos, pack,
        max_samples=max_samples,
    )
    n = len(input_ids)
    print(f"  {dataset_name}: {n} rows{' (packed)' if packed else ''}, "
          f"{n_trunc} truncated at max_length={max_length}, "
          f"loss tokens: {int(loss_mask.sum()):,}")

    if epochs is not None:
        max_steps = max(int(math.ceil(epochs * n / effective_batch)), 1)
        print(f"  --epochs={epochs} → max_steps={max_steps}")
    if is_main and wandb.run is not None:
        wandb.run.config.update({"max_steps": max_steps}, allow_val_change=True)

    state = model.to_state()
    graphstate = state.graphstate

    schedule = _build_schedule(learning_rate, max_steps, schedule_type, warmup_frac)
    print(f"  schedule: {schedule_type}, warmup_frac={warmup_frac}, "
          f"warmup_steps={int(warmup_frac * max_steps)}, total_steps={max_steps}")

    vocab_size = model.config.vocab_size
    def _decay_mask(params):
        return jax.tree.map(lambda p: p.shape[0] != vocab_size, params)

    optimizer = fused_clipped_adamw(
        learning_rate=schedule, b1=0.9, b2=adam_b2,
        weight_decay=weight_decay, mask=_decay_mask, clip_norm=clip_norm,
    )

    replicated = NamedSharding(mesh, PartitionSpec())
    param_shardings = jax.tree.map(lambda p: p.sharding, graphstate)
    opt_out_shardings = _FusedClippedAdamWState(
        m=param_shardings, v=param_shardings, count=replicated,
    )
    with mesh:
        opt_state = jax.jit(
            optimizer.init, out_shardings=opt_out_shardings,
        )(graphstate)

    use_remat = os.environ.get("SFT_REMAT", "").lower() in ("1", "true", "yes")
    train_step = make_sft_train_step(
        state, optimizer, gradient_accumulation_steps,
        remat=use_remat, loss_reduction=loss_reduction, packed=packed,
        loss_chunk_size=loss_chunk_size,
    )

    batch_sharding = NamedSharding(mesh, PartitionSpec(None, ("dp", "fsdp")))
    rng = np.random.default_rng(seed=seed)
    epoch_idx = rng.permutation(n)
    epoch_pos = 0

    if n < effective_batch:
        raise ValueError(
            f"dataset has {n} examples but effective_batch={effective_batch}. "
            f"Reduce --total_batch_size or --gradient_accumulation_steps, or "
            f"use a larger dataset."
        )

    def _next_batch():
        nonlocal epoch_idx, epoch_pos
        if epoch_pos + effective_batch > n:
            epoch_idx = rng.permutation(n)
            epoch_pos = 0
        idx = epoch_idx[epoch_pos:epoch_pos + effective_batch]
        epoch_pos += effective_batch
        shape = (gradient_accumulation_steps, total_batch_size, max_length)
        batch = {
            "input_ids": input_ids[idx].reshape(shape),
            "loss_mask": loss_mask[idx].reshape(shape),
        }
        if packed:
            batch["segment_ids"] = mask_arr[idx].reshape(shape).astype(np.int32)
        else:
            batch["attention_mask"] = mask_arr[idx].reshape(shape)
        return batch

    def should_log(s):
        if s == 1:
            return True
        if s <= 20:
            return s % 5 == 0
        return s % log_every == 0

    print(f"Training {dataset_name}: {max_steps} steps, "
          f"effective_batch={effective_batch} ({total_batch_size}x{gradient_accumulation_steps} GA), "
          f"max_length={max_length}, lr={learning_rate}, remat={use_remat}")

    def _device_put_batch(batch_np):
        return {k: jax.device_put(v, batch_sharding) for k, v in batch_np.items()}

    # Double-buffered training loop: prefetch step N+1's batch onto device
    # while step N's compute runs on TPU. device_put returns immediately
    # (async to TPU), so the H2D transfer overlaps with TPU compute.
    next_microbatches = _device_put_batch(_next_batch())

    # Per-step throughput is constant from shape (BS×ML×GA tokens/step). Wall
    # time is the only variable, so tok/s = padded_tokens / dt. We log:
    #   train/tokens_per_sec       — padded throughput (model FLOPs/sec scale)
    #   train/cum_tokens           — running total of padded tokens consumed
    #   train/cum_loss_tokens_M    — running total of loss-mask tokens (millions)
    # Loss-mask tokens are the "useful" tokens (assistant content); padded
    # tokens are what HFU benchmarks count.
    padded_tok_per_step = effective_batch * max_length
    cum_tokens = 0
    cum_loss_tokens = 0
    last_log_step = 0
    last_log_time = time.time()

    with mesh:
        for step in range(1, max_steps + 1):
            current = next_microbatches
            # Prefetch next step's batch while current step runs.
            if step < max_steps:
                next_microbatches = _device_put_batch(_next_batch())
            t0 = time.time()
            graphstate, opt_state, loss, loss_per_tok, norm_metrics = train_step(
                graphstate, opt_state, current,
            )
            if step == 1:
                jax.block_until_ready((graphstate, opt_state, loss))

            if is_main and should_log(step):
                loss_val = float(loss)
                lpt = float(loss_per_tok)
                ppl = float(math.exp(min(lpt, 20.0)))
                dt = time.time() - t0  # this step's wall time
                # Window-averaged throughput across steps since last log: more
                # stable than per-step dt (which catches one async drain).
                steps_since_log = step - last_log_step
                window_secs = time.time() - last_log_time
                window_tok = steps_since_log * padded_tok_per_step
                tok_per_sec = window_tok / max(window_secs, 1e-6)
                last_log_step = step
                last_log_time = time.time()

                # Update cumulative counters: estimate loss tokens from
                # microbatch's loss_mask sum (cheap host computation).
                cum_tokens = step * padded_tok_per_step
                # n_toks (loss-mask sum across all microbatches in this step)
                # is computed inside compute_loss_sum; loss_per_tok = sum/n
                # so n_toks = sum_loss / loss_per_tok. With "sum" reduction
                # `loss_val` is sum_loss; with "mean" `loss_val == loss_per_tok`
                # — derive n_toks from the global loss-mask total instead.
                if loss_reduction == "sum" and lpt > 0:
                    n_toks_step = int(loss_val / max(lpt, 1e-9))
                else:
                    n_toks_step = padded_tok_per_step  # upper bound
                cum_loss_tokens += n_toks_step

                lr_now = float(schedule(step))
                norm_host = {k: float(v) for k, v in norm_metrics.items()}
                g_global = norm_host.get("grad_norm.global", float("nan"))
                p_global = norm_host.get("param_norm.global", float("nan"))
                print(f"  step {step}/{max_steps}  loss={loss_val:.4f}  "
                      f"loss/tok={lpt:.4f}  ppl={ppl:.3f}  lr={lr_now:.2e}  "
                      f"|g|={g_global:.3f}  |p|={p_global:.3f}  "
                      f"tok/s={tok_per_sec/1e3:.0f}k  dt={dt:.2f}s")
                if wandb.run is not None:
                    wandb.log({
                        "train/loss": loss_val,
                        "train/loss_per_tok": lpt,
                        "train/perplexity": ppl,
                        "train/lr": lr_now, "train/dt": dt,
                        "train/tokens_per_sec": tok_per_sec,
                        "train/cum_tokens": cum_tokens,
                        "train/cum_loss_tokens_M": cum_loss_tokens / 1e6,
                        **{f"train/{k}": v for k, v in norm_host.items()},
                        **get_hbm_stats(),
                    }, step=step)

    if save_model:
        print(f"Saving SFT model to {save_gcs_path}")
        save_sft_model(
            state, graphstate, tokenizer, mesh, save_gcs_path,
            meta={
                "model": model_name, "base_ckpt": base_ckpt,
                "dataset": dataset_name, "run_name": run_name,
                "max_steps": max_steps, "learning_rate": learning_rate,
                "total_batch_size": total_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_length": max_length, "weight_decay": weight_decay,
            },
        )

    del opt_state, graphstate
    gc.collect()
    if is_main and wandb.run is not None:
        wandb.finish()


def run_sft(args):
    is_main = jax.process_index() == 0
    if not is_main:
        builtins.print = lambda *a, **kw: None

    gcs_bucket = os.environ.get("GCS_BUCKET")
    if not gcs_bucket:
        raise RuntimeError("GCS_BUCKET env var is required")

    model_spec = get_model_spec(args.model)
    # SFT_SHARDING="-1,1,1,1,1" gives full DP — used by per-host launcher
    # where FSDP's per-layer all-gather kills perf on a 4-chip mesh.
    _sharding_env = os.environ.get("SFT_SHARDING")
    if _sharding_env:
        new_sharding = tuple(int(x) for x in _sharding_env.split(","))
        print(f"  sharding: {model_spec.sharding} → {new_sharding} (SFT_SHARDING)")
        model_spec.sharding = new_sharding

    checkpoints = args.checkpoints or model_spec.checkpoints
    if not checkpoints:
        raise RuntimeError(f"No checkpoints for {args.model}; pass --checkpoints")

    print(f"=== SFT: {args.model} on {args.datasets} ===")
    print(f"  checkpoints ({len(checkpoints)}): {checkpoints}")
    print(f"  lr={args.learning_rate}  bs={args.total_batch_size}x{args.gradient_accumulation_steps} GA  "
          f"max_length={args.max_length}  wd={args.weight_decay}")

    for ckpt_idx, ckpt in enumerate(checkpoints):
        print(f"\n{'=' * 60}\nCheckpoint {ckpt_idx + 1}/{len(checkpoints)}: {ckpt}\n{'=' * 60}")

        # Three ckpt formats:
        #   1) gs://...                        — GCS path (EasyDeL midtrain
        #      tensorstore w/ model/ subdir, OR HF flat layout).
        #   2) <hf_id>@<revision>              — HF Hub revision (overrides spec.hf_id).
        #   3) <revision>                      — bare revision against spec.hf_id.
        # Label-derivation rules (used in save path + wandb run name):
        #   - gs://.../{rev}/step_NNN/         → "{rev}_step_NNN" (midtrain step saves)
        #   - gs://.../{org}--{model}--{rev}   → "{rev}" (HF flat in our GCS bucket)
        #   - gs://.../{name}/                 → "{name}"
        if ckpt.startswith("gs://"):
            parts = ckpt.rstrip("/").rsplit("/", 2)
            last, second = parts[-1], parts[-2] if len(parts) >= 2 else ""
            if last.startswith("step_") and second:
                ckpt_label = f"{second}_{last}"
            elif last.count("--") >= 2:
                ckpt_label = last.split("--", 1)[1]
            else:
                ckpt_label = last
            # Disambiguate midtrained ckpts from same-named HF revisions in
            # save paths and wandb run names — same revision name can be both
            # an HF stage1 ckpt AND our midtrained version of that same step.
            if "/midtrained/" in ckpt and not ckpt_label.startswith("midtrain_"):
                ckpt_label = f"midtrain_{ckpt_label}"
            load_kwargs = {"checkpoint_path": ckpt}
        elif "@" in ckpt:
            hf_id, revision = ckpt.rsplit("@", 1)
            ckpt_label = revision
            if hf_id != model_spec.hf_id:
                model_spec.hf_id = hf_id
            load_kwargs = {"revision": revision}
        else:
            ckpt_label = ckpt
            load_kwargs = {"revision": ckpt}

        for ds_idx, dataset_name in enumerate(args.datasets):
            print(f"\n--- {dataset_name} on {ckpt_label} ({ds_idx + 1}/{len(args.datasets)}) ---")
            model, tokenizer, mesh = load_model(
                spec=model_spec, max_length=args.max_length, **load_kwargs,
            )
            save_path = (
                f"gs://{gcs_bucket}/sft/{args.model}/{ckpt_label}/"
                f"{dataset_name}/{args.run_name}/"
            )
            sft_one(
                model=model, tokenizer=tokenizer, mesh=mesh,
                dataset_name=dataset_name, run_name=args.run_name,
                base_ckpt=ckpt_label, model_name=args.model,
                learning_rate=args.learning_rate,
                total_batch_size=args.total_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_steps=args.max_steps, max_length=args.max_length,
                weight_decay=args.weight_decay, log_every=args.log_every,
                save_gcs_path=save_path, save_model=args.save_model,
                epochs=args.epochs, schedule_type=args.schedule,
                warmup_frac=args.warmup_frac,
                add_bos=args.add_bos, seed=args.seed,
                loss_reduction=args.loss_reduction,
                adam_b2=args.adam_b2,
                clip_norm=(None if args.clip_norm.lower() in ("none", "")
                           else float(args.clip_norm)),
                pack=args.pack,
                loss_chunk_size=args.loss_chunk_size,
                max_samples=args.max_samples,
            )
            del model
            gc.collect()
            sync_global_devices(f"end_dataset_{ckpt_label}_{dataset_name}")

    print("\n=== SFT pipeline complete ===")


def parse_args():
    p = argparse.ArgumentParser(description="SFT pipeline (checkpoint-outer, datasets-inner)")
    p.add_argument("--model", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--checkpoints", nargs="+", default=None)
    p.add_argument("--run_name", required=True)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--epochs", type=float, default=2.0,
                   help="Overrides --max_steps with ceil(epochs * n / effective_batch).")
    p.add_argument("--total_batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=False,
                   help="Save fine-tuned model to GCS (~3GB). Default off; pass when needed for eval.")
    p.add_argument("--add_bos", action=argparse.BooleanOptionalAction, default=True,
                   help="Prepend BOS to each sequence if not already present.")
    p.add_argument("--seed", type=int, default=1,
                   help="Seed for batch shuffling. Allen AI uses 1 for OLMo-2 SFT.")
    p.add_argument("--loss_reduction", choices=["sum", "mean"], default="mean",
                   help="mean = literal OLMo-2-0425-1B launch (open-instruct default); "
                        "sum = OLMo-2-7B/13B/32B + Tulu-3 launch scripts (Allen AI's "
                        "stated 'best practice'). 1B published checkpoint used mean.")
    p.add_argument("--adam_b2", type=float, default=0.999,
                   help="Adam β2. 0.999 matches OLMo-2 SFT; 0.95 matches our midtrain.")
    p.add_argument("--clip_norm", type=str, default="none",
                   help="Global grad clip norm. 'none' disables clipping (OLMo-2 SFT).")
    p.add_argument("--pack", action=argparse.BooleanOptionalAction, default=False,
                   help="Pack chats into ML-length rows with block-diagonal attention "
                        "(EasyDeL MaskInfo segment_ids). Mathematically identical to "
                        "unpacked training; ~5-7x throughput on Tulu-3.")
    p.add_argument("--loss_chunk_size", type=int, default=0,
                   help="Chunk size along T for cross-entropy. >0 enables chunked CE: "
                        "forward emits last_hidden_state, lm_head + CE run in a "
                        "jax.lax.scan over T-chunks, dropping peak logits memory "
                        "B*T*V → B*chunk*V. Must divide max_length. Try 512 first.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap dataset size (debug). None = full dataset.")
    return p.parse_args()


def main():
    run_sft(parse_args())


if __name__ == "__main__":
    main()
