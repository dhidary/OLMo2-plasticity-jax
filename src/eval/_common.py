"""Shared helpers for ARC / HellaSwag / GSM8K benches.

Importing this module sets up the JAX persistent compile cache. Import it
before `jax.numpy` etc. so the config takes effect on first JIT.
"""

from __future__ import annotations

import datetime
import json
import os
import sys

# ejkernel routes JAX cache writes to ~/.cache/ejkernel-cache regardless of dir we set.
_GCS_BUCKET = os.environ.get("GCS_BUCKET")
if _GCS_BUCKET:
    _JAX_CACHE_DIR = f"gs://{_GCS_BUCKET}/jax-compile-cache"
else:
    _JAX_CACHE_DIR = os.path.expanduser("~/.cache/jax-compile-cache")
    os.makedirs(_JAX_CACHE_DIR, exist_ok=True)
_EJKERNEL_CACHE_DIR = os.path.expanduser("~/.cache/ejkernel-cache/ejit_compiled_functions")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_compilation_cache_dir", _JAX_CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

_n = len(os.listdir(_EJKERNEL_CACHE_DIR)) if os.path.isdir(_EJKERNEL_CACHE_DIR) else 0
print(f"[compile-cache] {'WARM' if _n else 'COLD'} - {_n} entries in {_EJKERNEL_CACHE_DIR}")


def split_revision(model_path: str):
    if model_path.startswith("gs://") or "@" not in model_path:
        return model_path, None
    head, rev = model_path.rsplit("@", 1)
    return head, rev


def resolve_model_path(model_path: str) -> str:
    if not model_path.startswith("gs://"):
        return model_path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from sft.model import _download_from_gcs
    return _download_from_gcs(model_path)


def load_easydel(model_path: str, max_pos: int = 2048, kvdtype: bool = False):
    """Load model + tokenizer + (1,16,1,1,1) mesh on v5e-16 FSDP-only."""
    import easydel as ed
    from transformers import AutoTokenizer
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from sft.model import create_mesh

    hf_id, revision = split_revision(model_path)
    local_path = resolve_model_path(hf_id)
    hf_kwargs = {"revision": revision} if revision else {}
    if revision:
        print(f"  loading from HF revision: {revision}")

    # Some of our SFT'd ckpts have malformed tokenizer files (vocab_file=None) that
    # crash transformers' convert_slow_tokenizer. Fall back to released SFT's
    # tokenizer from HF — same OLMo-2 1B vocab, just properly serialized.
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_path, fix_mistral_regex=True, **hf_kwargs)
    except (AttributeError, TypeError):
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_path, **hf_kwargs)
        except (AttributeError, TypeError):
            print(f"  local tokenizer broken at {local_path}; loading allenai/OLMo-2-0425-1B-SFT tokenizer instead")
            tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-SFT", **hf_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    axis_names = ("dp", "fsdp", "ep", "tp", "sp")
    n_devices = jax.device_count()
    mesh = create_mesh((1, n_devices, 1, 1, 1))

    cfg = dict(
        attn_mechanism=ed.AttentionMechanisms.SDPA,
        attn_dtype=jnp.bfloat16,
        freq_max_position_embeddings=max_pos,
        mask_max_position_embeddings=max_pos,
    )
    if kvdtype:
        cfg["kvdtype"] = jnp.bfloat16

    with mesh:
        model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
            local_path,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            sharding_axis_dims=(1, n_devices, 1, 1, 1),
            sharding_axis_names=axis_names,
            config_kwargs=ed.EasyDeLBaseConfigDict(**cfg),
            **hf_kwargs,
        )

    # HF Olmo2Config doesn't expose head_dim but EasyDeL's rotary code reads it.
    if not hasattr(model.config, "head_dim"):
        model.config.head_dim = model.config.hidden_size // model.config.num_attention_heads

    return model, tokenizer, mesh


def load_esurge(model_path: str, max_model_len: int,
                page_size: int = 32, max_num_seqs: int = 8,
                hbm_utilization: float = 0.7):
    """Load model + tokenizer + eSurge continuous-batching engine for generative benches.

    Wraps `load_easydel` and constructs `ed.eSurge`. Returns (engine, tokenizer).
    """
    import easydel as ed

    model, tokenizer, _mesh = load_easydel(model_path, max_pos=max_model_len, kvdtype=True)

    # Auto-size FSDP axis to actual device count (4 on single-host v5e/v6e, 16 on multi-host)
    n_devices = jax.device_count()
    engine = ed.eSurge(
        model=model,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        sharding_axis_dims=(1, n_devices, 1, 1, 1),
        dtype=jnp.bfloat16,
        hbm_utilization=hbm_utilization,
        page_size=page_size,
        max_num_seqs=max_num_seqs,
        auto_shard_model=False,
        compile_runner=True,
        runner_verbose=False,
        silent_mode=False,
    )
    return engine, tokenizer


def make_letter_score_fn(model, mesh, letter_token_ids):
    """JIT'd forward returning logits[batch, last_idx, letter_ids] — shape [B, K].

    For single-token MC scoring (ARC, MMLU): one forward through the prompt
    ending in 'Answer:', read the logit at the last content position (which
    predicts the next token), index into the K letter token IDs, argmax.

    Saves ~4× compute vs running 4 separate forwards with appended ' A'/' B'/...
    """
    from flax import nnx
    from jax.sharding import NamedSharding, PartitionSpec

    graphdef, params = nnx.split(model)
    param_shardings = jax.tree.map(lambda p: p.sharding, params)
    batch_sharding = NamedSharding(mesh, PartitionSpec(("dp", "fsdp")))
    replicated = NamedSharding(mesh, PartitionSpec())

    letters_arr = jnp.array(letter_token_ids, dtype=jnp.int32)

    def _score(params, input_ids, attention_mask, last_idx):
        m = nnx.merge(graphdef, params)
        logits = m(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, L, V]
        idx = last_idx[:, None, None].astype(jnp.int32)
        last_logits = jnp.take_along_axis(logits, idx, axis=1).squeeze(1)  # [B, V]
        return last_logits[:, letters_arr]  # [B, K]

    score = jax.jit(
        _score,
        in_shardings=(param_shardings, batch_sharding, batch_sharding, batch_sharding),
        out_shardings=replicated,
    )
    return score, params


def make_score_fn(model, mesh):
    """JIT-compiled forward returning per-sequence sum log-prob over target_mask."""
    from flax import nnx
    from jax.sharding import NamedSharding, PartitionSpec

    graphdef, params = nnx.split(model)
    param_shardings = jax.tree.map(lambda p: p.sharding, params)
    batch_sharding = NamedSharding(mesh, PartitionSpec(("dp", "fsdp")))
    replicated = NamedSharding(mesh, PartitionSpec())

    def _score(params, input_ids, attention_mask, target_mask):
        m = nnx.merge(graphdef, params)
        logits = m(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = target_mask[:, 1:].astype(jnp.float32)
        log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
        tok_lp = jnp.take_along_axis(log_probs, shift_labels[:, :, None], axis=-1).squeeze(-1)
        return jnp.sum(tok_lp * shift_mask, axis=-1)

    score = jax.jit(
        _score,
        in_shardings=(param_shardings, batch_sharding, batch_sharding, batch_sharding),
        out_shardings=replicated,
    )
    return score, params


def write_summary(task: str, label: str, model_path: str, n: int, num_shots: int, metrics: dict):
    payload = {
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": task,
        "label": label,
        "model_path": model_path,
        "n": n,
        "num_shots": num_shots,
        "metrics": metrics,
    }
    with open("/tmp/bench_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
