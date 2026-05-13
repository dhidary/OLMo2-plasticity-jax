"""Model loading utilities."""

import os
import shutil
import subprocess

import easydel as ed
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from transformers import AutoTokenizer

from sft.registry import ModelSpec


GCS_BUCKET = os.environ.get("GCS_BUCKET")

DTYPE_MAP = {
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
}

ATTN_MAP = {
    "vanilla": ed.AttentionMechanisms.VANILLA,
    "flash": ed.AttentionMechanisms.FLASH_ATTN2,
    "paged": ed.AttentionMechanisms.PAGED_ATTENTION,
    "sdpa": ed.AttentionMechanisms.SDPA,
    "ring": ed.AttentionMechanisms.RING,
}
# Optional, TPU-optimized mechanisms — added if the enum exists in this EasyDeL version.
for _opt_name in ("SPLASH", "BLOCKWISE"):
    _mech = getattr(ed.AttentionMechanisms, _opt_name, None)
    if _mech is not None:
        ATTN_MAP[_opt_name.lower()] = _mech


# --- Patch registry ---

def _patch_head_dim(model):
    if not hasattr(model.config, "head_dim"):
        model.config.head_dim = (
            model.config.hidden_size // model.config.num_attention_heads
        )
        print(f"  patched head_dim={model.config.head_dim}")


PATCH_FNS: dict[str, callable] = {
    "head_dim": _patch_head_dim,
}


def _apply_patches(model, patches: list[str]):
    for patch_name in patches:
        fn = PATCH_FNS.get(patch_name)
        if fn is None:
            raise ValueError(
                f"Unknown patch: {patch_name}. "
                f"Available: {list(PATCH_FNS)}"
            )
        fn(model)


# --- GCS download with caching ---

def _download_from_gcs(checkpoint_path: str, gcs_bucket: str = None) -> str:
    """Download a checkpoint from GCS, caching locally. Returns local path."""
    if gcs_bucket is None:
        gcs_bucket = GCS_BUCKET

    gcs_path = checkpoint_path
    if not gcs_path.startswith("gs://"):
        gcs_path = f"gs://{gcs_bucket}/{gcs_path}"

    cache_base = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "gcs_checkpoints",
    )
    cache_dir = os.path.join(
        cache_base,
        gcs_path.replace("gs://", "").replace("/", "_"),
    )

    if os.path.exists(cache_dir):
        print(f"Using cached checkpoint at {cache_dir}", flush=True)
        return cache_dir

    print(f"Downloading checkpoint from {gcs_path} to {cache_dir}...", flush=True)

    tmp_dir = cache_dir + ".tmp"

    # `gcloud storage cp -r` occasionally hangs in post-transfer cleanup after
    # printing "Average throughput" — the bytes are on disk but the CLI never
    # exits. Wrap it in a hard timeout + retry so a wedged download can't stall
    # resume forever; the partial tmp_dir is reused across attempts.
    MAX_ATTEMPTS = 6
    ATTEMPT_TIMEOUT = 900  # 15 min; 10GB at 100MB/s is ~100s
    # Back off between retries — without it 4 hosts retry in lockstep and re-hit
    # the same GCS rate-limit window. Staggering by process_index spreads load.
    import time as _time
    BACKOFF_BASE = 15  # seconds; multiplied by attempt number
    # Strip venv leakage so gcloud uses its own bundled protobuf, not the
    # incompatible one our venv installed (otherwise: `module 'google._upb._message'
    # has no attribute 'MessageMapContainer'`).
    clean_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")}

    # Detect layout: midtrained EasyDeL ckpts have a `model/` subdir holding
    # the tensorstore-sharded weights, alongside `opt_state/` we want to skip.
    # HF safetensors ckpts are flat (`*.safetensors` + `*.json` at top level).
    # Listing the prefix once is cheap and lets us pick the right cp pattern.
    prefix = gcs_path.rstrip("/")
    list_proc = subprocess.run(
        ["gcloud", "storage", "ls", f"{prefix}/"],
        capture_output=True, text=True, check=False, env=clean_env,
    )
    has_model_subdir = any(
        line.rstrip("/").endswith("/model")
        for line in list_proc.stdout.splitlines()
    )
    if has_model_subdir:
        # EasyDeL midtrained format: skip opt_state/ (multi-GB resume state,
        # not needed for inference / for SFT fwd init).
        download_ops = [
            ["gcloud", "storage", "cp", "-r", f"{prefix}/model", tmp_dir],
            ["gcloud", "storage", "cp", f"{prefix}/*.json", f"{prefix}/*.txt", tmp_dir],
        ]
    else:
        # HF flat format: copy everything in the prefix.
        download_ops = [
            ["gcloud", "storage", "cp", "-r", f"{prefix}/*", tmp_dir],
        ]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            for op in download_ops:
                subprocess.run(
                    op,
                    check=True,
                    timeout=ATTEMPT_TIMEOUT,
                    env=clean_env,
                    capture_output=True,
                    text=True,
                )
            break
        except subprocess.TimeoutExpired:
            print(f"gcloud cp timed out after {ATTEMPT_TIMEOUT}s (attempt {attempt}/{MAX_ATTEMPTS}), retrying", flush=True)
            if attempt == MAX_ATTEMPTS:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise RuntimeError(
                    f"gcloud cp hung {MAX_ATTEMPTS} times downloading {gcs_path}"
                )
        except subprocess.CalledProcessError as e:
            sleep_s = BACKOFF_BASE * attempt
            print(f"gcloud cp exit {e.returncode} (attempt {attempt}/{MAX_ATTEMPTS}); sleeping {sleep_s}s then retrying", flush=True)
            if attempt == MAX_ATTEMPTS:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise RuntimeError(
                    f"gcloud cp failed downloading {gcs_path} (exit {e.returncode}) after {MAX_ATTEMPTS} attempts.\n"
                    f"--- gcloud stdout (tail) ---\n{(e.stdout or '')[-2000:]}\n"
                    f"--- gcloud stderr (tail) ---\n{(e.stderr or '')[-2000:]}"
                ) from e
            _time.sleep(sleep_s)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
    os.rename(tmp_dir, cache_dir)

    print(f"Downloaded checkpoint to {cache_dir}", flush=True)
    return cache_dir


# --- HF model regional mirror ---

def _mirror_hf_to_gcs(hf_id: str, revision: str, gcs_bucket: str = None) -> str:
    """Two-tier cache for HF model revisions: local → regional GCS → HF.

    Avoids cross-region NAT egress by keeping a per-region copy in GCS.
    Returns a local directory path suitable for `from_pretrained(local_dir)`.
    """
    if gcs_bucket is None:
        gcs_bucket = GCS_BUCKET

    key = f"{hf_id.replace('/', '--')}--{revision}"
    cache_base = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "gcs_hf_mirror",
    )
    local_dir = os.path.join(cache_base, key)
    marker = os.path.join(local_dir, ".complete")

    if os.path.exists(marker):
        print(f"Using cached HF mirror at {local_dir}", flush=True)
        return local_dir

    if gcs_bucket is None:
        raise RuntimeError("GCS_BUCKET unset; cannot use HF mirror layer")

    gcs_path = f"gs://{gcs_bucket}/hf_models/{key}"
    clean_env = {
        k: v for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")
    }

    gcs_hit = subprocess.run(
        ["gcloud", "storage", "ls", f"{gcs_path}/.complete"],
        capture_output=True, text=True, check=False, env=clean_env,
    ).returncode == 0

    os.makedirs(local_dir, exist_ok=True)
    if gcs_hit:
        print(f"HF mirror hit in regional GCS: {gcs_path}", flush=True)
        subprocess.run(
            ["gcloud", "storage", "cp", "-r", f"{gcs_path}/*", local_dir],
            check=True, env=clean_env,
        )
        return local_dir

    from huggingface_hub import snapshot_download
    print(f"HF mirror miss → downloading {hf_id}@{revision} then uploading to {gcs_path}", flush=True)
    snapshot_download(
        repo_id=hf_id,
        revision=revision,
        local_dir=local_dir,
    )
    open(marker, "w").close()
    print(f"Uploading mirror to {gcs_path}", flush=True)
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", f"{local_dir}/*", f"{gcs_path}/"],
        check=False, env=clean_env,
    )
    return local_dir


# --- Mesh creation ---

def create_mesh(sharding_axis_dims: tuple[int, ...] = (-1, 1, 1, 1, 1)) -> Mesh:
    """Create a JAX device mesh for sharding."""
    axis_names = ("dp", "fsdp", "ep", "tp", "sp")
    dims = list(sharding_axis_dims)

    total_devices = jax.device_count()
    specified = 1
    neg_idx = None
    for i, d in enumerate(dims):
        if d == -1:
            neg_idx = i
        else:
            specified *= d
    if neg_idx is not None:
        dims[neg_idx] = total_devices // specified

    print(f"Creating mesh with shape {tuple(dims)} on {total_devices} devices")
    # allow_split_physical_axes=True: without this, mesh_utils requires each
    # logical axis size to equal a product of *whole* physical axes. On v6e-16
    # (physical torus with axes like 4x4) a logical shape like (1, 8, 1, 2, 1)
    # is rejected because 8 can't be expressed as a whole-axis product.
    # Splitting physical axes across logical axes keeps the mesh valid at the
    # cost of a slightly less locality-optimal layout.
    devices = mesh_utils.create_device_mesh(
        tuple(dims), allow_split_physical_axes=True,
    )
    return Mesh(devices, axis_names)


# --- Model loading ---

def load_model(
    checkpoint_path: str | None = None,
    spec: ModelSpec = None,
    max_length: int | None = None,
    dtype: str | None = None,
    param_dtype: str | None = None,
    attn_mechanism: str | None = None,
    revision: str | None = None,
):
    """Load model and tokenizer.

    If revision is provided, loads directly from HF (spec.hf_id @ revision).
    Otherwise loads from GCS via checkpoint_path.

    dtype: compute dtype (matmul/attention). Defaults to spec.dtype.
    param_dtype: storage dtype for parameters. Defaults to dtype (no
      separation). Pass "float32" for fp32 master weights + bf16 compute
      (OLMo2's amp_bf16 setup).

    Returns:
        (model, tokenizer, mesh) tuple.
    """
    max_length = max_length or spec.max_length
    dtype = dtype or spec.dtype
    param_dtype = param_dtype or dtype
    attn_mechanism = attn_mechanism or spec.attn_mechanism

    jnp_dtype = DTYPE_MAP[dtype]
    jnp_param_dtype = DTYPE_MAP[param_dtype]
    if attn_mechanism in ATTN_MAP:
        attn = ATTN_MAP[attn_mechanism]
    else:
        fallback = "sdpa"
        print(
            f"WARNING: attn_mechanism={attn_mechanism!r} not available in this "
            f"EasyDeL build (options: {list(ATTN_MAP)}). Falling back to {fallback!r}.",
            flush=True,
        )
        attn_mechanism = fallback
        attn = ATTN_MAP[fallback]

    if revision:
        if GCS_BUCKET:
            model_path = _mirror_hf_to_gcs(spec.hf_id, revision)
            hf_kwargs = {}
            print(f"Loading from regional HF mirror: {model_path}")
        else:
            model_path = spec.hf_id
            hf_kwargs = {"revision": revision}
            print(f"Loading from HF: {model_path} @ {revision}")
    else:
        model_path = _download_from_gcs(checkpoint_path)
        hf_kwargs = {}

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, **hf_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    mesh = create_mesh(spec.sharding)

    # Work out whether TP is active. If it is, we need to (a) tell EasyDeL
    # about our 5-axis mesh (its default is 4 axes: dp/fsdp/tp/sp, no ep),
    # and (b) switch the partition rules to the TP-aware variant, since
    # EasyDeL defaults to `get_partition_rules(fully_sharded_data_parallel=True)`
    # which replaces every `tp` with `fsdp`/`sp` and makes TP a no-op.
    axis_names = ("dp", "fsdp", "ep", "tp", "sp")
    tp_idx = axis_names.index("tp")
    tp_size_from_spec = spec.sharding[tp_idx]
    use_tp_rules = tp_size_from_spec != 1  # -1 auto-fill won't land on tp here

    partition_rules = None
    if use_tp_rules:
        print(f"  TP={tp_size_from_spec}, loading config to fetch TP-aware partition rules")
        cfg = ed.AutoEasyDeLConfig.from_pretrained(model_path, **hf_kwargs)
        partition_rules = cfg.get_partition_rules(fully_sharded_data_parallel=False)

    print(f"Loading model from {model_path}...")
    print(f"  compute dtype: {dtype}")
    print(f"  param dtype:   {param_dtype}")
    print(f"  sharding: {spec.sharding}  (axes: {axis_names})")
    print(f"  attention: {attn_mechanism}")
    print(f"  devices: {jax.device_count()}")
    print(f"  TP partition rules: {'on' if use_tp_rules else 'off (FSDP-only)'}")

    with mesh:
        model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
            model_path,
            dtype=jnp_dtype,
            param_dtype=jnp_param_dtype,
            sharding_axis_dims=spec.sharding,
            sharding_axis_names=axis_names,
            partition_rules=partition_rules,
            config_kwargs=ed.EasyDeLBaseConfigDict(
                attn_mechanism=attn,
                attn_dtype=jnp_dtype,
                freq_max_position_embeddings=max_length,
                mask_max_position_embeddings=max_length,
                gradient_checkpointing=ed.EasyDeLGradientCheckPointers.CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS,
            ),
            **hf_kwargs,
        )

    _apply_patches(model, spec.patches)

    print("Model loaded successfully.")
    return model, tokenizer, mesh