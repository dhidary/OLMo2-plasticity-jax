"""Standalone holdout-loss eval. Loads a saved SFT model and computes
completion-only CE/perplexity on the 200 GSM* dev items bench_gsm8k.py
excludes — so bench operates on the untouched 1119, our eval on the held-200.
Standalone (not inline in sft.train) sidesteps the NNX tracer leak that
hits when train_step and a separate eval-jit share state.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import jax

import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from jax.sharding import NamedSharding, PartitionSpec

_SFT_SINGLE_HOST = os.environ.get("SFT_SINGLE_HOST", "").lower() in ("1", "true", "yes")
if not _SFT_SINGLE_HOST:
    jax.distributed.initialize()

sys.path.insert(0, os.path.expanduser("~/plasticity/src"))

from sft.data import _fmt_gsm8k
from sft.model import load_model
from sft.registry import get_model_spec
from sft.train import tokenize_sft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Model name from sft.registry")
    p.add_argument("--checkpoint", required=True, help="gs:// path to saved SFT model")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    is_main = jax.process_index() == 0
    spec = get_model_spec(args.model)
    _sharding_env = os.environ.get("SFT_SHARDING")
    if _sharding_env:
        spec.sharding = tuple(int(x) for x in _sharding_env.split(","))

    print(f"Loading {args.checkpoint}...")
    model, tokenizer, mesh = load_model(
        checkpoint_path=args.checkpoint, spec=spec, max_length=args.max_length,
    )

    # 200 GSM* held-out (Random(42).sample, matches bench_gsm8k.py exclusion).
    full_test = load_dataset("openai/gsm8k", "main", split="test")
    import random as _rnd
    held_out_idx = sorted(_rnd.Random(42).sample(range(len(full_test)), 200))
    ds_eval = full_test.select(held_out_idx).map(
        _fmt_gsm8k, remove_columns=full_test.column_names,
    )

    ev_input, ev_attn, ev_loss, _ = tokenize_sft(ds_eval, tokenizer, args.max_length)
    n_ev = len(ev_input)
    if is_main:
        print(f"Held-out: {n_ev} examples, {int(ev_loss.sum()):,} completion tokens")

    state = model.to_state()
    graphstate = state.graphstate

    # Single jit on a fresh model — no train_step has run, no tracer leak.
    @jax.jit
    def eval_step(params, batch):
        bf16_params = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            params,
        )
        logits = state.merge(bf16_params)(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        ).logits
        shift_labels = jnp.pad(batch["input_ids"][:, 1:], ((0, 0), (0, 1)))
        shift_loss = jnp.pad(batch["loss_mask"][:, 1:], ((0, 0), (0, 1)))
        log_z = jax.scipy.special.logsumexp(logits, axis=-1)
        logit_at_label = jnp.take_along_axis(
            logits, shift_labels[..., None], axis=-1,
        ).squeeze(-1)
        token_losses = log_z - logit_at_label
        n_toks = shift_loss.sum().astype(jnp.float32)
        sum_loss = (token_losses * shift_loss).sum().astype(jnp.float32)
        return sum_loss, n_toks

    batch_sharding = NamedSharding(mesh, PartitionSpec(("dp", "fsdp")))
    sum_loss_acc = 0.0
    n_toks_acc = 0
    bs = args.batch_size
    t0 = time.time()

    def _pad(arr, pad):
        if pad == 0:
            return arr
        return np.concatenate([arr, np.zeros((pad, args.max_length), dtype=np.int32)])

    with mesh:
        for i in range(0, n_ev, bs):
            end = min(i + bs, n_ev)
            pad = bs - (end - i)
            sl = slice(i, end)
            batch = {
                "input_ids": jax.device_put(_pad(ev_input[sl], pad), batch_sharding),
                "attention_mask": jax.device_put(_pad(ev_attn[sl], pad), batch_sharding),
                "loss_mask": jax.device_put(_pad(ev_loss[sl], pad), batch_sharding),
            }
            sum_loss, n_toks = eval_step(graphstate, batch)
            sum_loss_acc += float(sum_loss)
            n_toks_acc += int(n_toks)

    mean_loss = sum_loss_acc / max(n_toks_acc, 1)
    ppl = math.exp(min(mean_loss, 20.0))

    if is_main:
        print(f"\n=== holdout eval ===")
        print(f"  checkpoint:  {args.checkpoint}")
        print(f"  loss:        {mean_loss:.4f}")
        print(f"  perplexity:  {ppl:.4f}")
        print(f"  tokens:      {n_toks_acc:,}")
        print(f"  wall:        {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
