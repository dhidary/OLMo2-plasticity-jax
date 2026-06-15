# OLMo2 JAX Plasticity Experiments

JAX/EasyDeL pipeline for continued pretraining/ midtraining, SFT, and
OLMES evaluation of OLMo 2 1B on Google Cloud TPUs.

## Findings

> Given 50B-token Dolmino midtraining, the peak post-midtrain performance on
> GSM8K, AGIEval, WinoGrande, MMLU, and TriviaQA occurs after
> 1200k–1600k steps, well before pre-training ends at 1907k steps — and at
> those bases our midtrain also beats the released midtrained checkpoint (full pretraining followed by midtraining) on some knowledge / reasoning tasks:
>
> | task       | best midtrain | offical released `stage2-ingredient3` (1907k) | gap     |
> |---         |---                |---                            |---      |
> | GSM8K      | 0.4906 (base 1200k) | 0.4370                      | **+5.4pt** |
> | WinoGrande | 0.6701 (base 1400k) | 0.6654                      | +0.5pt  |
> | MMLU       | 0.4489 (base 1600k) | 0.4438                      | +0.5pt  |
> | AGIEval    | 0.3711 (base 1200k) | 0.3696                      | +0.2pt  |
>
> The latest stage-1 checkpoint isn't the most plastic starting point for
> downstream training.

Full numbers in [`results/README.md`](results/README.md).

## Setup

Requires a multi-host TPU (validated on v4-32, v5e-16, v6e-16), a GCS bucket,
WandB, and an HF token. Set `GCS_BUCKET`, `GCP_PROJECT`, `TPU_NAME`,
`TPU_ZONE`, `WANDB_API_KEY`, `HF_TOKEN` in the environment.

Each entry point is a plain Python module — write your own launch wrappers
to suit your TPU orchestration. Module entry points:

- `python -m midtraining.train` — continued pretrain
- `python -m sft.train` — SFT
- `python -m eval.mid.bench_<task>` — base-model bench (one per OLMES task)
- `python -m eval.sft_evals.bench_<task>_sft` — chat-format SFT bench

Helper scripts:

- `src/midtraining/scripts/repack_dolmino.py` — pretokenize + shard
  OLMo's `dolmino50` mix to GCS.
- `src/midtraining/scripts/bake_global_shuffle.py` — physically apply the
  PCG64 shuffle so training can stream shards sequentially.
- `src/sft/scripts/eval_holdout_loss.py` — SFT-loss on a held-out split.
- `src/eval/scripts/aggregate_results.py` — pull bench JSONs from GCS and
  rebuild the `results/` tables.
- `src/eval/scripts/aggregate_our_sft.py` — same for the SFT-suite results.

Models live in `src/sft/registry.py` (`olmo-1b`, `olmo2-1b`, `olmo2-13b`,
`olmo3-7b`, `smollm3-3b`). SFT datasets in `src/sft/data.py`. Add either by
appending a spec to the registry.

Upstream: OLMo 2 (AI2) for base + stage-2 checkpoints and the OLMES protocol;
EasyDeL for the JAX/Flax models; Tülu 3 for the chat-format eval suite.

## Support
This work was supported by Google’s TPU Research Cloud (TRC) program: https://sites.research.google/trc/.
