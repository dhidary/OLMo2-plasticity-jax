# Plasticity

JAX/EasyDeL pipeline for **continued pretraining ("midtraining"), SFT, and
OLMES-faithful evaluation** of OLMo 2 1B on Google Cloud TPUs.

## Findings

> Under a fixed 50B-token Dolmino midtrain, the peak post-midtrain score on
> **GSM8K, AGIEval, WinoGrande, MMLU, and TriviaQA** lands on stage-1
> checkpoints at **1200k–1600k**, well before stage-1 ends (1907k) — and at
> those bases our midtrain also **beats the released `stage2-ingredient3`**
> (AI2's published midtrain on stage-1 final) on knowledge / reasoning
> tasks:
>
> | task       | our best midtrain | released `stage2-ingredient3` | gap     |
> |---         |---                |---                            |---      |
> | GSM8K      | 0.4906 (base 1200k) | 0.4370                      | **+5.4pt** |
> | WinoGrande | 0.6701 (base 1400k) | 0.6654                      | +0.5pt  |
> | MMLU       | 0.4489 (base 1600k) | 0.4438                      | +0.5pt  |
> | AGIEval    | 0.3711 (base 1200k) | 0.3696                      | +0.2pt  |
>
> The latest stage-1 checkpoint isn't the most plastic starting point for
> downstream training.

Full numbers in [`results/README.md`](results/README.md).

## Repo uses

- **JAX-on-TPU recipe for OLMo 2** — HF→EasyDeL loading from GCS, JAX mesh
  setup, per-host DP vs multi-host FSDP, FusedClippedAdamW (`src/sft/model.py`,
  `src/sft/registry.py`, `src/midtraining/train.py`).
- **EasyDeL training and eval examples** — every bench is a self-contained
  EasyDeL script; `_common.py` and `_sft_common.py` factor out the shared
  mesh and scoring helpers.
- **OLMES-faithful reimplementation in JAX** — matches OLMES on the released
  OLMo 2 1B base to within engine noise across 9 tasks (see
  [`results/README.md`](results/README.md)). Useful as a reference for
  `*::olmes` / `*::tulu` configs outside the HF/vLLM stack.
- **Empirical plasticity data on OLMo 2 1B** — which intermediate stage-1
  checkpoints retain the most downstream-learning capacity. See "Headline
  finding" above and `results/README.md`.

## Pipelines

| pipeline       | what it does | code |
|---             |---           |---   |
| `midtraining/` | Custom JAX/optax loop. 50B-token continued pretrain on Dolmino over an arbitrary base checkpoint. Resumable, preemption-safe. | `src/midtraining/` |
| `sft/`         | Custom JAX/optax SFT loop. Completion-only loss, chat templates, FSDP / per-host DP. | `src/sft/` |
| `eval/`        | OLMES-matched benches: ARC-C, HellaSwag, MMLU(+Pro), WinoGrande, GSM8K, NQ, TriviaQA, AGIEval, plus the chat SFT suite (BBH, MATH, PopQA, IFEval, TruthfulQA). | `src/eval/` |

All three share `sft/registry.py` (model specs), `sft/model.py` (HF→EasyDeL
loader, JAX mesh), and a GCS layout.

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
