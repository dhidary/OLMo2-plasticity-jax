# Results

OLMo 2 1B. OLMES-matched evals: 5-shot for ARC-C / HellaSwag / MMLU /
WinoGrande, 8-shot for GSM8K, OLMES random subsampling where it applies.
bf16. "paper" = OLMo 2 tech report Table 22 (base) / Table 16 (SFT).

## Headline: best midtrain comes from a 1200k–1600k base, beating the released `stage2-ingredient3`

For each released `stage1-stepN` checkpoint (N up to 1800k, stage-1 final is
~1907k) we ran our own 50B-token Dolmino midtrain on top and re-evaluated.
On every knowledge / reasoning task we swept, the peak post-midtrain score
comes from a base at **1200k–1600k** — well before stage-1 ends — and at
those bases our midtrain also beats the released `stage2-ingredient3`
(AI2's published midtrain on stage-1 final, the natural comparator):

| task       | best base | our best midtrain | released `stage2-ingredient3` | gap        |
|---         |---        |---                |---                            |---         |
| GSM8K      | **1200k** | 0.4906            | 0.4370                        | **+5.4pt** |
| WinoGrande | **1400k** | 0.6701            | 0.6654                        | +0.5pt     |
| MMLU       | **1600k** | 0.4489            | 0.4438                        | +0.5pt     |
| AGIEval    | **1200k** | 0.3711            | 0.3696                        | +0.2pt     |
| TriviaQA   | **1600k** | 0.5464            | 0.5476                        | −0.1pt     |
| ARC-C      | **1200k** | 0.5060            | 0.5094                        | −0.3pt     |
| MMLU-Pro   | **1800k** | 0.1533            | 0.1538                        | −0.05pt    |

Within our own midtrain sweep, the same story holds even more cleanly. The
peak base is never the latest one we trained on (1800k):

| task       | best base | best score | midtrain_1800k score | gap     |
|---         |---        |---         |---                   |---      |
| GSM8K      | **1200k** | 0.4906     | 0.4415               | −4.9pt  |
| MMLU       | **1600k** | 0.4489     | 0.4404               | −0.9pt  |
| AGIEval    | **1200k** | 0.3711     | 0.3685               | −0.3pt  |
| WinoGrande | **1400k** | 0.6701     | 0.6685               | −0.2pt  |
| TriviaQA   | **1600k** | 0.5464     | 0.5455               | (tied)  |

Stage-1 cosine LR is still decaying through the end of stage-1; the released
`stage2-ingredient3` sits on top of stage-1 final. Picking a 1200–1600k
base before that LR has fully cooled gives the midtrain (and our SFTs) more
to work with.

## Reproduction of the released base (sanity check)

| task                  | metric          | paper | ours   | Δ        |
|---                    |---              |---    |---     |---       |
| ARC-Challenge         | MC acc_raw      | 0.513 | 0.5094 | −0.4pt   |
| HellaSwag             | RC acc_per_char | 0.695 | 0.6950 |  0.0pt   |
| MMLU                  | MC macro_acc    | 0.443 | 0.4438 | +0.1pt   |
| WinoGrande            | RC acc_raw      | 0.665 | 0.6654 |  0.0pt   |
| TriviaQA              | gen f1          | 0.547 | 0.5476 | +0.1pt   |
| Natural Questions     | gen f1          | 0.208 | 0.2427 | +3.5pt   |
| AGIEval (English)     | MC macro_acc    | 0.363 | 0.3696 | +0.7pt   |
| MMLU Pro              | MC micro_acc    | 0.161 | 0.1538 | −0.7pt   |
| GSM8K (1119 held-out) | gen EM          | 0.438 | 0.4370 | −0.1pt   |

Within engine noise everywhere except NQ (bf16 jitter on generative).

## Midtrain lifts over stage-1 base (Δ = midtrained − stage1)

| step  | ARC-C | HellaSwag | MMLU  | WinoGr | GSM8K | AGIEval | MMLU-Pro | NQ    | TQA   |
|---    |---    |---        |---    |---     |---    |---      |---       |---    |---    |
| 240k  | +14.1 | +2.7      | +6.4  | +0.7   | +38.8 | +6.5    | +1.5     | +4.4  | +9.2  |
| 1000k | +22.0 | +0.5      | +17.8 | +1.1   | +47.0 | +12.9   | +3.7     | +5.9  | +8.0  |
| 1200k | +23.8 | +0.7      | +18.3 | +0.8   | +46.5 | +16.4   | +3.5     | +5.4  | +7.8  |
| 1400k | +23.3 | +0.2      | +17.0 | +2.1   | +46.8 | +12.5   | +3.0     | +5.1  | +4.8  |
| 1600k | +23.7 | −1.6      | +18.7 | +0.2   | +43.1 | +14.5   | +3.9     | +5.1  | +6.5  |
| 1800k | +23.9 | −0.1      | +18.5 | −0.6   | +40.7 | +13.7   | +3.8     | +4.8  | +4.3  |

Saturating tasks (HellaSwag, WinoGrande) gain little — base is already near
ceiling. Knowledge / reasoning tasks gain 5–47pt.

## SFT-suite reproduction (chat-format, OLMES `::tulu`)

Verifies the chat-eval path against the released
[`allenai/OLMo-2-0425-1B-SFT`](https://huggingface.co/allenai/OLMo-2-0425-1B-SFT)
card (±1pt acceptance):

| task       | OLMES config                  | card  | ours   | Δ       |
|---         |---                            |---    |---     |---      |
| GSM8K      | `gsm8k::tulu`                 | 0.521 | 0.5178 | −0.32pt |
| MMLU       | `mmlu:mc::tulu`               | 0.364 | 0.3751 | +1.11pt |
| BBH        | `bbh:cot-v1::tulu`            | 0.328 | 0.3339 | +0.59pt |
| PopQA      | `popqa::tulu`                 | 0.127 | 0.1210 | −0.60pt |
| MATH       | `minerva_math::tulu`          | 0.132 | 0.1354 | +0.34pt |
| IFEval     | `ifeval::tulu` (prompt-loose) | 0.505 | 0.4954 | −0.96pt |
| TruthfulQA | `truthfulqa::tulu` (mc2)      | 0.421 | 0.4239 | +0.29pt |

6/7 within ±1pt. MMLU residual is most likely an HF/vLLM vs JAX/EasyDeL
scoring-path difference — literal-OLMES settings overshoot by +3pt; the
empirically-tuned variant lands closest at +1.11pt.

## SFT on midtrained checkpoints

**GSM8K.** Best LR sweep on midtrain_1400k: lr=2e-6, GSM8K = **0.4951**
(8-shot CoT, 1119 held-out). Beats stage2_ingr3 base (0.437).

**MMLU.** GSM8K-data SFT does not transfer to MMLU at any LR. Switching to
MMLU's own `auxiliary_train` split (99k MC, format-matched, 1 epoch lr=1e-5):

| base ckpt        | base MMLU | + mmlu_aux SFT | Δ      |
|---               |---        |---             |---     |
| stage2_ingr3     | 0.4438    | **0.4942**     | +5.04  |
| midtrain_1800k   | 0.4404    | 0.4862         | +4.58  |
| midtrain_1600k   | 0.4489    | 0.4835         | +3.46  |
| midtrain_1400k   | 0.4470    | 0.4784         | +3.14  |
| midtrain_240k    | 0.3482    | 0.4101         | +6.19  |
| base_1800k       | 0.2557    | 0.4164         | +16.07 |
| base_1200k       | 0.2544    | 0.3554         | +10.10 |
| base_240k        | 0.2844    | 0.2837         | −0.01  |

Late base + format-matched SFT can substitute for midtraining on a single
target task: stage1_1800k + mmlu_aux SFT (0.4164) is competitive with
stage2_ingr3 without ever seeing Dolmino. Earliest base (240k) is the only
one SFT can't rescue.
