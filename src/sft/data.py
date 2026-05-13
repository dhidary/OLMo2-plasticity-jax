"""SFT dataset registry.

Conversational chat-format datasets only. Each row has a `messages` field
(list of {role, content} dicts). The trainer renders via
`tokenizer.apply_chat_template` and masks loss to assistant content.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, load_dataset as hf_load_dataset


@dataclass
class SftDatasetSpec:
    hf_id: str
    hf_subset: str | None = None
    train_split: str = "train"
    default_max_length: int = 4096


SFT_DATASETS: dict[str, SftDatasetSpec] = {
    # Allen AI's official OLMo-2-0425-1B SFT dataset (post-stage2 → SFT).
    # 866,138 rows, conversational `messages` field, multi-domain
    # (chat / math / code / IF / safety).
    "tulu3_olmo2": SftDatasetSpec(
        hf_id="allenai/tulu-3-sft-olmo-2-mixture-0225",
        default_max_length=4096,
    ),
}


def get_sft_spec(name: str) -> SftDatasetSpec:
    if name not in SFT_DATASETS:
        raise ValueError(
            f"Unknown SFT dataset: {name}. Available: {list(SFT_DATASETS)}"
        )
    return SFT_DATASETS[name]


def load_sft_dataset(name: str, max_samples: int | None = None) -> Dataset:
    spec = get_sft_spec(name)
    if spec.hf_subset:
        ds = hf_load_dataset(spec.hf_id, spec.hf_subset, split=spec.train_split)
    else:
        ds = hf_load_dataset(spec.hf_id, split=spec.train_split)
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(max_samples))
    return ds
