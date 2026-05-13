"""Model registry — model specs and checkpoint metadata."""

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    """Properties of the model itself — not training hyperparams."""
    hf_id: str
    sharding: tuple = (-1, 1, 1, 1, 1)
    patches: list[str] = field(default_factory=list)
    dtype: str = "bfloat16"
    attn_mechanism: str = "sdpa"
    max_length: int = 4096
    checkpoints: list[str] = field(default_factory=list)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "olmo-1b": ModelSpec(
        hf_id="allenai/OLMo-1B",
        checkpoints=[
            "step0-tokens0B",
            "step5000-tokens21B",
            "step20000-tokens84B",
            "step50000-tokens210B",
            "step100000-tokens419B",
            "step200000-tokens838B",
            "step449000-tokens1882B",
        ],
    ),
    "olmo2-1b": ModelSpec(
        hf_id="allenai/OLMo-2-0425-1B",
        sharding=(1, -1, 1, 1, 1),
        patches=["head_dim"],
        attn_mechanism="splash",
        checkpoints=[
            "stage1-step1000000-tokens2098B",
            "stage1-step1200000-tokens2517B",
            "stage1-step1400000-tokens2937B",
            "stage1-step1600000-tokens3356B",
            "stage1-step1800000-tokens3775B",
            "stage2-ingredient3-step3000-tokens7B",
            "stage2-ingredient3-step23852-tokens51B",
        ],
    ),
    "olmo2-13b": ModelSpec(
        hf_id="allenai/OLMo-2-0425-13B",
        sharding=(1, -1, 1, 1, 1),
        patches=["head_dim"],
        checkpoints=[
            "stage1-step0-tokens0B",
            "stage1-step10000-tokens42B",
            "stage1-step100000-tokens419B",
            "stage1-step500000-tokens2097B",
            "stage1-step953674-tokens4001B",
            "stage2-ingredient3-step5000-tokens21B",
            "stage2-ingredient3-step11931-tokens51B",
        ],
    ),
    "smollm3-3b": ModelSpec(
        hf_id="HuggingFaceTB/SmolLM3-3B-checkpoints",
        checkpoints=[
            "stage1-step-40000",
            "stage1-step-3440000",
            "stage2-step-3480000",
            "stage2-step-4200000",
            "stage3-step-4240000",
            "stage3-step-4480000",
            "stage3-step-4720000",
            "lc-32k-to-64k-step-20000",
            "it-mid-training",
            "it-LC-expert",
        ],
    ),
    "olmo3-7b": ModelSpec(
        hf_id="allenai/Olmo-3-1025-7B",
        sharding=(1, -1, 1, 1, 1),
        checkpoints=[],
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]