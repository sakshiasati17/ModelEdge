"""
PEFT adapter helpers, kept free of torch/transformers so they are testable in CI.

Unsloth QLoRA training records a pre-quantized 4-bit checkpoint (e.g.
"unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit") as the adapter's base model.
Merging the adapter into those 4-bit weights loses precision, and the result
is not a true FP16 model. resolve_base_model maps it back to the full-precision
checkpoint so the merge happens in FP16.
"""

import json
import os
import re
from pathlib import Path

# Override the resolved base model entirely (e.g. a local FP16 copy).
BASE_MODEL_ENV = "MODELEDGE_BASE_MODEL"

_QUANT_SUFFIX = re.compile(r"(-unsloth)?-bnb-4bit$", re.IGNORECASE)


def is_peft_adapter(model_path: str) -> bool:
    return (Path(model_path) / "adapter_config.json").exists()


def adapter_base_model(adapter_path: str) -> str:
    """base_model_name_or_path exactly as recorded in adapter_config.json."""
    with open(Path(adapter_path) / "adapter_config.json") as f:
        return json.load(f)["base_model_name_or_path"]


def is_prequantized_name(model_name: str) -> bool:
    return bool(_QUANT_SUFFIX.search(model_name))


def full_precision_name(model_name: str) -> str:
    """Strip Unsloth's "-bnb-4bit" / "-unsloth-bnb-4bit" suffix, if present."""
    return _QUANT_SUFFIX.sub("", model_name)


def resolve_base_model(adapter_path: str) -> str:
    """The full-precision base model to merge the adapter into."""
    override = os.environ.get(BASE_MODEL_ENV)
    if override:
        return override
    return full_precision_name(adapter_base_model(adapter_path))
