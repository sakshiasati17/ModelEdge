"""Shared model loader for benchmarking — handles plain models and PEFT adapters."""

import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _is_peft_adapter(model_path: str) -> bool:
    return (Path(model_path) / "adapter_config.json").exists()


def _get_base_model_name(adapter_path: str) -> str:
    with open(Path(adapter_path) / "adapter_config.json") as f:
        return json.load(f)["base_model_name_or_path"]


def load_model_and_tokenizer(model_path: str, quant_bits: Optional[int]):
    """Load a model from a local path or HF hub ID.

    If the path contains adapter_config.json (PEFT/LoRA adapter), loads the
    base model and merges the adapter before returning, so quantization is
    applied to a standalone merged model rather than through the adapter.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if quant_bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16
        )
    elif quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

    if _is_peft_adapter(model_path):
        from peft import PeftModel
        base_name = _get_base_model_name(model_path)
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        if bnb_config:
            # Re-load merged model with quantization
            import tempfile, os
            with tempfile.TemporaryDirectory() as tmp:
                model.save_pretrained(tmp)
                tokenizer.save_pretrained(tmp)
                model = AutoModelForCausalLM.from_pretrained(
                    tmp, quantization_config=bnb_config, device_map="auto"
                )
    else:
        kwargs = {"device_map": "auto"}
        if bnb_config:
            kwargs["quantization_config"] = bnb_config
        else:
            kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    model.eval()
    return model, tokenizer
