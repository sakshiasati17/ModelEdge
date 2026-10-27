"""Shared model loader for benchmarking — handles plain models and PEFT adapters."""

import gc
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmarking.adapter_utils import is_peft_adapter, resolve_base_model


def bnb_config_for(quant_bits: int | None) -> BitsAndBytesConfig | None:
    if quant_bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    return None


def merge_adapter(adapter_path: str, device_map: str = "auto"):
    """Merge a LoRA adapter into its full-precision FP16 base model."""
    from peft import PeftModel

    base_name = resolve_base_model(adapter_path)
    print(f"[loader] Merging adapter {adapter_path} into FP16 base {base_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.float16, device_map=device_map
    )
    return PeftModel.from_pretrained(base, adapter_path).merge_and_unload()


def load_model_and_tokenizer(model_path: str, quant_bits: int | None):
    """Load a model from a local path or HF hub ID.

    If the path contains adapter_config.json (PEFT/LoRA adapter), the adapter
    is merged into the full-precision base first, so quantization is applied
    to a standalone merged model rather than through the adapter.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = bnb_config_for(quant_bits)

    if is_peft_adapter(model_path) and bnb_config:
        # Merge on CPU so the FP16 copy never occupies (and pollutes) GPU memory,
        # then reload the merged weights quantized.
        merged = merge_adapter(model_path, device_map="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            merged.save_pretrained(tmp)
            del merged
            gc.collect()
            model = AutoModelForCausalLM.from_pretrained(
                tmp, quantization_config=bnb_config, device_map="auto"
            )
    elif is_peft_adapter(model_path):
        model = merge_adapter(model_path)
    else:
        kwargs = {"device_map": "auto"}
        if bnb_config:
            kwargs["quantization_config"] = bnb_config
        else:
            kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    model.eval()
    return model, tokenizer
