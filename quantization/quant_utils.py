"""BitsAndBytes INT8 and INT4 quantization helpers."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarking._model_loader import bnb_config_for, merge_adapter
from benchmarking.adapter_utils import is_peft_adapter


def _load_base_model(model_path: str):
    """Load model as plain FP16, merging any PEFT adapter into its full-precision base."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if is_peft_adapter(model_path):
        # CPU merge: only the merged weights are saved, nothing needs the GPU.
        model = merge_adapter(model_path, device_map="cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="cpu"
        )
    return model, tokenizer


def export_bnb_int8(model_path: str, output_path: str):
    """Save merged FP16 weights — load with load_in_8bit=True at inference time."""
    model, tokenizer = _load_base_model(model_path)

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    quant_info = {
        "quantization_bits": 8,
        "quantization_method": "bitsandbytes",
        "load_instruction": "Use BitsAndBytesConfig(load_in_8bit=True) when loading.",
    }
    with open(out / "quant_info.json", "w") as f:
        json.dump(quant_info, f, indent=2)

    print(f"[quant] INT8-ready model saved → {out}")
    print("[quant] Note: weights are FP16 on disk. Pass load_in_8bit=True at load time.")


def export_bnb_int4(model_path: str, output_path: str):
    """Save merged FP16 weights — load with load_in_4bit=True (BitsAndBytes) at inference time."""
    model, tokenizer = _load_base_model(model_path)

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    quant_info = {
        "quantization_bits": 4,
        "quantization_method": "bitsandbytes",
        "load_instruction": "Use BitsAndBytesConfig(load_in_4bit=True) when loading.",
    }
    with open(out / "quant_info.json", "w") as f:
        json.dump(quant_info, f, indent=2)

    print(f"[quant] INT4-ready model saved → {out}")
    print("[quant] Note: weights are FP16 on disk. Pass load_in_4bit=True at load time.")


def verify_quantized_model(model_path: str, bits: int):
    """Run a quick forward pass to confirm the quantized model loads correctly."""
    print(f"[quant] Verifying {bits}-bit model at {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer("What is hypertension?", return_tensors="pt")

    bnb_config = bnb_config_for(bits)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto"
    )

    with torch.no_grad():
        out = model.generate(**inputs.to(model.device), max_new_tokens=20)

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"[quant] Verification output: {decoded[:120]}")
    print("[quant] Verification passed.")
