"""BitsAndBytes INT8 and INT4 quantization helpers."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _load_base_model(model_path: str):
    """Load model as plain FP16, merging any PEFT adapter so weights are standalone."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    try:
        from peft import PeftModel
        base_name = _get_base_model_name(model_path)
        if base_name:
            base = AutoModelForCausalLM.from_pretrained(
                base_name, torch_dtype=torch.float16, device_map="auto"
            )
            model = PeftModel.from_pretrained(base, model_path)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
    return model, tokenizer


def _get_base_model_name(adapter_path: str) -> str | None:
    """Read base_model_name_or_path from adapter_config.json if present."""
    cfg_path = Path(adapter_path) / "adapter_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("base_model_name_or_path")
    return None


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


def export_awq_int4(model_path: str, output_path: str, calib_data_path: str):
    """Save merged FP16 weights — load with load_in_4bit=True at inference time.
    AutoAWQ is deprecated and broken on transformers>=4.52; using BitsAndBytes instead."""
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

    if bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto"
    )

    with torch.no_grad():
        out = model.generate(**inputs.to(model.device), max_new_tokens=20)

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"[quant] Verification output: {decoded[:120]}")
    print("[quant] Verification passed.")
