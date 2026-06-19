"""BitsAndBytes INT8 and AWQ INT4 quantization helpers."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def export_bnb_int8(model_path: str, output_path: str):
    """
    Save a copy of the model with a BitsAndBytes INT8 config embedded.

    BNB INT8 is load-time only — weights on disk stay in FP16.
    We copy the model weights as-is and write a config that tells
    transformers to apply INT8 quantization at load time.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load in FP16 (not quantized) so save_pretrained works
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # Record that this path should be loaded with INT8 at inference time
    quant_info = {
        "quantization_bits": 8,
        "quantization_method": "bitsandbytes",
        "load_instruction": "Use BitsAndBytesConfig(load_in_8bit=True) when loading this model.",
    }
    with open(out / "quant_info.json", "w") as f:
        json.dump(quant_info, f, indent=2)

    print(f"[quant] INT8-ready model saved → {out}")
    print("[quant] Note: weights are FP16 on disk. Pass load_in_8bit=True at load time.")


def export_awq_int4(model_path: str, output_path: str, calib_data_path: str):
    """Save model for INT4 BitsAndBytes loading (AutoAWQ is deprecated/broken on transformers>=4.52).
    Weights stay FP16 on disk; pass load_in_4bit=True at load time."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
    )

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    quant_info = {
        "quantization_bits": 4,
        "quantization_method": "bitsandbytes",
        "load_instruction": "Use BitsAndBytesConfig(load_in_4bit=True) when loading this model.",
    }
    with open(out / "quant_info.json", "w") as f:
        json.dump(quant_info, f, indent=2)

    print(f"[quant] INT4-ready model saved → {out}")
    print("[quant] Note: weights are FP16 on disk. Pass load_in_4bit=True at load time.")


def _load_calib_texts(jsonl_path: str, max_samples: int = 128) -> list[str]:
    texts = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            texts.append(rec.get("text", rec.get("instruction", "")))
            if len(texts) >= max_samples:
                break
    return texts


def verify_quantized_model(model_path: str, bits: int):
    """Run a quick forward pass to confirm the quantized model loads correctly."""
    print(f"[quant] Verifying {bits}-bit model at {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer("What is hypertension?", return_tensors="pt")

    if bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto"
        )
    else:
        bnb4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb4, device_map="auto"
        )

    with torch.no_grad():
        out = model.generate(**inputs.to(model.device), max_new_tokens=20)

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"[quant] Verification output: {decoded[:120]}")
    print("[quant] Verification passed.")
