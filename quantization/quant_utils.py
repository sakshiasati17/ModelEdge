"""BitsAndBytes INT8 and AWQ INT4 quantization helpers."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def export_bnb_int8(model_path: str, output_path: str):
    """Load a model and re-save with INT8 BitsAndBytes config."""
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    config_extra = {"quantization_bits": 8, "quantization_method": "bitsandbytes"}
    with open(out / "quant_info.json", "w") as f:
        json.dump(config_extra, f, indent=2)

    print(f"[quant] INT8 model saved → {out}")


def export_awq_int4(model_path: str, output_path: str, calib_data_path: str):
    """Quantize to INT4 using AutoAWQ with calibration data."""
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise RuntimeError(
            "autoawq is required for INT4 quantization. "
            "Install it with: pip install autoawq"
        )

    calib_texts = _load_calib_texts(calib_data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out))
    tokenizer.save_pretrained(str(out))

    config_extra = {"quantization_bits": 4, "quantization_method": "awq"}
    with open(out / "quant_info.json", "w") as f:
        json.dump(config_extra, f, indent=2)

    print(f"[quant] INT4 AWQ model saved → {out}")


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
        try:
            from awq import AutoAWQForCausalLM

            model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=True)
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    with torch.no_grad():
        out = model.generate(**inputs.to(model.device), max_new_tokens=20)

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"[quant] Verification output: {decoded[:120]}")
    print("[quant] Verification passed.")
