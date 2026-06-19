"""VRAM and RAM memory profiling for a loaded model."""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def run_memory_benchmark(
    model_path: str,
    quant_bits: Optional[int] = None,
) -> dict:
    print(f"[memory] Profiling {model_path} ...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant_bits == 8:
        bnb = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map="auto"
        )
    elif quant_bits == 4:
        bnb4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb4, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

    model.eval()

    inputs = tokenizer("What are symptoms of hypertension?", return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32)

    vram_gb = _peak_vram_gb()

    n_params = sum(p.numel() for p in model.parameters())
    param_gb = n_params * 2 / (1024**3)  # FP16 baseline estimate

    results = {
        "vram_gb": vram_gb,
        "param_count_M": n_params / 1e6,
        "param_size_fp16_gb": param_gb,
        "quantization_bits": quant_bits,
    }
    print(f"[memory] VRAM peak: {vram_gb:.2f} GB  |  Params: {n_params/1e6:.0f}M")
    return results
