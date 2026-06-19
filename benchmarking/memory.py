"""VRAM and RAM memory profiling for a loaded model."""

from typing import Optional

import torch

from benchmarking._model_loader import load_model_and_tokenizer


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

    model, tokenizer = load_model_and_tokenizer(model_path, quant_bits)

    inputs = tokenizer("What are symptoms of hypertension?", return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32)

    vram_gb = _peak_vram_gb()
    n_params = sum(p.numel() for p in model.parameters())
    param_gb = n_params * 2 / (1024**3)

    results = {
        "vram_gb": vram_gb,
        "param_count_M": n_params / 1e6,
        "param_size_fp16_gb": param_gb,
        "quantization_bits": quant_bits,
    }
    print(f"[memory] VRAM peak: {vram_gb:.2f} GB  |  Params: {n_params/1e6:.0f}M")
    return results
