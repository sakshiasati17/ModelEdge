"""VRAM profiling for a loaded model.

Run in a fresh process per model state (run_benchmark does this) so the peak
reflects only that model. Totals are summed over all visible GPUs, since
device_map="auto" may split a model across both T4s.
"""

import torch

from benchmarking._model_loader import load_model_and_tokenizer

GB = 1024**3


def _gpu_ids() -> list[int]:
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def reset_peak_vram():
    for i in _gpu_ids():
        torch.cuda.reset_peak_memory_stats(i)


def allocated_vram_gb() -> float:
    return sum(torch.cuda.memory_allocated(i) for i in _gpu_ids()) / GB


def peak_vram_gb() -> float:
    return sum(torch.cuda.max_memory_allocated(i) for i in _gpu_ids()) / GB


def run_memory_benchmark(
    model_path: str | None,
    quant_bits: int | None = None,
    model=None,
    tokenizer=None,
) -> dict:
    """Weights footprint plus peak VRAM over loading and a short generation.

    Pass model/tokenizer to profile an already-loaded model; the caller must
    then have called reset_peak_vram() before loading it.
    """
    if model is None:
        print(f"[memory] Profiling {model_path} ...")
        reset_peak_vram()
        model, tokenizer = load_model_and_tokenizer(model_path, quant_bits)

    weights_gb = allocated_vram_gb()

    inputs = tokenizer("What are symptoms of hypertension?", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)

    results = {
        "vram_gb": peak_vram_gb(),
        "vram_weights_gb": weights_gb,
        # get_memory_footprint counts packed 4/8-bit weights correctly,
        # unlike summing numel() * 2.
        "model_footprint_gb": model.get_memory_footprint() / GB,
        "n_gpus": len(_gpu_ids()),
        "quantization_bits": quant_bits,
    }
    print(
        f"[memory] VRAM peak: {results['vram_gb']:.2f} GB  |  "
        f"weights: {weights_gb:.2f} GB  |  footprint: {results['model_footprint_gb']:.2f} GB"
    )
    return results
