"""Latency benchmarking: p50 / p95 / throughput measurements."""

import time
from typing import Optional

import numpy as np
import torch

from benchmarking._model_loader import load_model_and_tokenizer

SAMPLE_PROMPTS = [
    "What is the first-line treatment for type 2 diabetes?",
    "Explain the mechanism of action of beta-blockers.",
    "What are the diagnostic criteria for sepsis?",
    "Describe the symptoms of myocardial infarction.",
    "What is the recommended dose of aspirin for secondary prevention?",
    "How does metformin reduce blood glucose levels?",
    "What are the contraindications for thrombolytic therapy?",
    "Explain the pathophysiology of heart failure.",
]


def _timed_generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> float:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return (time.perf_counter() - start) * 1000  # ms


def run_latency_benchmark(
    model_path: str,
    quant_bits: Optional[int] = None,
    n_warmup: int = 3,
    n_runs: int = 20,
    max_new_tokens: int = 128,
) -> dict:
    print(f"[latency] Loading model from {model_path} ...")
    model, tokenizer = load_model_and_tokenizer(model_path, quant_bits)

    print(f"[latency] Warming up ({n_warmup} runs) ...")
    for i in range(n_warmup):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        _timed_generate(model, tokenizer, prompt, max_new_tokens)

    latencies = []
    for i in range(n_runs):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        ms = _timed_generate(model, tokenizer, prompt, max_new_tokens)
        latencies.append(ms)
        if (i + 1) % 5 == 0:
            print(f"[latency] {i+1}/{n_runs} done")

    latencies = np.array(latencies)
    results = {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "mean_ms": float(np.mean(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "n_runs": n_runs,
        "max_new_tokens": max_new_tokens,
    }
    print(f"[latency] p50={results['p50_ms']:.1f}ms  p95={results['p95_ms']:.1f}ms")
    return results

