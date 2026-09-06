"""
Full benchmark suite — runs latency, memory, and accuracy for each model state.

Usage:
    python benchmarking/run_benchmark.py --models fp16 int8 int4
    python benchmarking/run_benchmark.py --models fp16 int8 int4 --output results/
"""

import argparse
import json
from pathlib import Path

from benchmarking.accuracy import run_accuracy_benchmark
from benchmarking.latency import run_latency_benchmark
from benchmarking.memory import run_memory_benchmark

import os as _os

_BASE = _os.environ.get("MODELEDGE_OUTPUT_DIR", "outputs")

MODEL_PATHS = {
    "fp16": f"{_BASE}/finetuned",
    "int8": f"{_BASE}/finetuned_int8",
    "int4": f"{_BASE}/finetuned_int4",
    "base": "unsloth/Llama-3.2-3B-Instruct",
}


def run_all(model_keys: list[str], test_file: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for key in model_keys:
        path = MODEL_PATHS.get(key)
        if path is None:
            print(f"[bench] Unknown model key '{key}' — skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"[bench] Benchmarking: {key}  ({path})")
        print(f"{'='*60}")

        quant_bits = {"fp16": None, "int8": 8, "int4": 4, "base": None}.get(key)

        lat = run_latency_benchmark(path, quant_bits=quant_bits)
        mem = run_memory_benchmark(path, quant_bits=quant_bits)
        acc = run_accuracy_benchmark(path, test_file, quant_bits=quant_bits)

        all_results[key] = {
            "model_path": path,
            "quantization_bits": quant_bits,
            **lat,
            **mem,
            **acc,
        }

        result_path = output_dir / f"{key}_results.json"
        with open(result_path, "w") as f:
            json.dump(all_results[key], f, indent=2)
        print(f"[bench] Saved → {result_path}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    _print_summary_table(all_results)
    print(f"\n[bench] Full results → {summary_path}")


def _print_summary_table(results: dict):
    headers = ["model", "p50_ms", "p95_ms", "vram_gb", "accuracy", "hallucination_rate"]
    row_fmt = "{:<10} {:>9} {:>9} {:>9} {:>10} {:>20}"
    print("\n" + row_fmt.format(*headers))
    print("-" * 72)
    for key, r in results.items():
        print(
            row_fmt.format(
                key,
                f"{r.get('p50_ms', 0):.1f}",
                f"{r.get('p95_ms', 0):.1f}",
                f"{r.get('vram_gb', 0):.2f}",
                f"{r.get('accuracy', 0):.3f}",
                f"{r.get('hallucination_rate', 0):.3f}",
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["base", "fp16", "int8", "int4"],
        default=["base", "fp16", "int8", "int4"],
    )
    parser.add_argument("--test_file", default="data/processed/medqa/test.jsonl")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    run_all(args.models, args.test_file, Path(args.output))


if __name__ == "__main__":
    main()
