"""
Full benchmark suite — runs latency, memory, and accuracy for each model state.

Each model state is benchmarked in its own subprocess: the model is loaded
once, all three benchmarks run on it, and the process exits, so GPU memory
from one state can never leak into another state's VRAM numbers.

Usage:
    python -m benchmarking.run_benchmark --models fp16 int8 int4
    python -m benchmarking.run_benchmark --models fp16 int8 int4 --output results/
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_BASE = os.environ.get("MODELEDGE_OUTPUT_DIR", "outputs")

# int8/int4 load the adapter directly: the loader merges it into the FP16 base
# and quantizes at load time, so stale quantization.quantize exports can't be
# benchmarked by mistake.
MODEL_PATHS = {
    "fp16": f"{_BASE}/finetuned",
    "int8": f"{_BASE}/finetuned",
    "int4": f"{_BASE}/finetuned",
    "base": "unsloth/Llama-3.2-3B-Instruct",
}

QUANT_BITS = {"fp16": None, "int8": 8, "int4": 4, "base": None}


def benchmark_one(key: str, test_file: str, output_dir: Path, max_samples: int) -> dict:
    """Load one model state once and run every benchmark on it (worker process)."""
    # Imported here so the parent process never initializes CUDA.
    from benchmarking._model_loader import load_model_and_tokenizer
    from benchmarking.accuracy import run_accuracy_benchmark
    from benchmarking.latency import run_latency_benchmark
    from benchmarking.memory import reset_peak_vram, run_memory_benchmark

    path = MODEL_PATHS[key]
    quant_bits = QUANT_BITS[key]

    reset_peak_vram()
    model, tokenizer = load_model_and_tokenizer(path, quant_bits)

    # Memory first: its peak covers loading plus one short generation.
    mem = run_memory_benchmark(path, quant_bits, model=model, tokenizer=tokenizer)
    lat = run_latency_benchmark(path, quant_bits, model=model, tokenizer=tokenizer)
    acc = run_accuracy_benchmark(
        path,
        test_file,
        quant_bits,
        max_samples=max_samples,
        model=model,
        tokenizer=tokenizer,
        predictions_path=str(output_dir / f"{key}_predictions.jsonl"),
    )

    return {"model_path": path, **lat, **mem, **acc, "quantization_bits": quant_bits}


def run_all(model_keys: list[str], test_file: str, output_dir: Path, max_samples: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    failed = []

    for key in model_keys:
        print(f"\n{'='*60}")
        print(f"[bench] Benchmarking: {key}  ({MODEL_PATHS[key]})")
        print(f"{'='*60}", flush=True)

        result_path = output_dir / f"{key}_results.json"
        result_path.unlink(missing_ok=True)
        cmd = [
            sys.executable, "-m", "benchmarking.run_benchmark",
            "--worker", key,
            "--test_file", test_file,
            "--output", str(output_dir),
            "--max_samples", str(max_samples),
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0 or not result_path.exists():
            print(f"[bench] {key} FAILED (exit code {proc.returncode})")
            failed.append(key)
            continue

        with open(result_path) as f:
            all_results[key] = json.load(f)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    _print_summary_table(all_results)
    print(f"\n[bench] Full results → {summary_path}")
    if failed:
        sys.exit(f"[bench] Failed model states: {', '.join(failed)}")


def _print_summary_table(results: dict):
    headers = ["model", "p50_ms", "p95_ms", "vram_gb", "accuracy", "option_acc", "hallucination_rate"]
    row_fmt = "{:<10} {:>9} {:>9} {:>9} {:>10} {:>10} {:>20}"
    print("\n" + row_fmt.format(*headers))
    print("-" * 83)
    for key, r in results.items():
        print(
            row_fmt.format(
                key,
                f"{r.get('p50_ms', 0):.1f}",
                f"{r.get('p95_ms', 0):.1f}",
                f"{r.get('vram_gb', 0):.2f}",
                f"{r.get('accuracy', 0):.3f}",
                "n/a" if r.get("option_accuracy") is None else f"{r['option_accuracy']:.3f}",
                f"{r.get('hallucination_rate', 0):.3f}",
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_PATHS),
        default=["base", "fp16", "int8", "int4"],
    )
    parser.add_argument("--test_file", default="data/processed/medqa/test.jsonl")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--max_samples", type=int, default=200)
    # Internal: benchmark a single state in this process (spawned by run_all).
    parser.add_argument("--worker", choices=list(MODEL_PATHS), help=argparse.SUPPRESS)
    args = parser.parse_args()

    output_dir = Path(args.output)
    if args.worker:
        result = benchmark_one(args.worker, args.test_file, output_dir, args.max_samples)
        with open(output_dir / f"{args.worker}_results.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    run_all(args.models, args.test_file, output_dir, args.max_samples)


if __name__ == "__main__":
    main()
