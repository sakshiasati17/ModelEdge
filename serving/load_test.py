"""
Concurrent load tester for the ModelEdge FastAPI server.

Tests with 10 / 50 / 100 concurrent requests and reports
throughput, p50, and p95 latency.

Usage:
    python serving/load_test.py --url http://localhost:8000 --concurrency 10 50 100
"""

import argparse
import asyncio
import json
import time
from statistics import mean, quantiles

import httpx

SAMPLE_QUESTIONS = [
    "What is the first-line treatment for type 2 diabetes?",
    "Explain the mechanism of action of ACE inhibitors.",
    "What are the diagnostic criteria for pneumonia?",
    "Describe the signs and symptoms of appendicitis.",
    "What is the recommended antibiotic for community-acquired pneumonia?",
    "How does warfarin work and what are its key interactions?",
    "What are the indications for coronary artery bypass grafting?",
    "Explain the pathophysiology of chronic kidney disease.",
]


async def single_request(client: httpx.AsyncClient, base_url: str, idx: int) -> dict:
    payload = {
        "question": SAMPLE_QUESTIONS[idx % len(SAMPLE_QUESTIONS)],
        "max_tokens": 128,
        "temperature": 0.1,
    }
    start = time.perf_counter()
    try:
        resp = await client.post(f"{base_url}/infer", json=payload, timeout=120.0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"ok": True, "latency_ms": elapsed_ms, "status": resp.status_code}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"ok": False, "latency_ms": elapsed_ms, "error": str(e)}


async def run_concurrent_batch(
    base_url: str, concurrency: int, n_requests: int
) -> dict:
    print(f"\n[load] concurrency={concurrency}  total_requests={n_requests}")
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(idx):
        async with semaphore:
            async with httpx.AsyncClient() as client:
                return await single_request(client, base_url, idx)

    start = time.perf_counter()
    results = await asyncio.gather(*[bounded(i) for i in range(n_requests)])
    wall_time = time.perf_counter() - start

    ok = [r for r in results if r["ok"]]
    failed = len(results) - len(ok)
    latencies = [r["latency_ms"] for r in ok]

    if latencies:
        p50, p95 = quantiles(latencies, n=100)[49], quantiles(latencies, n=100)[94]
    else:
        p50, p95 = 0, 0

    throughput = len(ok) / wall_time

    summary = {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "success": len(ok),
        "failed": failed,
        "throughput_rps": throughput,
        "p50_ms": p50,
        "p95_ms": p95,
        "mean_ms": mean(latencies) if latencies else 0,
        "wall_time_s": wall_time,
    }
    print(
        f"  success={len(ok)}/{n_requests}  "
        f"throughput={throughput:.2f} req/s  "
        f"p50={p50:.0f}ms  p95={p95:.0f}ms"
    )
    return summary


async def main_async(base_url: str, concurrency_levels: list[int], n_requests: int):
    all_results = {}
    for c in concurrency_levels:
        r = await run_concurrent_batch(base_url, c, n_requests)
        all_results[f"concurrency_{c}"] = r

    out_path = "results/load_test.json"
    import os; os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[load] Results saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--n_requests", type=int, default=200)
    args = parser.parse_args()

    asyncio.run(main_async(args.url, args.concurrency, args.n_requests))


if __name__ == "__main__":
    main()
