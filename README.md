# ModelEdge

**An end-to-end LLM optimization pipeline for medical question answering** — fine-tuning, quantization, benchmarking, and production-style serving in one repository.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active%20development-yellow.svg)]()

---

## Overview

ModelEdge takes a base instruction-tuned language model (Llama-3.2-3B / Phi-3.5-mini), fine-tunes it on medical QA data using QLoRA, and measures the impact of fine-tuning and quantization across three dimensions: **latency, memory footprint, and task accuracy / hallucination rate**.

The project is built around a real production deployment pattern — **vLLM + FastAPI behind Kubernetes** — so the same pipeline that produces the model is also responsible for serving it under load.

### Why medical QA

- Base models are confidently wrong on medical facts — a measurable hallucination problem
- Fine-tuning produces a clear, quantifiable improvement
- Evaluation is unambiguous: hallucination rate before vs. after, accuracy before vs. after

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          ModelEdge Pipeline                           │
├───────────────┬───────────────┬───────────────┬──────────────────────┤
│    Dataset    │  Fine-Tuning  │ Quantization  │    Benchmarking       │
│  Preparation  │  QLoRA +      │ FP16→INT8→    │  Latency / Memory /   │
│   (MedQA)     │  Unsloth      │    INT4       │  Accuracy / Halluc.   │
├───────────────┴───────────────┴───────────────┴──────────────────────┤
│         vLLM Serving + FastAPI + Streamlit Dashboard                  │
├────────────────────────────────────────────────────────────────────-─┤
│              Kubernetes (Deployments, Services, HPA)                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Components

| # | Module | Description |
|---|--------|-------------|
| 1 | [`data/`](data/) | Downloads MedQA / PubMedQA, cleans records, formats to Alpaca instruction style |
| 2 | [`training/`](training/) | QLoRA fine-tuning with Unsloth, multi-GPU via HuggingFace Accelerate |
| 3 | [`quantization/`](quantization/) | FP16 → INT8 (BitsAndBytes) → INT4 (AWQ) |
| 4 | [`benchmarking/`](benchmarking/) | Latency (p50/p95), VRAM footprint, task accuracy, hallucination rate |
| 5 | [`serving/`](serving/) | vLLM-backed FastAPI, Streamlit comparison dashboard, concurrent load tester |
| 6 | [`deployment/`](deployment/) | Kubernetes manifests — Deployments, Services, ConfigMap, HPA |

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Base model | Llama-3.2-3B / Phi-3.5-mini |
| Fine-tuning | HuggingFace Transformers, PEFT, Accelerate |
| Training speed-up | **Unsloth** (~2× faster QLoRA) |
| Quantization | BitsAndBytes (INT8), AWQ (INT4) |
| Experiment tracking | Weights & Biases |
| Inference serving | vLLM |
| API layer | FastAPI |
| Dashboard | Streamlit + Plotly |
| Orchestration | Kubernetes (Deployments, Services, HPA) |
| Training hardware | Kaggle 2× T4 (free tier) |
| Core | Python, Pandas, NumPy |

---

## Project Structure

```
ModelEdge/
├── data/
│   ├── prepare_dataset.py      # Download + clean MedQA / PubMedQA
│   └── dataset_utils.py        # Alpaca / chat-template formatters
├── training/
│   ├── finetune.py             # QLoRA training loop (Accelerate-wrapped)
│   ├── config.yaml             # Hyperparameters
│   └── trainer_utils.py        # Model loading, tokenization, W&B, checkpointing
├── quantization/
│   ├── quantize.py             # INT8 + INT4 entry point
│   └── quant_utils.py          # BitsAndBytes + AWQ helpers
├── benchmarking/
│   ├── run_benchmark.py        # Full benchmark suite across model states
│   ├── latency.py              # p50 / p95 latency
│   ├── accuracy.py             # Task accuracy + hallucination scorer
│   └── memory.py               # VRAM / RAM profiling
├── serving/
│   ├── api.py                  # FastAPI wrapper around vLLM
│   ├── dashboard.py            # Streamlit comparison dashboard
│   └── load_test.py            # Concurrent load tester (10/50/100 req)
├── deployment/
│   ├── deployment.yaml         # vLLM + FastAPI Deployments
│   ├── service.yaml            # LoadBalancer / ClusterIP / PVC
│   ├── configmap.yaml          # Shared runtime config
│   ├── hpa.yaml                # Horizontal Pod Autoscalers
│   └── minikube_test.sh        # Local cluster validation script
├── notebooks/
│   └── kaggle_training.ipynb   # End-to-end training notebook (Kaggle 2x T4)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Prepare the dataset
python data/prepare_dataset.py --dataset medqa --output data/processed/

# 2. Fine-tune (recommended: run notebooks/kaggle_training.ipynb on Kaggle 2x T4)
accelerate launch --num_processes 2 training/finetune.py --config training/config.yaml

# 3. Quantize
python quantization/quantize.py --model outputs/finetuned --bits 8
python quantization/quantize.py --model outputs/finetuned --bits 4

# 4. Benchmark base / fine-tuned / quantized states
python benchmarking/run_benchmark.py --models base fp16 int8 int4

# 5. Serve
uvicorn serving.api:app --host 0.0.0.0 --port 8000
streamlit run serving/dashboard.py
```

### Run on Kubernetes (local, via Minikube)

```bash
bash deployment/minikube_test.sh
kubectl get pods
kubectl get hpa
```

---

## Benchmark Results

QLoRA fine-tune of `unsloth/Llama-3.2-3B-Instruct` on MedQA-USMLE (9,733 train / 1,145 val), 3 epochs / 423 steps on Kaggle 2× T4. Run: 2026-06-22 → 06-23. Train loss 1.94 → 1.14, best eval loss **1.21**. Accuracy = exact-match (ground-truth answer string contained in first 64 generated tokens), evaluated on 200 held-out test questions.

| Model State     | Latency p50 | Latency p95 | VRAM (GB) | Accuracy | Hallucination Rate |
|-----------------|-------------|-------------|-----------|----------|---------------------|
| Base FP16       | 6634.6 ms   | 6689.4 ms   | 5.98      | 0.635    | 0.035               |
| Fine-tuned FP16 | 9864.3 ms   | 9956.0 ms   | 3.95      | 0.550    | 0.095               |
| Fine-tuned INT8 | 10034.2 ms  | 10153.5 ms  | 7.62      | 0.550    | 0.095               |
| Fine-tuned INT4 | 10022.0 ms  | 10097.2 ms  | 8.93      | 0.550    | 0.095               |

> ⚠️ **These numbers expose two problems, not a clean win.** (1) The fine-tuned model scores **lower** than base on exact-match accuracy (0.55 vs 0.64) and hallucinates **more** (0.095 vs 0.035) — fine-tuning taught it to answer verbosely, so the short ground-truth string is less often a substring and the confidence-phrase hallucination heuristic fires more. Eval *loss* improved, but the downstream MCQ metric regressed. (2) The quantization VRAM column is inverted (INT4 8.93 GB > FP16 base 5.98 GB) and FP16/INT8/INT4 accuracy is byte-identical — both are artifacts of the benchmark's load path (the adapter is merged in FP16 and re-loaded, so peak VRAM double-counts and all finetuned variants collapse to effectively the same quantized weights). See [Failure Analysis](#failure-analysis). Full run log and qualitative samples in [`RESULTS.md`](RESULTS.md).

---

## Failure Analysis

The first benchmark run surfaced three issues worth fixing before drawing conclusions about fine-tuning quality:

1. **Verbosity regression.** The fine-tuned model rambles instead of answering concisely — e.g. for a ground truth of `Captopril` it returns a paragraph on hydrochlorothiazide. The `exact_match` metric (`ground_truth.lower() in prediction.lower()`) penalizes this, and the long, assertive prose trips the hallucination heuristic. The fix is either a stricter generation/stop config at eval time, or an option-letter–based scorer instead of raw substring match.
2. **Quantization not measured in isolation.** `benchmarking/_model_loader.py` merges the LoRA adapter in FP16 and then re-saves/re-loads it under BitsAndBytes, so peak VRAM double-counts the transient FP16 copy (INT4 reports *more* VRAM than FP16 base) and all three finetuned precisions converge to identical accuracy. Quantized memory/accuracy should be profiled in a fresh process per model, or by loading the merged checkpoint once and quantizing in place.
3. **Latency inversion.** Base (FP16, ~6.6 s) is faster than the finetuned/quantized variants (~10 s) — BitsAndBytes dequant overhead on T4 dominates, so quantization here trades speed for (intended) memory savings rather than improving latency.

A per-question failure taxonomy (rare drug names, multi-step reasoning, numerical dosage errors) will be added alongside the Streamlit dashboard once the scorer and quant-profiling fixes above are in.

---

## Roadmap

| Stage | Status |
|-------|--------|
| Dataset preparation pipeline | ✅ Complete |
| QLoRA fine-tuning (Unsloth + Accelerate, multi-GPU ready) | ✅ Complete |
| Quantization pipeline (INT8 / INT4) | ✅ Complete |
| Benchmarking suite | ✅ Complete |
| Serving layer (vLLM + FastAPI + Streamlit) | ✅ Complete |
| Kubernetes deployment manifests | ✅ Complete |
| Training run on Kaggle 2x T4 | ✅ Complete (3 epochs, eval loss 1.21) |
| Quantized model benchmarks (real numbers) | ✅ Complete — surfaced metric/loader bugs (see Failure Analysis) |
| Load testing under Kubernetes | ⏳ Pending |
| Failure analysis | 🔄 In progress — fixing scorer + quant profiling |

---

## License

MIT
