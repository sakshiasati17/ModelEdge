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

| Model State | Latency p50 | VRAM (GB) | Accuracy | Hallucination Rate |
|-------------|-------------|-----------|----------|---------------------|
| Base FP16   | TBD         | TBD       | TBD      | TBD                 |
| Fine-tuned FP16 | TBD     | TBD       | TBD      | TBD                 |
| Fine-tuned INT4 | TBD     | TBD       | TBD      | TBD                 |

> Results will be populated after the training run on Kaggle 2x T4 completes — see [Roadmap](#roadmap).

---

## Failure Analysis

After benchmarking, a sample of incorrect responses will be manually reviewed and categorized by failure type (e.g. rare drug names, multi-step reasoning, numerical dosage errors). This taxonomy will be added here alongside the Streamlit dashboard.

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
| Training run on Kaggle 2x T4 | ⏳ In progress |
| Quantized model benchmarks (real numbers) | ⏳ Pending training |
| Load testing under Kubernetes | ⏳ Pending training |
| Failure analysis | ⏳ Pending benchmarks |

---

## License

MIT
