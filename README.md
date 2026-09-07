# ModelEdge

**End-to-end LLM optimization pipeline for medical QA** — QLoRA fine-tuning → quantization → benchmarking → serving, built on Llama-3.2-3B.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

ModelEdge fine-tunes an instruction-tuned LLM (Llama-3.2-3B) on medical exam questions with QLoRA, then measures how fine-tuning and quantization trade off **latency, VRAM, accuracy, and hallucination rate**. A serving layer (vLLM + FastAPI + Streamlit) and Kubernetes manifests are included — implemented, but not yet run under load.

**Pipeline:** MedQA data → QLoRA fine-tune (Unsloth) → quantize INT8/INT4 (BitsAndBytes) → benchmark 4 model states → serve (vLLM + FastAPI) + dashboard (Streamlit).

---

## Components

| Module | Role |
|--------|------|
| [`data/`](data/) | Download MedQA / PubMedQA, clean, format to Alpaca instruction style |
| [`training/`](training/) | QLoRA fine-tuning with Unsloth + TRL `SFTTrainer` (single-GPU) |
| [`quantization/`](quantization/) | Export merged model, loaded as INT8 / INT4 via BitsAndBytes |
| [`benchmarking/`](benchmarking/) | Latency (p50/p95), VRAM, exact-match accuracy, hallucination rate |
| [`serving/`](serving/) | vLLM-backed FastAPI + Streamlit dashboard + load tester |
| [`deployment/`](deployment/) | Kubernetes manifests — Deployments, Services, ConfigMap, HPA |

**Stack:** Transformers · PEFT · Unsloth · TRL · BitsAndBytes · Weights & Biases · vLLM · FastAPI · Streamlit · Docker · Kubernetes. Trained on Kaggle 2× T4 (free tier).

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Prepare data
python -m data.prepare_dataset --dataset medqa --output data/processed/

# 2. Fine-tune (or run notebooks/kaggle_training.ipynb on Kaggle 2x T4)
python -m training.finetune --config training/config.yaml

# 3. Quantize
python quantization/quantize.py --model outputs/finetuned --bits 8
python quantization/quantize.py --model outputs/finetuned --bits 4

# 4. Benchmark base / fine-tuned / quantized states
python benchmarking/run_benchmark.py --models base fp16 int8 int4

# 5. Serve + dashboard
uvicorn serving.api:app --host 0.0.0.0 --port 8000
streamlit run serving/dashboard.py
```

Output paths default to `outputs/`; set `MODELEDGE_OUTPUT_DIR` to override (the Kaggle notebook uses `/kaggle/working/outputs`).

---

## Benchmark Results

Fine-tuned on 9,733 MedQA-USMLE examples (3 epochs, single T4), evaluated on 200 held-out questions:

| Model State     | Latency p50 | VRAM (GB)* | Accuracy** | Hallucination Rate |
|-----------------|-------------|------------|------------|---------------------|
| Base FP16       | 6,634.6 ms  | 5.98       | 0.635      | 0.035               |
| Fine-tuned FP16 | 9,864.3 ms  | 3.95       | 0.550      | 0.095               |
| Fine-tuned INT8 | 10,034.2 ms | 7.62       | 0.550      | 0.095               |
| Fine-tuned INT4 | 10,022.0 ms | 8.93       | 0.550      | 0.095               |

Training: loss 1.94 → 1.14, eval loss 1.26 → 1.21 over 423 steps (~5.8 h). Adapter: 24.3M of 3.24B params (0.75%), 93 MB.

> **\* VRAM figures are not yet reliable** — states were profiled in one shared process with an in-place merge/reload, so lower precisions report inflated peaks (INT4 > FP16). Per-process isolation is open work.
>
> **\*\* Accuracy is exact substring match**, which penalizes correct-but-verbose answers. Reviewing outputs, the fine-tuned drop is partly this scoring artifact and partly genuine error. An option-letter / judge-based scorer (`choices` are now persisted per record to enable it) is the planned fix.

---

## Roadmap

| Stage | Status |
|-------|--------|
| Data prep, QLoRA fine-tuning, quantization, benchmarking | ✅ Complete |
| Kaggle training run + real benchmarks | ✅ Complete |
| Serving layer + Kubernetes manifests | ✅ Implemented (not yet run under load) |
| Option-based accuracy + isolated VRAM profiling | ⏳ Pending |
| Load testing + failure analysis | ⏳ Pending |

---

## License

MIT
