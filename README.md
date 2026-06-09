# ModelEdge

End-to-end model fine-tuning and optimization pipeline for medical question answering.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          ModelEdge Pipeline                          │
├───────────────┬───────────────┬───────────────┬──────────────────────┤
│    Dataset    │  Fine-Tuning  │ Quantization  │    Benchmarking      │
│  Preparation  │  QLoRA +      │ FP16→INT8→    │  Latency / Memory /  │
│  (MedQA)     │  Unsloth      │    INT4       │    Accuracy          │
├───────────────┴───────────────┴───────────────┴──────────────────────┤
│               vLLM Serving  +  FastAPI  +  Streamlit Dashboard       │
└──────────────────────────────────────────────────────────────────────┘
```

## Components

| # | Module | Description |
|---|--------|-------------|
| 1 | `data/` | MedQA download, cleaning, Alpaca instruction format |
| 2 | `training/` | QLoRA fine-tuning with Unsloth (2× faster) |
| 3 | `quantization/` | FP16 → INT8 (BitsAndBytes) → INT4 (AWQ) |
| 4 | `benchmarking/` | Latency p50/p95, VRAM, task accuracy, hallucination rate |
| 5 | `serving/` | vLLM + FastAPI + Streamlit comparison dashboard |

## Tech Stack

| Layer | Tool |
|-------|------|
| Model | Llama-3.2-3B / Phi-3.5-mini |
| Fine-tuning | HuggingFace Transformers, PEFT |
| Speed boost | **Unsloth** (2× faster QLoRA) |
| Quantization | BitsAndBytes (INT8), AWQ (INT4) |
| Experiment tracking | W&B (Weights & Biases) |
| Serving | vLLM |
| API | FastAPI |
| Dashboard | Streamlit |
| Hardware | Google Colab T4 (free) |

## Quick Start

```bash
pip install -r requirements.txt

# 1. Prepare dataset
python data/prepare_dataset.py --dataset medqa --output data/processed/

# 2. Fine-tune (run on Colab T4 — see notebooks/colab_training.ipynb)
python training/finetune.py --config training/config.yaml

# 3. Quantize
python quantization/quantize.py --model outputs/finetuned --bits 8
python quantization/quantize.py --model outputs/finetuned --bits 4

# 4. Benchmark all three model states
python benchmarking/run_benchmark.py --models fp16 int8 int4

# 5. Serve + dashboard
uvicorn serving.api:app --host 0.0.0.0 --port 8000
streamlit run serving/dashboard.py
```

## Benchmark Results

| Model State | Latency p50 | VRAM (GB) | Accuracy | Hallucination Rate |
|-------------|-------------|-----------|----------|--------------------|
| Base FP16   | ~320 ms     | 6.2       | 42 %     | 38 %               |
| FT FP16     | ~320 ms     | 6.2       | 71 %     | 12 %               |
| FT INT4     | ~140 ms     | 2.5       | 68 %     | 14 %               |

> Numbers above are illustrative. Run `benchmarking/run_benchmark.py` for real results on your hardware.

## Project Structure

```
ModelEdge/
├── data/
│   ├── prepare_dataset.py      # Download + clean MedQA / PubMedQA
│   ├── dataset_utils.py        # Alpaca / chat-template formatters
│   └── processed/              # Generated data (gitignored)
├── training/
│   ├── finetune.py             # QLoRA training entry point
│   ├── config.yaml             # Hyperparameters
│   └── trainer_utils.py        # W&B callbacks, checkpointing
├── quantization/
│   ├── quantize.py             # INT8 + INT4 pipelines
│   └── quant_utils.py          # BitsAndBytes + AWQ helpers
├── benchmarking/
│   ├── run_benchmark.py        # Full benchmark suite
│   ├── latency.py              # Throughput / latency metrics
│   ├── accuracy.py             # Task accuracy + hallucination scorer
│   └── memory.py               # VRAM / RAM profiling
├── serving/
│   ├── api.py                  # FastAPI wrapper
│   ├── dashboard.py            # Streamlit comparison dashboard
│   └── load_test.py            # Concurrent load tester
├── notebooks/
│   └── colab_training.ipynb    # Colab T4 training notebook
├── requirements.txt
└── README.md
```

## Week-by-Week Build Plan

| Week | Focus | Key Deliverable |
|------|-------|-----------------|
| 1 | Data + fine-tuning | Adapter weights saved, W&B loss curves |
| 2 | Evaluation + quantization | INT8 / INT4 models, benchmark numbers |
| 3 | Serving + dashboard | vLLM live, Streamlit tradeoff chart |
| 4 | Depth + polish | Failure analysis, clean GitHub, README |

## What's Left ❌ (needs GPU — use Kaggle 2x T4, free)

### New additions since scaffold ➕
- [ ] **Add HuggingFace Accelerate to `training/finetune.py`** — wrap model, optimizer,
      dataloader with `accelerator.prepare()`. Makes training multi-GPU ready.
      Use `tensor_parallel_size=2` in vLLM for both T4s.
- [ ] **Migrate from Colab notebook to Kaggle** — upload `notebooks/kaggle_training.ipynb`
      to Kaggle, set GPU T4 x2, add WANDB_API_KEY to Kaggle secrets

### Execution steps
1. [ ] **Run training on Kaggle 2x T4** — connect W&B, run finetune.py,
       save adapter weights to `/kaggle/working/outputs/finetuned`
2. [ ] **Run quantization** — once finetuned weights exist:
       ```bash
       python quantization/quantize.py --model outputs/finetuned --bits 8
       python quantization/quantize.py --model outputs/finetuned --bits 4
       ```
3. [ ] **Run benchmarks** — fill `results/summary.json` with real numbers:
       ```bash
       python benchmarking/run_benchmark.py --models base fp16 int8 int4
       ```
4. [ ] **Stand up vLLM with both T4s** — use `tensor_parallel_size=2`,
       run `load_test.py` at 1/10/50 concurrent requests
5. [ ] **Failure analysis** — review 30 wrong answers, categorize into failure
       types, add taxonomy to README + Streamlit dashboard
6. [ ] **Update README** — replace placeholder benchmark table with real
       p50/p95 latency, VRAM, accuracy, hallucination rate numbers + W&B run link
