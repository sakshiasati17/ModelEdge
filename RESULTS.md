# ModelEdge — Run Results

Full record of the first end-to-end run: QLoRA fine-tune → INT8/INT4 quantization → benchmark suite, executed on Kaggle 2× T4.

- **Date:** 2026-06-22 22:02 → 2026-06-23 04:xx UTC
- **Base model:** `unsloth/Llama-3.2-3B-Instruct`
- **Branch trained from:** `dev/modeledge-pipeline` @ `02a6302`
- **W&B run:** `qlora-llama3.2-3b` — project `modeledge-medqa` (entity `sakshiasati51-university-of-colorado-boulder`), run id `5tlb6hoq`

---

## 1. Dataset

`GBaker/MedQA-USMLE-4-options`, cleaned and formatted to Alpaca instruction style.

| Split | Records |
|-------|---------|
| Train | 9,733   |
| Val   | 1,145   |
| Test  | 573     |

(11,451 valid records after cleaning; 11,451 → 9,733 / 1,145 / 573.)

---

## 2. Fine-tuning (QLoRA, Unsloth)

- LoRA: r=16, alpha=32, dropout=0, all attention + MLP projections (24.3M trainable params, 0.75% of 3.24B)
- 3 epochs, 423 steps, effective batch 16 (2 × grad-accum 8), packing enabled (bfd)
- LR 2e-4 cosine, warmup 28, adamw_8bit, fp16
- Total runtime: **5h 48m** (`train_runtime` 20,880 s)

| Metric | Value |
|--------|-------|
| Train loss (start → end) | 1.94 → 1.14 (mean 1.227) |
| Best eval loss | **1.2107** (epoch 2.84) |
| Eval loss trajectory | 1.2614 → 1.2280 → 1.2148 → 1.2107 |
| total_flos | 1.16e17 |

Adapter (93 MB) saved to `/kaggle/working/outputs/finetuned`.

---

## 3. Quantization

| Target | Tool | Output | Status |
|--------|------|--------|--------|
| INT8 | BitsAndBytes | `outputs/finetuned_int8` | ✅ verified |
| INT4 | BitsAndBytes (AWQ path replaced) | `outputs/finetuned_int4_awq` | ✅ verified |

Both verified with a "What is hypertension?" smoke generation. Note printed by the script: *weights are FP16 on disk; quantization is applied at load time via `load_in_8bit`/`load_in_4bit`.* This detail matters for the benchmark anomalies below.

---

## 4. Benchmark suite (200 test questions, greedy decoding)

| Model | p50 (ms) | p95 (ms) | VRAM (GB) | Params | Accuracy | Hallucination |
|-------|----------|----------|-----------|--------|----------|---------------|
| base  | 6634.6   | 6689.4   | 5.98      | 3213M  | 0.635    | 0.035         |
| fp16  | 9864.3   | 9956.0   | 3.95      | 1841M  | 0.550    | 0.095         |
| int8  | 10034.2  | 10153.5  | 7.62      | 1841M  | 0.550    | 0.095         |
| int4  | 10022.0  | 10097.2  | 8.93      | 1841M  | 0.550    | 0.095         |

Accuracy metric = `exact_match`: ground-truth answer string present (case-insensitive) in the first 64 generated tokens. Hallucination = non-match response containing a confidence phrase (`definitely`, `the answer is`, `clearly`, …).

### Anomalies — read these before trusting the table

1. **Fine-tuning regressed the task metric.** Accuracy dropped 0.635 → 0.550 and hallucination rose 0.035 → 0.095, even though eval *loss* improved monotonically. Cause: the tuned model answers verbosely (see samples), so the short ground-truth string is less often a literal substring, and the assertive prose triggers the hallucination heuristic. This is a metric/format mismatch, not necessarily a worse model — `eval_loss` says it learned the data.

2. **Quantization VRAM is inverted and accuracy is identical across precisions.** INT4 (8.93 GB) reports *more* VRAM than the FP16 base (5.98 GB), and fp16/int8/int4 share byte-identical accuracy and hallucination. Both stem from `_model_loader.py`: the LoRA adapter is merged in FP16 and then re-saved/re-loaded under BitsAndBytes in the same process. Peak VRAM (`max_memory_allocated`) double-counts the transient FP16 merged copy, and all three finetuned variants collapse to effectively the same quantized weights (note the 1841M "param count" — bitsandbytes packing — for all three, vs 3213M for base). Quantization needs to be profiled one model per fresh process.

3. **Latency inversion.** Base FP16 (~6.6 s) beats every finetuned/quantized variant (~10 s). On T4, BitsAndBytes dequant overhead dominates; quantization here buys (intended) memory, not speed.

---

## 5. Qualitative samples (fine-tuned, first 5 test questions)

| Ground truth | Prediction (truncated) | Match |
|--------------|------------------------|-------|
| Fresh frozen plasma | "Desmopressin is the best choice for this patient's bleeding…" | ✗ |
| Signs of pneumonia | "Signs of pneumonia" is the correct answer because… | ✓ |
| Lactic acidosis | "…'Infections' is a side effect of this drug." | ✗ |
| Erythrogenic toxin-induced cytokine release | "Erythrogenic toxin-induced cytokine release"… scarlet fever… | ✓ |
| Captopril | "Hydrochlorothiazide (HCTZ) is a thiazide diuretic…" | ✗ |

The pattern is clear: when the model leads with the answer it scores; when it leads with reasoning, the substring scorer misses it even when the eventual answer may be correct. This is the primary lever for the next iteration.

---

## 6. Artifacts

Saved on Kaggle under `/kaggle/working/saved_outputs/` (adapter, INT8, INT4 checkpoints + `results/` JSON). These are excluded from git (`outputs/`, `*.safetensors`); export from the Kaggle Output tab → Save as Dataset (`modeledge-outputs`) to persist them.
