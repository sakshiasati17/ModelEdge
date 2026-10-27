"""
Task accuracy and hallucination rate evaluation.

Scoring logic lives in benchmarking/scoring.py (see there for definitions).
"""

import json

import torch

from benchmarking._model_loader import load_model_and_tokenizer
from benchmarking.scoring import score_predictions

INSTRUCTION_PROMPT = (
    "Below is a medical question. Answer it accurately and concisely.\n\n"
    "### Instruction:\n{question}\n\n"
    "### Input:\n{choices}\n\n"
    "### Response:\n"
)


def _generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_records(test_file: str, max_samples: int) -> list[dict]:
    records = []
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records[:max_samples]


def run_accuracy_benchmark(
    model_path: str | None,
    test_file: str,
    quant_bits: int | None = None,
    max_samples: int = 200,
    model=None,
    tokenizer=None,
    predictions_path: str | None = None,
) -> dict:
    if model is None:
        print(f"[accuracy] Loading model from {model_path} ...")
        model, tokenizer = load_model_and_tokenizer(model_path, quant_bits)

    records = load_records(test_file, max_samples)
    print(f"[accuracy] Evaluating on {len(records)} samples ...")

    predictions = []
    for i, rec in enumerate(records):
        prompt = INSTRUCTION_PROMPT.format(
            question=rec.get("instruction", rec.get("question", "")),
            choices=rec.get("input", ""),
        )
        predictions.append(_generate_answer(model, tokenizer, prompt))
        if (i + 1) % 50 == 0:
            print(f"[accuracy] {i+1}/{len(records)} done")

    if predictions_path:
        with open(predictions_path, "w") as f:
            for rec, pred in zip(records, predictions, strict=True):
                f.write(json.dumps({
                    "question": rec.get("instruction", rec.get("question", "")),
                    "answer": rec.get("output", rec.get("answer", "")),
                    "choices": rec.get("choices"),
                    "prediction": pred,
                }) + "\n")

    results = score_predictions(records, predictions)
    if results["n_with_choices"] < len(records):
        print(
            f"[accuracy] WARNING: only {results['n_with_choices']}/{len(records)} records "
            "have choices — regenerate the test file with data.prepare_dataset."
        )
    option_acc = results["option_accuracy"]
    print(
        f"[accuracy] exact_match={results['accuracy']:.3f}  "
        f"option_accuracy={'n/a' if option_acc is None else f'{option_acc:.3f}'}  "
        f"hallucination_rate={results['hallucination_rate']:.3f}"
    )
    return results
