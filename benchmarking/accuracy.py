"""
Task accuracy and hallucination rate evaluation.

Metrics:
  - exact_match: answer string in model output (case-insensitive)
  - option_match: correct multiple-choice option selected
  - hallucination_rate: fraction of responses containing confident
    factual claims that contradict the ground truth answer
"""

import json
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

INSTRUCTION_PROMPT = (
    "Below is a medical question. Answer it accurately and concisely.\n\n"
    "### Instruction:\n{question}\n\n"
    "### Input:\n{choices}\n\n"
    "### Response:\n"
)

OPTION_LABELS = list("ABCDE")

HALLUCINATION_CONFIDENCE_PATTERNS = [
    r"\b(definitely|certainly|absolutely|always|never|clearly|obviously)\b",
    r"\b(the answer is|it is|this is|that is)\b",
]


def _load_model(model_path: str, quant_bits: Optional[int]):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant_bits == 8:
        bnb = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map="auto"
        )
    elif quant_bits == 4:
        bnb4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb4, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
    model.eval()
    return model, tokenizer


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


def exact_match(prediction: str, ground_truth: str) -> bool:
    return ground_truth.lower().strip() in prediction.lower()


def option_match(prediction: str, answer: str, choices: list[str]) -> bool:
    pred_lower = prediction.lower()
    for i, choice in enumerate(choices):
        label = OPTION_LABELS[i]
        if choice.lower() in answer.lower():
            if (
                label.lower() in pred_lower[:10]
                or choice.lower() in pred_lower
            ):
                return True
    return False


def is_hallucination(prediction: str, ground_truth: str) -> bool:
    if exact_match(prediction, ground_truth):
        return False
    confidence_hit = any(
        re.search(pat, prediction, re.IGNORECASE)
        for pat in HALLUCINATION_CONFIDENCE_PATTERNS
    )
    return confidence_hit


def run_accuracy_benchmark(
    model_path: str,
    test_file: str,
    quant_bits: Optional[int] = None,
    max_samples: int = 200,
) -> dict:
    print(f"[accuracy] Loading model from {model_path} ...")
    model, tokenizer = _load_model(model_path, quant_bits)

    records = []
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records = records[:max_samples]
    print(f"[accuracy] Evaluating on {len(records)} samples ...")

    exact_hits = 0
    option_hits = 0
    hallucination_count = 0

    for i, rec in enumerate(records):
        prompt = INSTRUCTION_PROMPT.format(
            question=rec.get("instruction", rec.get("question", "")),
            choices=rec.get("input", ""),
        )
        prediction = _generate_answer(model, tokenizer, prompt)
        answer = rec.get("output", rec.get("answer", ""))
        choices = rec.get("choices", [])

        if exact_match(prediction, answer):
            exact_hits += 1
        if choices and option_match(prediction, answer, choices):
            option_hits += 1
        if is_hallucination(prediction, answer):
            hallucination_count += 1

        if (i + 1) % 50 == 0:
            print(f"[accuracy] {i+1}/{len(records)} done")

    n = len(records)
    results = {
        "accuracy": exact_hits / n,
        "option_accuracy": option_hits / n if any(rec.get("choices") for rec in records) else None,
        "hallucination_rate": hallucination_count / n,
        "n_samples": n,
        "exact_hits": exact_hits,
        "hallucination_count": hallucination_count,
    }
    print(
        f"[accuracy] accuracy={results['accuracy']:.3f}  "
        f"hallucination_rate={results['hallucination_rate']:.3f}"
    )
    return results
