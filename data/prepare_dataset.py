"""
Dataset preparation pipeline.

Downloads MedQA or PubMedQA, cleans the records, and converts them to
Alpaca-style instruction format ready for fine-tuning.

Usage:
    python data/prepare_dataset.py --dataset medqa --output data/processed/
    python data/prepare_dataset.py --dataset pubmedqa --output data/processed/
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset

from data.dataset_utils import (
    format_alpaca,
    format_chat_template,
    split_dataset,
    validate_record,
)


DATASET_CONFIGS = {
    "medqa": {
        "hf_name": "bigbio/med_qa",
        "hf_config": "med_qa_en_bigbio_qa",
        "splits": ["train", "validation", "test"],
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answer",
    },
    "pubmedqa": {
        "hf_name": "bigbio/pubmed_qa",
        "hf_config": "pubmed_qa_labeled_fold0_bigbio_qa",
        "splits": ["train", "test"],
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answer",
    },
}


def load_raw_dataset(dataset_name: str):
    cfg = DATASET_CONFIGS[dataset_name]
    print(f"[data] Loading {cfg['hf_name']} ({cfg['hf_config']}) ...")
    raw = load_dataset(cfg["hf_name"], cfg["hf_config"])
    return raw, cfg


def clean_record(record: dict, cfg: dict) -> dict | None:
    """Return a normalised record or None to drop it."""
    question = record.get(cfg["question_field"], "").strip()
    choices = record.get(cfg["choices_field"], [])
    answer = record.get(cfg["answer_field"], "")

    if not question or not answer:
        return None

    if isinstance(choices, list) and len(choices) > 0:
        choice_texts = [
            c["text"] if isinstance(c, dict) else str(c) for c in choices
        ]
    else:
        choice_texts = []

    return {"question": question, "choices": choice_texts, "answer": str(answer).strip()}


def build_instruction_dataset(raw_dataset, cfg: dict, fmt: str = "alpaca") -> list[dict]:
    records = []
    for split_name in cfg["splits"]:
        if split_name not in raw_dataset:
            continue
        for row in raw_dataset[split_name]:
            cleaned = clean_record(row, cfg)
            if cleaned is None:
                continue
            if not validate_record(cleaned):
                continue
            if fmt == "alpaca":
                formatted = format_alpaca(cleaned)
            else:
                formatted = format_chat_template(cleaned)
            records.append(formatted)
    print(f"[data] {len(records)} valid records after cleaning.")
    return records


def save_splits(records: list[dict], output_dir: Path, seed: int = 42):
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = split_dataset(records, seed=seed)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for rec in split:
                f.write(json.dumps(rec) + "\n")
        print(f"[data] Saved {len(split)} records → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["medqa", "pubmedqa"], default="medqa")
    parser.add_argument("--output", default="data/processed/")
    parser.add_argument("--format", choices=["alpaca", "chat"], default="alpaca")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw, cfg = load_raw_dataset(args.dataset)
    records = build_instruction_dataset(raw, cfg, fmt=args.format)
    save_splits(records, Path(args.output) / args.dataset, seed=args.seed)
    print("[data] Done.")


if __name__ == "__main__":
    main()
