"""
Dataset preparation pipeline.

Downloads MedQA or PubMedQA, cleans the records, and converts them to
Alpaca-style instruction format ready for fine-tuning.

Usage:
    python -m data.prepare_dataset --dataset medqa --output data/processed/
    python -m data.prepare_dataset --dataset pubmedqa --output data/processed/
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from data.dataset_utils import (
    format_alpaca,
    format_chat_template,
    split_dataset,
    validate_record,
)


def _normalize_medqa(row: dict) -> dict | None:
    question = row.get("question", "").strip()
    answer = row.get("answer", "")
    options = row.get("options", {})

    if not question or not answer:
        return None

    if isinstance(options, dict) and options:
        choices = [options[k] for k in sorted(options.keys())]
    elif isinstance(options, list):
        choices = [str(o) for o in options]
    else:
        choices = []

    return {"question": question, "choices": choices, "answer": str(answer).strip()}


def _normalize_pubmedqa(row: dict) -> dict | None:
    question = row.get("question", "").strip()
    decision = row.get("final_decision", "").strip()
    long_answer = row.get("long_answer", "").strip()

    if not question or not decision:
        return None

    answer = long_answer if long_answer else decision
    return {"question": question, "choices": ["yes", "no", "maybe"], "answer": answer}


# Datasets are loaded directly from parquet/arrow files on the Hub
# (no custom loading scripts — those are no longer supported by `datasets`).
DATASET_CONFIGS = {
    "medqa": {
        "hf_name": "GBaker/MedQA-USMLE-4-options",
        "hf_config": None,
        "normalize": _normalize_medqa,
    },
    "pubmedqa": {
        "hf_name": "qiaojin/PubMedQA",
        "hf_config": "pqa_labeled",
        "normalize": _normalize_pubmedqa,
    },
}


def load_raw_dataset(dataset_name: str):
    cfg = DATASET_CONFIGS[dataset_name]
    print(f"[data] Loading {cfg['hf_name']} ({cfg['hf_config']}) ...")
    if cfg["hf_config"]:
        raw = load_dataset(cfg["hf_name"], cfg["hf_config"])
    else:
        raw = load_dataset(cfg["hf_name"])
    return raw, cfg


def build_instruction_dataset(raw_dataset, cfg: dict, fmt: str = "alpaca") -> list[dict]:
    records = []
    normalize = cfg["normalize"]

    for split_name in raw_dataset.keys():
        for row in raw_dataset[split_name]:
            cleaned = normalize(row)
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
