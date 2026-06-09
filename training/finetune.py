"""
QLoRA fine-tuning entry point using Unsloth for 2× training speed.

Usage:
    python training/finetune.py --config training/config.yaml
"""

import argparse
from pathlib import Path

import yaml

from training.trainer_utils import (
    build_trainer,
    load_model_and_tokenizer,
    load_train_val_datasets,
    save_adapter,
    setup_wandb,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    setup_wandb(cfg)

    print("[train] Loading model and tokenizer ...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    print("[train] Loading datasets ...")
    train_ds, val_ds = load_train_val_datasets(cfg, tokenizer)

    print(f"[train] Train: {len(train_ds)} | Val: {len(val_ds)}")

    trainer = build_trainer(model, tokenizer, train_ds, val_ds, cfg)

    print("[train] Starting fine-tuning ...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    output_dir = Path(cfg["training"]["output_dir"])
    save_adapter(model, tokenizer, output_dir)
    print(f"[train] Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
