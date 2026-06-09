"""Utilities: model loading, dataset tokenization, W&B setup, adapter saving."""

import json
import os
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError:
    from transformers import AutoModelForCausalLM

    UNSLOTH_AVAILABLE = False
    print("[train] Unsloth not found — falling back to standard HuggingFace loading.")


def setup_wandb(cfg: dict):
    wandb_cfg = cfg.get("wandb", {})
    os.environ.setdefault("WANDB_PROJECT", wandb_cfg.get("project", "modeledge"))
    run_name = wandb_cfg.get("run_name", None)
    wandb.init(
        project=wandb_cfg.get("project", "modeledge"),
        name=run_name,
        config=cfg,
    )


def load_model_and_tokenizer(cfg: dict):
    model_cfg = cfg["model"]
    name = model_cfg["name"]
    max_seq = model_cfg["max_seq_length"]

    if UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=name,
            max_seq_length=max_seq,
            dtype=None,
            load_in_4bit=model_cfg.get("load_in_4bit", True),
        )

        lora_cfg = cfg["lora"]
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
            use_gradient_checkpointing="unsloth",
            random_state=cfg["training"]["seed"],
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        lora_cfg = cfg["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"],
        )
        model = get_peft_model(model, peft_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_train_val_datasets(cfg: dict, tokenizer):
    data_cfg = cfg["data"]
    train_records = _load_jsonl(data_cfg["train_file"])
    val_records = _load_jsonl(data_cfg["val_file"])
    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)
    return train_ds, val_ds


def tokenize_dataset(dataset: Dataset, tokenizer, cfg: dict) -> Dataset:
    """Tokenize text column into input_ids / labels for the training loop."""
    max_len = cfg["model"]["max_seq_length"]
    text_col = cfg["data"].get("text_column", "text")

    def _tokenize(batch):
        enc = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format("torch")
    return tokenized


def save_adapter(model, tokenizer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
