"""Utilities: model loading, dataset prep, W&B setup, SFTTrainer construction."""

import json
import os
from pathlib import Path

# Unsloth MUST be imported before trl/transformers/peft so its patches apply.
# Without this, trl's SFTTrainer runs unpatched and fails on Unsloth's
# legacy tokenizer placeholder eos_token '<EOS_TOKEN>'.
try:
    import unsloth
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

if not UNSLOTH_AVAILABLE:
    print("[train] Unsloth not found — falling back to standard HuggingFace loading.")


def setup_wandb(cfg: dict):
    wandb_cfg = cfg.get("wandb", {})
    os.environ.setdefault("WANDB_PROJECT", wandb_cfg.get("project", "modeledge"))
    wandb.init(
        project=wandb_cfg.get("project", "modeledge"),
        name=wandb_cfg.get("run_name", None),
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
            name, torch_dtype=torch.float16, device_map="auto"
        )
        lora_cfg = cfg["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["lora_alpha"],
                lora_dropout=lora_cfg["lora_dropout"],
                target_modules=lora_cfg["target_modules"],
                bias=lora_cfg["bias"],
                task_type=lora_cfg["task_type"],
            ),
        )

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
    train_ds = Dataset.from_list(_load_jsonl(data_cfg["train_file"]))
    val_ds = Dataset.from_list(_load_jsonl(data_cfg["val_file"]))
    return train_ds, val_ds


def build_sft_trainer(model, tokenizer, train_ds, val_ds, cfg: dict) -> SFTTrainer:
    t = cfg["training"]

    # Use Unsloth's patched SFTTrainer API — Unsloth patches trl internally so
    # tokenizer=, dataset_text_field=, and max_seq_length= work regardless of
    # trl version, bypassing trl 5.x vocabulary validation that breaks on
    # Unsloth's legacy tokenizer placeholder eos_token '<EOS_TOKEN>'.
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field=cfg["data"].get("text_column", "text"),
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            output_dir=t["output_dir"],
            num_train_epochs=t["num_train_epochs"],
            per_device_train_batch_size=t["per_device_train_batch_size"],
            per_device_eval_batch_size=t["per_device_eval_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            learning_rate=t["learning_rate"],
            weight_decay=t["weight_decay"],
            warmup_steps=t.get("warmup_steps", int(0.03 * t.get("num_train_epochs", 3) * 305)),
            lr_scheduler_type=t["lr_scheduler_type"],
            optim=t["optim"],
            fp16=t["fp16"],
            bf16=t["bf16"],
            logging_steps=t["logging_steps"],
            eval_strategy="steps",
            eval_steps=t["eval_steps"],
            save_strategy="steps",
            save_steps=t["save_steps"],
            save_total_limit=t["save_total_limit"],
            load_best_model_at_end=t["load_best_model_at_end"],
            metric_for_best_model=t["metric_for_best_model"],
            report_to=t["report_to"],
            seed=t["seed"],
        ),
    )


def save_adapter(model, tokenizer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
