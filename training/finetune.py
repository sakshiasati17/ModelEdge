"""
QLoRA fine-tuning entry point using Unsloth + HuggingFace Accelerate.

Accelerate wraps the model, optimizer, and dataloader so the exact same
script runs on 1 GPU or 2x T4 (Kaggle) without any code changes.

Single GPU:
    python training/finetune.py --config training/config.yaml

Multi-GPU (2x T4 on Kaggle):
    accelerate launch --num_processes 2 training/finetune.py --config training/config.yaml
"""

import argparse
import math
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup

from training.trainer_utils import (
    load_model_and_tokenizer,
    load_train_val_datasets,
    save_adapter,
    setup_wandb,
    tokenize_dataset,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloader(dataset, tokenizer, batch_size: int, shuffle: bool) -> DataLoader:
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        pin_memory=True,
    )


def train(cfg: dict, resume_from_checkpoint: str | None = None):
    accelerator = Accelerator(mixed_precision="fp16")

    if accelerator.is_main_process:
        setup_wandb(cfg)
        accelerator.print(f"[train] Accelerate: {accelerator.num_processes} process(es)")

    t = cfg["training"]

    accelerator.print("[train] Loading model and tokenizer ...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    accelerator.print("[train] Loading + tokenizing datasets ...")
    train_ds, val_ds = load_train_val_datasets(cfg, tokenizer)
    train_ds = tokenize_dataset(train_ds, tokenizer, cfg)
    val_ds = tokenize_dataset(val_ds, tokenizer, cfg)

    train_loader = build_dataloader(train_ds, tokenizer, t["per_device_train_batch_size"], shuffle=True)
    val_loader = build_dataloader(val_ds, tokenizer, t["per_device_eval_batch_size"], shuffle=False)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
    )

    total_steps = math.ceil(
        len(train_loader) / t["gradient_accumulation_steps"]
    ) * t["num_train_epochs"]
    warmup_steps = int(total_steps * t["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Accelerate wraps everything for multi-GPU ──────────────────────────
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    # ───────────────────────────────────────────────────────────────────────

    global_step = 0
    best_val_loss = float("inf")
    output_dir = Path(t["output_dir"])

    for epoch in range(t["num_train_epochs"]):
        model.train()
        accum_loss = 0.0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_loss += loss.detach().float()

            if accelerator.sync_gradients:
                global_step += 1
                accum_loss = 0.0

                if global_step % t["logging_steps"] == 0 and accelerator.is_main_process:
                    import wandb
                    wandb.log(
                        {
                            "train/loss": accum_loss / t["logging_steps"],
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/step": global_step,
                        }
                    )
                    accelerator.print(
                        f"[train] epoch={epoch+1} step={global_step} "
                        f"loss={accum_loss / t['logging_steps']:.4f}"
                    )

                if global_step % t["eval_steps"] == 0:
                    val_loss = _evaluate(model, val_loader, accelerator)
                    if accelerator.is_main_process:
                        import wandb
                        wandb.log({"eval/loss": val_loss, "train/step": global_step})
                        accelerator.print(f"[train] eval_loss={val_loss:.4f}")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_adapter(
                                accelerator.unwrap_model(model),
                                tokenizer,
                                output_dir / "best",
                            )

                if global_step % t["save_steps"] == 0 and accelerator.is_main_process:
                    save_adapter(
                        accelerator.unwrap_model(model),
                        tokenizer,
                        output_dir / f"step_{global_step}",
                    )

    if accelerator.is_main_process:
        save_adapter(accelerator.unwrap_model(model), tokenizer, output_dir)
        accelerator.print(f"[train] Final adapter saved → {output_dir}")

    accelerator.end_training()


def _evaluate(model, val_loader, accelerator: Accelerator) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(**batch)
            loss = accelerator.gather(outputs.loss).mean()
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
