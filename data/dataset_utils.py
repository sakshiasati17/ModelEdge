"""Formatting helpers for instruction-tuning datasets."""

import random
from typing import Any


ALPACA_TEMPLATE = (
    "Below is a medical question. Answer it accurately and concisely.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

SYSTEM_MSG = (
    "You are a knowledgeable medical assistant. "
    "Answer each question accurately, cite relevant facts, "
    "and clearly state when you are uncertain."
)


def format_alpaca(record: dict) -> dict:
    choices_text = ""
    if record["choices"]:
        labels = "ABCDE"
        choices_text = "\n".join(
            f"{labels[i]}. {c}" for i, c in enumerate(record["choices"])
        )

    instruction = record["question"]
    inp = choices_text
    output = record["answer"]

    return {
        "text": ALPACA_TEMPLATE.format(
            instruction=instruction, input=inp, output=output
        ),
        "instruction": instruction,
        "input": inp,
        "output": output,
    }


def format_chat_template(record: dict) -> dict:
    choices_text = ""
    if record["choices"]:
        labels = "ABCDE"
        choices_text = "\n".join(
            f"{labels[i]}. {c}" for i, c in enumerate(record["choices"])
        )

    user_content = record["question"]
    if choices_text:
        user_content += f"\n\nOptions:\n{choices_text}"

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": record["answer"]},
    ]

    return {
        "messages": messages,
        "instruction": record["question"],
        "input": choices_text,
        "output": record["answer"],
    }


def validate_record(record: dict) -> bool:
    if len(record["question"]) < 10:
        return False
    if len(record["answer"]) < 1:
        return False
    return True


def split_dataset(
    records: list[dict],
    train_frac: float = 0.85,
    val_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test
