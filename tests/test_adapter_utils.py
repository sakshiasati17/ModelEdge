import json

import pytest

from benchmarking.adapter_utils import (
    BASE_MODEL_ENV,
    full_precision_name,
    is_peft_adapter,
    is_prequantized_name,
    resolve_base_model,
)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit", "unsloth/llama-3.2-3b-instruct"),
        ("unsloth/Llama-3.2-3B-Instruct-bnb-4bit", "unsloth/Llama-3.2-3B-Instruct"),
        ("unsloth/Llama-3.2-3B-Instruct", "unsloth/Llama-3.2-3B-Instruct"),
    ],
)
def test_full_precision_name(name, expected):
    assert full_precision_name(name) == expected
    assert is_prequantized_name(name) == (name != expected)


def _write_adapter(tmp_path, base):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    return str(tmp_path)


def test_resolve_base_model_strips_4bit_suffix(tmp_path, monkeypatch):
    monkeypatch.delenv(BASE_MODEL_ENV, raising=False)
    path = _write_adapter(tmp_path, "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit")
    assert is_peft_adapter(path)
    assert resolve_base_model(path) == "unsloth/llama-3.2-3b-instruct"


def test_resolve_base_model_env_override(tmp_path, monkeypatch):
    path = _write_adapter(tmp_path, "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit")
    monkeypatch.setenv(BASE_MODEL_ENV, "/models/llama-fp16")
    assert resolve_base_model(path) == "/models/llama-fp16"


def test_is_peft_adapter_false_for_plain_dir(tmp_path):
    assert not is_peft_adapter(str(tmp_path))
