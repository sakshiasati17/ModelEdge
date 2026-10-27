"""
Merge a LoRA adapter into its full-precision base and save standalone FP16
weights. This is what vLLM serves: it cannot load a bare adapter directory.

Usage:
    python -m quantization.merge --model outputs/finetuned --output outputs/finetuned_merged
"""

import argparse
from pathlib import Path

from quantization.quant_utils import _load_base_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to fine-tuned adapter dir")
    parser.add_argument("--output", default=None, help="Output dir (default: <model>_merged)")
    args = parser.parse_args()

    model_path = Path(args.model)
    out = Path(args.output) if args.output else model_path.parent / f"{model_path.name}_merged"

    model, tokenizer = _load_base_model(str(model_path))
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[merge] FP16 merged model saved → {out}")


if __name__ == "__main__":
    main()
