"""
Quantization pipeline: FP16 → INT8 or INT4 (BitsAndBytes).

Usage:
    python quantization/quantize.py --model outputs/finetuned --bits 8
    python quantization/quantize.py --model outputs/finetuned --bits 4
"""

import argparse
from pathlib import Path

from quantization.quant_utils import (
    export_bnb_int4,
    export_bnb_int8,
    verify_quantized_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to fine-tuned model dir")
    parser.add_argument("--bits", type=int, choices=[8, 4], required=True)
    parser.add_argument("--output", default=None, help="Output dir (auto-named if omitted)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if args.output:
        out_path = Path(args.output)
    else:
        suffix = "int8" if args.bits == 8 else "int4"
        out_path = model_path.parent / f"{model_path.name}_{suffix}"

    if args.bits == 8:
        print(f"[quant] Quantizing to INT8 (BitsAndBytes) → {out_path}")
        export_bnb_int8(str(model_path), str(out_path))
    else:
        print(f"[quant] Quantizing to INT4 (BitsAndBytes) → {out_path}")
        export_bnb_int4(str(model_path), str(out_path))

    verify_quantized_model(str(out_path), args.bits)
    print("[quant] Done.")


if __name__ == "__main__":
    main()
