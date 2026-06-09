"""
Quantization pipeline: FP16 → INT8 (BitsAndBytes) or INT4 (AWQ).

Usage:
    python quantization/quantize.py --model outputs/finetuned --bits 8
    python quantization/quantize.py --model outputs/finetuned --bits 4 --awq_calib_data data/processed/medqa/val.jsonl
"""

import argparse
from pathlib import Path

from quantization.quant_utils import (
    export_awq_int4,
    export_bnb_int8,
    verify_quantized_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to fine-tuned model dir")
    parser.add_argument("--bits", type=int, choices=[8, 4], required=True)
    parser.add_argument("--output", default=None, help="Output dir (auto-named if omitted)")
    parser.add_argument(
        "--awq_calib_data",
        default="data/processed/medqa/val.jsonl",
        help="Calibration data for AWQ INT4",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if args.output:
        out_path = Path(args.output)
    else:
        suffix = "int8" if args.bits == 8 else "int4_awq"
        out_path = model_path.parent / f"{model_path.name}_{suffix}"

    if args.bits == 8:
        print(f"[quant] Quantizing to INT8 (BitsAndBytes) → {out_path}")
        export_bnb_int8(str(model_path), str(out_path))
    else:
        print(f"[quant] Quantizing to INT4 (AWQ) → {out_path}")
        export_awq_int4(str(model_path), str(out_path), args.awq_calib_data)

    verify_quantized_model(str(out_path), args.bits)
    print("[quant] Done.")


if __name__ == "__main__":
    main()
