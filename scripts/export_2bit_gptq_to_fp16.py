import argparse
import os

from gptqmodel import GPTQModel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a GPTQ-quantized model (e.g., 2-bit) to a dequantized HF fp16 checkpoint."
    )
    parser.add_argument(
        "--in_quant_dir",
        type=str,
        default=os.environ.get("IN_QUANT_DIR", "/root/autodl-tmp/models/source/Llama-2-7b-EfficientQAT-w2g64-GPTQ"),
        help="Path (local) or repo id (HF) of the quantized GPTQ model.",
    )
    parser.add_argument(
        "--out_fp16_dir",
        type=str,
        default=os.environ.get("OUT_FP16_DIR", "/root/autodl-tmp/models/output/Llama-2-7b-fp16-from-2bit"),
        help="Output directory for the dequantized HF checkpoint.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code when loading configs/models.",
    )

    args = parser.parse_args()

    # NOTE: GPTQModel.export(format="hf") internally loads the quantized model with backend=TORCH
    # and uses TorchQuantLinear dequantization (weights saved as fp16 on CPU).
    GPTQModel.export(
        model_id_or_path=args.in_quant_dir,
        target_path=args.out_fp16_dir,
        format="hf",
        trust_remote_code=args.trust_remote_code,
    )

    print(f"exported_fp16: {args.out_fp16_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
