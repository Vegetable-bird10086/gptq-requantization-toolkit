import argparse
import os
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_calibration_dataset_from_text(
    tokenizer,
    text: str,
    seq_len: int,
    num_samples: int,
) -> List[Dict[str, Any]]:
    enc = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    ids: List[int] = enc["input_ids"]

    if len(ids) < max(2, seq_len):
        raise ValueError(f"Not enough tokens in calibration text: got {len(ids)} tokens")

    blocks: List[Dict[str, Any]] = []
    step = seq_len
    for start in range(0, len(ids) - seq_len + 1, step):
        chunk = ids[start : start + seq_len]
        input_ids = torch.tensor([chunk], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        blocks.append({"input_ids": input_ids, "attention_mask": attention_mask})
        if num_samples is not None and num_samples > 0 and len(blocks) >= num_samples:
            break

    if len(blocks) == 0:
        raise ValueError("Calibration dataset ended up empty.")

    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize a fp16 HF checkpoint into a 4-bit GPTQ model using gptqmodel.")
    parser.add_argument(
        "--fp16_model_dir",
        type=str,
        default=os.environ.get("FP16_MODEL_DIR", "/root/autodl-tmp/models/output/Llama-2-7b-fp16-from-2bit"),
        help="Local path (or repo id) of the *unquantized* (fp16) HF checkpoint.",
    )
    parser.add_argument(
        "--out_quant_dir",
        type=str,
        default=os.environ.get("OUT_QUANT_DIR", "/root/autodl-tmp/models/output/Llama-2-7b-GPTQ-4bit"),
        help="Output directory for the 4-bit GPTQ model.",
    )
    parser.add_argument(
        "--calib_text_file",
        type=str,
        default=os.environ.get("CALIB_TEXT_FILE", None),
        required=False,
        help="Plain-text file used for calibration (recommended: WikiText train/valid/test raw text).",
    )
    parser.add_argument("--calib_seq_len", type=int, default=2048, help="Sequence length for calibration blocks")
    parser.add_argument(
        "--calib_num_samples",
        type=int,
        default=128,
        help="Number of calibration blocks. Increase for better quality; costs more time/VRAM.",
    )

    parser.add_argument("--bits", type=int, default=4, help="GPTQ bits (use 4 for 4-bit)")
    parser.add_argument("--group_size", type=int, default=128, help="GPTQ group size (commonly 64 or 128)")
    parser.add_argument("--desc_act", action="store_true", help="Enable desc_act (often improves quality)")
    parser.add_argument("--no_desc_act", action="store_true", help="Disable desc_act")
    parser.add_argument("--sym", action="store_true", help="Symmetric quantization")
    parser.add_argument("--no_sym", action="store_true", help="Asymmetric quantization")
    parser.add_argument("--true_sequential", action="store_true", help="Enable true_sequential")
    parser.add_argument("--no_true_sequential", action="store_true", help="Disable true_sequential")

    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device used for quantization forward passes (usually cuda:0).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=os.environ.get("GPTQ_BACKEND", "auto"),
        help="gptqmodel backend for quantization (e.g. auto, torch, triton).",
    )

    parser.add_argument("--batch_size", type=int, default=1, help="Calibration batch size (keep 1 for safety)")
    parser.add_argument("--buffered_fwd", action="store_true", help="Buffer forward inputs to CPU (lower VRAM, slower)")
    parser.add_argument("--disable_gpu_cache", action="store_true", help="Disable calibration GPU cache")
    parser.add_argument("--no_auto_gc", action="store_true", help="Disable auto torch/cuda GC")
    parser.add_argument("--trust_remote_code", action="store_true")

    args = parser.parse_args()

    if args.calib_text_file is None:
        raise SystemExit("Missing --calib_text_file (need some calibration text).")

    tokenizer = AutoTokenizer.from_pretrained(args.fp16_model_dir, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    text = _read_text(args.calib_text_file)
    calib_dataset = build_calibration_dataset_from_text(
        tokenizer=tokenizer,
        text=text,
        seq_len=args.calib_seq_len,
        num_samples=args.calib_num_samples,
    )

    desc_act = True if args.desc_act else False if args.no_desc_act else True
    sym = True if args.sym else False if args.no_sym else True
    true_sequential = True if args.true_sequential else False if args.no_true_sequential else True

    qcfg = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=desc_act,
        sym=sym,
        true_sequential=true_sequential,
        device=args.device,
    )

    model = GPTQModel.load(
        model_id_or_path=args.fp16_model_dir,
        quantize_config=qcfg,
        backend=args.backend,
        trust_remote_code=args.trust_remote_code,
    )

    model.quantize(
        calibration_dataset=calib_dataset,
        batch_size=args.batch_size,
        calibration_enable_gpu_cache=not args.disable_gpu_cache,
        buffered_fwd=args.buffered_fwd,
        auto_gc=not args.no_auto_gc,
    )

    model.save_quantized(args.out_quant_dir)
    print(f"saved_4bit_gptq: {args.out_quant_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
