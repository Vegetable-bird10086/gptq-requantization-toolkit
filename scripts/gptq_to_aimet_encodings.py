#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from gptqmodel import GPTQModel
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.utils.backend import BACKEND

try:
    import onnx
except Exception:  # pragma: no cover
    onnx = None


def unpack_qzeros(
    packed_qzeros: torch.Tensor,
    bits: int,
    out_features: int,
    pack_dtype_bits: int,
    qzero_format: int,
) -> torch.Tensor:
    packed_qzeros = packed_qzeros.to(torch.int64)
    if qzero_format == 1:
        offset = sum(1 << (bits * i) for i in range(pack_dtype_bits // bits))
        packed_qzeros = packed_qzeros + offset
    elif qzero_format != 2:
        raise ValueError(f"Unsupported qzero_format: {qzero_format}")

    pack_factor = pack_dtype_bits // bits
    shifts = torch.arange(0, pack_dtype_bits, bits, dtype=torch.int64, device=packed_qzeros.device)
    unpacked = (packed_qzeros.unsqueeze(-1) >> shifts.view(1, 1, pack_factor)) & ((1 << bits) - 1)
    zeros = unpacked.reshape(packed_qzeros.shape[0], packed_qzeros.shape[1] * pack_factor)[:, :out_features]
    return zeros.to(torch.int32).contiguous()


def to_float(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]


def build_param_entry_v1(name: str, scale: list[float], offset: list[int], bitwidth: int, symmetric: bool) -> dict[str, Any]:
    max_vals: list[float] = []
    min_vals: list[float] = []
    for s, o in zip(scale, offset, strict=False):
        qmax = (2**bitwidth) - 1
        min_vals.append((0.0 - o) * s)
        max_vals.append((qmax - o) * s)

    return {
        "name": name,
        "dtype": "INT",
        "bitwidth": bitwidth,
        "is_symmetric": symmetric,
        "enc_type": "PER_CHANNEL" if len(scale) > 1 else "PER_TENSOR",
        "scale": scale,
        "offset": offset,
        "min": min_vals,
        "max": max_vals,
    }


def build_param_entry_legacy(scale: list[float], offset: list[int], bitwidth: int, symmetric: bool) -> dict[str, Any]:
    max_vals: list[float] = []
    min_vals: list[float] = []
    for s, o in zip(scale, offset, strict=False):
        qmax = (2**bitwidth) - 1
        min_vals.append((0.0 - o) * s)
        max_vals.append((qmax - o) * s)

    return {
        "bitwidth": bitwidth,
        "dtype": "int",
        "is_symmetric": str(bool(symmetric)),
        "scale": scale,
        "offset": offset,
        "min": min_vals,
        "max": max_vals,
    }


def get_onnx_initializer_names(onnx_path: str | None) -> set[str]:
    if not onnx_path:
        return set()
    if onnx is None:
        raise RuntimeError("onnx is not installed, but --onnx-path was provided.")
    model = onnx.load(onnx_path)
    return {x.name for x in model.graph.initializer}


def match_initializer_name(base_name: str, initializers: set[str]) -> str:
    if not initializers:
        return base_name

    candidates = [
        f"{base_name}.scales",
        f"{base_name}.qweight",
        f"{base_name}.qweight_int",
        base_name,
    ]
    for c in candidates:
        if c in initializers:
            return c

    for n in initializers:
        if base_name in n and ("scale" in n or "weight" in n):
            return n

    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AIMET-style model.encodings from GPTQ qlinear parameters.")
    parser.add_argument("--in-quant-dir", required=True, type=str)
    parser.add_argument("--out-encodings", required=True, type=str)
    parser.add_argument("--onnx-path", type=str, default=None, help="Optional ONNX path for initializer name alignment")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--add-activation-placeholders",
        action="store_true",
        help="Add minimal activation encodings for common LLM I/O names.",
    )
    args = parser.parse_args()

    onnx_initializers = get_onnx_initializer_names(args.onnx_path)

    gptq_model = GPTQModel.load(
        model_id_or_path=args.in_quant_dir,
        backend=BACKEND.TORCH,
        device=args.device,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )

    global_sym = False
    try:
        quant_cfg = getattr(gptq_model, "quantize_config", None)
        if quant_cfg is None:
            quant_cfg = getattr(getattr(gptq_model, "model", None), "quantize_config", None)
        if quant_cfg is not None:
            if hasattr(quant_cfg, "sym"):
                global_sym = bool(getattr(quant_cfg, "sym"))
            elif isinstance(quant_cfg, dict):
                global_sym = bool(quant_cfg.get("sym", False))
    except Exception:
        global_sym = False

    param_enc_v1: list[dict[str, Any]] = []
    param_enc_legacy: dict[str, Any] = {}

    count = 0
    for module_name, module in gptq_model.model.named_modules():
        if not isinstance(module, TorchQuantLinear):
            continue

        bits = int(module.bits)
        symmetric = bool(getattr(module, "sym", global_sym))
        out_features = int(module.out_features)
        pack_dtype_bits = int(module.pack_dtype_bits)
        qzero_format = int(module.qzero_format())

        scales = module.scales.detach().cpu().to(torch.float32)
        zeros = unpack_qzeros(
            module.qzeros.detach().cpu(),
            bits=bits,
            out_features=out_features,
            pack_dtype_bits=pack_dtype_bits,
            qzero_format=qzero_format,
        ).to(torch.int32)

        if scales.dim() == 2 and scales.shape[0] == 1:
            scale_vec = scales[0]
            zero_vec = zeros[0]
        else:
            scale_vec = scales.reshape(-1)
            zero_vec = zeros.reshape(-1)[: scale_vec.numel()]

        scale_list = to_float(scale_vec)
        zero_list = [int(v) for v in zero_vec.reshape(-1).tolist()[: len(scale_list)]]

        enc_name = match_initializer_name(module_name, onnx_initializers)

        param_enc_v1.append(
            build_param_entry_v1(
                name=enc_name,
                scale=scale_list,
                offset=zero_list,
                bitwidth=bits,
                symmetric=symmetric,
            )
        )
        param_enc_legacy[enc_name] = build_param_entry_legacy(
            scale=scale_list,
            offset=zero_list,
            bitwidth=bits,
            symmetric=symmetric,
        )
        count += 1

    activation_enc_v1: list[dict[str, Any]] = []
    activation_enc_legacy: dict[str, Any] = {}

    if args.add_activation_placeholders:
        placeholders = {
            "input_ids": (16, 1.0, 0),
            "attention_mask": (16, 0.125, 0),
            "position_ids_cos": (16, 0.0625, 0),
            "position_ids_sin": (16, 0.0625, 0),
            "logits": (16, 0.125, 0),
        }
        for name, (bitwidth, scale, offset) in placeholders.items():
            activation_enc_v1.append(
                {
                    "name": name,
                    "dtype": "INT",
                    "bitwidth": bitwidth,
                    "is_symmetric": False,
                    "enc_type": "PER_TENSOR",
                    "scale": [scale],
                    "offset": [offset],
                    "min": [(-offset) * scale],
                    "max": [((2**bitwidth) - 1 - offset) * scale],
                }
            )
            activation_enc_legacy[name] = {
                "bitwidth": bitwidth,
                "dtype": "int",
                "is_symmetric": "False",
                "scale": [scale],
                "offset": [offset],
                "min": [(-offset) * scale],
                "max": [((2**bitwidth) - 1 - offset) * scale],
            }

    v1 = {
        "version": "1.0.0",
        "activation_encodings": activation_enc_v1,
        "param_encodings": param_enc_v1,
    }
    legacy = {
        "version": "0.6.1",
        "activation_encodings": activation_enc_legacy,
        "param_encodings": param_enc_legacy,
    }

    out_path = Path(args.out_encodings)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(v1, f, indent=2)

    legacy_path = out_path.with_suffix(out_path.suffix + ".legacy.json")
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(legacy, f, indent=2)

    print(f"processed_qlinear_modules: {count}")
    print(f"wrote_encodings_v1: {out_path}")
    print(f"wrote_encodings_legacy: {legacy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
