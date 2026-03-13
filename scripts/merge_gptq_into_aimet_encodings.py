#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class GptqParam:
    module_name: str
    bitwidth: int
    symmetric: bool
    scale: list[float]
    offset: list[int]


def unpack_qzeros(
    packed_qzeros: torch.Tensor,
    bits: int,
    out_features: int,
    pack_dtype_bits: int,
    qzero_format: int,
) -> torch.Tensor:
    packed_qzeros = packed_qzeros.to(torch.int64)
    if qzero_format == 1:
        add_offset = sum(1 << (bits * i) for i in range(pack_dtype_bits // bits))
        packed_qzeros = packed_qzeros + add_offset
    elif qzero_format != 2:
        raise ValueError(f"Unsupported qzero_format: {qzero_format}")

    pack_factor = pack_dtype_bits // bits
    shifts = torch.arange(0, pack_dtype_bits, bits, dtype=torch.int64, device=packed_qzeros.device)
    unpacked = (packed_qzeros.unsqueeze(-1) >> shifts.view(1, 1, pack_factor)) & ((1 << bits) - 1)
    zeros = unpacked.reshape(packed_qzeros.shape[0], packed_qzeros.shape[1] * pack_factor)[:, :out_features]
    return zeros.to(torch.int32).contiguous()


def calc_min_max(scale: list[float], offset: list[int], bitwidth: int) -> tuple[list[float], list[float]]:
    qmax = (2**bitwidth) - 1
    mins = [float((0 - o) * s) for s, o in zip(scale, offset, strict=False)]
    maxs = [float((qmax - o) * s) for s, o in zip(scale, offset, strict=False)]
    return mins, maxs


def extract_gptq_params(
    gptq_dir: str,
    device: str,
    attn_implementation: str,
    trust_remote_code: bool,
) -> list[GptqParam]:
    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from gptqmodel.utils.backend import BACKEND

    gptq_model = GPTQModel.load(
        model_id_or_path=gptq_dir,
        backend=BACKEND.TORCH,
        device=device,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )

    global_sym = False
    quant_cfg = getattr(gptq_model, "quantize_config", None)
    if quant_cfg is None:
        quant_cfg = getattr(getattr(gptq_model, "model", None), "quantize_config", None)
    if quant_cfg is not None:
        if hasattr(quant_cfg, "sym"):
            global_sym = bool(getattr(quant_cfg, "sym"))
        elif isinstance(quant_cfg, dict):
            global_sym = bool(quant_cfg.get("sym", False))

    out: list[GptqParam] = []
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

        scale = [float(v) for v in scale_vec.tolist()]
        offset = [int(v) for v in zero_vec.tolist()[: len(scale)]]

        out.append(
            GptqParam(
                module_name=module_name,
                bitwidth=bits,
                symmetric=symmetric,
                scale=scale,
                offset=offset,
            )
        )
    return out


def extract_gptq_params_from_encodings(gptq_encodings_path: str) -> list[GptqParam]:
    with open(gptq_encodings_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    raw = d.get("param_encodings", [])
    out: list[GptqParam] = []
    if isinstance(raw, list):
        for entry in raw:
            name = entry.get("name")
            if not isinstance(name, str):
                continue
            scale = entry.get("scale", [])
            offset = entry.get("offset", [])
            bitwidth = int(entry.get("bitwidth", 4))
            symmetric = bool(entry.get("is_symmetric", False))
            if not isinstance(scale, list) or not isinstance(offset, list):
                continue
            out.append(
                GptqParam(
                    module_name=name,
                    bitwidth=bitwidth,
                    symmetric=symmetric,
                    scale=[float(x) for x in scale],
                    offset=[int(x) for x in offset],
                )
            )
    elif isinstance(raw, dict):
        for name, entry in raw.items():
            scale = entry.get("scale", [])
            offset = entry.get("offset", [])
            bitwidth = int(entry.get("bitwidth", 4))
            symmetric = str(entry.get("is_symmetric", "False")).lower() == "true"
            if not isinstance(scale, list) or not isinstance(offset, list):
                continue
            out.append(
                GptqParam(
                    module_name=name,
                    bitwidth=bitwidth,
                    symmetric=symmetric,
                    scale=[float(x) for x in scale],
                    offset=[int(x) for x in offset],
                )
            )
    return out


def candidate_names(module_name: str) -> list[str]:
    base = module_name
    slash = module_name.replace(".", "/")
    return [
        base,
        f"{base}.weight",
        f"{base}.qweight",
        f"{base}.scales",
        slash,
        f"{slash}.weight",
        f"{slash}.qweight",
        f"{slash}.scales",
    ]


def match_name(module_name: str, available: set[str]) -> str | None:
    cands = candidate_names(module_name)
    for c in cands:
        if c in available:
            return c

    # conservative suffix fallback (full candidate only)
    for c in cands:
        for name in available:
            if name.endswith(c):
                return name
    return None


def _sha_key_index(name: str) -> int:
    m = re.search(r"_sha\.(\d+)\.weight$", name)
    return int(m.group(1)) if m else -1


def structural_target_names(module_name: str, available: set[str]) -> list[str]:
    m = re.match(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(.+)$", module_name)
    if not m:
        return []

    layer_idx = m.group(1)
    block = m.group(2)
    proj = m.group(3)
    prefix = f"model.model.model.layers.{layer_idx}."

    if block == "self_attn" and proj in {"q_proj", "k_proj", "v_proj"}:
        base = f"{prefix}self_attn.{proj}_sha."
        keys = [k for k in available if k.startswith(base) and k.endswith(".weight")]
        return sorted(keys, key=_sha_key_index)

    if block == "self_attn" and proj == "o_proj":
        key = f"{prefix}self_attn.o_proj_conv.weight"
        return [key] if key in available else []

    if block == "mlp" and proj in {"gate_proj", "up_proj", "down_proj"}:
        key = f"{prefix}mlp.{proj}_conv.weight"
        return [key] if key in available else []

    return []


def remap_module_name_for_layer_offset(module_name: str, layer_offset: int) -> str | None:
    if layer_offset == 0:
        return module_name

    m = re.match(r"^model\.layers\.(\d+)\.(.+)$", module_name)
    if not m:
        return module_name

    global_idx = int(m.group(1))
    local_idx = global_idx - layer_offset
    if local_idx < 0:
        return None
    return f"model.layers.{local_idx}.{m.group(2)}"


def update_v1_param_entry(entry: dict[str, Any], src: GptqParam) -> dict[str, Any]:
    mins, maxs = calc_min_max(src.scale, src.offset, src.bitwidth)
    new_entry = dict(entry)
    new_entry["bitwidth"] = src.bitwidth
    new_entry["is_symmetric"] = src.symmetric
    new_entry["scale"] = src.scale
    new_entry["offset"] = src.offset
    new_entry["min"] = mins
    new_entry["max"] = maxs
    new_entry["enc_type"] = "PER_CHANNEL" if len(src.scale) > 1 else "PER_TENSOR"
    return new_entry


def update_legacy_param_entry(entry: Any, src: GptqParam) -> Any:
    mins, maxs = calc_min_max(src.scale, src.offset, src.bitwidth)

    if isinstance(entry, list):
        if len(entry) > 0 and isinstance(entry[0], dict):
            template = dict(entry[0])
        else:
            template = {"dtype": "int"}

        n = min(len(src.scale), len(src.offset), len(mins), len(maxs))
        out: list[dict[str, Any]] = []
        for i in range(n):
            cur = dict(template)
            cur["bitwidth"] = src.bitwidth
            cur["is_symmetric"] = str(bool(src.symmetric))
            cur["scale"] = float(src.scale[i])
            cur["offset"] = int(src.offset[i])
            cur["min"] = float(mins[i])
            cur["max"] = float(maxs[i])
            out.append(cur)
        return out

    if isinstance(entry, dict):
        new_entry = dict(entry)
        new_entry["bitwidth"] = src.bitwidth
        new_entry["is_symmetric"] = str(bool(src.symmetric))
        new_entry["scale"] = float(src.scale[0]) if src.scale else 0.0
        new_entry["offset"] = int(src.offset[0]) if src.offset else 0
        new_entry["min"] = float(mins[0]) if mins else 0.0
        new_entry["max"] = float(maxs[0]) if maxs else 0.0
        return new_entry

    raise RuntimeError(f"Unsupported legacy param entry type: {type(entry)}")


def _entry_list_len(entry: Any) -> int:
    if isinstance(entry, list):
        return len(entry)
    if isinstance(entry, dict):
        return 1
    return 0


def _slice_param(src: GptqParam, start: int, end: int) -> GptqParam:
    return GptqParam(
        module_name=src.module_name,
        bitwidth=src.bitwidth,
        symmetric=src.symmetric,
        scale=src.scale[start:end],
        offset=src.offset[start:end],
    )


def _split_src_for_targets(src: GptqParam, entries: dict[str, Any], targets: list[str]) -> tuple[list[GptqParam], str | None]:
    target_lens = [_entry_list_len(entries[t]) for t in targets]
    total_need = sum(target_lens)
    total_have = min(len(src.scale), len(src.offset))
    if total_have != total_need:
        reason = (
            f"length_mismatch: src_len={total_have}, target_total={total_need}, "
            f"targets={len(targets)}"
        )
        return [], reason

    out: list[GptqParam] = []
    pos = 0
    for ln in target_lens:
        out.append(_slice_param(src, pos, pos + ln))
        pos += ln
    return out, None


def merge_encodings(
    encodings_path: Path,
    gptq_params: list[GptqParam],
    mapping_mode: str = "exact",
    layer_offset: int = 0,
) -> dict[str, Any]:
    with open(encodings_path, "r", encoding="utf-8") as f:
        enc = json.load(f)

    updated = 0
    updated_target_keys = 0
    unmatched: list[str] = []
    mapping_failures: list[dict[str, Any]] = []
    module_to_targets: dict[str, list[str]] = {}

    if isinstance(enc.get("param_encodings"), list):
        entries = enc["param_encodings"]
        name_to_idx = {x.get("name"): i for i, x in enumerate(entries)}
        available = {k for k in name_to_idx.keys() if isinstance(k, str)}

        for p in gptq_params:
            m = match_name(p.module_name, available)
            if m is None:
                unmatched.append(p.module_name)
                continue
            idx = name_to_idx[m]
            entries[idx] = update_v1_param_entry(entries[idx], p)
            updated += 1
            updated_target_keys += 1
            module_to_targets[p.module_name] = [m]
    elif isinstance(enc.get("param_encodings"), dict):
        entries = enc["param_encodings"]
        available = set(entries.keys())

        for p in gptq_params:
            mapped_module_name = remap_module_name_for_layer_offset(p.module_name, layer_offset)
            if mapped_module_name is None:
                unmatched.append(p.module_name)
                continue

            targets: list[str] = []

            if mapping_mode in {"exact", "auto"}:
                m = match_name(mapped_module_name, available)
                if m is not None:
                    targets = [m]

            if not targets and mapping_mode in {"structural", "auto"}:
                targets = structural_target_names(mapped_module_name, available)

            if not targets:
                unmatched.append(p.module_name)
                continue

            parts, split_err = _split_src_for_targets(p, entries, targets)
            if split_err is not None:
                mapping_failures.append(
                    {
                        "module": p.module_name,
                        "targets": targets,
                        "reason": split_err,
                    }
                )
                continue

            for t, src_part in zip(targets, parts, strict=False):
                entries[t] = update_legacy_param_entry(entries[t], src_part)

            updated += 1
            updated_target_keys += len(targets)
            module_to_targets[p.module_name] = targets
    else:
        raise RuntimeError("Unsupported model.encodings format: param_encodings is neither list nor dict.")

    enc["_gptq_merge_report"] = {
        "mapping_mode": mapping_mode,
        "layer_offset": layer_offset,
        "gptq_modules": len(gptq_params),
        "updated_entries": updated,
        "updated_target_keys": updated_target_keys,
        "unmatched_modules": unmatched,
        "mapping_failures": mapping_failures,
        "module_to_targets": module_to_targets,
    }
    return enc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge GPTQ weight quantization parameters into an existing AIMET model.encodings. "
            "Activation encodings from the AIMET checkpoint are preserved."
        )
    )
    parser.add_argument("--gptq-dir", required=True, type=str)
    parser.add_argument(
        "--gptq-encodings",
        type=str,
        default=None,
        help="If set, read GPTQ param encodings from this file instead of loading gptqmodel.",
    )
    parser.add_argument("--aimet-checkpoint", required=True, type=str)
    parser.add_argument("--out-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail if any GPTQ module cannot be matched.")
    parser.add_argument(
        "--mapping-mode",
        type=str,
        default="auto",
        choices=["exact", "structural", "auto"],
        help="How to map GPTQ module names to AIMET param_encodings keys.",
    )
    parser.add_argument(
        "--layer-offset",
        type=int,
        default=0,
        help="Subtract this offset from GPTQ layer index before matching (for shard-local layer numbering).",
    )
    args = parser.parse_args()

    src_ckpt = Path(args.aimet_checkpoint)
    if not src_ckpt.is_dir():
        raise RuntimeError(f"AIMET checkpoint directory not found: {src_ckpt}")
    src_enc = src_ckpt / "model.encodings"
    if not src_enc.exists():
        raise RuntimeError(f"model.encodings not found in {src_ckpt}")

    dst_ckpt = Path(args.out_checkpoint) if args.out_checkpoint else src_ckpt.with_name(src_ckpt.name + "_gptq_merged")
    if dst_ckpt.resolve() != src_ckpt.resolve():
        if dst_ckpt.exists():
            shutil.rmtree(dst_ckpt)
        shutil.copytree(src_ckpt, dst_ckpt)

    if args.gptq_encodings:
        gptq_params = extract_gptq_params_from_encodings(args.gptq_encodings)
    else:
        gptq_params = extract_gptq_params(
            gptq_dir=args.gptq_dir,
            device=args.device,
            attn_implementation=args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
        )

    enc_path = dst_ckpt / "model.encodings"
    merged = merge_encodings(
        enc_path,
        gptq_params,
        mapping_mode=args.mapping_mode,
        layer_offset=args.layer_offset,
    )

    report = merged.get("_gptq_merge_report", {})
    unmatched = report.get("unmatched_modules", [])
    if args.strict and unmatched:
        raise RuntimeError(f"Strict mode: unmatched GPTQ modules: {len(unmatched)}")

    with open(enc_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    report_path = dst_ckpt / "gptq_merge_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"gptq_modules: {report.get('gptq_modules', 0)}")
    print(f"updated_entries: {report.get('updated_entries', 0)}")
    print(f"updated_target_keys: {report.get('updated_target_keys', 0)}")
    print(f"mapping_failures: {len(report.get('mapping_failures', []))}")
    print(f"unmatched_modules: {len(unmatched)}")
    print(f"merged_encodings: {enc_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
