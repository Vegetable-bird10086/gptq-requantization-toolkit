import argparse
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.quantization.quantizer import Quantizer
from gptqmodel.utils.model import create_quant_layer, make_quant

from weight_only_quantize import (
    DEFAULT_QUANT_PARAM_CACHE,
    _create_progress,
    _fake_quantize_group,
    _read_text,
    _search_quant_params,
    build_calibration_dataset_from_text,
    collect_activation_channel_rms,
)


@torch.no_grad()
def _dequantize_quant_layer(module: BaseQuantLinear) -> torch.Tensor:
    if not hasattr(module, "dequantize_weight"):
        raise TypeError(
            f"Source quant layer {type(module)} does not expose dequantize_weight(). "
            "Please load the source model with backend=torch."
        )
    return module.dequantize_weight().T.contiguous().to(torch.float16)


def _build_linear_for_pack(module: BaseQuantLinear, weight: torch.Tensor) -> nn.Linear:
    linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
    linear.weight.data.copy_(weight)
    if module.bias is not None:
        linear.bias.data.copy_(module.bias.data)
    return linear


def _load_cache(cache_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not cache_path:
        return None
    return torch.load(cache_path, map_location="cpu")


def _source_qcfg_attr(model, module: BaseQuantLinear, name: str, default):
    if hasattr(module, name):
        value = getattr(module, name)
        if value is not None:
            return value
    qcfg = getattr(model, "quantize_config", None)
    if qcfg is not None and hasattr(qcfg, name):
        value = getattr(qcfg, name)
        if value is not None:
            return value
    return default


def _unpack_qweight(qweight: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=qweight.device)
    unpacked = (qweight.to(torch.int32).unsqueeze(1) >> shifts.view(1, pack_factor, 1)) & ((1 << bits) - 1)
    return unpacked.reshape(qweight.shape[0] * pack_factor, qweight.shape[1]).contiguous()


def _pack_qweight(int_weight: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    if int_weight.shape[0] % pack_factor != 0:
        raise ValueError(f"Invalid int_weight rows {int_weight.shape[0]} for {bits}-bit packing")
    reshaped = int_weight.to(torch.int32).reshape(int_weight.shape[0] // pack_factor, pack_factor, int_weight.shape[1])
    packed = torch.zeros((reshaped.shape[0], reshaped.shape[2]), dtype=torch.int32)
    for j in range(pack_factor):
        packed |= reshaped[:, j, :] << (bits * j)
    return packed.contiguous()


def _unpack_qzeros(qzeros: torch.Tensor, bits: int, out_features: int) -> torch.Tensor:
    pack_factor = 32 // bits
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=qzeros.device)
    unpacked = (qzeros.to(torch.int32).unsqueeze(-1) >> shifts.view(1, 1, pack_factor)) & ((1 << bits) - 1)
    return unpacked.reshape(qzeros.shape[0], qzeros.shape[1] * pack_factor)[:, :out_features].contiguous()


def _pack_qzeros(zeros: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    if zeros.shape[1] % pack_factor != 0:
        raise ValueError(f"Invalid zero columns {zeros.shape[1]} for {bits}-bit packing")
    reshaped = zeros.to(torch.int32).reshape(zeros.shape[0], zeros.shape[1] // pack_factor, pack_factor)
    packed = torch.zeros((reshaped.shape[0], reshaped.shape[1]), dtype=torch.int32)
    for j in range(pack_factor):
        packed |= reshaped[:, :, j] << (bits * j)
    return packed.contiguous()


def _direct_code_lift(orig: BaseQuantLinear, target_bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    source_bits = int(orig.bits)
    src_maxq = (1 << source_bits) - 1
    dst_maxq = (1 << target_bits) - 1
    if dst_maxq % src_maxq != 0:
        raise ValueError(f"Cannot exactly lift {source_bits}-bit codes into {target_bits}-bit codes")

    factor = dst_maxq // src_maxq
    qweight_int = _unpack_qweight(orig.qweight.detach().cpu(), source_bits)
    qzeros_int = _unpack_qzeros(orig.qzeros.detach().cpu(), source_bits, int(orig.out_features))
    scales = orig.scales.detach().cpu().to(torch.float32) / factor
    zeros = (qzeros_int * factor).to(torch.int32)
    g_idx = orig.g_idx.detach().cpu().to(torch.int32)
    qweight = (qweight_int * factor).to(torch.int32)
    return qweight, scales, zeros, g_idx, factor


def _direct_repack(orig: BaseQuantLinear, target_bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_bits = int(orig.bits)
    if target_bits < source_bits:
        raise ValueError(f"Cannot repack {source_bits}-bit codes into narrower {target_bits}-bit container")

    qweight = _unpack_qweight(orig.qweight.detach().cpu(), source_bits).to(torch.int32)
    zeros = _unpack_qzeros(orig.qzeros.detach().cpu(), source_bits, int(orig.out_features)).to(torch.int32)
    scales = orig.scales.detach().cpu().to(torch.float32)
    g_idx = orig.g_idx.detach().cpu().to(torch.int32)
    maxq = (1 << target_bits) - 1
    if torch.any(qweight < 0) or torch.any(qweight > maxq):
        raise ValueError(f"Source qweight codes exceed target {target_bits}-bit range")
    if torch.any(zeros < 0) or torch.any(zeros > maxq):
        raise ValueError(f"Source qzeros codes exceed target {target_bits}-bit range")
    return qweight, scales, zeros, g_idx


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct GPTQ-to-GPTQ re-quantization without exporting an fp16 checkpoint.")
    parser.add_argument("--in_quant_dir", type=str, required=True, help="Input GPTQ model directory.")
    parser.add_argument("--out_quant_dir", type=str, required=True, help="Output GPTQ model directory.")
    parser.add_argument(
        "--requant_from_cache",
        type=str,
        default=None,
        help="Optional path to an existing quant_params.pt. When provided, repack directly from this cache.",
    )
    parser.add_argument(
        "--quant_param_cache",
        type=str,
        default=None,
        help="Optional output path to save searched quantization parameters. Defaults to <out_quant_dir>/quant_params.pt in search mode.",
    )
    parser.add_argument("--direct_repack", action="store_true", help="Keep source integer codes and scales unchanged, and only repack them into a wider target-bit container.")
    parser.add_argument("--direct_code_lift", action="store_true", help="Exactly lift source integer codes/scale/zero-point into a higher-bit GPTQ representation without re-searching parameters.")
    parser.add_argument("--bits", type=int, default=4, help="Target bits.")
    parser.add_argument("--group_size", type=int, default=64, help="Target group size.")
    parser.add_argument("--per_channel", action="store_true", help="Use per-channel quantization. Equivalent to --group_size -1.")
    parser.add_argument("--desc_act", action="store_true")
    parser.set_defaults(sym=False)
    quant_mode_group = parser.add_mutually_exclusive_group()
    quant_mode_group.add_argument("--sym", dest="sym", action="store_true", help="Use symmetric quantization.")
    quant_mode_group.add_argument("--asym", dest="sym", action="store_false", help="Use asymmetric quantization.")
    parser.add_argument("--mse", type=float, default=2.0)
    parser.add_argument("--mse_grid", type=int, default=100)
    parser.add_argument("--maxshrink", type=float, default=0.8)
    parser.add_argument("--clip_ratio", type=float, default=1.0)
    parser.add_argument("--clip_search_grid", type=int, default=9)
    parser.add_argument("--refine_scale_grid", type=int, default=7)
    parser.add_argument("--refine_scale_range", type=float, default=0.15)
    parser.add_argument("--refine_zero_radius", type=int, default=4)
    parser.add_argument("--refine_rounds", type=int, default=2)
    parser.add_argument("--act_aware", action="store_true")
    parser.add_argument("--act_aware_alpha", type=float, default=1.0)
    parser.add_argument("--quantize_lm_head", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--backend", type=str, default=os.environ.get("GPTQ_BACKEND", "auto"))
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--calib_text_file", type=str, default=None)
    parser.add_argument("--calib_seq_len", type=int, default=512)
    parser.add_argument("--calib_num_samples", type=int, default=1)
    args = parser.parse_args()

    if args.per_channel:
        args.group_size = -1

    cache = _load_cache(args.requant_from_cache)
    if args.direct_repack and args.direct_code_lift:
        raise ValueError("--direct_repack cannot be combined with --direct_code_lift")
    if args.direct_code_lift and cache is not None:
        raise ValueError("--direct_code_lift cannot be combined with --requant_from_cache")
    if args.direct_repack and cache is not None:
        raise ValueError("--direct_repack cannot be combined with --requant_from_cache")
    if cache is not None:
        meta = cache["metadata"]
        args.bits = int(meta["bits"])
        args.group_size = int(meta["group_size"])
        args.desc_act = bool(meta["desc_act"])
        args.sym = bool(meta["sym"])
        args.mse = float(meta["mse"])

    if args.act_aware and args.calib_text_file is None and cache is None and not args.direct_code_lift:
        raise ValueError("--act_aware requires --calib_text_file when not reusing a cache.")

    model = GPTQModel.load(
        model_id_or_path=args.in_quant_dir,
        backend="torch",
        trust_remote_code=args.trust_remote_code,
    )

    orig_modules: Dict[str, BaseQuantLinear] = {}
    target_names = []
    for name, mod in model.named_modules():
        if not args.quantize_lm_head and name.endswith("lm_head"):
            continue
        if isinstance(mod, BaseQuantLinear):
            orig_modules[name] = mod
            target_names.append(name)

    if len(target_names) == 0:
        raise RuntimeError("No GPTQ quantized linear modules found in source model")

    first_module = orig_modules[target_names[0]]
    if args.direct_repack or args.direct_code_lift:
        args.group_size = int(_source_qcfg_attr(model, first_module, "group_size", args.group_size))
        args.sym = bool(_source_qcfg_attr(model, first_module, "sym", args.sym))
        args.desc_act = bool(_source_qcfg_attr(model, first_module, "desc_act", args.desc_act))

    qcfg = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        sym=args.sym,
        device=args.device,
        mse=args.mse,
        mock_quantization=False,
    )

    activation_channel_rms: Dict[str, torch.Tensor] = {}
    if args.act_aware and cache is None and not args.direct_code_lift and not args.direct_repack:
        tokenizer = AutoTokenizer.from_pretrained(args.in_quant_dir, use_fast=True, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        text = _read_text(args.calib_text_file)
        calib_dataset = build_calibration_dataset_from_text(
            tokenizer=tokenizer,
            text=text,
            seq_len=args.calib_seq_len,
            num_samples=args.calib_num_samples,
        )
        activation_channel_rms = collect_activation_channel_rms(
            model=model,
            target_modules=orig_modules,
            calib_dataset=calib_dataset,
            show_progress=not args.no_progress,
        )
        print(f"activation_aware_layers: {len(activation_channel_rms)}", flush=True)

    quant_result_placeholder = {n: {} for n in target_names}
    quant_linear_cls = make_quant(
        model,
        quant_result=quant_result_placeholder,
        qcfg=qcfg,
        backend=args.backend,
        lm_head_name=None,
        pack=False,
    )
    if not hasattr(quant_linear_cls, "pack"):
        quant_linear_cls = TorchQuantLinear

    create_quant_layer(
        linear_cls=quant_linear_cls,
        bits=qcfg.bits,
        desc_act=qcfg.desc_act,
        dynamic=qcfg.dynamic,
        group_size=qcfg.group_size,
        module=model,
        quant_result=quant_result_placeholder,
        sym=qcfg.sym,
        device=qcfg.device,
        lm_head_name=None,
        pack_dtype=qcfg.pack_dtype,
        backend=args.backend,
        adapter=qcfg.adapter,
    )

    quant_param_cache: Dict[str, Any] = {
        "metadata": {
            "bits": qcfg.bits,
            "group_size": qcfg.group_size,
            "desc_act": qcfg.desc_act,
            "sym": qcfg.sym,
            "mse": qcfg.mse,
            "mse_grid": args.mse_grid,
            "maxshrink": args.maxshrink,
            "clip_ratio": args.clip_ratio,
            "clip_search_grid": args.clip_search_grid,
            "refine_scale_grid": args.refine_scale_grid,
            "refine_scale_range": args.refine_scale_range,
            "refine_zero_radius": args.refine_zero_radius,
            "refine_rounds": args.refine_rounds,
            "act_aware": args.act_aware,
            "act_aware_alpha": args.act_aware_alpha,
            "per_channel": args.per_channel,
            "quantize_lm_head": args.quantize_lm_head,
            "backend": str(args.backend),
            "in_quant_dir": args.in_quant_dir,
            "search_method": "direct_repack" if args.direct_repack else ("direct_code_lift" if args.direct_code_lift else ("direct_from_cache" if cache is not None else ("activation_aware_direct_gptq_to_gptq" if args.act_aware else "direct_gptq_to_gptq"))),
        },
        "layers": {},
    }

    dst_named_modules = dict(model.named_modules())
    layer_progress = None if args.no_progress else _create_progress(total=len(target_names), desc="Direct requantizing layers", leave=True)

    cache_layers = cache["layers"] if cache is not None else {}
    for layer_idx, name in enumerate(target_names, start=1):
        orig = orig_modules[name]
        quant_mod = dst_named_modules[name]
        out_features = int(orig.out_features)
        in_features = int(orig.in_features)
        factor = None

        if args.direct_repack:
            qweight, scales, zeros, g_idx = _direct_repack(orig, qcfg.bits)
            group_size = int(orig.group_size)
            quant_mod.qweight.data.copy_(
                _pack_qweight(qweight, qcfg.bits).to(device=quant_mod.qweight.device, dtype=quant_mod.qweight.dtype)
            )
            quant_mod.qzeros.data.copy_(
                _pack_qzeros(zeros, qcfg.bits).to(device=quant_mod.qzeros.device, dtype=quant_mod.qzeros.dtype)
            )
            quant_mod.scales.data.copy_(scales.to(device=quant_mod.scales.device, dtype=quant_mod.scales.dtype))
            quant_mod.g_idx.data.copy_(g_idx.to(device=quant_mod.g_idx.device, dtype=quant_mod.g_idx.dtype))
            if orig.bias is not None and quant_mod.bias is not None:
                quant_mod.bias.data.copy_(orig.bias.detach().to(device=quant_mod.bias.device, dtype=quant_mod.bias.dtype))
        elif args.direct_code_lift:
            qweight, scales, zeros, g_idx, factor = _direct_code_lift(orig, qcfg.bits)
            group_size = int(orig.group_size)
            quant_mod.qweight.data.copy_(qweight.to(device=quant_mod.qweight.device, dtype=quant_mod.qweight.dtype))
            quant_mod.qzeros.data.copy_(
                _pack_qzeros(zeros, qcfg.bits).to(device=quant_mod.qzeros.device, dtype=quant_mod.qzeros.dtype)
            )
            quant_mod.scales.data.copy_(scales.to(device=quant_mod.scales.device, dtype=quant_mod.scales.dtype))
            quant_mod.g_idx.data.copy_(g_idx.to(device=quant_mod.g_idx.device, dtype=quant_mod.g_idx.dtype))
            if orig.bias is not None and quant_mod.bias is not None:
                quant_mod.bias.data.copy_(orig.bias.detach().to(device=quant_mod.bias.device, dtype=quant_mod.bias.dtype))
        elif cache is not None:
            W = _dequantize_quant_layer(orig)
            if name not in cache_layers:
                raise KeyError(f"Layer {name} missing in cache file {args.requant_from_cache}")
            info = cache_layers[name]
            group_size = int(info["group_size"])
            scales = info["scales"].to(torch.float32)
            zeros = info["zeros"].to(torch.int32)
            g_idx = info["g_idx"].to(torch.int32)
            Wq = torch.empty_like(W)
            for g in range(scales.shape[0]):
                s = slice(g * group_size, min((g + 1) * group_size, in_features))
                Wq[:, s] = _fake_quantize_group(W[:, s], scales[g], zeros[g], bits=qcfg.bits)
        else:
            W = _dequantize_quant_layer(orig)
            layer_channel_weights = activation_channel_rms.get(name)
            group_size = qcfg.group_size if qcfg.group_size != -1 else in_features
            if in_features % group_size != 0:
                group_size = in_features
            num_groups = in_features // group_size
            scales = torch.zeros((num_groups, out_features), dtype=torch.float32)
            zeros = torch.zeros((num_groups, out_features), dtype=torch.int32)
            Wq = torch.empty_like(W)

            quantizer = Quantizer(qcfg=qcfg, shape=out_features, name=name)
            quantizer.configure(
                perchannel=True,
                grid=args.mse_grid,
                maxshrink=args.maxshrink,
                bits=qcfg.bits,
                sym=qcfg.sym,
            )

            group_progress = None
            if not args.no_progress:
                group_progress = _create_progress(total=num_groups, desc=f"Layer {layer_idx}/{len(target_names)}", leave=False)

            for g in range(num_groups):
                s = slice(g * group_size, (g + 1) * group_size)
                chunk = W[:, s]
                chunk_channel_weights = None
                if layer_channel_weights is not None:
                    chunk_channel_weights = layer_channel_weights[s].to(chunk.device, dtype=torch.float32)
                    chunk_channel_weights = chunk_channel_weights.pow(args.act_aware_alpha)
                    chunk_channel_weights = chunk_channel_weights / chunk_channel_weights.mean().clamp_min(1e-6)

                if args.clip_ratio < 1.0:
                    absmax = chunk.abs().amax(dim=1, keepdim=True)
                    clip_bound = absmax * args.clip_ratio
                    chunk_for_search = torch.clamp(chunk, min=-clip_bound, max=clip_bound)
                else:
                    chunk_for_search = chunk

                quantizer.find_params(chunk_for_search, weight=True)
                base_scale = quantizer.scale.squeeze(1).to(torch.float32)
                base_zero = quantizer.zero.squeeze(1).to(torch.float32)
                scale, zero, qchunk = _search_quant_params(
                    chunk=chunk,
                    search_chunk=chunk_for_search,
                    bits=qcfg.bits,
                    sym=qcfg.sym,
                    init_scale=base_scale,
                    init_zero=base_zero,
                    maxshrink=args.maxshrink,
                    clip_grid=args.clip_search_grid,
                    refine_scale_grid=args.refine_scale_grid,
                    refine_scale_range=args.refine_scale_range,
                    refine_zero_radius=args.refine_zero_radius,
                    refine_rounds=args.refine_rounds,
                    error_power=max(float(args.mse), 1.0),
                    channel_weights=chunk_channel_weights,
                )
                scales[g] = scale
                zeros[g] = zero
                Wq[:, s] = qchunk.to(W.dtype)
                if group_progress is not None:
                    group_progress.update(1)

            if group_progress is not None:
                group_progress.close()
            g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)

        if not args.direct_code_lift and not args.direct_repack:
            linear_for_pack = _build_linear_for_pack(orig, Wq)
            quant_mod.pack(
                linear=linear_for_pack,
                scales=scales.T.contiguous(),
                zeros=zeros.T.contiguous(),
                g_idx=g_idx,
            )

        quant_param_cache["layers"][name] = {
            "scales": scales.cpu().contiguous(),
            "zeros": zeros.cpu().contiguous(),
            "g_idx": g_idx.cpu().contiguous(),
            "group_size": int(group_size),
            "in_features": int(in_features),
            "out_features": int(out_features),
            "layer_type": orig.__class__.__name__,
            "act_aware": name in activation_channel_rms,
            "repack_mode": "container-only" if args.direct_repack else None,
            "code_lift_factor": factor if args.direct_code_lift else None,
        }
        if layer_progress is not None:
            layer_progress.update(1)

    if layer_progress is not None:
        layer_progress.close()

    model.quantize_config = qcfg
    model.quantized = True
    model.qlinear_kernel = quant_linear_cls
    model.load_quantized_model = False
    if not hasattr(model, "quant_log"):
        model.quant_log = []

    model.save_quantized(args.out_quant_dir)

    if cache is None:
        quant_param_cache_path = args.quant_param_cache or os.path.join(args.out_quant_dir, DEFAULT_QUANT_PARAM_CACHE)
        os.makedirs(os.path.dirname(quant_param_cache_path), exist_ok=True)
        torch.save(quant_param_cache, quant_param_cache_path)
        print(f"saved_quant_param_cache: {quant_param_cache_path}")
    else:
        print(f"loaded_quant_param_cache: {args.requant_from_cache}")

    print(f"saved_direct_requantized_model: {args.out_quant_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
