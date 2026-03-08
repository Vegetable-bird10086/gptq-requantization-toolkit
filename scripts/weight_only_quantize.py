import argparse
import os
import time
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.quantizer import Quantizer
from gptqmodel.utils.model import make_quant
import torch.nn as nn
import transformers

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


DEFAULT_QUANT_PARAM_CACHE = "quant_params.pt"


class _SimpleProgress:
    def __init__(self, total: int, desc: str, leave: bool = True):
        self.total = max(int(total), 0)
        self.desc = desc
        self.leave = leave
        self.n = 0
        self._last_print = 0.0
        if self.total > 0:
            print(f"[{self.desc}] 0/{self.total}", flush=True)

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.time()
        should_print = self.n >= self.total or (now - self._last_print) >= 1.0
        if should_print and self.total > 0:
            print(f"[{self.desc}] {self.n}/{self.total}", flush=True)
            self._last_print = now

    def set_postfix_str(self, _: str) -> None:
        return

    def close(self) -> None:
        if self.leave and self.total > 0 and self.n < self.total:
            print(f"[{self.desc}] {self.n}/{self.total}", flush=True)


def _create_progress(total: int, desc: str, leave: bool = True):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, leave=leave, dynamic_ncols=True)
    return _SimpleProgress(total=total, desc=desc, leave=leave)


def _fake_quantize_group(chunk: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: int) -> torch.Tensor:
    scale = scale.to(device=chunk.device, dtype=torch.float32).unsqueeze(1)
    zero = zero.to(device=chunk.device, dtype=torch.float32).unsqueeze(1)
    q = torch.clamp(torch.round(chunk.float() / scale) + zero, 0, maxq)
    return (scale * (q - zero)).to(chunk.dtype)


def _quant_error(original: torch.Tensor, quantized: torch.Tensor, power: float) -> torch.Tensor:
    diff = (quantized.float() - original.float()).abs()
    if power == 1.0:
        return diff.sum(dim=1)
    return diff.pow(power).sum(dim=1)


def _weighted_quant_error(
    original: torch.Tensor,
    quantized: torch.Tensor,
    power: float,
    channel_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    diff = (quantized.float() - original.float()).abs()
    if power != 1.0:
        diff = diff.pow(power)
    if channel_weights is not None:
        weights = channel_weights.to(device=diff.device, dtype=diff.dtype).view(1, -1)
        diff = diff * weights
    return diff.sum(dim=1)


def _maybe_update_best(
    original: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: int,
    power: float,
    channel_weights: Optional[torch.Tensor],
    best_err: torch.Tensor,
    best_scale: torch.Tensor,
    best_zero: torch.Tensor,
    best_q: torch.Tensor,
) -> None:
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    zero = torch.clamp(torch.round(zero), 0, maxq)
    q = _fake_quantize_group(original, scale, zero, maxq)
    err = _weighted_quant_error(original, q, power, channel_weights=channel_weights)
    improved = err < best_err
    if torch.any(improved):
        best_err[improved] = err[improved]
        best_scale[improved] = scale[improved]
        best_zero[improved] = zero[improved]
        best_q[improved] = q[improved]


def _search_quant_params(
    chunk: torch.Tensor,
    *,
    search_chunk: torch.Tensor,
    bits: int,
    sym: bool,
    init_scale: torch.Tensor,
    init_zero: torch.Tensor,
    maxshrink: float,
    clip_grid: int,
    refine_scale_grid: int,
    refine_scale_range: float,
    refine_zero_radius: int,
    refine_rounds: int,
    error_power: float,
    channel_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original = chunk.float()
    search_source = search_chunk.float()
    maxq = 2 ** bits - 1

    init_scale = init_scale.to(device=original.device, dtype=torch.float32)
    init_scale = torch.where(init_scale > 0, init_scale, torch.ones_like(init_scale))
    init_zero = init_zero.to(device=original.device, dtype=torch.float32)

    best_scale = init_scale.clone()
    best_zero = torch.clamp(torch.round(init_zero), 0, maxq)
    best_q = _fake_quantize_group(original, best_scale, best_zero, maxq)
    best_err = _weighted_quant_error(original, best_q, error_power, channel_weights=channel_weights)

    xmin = torch.minimum(search_source.amin(dim=1), torch.zeros(original.shape[0], device=original.device))
    xmax = torch.maximum(search_source.amax(dim=1), torch.zeros(original.shape[0], device=original.device))
    degenerate = (xmin == 0) & (xmax == 0)
    xmin = torch.where(degenerate, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(degenerate, torch.full_like(xmax, 1.0), xmax)

    if clip_grid > 1:
        shrink_factors = torch.linspace(
            max(min(maxshrink, 1.0), 1e-3),
            1.0,
            steps=clip_grid,
            device=original.device,
            dtype=torch.float32,
        )
        if sym:
            absmax = torch.maximum(xmax.abs(), xmin.abs())
            zero = torch.full_like(best_scale, (maxq + 1) / 2)
            for factor in shrink_factors:
                scale = (2.0 * absmax * factor) / maxq
                _maybe_update_best(
                    original,
                    scale=scale,
                    zero=zero,
                    maxq=maxq,
                    power=error_power,
                    channel_weights=channel_weights,
                    best_err=best_err,
                    best_scale=best_scale,
                    best_zero=best_zero,
                    best_q=best_q,
                )
        else:
            for lower_factor in shrink_factors:
                xmin1 = xmin * lower_factor
                for upper_factor in shrink_factors:
                    xmax1 = xmax * upper_factor
                    scale = (xmax1 - xmin1) / maxq
                    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                    zero = -xmin1 / scale
                    _maybe_update_best(
                        original,
                        scale=scale,
                        zero=zero,
                        maxq=maxq,
                        power=error_power,
                        channel_weights=channel_weights,
                        best_err=best_err,
                        best_scale=best_scale,
                        best_zero=best_zero,
                        best_q=best_q,
                    )

    if refine_scale_grid > 1 and refine_rounds > 0:
        scale_factors = torch.linspace(
            max(1.0 - refine_scale_range, 1e-3),
            1.0 + refine_scale_range,
            steps=refine_scale_grid,
            device=original.device,
            dtype=torch.float32,
        )
        zero_offsets = torch.arange(
            -refine_zero_radius,
            refine_zero_radius + 1,
            device=original.device,
            dtype=torch.float32,
        )
        for _ in range(refine_rounds):
            current_scale = best_scale.clone()
            current_zero = best_zero.clone()
            for scale_factor in scale_factors:
                scale = current_scale * scale_factor
                if sym:
                    zero = current_zero
                    _maybe_update_best(
                        original,
                        scale=scale,
                        zero=zero,
                        maxq=maxq,
                        power=error_power,
                        channel_weights=channel_weights,
                        best_err=best_err,
                        best_scale=best_scale,
                        best_zero=best_zero,
                        best_q=best_q,
                    )
                    continue

                for zero_offset in zero_offsets:
                    zero = current_zero + zero_offset
                    _maybe_update_best(
                        original,
                        scale=scale,
                        zero=zero,
                        maxq=maxq,
                        power=error_power,
                        channel_weights=channel_weights,
                        best_err=best_err,
                        best_scale=best_scale,
                        best_zero=best_zero,
                        best_q=best_q,
                    )

    return best_scale.to(torch.float32), best_zero.to(torch.int32), best_q.to(chunk.dtype)


def _get_model_device(module: torch.nn.Module) -> torch.device:
    param = next(module.parameters(), None)
    if param is not None:
        return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def collect_activation_channel_rms(
    model: torch.nn.Module,
    target_modules: Dict[str, torch.nn.Module],
    calib_dataset: List[Dict[str, Any]],
    show_progress: bool,
) -> Dict[str, torch.Tensor]:
    if len(calib_dataset) == 0:
        return {}

    device = _get_model_device(model)
    sumsqs: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    hooks = []

    def _hook_factory(module_name: str):
        def _hook(_module, inputs, _output):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            if x.numel() == 0:
                return
            x2d = x.detach().reshape(-1, x.shape[-1]).float()
            if module_name not in sumsqs:
                sumsqs[module_name] = torch.zeros(x2d.shape[-1], dtype=torch.float64, device=x2d.device)
                counts[module_name] = 0
            sumsqs[module_name].add_(x2d.pow(2).sum(dim=0, dtype=torch.float64))
            counts[module_name] += x2d.shape[0]

        return _hook

    for module_name, module_ref in target_modules.items():
        hooks.append(module_ref.register_forward_hook(_hook_factory(module_name)))

    model.eval()
    progress = None
    if show_progress:
        progress = _create_progress(total=len(calib_dataset), desc="Collecting activation stats", leave=True)

    try:
        for sample in calib_dataset:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in sample.items()
            }
            model(**batch, use_cache=False)
            if progress is not None:
                progress.update(1)
    finally:
        for hook in hooks:
            hook.remove()
        if progress is not None:
            progress.close()

    stats: Dict[str, torch.Tensor] = {}
    for module_name, sumsq in sumsqs.items():
        count = max(counts.get(module_name, 0), 1)
        rms = torch.sqrt((sumsq / count).to(torch.float32) + 1e-8)
        rms = rms / rms.mean().clamp_min(1e-6)
        stats[module_name] = rms.cpu()
    return stats


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
    parser = argparse.ArgumentParser(description="Weight-only quantize a fp16 HF checkpoint into GPTQ format (fast).")
    parser.add_argument(
        "--fp16_model_dir",
        type=str,
        default=os.environ.get("FP16_MODEL_DIR", "/root/autodl-tmp/models/output/Llama-2-7b-fp16-from-2bit"),
        help="Local path (or repo id) of the *unquantized* (fp16) HF checkpoint.",
    )
    parser.add_argument(
        "--out_quant_dir",
        type=str,
        default=os.environ.get("OUT_QUANT_DIR", "/root/autodl-tmp/models/output/Llama-2-7b-GPTQ-4bit-weightonly"),
        help="Output directory for the GPTQ-format quantized model.",
    )
    parser.add_argument(
        "--quant_param_cache",
        type=str,
        default=None,
        help="Optional path to save searched quantization parameters (scale/zero/g_idx/clip metadata). Defaults to <out_quant_dir>/quant_params.pt.",
    )
    parser.add_argument(
        "--calib_text_file",
        type=str,
        default=os.environ.get("CALIB_TEXT_FILE", None),
        required=False,
        help="Plain-text file used for minimal calibration (optional, but some code paths warn if missing).",
    )
    parser.add_argument("--calib_seq_len", type=int, default=512, help="Sequence length for dummy calibration blocks")
    parser.add_argument("--calib_num_samples", type=int, default=1, help="Number of calibration blocks (weight-only needs none, but keep >=1)")

    parser.add_argument("--bits", type=int, default=4, help="Bits to quantize to (4)")
    parser.add_argument("--group_size", type=int, default=64, help="Group size for per-group quantization")
    parser.add_argument(
        "--per_channel",
        action="store_true",
        help="Use per-channel quantization over the full input dimension of each layer. Equivalent to `--group_size -1`.",
    )
    parser.add_argument("--desc_act", action="store_true", help="Enable desc_act (kept for compatibility)")
    parser.set_defaults(sym=False)
    quant_mode_group = parser.add_mutually_exclusive_group()
    quant_mode_group.add_argument(
        "--sym",
        dest="sym",
        action="store_true",
        help="Use symmetric quantization.",
    )
    quant_mode_group.add_argument(
        "--asym",
        dest="sym",
        action="store_false",
        help="Use asymmetric quantization (default).",
    )
    parser.add_argument(
        "--mse",
        type=float,
        default=2.0,
        help="Enable MSE-based scale search when > 0. Default 2.0 means squared-error search.",
    )
    parser.add_argument(
        "--mse_grid",
        type=int,
        default=100,
        help="Number of candidate clipping scales used during MSE search.",
    )
    parser.add_argument(
        "--maxshrink",
        type=float,
        default=0.8,
        help="Lower bound for clipping search. Smaller means stronger clipping can be explored.",
    )
    parser.add_argument(
        "--clip_ratio",
        type=float,
        default=1.0,
        help="Optional manual absmax clipping before MSE search. 1.0 disables extra clipping.",
    )
    parser.add_argument(
        "--clip_search_grid",
        type=int,
        default=9,
        help="Number of independent lower/upper clipping candidates for scale/zero search. >1 enables asymmetric bound search.",
    )
    parser.add_argument(
        "--refine_scale_grid",
        type=int,
        default=7,
        help="Number of local scale multipliers used after the initial search.",
    )
    parser.add_argument(
        "--refine_scale_range",
        type=float,
        default=0.15,
        help="Local scale refinement range. 0.15 means searching around [0.85, 1.15] * scale.",
    )
    parser.add_argument(
        "--refine_zero_radius",
        type=int,
        default=4,
        help="Local zero-point integer search radius around the current best candidate.",
    )
    parser.add_argument(
        "--refine_rounds",
        type=int,
        default=2,
        help="Number of local search refinement rounds for scale/zero-point.",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="Use calibration activations to weight reconstruction error when searching scale/zero-point.",
    )
    parser.add_argument(
        "--act_aware_alpha",
        type=float,
        default=1.0,
        help="Exponent applied to per-input-channel RMS weights. 1.0 means linear RMS weighting.",
    )
    parser.add_argument(
        "--quantize_lm_head",
        action="store_true",
        help="Also quantize `lm_head`. Disabled by default because naive RTN on `lm_head` often severely hurts generation quality.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable quantization progress display.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device reported in QuantizeConfig (used by packing selection).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=os.environ.get("GPTQ_BACKEND", "auto"),
        help="gptqmodel backend for packing (auto/torch/triton).",
    )
    parser.add_argument("--trust_remote_code", action="store_true")

    args = parser.parse_args()

    if args.per_channel:
        args.group_size = -1

    if args.act_aware and args.calib_text_file is None:
        raise ValueError("--act_aware requires --calib_text_file so activation statistics can be collected.")

    tokenizer = AutoTokenizer.from_pretrained(args.fp16_model_dir, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build a tiny calibration dataset if provided (not used for true quantization here)
    if args.calib_text_file is not None:
        text = _read_text(args.calib_text_file)
        calib_dataset = build_calibration_dataset_from_text(
            tokenizer=tokenizer,
            text=text,
            seq_len=args.calib_seq_len,
            num_samples=args.calib_num_samples,
        )
    else:
        # create a minimal dummy batch with pad token to satisfy prepare_dataset expectations
        dummy = torch.tensor([[tokenizer.pad_token_id] * args.calib_seq_len], dtype=torch.long)
        calib_dataset = [{"input_ids": dummy, "attention_mask": torch.ones_like(dummy)}]

    qcfg = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        sym=args.sym,
        device=args.device,
        mse=args.mse,
        mock_quantization=False,
    )

    # Load model (as GPTQModel so .save_quantized exists)
    model = GPTQModel.load(
        model_id_or_path=args.fp16_model_dir,
        quantize_config=qcfg,
        backend=args.backend,
        trust_remote_code=args.trust_remote_code,
    )

    # collect candidate linear modules to quantize (only weight matrices)
    orig_modules = {}
    target_names = []
    for name, mod in model.named_modules():
        # 默认跳过 lm_head，保留全精度输出头。
        # 对当前这版朴素 RTN 而言，量化 lm_head 往往会直接导致 logits 崩坏、<unk> 泛滥和 PPL 爆炸。
        if not args.quantize_lm_head and name.endswith("lm_head"):
            continue
        # pick nn.Linear and transformers.Conv1D-like modules with `weight`
        if isinstance(mod, (nn.Linear, transformers.pytorch_utils.Conv1D)):
            orig_modules[name] = mod
            target_names.append(name)

    if len(target_names) == 0:
        raise RuntimeError("No linear modules found to quantize")

    activation_channel_rms: Dict[str, torch.Tensor] = {}
    if args.act_aware:
        activation_channel_rms = collect_activation_channel_rms(
            model=model,
            target_modules=orig_modules,
            calib_dataset=calib_dataset,
            show_progress=not args.no_progress,
        )
        print(f"activation_aware_layers: {len(activation_channel_rms)}", flush=True)

    # use gptqmodel kernel selection to get proper quant linear class
    quant_result_placeholder = {n: {} for n in target_names}
    quant_linear_cls = make_quant(
        model,
        quant_result=quant_result_placeholder,
        qcfg=qcfg,
        backend=args.backend,
        lm_head_name=None,
        pack=False,
    )

    # ensure the chosen kernel supports `pack()` (fallback to TorchQuantLinear)
    if not hasattr(quant_linear_cls, "pack"):
        from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
        quant_linear_cls = TorchQuantLinear

    # Replace target modules with quant layers using create_quant_layer via make_quant helper
    # (call create_quant_layer by reusing make_quant path: pass same quant_result)
    # create_quant_layer will replace entries present in quant_result_placeholder
    # The call above returned compatible quant linear class; now explicitly create layers
    from gptqmodel.utils.model import create_quant_layer
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
            "fp16_model_dir": args.fp16_model_dir,
            "search_method": "activation_aware_independent_clip_then_local_scale_zero_refine" if args.act_aware else "independent_clip_then_local_scale_zero_refine",
        },
        "layers": {},
    }

    # Now compute per-module group-wise min-max scales/zeros and call pack()
    layer_progress = None if args.no_progress else _create_progress(total=len(target_names), desc="Quantizing layers", leave=True)
    quant_start_time = time.time()

    for layer_idx, name in enumerate(target_names, start=1):
        orig = orig_modules[name]
        quant_mod = dict(model.named_modules())[name]
        layer_channel_weights = activation_channel_rms.get(name)

        if layer_progress is not None:
            layer_progress.set_postfix_str(f"{layer_idx}/{len(target_names)} {name}")

        # get weight matrix in shape (out_features, in_features)
        W = orig.weight.data.clone()
        if isinstance(orig, transformers.pytorch_utils.Conv1D):
            W = W.T

        # Build dequantized quantized weights first. This is important when using
        # clipping / MSE search because `pack()` itself does not clamp integers;
        # packing the original outlier weights with a clipped scale would overflow.
        Wq = torch.empty_like(W)

        out_features, in_features = W.shape
        group_size = qcfg.group_size if qcfg.group_size != -1 else in_features
        if in_features % group_size != 0:
            # fall back to full-group
            group_size = in_features
        num_groups = in_features // group_size

        scales = torch.zeros((num_groups, out_features), dtype=torch.float32)
        zeros = torch.zeros((num_groups, out_features), dtype=torch.int32)

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

            # Optional extra clipping before the built-in MSE clipping search.
            # This is applied per output channel to reduce outlier sensitivity.
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

            # Use the refined searched parameters to fake-quantize the original
            # weight chunk, so clipping error is accounted for during search.
            Wq[:, s] = qchunk.to(W.dtype)

            if group_progress is not None:
                group_progress.update(1)

        if group_progress is not None:
            group_progress.close()

        # g_idx maps each input feature to group index
        g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)

        # pack API expects scales/zeros transposed here and will transpose them internally.
        # We pass a temporary layer carrying Wq so the packed integers match the
        # searched/clipped quantized weights exactly.
        if isinstance(orig, nn.Linear):
            linear_for_pack = nn.Linear(orig.in_features, orig.out_features, bias=orig.bias is not None)
            linear_for_pack.weight.data.copy_(Wq)
            if orig.bias is not None:
                linear_for_pack.bias.data.copy_(orig.bias.data)
        elif isinstance(orig, transformers.pytorch_utils.Conv1D):
            linear_for_pack = transformers.pytorch_utils.Conv1D(orig.weight.shape[1], orig.weight.shape[0])
            linear_for_pack.weight.data.copy_(Wq.T)
            if orig.bias is not None:
                linear_for_pack.bias.data.copy_(orig.bias.data)
        else:
            raise TypeError(f"Unsupported layer type for packing: {type(orig)}")

        quant_mod.pack(linear=linear_for_pack, scales=scales.T.contiguous(), zeros=zeros.T.contiguous(), g_idx=g_idx)

        quant_param_cache["layers"][name] = {
            "scales": scales.cpu().contiguous(),
            "zeros": zeros.cpu().contiguous(),
            "g_idx": g_idx.cpu().contiguous(),
            "group_size": int(group_size),
            "in_features": int(in_features),
            "out_features": int(out_features),
            "clip_ratio": float(args.clip_ratio),
            "layer_type": orig.__class__.__name__,
            "act_aware": layer_channel_weights is not None,
        }

        if layer_progress is not None:
            layer_progress.update(1)

    if layer_progress is not None:
        layer_progress.close()

    # mark model as quantized and set required metadata so save_quantized proceeds
    model.quantize_config = qcfg
    model.quantized = True
    model.qlinear_kernel = quant_linear_cls
    model.load_quantized_model = False
    if not hasattr(model, "quant_log"):
        model.quant_log = []

    # Save quantized model in GPTQ format using GPTQModel helper
    model.save_quantized(args.out_quant_dir)

    quant_param_cache_path = args.quant_param_cache or os.path.join(args.out_quant_dir, DEFAULT_QUANT_PARAM_CACHE)
    os.makedirs(os.path.dirname(quant_param_cache_path), exist_ok=True)
    torch.save(quant_param_cache, quant_param_cache_path)

    print(f"saved_weight_only_gptq: {args.out_quant_dir}")
    print(f"saved_quant_param_cache: {quant_param_cache_path}")
    print(f"quantized_lm_head: {args.quantize_lm_head}")
    print(f"quant_mode: {'sym' if args.sym else 'asym'}")
    print(f"per_channel: {args.per_channel}")
    print(f"mse: {args.mse}")
    print(f"mse_grid: {args.mse_grid}")
    print(f"maxshrink: {args.maxshrink}")
    print(f"clip_ratio: {args.clip_ratio}")
    print(f"act_aware: {args.act_aware}")
    print(f"act_aware_alpha: {args.act_aware_alpha}")
    print(f"quantization_elapsed_sec: {time.time() - quant_start_time:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
