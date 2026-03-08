import argparse
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import transformers

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils.model import make_quant


def _fake_quantize_group(chunk: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int) -> torch.Tensor:
    maxq = 2 ** bits - 1
    scale = scale.to(device=chunk.device, dtype=torch.float32).unsqueeze(1)
    zero = zero.to(device=chunk.device, dtype=torch.float32).unsqueeze(1)
    q = torch.clamp(torch.round(chunk.float() / scale) + zero, 0, maxq)
    return (scale * (q - zero)).to(chunk.dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast 4-bit re-quantization from saved scale/zero cache.")
    parser.add_argument("--fp16_model_dir", type=str, required=True, help="Path to fp16 model directory.")
    parser.add_argument("--quant_param_cache", type=str, required=True, help="Path to quant_params.pt saved by weight_only_quantize.py.")
    parser.add_argument("--out_quant_dir", type=str, required=True, help="Output GPTQ-format directory.")
    parser.add_argument("--backend", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    cache: Dict[str, Any] = torch.load(args.quant_param_cache, map_location="cpu")
    meta = cache["metadata"]
    layers = cache["layers"]

    qcfg = QuantizeConfig(
        bits=int(meta["bits"]),
        group_size=int(meta["group_size"]),
        desc_act=bool(meta["desc_act"]),
        sym=bool(meta["sym"]),
        mse=float(meta["mse"]),
        mock_quantization=False,
    )

    model = GPTQModel.load(
        model_id_or_path=args.fp16_model_dir,
        quantize_config=qcfg,
        backend=args.backend,
        trust_remote_code=args.trust_remote_code,
    )

    target_names = list(layers.keys())
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
        from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
        quant_linear_cls = TorchQuantLinear

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

    named_modules = dict(model.named_modules())
    for name, info in layers.items():
        orig = named_modules[name]
        quant_mod = named_modules[name]
        if hasattr(orig, "qweight"):
            # `named_modules` now points at the quant layer; fetch original layer from saved fp16 model path is not available,
            # so use the quant layer replacement's source stored before replacement via module tree lookup on GPTQ model.
            pass

    # Rebuild a map of original fp16 layers by reloading a fresh plain model handle.
    src_model = GPTQModel.load(
        model_id_or_path=args.fp16_model_dir,
        quantize_config=qcfg,
        backend=args.backend,
        trust_remote_code=args.trust_remote_code,
    )
    src_named_modules = dict(src_model.named_modules())
    dst_named_modules = dict(model.named_modules())

    for name, info in layers.items():
        orig = src_named_modules[name]
        quant_mod = dst_named_modules[name]

        W = orig.weight.data.clone()
        if isinstance(orig, transformers.pytorch_utils.Conv1D):
            W = W.T

        group_size = int(info["group_size"])
        in_features = int(info["in_features"])
        out_features = int(info["out_features"])
        scales = info["scales"].to(torch.float32)
        zeros = info["zeros"].to(torch.int32)
        g_idx = info["g_idx"].to(torch.int32)

        Wq = torch.empty_like(W)
        num_groups = scales.shape[0]
        for g in range(num_groups):
            s = slice(g * group_size, min((g + 1) * group_size, in_features))
            chunk = W[:, s]
            Wq[:, s] = _fake_quantize_group(chunk, scales[g], zeros[g], bits=qcfg.bits)

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

        quant_mod.pack(
            linear=linear_for_pack,
            scales=scales.T.contiguous(),
            zeros=zeros.T.contiguous(),
            g_idx=g_idx,
        )

    model.quantize_config = qcfg
    model.quantized = True
    model.qlinear_kernel = quant_linear_cls
    model.load_quantized_model = False
    if not hasattr(model, "quant_log"):
        model.quant_log = []

    model.save_quantized(args.out_quant_dir)
    print(f"saved_requantized_model: {args.out_quant_dir}")
    print(f"loaded_quant_param_cache: {args.quant_param_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
