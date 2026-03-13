#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from safetensors import safe_open


def find_tensor_file(model_dir: Path, tensor_name: str) -> Path:
    index_file = model_dir / 'model.safetensors.index.json'
    if index_file.exists():
        with index_file.open('r', encoding='utf-8') as f:
            weight_map = json.load(f)['weight_map']
        if tensor_name not in weight_map:
            raise KeyError(f'{tensor_name} not found in index')
        return model_dir / weight_map[tensor_name]

    files = sorted(model_dir.glob('*.safetensors'))
    for file in files:
        with safe_open(file, framework='pt', device='cpu') as f:
            if tensor_name in f.keys():
                return file
    raise KeyError(f'{tensor_name} not found in {model_dir}')


def load_tensor(model_dir: Path, tensor_name: str) -> torch.Tensor:
    file = find_tensor_file(model_dir, tensor_name)
    with safe_open(file, framework='pt', device='cpu') as f:
        return f.get_tensor(tensor_name)


def load_quant_config(model_dir: Path) -> dict:
    with (model_dir / 'quantize_config.json').open('r', encoding='utf-8') as f:
        return json.load(f)


def unpack_qweight(packed_qweight: torch.Tensor, bits: int) -> torch.Tensor:
    if bits not in (2, 4, 8):
        raise NotImplementedError(f'Only 2/4/8-bit GPTQ qweight is supported here, got bits={bits}')
    pack_factor = 32 // bits
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    unpacked = (packed_qweight.to(torch.int32).unsqueeze(1) >> shifts.view(1, pack_factor, 1)) & ((1 << bits) - 1)
    return unpacked.reshape(packed_qweight.shape[0] * pack_factor, packed_qweight.shape[1]).contiguous()


def unpack_qzeros(packed_qzeros: torch.Tensor, bits: int, out_features: int, checkpoint_format: str) -> torch.Tensor:
    if bits not in (2, 4, 8):
        raise NotImplementedError(f'Only 2/4/8-bit GPTQ qzeros is supported here, got bits={bits}')
    packed_qzeros = packed_qzeros.to(torch.int32)
    if checkpoint_format == 'gptq':
        pack_bits = packed_qzeros.element_size() * 8
        offset = sum(1 << (bits * i) for i in range(pack_bits // bits))
        packed_qzeros = packed_qzeros + offset
    elif checkpoint_format != 'gptq_v2':
        raise ValueError(f'Unsupported checkpoint_format: {checkpoint_format}')
    pack_factor = 32 // bits
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    unpacked = (packed_qzeros.unsqueeze(-1) >> shifts.view(1, 1, pack_factor)) & ((1 << bits) - 1)
    zeros = unpacked.reshape(packed_qzeros.shape[0], packed_qzeros.shape[1] * pack_factor)[:, :out_features].contiguous()
    return zeros


def load_g_idx_or_default(model_dir: Path, tensor_prefix: str, in_features: int, group_size: int) -> torch.Tensor:
    try:
        g_idx = load_tensor(model_dir, tensor_prefix + '.g_idx').cpu().to(torch.int32)
        if g_idx.numel() == 0:
            raise ValueError('empty g_idx')
        return g_idx
    except Exception:
        return (torch.arange(in_features, dtype=torch.int32) // group_size).contiguous()


class GPTQSingleLinearExact(torch.nn.Module):
    def __init__(self, qweight_int: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, g_idx: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer('qweight_int', qweight_int.to(torch.int32))
        self.register_buffer('scales', scales)
        self.register_buffer('zero_points', zero_points.to(torch.int32))
        self.register_buffer('g_idx', g_idx.to(torch.int32))
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        g_idx = self.g_idx.long()
        qweight = self.qweight_int.to(dtype)
        zero_points = self.zero_points.to(dtype)
        scales = self.scales.to(dtype)
        return scales[g_idx] * (qweight - zero_points[g_idx])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.qweight_int.shape[1],)
        x_2d = x.reshape(-1, x.shape[-1])
        weight = self.dequantize_weight(dtype=x.dtype)
        out = torch.matmul(x_2d, weight).reshape(out_shape)
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate exporting a single GPTQ linear layer to ONNX by explicitly unpacking qweight/qzeros and reconstructing TorchQuantLinear-style dequantization.')
    parser.add_argument('--model_dir', type=Path, required=True)
    parser.add_argument('--fp16_dir', type=Path, required=True)
    parser.add_argument('--tensor_prefix', type=str, default='model.layers.0.self_attn.q_proj')
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--opset', type=int, default=18)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    quant_cfg = load_quant_config(args.model_dir)
    bits = int(quant_cfg['bits'])
    group_size = int(quant_cfg['group_size'])
    checkpoint_format = str(quant_cfg.get('checkpoint_format', 'gptq_v2'))

    packed_qweight = load_tensor(args.model_dir, args.tensor_prefix + '.qweight').cpu()
    scales = load_tensor(args.model_dir, args.tensor_prefix + '.scales').cpu().to(torch.float32)
    packed_qzeros = load_tensor(args.model_dir, args.tensor_prefix + '.qzeros').cpu()
    bias = None
    try:
        bias = load_tensor(args.model_dir, args.tensor_prefix + '.bias').cpu()
        if bias.numel() == 0:
            bias = None
    except Exception:
        pass

    qweight_int = unpack_qweight(packed_qweight, bits=bits)
    zero_points = unpack_qzeros(
        packed_qzeros,
        bits=bits,
        out_features=scales.shape[1],
        checkpoint_format=checkpoint_format,
    )
    g_idx = load_g_idx_or_default(args.model_dir, args.tensor_prefix, in_features=qweight_int.shape[0], group_size=group_size)

    module = GPTQSingleLinearExact(
        qweight_int=qweight_int,
        scales=scales,
        zero_points=zero_points,
        g_idx=g_idx,
        bias=bias,
    ).eval()

    fp16_weight = load_tensor(args.fp16_dir, args.tensor_prefix + '.weight').cpu().float()
    fp16_bias = None
    try:
        fp16_bias = load_tensor(args.fp16_dir, args.tensor_prefix + '.bias').cpu().float()
        if fp16_bias.numel() == 0:
            fp16_bias = None
    except Exception:
        pass

    in_features = fp16_weight.shape[1]
    x = torch.randn(args.batch, in_features, dtype=torch.float32)
    with torch.no_grad():
        dequant_weight = module.dequantize_weight(dtype=torch.float32)
        y_quant = module(x)
        y_ref = torch.nn.functional.linear(x, fp16_weight, fp16_bias)
    weight_max_abs = (dequant_weight.T - fp16_weight).abs().max().item()
    weight_mean_abs = (dequant_weight.T - fp16_weight).abs().mean().item()
    max_abs = (y_quant - y_ref).abs().max().item()
    mean_abs = (y_quant - y_ref).abs().mean().item()
    print(f'checkpoint_format: {checkpoint_format}')
    print(f'weight_max_abs_diff_vs_fp16: {weight_max_abs:.8f}')
    print(f'weight_mean_abs_diff_vs_fp16: {weight_mean_abs:.8f}')
    print(f'max_abs_diff: {max_abs:.8f}')
    print(f'mean_abs_diff: {mean_abs:.8f}')
    print(f'bits: {bits}, group_size: {group_size}, qweight_shape: {tuple(qweight_int.shape)}')
    print(f'scales_shape: {tuple(scales.shape)}, zero_points_shape: {tuple(zero_points.shape)}, g_idx_shape: {tuple(g_idx.shape)}')

    onnx_path = args.out_dir / 'single_linear.onnx'
    torch.onnx.export(
        module,
        (x,),
        str(onnx_path),
        input_names=['x'],
        output_names=['y'],
        dynamic_axes={'x': {0: 'batch'}, 'y': {0: 'batch'}},
        opset_version=args.opset,
        external_data=False,
        do_constant_folding=False,
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))

    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {'x': x.numpy()})[0]
    ort_max = np.max(np.abs(ort_out - y_quant.numpy()))
    print(f'onnx_path: {onnx_path}')
    print(f'ort_max_abs_diff_vs_torch: {ort_max:.8f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
