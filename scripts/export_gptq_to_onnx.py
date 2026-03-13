#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from gptqmodel import GPTQModel
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.utils.backend import BACKEND
from transformers.models.llama import modeling_llama


def unpack_qweight(packed_qweight: torch.Tensor, bits: int, pack_dtype_bits: int) -> torch.Tensor:
    if bits not in (2, 4, 8):
        raise NotImplementedError(f'Only 2/4/8-bit GPTQ qweight is supported here, got bits={bits}')
    pack_factor = pack_dtype_bits // bits
    shifts = torch.arange(0, pack_dtype_bits, bits, dtype=torch.int64, device=packed_qweight.device)
    unpacked = (packed_qweight.to(torch.int64).unsqueeze(1) >> shifts.view(1, pack_factor, 1)) & ((1 << bits) - 1)
    return unpacked.reshape(packed_qweight.shape[0] * pack_factor, packed_qweight.shape[1]).to(torch.int32).contiguous()


def unpack_qzeros(packed_qzeros: torch.Tensor, bits: int, out_features: int, pack_dtype_bits: int, qzero_format: int) -> torch.Tensor:
    if bits not in (2, 4, 8):
        raise NotImplementedError(f'Only 2/4/8-bit GPTQ qzeros is supported here, got bits={bits}')

    packed_qzeros = packed_qzeros.to(torch.int64)
    if qzero_format == 1:
        offset = sum(1 << (bits * i) for i in range(pack_dtype_bits // bits))
        packed_qzeros = packed_qzeros + offset
    elif qzero_format != 2:
        raise ValueError(f'Unsupported qzero_format: {qzero_format}')

    pack_factor = pack_dtype_bits // bits
    shifts = torch.arange(0, pack_dtype_bits, bits, dtype=torch.int64, device=packed_qzeros.device)
    unpacked = (packed_qzeros.unsqueeze(-1) >> shifts.view(1, 1, pack_factor)) & ((1 << bits) - 1)
    zeros = unpacked.reshape(packed_qzeros.shape[0], packed_qzeros.shape[1] * pack_factor)[:, :out_features]
    return zeros.to(torch.int32).contiguous()


class GPTQLinearOnnxExact(torch.nn.Module):
    def __init__(
        self,
        qweight_int: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        g_idx: torch.Tensor,
        bias: torch.Tensor | None,
    ):
        super().__init__()
        self.register_buffer('qweight_int', qweight_int.to(torch.int32).cpu())
        self.register_buffer('scales', scales.detach().to(torch.float32).cpu())
        self.register_buffer('zero_points', zero_points.to(torch.int32).cpu())
        self.register_buffer('g_idx', g_idx.to(torch.int32).cpu())
        if bias is not None:
            self.register_buffer('bias', bias.detach().cpu())
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


class CausalLMOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits


def install_export_friendly_llama_mask() -> None:
    def simple_causal_mask(
        config,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        past_key_values,
        position_ids: torch.Tensor | None = None,
        or_mask_function=None,
        and_mask_function=None,
    ) -> torch.Tensor:
        del config, cache_position, past_key_values, position_ids, or_mask_function, and_mask_function
        batch_size, query_length = input_embeds.shape[:2]
        key_length = query_length if attention_mask is None else attention_mask.shape[-1]
        dtype = input_embeds.dtype
        neg = torch.finfo(dtype).min

        upper = torch.triu(
            torch.ones((query_length, key_length), dtype=torch.bool, device=input_embeds.device),
            diagonal=1,
        )
        causal_mask = torch.zeros((query_length, key_length), dtype=dtype, device=input_embeds.device)
        causal_mask = causal_mask.masked_fill(upper, neg)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_length, key_length).contiguous()

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid = attention_mask[:, None, None, :key_length].to(torch.bool)
                causal_mask = causal_mask.masked_fill(~valid, neg)
            elif attention_mask.dim() == 4:
                causal_mask = causal_mask + attention_mask[:, :, :query_length, :key_length].to(dtype)

        return causal_mask

    modeling_llama.create_causal_mask = simple_causal_mask


def convert_torch_quant_linear(module: TorchQuantLinear) -> GPTQLinearOnnxExact:
    if getattr(module, 'adapter', None) is not None:
        raise NotImplementedError('Adapter-attached TorchQuantLinear is not supported by this exporter')

    bits = int(module.bits)
    pack_dtype_bits = int(module.pack_dtype_bits)
    out_features = int(module.out_features)

    qweight_int = unpack_qweight(module.qweight.detach().cpu(), bits=bits, pack_dtype_bits=pack_dtype_bits)
    zero_points = unpack_qzeros(
        module.qzeros.detach().cpu(),
        bits=bits,
        out_features=out_features,
        pack_dtype_bits=pack_dtype_bits,
        qzero_format=int(module.qzero_format()),
    )
    scales = module.scales.detach().cpu().to(torch.float32)
    g_idx = module.g_idx.detach().cpu().to(torch.int32)
    bias = None if module.bias is None else module.bias.detach().cpu()
    return GPTQLinearOnnxExact(
        qweight_int=qweight_int,
        scales=scales,
        zero_points=zero_points,
        g_idx=g_idx,
        bias=bias,
    )


def replace_torch_quant_linears(root: torch.nn.Module) -> int:
    replaced = 0
    for name, child in list(root.named_children()):
        if isinstance(child, TorchQuantLinear):
            setattr(root, name, convert_torch_quant_linear(child))
            replaced += 1
        else:
            replaced += replace_torch_quant_linears(child)
    return replaced


def build_dummy_inputs(model: torch.nn.Module, batch_size: int, seq_len: int) -> torch.Tensor:
    vocab_size = int(getattr(model.config, 'vocab_size', 32000))
    input_ids = (torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len) % max(vocab_size, 1)).contiguous()
    return input_ids


def main() -> int:
    parser = argparse.ArgumentParser(description='Export a GPTQ model to ONNX by replacing every TorchQuantLinear with a custom nn.Module that explicitly performs GPTQ dequantization: qweight/scales/qzeros -> (q - zp) * scale -> matmul.')
    parser.add_argument('--in_quant_dir', type=str, required=True, help='Input GPTQ model directory')
    parser.add_argument('--out_onnx_dir', type=str, required=True, help='Output ONNX directory')
    parser.add_argument('--opset', type=int, default=18, help='ONNX opset version')
    parser.add_argument('--batch_size', type=int, default=1, help='Dummy batch size for export')
    parser.add_argument('--seq_len', type=int, default=8, help='Dummy sequence length for export')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device used to trace the model')
    parser.add_argument('--attn_implementation', type=str, default='eager', help='Attention implementation used when loading the HF model, e.g. eager or sdpa')
    parser.add_argument('--external_data', action='store_true', help='Store large initializers in external data files')
    parser.add_argument('--disable_validation', action='store_true', help='Skip ONNX Runtime session validation')
    parser.add_argument('--dry_run_replace_only', action='store_true', help='Only load the GPTQ model and replace TorchQuantLinear modules without exporting')
    parser.add_argument('--trust_remote_code', action='store_true')
    args = parser.parse_args()

    out_onnx_dir = Path(args.out_onnx_dir)
    out_onnx_dir.mkdir(parents=True, exist_ok=True)

    install_export_friendly_llama_mask()

    gptq_model = GPTQModel.load(
        model_id_or_path=args.in_quant_dir,
        backend=BACKEND.TORCH,
        device=args.device,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model = gptq_model.model.eval()
    model.config.use_cache = False
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.use_cache = False

    replaced = replace_torch_quant_linears(model)
    print(f'replaced_torch_quant_linear: {replaced}')
    if replaced == 0:
        raise RuntimeError('No TorchQuantLinear modules were found to replace')

    if hasattr(model.config, 'quantization_config'):
        delattr(model.config, 'quantization_config')

    if args.dry_run_replace_only:
        return 0

    wrapper = CausalLMOnnxWrapper(model).eval()
    input_ids = build_dummy_inputs(model, batch_size=args.batch_size, seq_len=args.seq_len)
    if args.device == 'cuda':
        wrapper = wrapper.cuda()
        input_ids = input_ids.cuda()

    onnx_path = out_onnx_dir / 'model.onnx'
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids,),
            str(onnx_path),
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'},
            },
            opset_version=args.opset,
            external_data=args.external_data,
            do_constant_folding=False,
            dynamo=False,
        )

    onnx.checker.check_model(str(onnx_path))
    print(f'exported_onnx: {onnx_path}')

    if not args.disable_validation:
        sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        print(f'ort_inputs: {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}')
        print(f'ort_outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
