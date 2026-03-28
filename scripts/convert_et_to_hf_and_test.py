#!/usr/bin/env python3
"""Convert Executorch-style state_dict back to HF naming and run quick generation + PPL test.

Saves: /root/autodl-tmp/llama3.2-1b-2bit/dequantized_from_executorch_hf.pth
"""
import math
import os
import importlib.util
import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# reverse mapping: ET -> HF
def et_to_hf_key(key: str) -> str:
    # top-level
    if key == "tok_embeddings.weight":
        return "model.embed_tokens.weight"
    if key == "norm.weight":
        return "model.norm.weight"
    if key == "output.weight":
        return "lm_head.weight"

    # layers prefix
    key = key.replace("layers.", "model.layers.")

    # attention
    key = key.replace(".attention.wq.weight", ".self_attn.q_proj.weight")
    key = key.replace(".attention.wk.weight", ".self_attn.k_proj.weight")
    key = key.replace(".attention.wv.weight", ".self_attn.v_proj.weight")
    key = key.replace(".attention.wo.weight", ".self_attn.o_proj.weight")

    # ffn
    key = key.replace(".feed_forward.w1.weight", ".mlp.gate_proj.weight")
    key = key.replace(".feed_forward.w3.weight", ".mlp.up_proj.weight")
    key = key.replace(".feed_forward.w2.weight", ".mlp.down_proj.weight")

    # norms
    key = key.replace(".attention_norm.weight", ".input_layernorm.weight")
    key = key.replace(".ffn_norm.weight", ".post_attention_layernorm.weight")

    return key


def convert(et_sd: dict) -> dict:
    hf_sd = {}
    for k, v in et_sd.items():
        new_k = et_to_hf_key(k)
        hf_sd[new_k] = v
    return hf_sd


@torch.no_grad()
def compute_ppl_sliding_window(hf_model, tokenizer, input_ids: torch.Tensor, device: torch.device, max_length: int = 2048, stride: int = 512):
    hf_model.eval()
    seq_len_total = input_ids.size(1)
    if seq_len_total < 2:
        return float('nan'), float('nan'), 0

    nll_sum = 0.0
    token_count = 0

    for i in range(0, seq_len_total, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len_total)
        trg_len = end_loc - i
        if trg_len <= 0:
            continue

        input_ids_slice = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100

        outputs = hf_model(input_ids=input_ids_slice, labels=target_ids, use_cache=False)
        loss = outputs.loss
        nll_sum += loss.item() * trg_len
        token_count += trg_len

        if end_loc == seq_len_total:
            break

    mean_nll = nll_sum / max(token_count, 1)
    ppl = float(math.exp(mean_nll))
    return ppl, mean_nll, token_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--et-pth', default='/root/autodl-tmp/llama3.2-1b-2bit/executorch_model.pth')
    parser.add_argument('--config-dir', default='/root/autodl-tmp/llama3.2-1b')
    parser.add_argument('--tokenizer-dir', default='/root/autodl-tmp/llama3.2-1b')
    parser.add_argument('--out-pth', default='/root/autodl-tmp/llama3.2-1b/dequantized_from_executorch_hf.pth')
    parser.add_argument('--prompt', default='I would like to learn python, could you teach me with a simple example?')
    parser.add_argument('--ppl-text', default='/root/autodl-tmp/gptq-requantization-toolkit/data/wiki.test.raw')
    parser.add_argument('--max-ppl-tokens', type=int, default=2000)
    args = parser.parse_args()

    et_obj = torch.load(args.et_pth, map_location='cpu')
    if isinstance(et_obj, dict) and 'state_dict' in et_obj:
        et_sd = et_obj['state_dict']
    else:
        et_sd = et_obj

    print(f"Loaded ET pth: {args.et_pth}  tensors={len(et_sd)}")

    hf_sd = convert(et_sd)
    print(f"Converted to HF-style keys: tensors={len(hf_sd)}")

    # save HF-style state_dict
    torch.save(hf_sd, args.out_pth)
    print(f"Saved HF-style state_dict to: {args.out_pth}")

    # Load config and instantiate model then load state_dict
    config = AutoConfig.from_pretrained(args.config_dir)
    model = AutoModelForCausalLM.from_config(config)

    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    print(f"load_state_dict missing_keys: {len(missing)} unexpected_keys: {len(unexpected)}")
    if len(missing) > 0:
        print('Some missing keys (first 20):', missing[:20])
    if len(unexpected) > 0:
        print('Some unexpected keys (first 20):', unexpected[:20])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # generation test
    inputs = tokenizer(args.prompt, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=False)
    new = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    print('\n=== Generation Test ===')
    print('prompt:', args.prompt)
    print('generated (full):')
    print(full)
    print('generated (new):')
    print(new)

    # PPL test on first N tokens of text file (sliding window)
    if os.path.exists(args.ppl_text):
        with open(args.ppl_text, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()
        enc = tokenizer(full_text, return_tensors='pt', add_special_tokens=False)
        input_ids = enc['input_ids']
        if args.max_ppl_tokens and input_ids.size(1) > args.max_ppl_tokens:
            input_ids = input_ids[:, : args.max_ppl_tokens]
        ppl, mean_nll, token_count = compute_ppl_sliding_window(model, tokenizer, input_ids, device)
        print('\n=== PPL Test ===')
        print(f'tokens: {token_count} mean_nll: {mean_nll:.6f} ppl: {ppl:.4f}')
    else:
        print('\nPPL text file not found, skipping PPL test')


if __name__ == '__main__':
    main()
