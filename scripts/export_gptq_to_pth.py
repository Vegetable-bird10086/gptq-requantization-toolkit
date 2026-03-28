#!/usr/bin/env python3
"""Simple exporter: export GPTQ -> HF bf16 checkpoint and save a plain dict .pth

This script only performs the minimal steps required to export a GPTQ model
to a Hugging Face checkpoint, load it with `torch_dtype=bfloat16`, convert the
returned state_dict to a plain `dict`, print its type, and save it to the
requested `.pth` path.
"""
import argparse
import shutil
import tempfile
import torch
from gptqmodel import GPTQModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Export GPTQ quantized model to a .pth state_dict via HF bf16 export")
    parser.add_argument("--in_quant_dir", type=str, required=True, help="Path to GPTQ quantized model directory")
    parser.add_argument("--out_pth", type=str, required=True, help="Output .pth path to write state_dict")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="gptq_bf16_")
    try:
        print("Exporting quantized model to HF bf16 checkpoint (temporary):", tmpdir)
        GPTQModel.export(
            model_id_or_path=args.in_quant_dir,
            target_path=tmpdir,
            format="hf",
            trust_remote_code=args.trust_remote_code,
        )

        # Load using transformers with bfloat16 and save a plain dict .pth
        from transformers import AutoModel, AutoConfig
        try:
            from transformers import AutoModelForCausalLM as _AM
            ModelLoader = _AM
        except Exception:
            ModelLoader = AutoModel

        cfg = AutoConfig.from_pretrained(tmpdir, trust_remote_code=args.trust_remote_code)
        model = ModelLoader.from_pretrained(tmpdir, config=cfg, trust_remote_code=args.trust_remote_code, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

        # Ensure we save a plain dict (not OrderedDict) so `type(sd)` is <class 'dict'>
        sd = dict(model.state_dict())
        print("Loaded type:", type(sd))

        print(f"Saving state_dict ({len(sd)} tensors) to: {args.out_pth}")
        torch.save(sd, args.out_pth)
        print("Saved .pth successfully")

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
