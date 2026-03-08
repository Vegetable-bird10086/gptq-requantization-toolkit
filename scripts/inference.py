import argparse

import torch
from transformers import AutoTokenizer

from gptqmodel import GPTQModel


def main() -> int:
	parser = argparse.ArgumentParser(description="Run generation on a quantized GPTQ model.")
	parser.add_argument(
		"--model",
		type=str,
		default="/root/autodl-tmp/models/output/_weightonly-4bit",
		help="Local quantized model directory.",
	)
	parser.add_argument(
		"--prompt",
		type=str,
		default="The Large Language Model is",
		help="Prompt text.",
	)
	parser.add_argument("--max_new_tokens", type=int, default=64)
	parser.add_argument("--do_sample", action="store_true")
	parser.add_argument("--temperature", type=float, default=0.8)
	parser.add_argument("--top_p", type=float, default=0.95)
	parser.add_argument("--backend", type=str, default="auto")
	parser.add_argument("--trust_remote_code", action="store_true")
	args = parser.parse_args()

	model = GPTQModel.from_quantized(
		args.model,
		backend=args.backend,
		trust_remote_code=args.trust_remote_code,
	)

	tokenizer = getattr(model, "tokenizer", None)
	if tokenizer is None:
		tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)

	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id

	inputs = tokenizer(args.prompt, return_tensors="pt")
	if hasattr(model, "device"):
		inputs = {k: v.to(model.device) for k, v in inputs.items()}
	else:
		dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		inputs = {k: v.to(dev) for k, v in inputs.items()}

	generate_kwargs = dict(
		**inputs,
		max_new_tokens=args.max_new_tokens,
		do_sample=args.do_sample,
		pad_token_id=tokenizer.pad_token_id,
		eos_token_id=tokenizer.eos_token_id,
	)
	if args.do_sample:
		generate_kwargs.update(
			temperature=args.temperature,
			top_p=args.top_p,
		)

	with torch.inference_mode():
		out_ids = model.generate(**generate_kwargs)[0]

	full_text = tokenizer.decode(out_ids, skip_special_tokens=False)
	prompt_len = inputs["input_ids"].shape[1]
	new_text = tokenizer.decode(out_ids[prompt_len:], skip_special_tokens=False)

	print(f"model: {args.model}")
	print(f"prompt: {args.prompt}")
	print("=" * 80)
	print("full_output:")
	print(full_text)
	print("=" * 80)
	print("new_tokens_only:")
	print(new_text)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())