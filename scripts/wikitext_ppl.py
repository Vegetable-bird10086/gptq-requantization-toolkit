import argparse
import math
import os
from typing import Optional

import torch
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from transformers import AutoTokenizer

from gptqmodel import GPTQModel


def _pick_text_column(dataset) -> str:
    for candidate in ("text", "content"):
        if candidate in dataset.column_names:
            return candidate
    raise ValueError(f"Unknown text column. Available columns: {dataset.column_names}")


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _get_hf_model(gptq_model):
    # GPTQModel often wraps an underlying HF PreTrainedModel in `.model`.
    return getattr(gptq_model, "model", gptq_model)


@torch.no_grad()
def compute_ppl(
    hf_model,
    tokenizer,
    input_ids: torch.Tensor,
    device: torch.device,
    max_length: int,
    stride: int,
    show_progress: bool,
) -> tuple[float, float, int]:
    hf_model.eval()

    seq_len_total = input_ids.size(1)
    if seq_len_total < 2:
        raise ValueError("Not enough tokens to compute perplexity.")

    nll_sum = 0.0
    token_count = 0
    total_steps = len(range(0, seq_len_total, stride))

    iterator = range(0, seq_len_total, stride)
    progress_bar = None
    last_progress_print = -1
    if show_progress:
        try:
            from tqdm import tqdm

            progress_bar = tqdm(total=total_steps, desc="ppl", unit="step")
        except Exception:
            print(f"[ppl] 0/{total_steps} steps | 0/{seq_len_total} tokens | 0.0%", flush=True)

    for step_idx, i in enumerate(iterator, start=1):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len_total)
        trg_len = end_loc - i
        if trg_len <= 0:
            continue

        input_ids_slice = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100

        outputs = hf_model(
            input_ids=input_ids_slice,
            labels=target_ids,
            use_cache=False,
        )

        loss = outputs.loss
        # HF loss is mean over non-ignored labels, so multiply by trg_len for total NLL.
        nll_sum += loss.item() * trg_len
        token_count += trg_len

        if show_progress:
            percent = 100.0 * token_count / max(seq_len_total, 1)
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(tokens=f"{token_count}/{seq_len_total}", pct=f"{percent:.1f}%")
            else:
                should_print = step_idx == total_steps or step_idx == 1 or step_idx - last_progress_print >= 10
                if should_print:
                    print(
                        f"[ppl] {step_idx}/{total_steps} steps | {token_count}/{seq_len_total} tokens | {percent:.1f}%",
                        flush=True,
                    )
                    last_progress_print = step_idx

        if end_loc == seq_len_total:
            break

    if progress_bar is not None:
        progress_bar.close()

    mean_nll = nll_sum / max(token_count, 1)
    ppl = float(math.exp(mean_nll))
    return ppl, mean_nll, token_count


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute WikiText perplexity for a GPTQ model.")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/source/Llama-2-7b-EfficientQAT-w2g64-GPTQ"),
        help="Local path or HF repo id. Defaults to env MODEL_DIR or your local folder.",
    )
    parser.add_argument("--dataset", type=str, default="wikitext", help="HF dataset name")
    parser.add_argument(
        "--subset",
        type=str,
        default="wikitext-2-raw-v1",
        help="HF dataset config/subset (default: wikitext-2-raw-v1)",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate")
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="Offline mode: path to a local plain-text file to evaluate (overrides --dataset/--subset/--split).",
    )
    parser.add_argument("--max_length", type=int, default=2048, help="Model context window for eval")
    parser.add_argument(
        "--max_eval_tokens",
        type=int,
        default=None,
        help="Optional cap on total number of tokens to evaluate (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride for sliding window. Smaller = more accurate, slower.",
    )
    parser.add_argument("--use_fast", action="store_true", help="Prefer fast tokenizer if available")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bar")

    args = parser.parse_args(argv)

    quant_dir = args.model

    tokenizer = AutoTokenizer.from_pretrained(quant_dir, use_fast=args.use_fast)
    if tokenizer.pad_token_id is None:
        # Common for LLaMA tokenizers; needed for some batch operations.
        tokenizer.pad_token_id = tokenizer.eos_token_id

    gptq_model = GPTQModel.from_quantized(quant_dir)
    device = getattr(gptq_model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    hf_model = _get_hf_model(gptq_model)

    if args.text_file:
        full_text = _read_text_file(args.text_file)
    else:
        try:
            ds = load_dataset(args.dataset, args.subset, split=args.split)
        except (ConnectionError, OSError, DatasetNotFoundError) as e:
            raise SystemExit(
                "Failed to load dataset from Hugging Face Hub. "
                "This environment appears to be offline or cannot reach huggingface.co.\n\n"
                "Options:\n"
                "1) Download WikiText on a machine with internet, copy the test text file here, then run with: \n"
                "   python wikitext_ppl.py --model <MODEL_DIR> --text_file <PATH_TO_TEST_TXT>\n"
                "2) If you have an HTTP proxy / network access, configure it and re-run.\n\n"
                f"Original error: {type(e).__name__}: {e}"
            )

        text_col = _pick_text_column(ds)
        # WikiText contains many empty lines; join with newlines to preserve boundaries.
        full_text = "\n\n".join([t for t in ds[text_col] if isinstance(t, str) and t.strip()])

    enc = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
        verbose=False,
    )
    input_ids = enc["input_ids"]

    if args.max_eval_tokens is not None and args.max_eval_tokens > 0:
        input_ids = input_ids[:, : args.max_eval_tokens]

    ppl, mean_nll, token_count = compute_ppl(
        hf_model=hf_model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        device=device,
        max_length=args.max_length,
        stride=args.stride,
        show_progress=not args.no_progress,
    )

    print(f"model: {quant_dir}")
    if args.text_file:
        print(f"text_file: {args.text_file}")
    else:
        print(f"dataset: {args.dataset}/{args.subset} split={args.split}")
    print(f"tokens: {token_count}")
    print(f"mean_nll: {mean_nll:.6f}")
    print(f"ppl: {ppl:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
