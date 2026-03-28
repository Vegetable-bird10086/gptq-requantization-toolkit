"""README (简化并同步 scripts/)"""

# Llama-2 GPTQ Re-Quantization Toolkit

一个面向 GPTQ 重量化 / 重新打包 / 导出到 HF checkpoint 的实验性工具集。



---

## 脚本说明

- `scripts/convert_et_to_hf_and_test.py`：
  - 作用：把 Executorch 风格的 state_dict 转为 Hugging Face 命名，并做一次简短的生成 + WikiText PPL 测试。
  - 示例：
    ```bash
    python scripts/convert_et_to_hf_and_test.py \
      --et-pth /path/to/executorch_model.pth \
      --config-dir /path/to/config_dir \
      --tokenizer-dir /path/to/tokenizer_dir \
      --out-pth /path/to/out_hf.pth \
      --prompt "Hello" \
      --ppl-text /path/to/wiki.test.raw
    ```

- `scripts/convert_llama_model.py`：
  - 作用：将 HF 命名的 state_dict 转为 Executorch/旧格式并保存（脚本内有默认路径，直接运行会执行转换）。
  - 示例：
    ```bash
    python scripts/convert_llama_model.py
    ```

- `scripts/export_2bit_gptq_to_fp16.py`：
  - 作用：把 GPTQ（例如 2-bit）模型导出为 Hugging Face 格式的 FP16 checkpoint（使用 `GPTQModel.export(format="hf")`）。
  - 示例：
    ```bash
    python scripts/export_2bit_gptq_to_fp16.py \
      --in_quant_dir /path/to/quant_dir \
      --out_fp16_dir /path/to/out_fp16_dir \
      [--trust_remote_code]
    ```

- `scripts/export_gptq_to_pth.py`：
  - 作用：通过先导出到 HF（bf16），再加载并保存为 plain `.pth` 字典（便于后续离线处理或检查）。
  - 参数：`--in_quant_dir`、`--out_pth`、`--trust_remote_code`。
  - 示例：
    ```bash
    python scripts/export_gptq_to_pth.py \
      --in_quant_dir /path/to/quant_dir \
      --out_pth /path/to/out.pth
    ```

- `scripts/direct_requantize_gptq.py`：
  - 作用：在 GPTQ 格式之间做直接重打包 / 提升 / 重新搜索（可无须落盘 FP16）。
  - 主要模式：
    - `--direct_repack`：保留原始整数码，直接打包到更高位宽容器；
    - `--direct_code_lift`：按比例把低位码值提升到高位空间；
    - 未指定以上参数则执行基于搜索的重新量化（可输出 `quant_params.pt`）。
  - 示例：
    ```bash
    python scripts/direct_requantize_gptq.py \
      --in_quant_dir /path/to/in_quant_dir \
      --out_quant_dir /path/to/out_quant_dir \
      --direct_repack
    ```

- `scripts/fast_requantize_from_cache.py`：
  - 作用：给定已保存的 `quant_params.pt`，在不重新搜索的情况下对 FP16 权重快速生成 GPTQ 模型。
  - 参数：`--fp16_model_dir`、`--quant_param_cache`、`--out_quant_dir`。
  - 示例：
    ```bash
    python scripts/fast_requantize_from_cache.py \
      --fp16_model_dir /path/to/fp16_dir \
      --quant_param_cache /path/to/quant_params.pt \
      --out_quant_dir /path/to/out_quant_dir
    ```

- `scripts/quantize_fp16_to_4bit_gptq.py`：
  - 作用：使用 `gptqmodel` 的标准流程把 FP16 checkpoint 量化为 4-bit GPTQ（含校准数据输入）。
  - 常用参数：`--fp16_model_dir`、`--out_quant_dir`、`--calib_text_file`、`--bits`、`--group_size`、`--desc_act`、`--sym` 等。
  - 示例：
    ```bash
    python scripts/quantize_fp16_to_4bit_gptq.py \
      --fp16_model_dir /path/to/fp16_dir \
      --out_quant_dir /path/to/out_quant_dir \
      --calib_text_file /path/to/wiki.train.raw
    ```

- `scripts/weight_only_quantize.py`：
  - 作用：基于 weight-only（RTN / per-channel）策略进行量化，支持更细粒度的 `scale/zero` 搜索与 activation-aware 权重。
  - 常用参数见脚本头部，支持导出 `quant_params.pt` 用于后续快速量化。
  - 示例（per-channel + activation-aware）：
    ```bash
    python scripts/weight_only_quantize.py \
      --fp16_model_dir /path/to/fp16_dir \
      --out_quant_dir /path/to/out_quant_dir \
      --calib_text_file /path/to/wiki.train.raw \
      --per_channel --act_aware
    ```

- `scripts/remove_pad_row.py`：
  - 作用：从 `.pth` 文件中删除/缩减最后一行（用于去除填充行以减小文件或兼容某些导出格式）。
  - 示例：
    ```bash
    python scripts/remove_pad_row.py --pth /path/to/model.pth --keys tok_embeddings.weight output.weight
    ```

- `scripts/inference.py`：
  - 作用：对量化后的 GPTQ 模型做简单文本生成测试。
  - 示例：
    ```bash
    python scripts/inference.py --model /path/to/quant_model_dir --prompt "Hello"
    ```

- `scripts/wikitext_ppl.py`：
  - 作用：在 WikiText（或本地文本文件）上评估模型 perplexity。
  - 示例（本地文本）：
    ```bash
    python scripts/wikitext_ppl.py --model /path/to/quant_model_dir --text_file /path/to/wiki.test.raw
    ```

---

## 环境依赖（简要）

- Python 3.10+
- PyTorch
- transformers
- gptqmodel

最小安装示例：

```bash
pip install gptqmodel transformers
```

---




