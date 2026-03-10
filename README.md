# Llama-2 GPTQ Re-Quantization Toolkit

一个面向 `Llama-2-7b-EfficientQAT-w2g64-GPTQ` 的实验仓库，目标是支持以下工作流：

1. 将已有的 `2-bit GPTQ` 模型导出为 `FP16`。
2. 基于 `FP16` 模型生成新的 `4-bit GPTQ` 模型。
3. 不落盘 `FP16`，直接把 `2-bit GPTQ` 模型重打包或重映射为 `4-bit GPTQ`。
3. 支持两类 4bit 量化方式：
   - 标准 `GPTQ` 量化
   - `weight-only RTN / per-channel` 量化
4. 将离线搜索得到的量化参数缓存下来，后续对同一份 `FP16` 权重做快速重新量化。
5. 通过文本生成与 WikiText PPL 对量化效果做验证。

---

## 仓库结构

- [scripts/export_2bit_gptq_to_fp16.py](scripts/export_2bit_gptq_to_fp16.py)  
  将已有的 GPTQ 模型导出为 Hugging Face `FP16` checkpoint。

- [scripts/quantize_fp16_to_4bit_gptq.py](scripts/quantize_fp16_to_4bit_gptq.py)  
  使用 `gptqmodel` 的标准 GPTQ 流程，将 `FP16` 模型量化为 `4-bit GPTQ`。

- [scripts/direct_requantize_gptq.py](scripts/direct_requantize_gptq.py)  
  直接从已有 `2-bit GPTQ` 模型生成 `4-bit GPTQ`，支持三种模式：
  - `--direct_repack`：仅做容器重打包，保持原始整数码值与 `scale / zero-point`
  - `--direct_code_lift`：把 2bit 码值按整数比例嵌入到 4bit 码值空间
  - 基于搜索或缓存的直接重量化

- [scripts/weight_only_quantize.py](scripts/weight_only_quantize.py)  
  自定义 `weight-only` 量化脚本，支持：
  - RTN 风格量化
  - `per-channel` 量化
  - `scale / zero-point` 搜索
  - activation-aware weighted search
  - 保存 `quant_params.pt` 以支持后续快速重量化

- [scripts/fast_requantize_from_cache.py](scripts/fast_requantize_from_cache.py)  
  从已保存的 `quant_params.pt` 快速重新量化同一份 `FP16` 权重。

- [scripts/inference.py](scripts/inference.py)  
  对量化后的 GPTQ 模型做简单生成测试。

- [scripts/wikitext_ppl.py](scripts/wikitext_ppl.py)  
  在 WikiText 文本上评估模型 perplexity。

- [data/wiki.test.raw](data/wiki.test.raw)  
  用于 PPL 评估或校准实验的纯文本文件。

- [artifacts](artifacts)  
  用于保存 `quant_params.pt` 等中间缓存产物。

- `models/source/`  
  输入模型目录，通常放原始 GPTQ 模型。

- `models/output/`  
  输出模型目录，通常放导出的 `FP16` 模型和后续量化结果。

---

## 环境依赖

建议环境：

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- `gptqmodel`
- CUDA 环境（推荐）

最小安装示例：

```bash
pip install gptqmodel
```

如需固定依赖，也可以参考 [requirements.txt](requirements.txt)。

---

## 快速开始

### 1. 将 2-bit GPTQ 模型导出为 FP16

使用 [scripts/export_2bit_gptq_to_fp16.py](scripts/export_2bit_gptq_to_fp16.py) 可以把已有的 GPTQ 模型反量化并导出为 Hugging Face 格式的 `FP16` checkpoint。

基础用法：

```bash
python scripts/export_2bit_gptq_to_fp16.py \
  --in_quant_dir /path/to/models/source/Llama-2-7b-EfficientQAT-w2g64-GPTQ \
  --out_fp16_dir /path/to/models/output/Llama-2-7b-fp16-from-2bit
```

如果加载模型时需要开启 `remote code`：

```bash
python scripts/export_2bit_gptq_to_fp16.py \
  --in_quant_dir /path/to/models/source/Llama-2-7b-EfficientQAT-w2g64-GPTQ \
  --out_fp16_dir /path/to/models/output/Llama-2-7b-fp16-from-2bit \
  --trust_remote_code
```

主要参数说明：

- `--in_quant_dir`：输入的 GPTQ 模型目录，支持本地路径或 Hugging Face repo id。
- `--out_fp16_dir`：导出的 `FP16` 模型保存目录。
- `--trust_remote_code`：加载配置或模型时启用 `trust_remote_code`。

脚本运行成功后会打印：

```text
exported_fp16: /path/to/output
```

### 2. 直接将 2-bit GPTQ 转成 4-bit GPTQ

如果目标是“磁盘上保留 2bit 模型，但部署或后端推理需要 4bit 容器格式”，推荐优先使用 [scripts/direct_requantize_gptq.py](scripts/direct_requantize_gptq.py)。

#### 2.1 容器重打包

这种模式只会：

- 解包原始 2bit 的 `qweight / qzeros`
- 重新按 4bit 容器格式打包
- 保持原始整数码值、`scales`、`g_idx` 不变

适合以下目标：

- 需要给只接受 4bit 容器格式的后端使用
- 但希望数值语义严格保持原始 2bit GPTQ 模型

```bash
python scripts/direct_requantize_gptq.py \
  --in_quant_dir /path/to/models/source/Llama-3-8b-2bit-GPTQ \
  --out_quant_dir /path/to/models/output/Llama-3-8b-4bit-repack \
  --direct_repack
```

#### 2.2 整数码值提升

这种模式会把 2bit 码值与零点映射到 4bit 码值空间，同时按比例调整 `scale`。

```bash
python scripts/direct_requantize_gptq.py \
  --in_quant_dir /path/to/models/source/Llama-3-8b-2bit-GPTQ \
  --out_quant_dir /path/to/models/output/Llama-3-8b-4bit-code-lift \
  --direct_code_lift
```

#### 2.3 直接搜索新的 4bit 参数

如果希望不经过落盘 `FP16`，但仍然重新搜索新的 4bit `scale / zero-point`，可以直接运行：

```bash
python scripts/direct_requantize_gptq.py \
  --in_quant_dir /path/to/models/source/Llama-3-8b-2bit-GPTQ \
  --out_quant_dir /path/to/models/output/Llama-3-8b-4bit-direct-search \
  --group_size 64 \
  --asym
```

如需复用已有的 `quant_params.pt`：

```bash
python scripts/direct_requantize_gptq.py \
  --in_quant_dir /path/to/models/source/Llama-3-8b-2bit-GPTQ \
  --out_quant_dir /path/to/models/output/Llama-3-8b-4bit-from-cache \
  --requant_from_cache /path/to/artifacts/quant_params.pt
```

输出目录中会包含：

- 新的 4bit GPTQ 模型文件
- `quantize_config.json`
- `quant_params.pt`（当不是 cache 模式时）

### 3. 使用标准 GPTQ 方法量化为 4bit

```bash
python scripts/quantize_fp16_to_4bit_gptq.py \
  --fp16_model_dir /path/to/models/output/Llama-2-7b-fp16-from-2bit \
  --out_quant_dir /path/to/models/output/Llama-2-7b-GPTQ-4bit \
  --calib_text_file /path/to/data/wiki.test.raw
```

常用可选参数：

- `--bits 4`
- `--group_size 128`
- `--desc_act`
- `--sym` 或 `--no_sym`
- `--true_sequential` 或 `--no_true_sequential`

### 4. 使用 `weight-only` 方法量化为 4bit

基础示例：

```bash
python scripts/weight_only_quantize.py \
  --fp16_model_dir /path/to/models/output/Llama-2-7b-fp16-from-2bit \
  --out_quant_dir /path/to/models/output/_weightonly-4bit \
  --asym
```

`per-channel + activation-aware` 示例：

```bash
python scripts/weight_only_quantize.py \
  --fp16_model_dir /path/to/models/output/Llama-2-7b-fp16-from-2bit \
  --out_quant_dir /path/to/models/output/_weightonly-4bit-perchannel-actaware \
  --calib_text_file /path/to/data/wiki.test.raw \
  --calib_seq_len 512 \
  --calib_num_samples 8 \
  --per_channel \
  --asym \
  --mse 2.0 \
  --mse_grid 128 \
  --maxshrink 0.7 \
  --clip_ratio 0.98 \
  --clip_search_grid 11 \
  --refine_scale_grid 9 \
  --refine_scale_range 0.15 \
  --refine_zero_radius 6 \
  --refine_rounds 3 \
  --act_aware \
  --act_aware_alpha 1.0
```

量化完成后，输出目录中通常会包含：

- GPTQ 格式模型文件
- `quant_params.pt`

### 5. 从缓存参数快速重新量化

```bash
python scripts/fast_requantize_from_cache.py \
  --fp16_model_dir /path/to/models/output/Llama-2-7b-fp16-from-2bit \
  --quant_param_cache /path/to/artifacts/quant_params.pt \
  --out_quant_dir /path/to/models/output/_weightonly-4bit-fast
```

这个流程适用于：

- 原始 `FP16` 权重保持不变
- 离线已完成参数搜索
- 在线只想快速量化到 `4bit`

### 6. 文本生成测试

```bash
python scripts/inference.py \
  --model /path/to/models/output/_weightonly-4bit-fast \
  --prompt "The Large Language Model is"
```

如需采样生成，可额外传入：

```bash
python scripts/inference.py \
  --model /path/to/models/output/_weightonly-4bit-fast \
  --prompt "The Large Language Model is" \
  --do_sample \
  --temperature 0.8 \
  --top_p 0.95
```

### 7. 评估 WikiText PPL

使用本地文本文件评估：

```bash
python scripts/wikitext_ppl.py \
  --model /path/to/models/output/_weightonly-4bit-fast \
  --text_file /path/to/data/wiki.test.raw \
  --max_length 2048 \
  --stride 512
```

如果环境可访问 Hugging Face，也可以直接使用数据集：

```bash
python scripts/wikitext_ppl.py \
  --model /path/to/models/output/_weightonly-4bit-fast \
  --dataset wikitext \
  --subset wikitext-2-raw-v1 \
  --split test
```

---

## 推荐工作流

对于当前仓库，更推荐下面的实验顺序：

### 路线 A：标准 FP16 中转路线

1. 从 `2-bit GPTQ` 导出 `FP16`
2. 用 [scripts/weight_only_quantize.py](scripts/weight_only_quantize.py) 做一次高质量离线搜索
3. 保存 `quant_params.pt`
4. 后续部署阶段用 [scripts/fast_requantize_from_cache.py](scripts/fast_requantize_from_cache.py) 对同一份 `FP16` 权重做快速量化
5. 用 [scripts/inference.py](scripts/inference.py) 和 [scripts/wikitext_ppl.py](scripts/wikitext_ppl.py) 验证效果

### 路线 B：直接 GPTQ -> GPTQ 路线

1. 直接运行 [scripts/direct_requantize_gptq.py](scripts/direct_requantize_gptq.py)
2. 若目标是后端兼容性转换，优先使用 `--direct_repack`
3. 若目标是整数码值嵌入，使用 `--direct_code_lift`
4. 若目标是直接搜索新的 4bit 参数，则不加上述两个参数，或复用 `--requant_from_cache`
5. 再使用生成与 PPL 脚本验证结果

这条链路适合以下目标：

- 不修改 `FP16` 权重本身
- 基于离线缓存的量化参数快速得到 `4bit` 模型

直接 GPTQ -> GPTQ 路线适合以下目标：

- `tmac`、`qnn`、自定义 runtime 等只接受 4bit 容器格式
- 希望磁盘上仍保存 2bit GPTQ 原始模型
- 不想额外落盘完整 `FP16` checkpoint

---

## `weight_only_quantize.py` 的特点

相比标准 GPTQ 脚本，这个脚本提供了更多可控性：

- `per-channel` 量化
- 局部 `scale / zero-point` 细化搜索
- `quant_params.pt` 缓存导出
- activation-aware 搜索

---


