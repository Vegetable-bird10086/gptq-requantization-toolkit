# Llama-2 GPTQ Re-Quantization Toolkit

一个面向 `Llama-2-7b-EfficientQAT-w2g64-GPTQ` 的实验仓库，目标是支持以下工作流：

1. 将已有的 `2-bit GPTQ` 模型导出为 `FP16`。
2. 基于 `FP16` 模型生成新的 `4-bit GPTQ` 模型。
3. 支持两类 4bit 量化方式：
   - 标准 `GPTQ` 量化
   - `weight-only RTN / per-channel` 量化
4. 将离线搜索得到的量化参数缓存下来，后续对同一份 `FP16` 权重做快速重新量化。
5. 通过文本生成与 WikiText PPL 对量化效果做验证。

---

## 仓库建议

如果准备公开到 GitHub，建议按下面原则整理：

### 1. 不建议直接把大模型权重普通提交到 Git
当前仓库包含模型目录 `Llama-2-7b-EfficientQAT-w2g64-GPTQ/`。这类文件通常体积很大，并且可能受上游模型许可约束。

建议：
- 使用 **Git LFS** 管理 `.safetensors` 等大文件。
- 或只保留目录说明、下载脚本、模型来源说明，把权重放到 Hugging Face / Release / 私有对象存储。
- 在 README 中明确说明模型许可证和来源。

### 2. 建议补充 `.gitignore`
建议忽略：
- `__pycache__/`
- `*.pyc`
- 中间量化产物目录，例如 `_weightonly-*`、`Llama-2-7b-fp16-from-2bit/`、`Llama-2-7b-GPTQ-4bit/`
- 临时日志和实验输出

### 3. 建议把“输入模型”和“输出模型”分开
比较清晰的结构是：

- `scripts/`：脚本
- `data/`：如 `wiki.test.raw`
- `models/output/`：导出的 FP16 和量化结果
- `artifacts/`：`quant_params.pt` 等缓存

如果暂时不想重构目录，也可以先保留当前结构，但 README 里要讲清楚每个目录的角色。

### 4. 建议明确主工作流
这个仓库最有价值的部分不是单个脚本，而是完整链路：


当前仓库已经按这个思路整理为：

- [artifacts](artifacts)

---

  将已有的 GPTQ 模型导出为 Hugging Face FP16 checkpoint。

- [scripts/quantize_fp16_to_4bit_gptq.py](scripts/quantize_fp16_to_4bit_gptq.py)  
  使用 `gptqmodel` 的标准 GPTQ 流程，将 FP16 模型量化为 4bit GPTQ。

- [scripts/weight_only_quantize.py](scripts/weight_only_quantize.py)  
  自定义 `weight-only` 量化脚本，支持：
  - RTN 风格量化
  - `per-channel` 量化
  - `scale / zero-point` 搜索
  - activation-aware weighted search
  - 保存 `quant_params.pt` 以支持后续快速重量化
  对量化后的 GPTQ 模型做简单生成测试。
- [scripts/wikitext_ppl.py](scripts/wikitext_ppl.py)  
  在 WikiText 文本上评估模型 perplexity。

### 数据
- [data/wiki.test.raw](data/wiki.test.raw)  
  用于 PPL 评估或校准实验的纯文本文件。
### 模型目录

  仓库中附带的 GPTQ 模型目录，通常作为 `2-bit GPTQ -> FP16` 导出链路的起点。

---

## 环境依赖

建议环境：
- Python 3.10+
- PyTorch
- `gptqmodel`
- CUDA 环境（推荐）
一个最小安装示例：

```bash
pip install gptqmodel
```


---
## 快速开始

### 1. 将量化 GPTQ 模型导出为 FP16


### 2. 使用标准 GPTQ 方法量化为 4bit

```bash
python scripts/quantize_fp16_to_4bit_gptq.py \
  --fp16_model_dir /path/to/models/output/Llama-2-7b-fp16-from-2bit \
  --out_quant_dir /path/to/models/output/Llama-2-7b-GPTQ-4bit \
  --calib_text_file /path/to/data/wiki.test.raw
```


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

量化完成后，会在输出目录中保存：
- GPTQ 格式模型文件
- `quant_params.pt`（默认）

### 4. 从缓存参数快速重新量化

```bash
python scripts/fast_requantize_from_cache.py \
  --fp16_model_dir /path/to/models/output/Llama-2-7b-fp16-from-2bit \
  --quant_param_cache /path/to/artifacts/quant_params.pt \
  --out_quant_dir /path/to/models/output/_weightonly-4bit-fast
```

这个流程适用于：
- 原始 FP16 权重保持不变
- 离线已完成参数搜索
- 在线只想快速量化到 4bit

```bash
python scripts/inference.py \
  --model /path/to/models/output/_weightonly-4bit-fast \
  --prompt "The Large Language Model is"
```

### 6. 评估 WikiText PPL

```bash
  --max_length 2048 \
  --stride 512
```

---

## 推荐工作流

对于当前仓库，更推荐下面的实验顺序：

1. 从 `2-bit GPTQ` 导出 `FP16`
2. 用 [scripts/weight_only_quantize.py](scripts/weight_only_quantize.py) 做一次高质量离线搜索
3. 保存 `quant_params.pt`
4. 后续部署阶段用 [scripts/fast_requantize_from_cache.py](scripts/fast_requantize_from_cache.py) 对同一份 FP16 权重做快速量化
5. 用 [scripts/inference.py](scripts/inference.py) 和 [scripts/wikitext_ppl.py](scripts/wikitext_ppl.py) 验证效果

这条链路适合你的目标：
- 不修改 FP16 权重本身
- 再基于离线缓存的量化参数快速得到 4bit 模型

---

## `weight_only_quantize.py` 的特点

相比标准 GPTQ 脚本，这个脚本提供了更多可控性：

- `per-channel` 量化
- 局部 scale / zero-point 细化搜索
- `quant_params.pt` 缓存导出
- 进度显示


如果要在 GitHub 首页突出重点，建议在仓库简介里写成：
---
## 许可证与模型来源

如果公开仓库，建议补充：

- 本仓库脚本代码的许可证
- 上游模型许可证
- 是否允许再分发权重文件


---
- 增加 `requirements.txt` 或 `environment.yml`
- 增加 `examples/` 目录保存命令模板
- 增加 PPL/生成效果对比表

- PyTorch
- Hugging Face Transformers
- gptqmodel
3. 保存 `quant_params.pt`
