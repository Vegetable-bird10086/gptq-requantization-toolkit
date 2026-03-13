# 从 GPTQ 模型到 QNN 二进制：完整实战教程（Llama2-7B）

本教程覆盖完整链路：

1. 从 GPTQ 模型读取/准备量化 encodings。
2. 将 GPTQ 权重量化参数填入官方 `llama_sha_*.encodings`。
3. 校验替换正确性。
4. 用官方 QAI Hub 导出链路编译并链接。
5. 下载最终 QNN 上下文二进制（`linked_model.bin`）。

> 适用对象：当前仓库中已验证的 `llama_v2_7b_chat`（4 分片）流程。  
> 关键思路：**官方激活 encodings + GPTQ 权重 encodings + 官方 compile/link**。

---

## 0. 先理解你将得到什么

最终产物是 QNN 可部署二进制（每个分片一个）：

- `linked_model.bin`（Part1）
- `linked_model.bin`（Part2）
- `linked_model.bin`（Part3）
- `linked_model.bin`（Part4）

它们会被下载到：

- `artifacts/llama_gptqfilled_w4a16_hub_downloads/`

这不是 Hugging Face GPTQ checkpoint 原样导出；而是桥接后的部署产物。

---

## 1. 前置条件

### 1.1 必备目录（建议）

- GPTQ 模型目录（包含 `model.encodings`）：
  - `/root/autodl-tmp/models/output/Llama-2-7b-4bit`
- 本地 FP16 模型目录（供官方导出脚本加载）：
  - `/root/autodl-tmp/models/output/Llama-2-7b-fp16-from-2bit`
- 官方 Llama2 分片 encodings 目录：
  - `/root/.qaihm/qai-hub-models/models/llama_v2_7b_chat/v1/config`

### 1.2 QAI Hub 配置

```bash
qai-hub configure --api_token <YOUR_TOKEN>
```

确认文件存在：

```bash
ls /root/.qai_hub/client.ini
```

### 1.3 依赖与环境

- Python 3.10+
- `qai_hub_models`、`qai_hub`
- 本仓库脚本可运行

---

## 2. 方案 A：一键全流程（推荐）

仓库根目录提供了一键脚本：

- `../run_export_pipeline.sh`

直接运行：

```bash
cd /root/autodl-tmp/gptq-requantization-toolkit
bash ../run_export_pipeline.sh
```

常用选项：

```bash
# 跳过填充与校验（用于重复导出）
SKIP_FILL=1 bash ../run_export_pipeline.sh

# 导出但不下载二进制
DO_DOWNLOAD=0 bash ../run_export_pipeline.sh
```

该脚本会自动做 4 件事：

1. 填充官方 encodings（GPTQ -> `llama_sha_*.encodings`）
2. 校验一致性（长度/公式/覆盖率）
3. 运行官方 export（全量 4 分片）
4. 下载 link job 产物到本地

---

## 3. 方案 B：手动分步（便于调试）

## 3.1 填充官方 encodings

```bash
cd /root/autodl-tmp/gptq-requantization-toolkit

python scripts/fill_gptq_into_official_llama2_shards.py \
  --gptq-dir /root/autodl-tmp/models/output/Llama-2-7b-4bit \
  --gptq-encodings /root/autodl-tmp/models/output/Llama-2-7b-4bit/model.encodings \
  --official-config-dir /root/.qaihm/qai-hub-models/models/llama_v2_7b_chat/v1/config \
  --out-dir /root/autodl-tmp/artifacts/llama2_official_filled \
  --mapping-mode structural \
  --device cpu \
  --clean
```

输出重点：

- `artifacts/llama2_official_filled/sha_0_merged/model.encodings` ... `sha_3_merged/model.encodings`
- `fill_summary.json`

## 3.2 校验替换正确性

```bash
python scripts/validate_filled_llama2_encodings.py \
  --official-config-dir /root/.qaihm/qai-hub-models/models/llama_v2_7b_chat/v1/config \
  --filled-dir /root/autodl-tmp/artifacts/llama2_official_filled \
  --out-report /root/autodl-tmp/artifacts/llama2_official_filled/validation_report.json
```

你应重点看：

- `all_ok: true`
- `mapped_modules_union: 224`
- 每个 shard 的 `mapping_failures=0`、`length_mismatch=0`、`formula_fail=0`

## 3.3 准备 export 需要的命名目录

官方导出脚本按 `llama_sha_0..3.encodings` 命名读取：

```bash
mkdir -p /root/autodl-tmp/artifacts/llama2_official_filled/as_llama_sha
for i in 0 1 2 3; do
  cp -f \
    /root/autodl-tmp/artifacts/llama2_official_filled/sha_${i}_merged/model.encodings \
    /root/autodl-tmp/artifacts/llama2_official_filled/as_llama_sha/llama_sha_${i}.encodings
done
```

## 3.4 全量导出（4 分片）

```bash
mkdir -p /root/autodl-tmp/tmp /root/autodl-tmp/build/llama_gptqfilled_w4a16_full

cd /root/autodl-tmp
QAIHM_STORE_ROOT=/root/autodl-tmp \
TMPDIR=/root/autodl-tmp/tmp \
QAIHM_LLAMA2_LOCAL_MODEL_DIR=/root/autodl-tmp/models/output/Llama-2-7b-fp16-from-2bit \
QAIHM_LLAMA2_LOCAL_ENCODINGS_DIR=/root/autodl-tmp/artifacts/llama2_official_filled/as_llama_sha \
PYTHONPATH=/root/autodl-tmp/ai-hub-models \
python -m qai_hub_models.models.llama_v2_7b_chat.export \
  --model-cache-mode disable \
  --skip-profiling \
  --skip-inferencing \
  --skip-summary \
  --skip-downloading \
  --output-dir /root/autodl-tmp/build/llama_gptqfilled_w4a16_full
```

说明：

- `QAIHM_STORE_ROOT=/root/autodl-tmp` 用于把临时文件放到数据盘，避免系统盘空间不足。
- 脚本会提交 8 个 compile job（4 个 part × prompt/token）和 4 个 link job。

## 3.5 下载 link 产物

最简单：复用一键脚本中的下载逻辑。  
你也可以按 job id 手动下载（Python API）：

```python
import qai_hub as hub
job = hub.get_job("<LINK_JOB_ID>")
model = job.get_target_model()
model.download("/path/to/output/linked_model")
```

建议目录：

- `/root/autodl-tmp/artifacts/llama_gptqfilled_w4a16_hub_downloads/`

---

## 4. 脚本做了什么（关键原理）

### 4.1 `fill_gptq_into_official_llama2_shards.py`

它内部调用 `merge_gptq_into_aimet_encodings.py`，按以下规则映射：

- `self_attn.q_proj/k_proj/v_proj` -> `*_sha.N.weight`（N=0..31，按通道切片）
- `self_attn.o_proj` -> `o_proj_conv.weight`
- `mlp.gate/up/down_proj` -> `*_conv.weight`

分片层偏移：

- shard0: offset 0
- shard1: offset 8
- shard2: offset 16
- shard3: offset 24

### 4.2 `merge_gptq_into_aimet_encodings.py`

对每个目标通道写入：

- `bitwidth`
- `is_symmetric`
- `scale`
- `offset`
- `min`
- `max`

并保持 `activation_encodings` 不变。

`min/max` 按 4bit 公式计算：

- `min = (0 - offset) * scale`
- `max = (15 - offset) * scale`

### 4.3 `validate_filled_llama2_encodings.py`

自动检查：

- 键集合一致
- 通道长度一致
- `min/max` 公式一致
- 跨分片覆盖率

---

## 5. 常见问题

### Q1：为什么单分片报告里有很多 unmatched？

因为单分片只覆盖 8 层；四分片联合才覆盖全部 32 层。看 `mapped_modules_union` 才是全局结论。

### Q2：这是不是“原生 GPTQ checkpoint 直接导出”？

不是。是“官方激活 encodings + GPTQ 权重 encodings + 官方 compile/link”的桥接导出。

### Q3：为什么要四分片？

`llama_v2_7b_chat` 官方实现固定拆成 4 part（32 层按 8 层分片），以满足编译与部署约束。

### Q4：换模型能直接复用吗？

通用流程可复用；但结构映射规则（命名、切片方式、分片策略）通常要重新适配。

---

## 6. 最终你应该看到的关键产物

- 填充后的 encodings（供导出）：
  - `artifacts/llama2_official_filled/as_llama_sha/llama_sha_0.encodings`
  - `.../llama_sha_1.encodings`
  - `.../llama_sha_2.encodings`
  - `.../llama_sha_3.encodings`
- 校验报告：
  - `artifacts/llama2_official_filled/validation_report.json`
- 导出日志：
  - `build/llama_gptqfilled_w4a16_full/export.log`
- 下载产物：
  - `artifacts/llama_gptqfilled_w4a16_hub_downloads/<link_job_id>/linked_model.bin`
  - `artifacts/llama_gptqfilled_w4a16_hub_downloads/download_summary.json`

如果这几项都在，且验证报告 `all_ok=true`，就说明从 GPTQ 到 QNN 二进制的链路已经打通。
