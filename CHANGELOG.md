# Changelog — 基于 MiniMind 原仓库的改动记录

本文档记录相对于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 原始仓库所做的所有修改和新增工作。

---

## 1. 新增 MLA（Multi-head Latent Attention）模块

### 1.1 背景与动机

MLA 是 DeepSeek-V2/V3 中提出的高效注意力机制，其核心思想是对 Q、K、V 进行**低秩压缩**，并采用**解耦式旋转位置编码（Decoupled RoPE）**，在显著减少 KV Cache 显存开销的同时保持注意力表达能力。

原始 MiniMind 仓库仅支持标准 GQA（Grouped Query Attention）和 MoE。本次改动将 MLA 机制完整集成到 MiniMind 架构中，作为可选的注意力变体。

### 1.2 改动文件清单

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `model/model_minimind.py` | **核心改动** | Config 新增 4 个参数；Attention 类新增 MLA 分支；MiniMindModel 调整 RoPE 维度 |
| `trainer/train_pretrain.py` | 参数新增 | 新增 `--use_mla`, `--mla_kv_dim`, `--mla_q_dim`, `--mla_rope_dim` CLI 参数 |
| `trainer/train_full_sft.py` | 参数新增 | 同上 |
| `trainer/train_lora.py` | 参数新增 | 同上 |
| `trainer/train_dpo.py` | 参数新增 | 同上 |
| `trainer/train_ppo.py` | 参数新增 | 同上 |
| `trainer/train_grpo.py` | 参数新增 | 同上 |
| `trainer/train_spo.py` | 参数新增 | 同上 |
| `trainer/train_reason.py` | 参数新增 | 同上 |

### 1.3 新增配置参数

在 `MiniMindConfig` 中新增以下参数：

```python
use_mla: bool = False      # 是否启用 MLA 注意力机制
mla_kv_dim: int = -1       # KV 压缩潜在空间维度（推荐 128）
mla_q_dim: int = -1        # Q 压缩潜在空间维度（推荐 256）
mla_rope_dim: int = -1     # Decoupled RoPE 维度（推荐 128）
```

### 1.4 MLA 架构设计

#### 标准注意力 vs MLA 对比

| 维度 | 标准 GQA | MLA |
|---|---|---|
| Q 来源 | x → W_q → Q | x → W_q_down → c_q → W_q_up → Q（低秩） |
| K/V 来源 | x → W_k/W_v → K/V | x → W_kv_down → c_kv → W_k_up/W_v_up → K/V（低秩）|
| RoPE 施加对象 | 整个 Q 和 K | 独立投影的 q_pe, k_pe（Decoupled） |
| QK 点积维度 | head_dim | head_dim + rope_dim |
| KV Cache 内容 | 完整 K + V 张量 | 压缩态 c_kv + 带位置编码的 k_pe |

#### 核心组件

**1) KV 压缩**：通过 `mla_kv_proj` 将 hidden_size 维的输入压缩到 mla_kv_dim 维的潜在空间 c_kv，再从 c_kv 分别上投影恢复 K 和 V。KV Cache 中仅存储压缩后的 c_kv，解码时按需重建。

**2) Q 压缩**：通过 `mla_q_proj` 将输入压缩到 mla_q_dim 维后再上投影得到 Q。

**3) Decoupled RoPE**：由于 K 从压缩态 c_kv 恢复，直接对其施加 RoPE 会导致位置信息与压缩内容纠缠，使缓存的 c_kv 在不同位置无法复用。因此从原始输入 x 额外投影出独立的 q_pe 和 k_pe，**仅对它们施加 RoPE**，然后拼接到内容向量：

```
Q_final = [Q_nope ‖ q_pe]    （维度 = head_dim + rope_dim）
K_final = [K_nope ‖ k_pe]    （维度 = head_dim + rope_dim）
```

**4) KV Cache 策略**：缓存 `(c_kv, k_pe)` 而非完整的 K 和 V。k_pe 必须独立缓存，因为它由原始 x 投影而来，无法从 c_kv 重构。

#### 新增的投影层（Attention 类）

| 投影层 | 维度变换 | 作用 |
|---|---|---|
| `mla_kv_proj` | hidden_size → mla_kv_dim | KV 下投影（压缩） |
| `mla_q_proj` | hidden_size → mla_q_dim | Q 下投影（压缩） |
| `q_rope_proj` | hidden_size → num_heads × rope_dim | Q 的解耦位置编码投影 |
| `k_rope_proj` | hidden_size → num_kv_heads × rope_dim | K 的解耦位置编码投影 |

### 1.5 使用方式

启用 MLA 训练示例：

```bash
cd trainer

# 预训练
python train_pretrain.py --use_mla 1 --mla_kv_dim 128 --mla_q_dim 256 --mla_rope_dim 128

# SFT
python train_full_sft.py --use_mla 1 --mla_kv_dim 128 --mla_q_dim 256 --mla_rope_dim 128
```

关闭 MLA（默认行为，与原仓库一致）：

```bash
python train_pretrain.py  # use_mla 默认为 False
```

---

## 2. 数据集下载实践记录

### 2.1 背景

README 中提供的数据集下载地址为 [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset) 和 [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)。实际下载时发现 HuggingFace 上的文件名与 README 中的说明已有更新。

### 2.2 文件名映射

| README 中的名称 | HuggingFace 实际文件名 | 大小 | 用途 |
|---|---|---|---|
| `pretrain_hq.jsonl` ✨ | `pretrain_t2t.jsonl` | ~7.8 GB | 预训练数据 |
| `sft_mini_512.jsonl` ✨ | `sft_t2t_mini.jsonl` | ~1.6 GB | SFT 微调数据 |
| `dpo.jsonl` ✨ | `dpo.jsonl` | ~52 MB | DPO 偏好对齐数据 |
| `rlaif-mini.jsonl` ✨ | `rlaif.jsonl` | ~23 MB | RLAIF 强化学习数据 |

> ✨ 标记为 README 中推荐的必须下载项。

### 2.3 下载方式

使用 `huggingface_hub` Python 包下载：

```python
from huggingface_hub import hf_hub_download

files = ['pretrain_t2t.jsonl', 'sft_t2t_mini.jsonl', 'dpo.jsonl', 'rlaif.jsonl']

for f in files:
    hf_hub_download(
        repo_id='jingyaogong/minimind_dataset',
        filename=f,
        repo_type='dataset',
        local_dir='./dataset',
    )
```

### 2.4 注意事项

- 训练脚本中引用的数据文件名可能仍使用 README 中的旧名称，已通过软链接方式解决（见下文第 3 节）。
- HuggingFace 上还有 `pretrain_t2t_mini.jsonl`（较小的预训练数据子集）和 `sft_t2t.jsonl`（完整 SFT 数据）等可选文件。
- 无需设置 HF_TOKEN 即可下载，但设置后可获得更高的下载速率。

---

## 3. 阶段零修复：训练前置问题处理

### 3.1 数据文件名不匹配修复

HuggingFace 上的文件名已更新，但训练脚本的默认 `data_path` 仍使用旧名称。通过创建软链接解决，无需修改任何训练脚本：

```bash
cd dataset/
ln -sf pretrain_t2t.jsonl pretrain_hq.jsonl
ln -sf sft_t2t_mini.jsonl sft_mini_512.jsonl
ln -sf rlaif.jsonl rlaif-mini.jsonl
```

修复后所有训练脚本的默认 `data_path` 均可正常读取数据：

| 脚本默认名称 | 软链接目标 | 状态 |
|---|---|---|
| `pretrain_hq.jsonl` | → `pretrain_t2t.jsonl` (7.8 GB) | ✅ |
| `sft_mini_512.jsonl` | → `sft_t2t_mini.jsonl` (1.6 GB) | ✅ |
| `dpo.jsonl` | 原名一致 (52 MB) | ✅ |
| `rlaif-mini.jsonl` | → `rlaif.jsonl` (23 MB) | ✅ |

### 3.2 为 eval_llm.py 添加 MLA 支持

原始 `eval_llm.py` 不支持 MLA 参数，无法加载和评估 MLA 训练的模型。新增以下改动：

**argparse 新增 4 个参数**（与训练脚本保持一致）：
```python
parser.add_argument('--use_mla', default=0, type=int, choices=[0, 1])
parser.add_argument('--mla_kv_dim', type=int, default=128)
parser.add_argument('--mla_q_dim', type=int, default=256)
parser.add_argument('--mla_rope_dim', type=int, default=128)
```

**MiniMindConfig 初始化**新增传递：
```python
use_mla=bool(args.use_mla),
mla_kv_dim=args.mla_kv_dim,
mla_q_dim=args.mla_q_dim,
mla_rope_dim=args.mla_rope_dim,
```

使用方式：
```bash
# 评估标准 GQA 模型（默认，与原仓库一致）
python eval_llm.py --weight full_sft

# 评估 MLA 模型
python eval_llm.py --weight full_sft --use_mla 1
```

---

## 4. 模型转换脚本改进：支持 MLA 和 CLI 参数

### 4.1 背景

原始 `scripts/convert_model.py` 存在两个问题：
- 所有配置硬编码在 `__main__` 中，切换不同模型变体需要手动修改代码
- 不支持 MLA 模型转换（MLA 架构与 Llama 不兼容，无法走 Llama 转换路径）

### 4.2 改动内容

**新增 argparse CLI 参数**，支持所有模型变体：

| 参数 | 说明 |
|---|---|
| `--hidden_size` | 隐藏层维度（512/640/768） |
| `--num_hidden_layers` | 层数（8/16） |
| `--use_moe` | 是否为 MoE 模型 |
| `--use_mla` | 是否为 MLA 模型 |
| `--mla_kv_dim/q_dim/rope_dim` | MLA 维度参数 |
| `--weight` | 权重名称前缀 |
| `--input_dir` | 输入权重目录 |
| `--output_dir` | 输出目录（默认自动生成） |
| `--direction` | 转换方向（t2t / t2torch） |

**自动路径选择**：
- 标准 GQA 模型 → `convert_torch2transformers_llama`（Llama 格式，兼容第三方生态）
- MLA / MoE 模型 → `convert_torch2transformers_minimind`（自定义架构，需 `trust_remote_code=True`）

**重构函数签名**：两个转换函数的 `lm_config` 从全局变量改为显式参数传入。

### 4.3 使用方式

```bash
cd scripts/

# 标准 GQA 模型 → Llama 格式
python convert_model.py --weight full_sft

# MLA 模型 → MiniMind 原生格式
python convert_model.py --weight full_sft --use_mla 1

# MoE 模型
python convert_model.py --weight full_sft --use_moe 1 --hidden_size 640

# 768 维 Base 模型
python convert_model.py --weight full_sft --hidden_size 768 --num_hidden_layers 16
```

### 4.4 两种上传路径说明

| | Llama 格式（路径 A） | MiniMind 原生格式（路径 B） |
|---|---|---|
| 适用模型 | 标准 GQA | MLA / MoE |
| 加载方式 | `AutoModelForCausalLM.from_pretrained(...)` | 需加 `trust_remote_code=True` |
| 第三方兼容 | llama.cpp / vllm / ollama ✅ | 仅 transformers |
| 上传内容 | config.json + pytorch_model.bin | 同左 + model_minimind.py（模型代码） |

---

## 5. 新增量化评估脚本 `eval_benchmark.py`

### 5.1 背景

原始仓库缺乏量化的模型评测手段，`eval_llm.py` 仅做交互式文本生成，训练脚本无 validation loop。无法量化对比不同模型变体（如 GQA vs MLA）的能力差异。

### 5.2 评估维度

| 任务代号 | 评估维度 | 指标 | 方法 |
|---|---|---|---|
| `ppl` | 语言建模能力 | Perplexity (PPL) | 在 SFT 数据 held-out 集上计算交叉熵 |
| `mcq` | 知识与推理 | 准确率 (%) | 20 道中文选择题，概率法取 A/B/C/D 的 logits |
| `gen` | 生成质量 | 综合评分 (0-100) | 关键词命中率 + 重复度检测 + 格式规范性 |
| `eff` | 推理效率 | tokens/s, 峰值显存 | 多种输入长度下的推理速度和显存 |

### 5.3 使用方式

```bash
# 全面评估
python eval_benchmark.py --weight full_sft --tasks all

# MLA 模型评估
python eval_benchmark.py --weight full_sft --use_mla 1 --tasks all

# 仅 PPL
python eval_benchmark.py --weight full_sft --tasks ppl

# 指定评估数据和样本数
python eval_benchmark.py --weight full_sft --eval_data ../dataset/sft_mini_512.jsonl --eval_samples 1000
```

### 5.4 结果保存

- 评估结果自动保存到 `eval_results/` 目录（JSON 格式）
- 生成详情（含完整回答文本）单独保存为 `*_generations.json`

### 5.5 改动文件

| 文件 | 类型 |
|---|---|
| `eval_benchmark.py` | **新增** |
| `.gitignore` | 已包含 `eval_results/` 忽略（通过 `out` 规则） |

---

## 6. 训练可视化工具替换：SwanLab → WandB

将原始仓库使用的 SwanLab（`import swanlab as wandb`）替换为原生 Weights & Biases。

### 改动文件

| 文件 | 改动 |
|---|---|
| 9 个训练脚本 | `import swanlab as wandb` → `import wandb` |
| `train_pretrain.py` | 额外删除硬编码的 `wandb.login(key=...)` |
| `trainer_utils.py` | `wandb.get_run()`（swanlab API）→ `wandb.run`（原生 wandb API） |

---

## 7. 完整 Benchmark 评测框架

### 7.1 背景

原始仓库缺乏系统化的量化评测，`eval_benchmark.py` 只是简单合并脚本。本次构建独立的 `benchmark/` 目录，使用 HuggingFace 开源评测数据集，按训练阶段设计不同的评测方案。

### 7.2 新增文件

| 文件 | 说明 |
|---|---|
| `benchmark/README.md` | 评测框架完整说明文档 |
| `benchmark/run_all.py` | 一键运行所有评测 |
| `benchmark/eval_pretrain.py` | PPL 评测（pretrain/SFT 阶段） |
| `benchmark/eval_ceval.py` | C-Eval 52 科中文选择题评测（HuggingFace 数据源） |
| `benchmark/eval_generation.py` | 15 prompt × 5 能力维度的生成质量自动评分 |
| `benchmark/eval_efficiency.py` | 推理速度 / 显存峰值 / 多输入长度测试 |

### 7.3 评测设计

| 维度 | 数据来源 | 指标 | 适用阶段 |
|---|---|---|---|
| 语言建模 | held-out 训练数据 | PPL ↓ | Pretrain, SFT |
| 知识推理 | `ceval/ceval-exam`（52科，HuggingFace） | 准确率 ↑ | SFT, DPO |
| 生成质量 | 15 个固定 prompt | 自动评分 (0-100) ↑ | SFT, DPO, RLHF |
| 推理效率 | 合成输入（多种长度） | tokens/s, 显存 MB | 所有阶段 |

### 7.4 使用方式

```bash
cd benchmark/
python run_all.py --weight full_sft              # GQA 全面评测
python run_all.py --weight full_sft --use_mla 1   # MLA 全面评测
python run_all.py --weight pretrain --pretrain_mode  # 预训练模型仅 PPL
```

详见 `benchmark/README.md`。

---

## 后续计划

- [ ] MLA 消融实验：对比 MLA 与标准 GQA 在 MiniMind2-Small (26M) 上的训练效果和推理效率
- [ ] KV Cache 显存节省量化测试
- [x] ~~将 MLA 支持扩展到 `eval_llm.py`~~（已完成）
- [x] ~~改进 `convert_model.py` 支持 MLA 和 CLI 参数~~（已完成）
- [x] ~~新增量化评估脚本 `eval_benchmark.py`~~（已完成）
- [x] ~~SwanLab → WandB 替换~~（已完成）
- [x] ~~构建完整 benchmark 评测框架~~（已完成）
- [ ] 将 MLA 支持扩展到 `train_distillation.py`
