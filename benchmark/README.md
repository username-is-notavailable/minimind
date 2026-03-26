# MiniMind Benchmark — 量化评测框架

## 概述

本目录包含 MiniMind 模型在不同训练阶段的量化评测工具。所有评测脚本均为独立可运行，支持 GQA/MLA/MoE 全部模型变体。

---

## 目录结构

```
benchmark/
├── README.md              # 本说明文档
├── run_all.py             # 一键运行所有评测
├── eval_pretrain.py       # 预训练阶段评测（PPL）
├── eval_ceval.py          # C-Eval 中文学科评测（52科）
├── eval_generation.py     # 生成质量评测（自动评分）
├── eval_efficiency.py     # 推理效率评测（速度/显存）
└── results/               # 评测结果输出目录（JSON）
```

---

## 评测维度与适用阶段

| 评测脚本 | 评测维度 | 数据来源 | 适用阶段 | 核心指标 |
|---|---|---|---|---|
| `eval_pretrain.py` | 语言建模能力 | pretrain/SFT held-out 数据 | Pretrain, SFT | **Perplexity (PPL)** ↓ |
| `eval_ceval.py` | 知识与推理 | HuggingFace `ceval/ceval-exam`（52科） | SFT, DPO, RLHF | **准确率 (%)** ↑ |
| `eval_generation.py` | 生成质量 | 15 个固定 prompt（5 类能力） | SFT, DPO, RLHF | **综合评分 (0-100)** ↑ |
| `eval_efficiency.py` | 推理效率 | 合成输入 | 所有阶段 | **tokens/s**, **显存 (MB)** |

### 各训练阶段推荐评测组合

| 训练阶段 | 推荐评测 | 命令 |
|---|---|---|
| **Pretrain** | PPL | `python run_all.py --weight pretrain --pretrain_mode` |
| **SFT** | PPL + C-Eval + 生成 + 效率 | `python run_all.py --weight full_sft` |
| **DPO** | C-Eval + 生成 | `python run_all.py --weight dpo --tasks ceval,gen` |
| **GRPO/PPO** | C-Eval + 生成 | `python run_all.py --weight grpo --tasks ceval,gen` |

---

## 快速开始

```bash
cd minimind/benchmark/

# 一键评测 GQA baseline（SFT 后）
python run_all.py --weight full_sft

# 一键评测 MLA 模型
python run_all.py --weight full_sft --use_mla 1

# 仅运行 PPL 评测
python run_all.py --weight full_sft --tasks ppl

# 使用指定 Python 环境
python run_all.py --weight full_sft --python /home/aiscuser/.conda/envs/pre/bin/python
```

---

## 各评测详细说明

### 1. Perplexity 评测（`eval_pretrain.py`）

**原理**：在 held-out 数据上计算模型的交叉熵损失，转换为 PPL。PPL 越低表示模型越好地"理解"了文本分布。

**数据源**：使用训练数据文件的末尾 N 条作为 held-out 集（不影响训练过程）。

**指标**：
- `Perplexity (PPL)` — 困惑度，越低越好
- `Avg Loss` — 平均交叉熵损失

```bash
# 在预训练数据上评测
python eval_pretrain.py --weight pretrain --data_path ../dataset/pretrain_hq.jsonl

# 在 SFT 数据上评测
python eval_pretrain.py --weight full_sft --data_path ../dataset/sft_mini_512.jsonl

# 调整 held-out 样本数
python eval_pretrain.py --weight full_sft --max_samples 1000
```

**为什么重要**：PPL 是最公平的语言模型评测指标，不依赖于模型是否"知道"特定事实，直接衡量模型对文本分布的建模能力。

---

### 2. C-Eval 中文学科评测（`eval_ceval.py`）

**原理**：使用 HuggingFace 上的 [ceval/ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam) 标准化中文评测集，覆盖 **52 个学科**，4 大类别：
- **STEM**（17科）：数学、物理、化学、计算机等
- **社会科学**（11科）：经济、法律、政治等
- **人文**（9科）：历史、文学、哲学等
- **其他**（15科）：医学、会计、工程等

**评测方式**（概率法）：
1. 构造 prompt: `"问题：{question}\nA. xxx\nB. xxx\nC. xxx\nD. xxx\n答案："`
2. 取模型在最后一个 token 位置对 A/B/C/D 四个字母 token 的 logits
3. 选概率最高的作为模型预测答案
4. 与标准答案比较，计算准确率

**为什么用概率法而不是生成法**：小模型生成的答案格式不稳定（可能输出"答案是B"或"B选项正确"等），概率法更公平且与 lm-evaluation-harness 一致。

```bash
# val 集快速评测（每科 5-20 题）
python eval_ceval.py --weight full_sft --split val

# 指定科目
python eval_ceval.py --weight full_sft --subjects computer_network,operating_system

# 预训练模型（不使用 chat template）
python eval_ceval.py --weight pretrain --no_chat_template
```

**参考基线**：
- 随机猜测：25%
- MiniMind2 原作者在 C-Eval 上约 26.5%（接近随机，小模型正常表现）

---

### 3. 生成质量评测（`eval_generation.py`）

**原理**：使用 15 个固定 prompt（覆盖 5 类能力），让模型生成回复，通过多维度自动评分系统打分。

**5 类能力维度**：

| 能力 | Prompt 数 | 示例 |
|---|---|---|
| 事实问答 | 4 | "中国的首都是哪里？" |
| 科学解释 | 3 | "为什么天空是蓝色的？" |
| 逻辑推理 | 2 | "比较猫和狗作为宠物的优缺点" |
| 代码生成 | 2 | "用Python写斐波那契函数" |
| 创意写作 | 2 | "推荐一些中国的美食" |
| 自我认知 | 2 | "解释什么是机器学习" |

**自动评分系统（满分 100）**：

| 维度 | 满分 | 评分依据 |
|---|---|---|
| 长度适当性 | 20 | 回答长度是否合理（太短/太长扣分） |
| 关键词命中 | 40 | 预设关键词在回答中的出现比例 |
| 流畅度 | 20 | 重复片段检测（严重重复扣分） |
| 格式规范 | 20 | 是否有标点、分段、列表等结构 |

```bash
python eval_generation.py --weight full_sft
python eval_generation.py --weight dpo --use_mla 1
```

**输出**：每个 prompt 的评分 + 生成速度（tokens/s），按类别汇总平均分。

---

### 4. 推理效率评测（`eval_efficiency.py`）

**原理**：在不同输入长度下（16 到 512 tokens）测量模型的推理速度和显存占用。

**指标**：
- `throughput (tokens/s)` — 生成吞吐量
- `peak_memory (MB)` — GPU 显存峰值
- 不同输入长度下的性能变化曲线

```bash
python eval_efficiency.py --weight full_sft
python eval_efficiency.py --weight full_sft --use_mla 1 --gen_length 128
```

**为什么重要**：这是 MLA 与 GQA 对比的**核心指标**。MLA 通过 KV 压缩减少缓存大小，理论上在长序列推理时显存更低。

---

## 结果格式

所有结果保存为 JSON 到 `results/` 目录：

```
results/
├── pretrain_GQA_512.json
├── pretrain_MLA_512.json
├── ceval_full_sft_GQA_512_val.json
├── ceval_full_sft_MLA_512_val.json
├── generation_full_sft_GQA_512.json
├── generation_full_sft_MLA_512.json
├── efficiency_full_sft_GQA_512.json
├── efficiency_full_sft_MLA_512.json
└── summary_full_sft_GQA_512.json       # run_all.py 汇总
```

---

## GQA vs MLA 消融对比结果记录

> 以下表格在完成实验后填写

### 预训练阶段

| 指标 | GQA Baseline | MLA | 差异 |
|---|---|---|---|
| 参数量 (M) | | | |
| PPL (pretrain held-out) | | | |

### SFT 阶段

| 指标 | GQA Baseline | MLA | 差异 |
|---|---|---|---|
| PPL (SFT held-out) | | | |
| C-Eval 准确率 | | | |
| C-Eval STEM | | | |
| C-Eval 社会科学 | | | |
| C-Eval 人文 | | | |
| 生成评分 (avg) | | | |
| 生成-事实问答 | | | |
| 生成-科学解释 | | | |
| 生成-代码生成 | | | |
| 生成-创意写作 | | | |

### 推理效率

| 指标 | GQA Baseline | MLA | 差异 |
|---|---|---|---|
| 速度@input=32 (tok/s) | | | |
| 速度@input=128 (tok/s) | | | |
| 速度@input=512 (tok/s) | | | |
| 显存@input=32 (MB) | | | |
| 显存@input=128 (MB) | | | |
| 显存@input=512 (MB) | | | |

---

## 与 lm-evaluation-harness 的关系

本框架是**轻量化的内置评测**，直接加载原生 `.pth` 权重，不需要转换模型格式。

如需使用社区标准的 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)（支持更多 benchmark），需先将模型转换为 Transformers 格式：

```bash
# 1. 转换模型
cd scripts/
python convert_model.py --weight full_sft [--use_mla 1]

# 2. 使用 lm-evaluation-harness
pip install lm-eval
lm_eval --model hf \
    --model_args pretrained=../MiniMind2-Small,device=cuda,dtype=auto \
    --tasks ceval* \
    --batch_size 8 \
    --trust_remote_code
```

本框架的 C-Eval 评测与 lm-evaluation-harness 使用**相同的概率法**（logits对比），结果可直接比较。

---

## CLI 参数速查

所有脚本的公共参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--weight` | `full_sft` | 权重前缀 |
| `--save_dir` | `out` | 权重目录 |
| `--hidden_size` | `512` | 隐藏维度 |
| `--num_hidden_layers` | `8` | 层数 |
| `--use_mla` | `0` | 启用 MLA |
| `--use_moe` | `0` | 启用 MoE |
| `--mla_kv_dim` | `128` | MLA KV 维度 |
| `--mla_q_dim` | `256` | MLA Q 维度 |
| `--mla_rope_dim` | `128` | MLA RoPE 维度 |
| `--device` | `cuda` | 计算设备 |
