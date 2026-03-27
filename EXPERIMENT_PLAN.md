# MiniMind MLA 消融实验计划

## 实验目标

在 MiniMind2-Small (26M) 和 MiniMind2-1B (~988M) 两个规模上，对比标准 GQA（Grouped Query Attention）与自定义 MLA（Multi-head Latent Attention）注意力机制的训练效果和推理效率差异。

实验覆盖完整的 LLM 训练流程：环境配置 → 数据准备 → 预训练 → SFT → RLHF/RLAIF → 量化评估 → 模型转换与发布。

---

## 阶段一：环境配置

### 1.1 硬件环境

| 项目 | 配置 |
|---|---|
| GPU | 8 × NVIDIA A100-SXM4-80GB |
| CUDA | 12.8 |
| 分词器 | 已有（`model/tokenizer.json`，vocab_size=6400，无需重新训练） |

### 1.2 Python 环境

当前机器默认 Python 没有 torch，需使用 conda 环境：

```bash
conda activate pre  # torch 2.8.0
```

安装项目依赖：

```bash
cd minimind
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

核心依赖清单：

| 包名 | 最低版本 | 用途 |
|---|---|---|
| `torch` | ≥2.6.0 | 训练与推理 |
| `transformers` | ≥4.45.0 | 模型加载与转换 |
| `datasets` | ≥3.0.0 | HuggingFace 数据集（C-Eval 评测） |
| `huggingface_hub` | ≥0.20.0 | 数据/模型下载上传 |
| `wandb` | ≥0.18.0 | 训练曲线记录 |

验证 GPU 可用性：

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
```

### 1.3 训练可视化（WandB）

已将原仓库的 SwanLab 替换为原生 [Weights & Biases](https://wandb.ai/)。

```bash
pip install wandb -q
wandb login
# 输入你的 API Key（从 https://wandb.ai/authorize 获取）
```

所有训练脚本添加 `--use_wandb` 即可启用记录。自动记录的指标：

| 指标名 | 含义 |
|---|---|
| `loss` | 总损失（含 aux_loss） |
| `logits_loss` | 语言建模损失 |
| `aux_loss` | MoE 负载均衡损失（非 MoE 模型为 0） |
| `learning_rate` | 当前学习率 |
| `epoch_time` | 预估剩余训练时间（分钟） |

WandB 项目命名规范（便于对比）：

| 实验 | `--wandb_project` 参数 |
|---|---|
| GQA Baseline 预训练 | `"MiniMind-Pretrain-GQA"` |
| GQA Baseline SFT | `"MiniMind-SFT-GQA"` |
| MLA 预训练 | `"MiniMind-Pretrain-MLA"` |
| MLA SFT | `"MiniMind-SFT-MLA"` |
| DPO | `"MiniMind-DPO-GQA"` / `"MiniMind-DPO-MLA"` |
| GRPO | `"MiniMind-GRPO-GQA"` / `"MiniMind-GRPO-MLA"` |

### 1.4 目录结构约定

```
minimind/                    # 项目根目录
├── model/                   # 模型定义 + 分词器
├── trainer/                 # 训练脚本（在此目录下运行训练）
├── dataset/                 # 数据集文件（需下载）
├── benchmark/               # 评测脚本（在此目录下运行评测）
├── scripts/                 # 转换脚本（在此目录下运行转换）
├── out/                     # 训练输出的权重文件
├── eval_llm.py              # 交互式评估（在项目根目录运行）
└── eval_benchmark.py        # 快速评估（在项目根目录运行）
```

**脚本运行目录约定**（重要，路径依赖于此）：

| 脚本类型 | 运行目录 | 数据路径前缀 |
|---|---|---|
| 训练脚本 `train_*.py` | `minimind/trainer/` | `../dataset/` |
| `eval_llm.py` | `minimind/` | `./dataset/` |
| benchmark 脚本 | `minimind/benchmark/` | `../dataset/` |
| `convert_model.py` | `minimind/scripts/` | `../out/` |

---

## 阶段二：数据准备

### 2.1 原作者数据集下载（26M / 104M 模型训练用）

数据集托管在 HuggingFace 和 ModelScope 上。

**下载地址**：
- HuggingFace: https://huggingface.co/datasets/jingyaogong/minimind_dataset
- ModelScope: https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files

**方式一：Python 脚本下载（推荐）**

```python
from huggingface_hub import hf_hub_download
import os

os.makedirs('./dataset', exist_ok=True)

# 必须下载的 4 个核心文件
core_files = {
    'pretrain_t2t.jsonl':   '预训练数据 (~7.8GB)',
    'sft_t2t_mini.jsonl':   'SFT 微调数据 (~1.6GB)',
    'dpo.jsonl':            'DPO 偏好对齐数据 (~52MB)',
    'rlaif.jsonl':          'RLAIF 强化学习数据 (~23MB)',
}

for filename, desc in core_files.items():
    print(f'Downloading {filename} — {desc}')
    hf_hub_download(
        repo_id='jingyaogong/minimind_dataset',
        filename=filename,
        repo_type='dataset',
        local_dir='./dataset',
    )
    print(f'  Done: {filename}')
```

**方式二：huggingface-cli 命令行下载**

```bash
# 下载全部文件
huggingface-cli download jingyaogong/minimind_dataset --repo-type dataset --local-dir ./dataset

# 或逐个下载
huggingface-cli download jingyaogong/minimind_dataset pretrain_t2t.jsonl --repo-type dataset --local-dir ./dataset
huggingface-cli download jingyaogong/minimind_dataset sft_t2t_mini.jsonl --repo-type dataset --local-dir ./dataset
huggingface-cli download jingyaogong/minimind_dataset dpo.jsonl --repo-type dataset --local-dir ./dataset
huggingface-cli download jingyaogong/minimind_dataset rlaif.jsonl --repo-type dataset --local-dir ./dataset
```

**方式三：ModelScope 下载（国内网络更快）**

```bash
pip install modelscope
modelscope download --dataset gongjy/minimind_dataset --local_dir ./dataset
```

**可选的扩展数据文件**（如需更充分训练）：

| HF 文件名 | 大小 | 用途 |
|---|---|---|
| `pretrain_t2t_mini.jsonl` | ~1.6GB | 精简预训练数据子集 |
| `sft_t2t.jsonl` | ~5.5GB | 完整 SFT 数据 |
| `sft_1024.jsonl` | ~5.6GB | Qwen2.5 蒸馏数据（长度 ≤1024） |
| `sft_2048.jsonl` | ~9GB | Qwen2.5 蒸馏数据（长度 ≤2048） |
| `r1_mix_1024.jsonl` | ~340MB | DeepSeek-R1 蒸馏数据 |
| `lora_identity.jsonl` | ~23KB | 自我认知数据（LoRA 训练用） |
| `lora_medical.jsonl` | ~34MB | 医疗问答数据（LoRA 训练用） |

### 2.2 文件名映射（软链接）

HuggingFace 上的实际文件名与训练脚本默认的 `data_path` 参数不一致，需要创建软链接：

```bash
cd minimind/dataset/

# 创建软链接（不修改任何训练脚本）
ln -sf pretrain_t2t.jsonl pretrain_hq.jsonl
ln -sf sft_t2t_mini.jsonl sft_mini_512.jsonl
ln -sf rlaif.jsonl rlaif-mini.jsonl
# dpo.jsonl 文件名一致，无需处理
```

**映射关系**：

| 训练脚本默认名称 | 软链接目标（HF 实际文件名） | 大小 |
|---|---|---|
| `pretrain_hq.jsonl` | → `pretrain_t2t.jsonl` | ~7.8GB |
| `sft_mini_512.jsonl` | → `sft_t2t_mini.jsonl` | ~1.6GB |
| `rlaif-mini.jsonl` | → `rlaif.jsonl` | ~23MB |
| `dpo.jsonl` | 原名一致 | ~52MB |

### 2.3 数据格式说明

各阶段数据使用不同的 JSONL 格式，每行一个 JSON 对象：

**预训练数据**（`pretrain_hq.jsonl`）：
```json
{"text": "如何才能摆脱拖延症？ 治愈拖延症并不容易，但以下建议可能有所帮助..."}
```

**SFT / RLAIF 数据**（`sft_mini_512.jsonl` / `rlaif-mini.jsonl`）：
```json
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
```

**DPO 数据**（`dpo.jsonl`）：
```json
{
    "chosen": [
        {"content": "Q", "role": "user"},
        {"content": "good answer", "role": "assistant"}
    ],
    "rejected": [
        {"content": "Q", "role": "user"},
        {"content": "bad answer", "role": "assistant"}
    ]
}
```

### 2.4 1B 模型扩展数据准备

现有数据约 0.9B tokens，训练 1B 模型会严重过拟合（仅为 Chinchilla 最优量的 4.5%）。需要将预训练数据扩充到 **5-10B tokens**。

**推荐数据源**（HuggingFace 开源中文语料）：

| 数据集 | HuggingFace ID | 规模 | 说明 |
|---|---|---|---|
| WanJuan-CC | `opendatalab/WanJuan-CC` | ~100B tokens | 清洗后的中文网页数据（取子集即可） |
| SkyPile | `Skywork/SkyPile-150B` | ~150B tokens | 中文网页（取前 10GB 即可） |
| Chinese-Web-Text | `CASIA-LM/ChineseWebText` | ~50B tokens | 高质量中文网络文本 |

**推荐数据配比**（以 10B tokens 为目标）：

| 数据类型 | 占比 | Tokens (估) | 来源 |
|---|---|---|---|
| 中文网页文本 | 45% | ~4.5B | WanJuan-CC + SkyPile |
| 百科/知识类 | 15% | ~1.5B | Wikipedia 中文 + 现有 pretrain_hq |
| 书籍/文学 | 15% | ~1.5B | Chinese-Web-Text 筛选 |
| 对话/问答 | 10% | ~1.0B | 现有 sft_t2t_mini |
| 代码 | 5% | ~0.5B | StarCoder 中文子集 |
| 新闻/时事 | 5% | ~0.5B | WanJuan-CC 新闻子集 |
| 学术/专业 | 5% | ~0.5B | Chinese-Web-Text 筛选 |

**下载与合并流程**：

```bash
cd minimind

# 1. 下载额外数据（以 WanJuan-CC 为例，取子集）
python -c "
from datasets import load_dataset
import json, os

os.makedirs('./dataset', exist_ok=True)
ds = load_dataset('opendatalab/WanJuan-CC', split='train', streaming=True)

with open('./dataset/pretrain_extra.jsonl', 'w') as f:
    for i, item in enumerate(ds):
        if i >= 5_000_000:  # 取 500 万条
            break
        f.write(json.dumps({'text': item['content']}, ensure_ascii=False) + '\n')
print(f'Downloaded {i+1} items')
"

# 2. 合并为 1B 预训练数据
cat ./dataset/pretrain_hq.jsonl ./dataset/pretrain_extra.jsonl > ./dataset/pretrain_1b.jsonl

# 3. 查看数据量
wc -l ./dataset/pretrain_1b.jsonl
```

> 注意：下载大规模数据可能需要较长时间，建议提前准备。数据格式必须与原始预训练数据一致（每行一个 `{"text": "..."}` JSON 对象）。

### 2.5 1B 专用分词器训练（可选）

原始 6400 词表为 26M 模型设计。1B 模型中 embedding 层仅占 1.3%，词表大小不再是瓶颈。使用 32K 词表可显著提升编码效率：

| | 6400 词表 | 32000 词表 |
|---|---|---|
| embed 占比 (1B) | 1.3% | 6.6% |
| 中文编码效率 | ~1.6 字符/token | ~2.5-3.0 字符/token |
| 512 tokens 覆盖 | ~820 字符 | ~1400 字符 |

```bash
cd minimind/trainer
python train_tokenizer_1b.py \
    --data_path ../dataset/pretrain_1b.jsonl \
    --vocab_size 32000 \
    --max_lines 200000 \
    --output_dir ../model_1b_tokenizer
```

> **注意**：使用新分词器训练的 1B 模型与 6400 词表的 26M 模型 **PPL 不直接可比**。如需公平消融对比 MLA，应在同一分词器下对比 GQA vs MLA。

### 2.6 Reward Model 下载（GRPO/PPO 阶段使用）

GRPO 和 PPO 训练需要外部奖励模型。推荐使用 InternLM2-1.8B-Reward（约 3.6GB）：

```bash
cd ..  # 回到 minimind 的上级目录

# HuggingFace 下载
git clone https://huggingface.co/internlm/internlm2-1_8b-reward

# 或 ModelScope 下载（国内更快）
git clone https://www.modelscope.cn/Shanghai_AI_Laboratory/internlm2-1_8b-reward.git
```

最终目录结构：

```
project/
├── minimind/                    # MiniMind 项目
└── internlm2-1_8b-reward/       # 奖励模型（与 minimind 同级）
    ├── config.json
    ├── model.safetensors
    └── ...
```

### 2.7 验证数据就绪

```bash
cd minimind/dataset/

# 确认所有文件就绪
ls -lh pretrain_hq.jsonl sft_mini_512.jsonl dpo.jsonl rlaif-mini.jsonl

# 验证数据格式（检查前 2 行）
head -n 2 pretrain_hq.jsonl | python -m json.tool --no-ensure-ascii
head -n 2 sft_mini_512.jsonl | python -m json.tool --no-ensure-ascii
head -n 2 dpo.jsonl | python -m json.tool --no-ensure-ascii
head -n 2 rlaif-mini.jsonl | python -m json.tool --no-ensure-ascii
```

---

## 阶段三：26M 模型 Baseline 训练（标准 GQA）

### 3.1 模型规格速查表

| 模型名称 | 参数量 | `--hidden_size` | `--num_hidden_layers` | 权重文件名 |
|---|---|---|---|---|
| MiniMind2-Small | **26M** | `512`（默认） | `8`（默认） | `*_512.pth` |
| MiniMind2-Base | **104M** | `768` | `16` | `*_768.pth` |
| MiniMind2-1B | **~988M** | `2048` | `22` | `*_2048.pth` |

**原则**：训练时用什么参数，后续加载/评估/转换时必须用完全相同的参数。

### 3.2 GQA Baseline 模型配置

| 参数 | 值 |
|---|---|
| hidden_size | 512 |
| num_hidden_layers | 8 |
| num_attention_heads | 8 |
| num_key_value_heads | 2 |
| vocab_size | 6400 |
| use_mla | False |
| use_moe | False |

### 3.3 预训练

```bash
conda activate pre
cd minimind/trainer

torchrun --nproc_per_node 4 train_pretrain.py \
    --use_wandb --wandb_project "MiniMind-Pretrain-GQA"
```

关键训练参数（均为默认值，无需额外指定）：

| 参数 | 值 |
|---|---|
| `data_path` | `../dataset/pretrain_hq.jsonl`（7.8GB） |
| `max_seq_len` | 340 |
| `batch_size` | 32 |
| `learning_rate` | 5e-4 |
| `accumulation_steps` | 8 |
| `dtype` | bfloat16 |
| `epochs` | 1 |
| `save_interval` | 1000 步 |

- **预计耗时**：~20min（4×A100）
- **产出**：`out/pretrain_512.pth`

### 3.4 SFT 微调

```bash
torchrun --nproc_per_node 4 train_full_sft.py \
    --use_wandb --wandb_project "MiniMind-SFT-GQA"
```

关键参数差异（与预训练对比）：

| 参数 | 预训练 | SFT |
|---|---|---|
| `data_path` | `pretrain_hq.jsonl` | `sft_mini_512.jsonl`（1.6GB） |
| `batch_size` | 32 | 16 |
| `learning_rate` | 5e-4 | 1e-6 |
| `accumulation_steps` | 8 | 1 |
| `epochs` | 1 | 2 |
| `from_weight` | none | `pretrain`（自动加载 `pretrain_512.pth`） |

- **预计耗时**：~15min（4×A100）
- **产出**：`out/full_sft_512.pth`

### 3.5 评估

```bash
cd minimind  # 回到项目根目录
python eval_llm.py --weight full_sft --show_speed 1
```

- 选择 `[0] 自动测试`，固定种子 2026
- 记录 8 个预设 prompt 的输出文本和 tokens/s

### 3.6 备份权重

```bash
mkdir -p out/baseline
cp out/pretrain_512.pth out/baseline/
cp out/full_sft_512.pth out/baseline/
```

> **重要**：备份必须在阶段四开始前完成，否则 MLA 训练将覆盖同名文件。

---

## 阶段四：26M 模型 MLA 训练

### 4.1 MLA 模型配置

在 GQA Baseline 基础上，额外启用 MLA 注意力：

| 参数 | 值 | 说明 |
|---|---|---|
| **use_mla** | **True** | 启用 MLA |
| **mla_kv_dim** | **128** | KV 压缩潜在空间维度 |
| **mla_q_dim** | **256** | Q 压缩潜在空间维度 |
| **mla_rope_dim** | **128** | Decoupled RoPE 维度 |

MLA 模型的 CLI 参数模板（所有训练脚本通用）：

```
--use_mla 1 --mla_kv_dim 128 --mla_q_dim 256 --mla_rope_dim 128
```

### 4.2 预训练

```bash
cd minimind/trainer

torchrun --nproc_per_node 4 train_pretrain.py \
    --use_mla 1 --mla_kv_dim 128 --mla_q_dim 256 --mla_rope_dim 128 \
    --use_wandb --wandb_project "MiniMind-Pretrain-MLA"
```

- 其余超参与阶段三完全一致
- **产出**：`out/pretrain_512.pth`（会覆盖旧文件，需先完成步骤 3.6 备份）

### 4.3 SFT 微调

```bash
torchrun --nproc_per_node 4 train_full_sft.py \
    --use_mla 1 --mla_kv_dim 128 --mla_q_dim 256 --mla_rope_dim 128 \
    --use_wandb --wandb_project "MiniMind-SFT-MLA"
```

### 4.4 评估

```bash
cd minimind
python eval_llm.py --weight full_sft --use_mla 1 --show_speed 1
```

### 4.5 备份权重

```bash
mkdir -p out/mla
cp out/pretrain_512.pth out/mla/
cp out/full_sft_512.pth out/mla/
```

---

## 阶段五：26M 消融对比评测

使用 `benchmark/` 目录下的评测框架，对阶段三和阶段四产出的两组模型进行系统化量化对比。

### 5.1 运行量化评测

```bash
cd minimind/benchmark

# GQA Baseline 全面评估
python run_all.py --weight full_sft --save_dir ../out/baseline

# MLA 全面评估
python run_all.py --weight full_sft --save_dir ../out/mla --use_mla 1
```

### 5.2 评测维度

| 任务 | 数据来源 | 指标 | 说明 |
|---|---|---|---|
| `ppl` | SFT held-out 数据 | Perplexity ↓ | 语言建模能力 |
| `ceval` | HuggingFace `ceval/ceval-exam`（52 科） | 准确率 ↑ | 知识与推理（概率法取 A/B/C/D logits） |
| `gen` | 15 个固定 prompt（5 类能力） | 综合评分 (0-100) ↑ | 关键词命中 + 重复度 + 格式规范 |
| `eff` | 合成输入（多种长度） | tokens/s, 显存 MB | 推理效率 |

结果自动保存到 `eval_results/` 目录（JSON 格式）。

### 5.3 对话质量对比

使用以下 8 个固定 prompt，种子固定 2026：

1. 你有什么特长？
2. 为什么天空是蓝色的
3. 请用Python写一个计算斐波那契数列的函数
4. 解释一下"光合作用"的基本过程
5. 如果明天下雨，我应该如何出门
6. 比较一下猫和狗作为宠物的优缺点
7. 解释什么是机器学习
8. 推荐一些中国的美食

评估方式：
- 人工主观评分（准确性、完整性、逻辑性）
- 可选：将两组输出提交给 GPT-4/DeepSeek-R1 进行盲评打分

### 5.4 指标汇总模板

| 指标 | GQA Baseline | MLA |
|---|---|---|
| 总参数量 | ____M | ____M |
| 预训练最终 Loss | ____ | ____ |
| SFT 最终 Loss | ____ | ____ |
| Perplexity (PPL) ↓ | ____ | ____ |
| C-Eval 准确率 ↑ | ____% | ____% |
| 生成质量评分 ↑ | ____/100 | ____/100 |
| 推理速度 (tokens/s) | ____ | ____ |
| 推理显存峰值 (MB) | ____ | ____ |

---

## 阶段六：RLHF / RLAIF 后训练

基于阶段五中表现更好的模型变体（或两者均训练），继续后训练。

### 6.1 DPO（离线偏好优化）

```bash
cd minimind/trainer

# GQA 模型 DPO
torchrun --nproc_per_node 4 train_dpo.py \
    --use_wandb --wandb_project "MiniMind-DPO-GQA"

# MLA 模型 DPO（补充 MLA 参数）
torchrun --nproc_per_node 4 train_dpo.py \
    --use_mla 1 --mla_kv_dim 128 --mla_q_dim 256 --mla_rope_dim 128 \
    --use_wandb --wandb_project "MiniMind-DPO-MLA"
```

DPO 关键参数（均为默认值）：

| 参数 | 值 |
|---|---|
| `data_path` | `../dataset/dpo.jsonl`（52MB） |
| `from_weight` | `full_sft`（自动加载 `full_sft_512.pth`） |
| `batch_size` | 4 |
| `learning_rate` | 4e-8 |
| `max_seq_len` | 1024 |
| `beta` | 0.1（KL 惩罚系数） |
| `save_interval` | 100 步 |

- **产出**：`out/dpo_512.pth`

### 6.2 GRPO（在线强化学习）

需要提前完成 2.6 节的 Reward Model 下载。

```bash
# GQA 模型 GRPO
torchrun --nproc_per_node 4 train_grpo.py \
    --use_wandb --wandb_project "MiniMind-GRPO-GQA"

# MLA 模型 GRPO
torchrun --nproc_per_node 4 train_grpo.py \
    --use_mla 1 --mla_kv_dim 128 --mla_q_dim 256 --mla_rope_dim 128 \
    --use_wandb --wandb_project "MiniMind-GRPO-MLA"
```

GRPO 关键参数（均为默认值）：

| 参数 | 值 |
|---|---|
| `data_path` | `../dataset/rlaif-mini.jsonl`（23MB） |
| `batch_size` | 2 |
| `learning_rate` | 8e-8 |
| `max_seq_len` | 66（prompt 最大长度） |
| `max_gen_len` | 1536（生成最大长度） |
| `num_generations` | 8（每个 prompt 生成的候选数） |
| `beta` | 0.02（KL 惩罚系数） |
| `reward_model_path` | `../../internlm2-1_8b-reward` |

- 基础权重自动加载 `full_sft_512.pth`（reasoning=0）或 `reason_512.pth`（reasoning=1）
- **产出**：`out/grpo_512.pth`

### 6.3 PPO（在线强化学习）

```bash
torchrun --nproc_per_node 4 train_ppo.py \
    --use_wandb --wandb_project "MiniMind-PPO"
```

PPO 关键参数：

| 参数 | 默认值 |
|---|---|
| `data_path` | `../dataset/rlaif-mini.jsonl` |
| `batch_size` | 2 |
| `learning_rate` | 8e-8（Actor） |
| `critic_learning_rate` | 8e-8（Critic） |
| `clip_epsilon` | 0.1 |
| `vf_coef` | 0.5 |
| `kl_coef` | 0.02 |

- PPO 需要同时维护 4 个模型：actor, old_actor, ref_model, critic
- **产出**：`out/ppo_actor_512.pth`

### 6.4 后训练评测

```bash
cd minimind

# DPO 模型评估
python eval_llm.py --weight dpo --show_speed 1

# GRPO 模型评估
python eval_llm.py --weight grpo --show_speed 1
```

---

## 阶段七：1B 模型扩展实验

### 7.1 可行性分析

| 维度 | 结论 |
|---|---|
| 算力 | 8×A100-80GB 绰绰有余（单卡即可放下 1B 训练） |
| 代码 | 无需修改，通过 CLI 参数调整 |
| 训练时间 | 4×A100 约 2-3h |
| **数据量** | **⚠️ 现有 0.9B tokens 严重不足，需先完成阶段二 2.4 节的数据准备** |

### 7.2 1B 模型配置

| 参数 | GQA | MLA | 说明 |
|---|---|---|---|
| `hidden_size` | 2048 | 2048 | |
| `num_hidden_layers` | 22 | 22 | |
| `num_attention_heads` | 16 | 16 | |
| `num_key_value_heads` | 4 | 4 | GQA 分组 |
| `vocab_size` | 6400 | 6400 | 与 26M 保持一致以便对比 |
| `mla_kv_dim` | — | **512** | hidden_size / 4 |
| `mla_q_dim` | — | **1024** | hidden_size / 2 |
| `mla_rope_dim` | — | **256** | hidden_size / 8 |
| 总参数量 | ~988M | ~988M+ | |
| 训练显存 | ~18GB | ~18GB | 单卡 A100 即可 |

MLA 维度按 hidden_size 比例放大的经验公式：
- `kv_dim = hidden_size / 4`
- `q_dim = hidden_size / 2`
- `rope_dim = hidden_size / 8`

### 7.3 1B GQA 预训练 + SFT

```bash
conda activate pre
cd minimind/trainer

# 预训练
torchrun --nproc_per_node 4 train_pretrain.py \
    --hidden_size 2048 --num_hidden_layers 22 \
    --data_path ../dataset/pretrain_1b.jsonl \
    --max_seq_len 512 --batch_size 8 \
    --use_wandb --wandb_project "MiniMind-Pretrain-1B-GQA"

# SFT
torchrun --nproc_per_node 4 train_full_sft.py \
    --hidden_size 2048 --num_hidden_layers 22 \
    --use_wandb --wandb_project "MiniMind-SFT-1B-GQA"

# 备份
cd minimind
mkdir -p out/1b_baseline
cp out/pretrain_2048.pth out/1b_baseline/
cp out/full_sft_2048.pth out/1b_baseline/
```

### 7.4 1B MLA 预训练 + SFT

```bash
cd minimind/trainer

# 预训练
torchrun --nproc_per_node 4 train_pretrain.py \
    --hidden_size 2048 --num_hidden_layers 22 \
    --use_mla 1 --mla_kv_dim 512 --mla_q_dim 1024 --mla_rope_dim 256 \
    --data_path ../dataset/pretrain_1b.jsonl \
    --max_seq_len 512 --batch_size 8 \
    --use_wandb --wandb_project "MiniMind-Pretrain-1B-MLA"

# SFT
torchrun --nproc_per_node 4 train_full_sft.py \
    --hidden_size 2048 --num_hidden_layers 22 \
    --use_mla 1 --mla_kv_dim 512 --mla_q_dim 1024 --mla_rope_dim 256 \
    --use_wandb --wandb_project "MiniMind-SFT-1B-MLA"

# 备份
cd minimind
mkdir -p out/1b_mla
cp out/pretrain_2048.pth out/1b_mla/
cp out/full_sft_2048.pth out/1b_mla/
```

### 7.5 1B 评测

```bash
cd minimind/benchmark

# 1B GQA
python run_all.py --weight full_sft --save_dir ../out/1b_baseline \
    --hidden_size 2048 --num_hidden_layers 22

# 1B MLA
python run_all.py --weight full_sft --save_dir ../out/1b_mla \
    --hidden_size 2048 --num_hidden_layers 22 \
    --use_mla 1 --mla_kv_dim 512 --mla_q_dim 1024 --mla_rope_dim 256
```

### 7.6 跨规模消融对比

| 指标 | 26M GQA | 26M MLA | 1B GQA | 1B MLA |
|---|---|---|---|---|
| 参数量 | 25.8M | ____M | ~988M | ____M |
| PPL ↓ | ____ | ____ | ____ | ____ |
| C-Eval 准确率 ↑ | ____ | ____ | ____ | ____ |
| 生成评分 ↑ | ____ | ____ | ____ | ____ |
| 推理速度 (tok/s) | ____ | ____ | ____ | ____ |
| 推理显存 (MB) | ____ | ____ | ____ | ____ |

---

## 阶段八：模型转换与发布

### 8.1 Torch → Transformers 格式转换

```bash
cd minimind/scripts

# 26M GQA Baseline → Llama 格式（兼容 llama.cpp/vllm/ollama）
python convert_model.py --weight full_sft \
    --input_dir ../out/baseline --output_dir ../MiniMind2-Small-GQA

# 26M MLA → MiniMind 原生格式（需 trust_remote_code=True）
python convert_model.py --weight full_sft --use_mla 1 \
    --input_dir ../out/mla --output_dir ../MiniMind2-Small-MLA

# 1B GQA
python convert_model.py --weight full_sft \
    --hidden_size 2048 --num_hidden_layers 22 \
    --input_dir ../out/1b_baseline --output_dir ../MiniMind2-1B-GQA

# 1B MLA
python convert_model.py --weight full_sft --use_mla 1 \
    --hidden_size 2048 --num_hidden_layers 22 \
    --mla_kv_dim 512 --mla_q_dim 1024 --mla_rope_dim 256 \
    --input_dir ../out/1b_mla --output_dir ../MiniMind2-1B-MLA
```

**两种格式说明**：

| | Llama 格式 | MiniMind 原生格式 |
|---|---|---|
| 适用模型 | 标准 GQA | MLA / MoE |
| 加载方式 | `AutoModelForCausalLM.from_pretrained(...)` | 需加 `trust_remote_code=True` |
| 第三方兼容 | llama.cpp / vllm / ollama | 仅 transformers |

### 8.2 上传到 HuggingFace

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("your-username/MiniMind2-Small-GQA", repo_type="model")
api.upload_folder(folder_path="./MiniMind2-Small-GQA", repo_id="your-username/MiniMind2-Small-GQA")
```

### 8.3 验证加载

```python
from transformers import AutoModelForCausalLM

# GQA — 直接加载
model = AutoModelForCausalLM.from_pretrained("your-username/MiniMind2-Small-GQA")

# MLA — 需要 trust_remote_code
model = AutoModelForCausalLM.from_pretrained("your-username/MiniMind2-Small-MLA", trust_remote_code=True)
```

---

## 前置检查清单

- [x] MLA 代码集成到 `model/model_minimind.py`
- [x] 9 个训练脚本已添加 MLA CLI 参数
- [x] `eval_llm.py` 已支持 MLA 参数
- [x] `scripts/convert_model.py` 已支持 MLA 转换和 CLI 参数
- [x] 数据文件名软链接就绪
- [x] `.gitignore` 已忽略数据集和 checkpoints
- [ ] WandB 账号已登录
- [ ] 1B 扩展数据已下载（阶段七前置）
- [ ] Reward Model 已下载（阶段六 GRPO/PPO 前置）

---

## 断点续训说明

所有训练脚本均支持断点续训，添加 `--from_resume 1` 即可：

```bash
python train_pretrain.py --from_resume 1
python train_full_sft.py --from_resume 1
torchrun --nproc_per_node 4 train_pretrain.py --from_resume 1
```

机制说明：
- 检查点自动保存到 `./checkpoints/` 目录
- 文件命名：`<权重名>_<维度>_resume.pth`
- 支持跨不同 GPU 数量恢复（自动调整 step）
- 支持 wandb run 连续性恢复

---

## 执行时间线

```
Day 1 — 26M 核心消融实验:
  阶段一  环境配置 + 验证
  阶段二  数据下载 + 软链接（如已完成则跳过）
  阶段三  GQA Baseline Pretrain (~20min) + SFT (~15min) + 评估 + 备份
  阶段四  MLA Pretrain (~20min) + SFT (~15min) + 评估 + 备份
  阶段五  量化评测 + 消融对比分析

Day 2 — 后训练与发布:
  阶段六  DPO / GRPO / PPO 后训练
  阶段八  模型转换 + HuggingFace 上传

Day 3 — 1B 扩展实验:
  阶段二  补充 1B 预训练数据（如未完成）
  阶段七  1B GQA Pretrain + SFT (~2-3h)
  阶段七  1B MLA Pretrain + SFT (~2-3h)
  阶段七  1B 评测 + 跨规模消融分析
  阶段八  1B 模型转换 + 上传
```

> 以上时间基于 4×A100 估算。使用全部 8 卡可进一步缩短至约一半。

---

## 风险与注意事项

1. **权重覆盖**：GQA 和 MLA 的输出文件名相同（`*_512.pth`），务必在训练 MLA 前完成 GQA 权重备份
2. **Python 环境**：当前机器默认 Python 没有 torch，需使用 `conda activate pre`
3. **wandb**：如需记录训练曲线，确保已登录（`wandb login`）
4. **Reward Model**：GRPO/PPO 需要 InternLM2-1.8B-Reward（约 3.6GB），必须放在 minimind 同级目录
5. **MLA 与 Llama 不兼容**：MLA 模型只能通过 MiniMind 原生格式上传 HuggingFace，无法转为 Llama 格式
6. **1B 数据量**：现有 0.9B tokens 训练 1B 模型会严重过拟合，务必先补充数据至 5B+ tokens
7. **1B MLA 维度**：需按 hidden_size 比例放大（kv_dim≈d/4, q_dim≈d/2, rope_dim≈d/8），过小会损失表达能力
8. **1B 权重文件名**：`*_2048.pth`（因 hidden_size=2048），与 26M 的 `*_512.pth` 不冲突

---

## 实验结果记录

### 26M 消融对比

| 指标 | GQA Baseline | MLA |
|---|---|---|
| 总参数量 | ____M | ____M |
| 注意力层参数量 | ____M | ____M |
| 预训练最终 Loss | ____ | ____ |
| SFT 最终 Loss | ____ | ____ |
| 训练总时间 | ____min | ____min |
| 显存峰值 (MB) | ____ | ____ |
| **PPL** ↓ | ____ | ____ |
| **C-Eval 准确率** ↑ | ____% | ____% |
| **生成质量评分** ↑ | ____/100 | ____/100 |
| 推理速度 (tokens/s) | ____ | ____ |
| 推理显存峰值 (MB) | ____ | ____ |

### 1B 消融对比

| 指标 | GQA | MLA |
|---|---|---|
| 总参数量 | ____M | ____M |
| 预训练最终 Loss | ____ | ____ |
| SFT 最终 Loss | ____ | ____ |
| PPL ↓ | ____ | ____ |
| C-Eval 准确率 ↑ | ____% | ____% |
| 生成质量评分 ↑ | ____/100 | ____/100 |
| 推理速度 (tokens/s) | ____ | ____ |
| 推理显存峰值 (MB) | ____ | ____ |

### 对话质量对比

> 种子固定 2026，使用 eval_llm.py 内置 8 个 prompt

| Prompt | GQA Baseline | MLA |
|---|---|---|
| 1. 你有什么特长？ | （待填写） | （待填写） |
| 2. 为什么天空是蓝色的 | （待填写） | （待填写） |
| 3. 请用Python写一个计算斐波那契数列的函数 | （待填写） | （待填写） |
| 4. 解释一下"光合作用"的基本过程 | （待填写） | （待填写） |
| 5. 如果明天下雨，我应该如何出门 | （待填写） | （待填写） |
| 6. 比较一下猫和狗作为宠物的优缺点 | （待填写） | （待填写） |
| 7. 解释什么是机器学习 | （待填写） | （待填写） |
| 8. 推荐一些中国的美食 | （待填写） | （待填写） |

### 综合结论

| 维度 | GQA Baseline | MLA | 结论 |
|---|---|---|---|
| 参数效率 | ____ | ____ | |
| 训练收敛速度 | ____ | ____ | |
| 推理速度 | ____ | ____ | |
| 显存占用 | ____ | ____ | |
| 对话质量 | ____ | ____ | |
| **总体结论** | | | |

### Loss 曲线截图

> 从 WandB Dashboard 下载，放到 `images/` 目录

| 阶段 | 截图 |
|---|---|
| 预训练 Loss 对比 | （待添加） |
| SFT Loss 对比 | （待添加） |
| Learning Rate 曲线 | （待添加） |
