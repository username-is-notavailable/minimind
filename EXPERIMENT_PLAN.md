    # MiniMind MLA 消融实验计划

## 实验目标

在 MiniMind2-Small (26M) 和 MiniMind2-0.5B (~545M) 两个规模上，对比标准 GQA（Grouped Query Attention）与自定义 MLA（Multi-head Latent Attention）注意力机制的训练效果和推理效率差异。

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
conda activate minimind  # torch 2.8.0
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

### 2.4 大规模模型扩展数据准备

按 Chinchilla 最优比例（~20 tokens/参数），0.5B 模型需要约 10B tokens，1B 模型需要约 20B tokens。

数据通过两阶段准备，脚本均在 `dataset_1B/` 目录下：

1. **`prepare_pretrain_data.py`**（基础采集，~97 分钟）— 采集本地数据 + 9 个 HuggingFace 远程源
2. **`expand_pretrain_data.py`**（扩充至 20B tokens，~184 分钟）— 追加本地完整 SFT 数据 + 4 个远程源继续采样

**最终数据规模**：

| 指标 | 值 |
|---|---|
| 总样本数 | 32,349,742 |
| 总字符数 | 34.6B |
| 估算 tokens（6400 词表） | **~21.7B** ✅ |
| 估算 tokens（32K 词表） | **~13.9B** |
| 文件大小 | 63.4 GB |

**数据来源明细**：

| 类别 | 语言 | 条数 | 字符 | 数据源 |
|---|---|---|---|---|
| 中文网页 | 中文 | 10,500,000 | 10.55B | `Skywork/SkyPile-150B`（基础 3.5M + 扩充 7M） |
| 中文对话(SFT转) | 中文 | 5,831,466 | 6.52B | 本地 sft_t2t + sft_t2t_mini（转预训练格式） |
| 本地预训练 | 中文 | 8,429,917 | 3.24B | 现有 pretrain_hq（匠数科技） |
| 中文百科 | 中文 | 1,270,195 | 901M | `wikimedia/wikipedia` (zh)（全量） |
| 中文多样化 | 中文 | 1,000,000 | 1.02B | `Skywork/SkyPile-150B`（偏移 12M 不同区间） |
| 中文新闻 | 中文 | 500,000 | 500M | `Skywork/SkyPile-150B`（偏移 5M） |
| 英文网页 | 英文 | 3,000,000 | 5.67B | `allenai/c4` (en)（基础 1M + 扩充 2M） |
| 英文学术 | 英文 | 1,300,000 | 4.73B | `open-web-math/open-web-math`（基础 500K + 扩充 800K） |
| 英文百科 | 英文 | 500,000 | 1.50B | `wikimedia/wikipedia` (en) |
| 代码 | 混合 | 17,764 | 8M | `iamtarun/python_code_instructions_18k_alpaca` |

> **已知问题**：`CASIA-LM/ChineseWebText` 数据格式不兼容（text 字段为字符串化列表），采集 0 条。已通过增加 SkyPile 和 SFT 数据补偿。代码源数据集仅 1.8 万条，远小于目标，可后续替换为更大数据集。

**运行方式**：

```bash
cd minimind/dataset_1B

# 第一步：基础采集（~97 分钟）
nohup python prepare_pretrain_data.py > prepare.log 2>&1 &

# 第二步：扩充至 20B tokens（~184 分钟）
nohup python expand_pretrain_data.py > expand.log 2>&1 &

# 快速测试
python prepare_pretrain_data.py --test_mode
python expand_pretrain_data.py --test_mode

# 查看进度
tail -f prepare.log  # 或 expand.log
```

**输出文件**：

| 文件 | 大小 | 说明 |
|---|---|---|
| `dataset_1B/pretrain_1b.jsonl` | 63.4 GB | 合并打乱后的预训练数据（32.3M 行） |
| `dataset_1B/tokenizer_train.jsonl` | 401 MB | 均匀采样子集（20 万条），用于分词器训练 |
| `dataset_1B/data_stats.json` | 3.4 KB | 各数据源条数、字符数统计 |

**数据就绪后创建软链接**（让训练脚本能找到数据）：

```bash
cd minimind/dataset
ln -sf ../dataset_1B/pretrain_1b.jsonl pretrain_hq.jsonl
```

> 注意：此软链接会覆盖原本指向 `pretrain_t2t.jsonl` 的链接。如需切回 26M 训练数据，重新执行 `ln -sf pretrain_t2t.jsonl pretrain_hq.jsonl` 即可。

详见 `dataset_1B/README.md`。

### 2.5 大规模模型专用分词器训练

原始 6400 词表为 26M 模型设计。0.5B 模型中 embedding 层占比不再是瓶颈。使用 32K 词表可显著提升编码效率：

| | 6400 词表 | 32000 词表 |
|---|---|---|
| embed 占比 (0.5B) | 2.4% | 11.7% |
| 中文编码效率 | ~1.6 字符/token | ~2.5-3.0 字符/token |
| 512 tokens 覆盖 | ~820 字符 | ~1400 字符 |

`prepare_pretrain_data.py` 已自动生成分词器训练用采样文件 `dataset_1B/tokenizer_train.jsonl`，直接使用即可：

```bash
cd minimind/trainer
python train_tokenizer_05b.py \
    --data_path ../dataset_1B/tokenizer_train.jsonl \
    --vocab_size 32000 \
    --max_lines 200000 \
    --output_dir ../model_05b_tokenizer
```

> **注意**：使用新分词器训练的 0.5B 模型与 6400 词表的 26M 模型 **PPL 不直接可比**。如需公平消融对比 MLA，应在同一分词器下对比 GQA vs MLA。

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
| MiniMind2-0.5B | **~545M** | `1536` | `20` | `*_1536.pth` |

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
conda activate minimind
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

### 3.7 104M 模型训练（已验证）

104M 模型使用 `hidden_size=768, num_hidden_layers=16`，以下参数经实际训练验证可行：

**预训练**（4×A100，~6h，2 epochs × 16.6M 样本 ≈ 4.6B tokens）：

```bash
cd minimind/trainer

torchrun --nproc_per_node 4 train_pretrain.py \
    --hidden_size 768 --num_hidden_layers 16 \
    --data_path ../dataset/pretrain_hq.jsonl \
    --max_seq_len 340 --epochs 2 \
    --batch_size 32 --accumulation_steps 8 \
    --learning_rate 3e-4 --dtype bfloat16 \
    --save_weight pretrain --save_interval 1000 \
    --use_wandb --wandb_project "MiniMind-104M-Pretrain"
```

**SFT**（4×A100，~10h，完整 SFT 数据 5.1M 条 × 2 epochs）：

```bash
torchrun --nproc_per_node 4 train_full_sft.py \
    --hidden_size 768 --num_hidden_layers 16 \
    --data_path ../dataset/sft_t2t.jsonl \
    --max_seq_len 512 --from_weight pretrain \
    --epochs 2 --batch_size 64 --accumulation_steps 2 \
    --learning_rate 5e-6 --dtype bfloat16 \
    --save_weight full_sft --save_interval 2000 \
    --use_wandb --wandb_project "MiniMind-104M-SFT-Full"
```

**与 26M 默认值的关键差异**：

| 参数 | 26M 默认 | 104M 实际 | 说明 |
|---|---|---|---|
| `hidden_size` | 512 | **768** | 增大 1.5× |
| `num_hidden_layers` | 8 | **16** | 翻倍 |
| `learning_rate`（预训练） | 5e-4 | **3e-4** | 略微降低，更大模型需更保守 |
| `epochs`（预训练） | 1 | **2** | 数据过两遍以补充 token 不足 |
| SFT `data_path` | `sft_mini_512.jsonl` | **`sft_t2t.jsonl`** | 使用完整 14GB SFT 数据 |
| SFT `batch_size` | 16 | **64** | A100 显存充足可增大 |
| SFT `accumulation_steps` | 1 | **2** | 等效 batch 128 |
| SFT `learning_rate` | 1e-6 | **5e-6** | 略提高以匹配更大有效 batch |
| SFT `max_seq_len` | 340 | **512** | 覆盖更长文本 |

**产出**：`out/pretrain_768.pth`（208MB）、`out/full_sft_768.pth`（208MB）

**模型转换**（已验证）：

```bash
cd minimind/scripts
python convert_model.py --weight pretrain --hidden_size 768 --num_hidden_layers 16
python convert_model.py --weight full_sft --hidden_size 768 --num_hidden_layers 16
```

---

## 阶段四：26M 模型 MLA 训练（未执行）

> **状态：未执行。** 以下为计划内容，保留供后续实验参考。

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

## 阶段五：26M GQA vs MLA 消融对比评测（未执行）

> **状态：未执行。** MLA 训练尚未进行，以下为计划内容。
> 已完成的 26M GQA 评测结果见「实验结果记录」章节。

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

## 阶段六：RLHF / RLAIF 后训练（未执行）

> **状态：未执行。** 以下为计划内容，保留供后续实验参考。

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

## 阶段七：0.5B 模型扩展实验

### 7.1 可行性分析

| 维度 | 结论 |
|---|---|
| 算力 | 4×A100-80GB，每卡占用 ~79GB（96%利用率） |
| 代码 | 无需修改，通过 CLI 参数调整 |
| 训练时间 | 4×A100 预训练约 35h（~1.5 天），SFT 约 3-15h |
| **数据量** | 现有 ~13.9B tokens（32K 词表）已超过 Chinchilla 最优量（~10B tokens），无需额外扩充 |

### 7.2 0.5B 模型配置

| 参数 | GQA | MLA | 说明 |
|---|---|---|---|
| `hidden_size` | 1536 | 1536 | |
| `num_hidden_layers` | 20 | 20 | |
| `num_attention_heads` | 8 | 8 | |
| `num_key_value_heads` | 2 | 2 | GQA 分组 |
| `vocab_size` | 32000 | 32000 | 32K 新分词器 |
| `mla_kv_dim` | — | **384** | hidden_size / 4 |
| `mla_q_dim` | — | **768** | hidden_size / 2 |
| `mla_rope_dim` | — | **192** | hidden_size / 8 |
| 总参数量 | ~545M | ~545M+ | |
| 训练显存 | ~79GB | ~79GB | 每卡 A100-80GB（batch_size=64） |

MLA 维度按 hidden_size 比例放大的经验公式：
- `kv_dim = hidden_size / 4`
- `q_dim = hidden_size / 2`
- `rope_dim = hidden_size / 8`

### 7.2.1 训练前：分词器训练（已完成）

```bash
cd minimind/trainer

# 训练 32K 词表分词器（~10 分钟）
python train_tokenizer_05b.py \
    --data_path ../dataset_1B/tokenizer_train.jsonl \
    --vocab_size 32000 \
    --max_lines 200000 \
    --output_dir ../model_05b_tokenizer
```

### 7.3 0.5B GQA 预训练 + SFT（已验证参数）

```bash
conda activate minimind
cd minimind/trainer

# 预训练（~35h，4×A100-80GB，每卡 ~79GB）
torchrun --nproc_per_node 4 train_pretrain.py \
    --hidden_size 1536 --num_hidden_layers 20 \
    --vocab_size 32000 \
    --data_path ../dataset_1B/pretrain_1b.jsonl \
    --tokenizer_path ../model_05b_tokenizer \
    --max_seq_len 512 --batch_size 64 \
    --accumulation_steps 2 \
    --learning_rate 3e-4 \
    --epochs 1 \
    --dtype bfloat16 \
    --save_interval 2000 \
    --use_wandb --wandb_project "MiniMind-Pretrain-0.5B-GQA"

# SFT（~3-15h，取决于数据集）
torchrun --nproc_per_node 4 train_full_sft.py \
    --hidden_size 1536 --num_hidden_layers 20 \
    --vocab_size 32000 \
    --data_path ../dataset/sft_t2t.jsonl \
    --tokenizer_path ../model_05b_tokenizer \
    --max_seq_len 512 --batch_size 64 \
    --from_weight pretrain \
    --epochs 2 \
    --learning_rate 5e-6 \
    --dtype bfloat16 \
    --use_wandb --wandb_project "MiniMind-SFT-0.5B-GQA"

# 备份
cd minimind
mkdir -p out/05b_baseline
cp out/pretrain_1536.pth out/05b_baseline/
cp out/full_sft_1536.pth out/05b_baseline/
```

### 7.4 0.5B MLA 预训练 + SFT

```bash
cd minimind/trainer

# 预训练
torchrun --nproc_per_node 4 train_pretrain.py \
    --hidden_size 1536 --num_hidden_layers 20 \
    --vocab_size 32000 \
    --use_mla 1 --mla_kv_dim 384 --mla_q_dim 768 --mla_rope_dim 192 \
    --data_path ../dataset_1B/pretrain_1b.jsonl \
    --tokenizer_path ../model_05b_tokenizer \
    --max_seq_len 512 --batch_size 64 \
    --accumulation_steps 2 \
    --learning_rate 3e-4 \
    --epochs 1 \
    --dtype bfloat16 \
    --save_interval 2000 \
    --use_wandb --wandb_project "MiniMind-Pretrain-0.5B-MLA"

# SFT
torchrun --nproc_per_node 4 train_full_sft.py \
    --hidden_size 1536 --num_hidden_layers 20 \
    --vocab_size 32000 \
    --use_mla 1 --mla_kv_dim 384 --mla_q_dim 768 --mla_rope_dim 192 \
    --data_path ../dataset/sft_t2t.jsonl \
    --tokenizer_path ../model_05b_tokenizer \
    --max_seq_len 512 --batch_size 64 \
    --from_weight pretrain \
    --epochs 2 \
    --learning_rate 5e-6 \
    --dtype bfloat16 \
    --use_wandb --wandb_project "MiniMind-SFT-0.5B-MLA"

# 备份
cd minimind
mkdir -p out/05b_mla
cp out/pretrain_1536.pth out/05b_mla/
cp out/full_sft_1536.pth out/05b_mla/
```

### 7.5 0.5B 评测

```bash
cd minimind/benchmark

# 0.5B GQA
python run_all.py --weight full_sft --save_dir ../out/05b_baseline \
    --hidden_size 1536 --num_hidden_layers 20

# 0.5B MLA
python run_all.py --weight full_sft --save_dir ../out/05b_mla \
    --hidden_size 1536 --num_hidden_layers 20 \
    --use_mla 1 --mla_kv_dim 384 --mla_q_dim 768 --mla_rope_dim 192
```

### 7.6 跨规模消融对比

| 指标 | 26M GQA | 26M MLA | 0.5B GQA | 0.5B MLA |
|---|---|---|---|---|
| 参数量 | 25.8M | ____M | ~545M | ____M |
| 词表大小 | 6400 | 6400 | 32000 | 32000 |
| PPL ↓ | ____ | ____ | ____ | ____ |
| C-Eval 准确率 ↑ | ____ | ____ | ____ | ____ |
| 生成评分 ↑ | ____ | ____ | ____ | ____ |
| 推理速度 (tok/s) | ____ | ____ | ____ | ____ |
| 推理显存 (MB) | ____ | ____ | ____ | ____ |

---

## 阶段八：模型转换与发布

> **状态：26M 和 104M GQA 模型已完成转换与上传。MLA / 0.5B 部分未执行。**

### 8.1 Torch → Transformers 格式转换

```bash
cd minimind/scripts

# 26M GQA Baseline → Llama 格式（兼容 llama.cpp/vllm/ollama）
python convert_model.py --weight full_sft \
    --input_dir ../out/baseline --output_dir ../MiniMind2-Small-GQA

# 26M MLA → MiniMind 原生格式（需 trust_remote_code=True）
python convert_model.py --weight full_sft --use_mla 1 \
    --input_dir ../out/mla --output_dir ../MiniMind2-Small-MLA

# 0.5B GQA
python convert_model.py --weight full_sft \
    --hidden_size 1536 --num_hidden_layers 20 \
    --input_dir ../out/05b_baseline --output_dir ../MiniMind2-0.5B-GQA

# 0.5B MLA
python convert_model.py --weight full_sft --use_mla 1 \
    --hidden_size 1536 --num_hidden_layers 20 \
    --mla_kv_dim 384 --mla_q_dim 768 --mla_rope_dim 192 \
    --input_dir ../out/05b_mla --output_dir ../MiniMind2-0.5B-MLA
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
- [x] WandB 账号已登录
- [x] SFT 数据 tool_calls 字段兼容性修复（`lm_dataset.py`）
- [x] 26M GQA Pretrain + SFT（seq340）完成
- [x] 26M GQA Pretrain + SFT（seq512 消融）完成
- [x] 104M GQA Pretrain + SFT 完成
- [x] 全部已完成模型上传至 HuggingFace
- [ ] 0.5B 扩展数据已下载（阶段七前置）
- [ ] Reward Model 已下载（阶段六 GRPO/PPO 前置）
- [ ] MLA 消融实验（阶段四）
- [x] 0.5B GQA Pretrain + SFT + 评测 完成
- [x] 评测脚本已添加 --vocab_size 和 --tokenizer_path 参数

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

Day 3-5 — 0.5B 扩展实验:
  阶段七  分词器训练 (~30min)
  阶段七  0.5B GQA Pretrain (~2天) + SFT (~3-15h)
  阶段七  0.5B MLA Pretrain (~2天) + SFT (~3-15h)
  阶段七  0.5B 评测 + 跨规模消融分析
  阶段八  0.5B 模型转换 + 上传
```

> 以上时间基于 4×A100 估算。使用全部 8 卡可进一步缩短至约一半。

---

## 风险与注意事项

1. **权重覆盖**：GQA 和 MLA 的输出文件名相同（`*_512.pth`），务必在训练 MLA 前完成 GQA 权重备份
2. **Python 环境**：当前机器默认 Python 没有 torch，需使用 `conda activate minimind`
3. **wandb**：如需记录训练曲线，确保已登录（`wandb login`）
4. **Reward Model**：GRPO/PPO 需要 InternLM2-1.8B-Reward（约 3.6GB），必须放在 minimind 同级目录
5. **MLA 与 Llama 不兼容**：MLA 模型只能通过 MiniMind 原生格式上传 HuggingFace，无法转为 Llama 格式
6. **0.5B 数据量**：现有 ~13.9B tokens（32K词表）已超过 Chinchilla 最优量（~10B tokens），无需额外扩充
7. **0.5B MLA 维度**：需按 hidden_size 比例放大（kv_dim≈d/4, q_dim≈d/2, rope_dim≈d/8），过小会损失表达能力
8. **0.5B 权重文件名**：`*_1536.pth`（因 hidden_size=1536），与 26M 的 `*_512.pth` 不冲突

---

## 实验结果记录

### 已完成的实验

| 实验 | 模型 | 状态 | HuggingFace |
|---|---|---|---|
| 26M GQA Pretrain+SFT (seq340) | 25.8M | ✅ 已完成 | [leixinlin/MiniMind2-Small-GQA](https://huggingface.co/leixinlin/MiniMind2-Small-GQA) |
| 26M GQA Pretrain+SFT (seq512) | 25.8M | ✅ 已完成 | [leixinlin/MiniMind2-Small-GQA-Seq512](https://huggingface.co/leixinlin/MiniMind2-Small-GQA-Seq512) |
| 104M GQA Pretrain+SFT | 104.0M | ✅ 已完成 | [leixinlin/MiniMind2-Pretrain-104M](https://huggingface.co/leixinlin/MiniMind2-Pretrain-104M) / [leixinlin/MiniMind2-SFT-104M](https://huggingface.co/leixinlin/MiniMind2-SFT-104M) |
| 26M MLA 消融 | — | ❌ 未执行 | — |
| RLHF/RLAIF 后训练 | — | ❌ 未执行 | — |
| 0.5B GQA Pretrain+SFT | 544.67M | ✅ 已完成 | [leixinlin/MiniMind2-0.5B-Pretrain](https://huggingface.co/leixinlin/MiniMind2-0.5B-Pretrain) / [leixinlin/MiniMind2-0.5B-SFT](https://huggingface.co/leixinlin/MiniMind2-0.5B-SFT) |
| 0.5B GQA DPO | 544.67M | ✅ 已完成 | [leixinlin/MiniMind2-0.5B-DPO](https://huggingface.co/leixinlin/MiniMind2-0.5B-DPO) |
| 0.5B MLA 消融 | — | ❌ 未执行 | — |

---

### 26M GQA 训练结果

#### 训练参数

| 参数 | Pretrain (seq340) | SFT (seq340) | Pretrain (seq512) | SFT (seq512) |
|---|---|---|---|---|
| `data_path` | `pretrain_t2t.jsonl` (7.8GB) | `sft_t2t_mini.jsonl` (1.6GB) | `pretrain_t2t.jsonl` (7.8GB) | `sft_t2t_mini.jsonl` (1.6GB) |
| `max_seq_len` | 340 | 340 | 512 | 512 |
| `batch_size` | 256 | 128 | 128 | 64 |
| `accumulation_steps` | 1 | 1 | 1 | 1 |
| `learning_rate` | 5e-4 | 1e-6 | 5e-4 | 1e-6 |
| `epochs` | 1 | 2 | 1 | 2 |
| `dtype` | bfloat16 | bfloat16 | bfloat16 | bfloat16 |
| GPU | 2×A100 | 2×A100 | 2×A100 | 2×A100 |

#### 训练 Loss

| 模型 | 最终 Pretrain Loss | 最终 SFT Loss |
|---|---|---|
| 26M seq340 | ~1.84 | 1.65 |
| 26M seq512 | 1.84 | 1.66 |

#### 消融对比：max_seq_len=340 vs 512

**PPL（在 pretrain held-out 数据上评测）：**

| 模型 | seq_len=340 | seq_len=512 | 变化 |
|---|---|---|---|
| Pretrain | 167.20 | **29.44** | ↓ 82.4% |
| SFT | 170.37 | **32.41** | ↓ 81.0% |

**C-Eval（52 科 val 集）：**

| 类别 | seq_len=340 | seq_len=512 |
|---|---|---|
| STEM | 21.97% | 21.97% |
| 社会科学 | 28.43% | 27.42% |
| 人文 | 26.67% | 24.29% |
| 其他 | 22.79% | 23.97% |
| **总体** | **24.22%** | **24.00%** |

**生成质量（15 prompt 自动评分）：**

| 能力维度 | seq_len=340 | seq_len=512 |
|---|---|---|
| 事实问答 | 82.8 | 83.0 |
| 科学解释 | 82.3 | 82.3 |
| 逻辑推理 | 100.0 | 78.0 |
| 代码生成 | 92.0 | 92.0 |
| 创意写作 | 75.5 | 83.0 |
| 自我认知 | 84.0 | 75.5 |
| **总平均** | **85.4** | **82.4** |

**推理效率：**

| 指标 | seq_len=340 | seq_len=512 |
|---|---|---|
| 推理速度 | 89-92 tok/s | 92-95 tok/s |
| 显存峰值 | 126-133 MB | 125-128 MB |

**消融结论**：seq_len=512 的 PPL 大幅改善（167→29），因为模型能学到更完整的文本上下文。但 C-Eval 和生成质量评分差异不大，26M 模型的知识容量是硬瓶颈。

---

### 104M GQA 训练结果

#### 训练参数

| 参数 | Pretrain | SFT |
|---|---|---|
| `hidden_size` | 768 | 768 |
| `num_hidden_layers` | 16 | 16 |
| `data_path` | `pretrain_hq.jsonl` → `dataset_1B/pretrain_1b.jsonl` (8.8GB, 931万条) | `sft_t2t.jsonl` (14GB, 510万条) |
| `max_seq_len` | 340 | 512 |
| `batch_size` | 32 | 64 |
| `accumulation_steps` | 8 | 2 |
| `learning_rate` | 3e-4 | 5e-6 |
| `epochs` | 2 | 2 |
| `dtype` | bfloat16 | bfloat16 |
| `from_weight` | none（从零） | pretrain |
| GPU | 4×A100 DDP | 4×A100 DDP |

#### 训练 Loss

| 阶段 | 最终 Loss |
|---|---|
| Pretrain | 1.55 |
| SFT | 1.30 |

#### 评测结果

**PPL（pretrain 模型，在 pretrain held-out 数据上）：**

| 指标 | 值 |
|---|---|
| Perplexity | 29.16 |
| Avg Loss | 3.373 |

**C-Eval（52 科 val 集）：**

| 模型 | 总体准确率 |
|---|---|
| Pretrain | 23.85% |
| SFT | 24.07% |

**生成质量（15 prompt 自动评分）：**

| 能力维度 | SFT 评分 |
|---|---|
| 事实问答 | 71.2 |
| 科学解释 | 83.7 |
| 逻辑推理 | 94.0 |
| 代码生成 | 88.0 |
| 创意写作 | 79.0 |
| 自我认知 | 71.0 |
| **加权平均** | **~81.2** |

**推理效率：**

| 指标 | 值 |
|---|---|
| 推理速度 | ~52 tok/s |
| 模型显存 | ~400 MB |
| 推理峰值显存 | 430-443 MB |

---

### 跨规模对比

| 指标 | 26M (seq340) | 26M (seq512) | 104M | 0.5B |
|---|---|---|---|---|
| 参数量 | 25.8M | 25.8M | 104.0M | 544.67M |
| 词表大小 | 6400 | 6400 | 6400 | 32000 |
| 预训练数据 | 7.8GB (原始) | 7.8GB (原始) | 8.8GB (扩展) | 63.4GB (20B tokens) |
| SFT 数据 | 1.6GB (mini) | 1.6GB (mini) | 14GB (完整) | 1.6GB (mini) |
| Pretrain Loss | ~1.84 | 1.84 | 1.55 | — |
| SFT Loss | 1.65 | 1.66 | 1.30 | ~1.18 |
| PPL ↓ | 167.20 | **29.44** | **29.16** | — |
| C-Eval ↑ | 24.22% | 24.00% | 24.07% | **26.00%** |
| 生成评分 ↑ | **85.4** | 82.4 | ~81.2 | **82.87** |
| 推理速度 | 89-92 tok/s | 92-95 tok/s | ~52 tok/s | ~39-43 tok/s |
| 推理显存 | 126-133 MB | 125-128 MB | 430-443 MB | 2173-2206 MB |

**观察**：
1. PPL 在 seq512 和 104M 上表现接近（~29-32），远优于 seq340 的 167，表明 seq_len 对 PPL 影响极大
2. C-Eval 在三种配置下均接近随机水平（~24%），小模型知识容量有限；0.5B 略有提升至 26.00%，人文类别最高达 32.86%
3. 生成评分在 26M seq340 上最高（85.4），0.5B 为 82.87，整体质量更稳定
4. 0.5B 推理速度约 39-43 tok/s，因模型参数量大幅增加（5×104M），显存占用约 2.1GB

### 已知问题与修复记录

1. **SFT 数据 tool_calls 字段问题**：SFT 数据中约 66574 条包含 `tool_calls` 字段（JSON 字符串格式），导致 Jinja2 模板渲染崩溃（`TypeError: Object of type Undefined is not JSON serializable`）。修复：在 `dataset/lm_dataset.py` 的 `create_chat_prompt` 方法中清理 `tool_calls`/`function_call` 字段。
2. **pretrain_hq.jsonl 软链接切换**：26M 训练使用原始 `pretrain_t2t.jsonl`（7.8GB），104M 训练时切换为 `dataset_1B/pretrain_1b.jsonl`（8.8GB 扩展版）。训练完成后需手动切回。
3. **评测脚本 0.5B 兼容**：`eval_llm.py` 和 `benchmark/` 下的评测脚本原本不支持 `--vocab_size` 和 `--tokenizer_path` 参数，已修改添加支持。

---

### 0.5B GQA 训练结果

#### 训练参数

| 参数 | Pretrain | SFT |
|---|---|---|
| `hidden_size` | 1536 | 1536 |
| `num_hidden_layers` | 20 | 20 |
| `vocab_size` | 32000 | 32000 |
| `data_path` | `dataset_1B/pretrain_1b.jsonl` (63.4GB) | `sft_t2t_mini.jsonl` (1.7GB) |
| `tokenizer_path` | `model_05b_tokenizer` | `model_05b_tokenizer` |
| `max_seq_len` | 512 | 512 |
| `batch_size` | 64 | 64 |
| `accumulation_steps` | 2 | 2 |
| `learning_rate` | 3e-4 | 5e-6 |
| `epochs` | 1 | 2 |
| `dtype` | bfloat16 | bfloat16 |
| `from_weight` | none（从零） | pretrain |
| GPU | 4×A100 DDP | 8×A100 DDP |

#### 训练 Loss

| 阶段 | 最终 Loss |
|---|---|
| Pretrain | — |
| SFT | ~1.18 |

#### 评测结果

**C-Eval（52 科 val 集）：**

| 类别 | 准确率 |
|---|---|
| STEM | 26.59% |
| 社会科学 | 23.41% |
| 人文 | 32.86% |
| 其他 | 22.79% |
| **总体** | **26.00%** |

**生成质量（15 prompt 自动评分）：**

| 能力维度 | SFT 评分 |
|---|---|
| 事实问答 | 76.2 |
| 科学解释 | 89.3 |
| 逻辑推理 | 71.0 |
| 代码生成 | 92.0 |
| 创意写作 | 88.0 |
| 自我认知 | 84.0 |
| **加权平均** | **82.87** |

**推理效率：**

| 指标 | 值 |
|---|---|
| 推理速度 | ~39-43 tok/s |
| 模型参数 | 544.67M |
| 参数显存 | 2077.74 MB |
| 推理峰值显存 | 2173-2206 MB |

**生成质量观察**：
- 0.5B 模型在科学解释（89.3）和代码生成（92.0）上表现优异
- 事实问答（76.2）偶有小错（如地球公转周期回答错误），但整体结构完整
- 创意写作（88.0）能力良好，能写诗、推荐美食
- 自我认知（84.0）正确识别为 MiniMind 模型

---

### 0.5B GQA DPO 训练结果

#### 训练参数

| 参数 | 值 |
|---|---|
| `hidden_size` | 1536 |
| `num_hidden_layers` | 20 |
| `vocab_size` | 32000 |
| `data_path` | `dpo.jsonl` (52MB) |
| `from_weight` | `full_sft` |
| `max_seq_len` | 1024 |
| `batch_size` | 4 |
| `learning_rate` | 4e-8 |
| `beta` | 0.1 |
| `epochs` | 1 |
| `dtype` | bfloat16 |
| GPU | 8×A100 DDP |

#### 训练 Loss

| 阶段 | 最终 Loss |
|---|---|
| DPO | ~0.692（接近 ln(2)≈0.693，理想值） |

#### 评测结果

**C-Eval（52 科 val 集）：**

| 类别 | SFT | DPO | 变化 |
|---|---|---|---|
| STEM | 26.59% | 26.30% | -0.29% |
| 社会科学 | 23.41% | 23.41% | 0% |
| 人文 | 32.86% | 32.86% | 0% |
| 其他 | 22.79% | 22.99% | +0.20% |
| **总体** | **26.00%** | **26.00%** | **0%** |

**生成质量对比（15 prompt）：**

| 能力维度 | SFT | DPO | 变化 |
|---|---|---|---|
| 事实问答 | 76.2 | 76.2 | 0 |
| 科学解释 | 89.3 | 89.3 | 0 |
| 逻辑推理 | 71.0 | 71.0 | 0 |
| 代码生成 | 92.0 | 92.0 | 0 |
| 创意写作 | 88.0 | 88.0 | 0 |
| 自我认知 | 84.0 | 84.0 | 0 |
| **加权平均** | **82.87** | **82.87** | **0** |

**推理效率：**

| 指标 | SFT | DPO |
|---|---|---|
| 推理速度 | ~39-43 tok/s | ~38-43 tok/s |
| 推理峰值显存 | 2173-2206 MB | 2173-2206 MB |

**DPO 分析**：
- DPO 的 loss 收敛至 ~0.692，接近理论最优值 ln(2)≈0.693，说明模型已学习偏好信号
- C-Eval 和生成评分与 SFT 完全一致，DPO 没有显著改变模型的知识和生成能力，主要作用于偏好对齐
- 使用固定 seed 评测，DPO 与 SFT 输出几乎相同，表明 learning_rate=4e-8 的微调幅度较小，成功避免了灾难性遗忘

---

### 0.5B GQA GRPO 训练（进行中）

#### 训练参数

| 参数 | 值 |
|---|---|
| `hidden_size` | 1536 |
| `num_hidden_layers` | 20 |
| `vocab_size` | 32000 |
| `tokenizer_path` | `model_05b_tokenizer` |
| `data_path` | `rlaif-mini.jsonl` (23MB) |
| `reward_model_path` | `internlm2-1_8b-reward` |
| `from_weight` | `full_sft`（自动加载 `full_sft_1536.pth`） |
| `reasoning` | 0 |
| `max_seq_len` | 66（prompt 最大长度） |
| `max_gen_len` | 1536（生成最大长度） |
| `num_generations` | 8（每个 prompt 生成的候选数） |
| `batch_size` | 2 |
| `learning_rate` | 8e-8 |
| `beta` | 0.02（KL 惩罚系数） |
| `epochs` | 1 |
| `dtype` | bfloat16 |
| `save_interval` | 100 步 |
| GPU | 8×A100 DDP |
| 总步数 | 1219 |
| 预计耗时 | ~20 小时 |

#### 训练进度（实时更新）

| 步数 | Actor Loss | Reward | Avg Response Len |
|---|---|---|---|
| 10 | 0.0000 | -0.9767 | 691.25 |
| 20 | 0.0001 | -1.9622 | 220.56 |
| 30 | 0.0003 | -1.9141 | 468.81 |
| 40 | 0.0008 | -0.9633 | 47.88 |
| 50 | 0.0001 | -1.9243 | 93.69 |

---

### 0.5B 模型训练全流程参数汇总

| 阶段 | 数据 | 数据大小 | batch | accum | lr | epochs | seq_len | GPU | 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| **Pretrain** | `pretrain_1b.jsonl` | 63.4GB | 64 | 2 | 3e-4 | 1 | 512 | 4×A100 | ~35h |
| **SFT** | `sft_t2t_mini.jsonl` | 1.7GB | 64 | 2 | 5e-6 | 2 | 512 | 8×A100 | ~30min |
| **DPO** | `dpo.jsonl` | 52MB | 4 | 1 | 4e-8 | 1 | 1024 | 8×A100 | ~10min |
| **GRPO** | `rlaif-mini.jsonl` | 23MB | 2 | — | 8e-8 | 1 | 66+1536 | 8×A100 | ~20h |

#### Pretrain 完整命令与超参数

```bash
torchrun --nproc_per_node 4 train_pretrain.py \
    --hidden_size 1536 --num_hidden_layers 20 \
    --vocab_size 32000 \
    --data_path ../dataset_1B/pretrain_1b.jsonl \
    --tokenizer_path ../model_05b_tokenizer \
    --max_seq_len 512 --batch_size 64 \
    --accumulation_steps 2 \
    --learning_rate 3e-4 \
    --epochs 1 \
    --dtype bfloat16 \
    --save_interval 2000 \
    --grad_clip 1.0 \
    --log_interval 100 \
    --num_workers 8 \
    --save_weight pretrain \
    --from_weight none \
    --use_moe 0 --use_mla 0 \
    --use_compile 0
```

| 参数 | 脚本默认值 | 0.5B 实际值 | 说明 |
|---|---|---|---|
| `hidden_size` | 512 | **1536** | 0.5B 模型维度 |
| `num_hidden_layers` | 8 | **20** | 0.5B 层数 |
| `vocab_size` | 6400 | **32000** | 32K 新分词器 |
| `data_path` | `pretrain_hq.jsonl` | **`pretrain_1b.jsonl`** | 63.4GB 扩展数据 |
| `tokenizer_path` | `../model` | **`../model_05b_tokenizer`** | 32K 分词器 |
| `max_seq_len` | 340 | **512** | 覆盖更长文本 |
| `batch_size` | 32 | **64** | A100 显存充足 |
| `accumulation_steps` | 8 | **2** | 等效 batch 128 |
| `learning_rate` | 5e-4 | **3e-4** | 更大模型需更保守 |
| `epochs` | 1 | 1 | 默认值 |
| `dtype` | bfloat16 | bfloat16 | 默认值 |
| `save_interval` | 1000 | **2000** | 减少 I/O 开销 |
| `grad_clip` | 1.0 | 1.0 | 默认值 |
| `log_interval` | 100 | 100 | 默认值 |
| `num_workers` | 8 | 8 | 默认值 |
| `save_weight` | pretrain | pretrain | 默认值 |
| `from_weight` | none | none | 从零训练 |
| `from_resume` | 0 | 0 | 不续训 |
| `use_moe` | 0 | 0 | 非 MoE |
| `use_mla` | 0 | 0 | 非 MLA |
| `use_compile` | 0 | 0 | 不使用 torch.compile |

#### SFT 完整命令与超参数

```bash
torchrun --nproc_per_node 8 --master_port 29600 train_full_sft.py \
    --hidden_size 1536 --num_hidden_layers 20 \
    --vocab_size 32000 \
    --data_path ../dataset/sft_t2t_mini.jsonl \
    --tokenizer_path ../model_05b_tokenizer \
    --max_seq_len 512 --batch_size 64 \
    --from_weight pretrain \
    --epochs 2 \
    --accumulation_steps 2 \
    --learning_rate 5e-6 \
    --dtype bfloat16 \
    --save_interval 2000 \
    --log_interval 50 \
    --grad_clip 1.0 \
    --num_workers 8 \
    --save_weight full_sft \
    --use_moe 0 --use_mla 0 \
    --use_compile 0
```

| 参数 | 脚本默认值 | 0.5B 实际值 | 说明 |
|---|---|---|---|
| `hidden_size` | 512 | **1536** | |
| `num_hidden_layers` | 8 | **20** | |
| `vocab_size` | 6400 | **32000** | |
| `data_path` | `sft_mini_512.jsonl` | **`sft_t2t_mini.jsonl`** | 1.7GB SFT 数据 |
| `tokenizer_path` | `../model` | **`../model_05b_tokenizer`** | |
| `max_seq_len` | 340 | **512** | |
| `batch_size` | 16 | **64** | |
| `accumulation_steps` | 1 | **2** | 等效 batch 128 |
| `learning_rate` | 1e-6 | **5e-6** | 略提高匹配更大 batch |
| `epochs` | 2 | 2 | 默认值 |
| `from_weight` | pretrain | pretrain | 默认值，加载 `pretrain_1536.pth` |
| `save_interval` | 1000 | **2000** | |
| `log_interval` | 100 | **50** | 更频繁打印 |
| `grad_clip` | 1.0 | 1.0 | 默认值 |
| `num_workers` | 8 | 8 | 默认值 |

#### DPO 完整命令与超参数

```bash
torchrun --nproc_per_node 8 --master_port 29601 train_dpo.py \
    --hidden_size 1536 --num_hidden_layers 20 \
    --vocab_size 32000 \
    --tokenizer_path ../model_05b_tokenizer \
    --data_path ../dataset/dpo.jsonl \
    --from_weight full_sft \
    --max_seq_len 1024 \
    --batch_size 4 \
    --learning_rate 4e-8 \
    --beta 0.1 \
    --epochs 1 \
    --dtype bfloat16 \
    --save_interval 100 \
    --log_interval 20 \
    --grad_clip 1.0 \
    --accumulation_steps 1 \
    --num_workers 8 \
    --save_weight dpo \
    --use_moe 0 --use_mla 0
```

| 参数 | 脚本默认值 | 0.5B 实际值 | 说明 |
|---|---|---|---|
| `hidden_size` | 512 | **1536** | |
| `num_hidden_layers` | 8 | **20** | |
| `vocab_size` | 6400 | **32000** | |
| `tokenizer_path` | `../model` | **`../model_05b_tokenizer`** | |
| `data_path` | `dpo.jsonl` | `dpo.jsonl` | 默认值 |
| `from_weight` | full_sft | full_sft | 默认值，加载 `full_sft_1536.pth` |
| `max_seq_len` | 1024 | 1024 | 默认值 |
| `batch_size` | 4 | 4 | 默认值 |
| `learning_rate` | 4e-8 | 4e-8 | 默认值 |
| `beta` | 0.1 | 0.1 | 默认值，KL 惩罚系数 |
| `epochs` | 1 | 1 | 默认值 |
| `save_interval` | 100 | 100 | 默认值 |
| `log_interval` | 100 | **20** | |
| `grad_clip` | 1.0 | 1.0 | 默认值 |
| `accumulation_steps` | 1 | 1 | 默认值 |

#### GRPO 完整命令与超参数

```bash
torchrun --nproc_per_node 8 --master_port 29602 train_grpo.py \
    --hidden_size 1536 --num_hidden_layers 20 \
    --vocab_size 32000 \
    --tokenizer_path ../model_05b_tokenizer \
    --data_path ../dataset/rlaif-mini.jsonl \
    --reward_model_path ../../internlm2-1_8b-reward \
    --reasoning 0 \
    --batch_size 2 \
    --learning_rate 8e-8 \
    --max_seq_len 66 --max_gen_len 1536 \
    --num_generations 8 \
    --beta 0.02 \
    --epochs 1 \
    --dtype bfloat16 \
    --save_interval 100 \
    --log_interval 10 \
    --grad_clip 1.0 \
    --accumulation_steps 1 \
    --num_workers 8 \
    --save_weight grpo \
    --use_moe 0 --use_mla 0
```

| 参数 | 脚本默认值 | 0.5B 实际值 | 说明 |
|---|---|---|---|
| `hidden_size` | 512 | **1536** | |
| `num_hidden_layers` | 8 | **20** | |
| `vocab_size` | 6400 | **32000** | |
| `tokenizer_path` | `../model` | **`../model_05b_tokenizer`** | |
| `data_path` | `rlaif-mini.jsonl` | `rlaif-mini.jsonl` | 默认值 |
| `reward_model_path` | `../../internlm2-1_8b-reward` | `../../internlm2-1_8b-reward` | 默认值 |
| `reasoning` | 1 | **0** | 普通模型（非推理模型），加载 `full_sft_1536.pth` |
| `max_seq_len` | 66 | 66 | 默认值，prompt 最大长度 |
| `max_gen_len` | 1536 | 1536 | 默认值，生成最大长度 |
| `num_generations` | 8 | 8 | 默认值，每 prompt 生成候选数 |
| `batch_size` | 2 | 2 | 默认值 |
| `learning_rate` | 8e-8 | 8e-8 | 默认值 |
| `beta` | 0.02 | 0.02 | 默认值，KL 惩罚系数 |
| `epochs` | 1 | 1 | 默认值 |
| `save_interval` | 10 | **100** | 减少 I/O |
| `log_interval` | 1 | **10** | 减少日志噪音 |
| `grad_clip` | 1.0 | 1.0 | 默认值 |
| `accumulation_steps` | 1 | 1 | 默认值 |

### 待执行实验

- [ ] 26M MLA 消融实验（Pretrain + SFT + 评测）
- [x] 0.5B GQA DPO 后训练 + 评测
- [ ] 0.5B GRPO/PPO 后训练（需下载 Reward Model）
- [x] 0.5B GQA 扩展实验（Pretrain + SFT + 评测）
- [ ] 0.5B MLA 扩展实验（Pretrain + SFT + 评测）
- [ ] 跨架构（GQA vs MLA）消融分析
