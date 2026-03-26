# MiniMind MLA 消融实验计划

## 实验目标

在 MiniMind2-Small (26M) 模型上，对比标准 GQA（Grouped Query Attention）与自定义 MLA（Multi-head Latent Attention）注意力机制的训练效果和推理效率差异。

---

## 环境信息

| 项目 | 配置 |
|---|---|
| GPU | 8 × NVIDIA A100-SXM4-80GB |
| CUDA | 12.8 |
| Python 环境 | `conda activate pre`（torch 2.8.0） |
| 分词器 | 已有（`model/tokenizer.json`，vocab_size=6400，无需重新训练） |
| 数据集 | 4 个核心数据集已下载并创建软链接 |

---

## 前置检查清单

- [x] MLA 代码集成到 `model/model_minimind.py`
- [x] 8 个训练脚本已添加 MLA CLI 参数
- [x] `eval_llm.py` 已支持 MLA 参数
- [x] `scripts/convert_model.py` 已支持 MLA 转换和 CLI 参数
- [x] 数据文件名软链接就绪（`pretrain_hq.jsonl` → `pretrain_t2t.jsonl` 等）
- [x] `.gitignore` 已忽略数据集和 checkpoints
- [ ] WandB 账号已登录（见下方训练可视化配置）

---

## 模型规格速查表

通过 `--hidden_size` 和 `--num_hidden_layers` 控制模型参数量，所有脚本统一：

| 模型名称 | 参数量 | `--hidden_size` | `--num_hidden_layers` | 权重文件名 | 说明 |
|---|---|---|---|---|---|
| MiniMind2-Small | **26M** | `512`（默认） | `8`（默认） | `*_512.pth` | 不传参数即为此配置 |
| MiniMind2-Base | **104M** | `768` | `16` | `*_768.pth` | |
| MiniMind2-1B | **~988M** | `2048` | `22` | `*_2048.pth` | 需补充预训练数据 |

**原则**：训练时用什么参数，后续加载/评估/转换时必须用完全相同的参数。

MLA 模型额外需要（以 26M 为例）：
```
--use_mla 1 --mla_kv_dim 128 --mla_q_dim 256 --mla_rope_dim 128
```

1B 模型的 MLA 维度按比例放大：
```
--use_mla 1 --mla_kv_dim 512 --mla_q_dim 1024 --mla_rope_dim 256
```

**脚本运行目录约定**：
- 训练脚本：在 `minimind/trainer/` 下运行（数据路径以 `../dataset/` 开头）
- eval_llm.py：在 `minimind/` 项目根目录运行
- benchmark 脚本：在 `minimind/benchmark/` 下运行
- convert_model.py：在 `minimind/scripts/` 下运行

---

## 训练可视化配置（WandB）

> 已将原始仓库的 SwanLab 替换为原生 [Weights & Biases (wandb)](https://wandb.ai/)。

### 设置步骤

```bash
conda activate pre
pip install wandb -q

# 登录 wandb
wandb login
# 输入你的 API Key（从 https://wandb.ai/authorize 获取）
```

### 训练脚本启用方式

所有训练脚本添加 `--use_wandb` 参数即可启用：

```bash
python train_pretrain.py --use_wandb
python train_full_sft.py --use_wandb
```

### 自动记录的指标

| 指标名 | 含义 | 记录频率 |
|---|---|---|
| `loss` | 总损失（含 aux_loss） | 每 `log_interval` 步 |
| `logits_loss` | 语言建模损失（= loss - aux_loss） | 同上 |
| `aux_loss` | MoE 负载均衡损失（非 MoE 模型为 0） | 同上 |
| `learning_rate` | 当前学习率 | 同上 |
| `epoch_time` | 预估剩余训练时间（分钟） | 同上 |

### 项目命名规范

为便于对比，建议使用以下 wandb project 命名：

| 实验 | `--wandb_project` 参数 |
|---|---|
| GQA Baseline 预训练 | `"MiniMind-Pretrain-GQA"` |
| GQA Baseline SFT | `"MiniMind-SFT-GQA"` |
| MLA 预训练 | `"MiniMind-Pretrain-MLA"` |
| MLA SFT | `"MiniMind-SFT-MLA"` |
| DPO | `"MiniMind-DPO"` |
| GRPO | `"MiniMind-GRPO"` |

示例：
```bash
# GQA Baseline 预训练
torchrun --nproc_per_node 4 train_pretrain.py --use_wandb --wandb_project "MiniMind-Pretrain-GQA"

# MLA 预训练
torchrun --nproc_per_node 4 train_pretrain.py --use_mla 1 --use_wandb --wandb_project "MiniMind-Pretrain-MLA"
```

### 断点续训时的 wandb 恢复

训练脚本在 checkpoint 中保存了 `wandb_id`。使用 `--from_resume 1` 续训时，会自动恢复到同一个 wandb run，保持曲线连续。

## 阶段一：Baseline 训练（标准 GQA）

### 模型配置

| 参数 | 值 |
|---|---|
| hidden_size | 512 |
| num_hidden_layers | 8 |
| num_attention_heads | 8 |
| num_key_value_heads | 2 |
| vocab_size | 6400 |
| use_mla | False |
| use_moe | False |

### 步骤 1.1 — 预训练

```bash
conda activate pre
cd minimind/trainer
torchrun --nproc_per_node 4 train_pretrain.py --use_wandb --wandb_project "MiniMind-Pretrain-GQA"
```

- **数据**：`pretrain_hq.jsonl`（7.8GB），max_seq_len=340
- **优化器**：AdamW，lr=5e-4，余弦退火
- **混合精度**：bfloat16
- **梯度累积**：8 步
- **预计耗时**：~20min（4×A100）
- **产出**：`out/pretrain_512.pth`

### 步骤 1.2 — SFT 微调

```bash
torchrun --nproc_per_node 4 train_full_sft.py --use_wandb --wandb_project "MiniMind-SFT-GQA"
```

- **数据**：`sft_mini_512.jsonl`（1.6GB），max_seq_len=340
- **起点**：加载 `pretrain_512.pth`
- **学习率**：1e-6
- **预计耗时**：~15min（4×A100）
- **产出**：`out/full_sft_512.pth`

### 步骤 1.3 — 评估

```bash
cd minimind  # 回到项目根目录
python eval_llm.py --weight full_sft --show_speed 1
```

- 选择 `[0] 自动测试`，固定种子 2026
- 记录 8 个预设 prompt 的输出文本
- 记录 tokens/s

### 步骤 1.4 — 备份权重

```bash
mkdir -p out/baseline
cp out/pretrain_512.pth out/baseline/
cp out/full_sft_512.pth out/baseline/
```

---

## 阶段二：MLA 训练

### 模型配置

| 参数 | 值 |
|---|---|
| hidden_size | 512 |
| num_hidden_layers | 8 |
| num_attention_heads | 8 |
| num_key_value_heads | 2 |
| vocab_size | 6400 |
| **use_mla** | **True** |
| **mla_kv_dim** | **128** |
| **mla_q_dim** | **256** |
| **mla_rope_dim** | **128** |

### 步骤 2.1 — 预训练

```bash
cd minimind/trainer  # 确保在 trainer 目录
torchrun --nproc_per_node 4 train_pretrain.py \
    --use_mla 1 --mla_kv_dim 128 --mla_q_dim 256 --mla_rope_dim 128 \
    --use_wandb --wandb_project "MiniMind-Pretrain-MLA"
```

- 其余超参与阶段一完全一致
- **产出**：`out/pretrain_512.pth`（会覆盖，需先完成步骤 1.4 备份）

### 步骤 2.2 — SFT 微调

```bash
torchrun --nproc_per_node 4 train_full_sft.py \
    --use_mla 1 --mla_kv_dim 128 --mla_q_dim 256 --mla_rope_dim 128 \
    --use_wandb --wandb_project "MiniMind-SFT-MLA"
```

### 步骤 2.3 — 评估

```bash
cd minimind  # 回到项目根目录
python eval_llm.py --weight full_sft --use_mla 1 --show_speed 1
```

- 同样选择 `[0] 自动测试`，固定种子 2026
- 记录相同 8 个 prompt 的输出文本
- 记录 tokens/s

### 步骤 2.4 — 备份权重

```bash
mkdir -p out/mla
cp out/pretrain_512.pth out/mla/
cp out/full_sft_512.pth out/mla/
```

---

## 阶段三：量化评估与消融对比

> 使用 `eval_benchmark.py` 进行全自动量化评估

### 3.0 量化评估

训练完成后，对两组模型分别运行：

```bash
cd minimind/benchmark  # 在 benchmark 目录下运行

# GQA Baseline 全面评估
python run_all.py --weight full_sft --save_dir out/baseline

# MLA 全面评估
python run_all.py --weight full_sft --save_dir out/mla --use_mla 1
```

评估脚本自动输出 4 类量化指标：

| 任务 | 说明 | 指标 |
|---|---|---|
| `ppl` | 在 SFT 数据 held-out 集上计算困惑度 | Perplexity、Avg Loss |
| `mcq` | 20 道中文选择题（概率法） | 准确率 |
| `gen` | 8 个固定 prompt 生成+自动评分 | 平均分（0-100）、关键词命中、流畅度 |
| `eff` | 不同输入长度的推理效率 | tokens/s、峰值显存 |

结果自动保存到 `eval_results/` 目录。

### 3.1 定量指标对比

| 指标 | 方法 | 工具 |
|---|---|---|
| **参数量** | 训练启动时自动打印 | 脚本内置 `get_model_params()` |
| **Perplexity** | held-out 数据集 | `eval_benchmark.py --tasks ppl` |
| **选择题准确率** | 20 道中文选择题 | `eval_benchmark.py --tasks mcq` |
| **生成质量评分** | 关键词+流畅度+格式自动评分 | `eval_benchmark.py --tasks gen` |
| **预训练 Loss** | wandb 曲线叠加 | wandb UI |
| **SFT Loss** | wandb 曲线叠加 | wandb UI |
| **推理速度** | 多种输入长度测试 | `eval_benchmark.py --tasks eff` |
| **显存占用** | 推理峰值显存 | `eval_benchmark.py --tasks eff` |

### 3.2 对话质量对比

使用以下 8 个固定 prompt（`eval_benchmark.py` 内置），种子固定为 2026：

1. 你有什么特长？
2. 为什么天空是蓝色的
3. 请用Python写一个计算斐波那契数列的函数
4. 解释一下"光合作用"的基本过程
5. 如果明天下雨，我应该如何出门
6. 比较一下猫和狗作为宠物的优缺点
7. 解释什么是机器学习
8. 推荐一些中国的美食

**评估方式**：
- 人工主观评分（准确性、完整性、逻辑性）
- 可选：将两组输出提交给 GPT-4/DeepSeek-R1 进行盲评打分

### 3.3 记录模板

```
| 指标 | GQA Baseline | MLA |
|---|---|---|
| 总参数量 | ？M | ？M |
| 预训练最终 Loss | ？ | ？ |
| SFT 最终 Loss | ？ | ？ |
| 推理速度 (tokens/s) | ？ | ？ |
| 训练显存峰值 (MB) | ？ | ？ |
| 推理显存峰值 (MB) | ？ | ？ |
```

---

## 阶段四（可选）：RLHF / RLAIF 后训练

基于阶段三中表现更好的模型变体，继续后训练：

### 4.1 DPO（离线偏好优化）

```bash
cd trainer
torchrun --nproc_per_node 4 train_dpo.py [--use_mla 1 ...] --use_wandb
```

- **数据**：`dpo.jsonl`（52MB）
- **起点**：`full_sft_512.pth`
- **学习率**：4e-8
- **产出**：`out/dpo_512.pth`

### 4.2 GRPO（在线强化学习）

需要额外下载 Reward Model（InternLM2-1.8B-Reward，放在 minimind 同级目录）。

```bash
torchrun --nproc_per_node 4 train_grpo.py [--use_mla 1 ...] --use_wandb
```

- **数据**：`rlaif-mini.jsonl`（23MB）
- **起点**：`full_sft_512.pth` 或 `dpo_512.pth`
- **产出**：`out/grpo_512.pth`

---

## 阶段五（可选）：模型转换与上传

### 5.1 转换

```bash
cd minimind/scripts  # 在 scripts 目录下运行

# GQA Baseline → Llama 格式（兼容 llama.cpp/vllm/ollama）
python convert_model.py --weight full_sft --input_dir ../out/baseline --output_dir ../MiniMind2-Small-GQA

# MLA 模型 → MiniMind 原生格式（需 trust_remote_code=True）
python convert_model.py --weight full_sft --use_mla 1 --input_dir ../out/mla --output_dir ../MiniMind2-Small-MLA
```

### 5.2 上传到 HuggingFace

```python
from huggingface_hub import HfApi

api = HfApi()

# 上传 GQA 模型
api.create_repo("your-username/MiniMind2-Small-GQA", repo_type="model")
api.upload_folder(folder_path="./MiniMind2-Small-GQA", repo_id="your-username/MiniMind2-Small-GQA")

# 上传 MLA 模型
api.create_repo("your-username/MiniMind2-Small-MLA", repo_type="model")
api.upload_folder(folder_path="./MiniMind2-Small-MLA", repo_id="your-username/MiniMind2-Small-MLA")
```

### 5.3 验证下载加载

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# GQA — 直接加载
model = AutoModelForCausalLM.from_pretrained("your-username/MiniMind2-Small-GQA")

# MLA — 需要 trust_remote_code
model = AutoModelForCausalLM.from_pretrained("your-username/MiniMind2-Small-MLA", trust_remote_code=True)
```

---

## 执行时间线

```
Day 1 — 核心实验 (26M 模型):
  09:00  阶段一 步骤1.1  预训练 Baseline         (~20min)
  09:25  阶段一 步骤1.2  SFT Baseline            (~15min)
  09:45  阶段一 步骤1.3  评估 + 步骤1.4 备份
  10:00  阶段二 步骤2.1  预训练 MLA              (~20min)
  10:25  阶段二 步骤2.2  SFT MLA                (~15min)
  10:45  阶段二 步骤2.3  评估 + 步骤2.4 备份
  11:00  阶段三         消融对比分析 + 记录结果

Day 2 — 可选扩展:
  阶段四  DPO / GRPO 后训练
  阶段五  模型转换 + HuggingFace 上传

Day 3 — 1B 扩展实验 (需先准备数据):
  阶段六 步骤6.1  补充预训练数据
  阶段六 步骤6.2  1B GQA 预训练      (~2-3h, 4×A100)
  阶段六 步骤6.3  1B MLA 预训练      (~2-3h, 4×A100)
  阶段六 步骤6.4  1B SFT + 评估
  阶段六 步骤6.5  跨规模消融分析
```

> 注：以上时间基于 4×A100 估算。使用全部 8 卡（`--nproc_per_node 8`）可进一步缩短至约一半时间。

---

## 阶段六：1B 模型扩展实验

### 6.0 背景与可行性

将模型从 26M/104M 扩展到 ~1B，验证 MLA 在更大参数量下的效果差异。

**可行性分析**：

| 维度 | 结论 |
|---|---|
| 算力 | ✅ 8×A100-80GB 绰绰有余（单卡即可放下 1B 训练） |
| 代码 | ✅ 无需修改，通过 CLI 参数即可调整 |
| 训练时间 | ✅ 4×A100 约 2-3h（可接受） |
| **数据量** | **⚠️ 现有 0.9B tokens 仅为 Chinchilla 最优量的 4.5%，需补充数据** |

### 6.1 数据准备

**最低目标**：将预训练数据从 0.9B tokens 扩充到 **5-10B tokens**。

**推荐数据源**（HuggingFace 开源中文语料）：

| 数据集 | 来源 | 规模 | 说明 |
|---|---|---|---|
| WanJuan-CC | `opendatalab/WanJuan-CC` | ~100B tokens | 清洗后的中文网页数据（可取子集） |
| SkyPile | `Skywork/SkyPile-150B` | ~150B tokens | 中文网页（取前 10GB 即可） |
| Chinese-Web-Text | `CASIA-LM/ChineseWebText` | ~50B tokens | 高质量中文网络文本 |

**数据准备流程**：

```bash
# 1. 下载数据（以 WanJuan 为例，取子集）
python -c "
from datasets import load_dataset
ds = load_dataset('opendatalab/WanJuan-CC', split='train', streaming=True)
import json, os
os.makedirs('../dataset', exist_ok=True)
with open('../dataset/pretrain_extra.jsonl', 'w') as f:
    for i, item in enumerate(ds):
        if i >= 5000000:  # 取 500 万条
            break
        f.write(json.dumps({'text': item['content']}, ensure_ascii=False) + '\n')
"

# 2. 合并数据
cat ../dataset/pretrain_hq.jsonl ../dataset/pretrain_extra.jsonl > ../dataset/pretrain_1b.jsonl

# 3. 创建软链接（供脚本使用）
cd ../dataset && ln -sf pretrain_1b.jsonl pretrain_hq.jsonl
```

> 注意：下载大规模数据可能需要较长时间，建议提前准备。

### 6.2 模型配置

**推荐的 1B 配置**（参考 Llama 架构比例）：

| 参数 | 值 | 说明 |
|---|---|---|
| `hidden_size` | 2048 | |
| `num_hidden_layers` | 22 | |
| `num_attention_heads` | 16 | |
| `num_key_value_heads` | 4 | GQA 分组 |
| `vocab_size` | 6400 | 保持不变 |
| **总参数量** | **~988M** | |
| fp16 显存 | ~1.8GB | |
| 训练显存 | ~18GB | 单卡 A100 即可 |

### 6.3 训练步骤

**步骤 6.3.1 — 1B GQA 预训练**

```bash
conda activate pre
cd minimind/trainer
torchrun --nproc_per_node 4 train_pretrain.py \
    --hidden_size 2048 --num_hidden_layers 22 \
    --data_path ../dataset/pretrain_1b.jsonl \
    --max_seq_len 512 \
    --batch_size 8 \
    --use_wandb --wandb_project "MiniMind-Pretrain-1B-GQA"
```

**步骤 6.3.2 — 1B GQA SFT**

```bash
torchrun --nproc_per_node 4 train_full_sft.py \
    --hidden_size 2048 --num_hidden_layers 22 \
    --use_wandb --wandb_project "MiniMind-SFT-1B-GQA"
```

**步骤 6.3.3 — 备份 1B GQA 权重**

```bash
cd minimind  # 回到项目根目录
mkdir -p out/1b_baseline
cp out/pretrain_2048.pth out/1b_baseline/
cp out/full_sft_2048.pth out/1b_baseline/
```

**步骤 6.3.4 — 1B MLA 预训练**

```bash
cd minimind/trainer  # 确保在 trainer 目录
torchrun --nproc_per_node 4 train_pretrain.py \
    --hidden_size 2048 --num_hidden_layers 22 \
    --use_mla 1 --mla_kv_dim 512 --mla_q_dim 1024 --mla_rope_dim 256 \
    --data_path ../dataset/pretrain_1b.jsonl \
    --max_seq_len 512 \
    --batch_size 8 \
    --use_wandb --wandb_project "MiniMind-Pretrain-1B-MLA"
```

> MLA 维度按比例放大：kv_dim=hidden_size/4, q_dim=hidden_size/2, rope_dim=hidden_size/8

**步骤 6.3.5 — 1B MLA SFT**

```bash
torchrun --nproc_per_node 4 train_full_sft.py \
    --hidden_size 2048 --num_hidden_layers 22 \
    --use_mla 1 --mla_kv_dim 512 --mla_q_dim 1024 --mla_rope_dim 256 \
    --use_wandb --wandb_project "MiniMind-SFT-1B-MLA"
```

**步骤 6.3.6 — 备份 1B MLA 权重**

```bash
cd minimind  # 回到项目根目录
mkdir -p out/1b_mla
cp out/pretrain_2048.pth out/1b_mla/
cp out/full_sft_2048.pth out/1b_mla/
```

### 6.4 评估

```bash
cd minimind/benchmark  # 在 benchmark 目录下运行

# 1B GQA 全面评测
python run_all.py --weight full_sft --save_dir out/1b_baseline \
    --hidden_size 2048 --num_hidden_layers 22

# 1B MLA 全面评测
python run_all.py --weight full_sft --save_dir out/1b_mla \
    --hidden_size 2048 --num_hidden_layers 22 \
    --use_mla 1 --mla_kv_dim 512 --mla_q_dim 1024 --mla_rope_dim 256
```

### 6.5 跨规模消融记录

> 对比 26M 和 1B 在 GQA/MLA 下的表现，验证 MLA 的优势是否随参数量增大而放大

| 指标 | 26M GQA | 26M MLA | 1B GQA | 1B MLA |
|---|---|---|---|---|
| 参数量 | 25.8M | ____M | ~988M | ____M |
| PPL ↓ | ____ | ____ | ____ | ____ |
| C-Eval 准确率 ↑ | ____ | ____ | ____ | ____ |
| 生成评分 ↑ | ____ | ____ | ____ | ____ |
| 推理速度 (tok/s) | ____ | ____ | ____ | ____ |
| 推理显存 (MB) | ____ | ____ | ____ | ____ |

---

## 风险与注意事项

1. **权重覆盖**：阶段一和阶段二的输出文件名相同（`pretrain_512.pth` / `full_sft_512.pth`），**务必在阶段二开始前完成步骤 1.4 的备份**
2. **Python 环境**：当前机器默认 Python 没有 torch，需使用 `conda activate pre`
3. **wandb**：如需记录训练曲线，确保已登录（`wandb login`）
4. **Reward Model**：阶段四 GRPO 需要额外下载 InternLM2-1.8B-Reward 模型（约 3.6GB），放在 minimind 同级目录
5. **MLA 与 Llama 不兼容**：MLA 模型只能通过 MiniMind 原生格式上传 HuggingFace，无法转为 Llama 格式
6. **1B 数据量风险**：现有 0.9B tokens 训练 1B 模型会严重过拟合，务必先补充预训练数据至 5B+ tokens
7. **1B MLA 维度选择**：MLA 的压缩维度需要按 hidden_size 比例放大（kv_dim≈d/4, q_dim≈d/2, rope_dim≈d/8），过小会损失表达能力
8. **1B 权重文件名**：1B 模型权重文件名为 `*_2048.pth`（因 hidden_size=2048），与 26M 模型的 `*_512.pth` 不冲突

---

## 实验结果记录

### 参数量对比

| 指标 | GQA Baseline | MLA |
|---|---|---|
| 总参数量 | ____M | ____M |
| 注意力层参数量 | ____M | ____M |
| 新增投影层 | 0 | 4 个 (mla_kv_proj + mla_q_proj + q_rope_proj + k_rope_proj) |

### 预训练结果

| 指标 | GQA Baseline | MLA |
|---|---|---|
| 最终 Loss | ____ | ____ |
| 最终 Logits Loss | ____ | ____ |
| 最终 Aux Loss | ____ | ____ |
| 训练总时间 | ____min | ____min |
| GPU 数量 | ____ | ____ |
| 显存峰值 (MB) | ____ | ____ |
| WandB 链接 | [WandB 链接]() | [WandB 链接]() |

### SFT 结果

| 指标 | GQA Baseline | MLA |
|---|---|---|
| 最终 Loss | ____ | ____ |
| 最终 Logits Loss | ____ | ____ |
| 训练总时间 | ____min | ____min |
| 显存峰值 (MB) | ____ | ____ |
| WandB 链接 | [WandB 链接]() | [WandB 链接]() |

### 推理性能对比

| 指标 | GQA Baseline | MLA |
|---|---|---|
| 推理速度 (tokens/s) | ____ | ____ |
| 推理显存 (MB) | ____ | ____ |
| 首 Token 延迟 (ms) | ____ | ____ |

### 量化评测对比（eval_benchmark.py）

| 指标 | GQA Baseline | MLA |
|---|---|---|
| **Perplexity (PPL)** ↓ | ____ | ____ |
| **选择题准确率** ↑ | ____% (___/20) | ____% (___/20) |
| **生成质量评分** ↑ | ____/100 | ____/100 |
| 生成速度 (tokens/s) | ____ | ____ |
| 推理显存峰值 (MB) | ____ | ____ |

### 对话质量对比

> 种子固定 2026，使用 eval_llm.py 内置的 8 个 prompt

#### Prompt 1: 你有什么特长？

**GQA Baseline**:
```
（待填写）
```

**MLA**:
```
（待填写）
```

#### Prompt 2: 为什么天空是蓝色的

**GQA Baseline**:
```
（待填写）
```

**MLA**:
```
（待填写）
```

#### Prompt 3: 请用Python写一个计算斐波那契数列的函数

**GQA Baseline**:
```
（待填写）
```

**MLA**:
```
（待填写）
```

#### Prompt 4: 解释一下“光合作用”的基本过程

**GQA Baseline**:
```
（待填写）
```

**MLA**:
```
（待填写）
```

#### Prompt 5: 如果明天下雨，我应该如何出门

**GQA Baseline**:
```
（待填写）
```

**MLA**:
```
（待填写）
```

#### Prompt 6: 比较一下猫和狗作为宠物的优缺点

**GQA Baseline**:
```
（待填写）
```

**MLA**:
```
（待填写）
```

#### Prompt 7: 解释什么是机器学习

**GQA Baseline**:
```
（待填写）
```

**MLA**:
```
（待填写）
```

#### Prompt 8: 推荐一些中国的美食

**GQA Baseline**:
```
（待填写）
```

**MLA**:
```
（待填写）
```

### 综合评价

| 维度 | GQA Baseline | MLA | 结论 |
|---|---|---|---|
| 参数效率 | ____ | ____ | |
| 训练收敛速度 | ____ | ____ | |
| 推理速度 | ____ | ____ | |
| 显存占用 | ____ | ____ | |
| 对话质量 | ____ | ____ | |
| **总体结论** | | | |

### Loss 曲线截图

> 曲线截图从 WandB Dashboard 下载，放到 `images/` 目录

| 阶段 | 截图 |
|---|---|
| 预训练 Loss 对比 | （待添加） |
| SFT Loss 对比 | （待添加） |
| Learning Rate 曲线 | （待添加） |
