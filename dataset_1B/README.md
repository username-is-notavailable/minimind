# dataset_1B — 1B 模型预训练数据

## 数据规模

**~21.7B tokens**（6400 词表）/ **~13.9B tokens**（32K 词表）/ 34.6B 字符 / 32.3M 样本 / 63.4 GB

按 Chinchilla 最优比例（~20 tokens/参数），988M 参数的 1B 模型需要约 20B tokens。当前数据量满足该目标。

## 数据来源

数据通过两阶段准备：`prepare_pretrain_data.py`（基础采集）+ `expand_pretrain_data.py`（扩充至 20B tokens）。

| 类别 | 语言 | 条数 | 字符 | 数据源 | 说明 |
|---|---|---|---|---|---|
| 中文网页 | 中文 | 10,500,000 | 10.55B | `Skywork/SkyPile-150B` | 基础 3.5M + 扩充 7M |
| 中文百科 | 中文 | 1,270,195 | 901M | `wikimedia/wikipedia` (zh) | 全量 |
| 中文对话(SFT转) | 中文 | 5,831,466 | 6.52B | 本地 sft_t2t + sft_t2t_mini | 转为预训练格式 |
| 中文多样化 | 中文 | 1,000,000 | 1.02B | `Skywork/SkyPile-150B` (偏移) | 不同区间补充 |
| 中文新闻 | 中文 | 500,000 | 500M | `Skywork/SkyPile-150B` (偏移) | |
| 本地预训练 | 中文 | 8,429,917 | 3.24B | 现有 pretrain_hq | |
| 英文网页 | 英文 | 3,000,000 | 5.67B | `allenai/c4` (en) | 基础 1M + 扩充 2M |
| 英文百科 | 英文 | 500,000 | 1.50B | `wikimedia/wikipedia` (en) | |
| 英文学术 | 英文 | 1,300,000 | 4.73B | `open-web-math/open-web-math` | 基础 500K + 扩充 800K |
| 代码 | 混合 | 17,764 | 8M | `iamtarun/python_code_instructions_18k_alpaca` | 源数据集仅此规模 |

> **已知问题**：`CASIA-LM/ChineseWebText` 数据格式不兼容（text 字段为字符串化列表），采集 0 条。已通过增加 SkyPile 和 SFT 数据补偿。

**语言分布**：中文 ~65%，英文 ~34%，代码 ~1%

## 使用方式

```bash
cd minimind/dataset_1B

# 第一步：基础数据采集（~97 分钟）
nohup python prepare_pretrain_data.py > prepare.log 2>&1 &

# 第二步：扩充至 20B tokens（~184 分钟）
nohup python expand_pretrain_data.py > expand.log 2>&1 &

# 快速测试（每源 100 条，验证流程）
python prepare_pretrain_data.py --test_mode

# 查看进度
tail -f prepare.log  # 或 expand.log
```

## 输出文件

| 文件 | 大小 | 说明 | 用途 |
|---|---|---|---|
| `pretrain_1b.jsonl` | 63.4 GB | 合并打乱后的预训练数据（32.3M 行） | 模型预训练 |
| `tokenizer_train.jsonl` | 401 MB | 从预训练数据均匀采样的子集（20 万条） | 分词器训练 |
| `data_stats.json` | 3.4 KB | 各数据源条数、字符数统计 | 记录参考 |

## 数据格式

所有数据统一为 MiniMind 预训练格式：

```json
{"text": "这是一段预训练文本..."}
```

## 后续步骤

1. **训练分词器**：
```bash
cd ../trainer
python train_tokenizer_1b.py --data_path ../dataset_1B/tokenizer_train.jsonl --vocab_size 32000
```

2. **创建软链接**（让训练脚本能找到数据）：
```bash
cd ../dataset
ln -sf ../dataset_1B/pretrain_1b.jsonl pretrain_hq.jsonl
```

3. **开始训练**（0.5B 模型，hidden_size=1536, 20 层）：
```bash
cd ../trainer
torchrun --nproc_per_node 4 train_pretrain.py \
    --hidden_size 1536 --num_hidden_layers 20 \
    --batch_size 16 --max_seq_len 512
```

## 清洗规则

| 规则 | 中文 | 英文 | 代码 |
|---|---|---|---|
| 最小长度 | 50 字符 | 50 字符 | 50 字符 |
| 最大长度（截断） | 8192 字符 | 8192 字符 | 8192 字符 |
| 语言过滤 | 中文字符 >30% | 中文字符 <10% | 仅检测代码特征 |
| 重复检测 | 去重复行 >50% | 同左 | 不检测 |
| 乱码过滤 | 特殊字符 <5% | 同左 | 同左 |
