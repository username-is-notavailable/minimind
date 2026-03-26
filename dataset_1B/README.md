# dataset_1B — 1B 模型预训练数据

## 数据配比

目标 **10B tokens**，中英混合（中文 ~75%，英文 ~20%，代码 ~5%）：

| 类别 | 语言 | 占比 | HuggingFace 数据源 |
|---|---|---|---|
| 中文网页 | 中文 | 35% | `Skywork/SkyPile-150B` |
| 中文百科/知识 | 中文 | 15% | `wikipedia/zh` + 现有 pretrain_hq |
| 中文对话 | 中文 | 10% | 现有 sft_t2t_mini |
| 中文文学 | 中文 | 10% | `CASIA-LM/ChineseWebText` |
| 英文网页 | 英文 | 10% | `allenai/c4` (en) |
| 英文百科 | 英文 | 5% | `wikipedia/en` |
| 代码 | 混合 | 5% | `bigcode/starcoderdata` (python) |
| 英文学术 | 英文 | 5% | `open-web-math/open-web-math` |
| 中文新闻 | 中文 | 5% | `Skywork/SkyPile-150B` (偏移子集) |

## 使用方式

```bash
cd minimind/dataset_1B

# 完整下载（耗时较长，建议后台运行）
nohup python prepare_pretrain_data.py > prepare.log 2>&1 &

# 快速测试（每源 100 条，验证流程）
python prepare_pretrain_data.py --test_mode

# 仅下载指定数据源
python prepare_pretrain_data.py --sources chinese_web,english_web,local

# 查看进度
tail -f prepare.log
```

## 输出文件

| 文件 | 说明 | 用途 |
|---|---|---|
| `pretrain_1b.jsonl` | 合并打乱后的预训练数据 | 模型预训练 |
| `tokenizer_train.jsonl` | 从预训练数据均匀采样的子集（20万条） | 分词器训练 |
| `data_stats.json` | 数据统计信息 | 记录各数据源条数、字符数 |

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

3. **开始训练**：
```bash
cd ../trainer
torchrun --nproc_per_node 4 train_pretrain.py \
    --hidden_size 2048 --num_hidden_layers 22 \
    --batch_size 8 --max_seq_len 512
```

## 清洗规则

| 规则 | 中文 | 英文 | 代码 |
|---|---|---|---|
| 最小长度 | 50 字符 | 50 字符 | 50 字符 |
| 最大长度（截断） | 8192 字符 | 8192 字符 | 8192 字符 |
| 语言过滤 | 中文字符 >30% | 中文字符 <10% | 仅检测代码特征 |
| 重复检测 | 去重复行 >50% | 同左 | 不检测 |
| 乱码过滤 | 特殊字符 <5% | 同左 | 同左 |
