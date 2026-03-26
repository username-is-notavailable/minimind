"""
prepare_pretrain_data.py — 1B 模型预训练数据下载、清洗与合并

输出：
  - dataset_1B/pretrain_1b.jsonl        — 合并后的预训练数据（供模型训练）
  - dataset_1B/tokenizer_train.jsonl    — 分词器训练用数据（采样子集）
  - dataset_1B/data_stats.json          — 数据统计信息

数据配比（10B tokens 目标，中文~75% 英文~20% 代码~5%）：
  ┌─────────────────┬──────┬──────────┬────────────────────────────────┐
  │ 类别             │ 占比  │ 采样条数   │ HuggingFace 数据源              │
  ├─────────────────┼──────┼──────────┼────────────────────────────────┤
  │ 中文网页          │ 35%  │ 3,500,000 │ Skywork/SkyPile-150B           │
  │ 中文百科/知识     │ 15%  │ 现有+wiki │ 现有 pretrain_hq + wikipedia/zh │
  │ 中文对话          │ 10%  │ 现有全部   │ 现有 sft_t2t_mini              │
  │ 中文文学          │ 10%  │ 1,000,000 │ CASIA-LM/ChineseWebText        │
  │ 英文网页          │ 10%  │ 1,000,000 │ allenai/c4 (en)                │
  │ 英文百科          │  5%  │ wiki全部   │ wikipedia/en (20220301)         │
  │ 代码             │  5%  │ 500,000   │ bigcode/starcoderdata (python)  │
  │ 英文学术          │  5%  │ 500,000   │ open-web-math/open-web-math    │
  │ 中文新闻/专业     │  5%  │ 500,000   │ Skywork/SkyPile-150B (子集)     │
  └─────────────────┴──────┴──────────┴────────────────────────────────┘

用法：
    cd minimind/dataset_1B

    # 下载并处理所有数据（完整版，耗时可能较长）
    python prepare_pretrain_data.py

    # 快速测试（每个源只取 1000 条）
    python prepare_pretrain_data.py --test_mode

    # 仅下载指定数据源
    python prepare_pretrain_data.py --sources chinese_web,english_web

    # 自定义输出
    python prepare_pretrain_data.py --output_dir ./my_data --tokenizer_samples 200000
"""

import os
import sys
import json
import time
import random
import argparse
import re
from collections import defaultdict


# ============================================================
# 数据源配置
# ============================================================

DATA_SOURCES = {
    'chinese_web': {
        'name': '中文网页',
        'hf_repo': 'Skywork/SkyPile-150B',
        'hf_config': None,
        'hf_split': 'train',
        'text_field': 'text',
        'max_samples': 3_500_000,
        'lang': 'zh',
        'category': '中文网页',
    },
    'chinese_wiki': {
        'name': '中文百科',
        'hf_repo': 'wikipedia',
        'hf_config': '20220301.zh',
        'hf_split': 'train',
        'text_field': 'text',
        'max_samples': 0,  # 0 = 全部
        'lang': 'zh',
        'category': '中文百科',
    },
    'chinese_literature': {
        'name': '中文文学',
        'hf_repo': 'CASIA-LM/ChineseWebText',
        'hf_config': None,
        'hf_split': 'train',
        'text_field': 'text',
        'max_samples': 1_000_000,
        'lang': 'zh',
        'category': '中文文学',
    },
    'english_web': {
        'name': '英文网页',
        'hf_repo': 'allenai/c4',
        'hf_config': 'en',
        'hf_split': 'train',
        'text_field': 'text',
        'max_samples': 1_000_000,
        'lang': 'en',
        'category': '英文网页',
    },
    'english_wiki': {
        'name': '英文百科',
        'hf_repo': 'wikipedia',
        'hf_config': '20220301.en',
        'hf_split': 'train',
        'text_field': 'text',
        'max_samples': 500_000,
        'lang': 'en',
        'category': '英文百科',
    },
    'code': {
        'name': '代码(Python)',
        'hf_repo': 'bigcode/starcoderdata',
        'hf_config': 'python',
        'hf_split': 'train',
        'text_field': 'content',
        'max_samples': 500_000,
        'lang': 'code',
        'category': '代码',
    },
    'english_academic': {
        'name': '英文学术',
        'hf_repo': 'open-web-math/open-web-math',
        'hf_config': None,
        'hf_split': 'train',
        'text_field': 'text',
        'max_samples': 500_000,
        'lang': 'en',
        'category': '英文学术',
    },
    'chinese_news': {
        'name': '中文新闻/专业',
        'hf_repo': 'Skywork/SkyPile-150B',
        'hf_config': None,
        'hf_split': 'train',
        'text_field': 'text',
        'max_samples': 500_000,
        'lang': 'zh',
        'category': '中文新闻',
        'offset': 5_000_000,  # 跳过前 500 万条（避免与 chinese_web 重复）
    },
}


# ============================================================
# 文本清洗
# ============================================================

def is_chinese_char(c):
    cp = ord(c)
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F or
            0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or
            0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


def chinese_char_ratio(text):
    if not text:
        return 0
    cn_count = sum(1 for c in text if is_chinese_char(c))
    return cn_count / len(text)


def clean_text(text, lang='zh', min_length=50, max_length=8192):
    """通用文本清洗"""
    if not text or not isinstance(text, str):
        return None

    # 去除首尾空白
    text = text.strip()

    # 长度过滤
    if len(text) < min_length:
        return None
    if len(text) > max_length:
        text = text[:max_length]

    # 语言过滤
    if lang == 'zh':
        if chinese_char_ratio(text) < 0.3:
            return None
    elif lang == 'en':
        if chinese_char_ratio(text) > 0.1:
            return None

    # 重复行检测（粗略）
    lines = text.split('\n')
    if len(lines) > 3:
        unique_lines = set(lines)
        if len(unique_lines) / len(lines) < 0.5:
            return None

    # 乱码检测（连续特殊字符过多）
    special_count = sum(1 for c in text if ord(c) > 0xFFFF or c in '\x00\x01\x02\x03')
    if special_count > len(text) * 0.05:
        return None

    return text


def clean_code(text, min_length=50, max_length=8192):
    """代码清洗"""
    if not text or not isinstance(text, str):
        return None

    text = text.strip()
    if len(text) < min_length:
        return None
    if len(text) > max_length:
        text = text[:max_length]

    # 至少包含一些代码特征
    code_indicators = ['def ', 'class ', 'import ', 'return ', 'if ', 'for ', '= ', '(', ')']
    if not any(ind in text for ind in code_indicators):
        return None

    return text


def process_sft_to_pretrain(conversations):
    """将 SFT 对话格式转为预训练文本"""
    parts = []
    for turn in conversations:
        role = turn.get('role', '')
        content = turn.get('content', '')
        if content:
            parts.append(content)
    text = '\n'.join(parts)
    return text if len(text) > 50 else None


# ============================================================
# 数据下载与处理
# ============================================================

def download_and_process_source(source_key, source_config, output_file, test_mode=False):
    """下载并处理单个数据源"""
    from datasets import load_dataset

    name = source_config['name']
    max_samples = 100 if test_mode else source_config['max_samples']
    offset = source_config.get('offset', 0)
    lang = source_config['lang']

    print(f"\n{'─' * 50}")
    print(f"📥 下载: {name}")
    print(f"   源: {source_config['hf_repo']}")
    print(f"   目标条数: {'全部' if max_samples == 0 else max_samples}")
    if offset:
        print(f"   偏移: 跳过前 {offset} 条")

    st = time.time()
    count = 0
    skipped = 0
    total_chars = 0

    try:
        ds = load_dataset(
            source_config['hf_repo'],
            source_config['hf_config'],
            split=source_config['hf_split'],
            streaming=True,
        )

        with open(output_file, 'a', encoding='utf-8') as f:
            for i, item in enumerate(ds):
                # 跳过 offset
                if i < offset:
                    continue

                # 达到目标数量
                if max_samples > 0 and count >= max_samples:
                    break

                # 提取文本
                text_field = source_config['text_field']
                raw_text = item.get(text_field, '')

                # 清洗
                if lang == 'code':
                    text = clean_code(raw_text)
                else:
                    text = clean_text(raw_text, lang=lang)

                if text is None:
                    skipped += 1
                    continue

                # 写入
                f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                count += 1
                total_chars += len(text)

                # 进度显示
                if count % 100_000 == 0:
                    elapsed = time.time() - st
                    speed = count / elapsed
                    print(f"   [{count:>10,}] 条 | 跳过 {skipped:,} | "
                          f"{total_chars / 1e9:.2f}GB | {speed:.0f} 条/s")

    except Exception as e:
        print(f"   ⚠️ 错误: {e}")
        print(f"   已保存 {count:,} 条，继续后续数据源")

    elapsed = time.time() - st
    print(f"   ✅ 完成: {count:,} 条 | 跳过 {skipped:,} | "
          f"{total_chars / 1e6:.1f}MB | {elapsed:.1f}s")

    return {
        'source': source_key,
        'name': name,
        'count': count,
        'skipped': skipped,
        'chars': total_chars,
        'time_sec': round(elapsed, 1),
    }


def process_local_files(output_file, dataset_dir='../dataset'):
    """处理现有本地数据文件"""
    stats = []

    # 1. 现有 pretrain_hq.jsonl
    pretrain_path = os.path.join(dataset_dir, 'pretrain_hq.jsonl')
    if os.path.exists(pretrain_path):
        print(f"\n{'─' * 50}")
        print(f"📂 处理本地文件: pretrain_hq.jsonl")
        count = 0
        total_chars = 0
        with open(pretrain_path, 'r', encoding='utf-8') as fin, \
             open(output_file, 'a', encoding='utf-8') as fout:
            for line in fin:
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '')
                    if len(text) > 50:
                        fout.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                        total_chars += len(text)
                except:
                    continue
        print(f"   ✅ {count:,} 条 | {total_chars / 1e6:.1f}MB")
        stats.append({'source': 'local_pretrain', 'name': '现有 pretrain_hq', 'count': count, 'chars': total_chars})

    # 2. 现有 sft 数据（转为 pretrain 格式）
    sft_path = os.path.join(dataset_dir, 'sft_t2t_mini.jsonl')
    if os.path.exists(sft_path):
        print(f"\n{'─' * 50}")
        print(f"📂 处理本地文件: sft_t2t_mini.jsonl → pretrain 格式")
        count = 0
        total_chars = 0
        with open(sft_path, 'r', encoding='utf-8') as fin, \
             open(output_file, 'a', encoding='utf-8') as fout:
            for line in fin:
                try:
                    data = json.loads(line.strip())
                    conversations = data.get('conversations', [])
                    text = process_sft_to_pretrain(conversations)
                    if text:
                        fout.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                        total_chars += len(text)
                except:
                    continue
        print(f"   ✅ {count:,} 条 | {total_chars / 1e6:.1f}MB")
        stats.append({'source': 'local_sft', 'name': '现有 sft_t2t_mini (转pretrain)', 'count': count, 'chars': total_chars})

    return stats


def shuffle_file(filepath):
    """对文件进行随机打乱（内存中处理）"""
    print(f"\n🔀 打乱数据顺序...")

    # 检查文件大小
    file_size = os.path.getsize(filepath) / (1024 ** 3)
    if file_size > 30:
        print(f"   文件过大 ({file_size:.1f}GB)，使用分块打乱...")
        shuffle_large_file(filepath)
        return

    st = time.time()
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"   ✅ {len(lines):,} 行已打乱 ({time.time() - st:.1f}s)")


def shuffle_large_file(filepath, chunk_size=1_000_000):
    """大文件分块打乱"""
    import tempfile
    st = time.time()

    # 读取所有行的偏移量
    offsets = []
    with open(filepath, 'rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(offset)

    random.shuffle(offsets)

    tmp_path = filepath + '.shuffled'
    with open(filepath, 'rb') as fin, open(tmp_path, 'wb') as fout:
        for offset in offsets:
            fin.seek(offset)
            fout.write(fin.readline())

    os.replace(tmp_path, filepath)
    print(f"   ✅ {len(offsets):,} 行已打乱 ({time.time() - st:.1f}s)")


def create_tokenizer_subset(pretrain_file, tokenizer_file, max_samples=200_000):
    """从预训练数据中采样子集用于分词器训练"""
    print(f"\n📝 创建分词器训练数据 (采样 {max_samples:,} 条)...")

    # 先计算总行数
    with open(pretrain_file, 'r', encoding='utf-8') as f:
        total = sum(1 for _ in f)

    # 均匀采样
    if total <= max_samples:
        # 数据不够，全部使用
        import shutil
        shutil.copy(pretrain_file, tokenizer_file)
        print(f"   ✅ 全部 {total:,} 条")
        return total

    step = total // max_samples
    count = 0
    with open(pretrain_file, 'r', encoding='utf-8') as fin, \
         open(tokenizer_file, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if i % step == 0 and count < max_samples:
                fout.write(line)
                count += 1

    print(f"   ✅ 从 {total:,} 条中采样 {count:,} 条 (步长 {step})")
    return count


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="1B 模型预训练数据下载与处理")
    parser.add_argument('--output_dir', default='.', type=str, help="输出目录")
    parser.add_argument('--dataset_dir', default='../dataset', type=str, help="现有数据集目录")
    parser.add_argument('--sources', default='all', type=str,
                        help="要下载的数据源，逗号分隔。可选: " + ','.join(DATA_SOURCES.keys()) + ",local,all")
    parser.add_argument('--test_mode', action='store_true', help="测试模式（每个源仅 100 条）")
    parser.add_argument('--tokenizer_samples', default=200_000, type=int, help="分词器训练采样数")
    parser.add_argument('--no_shuffle', action='store_true', help="不打乱数据")
    parser.add_argument('--no_tokenizer_subset', action='store_true', help="不创建分词器训练子集")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pretrain_file = os.path.join(args.output_dir, 'pretrain_1b.jsonl')
    tokenizer_file = os.path.join(args.output_dir, 'tokenizer_train.jsonl')
    stats_file = os.path.join(args.output_dir, 'data_stats.json')

    # 清空输出文件
    open(pretrain_file, 'w').close()

    if args.sources == 'all':
        source_keys = list(DATA_SOURCES.keys())
        include_local = True
    else:
        source_keys = [s.strip() for s in args.sources.split(',') if s.strip() in DATA_SOURCES]
        include_local = 'local' in args.sources or 'all' in args.sources

    print(f"{'=' * 60}")
    print(f"1B 模型预训练数据准备")
    print(f"{'=' * 60}")
    print(f"  输出文件: {pretrain_file}")
    print(f"  数据源: {source_keys}")
    print(f"  包含本地数据: {include_local}")
    print(f"  测试模式: {args.test_mode}")
    print(f"{'=' * 60}")

    all_stats = []
    total_start = time.time()

    # 1. 处理本地已有数据
    if include_local:
        local_stats = process_local_files(pretrain_file, args.dataset_dir)
        all_stats.extend(local_stats)

    # 2. 下载并处理 HuggingFace 数据
    for key in source_keys:
        if key not in DATA_SOURCES:
            print(f"  ⚠️ 未知数据源: {key}")
            continue
        config = DATA_SOURCES[key]
        stat = download_and_process_source(key, config, pretrain_file, args.test_mode)
        all_stats.append(stat)

    # 3. 打乱
    if not args.no_shuffle and os.path.exists(pretrain_file):
        shuffle_file(pretrain_file)

    # 4. 创建分词器训练子集
    if not args.no_tokenizer_subset and os.path.exists(pretrain_file):
        create_tokenizer_subset(pretrain_file, tokenizer_file, args.tokenizer_samples)

    # 5. 统计
    total_time = time.time() - total_start
    file_size = os.path.getsize(pretrain_file) / (1024 ** 3) if os.path.exists(pretrain_file) else 0

    total_count = sum(s.get('count', 0) for s in all_stats)
    total_chars = sum(s.get('chars', 0) for s in all_stats)
    estimated_tokens = total_chars / 1.6  # 中文 ~1.6 字符/token (6400 词表)

    summary = {
        'total_samples': total_count,
        'total_chars': total_chars,
        'estimated_tokens_6400': int(estimated_tokens),
        'estimated_tokens_32000': int(total_chars / 2.5),  # 32K 词表编码效率更高
        'file_size_gb': round(file_size, 2),
        'total_time_min': round(total_time / 60, 1),
        'sources': all_stats,
    }

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"数据准备完成!")
    print(f"{'=' * 60}")
    print(f"  总条数:      {total_count:,}")
    print(f"  总字符数:    {total_chars / 1e9:.2f}B")
    print(f"  文件大小:    {file_size:.2f} GB")
    print(f"  估计 tokens: {estimated_tokens / 1e9:.1f}B (6400词表) / {total_chars / 2.5 / 1e9:.1f}B (32K词表)")
    print(f"  耗时:        {total_time / 60:.1f} min")
    print(f"{'=' * 60}")
    print(f"\n📁 输出文件:")
    print(f"  预训练数据:    {pretrain_file}")
    if os.path.exists(tokenizer_file):
        print(f"  分词器训练数据: {tokenizer_file}")
    print(f"  数据统计:      {stats_file}")
    print(f"\n后续步骤:")
    print(f"  1. 训练分词器:  cd ../trainer && python train_tokenizer_1b.py --data_path ../dataset_1B/tokenizer_train.jsonl")
    print(f"  2. 创建软链接:  cd ../dataset && ln -sf ../dataset_1B/pretrain_1b.jsonl pretrain_hq.jsonl")
    print(f"  3. 开始训练:    cd ../trainer && torchrun ... train_pretrain.py --hidden_size 2048 --num_hidden_layers 22")


if __name__ == '__main__':
    main()
