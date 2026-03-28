#!/usr/bin/env python3
"""
expand_pretrain_data.py — 扩充预训练数据至 ~20B tokens (6400词表) / ~13.9B tokens (32K词表)

适用于 0.5B 模型（Chinchilla 最优 ~10B tokens，当前 13.9B tokens 已满足）
及 1B 模型（Chinchilla 最优 ~20B tokens，当前 21.7B tokens 已满足）

扩充策略（追加到现有 pretrain_1b.jsonl）:
  ┌──────────────────────┬────────────┬───────────┬──────────────────────────────────┐
  │ 来源                  │ 追加条数    │ 预估字符   │ 说明                              │
  ├──────────────────────┼────────────┼───────────┼──────────────────────────────────┤
  │ 本地 sft_t2t.jsonl    │ ~4,200,000 │ ~4.0B     │ 完整SFT数据转预训练格式（免下载）     │
  │ 中文网页 (SkyPile)    │ 7,000,000  │ ~7.0B     │ 从 offset=3,502,000 继续采样       │
  │ 英文网页 (C4)         │ 2,000,000  │ ~3.8B     │ 从 offset=1,002,000 继续采样       │
  │ 英文学术 (open-web-math)│ 800,000  │ ~2.9B     │ 从 offset=555,000 继续采样         │
  │ 中文补充 (SkyPile)    │ 1,000,000  │ ~1.0B     │ offset=12,000,000 不同区间          │
  └──────────────────────┴────────────┴───────────┴──────────────────────────────────┘
  预计扩充: ~18.7B 字符 → 合计 ~32.5B 字符 → ~20.3B tokens

用法:
    cd minimind/dataset_1B
    python expand_pretrain_data.py                # 完整扩充
    python expand_pretrain_data.py --test_mode    # 测试模式（每源100条）
"""

import os
import sys
import json
import time
import random
import argparse


# ============================================================
# 文本清洗（复用 prepare_pretrain_data.py 的逻辑）
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
    return sum(1 for c in text if is_chinese_char(c)) / len(text)


def clean_text(text, lang='zh', min_length=50, max_length=8192):
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if len(text) < min_length:
        return None
    if len(text) > max_length:
        text = text[:max_length]
    if lang == 'zh' and chinese_char_ratio(text) < 0.3:
        return None
    elif lang == 'en' and chinese_char_ratio(text) > 0.1:
        return None
    lines = text.split('\n')
    if len(lines) > 3 and len(set(lines)) / len(lines) < 0.5:
        return None
    if sum(1 for c in text if ord(c) > 0xFFFF or c in '\x00\x01\x02\x03') > len(text) * 0.05:
        return None
    return text


def process_sft_to_pretrain(conversations):
    parts = []
    for turn in conversations:
        content = turn.get('content', '')
        if content:
            parts.append(content)
    text = '\n'.join(parts)
    return text if len(text) > 50 else None


# ============================================================
# 扩充数据源配置
# ============================================================

EXPAND_SOURCES = {
    'local_sft_full': {
        'type': 'local_sft',
        'name': 'SFT完整数据(转pretrain)',
        'path': '../dataset/sft_t2t.jsonl',
    },
    'chinese_web_expand': {
        'type': 'hf',
        'name': '中文网页(扩充)',
        'hf_repo': 'Skywork/SkyPile-150B',
        'hf_config': None,
        'hf_split': 'train',
        'text_field': 'text',
        'max_samples': 7_000_000,
        'offset': 3_502_000,
        'lang': 'zh',
    },
    'english_web_expand': {
        'type': 'hf',
        'name': '英文网页(扩充)',
        'hf_repo': 'allenai/c4',
        'hf_config': 'en',
        'hf_split': 'train',
        'text_field': 'text',
        'max_samples': 2_000_000,
        'offset': 1_002_000,
        'lang': 'en',
    },
    'english_academic_expand': {
        'type': 'hf',
        'name': '英文学术(扩充)',
        'hf_repo': 'open-web-math/open-web-math',
        'hf_config': None,
        'hf_split': 'train',
        'text_field': 'text',
        'max_samples': 800_000,
        'offset': 555_000,
        'lang': 'en',
    },
    'chinese_diverse': {
        'type': 'hf',
        'name': '中文多样化补充',
        'hf_repo': 'Skywork/SkyPile-150B',
        'hf_config': None,
        'hf_split': 'train',
        'text_field': 'text',
        'max_samples': 1_000_000,
        'offset': 12_000_000,
        'lang': 'zh',
    },
}


# ============================================================
# 处理函数
# ============================================================

def process_local_sft(sft_path, output_file):
    """处理本地完整 SFT 数据，转为预训练格式并追加"""
    if not os.path.exists(sft_path):
        print(f"   ⚠️ 文件不存在: {sft_path}")
        return {'source': 'local_sft_full', 'name': 'SFT完整数据', 'count': 0, 'chars': 0}

    print(f"\n{'─' * 50}")
    print(f"📂 处理本地文件: {os.path.basename(sft_path)} → pretrain 格式")

    count = 0
    skipped = 0
    total_chars = 0

    # 已有的 sft_t2t_mini 条数（避免完全重复的粗略检查）
    seen_first_lines = set()
    mini_path = os.path.join(os.path.dirname(sft_path), 'sft_t2t_mini.jsonl')
    if os.path.exists(mini_path):
        print(f"   跳过已包含的 sft_t2t_mini 数据...")
        with open(mini_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 50000:  # 只采样前5万条做去重指纹
                    break
                try:
                    data = json.loads(line.strip())
                    convs = data.get('conversations', [])
                    if convs and convs[0].get('content'):
                        key = convs[0]['content'][:100]
                        seen_first_lines.add(key)
                except:
                    continue
        print(f"   已建立 {len(seen_first_lines):,} 条去重指纹")

    with open(sft_path, 'r', encoding='utf-8') as fin, \
         open(output_file, 'a', encoding='utf-8') as fout:
        for line in fin:
            try:
                data = json.loads(line.strip())
                conversations = data.get('conversations', [])

                # 粗略去重
                if conversations and conversations[0].get('content'):
                    key = conversations[0]['content'][:100]
                    if key in seen_first_lines:
                        skipped += 1
                        continue

                text = process_sft_to_pretrain(conversations)
                if text:
                    fout.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                    total_chars += len(text)
            except:
                continue

            if count > 0 and count % 500_000 == 0:
                print(f"   [{count:>10,}] 条 | 跳过 {skipped:,} | {total_chars / 1e9:.2f}GB")

    print(f"   ✅ {count:,} 条 | 跳过(去重) {skipped:,} | {total_chars / 1e6:.1f}MB")
    return {'source': 'local_sft_full', 'name': 'SFT完整数据(转pretrain)',
            'count': count, 'skipped': skipped, 'chars': total_chars}


def download_hf_source(source_key, config, output_file, test_mode=False):
    """下载 HuggingFace 数据源并追加到输出文件"""
    from datasets import load_dataset

    name = config['name']
    max_samples = 100 if test_mode else config['max_samples']
    offset = 0 if test_mode else config.get('offset', 0)
    lang = config['lang']

    print(f"\n{'─' * 50}")
    print(f"📥 下载: {name}")
    print(f"   源: {config['hf_repo']}")
    print(f"   目标条数: {max_samples:,}")
    if offset:
        print(f"   偏移: 跳过前 {offset:,} 条")

    st = time.time()
    count = 0
    skipped = 0
    total_chars = 0

    try:
        ds = load_dataset(
            config['hf_repo'],
            config.get('hf_config'),
            split=config['hf_split'],
            streaming=True,
        )

        with open(output_file, 'a', encoding='utf-8') as f:
            for i, item in enumerate(ds):
                if i < offset:
                    continue
                if count >= max_samples:
                    break

                raw_text = item.get(config['text_field'], '')
                text = clean_text(raw_text, lang=lang)
                if text is None:
                    skipped += 1
                    continue

                f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                count += 1
                total_chars += len(text)

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
        'source': source_key, 'name': name,
        'count': count, 'skipped': skipped,
        'chars': total_chars, 'time_sec': round(elapsed, 1),
    }


def shuffle_large_file(filepath):
    """大文件随机打乱"""
    print(f"\n🔀 打乱数据顺序...")
    st = time.time()

    offsets = []
    with open(filepath, 'rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(offset)

    print(f"   共 {len(offsets):,} 行，开始打乱...")
    random.shuffle(offsets)

    tmp_path = filepath + '.shuffled'
    with open(filepath, 'rb') as fin, open(tmp_path, 'wb') as fout:
        for i, offset in enumerate(offsets):
            fin.seek(offset)
            fout.write(fin.readline())
            if (i + 1) % 5_000_000 == 0:
                print(f"   已写入 {i + 1:,} / {len(offsets):,} 行")

    os.replace(tmp_path, filepath)
    print(f"   ✅ {len(offsets):,} 行已打乱 ({time.time() - st:.1f}s)")
    return len(offsets)


def create_tokenizer_subset(pretrain_file, tokenizer_file, max_samples=200_000):
    """从预训练数据中均匀采样子集用于分词器训练"""
    print(f"\n📝 创建分词器训练数据 (采样 {max_samples:,} 条)...")

    with open(pretrain_file, 'r', encoding='utf-8') as f:
        total = sum(1 for _ in f)

    if total <= max_samples:
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
    parser = argparse.ArgumentParser(description="扩充 1B 模型预训练数据至 ~20B tokens")
    parser.add_argument('--output_dir', default='.', type=str)
    parser.add_argument('--test_mode', action='store_true', help="测试模式（每源100条）")
    parser.add_argument('--sources', default='all', type=str,
                        help="指定扩充源，逗号分隔")
    parser.add_argument('--no_shuffle', action='store_true', help="不打乱数据")
    parser.add_argument('--tokenizer_samples', default=200_000, type=int)
    args = parser.parse_args()

    pretrain_file = os.path.join(args.output_dir, 'pretrain_1b.jsonl')
    tokenizer_file = os.path.join(args.output_dir, 'tokenizer_train.jsonl')
    stats_file = os.path.join(args.output_dir, 'data_stats.json')

    # 加载现有统计
    old_stats = {}
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            old_stats = json.load(f)

    old_samples = old_stats.get('total_samples', 0)
    old_chars = old_stats.get('total_chars', 0)
    old_sources = old_stats.get('sources', [])

    print(f"{'=' * 60}")
    print(f"1B 模型预训练数据扩充")
    print(f"{'=' * 60}")
    print(f"  现有数据:    {old_samples:,} 条 | {old_chars / 1e9:.2f}B 字符 | ~{old_chars / 1.6 / 1e9:.1f}B tokens")
    print(f"  目标:        ~20B tokens (6400词表) | ~32B 字符")
    print(f"  测试模式:    {args.test_mode}")
    print(f"{'=' * 60}")

    if args.sources == 'all':
        source_keys = list(EXPAND_SOURCES.keys())
    else:
        source_keys = [s.strip() for s in args.sources.split(',') if s.strip() in EXPAND_SOURCES]

    expand_stats = []
    total_start = time.time()

    for key in source_keys:
        config = EXPAND_SOURCES[key]
        if config.get('type') == 'local_sft':
            stat = process_local_sft(config['path'], pretrain_file)
        else:
            stat = download_hf_source(key, config, pretrain_file, args.test_mode)
        expand_stats.append(stat)

    # 打乱
    if not args.no_shuffle and os.path.exists(pretrain_file):
        total_lines = shuffle_large_file(pretrain_file)
    else:
        with open(pretrain_file, 'r') as f:
            total_lines = sum(1 for _ in f)

    # 重新创建分词器训练子集
    if os.path.exists(pretrain_file):
        create_tokenizer_subset(pretrain_file, tokenizer_file, args.tokenizer_samples)

    # 统计
    total_time = time.time() - total_start
    file_size = os.path.getsize(pretrain_file) / (1024 ** 3) if os.path.exists(pretrain_file) else 0

    new_count = sum(s.get('count', 0) for s in expand_stats)
    new_chars = sum(s.get('chars', 0) for s in expand_stats)
    total_samples = old_samples + new_count
    total_chars = old_chars + new_chars
    est_tokens_6400 = int(total_chars / 1.6)
    est_tokens_32000 = int(total_chars / 2.5)

    summary = {
        'total_samples': total_samples,
        'total_chars': total_chars,
        'estimated_tokens_6400': est_tokens_6400,
        'estimated_tokens_32000': est_tokens_32000,
        'file_size_gb': round(file_size, 2),
        'total_time_min': round(total_time / 60, 1),
        'sources': old_sources + expand_stats,
    }

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"数据扩充完成!")
    print(f"{'=' * 60}")
    print(f"  本次新增:    {new_count:,} 条 | {new_chars / 1e9:.2f}B 字符")
    print(f"  累计总条数:  {total_samples:,}")
    print(f"  累计字符数:  {total_chars / 1e9:.2f}B")
    print(f"  文件大小:    {file_size:.2f} GB")
    print(f"  估计 tokens: {est_tokens_6400 / 1e9:.1f}B (6400词表) / {est_tokens_32000 / 1e9:.1f}B (32K词表)")
    print(f"  耗时:        {total_time / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
