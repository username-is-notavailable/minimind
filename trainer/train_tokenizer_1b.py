"""
train_tokenizer_1b.py — 为 1B 模型训练更大词表的分词器

为什么需要新的分词器：
  - 原始 6400 词表是为 26M 模型设计的（避免 embed 占比过高）
  - 1B 模型中 embed 占比仅 1.3%，词表大小不再是瓶颈
  - 更大词表 → 更高编码效率 → 同样 tokens 覆盖更多文本 → 训练更高效
  - 32K 词表在 1B 模型中 embed 占比 6.6%（完全合理，主流模型均在此范围）

用法：
    cd minimind/trainer

    # 默认：在预训练数据上训练 32000 词表
    python train_tokenizer_1b.py

    # 自定义词表大小和数据
    python train_tokenizer_1b.py --vocab_size 32000 --data_path ../dataset/pretrain_1b.jsonl

    # 使用更多训练数据行
    python train_tokenizer_1b.py --max_lines 500000

    # 训练完成后，1B 训练使用新分词器：
    # torchrun --nproc_per_node 4 train_pretrain.py \
    #     --hidden_size 2048 --num_hidden_layers 22 \
    #     --tokenizer_path ../model_1b_tokenizer/ \
    #     ...
"""

import os
import sys
import json
import time
import argparse
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer


def get_texts(data_path, max_lines=None):
    """从 jsonl 文件流式读取文本，支持 pretrain 和 sft 格式"""
    count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_lines and count >= max_lines:
                break
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            if 'text' in data:
                text = data['text']
            elif 'conversations' in data:
                text = ' '.join(turn.get('content', '') for turn in data['conversations'])
            else:
                continue

            if len(text) > 20:
                yield text
                count += 1

    print(f"  已读取 {count} 条文本")


def train_tokenizer(data_path, output_dir, vocab_size, max_lines):
    """训练 BPE 分词器"""
    print(f"{'=' * 60}")
    print(f"分词器训练")
    print(f"  数据: {data_path}")
    print(f"  词表大小: {vocab_size}")
    print(f"  最大行数: {max_lines if max_lines else '不限'}")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 60}")

    # 初始化 BPE 分词器
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 特殊 token（与原始 MiniMind 保持一致）
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,  # 至少出现 2 次才加入词表
    )

    # 训练
    print("\n开始训练...")
    st = time.time()
    texts = get_texts(data_path, max_lines)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    elapsed = time.time() - st
    print(f"训练完成! 耗时: {elapsed:.1f}s")

    # 验证特殊 token
    assert tokenizer.token_to_id("<|endoftext|>") == 0, "特殊 token ID 不匹配"
    assert tokenizer.token_to_id("<|im_start|>") == 1, "特殊 token ID 不匹配"
    assert tokenizer.token_to_id("<|im_end|>") == 2, "特殊 token ID 不匹配"

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    tokenizer.model.save(output_dir)

    # tokenizer_config.json（与原始 MiniMind 格式完全一致）
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {"content": "<|endoftext|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "1": {"content": "<|im_start|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "2": {"content": "<|im_end|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        # chat_template 与原始 MiniMind 完全一致
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    }

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 分词器已保存到: {output_dir}")
    print(f"   词表大小: {tokenizer.get_vocab_size()}")
    return tokenizer


def eval_tokenizer(output_dir):
    """验证分词器的编码/解码一致性和编码效率"""
    from transformers import AutoTokenizer

    print(f"\n{'=' * 60}")
    print(f"分词器评估")
    print(f"{'=' * 60}")

    new_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    old_tokenizer = AutoTokenizer.from_pretrained('../model/')

    print(f"\n  新词表大小: {len(new_tokenizer)} | 旧词表大小: {len(old_tokenizer)}")

    # 1. 编解码一致性测试
    test_texts = [
        "你好，世界！Hello, world!",
        "大语言模型（LLM）是一种基于Transformer架构的深度学习模型。",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "北京烤鸭、四川火锅、广东早茶，都是中国的特色美食。",
        "E=mc²，这是爱因斯坦的质能方程。",
    ]

    print(f"\n  编码效率对比:")
    print(f"  {'文本':30s} {'旧(6400)':>10s} {'新':>10s} {'压缩比':>8s}")
    print(f"  {'-' * 62}")

    total_old, total_new = 0, 0
    for text in test_texts:
        old_ids = old_tokenizer.encode(text)
        new_ids = new_tokenizer.encode(text)
        ratio = len(old_ids) / len(new_ids) if len(new_ids) > 0 else 0
        total_old += len(old_ids)
        total_new += len(new_ids)

        display = text[:28] + '..' if len(text) > 30 else text
        print(f"  {display:30s} {len(old_ids):>8d}tk {len(new_ids):>8d}tk {ratio:>6.2f}x")

    overall_ratio = total_old / total_new if total_new > 0 else 0
    print(f"  {'总计':30s} {total_old:>8d}tk {total_new:>8d}tk {overall_ratio:>6.2f}x")

    # 2. 编解码一致性
    print(f"\n  编解码一致性测试:")
    all_pass = True
    for text in test_texts:
        ids = new_tokenizer.encode(text)
        decoded = new_tokenizer.decode(ids, skip_special_tokens=True)
        match = decoded.strip() == text.strip()
        if not match:
            all_pass = False
            print(f"  ❌ 不一致: '{text[:40]}...'")
    if all_pass:
        print(f"  ✅ 全部通过")

    # 3. Chat template 测试
    print(f"\n  Chat Template 测试:")
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
    ]
    try:
        result = new_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"  ✅ Chat template 正常")
        print(f"  输出: {result[:80]}...")
    except Exception as e:
        print(f"  ❌ Chat template 失败: {e}")

    # 4. 特殊 token 检查
    print(f"\n  特殊 Token:")
    for name, expected_id in [("pad/unk (<|endoftext|>)", 0), ("bos (<|im_start|>)", 1), ("eos (<|im_end|>)", 2)]:
        actual_id = new_tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<|im_start|>", "<|im_end|>"][expected_id])
        status = "✅" if actual_id == expected_id else "❌"
        print(f"  {status} {name}: ID={actual_id} (期望={expected_id})")


def main():
    parser = argparse.ArgumentParser(description="为 1B 模型训练更大词表的分词器")
    parser.add_argument('--data_path', default='../dataset/pretrain_hq.jsonl', type=str,
                        help="训练数据路径（应使用 1B 预训练数据）")
    parser.add_argument('--output_dir', default='../model_1b_tokenizer', type=str,
                        help="分词器输出目录")
    parser.add_argument('--vocab_size', default=32000, type=int,
                        help="词表大小（推荐 32000，主流 1B 模型标准）")
    parser.add_argument('--max_lines', default=100000, type=int,
                        help="最大训练行数（0=不限，建议 10 万-50 万条即可覆盖足够的词汇分布）")
    parser.add_argument('--skip_eval', action='store_true',
                        help="跳过评估步骤")
    args = parser.parse_args()

    train_tokenizer(args.data_path, args.output_dir, args.vocab_size,
                    args.max_lines if args.max_lines > 0 else None)

    if not args.skip_eval:
        eval_tokenizer(args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"后续步骤:")
    print(f"  1. 修改 MiniMindConfig 的 vocab_size 为 {args.vocab_size}")
    print(f"  2. 训练时指定分词器路径:")
    print(f"     --tokenizer_path {args.output_dir}")
    print(f"  3. 注意：使用新分词器训练的模型与旧分词器模型不兼容")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
