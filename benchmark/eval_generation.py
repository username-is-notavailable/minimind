"""
eval_generation.py — 生成质量评测

评测内容：
  - 多维度自动评分（关键词命中、流畅度、长度、格式）
  - 推理速度测量 (tokens/s)
  - 完整生成文本记录

适用模型权重：full_sft_*.pth, dpo_*.pth 等 SFT 后模型

用法：
    cd benchmark/
    python eval_generation.py --weight full_sft
    python eval_generation.py --weight full_sft --use_mla 1
    python eval_generation.py --weight dpo
"""

import os
import sys
import json
import time
import argparse
import torch
import warnings
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')

# 评测 prompt 集合，按能力维度分类
EVAL_PROMPTS = [
    # === 事实问答 ===
    {'prompt': '中国的首都是哪里？', 'keywords': ['北京'], 'category': '事实问答', 'difficulty': 'easy'},
    {'prompt': '世界上最高的山峰是什么？', 'keywords': ['珠穆朗玛', '8848', '喜马拉雅'], 'category': '事实问答', 'difficulty': 'easy'},
    {'prompt': '万有引力是谁提出的？', 'keywords': ['牛顿', 'Newton'], 'category': '事实问答', 'difficulty': 'easy'},
    {'prompt': '地球绕太阳一周需要多长时间？', 'keywords': ['一年', '365', '天'], 'category': '事实问答', 'difficulty': 'easy'},

    # === 科学解释 ===
    {'prompt': '为什么天空是蓝色的？', 'keywords': ['光', '散射', '大气', '波长', '瑞利'], 'category': '科学解释', 'difficulty': 'medium'},
    {'prompt': '解释一下光合作用的基本过程', 'keywords': ['植物', '阳光', '二氧化碳', '水', '氧气', '能量'], 'category': '科学解释', 'difficulty': 'medium'},
    {'prompt': '海水为什么是咸的？', 'keywords': ['盐', '矿物质', '氯化钠', '溶解'], 'category': '科学解释', 'difficulty': 'medium'},

    # === 逻辑推理 ===
    {'prompt': '如果所有的猫都是动物，小花是猫，那么小花是什么？', 'keywords': ['动物'], 'category': '逻辑推理', 'difficulty': 'easy'},
    {'prompt': '比较一下猫和狗作为宠物的优缺点', 'keywords': ['猫', '狗', '优', '缺'], 'category': '逻辑推理', 'difficulty': 'medium'},

    # === 代码生成 ===
    {'prompt': '用Python写一个计算斐波那契数列的函数', 'keywords': ['def', 'return', 'fib', 'n'], 'category': '代码生成', 'difficulty': 'medium'},
    {'prompt': '用Python写一个冒泡排序算法', 'keywords': ['def', 'for', 'swap', 'sort', 'return'], 'category': '代码生成', 'difficulty': 'medium'},

    # === 创意写作 ===
    {'prompt': '写一首关于春天的五言绝句', 'keywords': ['春', '花', '风', '绿'], 'category': '创意写作', 'difficulty': 'hard'},
    {'prompt': '推荐一些中国的美食', 'keywords': ['北京', '四川', '广东', '烤鸭', '火锅', '菜'], 'category': '创意写作', 'difficulty': 'easy'},

    # === 自我认知 ===
    {'prompt': '你是谁？', 'keywords': ['AI', '助手', '模型', '语言', '人工智能'], 'category': '自我认知', 'difficulty': 'easy'},
    {'prompt': '解释什么是机器学习', 'keywords': ['数据', '模型', '训练', '算法', '学习'], 'category': '自我认知', 'difficulty': 'medium'},
]


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=args.vocab_size,
        use_moe=bool(args.use_moe),
        use_mla=bool(args.use_mla),
        mla_kv_dim=args.mla_kv_dim,
        mla_q_dim=args.mla_q_dim,
        mla_rope_dim=args.mla_rope_dim,
    )
    model = MiniMindForCausalLM(config)
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    print(f"加载权重: {ckp}")
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    get_model_params(model, config)
    return model.eval().to(args.device), tokenizer


def score_response(response, keywords):
    """多维度自动评分，满分 100"""
    score = 0
    length = len(response)

    # 1. 长度适当性（20分）
    if length < 5:
        score += 0
    elif length < 20:
        score += 5
    elif length < 50:
        score += 12
    elif length <= 500:
        score += 20
    elif length <= 1000:
        score += 16
    else:
        score += 10

    # 2. 关键词命中率（40分）
    if keywords:
        hits = sum(1 for kw in keywords if kw.lower() in response.lower())
        score += int(hits / len(keywords) * 40)

    # 3. 流畅度 — 重复检测（20分）
    fluency = 20
    for seg_len in [8, 15, 25]:
        if length > seg_len * 3:
            segments = [response[i:i + seg_len] for i in range(0, length - seg_len, seg_len)]
            counts = Counter(segments)
            max_rep = max(counts.values()) if counts else 1
            if max_rep > 3:
                fluency = max(0, fluency - min(20, (max_rep - 3) * 4))
    score += fluency

    # 4. 格式与结构（20分）
    fmt = 0
    if any(p in response for p in ['。', '！', '？', '.', '!', '?']):
        fmt += 8
    if '\n' in response or any(f'{i}.' in response for i in range(1, 10)):
        fmt += 6
    if length > 30 and response[-10:] != response[:10]:
        fmt += 6
    score += fmt

    return min(100, score)


def main():
    parser = argparse.ArgumentParser(description="生成质量评测")
    parser.add_argument('--weight', default='full_sft', type=str)
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_mla', default=0, type=int, choices=[0, 1])
    parser.add_argument('--mla_kv_dim', type=int, default=128)
    parser.add_argument('--mla_q_dim', type=int, default=256)
    parser.add_argument('--mla_rope_dim', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=6400)
    parser.add_argument('--tokenizer_path', type=str, default='../model/')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    args = parser.parse_args()

    setup_seed(2026)
    model, tokenizer = load_model(args)

    arch = "MLA" if args.use_mla else "GQA"
    print(f"\n{'=' * 60}")
    print(f"生成质量评测 | 权重: {args.weight} | 架构: {arch}")
    print(f"{'=' * 60}")

    results = []
    category_scores = {}
    total_tokens = 0
    total_time = 0

    for item in EVAL_PROMPTS:
        setup_seed(2026)
        prompt = item['prompt']

        conversation = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt', truncation=True).to(args.device)

        torch.cuda.synchronize() if args.device == 'cuda' else None
        st = time.time()
        with torch.no_grad():
            gen_ids = model.generate(
                inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                max_new_tokens=args.max_new_tokens, do_sample=True,
                top_p=args.top_p, temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize() if args.device == 'cuda' else None
        gen_time = time.time() - st

        gen_token_count = len(gen_ids[0]) - len(inputs['input_ids'][0])
        response = tokenizer.decode(gen_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

        score = score_response(response, item['keywords'])
        total_tokens += gen_token_count
        total_time += gen_time

        cat = item['category']
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(score)

        results.append({
            'prompt': prompt,
            'category': cat,
            'difficulty': item['difficulty'],
            'response': response,
            'score': score,
            'gen_tokens': gen_token_count,
            'tokens_per_sec': round(gen_token_count / gen_time, 2) if gen_time > 0 else 0,
        })

        print(f"  [{cat:6s}] {prompt[:25]:25s} → 评分: {score:3d}/100 | "
              f"{gen_token_count:4d} tokens | {gen_token_count / gen_time:.1f} tok/s")

    # 汇总
    avg_score = sum(r['score'] for r in results) / len(results)
    avg_speed = total_tokens / total_time if total_time > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"分类评分:")
    cat_summary = {}
    for cat, scores in category_scores.items():
        cat_avg = sum(scores) / len(scores)
        cat_summary[cat] = round(cat_avg, 1)
        print(f"  {cat:10s}: {cat_avg:.1f}/100 ({len(scores)}题)")

    print(f"\n  总平均分: {avg_score:.1f}/100")
    print(f"  平均速度: {avg_speed:.1f} tokens/s")
    print(f"{'=' * 60}")

    summary = {
        'weight': args.weight,
        'architecture': arch,
        'hidden_size': args.hidden_size,
        'avg_score': round(avg_score, 2),
        'avg_speed': round(avg_speed, 2),
        'category_scores': cat_summary,
        'details': results,
    }

    os.makedirs('results', exist_ok=True)
    out_path = f'results/generation_{args.weight}_{arch}_{args.hidden_size}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n📁 结果已保存: {out_path}")


if __name__ == '__main__':
    main()
