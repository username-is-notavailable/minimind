"""
MiniMind 模型量化评估脚本

评估维度：
1. Perplexity (PPL) — 在预训练/SFT数据的held-out子集上计算
2. 选择题准确率 — 模拟C-Eval风格的多选题评测
3. 生成质量评分 — 固定prompt生成+关键词/长度/格式自动评分
4. 推理效率 — tokens/s、显存峰值、首token延迟

用法：
    # 评估标准 GQA 模型
    python eval_benchmark.py --weight full_sft

    # 评估 MLA 模型
    python eval_benchmark.py --weight full_sft --use_mla 1

    # 仅计算 PPL
    python eval_benchmark.py --weight full_sft --tasks ppl

    # 运行所有评测
    python eval_benchmark.py --weight full_sft --tasks all

    # 指定评测数据
    python eval_benchmark.py --weight full_sft --eval_data ../dataset/sft_t2t_mini.jsonl --eval_samples 500
"""

import os
import sys
import time
import json
import argparse
import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


# ============================================================
# 1. Perplexity 评估
# ============================================================

class PerplexityDataset(Dataset):
    """从 jsonl 文件加载评估数据（支持 pretrain 和 sft 格式）"""

    def __init__(self, data_path, tokenizer, max_length=512, max_samples=1000, offset=0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 从 offset 开始取样，模拟 held-out 集
        selected = lines[offset:offset + max_samples] if offset + max_samples <= len(lines) else lines[-max_samples:]

        for line in selected:
            data = json.loads(line.strip())
            if 'text' in data:
                # pretrain 格式
                text = data['text']
            elif 'conversations' in data:
                # sft 格式：拼接所有轮次
                text = self.tokenizer.apply_chat_template(
                    data['conversations'], tokenize=False, add_generation_prompt=False
                )
            else:
                continue

            tokens = self.tokenizer(text, return_tensors='pt', truncation=True,
                                    max_length=max_length, padding=False)
            if tokens['input_ids'].shape[1] > 10:  # 过滤太短的
                self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def compute_perplexity(model, tokenizer, data_path, max_length=512, max_samples=1000,
                       batch_size=1, device='cuda', offset=0):
    """计算模型在给定数据上的 Perplexity"""
    dataset = PerplexityDataset(data_path, tokenizer, max_length, max_samples, offset)
    print(f"  PPL 评估样本数: {len(dataset)}")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            input_ids = sample['input_ids'].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # token 数 = seq_len - 1（shift）
            num_tokens = input_ids.shape[1] - 1
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if (i + 1) % 100 == 0:
                current_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
                print(f"  [{i + 1}/{len(dataset)}] 当前 PPL: {current_ppl:.2f}")

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return {
        'perplexity': round(ppl, 4),
        'avg_loss': round(avg_loss, 6),
        'total_tokens': total_tokens,
        'num_samples': len(dataset),
    }


# ============================================================
# 2. 选择题准确率评估（模拟 C-Eval 风格）
# ============================================================

MULTIPLE_CHOICE_QUESTIONS = [
    {
        "question": "中国的首都是哪个城市？",
        "choices": ["A. 上海", "B. 北京", "C. 广州", "D. 深圳"],
        "answer": "B"
    },
    {
        "question": "地球绕太阳公转一周大约需要多长时间？",
        "choices": ["A. 24小时", "B. 30天", "C. 365天", "D. 100天"],
        "answer": "C"
    },
    {
        "question": "水的化学式是什么？",
        "choices": ["A. CO2", "B. NaCl", "C. O2", "D. H2O"],
        "answer": "D"
    },
    {
        "question": "光合作用主要发生在植物的哪个部位？",
        "choices": ["A. 根", "B. 茎", "C. 叶", "D. 花"],
        "answer": "C"
    },
    {
        "question": "以下哪个是哺乳动物？",
        "choices": ["A. 蛇", "B. 鲨鱼", "C. 鲸鱼", "D. 蜥蜴"],
        "answer": "C"
    },
    {
        "question": "万有引力定律是谁提出的？",
        "choices": ["A. 爱因斯坦", "B. 牛顿", "C. 伽利略", "D. 达尔文"],
        "answer": "B"
    },
    {
        "question": "以下哪个朝代在中国历史上最早？",
        "choices": ["A. 唐朝", "B. 宋朝", "C. 汉朝", "D. 明朝"],
        "answer": "C"
    },
    {
        "question": "二进制数 1010 转换为十进制是多少？",
        "choices": ["A. 8", "B. 10", "C. 12", "D. 14"],
        "answer": "B"
    },
    {
        "question": "以下哪种气体是温室气体？",
        "choices": ["A. 氧气", "B. 氮气", "C. 二氧化碳", "D. 氢气"],
        "answer": "C"
    },
    {
        "question": "TCP/IP协议中，HTTP协议工作在哪一层？",
        "choices": ["A. 物理层", "B. 网络层", "C. 传输层", "D. 应用层"],
        "answer": "D"
    },
    {
        "question": "以下哪位是《红楼梦》的作者？",
        "choices": ["A. 曹雪芹", "B. 施耐庵", "C. 罗贯中", "D. 吴承恩"],
        "answer": "A"
    },
    {
        "question": "人体最大的器官是什么？",
        "choices": ["A. 心脏", "B. 肝脏", "C. 皮肤", "D. 大脑"],
        "answer": "C"
    },
    {
        "question": "1+1等于几？",
        "choices": ["A. 1", "B. 2", "C. 3", "D. 4"],
        "answer": "B"
    },
    {
        "question": "以下哪种编程语言主要用于网页前端开发？",
        "choices": ["A. Python", "B. Java", "C. JavaScript", "D. C++"],
        "answer": "C"
    },
    {
        "question": "声音在以下哪种介质中传播最快？",
        "choices": ["A. 空气", "B. 水", "C. 真空", "D. 钢铁"],
        "answer": "D"
    },
    {
        "question": "珠穆朗玛峰位于哪两个国家的边境？",
        "choices": ["A. 中国和印度", "B. 中国和尼泊尔", "C. 印度和尼泊尔", "D. 中国和巴基斯坦"],
        "answer": "B"
    },
    {
        "question": "以下哪个不是Python的内置数据类型？",
        "choices": ["A. list", "B. dict", "C. array", "D. tuple"],
        "answer": "C"
    },
    {
        "question": "太阳系中最大的行星是哪个？",
        "choices": ["A. 土星", "B. 木星", "C. 天王星", "D. 海王星"],
        "answer": "B"
    },
    {
        "question": "DNA的全称是什么？",
        "choices": ["A. 核糖核酸", "B. 脱氧核糖核酸", "C. 氨基酸", "D. 腺嘌呤核苷酸"],
        "answer": "B"
    },
    {
        "question": "以下哪个城市不是直辖市？",
        "choices": ["A. 天津", "B. 重庆", "C. 成都", "D. 上海"],
        "answer": "C"
    },
]


def eval_multiple_choice(model, tokenizer, device='cuda'):
    """通过概率法评测选择题准确率（取A/B/C/D对应token的logits）"""
    model.eval()
    correct = 0
    total = len(MULTIPLE_CHOICE_QUESTIONS)
    results = []

    # 获取 A/B/C/D 的 token ids
    choice_tokens = {}
    for c in ['A', 'B', 'C', 'D']:
        ids = tokenizer.encode(c, add_special_tokens=False)
        choice_tokens[c] = ids[-1]  # 取最后一个 token

    with torch.no_grad():
        for q in MULTIPLE_CHOICE_QUESTIONS:
            # 构造 prompt
            prompt = f"问题：{q['question']}\n" + "\n".join(q['choices']) + "\n答案："
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = model(input_ids=inputs['input_ids'])
            # 取最后一个 token 的 logits
            last_logits = outputs.logits[0, -1, :]

            # 比较 A/B/C/D 的概率
            probs = {}
            for c in ['A', 'B', 'C', 'D']:
                probs[c] = last_logits[choice_tokens[c]].item()

            predicted = max(probs, key=probs.get)
            is_correct = predicted == q['answer']
            if is_correct:
                correct += 1

            results.append({
                'question': q['question'],
                'answer': q['answer'],
                'predicted': predicted,
                'correct': is_correct,
                'probs': {k: round(v, 4) for k, v in probs.items()}
            })

    accuracy = correct / total
    return {
        'accuracy': round(accuracy, 4),
        'correct': correct,
        'total': total,
        'details': results,
    }


# ============================================================
# 3. 生成质量评估（自动评分）
# ============================================================

GENERATION_PROMPTS = [
    {
        'prompt': '你有什么特长？',
        'keywords': ['帮助', '回答', '问题', '信息', '语言', 'AI', '人工智能'],
        'category': '自我认知'
    },
    {
        'prompt': '为什么天空是蓝色的',
        'keywords': ['光', '散射', '大气', '波长', '太阳', '分子'],
        'category': '科学知识'
    },
    {
        'prompt': '请用Python写一个计算斐波那契数列的函数',
        'keywords': ['def', 'return', 'fibonacci', 'fib', 'n'],
        'category': '代码生成'
    },
    {
        'prompt': '解释一下"光合作用"的基本过程',
        'keywords': ['植物', '阳光', '二氧化碳', '水', '氧气', '叶绿素', '能量'],
        'category': '科学知识'
    },
    {
        'prompt': '比较一下猫和狗作为宠物的优缺点',
        'keywords': ['猫', '狗', '优点', '缺点', '宠物'],
        'category': '分析比较'
    },
    {
        'prompt': '解释什么是机器学习',
        'keywords': ['数据', '模型', '训练', '算法', '预测', '学习', '特征'],
        'category': '技术概念'
    },
    {
        'prompt': '推荐一些中国的美食',
        'keywords': ['北京', '四川', '广东', '烤鸭', '火锅', '小笼包', '美食', '菜'],
        'category': '常识推荐'
    },
    {
        'prompt': '世界上最高的山峰是什么？',
        'keywords': ['珠穆朗玛', '8848', '喜马拉雅', '最高'],
        'category': '事实问答'
    },
]


def score_generation(response, keywords, category):
    """对单条生成结果进行自动评分（0-100分）"""
    score = 0

    # 1. 长度评分（20分）— 回答不能太短或太长
    length = len(response)
    if length < 10:
        score += 0
    elif length < 30:
        score += 5
    elif length < 100:
        score += 15
    elif length <= 500:
        score += 20
    else:
        score += 15  # 过长轻微扣分

    # 2. 关键词命中率（40分）
    hits = sum(1 for kw in keywords if kw.lower() in response.lower())
    keyword_ratio = hits / len(keywords) if keywords else 0
    score += int(keyword_ratio * 40)

    # 3. 流畅性评分（20分）— 基于重复度检测
    if length > 0:
        # 检测是否有严重重复（连续重复片段）
        repeat_penalty = 0
        for seg_len in [10, 20, 30]:
            if length > seg_len * 2:
                segments = [response[i:i + seg_len] for i in range(0, length - seg_len, seg_len)]
                from collections import Counter
                seg_counts = Counter(segments)
                max_repeat = max(seg_counts.values()) if seg_counts else 1
                if max_repeat > 2:
                    repeat_penalty = max(repeat_penalty, min(20, (max_repeat - 2) * 5))
        score += (20 - repeat_penalty)
    else:
        score += 0

    # 4. 格式评分（20分）— 有分段、有标点
    if '。' in response or '！' in response or '？' in response or '.' in response:
        score += 10  # 有句号等结束标点
    if '\n' in response or '1.' in response or '- ' in response:
        score += 5  # 有分段/列表
    if len(response) > 20 and not response.endswith(response[:20]):
        score += 5  # 结尾不是开头的重复

    return min(100, score)


def eval_generation(model, tokenizer, device='cuda', max_new_tokens=512, temperature=0.85, top_p=0.85):
    """生成并自动评分"""
    model.eval()
    results = []
    total_score = 0
    total_tokens = 0
    total_time = 0

    for item in GENERATION_PROMPTS:
        setup_seed(2026)
        prompt = item['prompt']

        conversation = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt', truncation=True).to(device)

        st = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0
            )
        gen_time = time.time() - st
        gen_tokens = len(generated_ids[0]) - len(inputs['input_ids'][0])

        response = tokenizer.decode(generated_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        score = score_generation(response, item['keywords'], item['category'])
        total_score += score
        total_tokens += gen_tokens
        total_time += gen_time

        results.append({
            'prompt': prompt,
            'category': item['category'],
            'response': response[:500],  # 截断保存
            'score': score,
            'gen_tokens': gen_tokens,
            'gen_time': round(gen_time, 2),
            'tokens_per_sec': round(gen_tokens / gen_time, 2) if gen_time > 0 else 0,
        })

    avg_score = total_score / len(GENERATION_PROMPTS)
    avg_speed = total_tokens / total_time if total_time > 0 else 0

    return {
        'avg_score': round(avg_score, 2),
        'avg_speed_tokens_per_sec': round(avg_speed, 2),
        'total_gen_tokens': total_tokens,
        'total_gen_time': round(total_time, 2),
        'details': results,
    }


# ============================================================
# 4. 推理效率评估
# ============================================================

def eval_efficiency(model, tokenizer, device='cuda', num_runs=5, prompt_lengths=[32, 64, 128, 256]):
    """测量推理效率指标"""
    model.eval()
    results = []

    for seq_len in prompt_lengths:
        # 构造指定长度的输入
        dummy_text = "你好" * (seq_len // 2)
        inputs = tokenizer(dummy_text, return_tensors='pt', truncation=True,
                           max_length=seq_len).to(device)
        actual_len = inputs['input_ids'].shape[1]

        # Warmup
        with torch.no_grad():
            model.generate(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                           max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.pad_token_id)

        torch.cuda.synchronize() if device == 'cuda' else None
        torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None

        times = []
        first_token_times = []
        gen_token_counts = []

        for _ in range(num_runs):
            torch.cuda.synchronize() if device == 'cuda' else None
            st = time.time()

            with torch.no_grad():
                generated_ids = model.generate(
                    inputs=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            torch.cuda.synchronize() if device == 'cuda' else None
            elapsed = time.time() - st
            gen_count = len(generated_ids[0]) - len(inputs['input_ids'][0])
            times.append(elapsed)
            gen_token_counts.append(gen_count)

        avg_time = sum(times) / len(times)
        avg_tokens = sum(gen_token_counts) / len(gen_token_counts)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device == 'cuda' else 0

        results.append({
            'input_length': actual_len,
            'avg_gen_tokens': round(avg_tokens, 1),
            'avg_time_sec': round(avg_time, 4),
            'tokens_per_sec': round(avg_tokens / avg_time, 2) if avg_time > 0 else 0,
            'peak_memory_mb': round(peak_mem, 1),
        })

    return {'efficiency': results}


# ============================================================
# 主函数
# ============================================================

def init_model(args):
    """初始化模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            use_mla=bool(args.use_mla),
            mla_kv_dim=args.mla_kv_dim,
            mla_q_dim=args.mla_q_dim,
            mla_rope_dim=args.mla_rope_dim,
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        print(f"加载权重: {ckp}")
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="MiniMind 模型量化评估")

    # 模型参数
    parser.add_argument('--load_from', default='model', type=str)
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--weight', default='full_sft', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_mla', default=0, type=int, choices=[0, 1])
    parser.add_argument('--mla_kv_dim', type=int, default=128)
    parser.add_argument('--mla_q_dim', type=int, default=256)
    parser.add_argument('--mla_rope_dim', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)

    # 评估参数
    parser.add_argument('--tasks', default='all', type=str,
                        help="评估任务，逗号分隔: ppl,mcq,gen,eff,all")
    parser.add_argument('--eval_data', default='../dataset/sft_mini_512.jsonl', type=str,
                        help="PPL评估用的数据文件")
    parser.add_argument('--eval_samples', default=500, type=int,
                        help="PPL评估样本数")
    parser.add_argument('--eval_offset', default=-1, type=int,
                        help="PPL评估数据偏移量（-1=取文件末尾）")
    parser.add_argument('--output', default=None, type=str,
                        help="评估结果保存路径（JSON）")

    args = parser.parse_args()

    # 解析任务
    if args.tasks == 'all':
        tasks = ['ppl', 'mcq', 'gen', 'eff']
    else:
        tasks = [t.strip() for t in args.tasks.split(',')]

    print("=" * 60)
    print("MiniMind 模型量化评估")
    print("=" * 60)
    mla_str = "MLA" if args.use_mla else "GQA"
    print(f"模型: {args.weight} | 架构: {mla_str} | hidden_size: {args.hidden_size}")
    print(f"评估任务: {tasks}")
    print("=" * 60)

    setup_seed(2026)
    model, tokenizer = init_model(args)

    all_results = {
        'model': args.weight,
        'architecture': mla_str,
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.num_hidden_layers,
        'use_mla': bool(args.use_mla),
    }

    # --- PPL ---
    if 'ppl' in tasks:
        print("\n" + "=" * 60)
        print("📊 [1/4] Perplexity 评估")
        print("=" * 60)
        if os.path.exists(args.eval_data):
            # 如果 offset=-1，取文件末尾的样本作为 held-out
            if args.eval_offset == -1:
                with open(args.eval_data, 'r') as f:
                    total_lines = sum(1 for _ in f)
                offset = max(0, total_lines - args.eval_samples)
                print(f"  数据文件: {args.eval_data} (共 {total_lines} 行)")
                print(f"  使用末尾 {args.eval_samples} 行作为 held-out 集 (offset={offset})")
            else:
                offset = args.eval_offset

            ppl_result = compute_perplexity(model, tokenizer, args.eval_data,
                                            max_samples=args.eval_samples, device=args.device,
                                            offset=offset)
            all_results['perplexity'] = ppl_result
            print(f"\n  ✅ Perplexity: {ppl_result['perplexity']:.4f}")
            print(f"     Avg Loss:  {ppl_result['avg_loss']:.6f}")
            print(f"     Tokens:    {ppl_result['total_tokens']}")
        else:
            print(f"  ⚠️ 数据文件不存在: {args.eval_data}")

    # --- MCQ ---
    if 'mcq' in tasks:
        print("\n" + "=" * 60)
        print("📊 [2/4] 选择题准确率评估")
        print("=" * 60)
        mcq_result = eval_multiple_choice(model, tokenizer, device=args.device)
        all_results['multiple_choice'] = {k: v for k, v in mcq_result.items() if k != 'details'}
        print(f"\n  ✅ 准确率: {mcq_result['accuracy']:.2%} ({mcq_result['correct']}/{mcq_result['total']})")
        # 打印每题详情
        for r in mcq_result['details']:
            mark = "✓" if r['correct'] else "✗"
            print(f"     {mark} {r['question'][:25]}... 答:{r['answer']} 预测:{r['predicted']} {r['probs']}")

    # --- Generation ---
    if 'gen' in tasks:
        print("\n" + "=" * 60)
        print("📊 [3/4] 生成质量评估")
        print("=" * 60)
        gen_result = eval_generation(model, tokenizer, device=args.device)
        all_results['generation'] = {k: v for k, v in gen_result.items() if k != 'details'}
        print(f"\n  ✅ 平均评分: {gen_result['avg_score']:.1f}/100")
        print(f"     平均速度: {gen_result['avg_speed_tokens_per_sec']:.1f} tokens/s")
        for r in gen_result['details']:
            print(f"     [{r['category']}] {r['prompt'][:20]}... → 评分:{r['score']}/100, "
                  f"{r['tokens_per_sec']:.1f} tok/s")

    # --- Efficiency ---
    if 'eff' in tasks:
        print("\n" + "=" * 60)
        print("📊 [4/4] 推理效率评估")
        print("=" * 60)
        eff_result = eval_efficiency(model, tokenizer, device=args.device)
        all_results['efficiency'] = eff_result
        print(f"\n  {'输入长度':>8} {'生成数':>8} {'耗时(s)':>10} {'速度(tok/s)':>12} {'显存(MB)':>10}")
        for r in eff_result['efficiency']:
            print(f"  {r['input_length']:>8} {r['avg_gen_tokens']:>8.1f} {r['avg_time_sec']:>10.4f} "
                  f"{r['tokens_per_sec']:>12.2f} {r['peak_memory_mb']:>10.1f}")

    # --- 保存结果 ---
    if args.output is None:
        os.makedirs('../eval_results', exist_ok=True)
        args.output = f'../eval_results/eval_{args.weight}_{mla_str}_{args.hidden_size}.json'

    # 不保存详细生成文本到JSON（太大），单独保存
    save_results = {k: v for k, v in all_results.items()}
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print(f"\n📁 评估结果已保存: {args.output}")

    # 保存生成详情（含完整文本）
    if 'gen' in tasks:
        gen_detail_path = args.output.replace('.json', '_generations.json')
        with open(gen_detail_path, 'w', encoding='utf-8') as f:
            json.dump(gen_result['details'], f, ensure_ascii=False, indent=2)
        print(f"📁 生成详情已保存: {gen_detail_path}")

    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
