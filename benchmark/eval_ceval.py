"""
eval_ceval.py — C-Eval 中文学科评测

使用 HuggingFace 上的 ceval/ceval-exam 数据集（52个学科），
通过概率法评测模型在中文选择题上的准确率。

评测方式：取 A/B/C/D 四个选项对应 token 的 logits，
         选概率最高的与标准答案比较。

适用模型权重：full_sft_*.pth, dpo_*.pth 等 SFT 后的模型

用法：
    cd benchmark/
    # 评测 val 集（快速，每科约5-20题）
    python eval_ceval.py --weight full_sft --split val

    # 评测 test 集（完整）
    python eval_ceval.py --weight full_sft --split test

    # MLA 模型
    python eval_ceval.py --weight full_sft --use_mla 1

    # 仅评测指定科目
    python eval_ceval.py --weight full_sft --subjects computer_network,operating_system
"""

import os
import sys
import json
import argparse
import torch
import warnings
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
from transformers import AutoTokenizer
from datasets import load_dataset, get_dataset_config_names

warnings.filterwarnings('ignore')

# C-Eval 学科分类
CEVAL_CATEGORIES = {
    'STEM': ['advanced_mathematics', 'college_chemistry', 'college_physics', 'computer_architecture',
             'computer_network', 'discrete_mathematics', 'electrical_engineer', 'high_school_biology',
             'high_school_chemistry', 'high_school_mathematics', 'high_school_physics',
             'middle_school_biology', 'middle_school_chemistry', 'middle_school_mathematics',
             'middle_school_physics', 'operating_system', 'probability_and_statistics'],
    '社会科学': ['business_administration', 'college_economics', 'education_science', 'high_school_geography',
               'high_school_politics', 'law', 'mao_zedong_thought', 'marxism', 'middle_school_geography',
               'middle_school_politics', 'teacher_qualification'],
    '人文': ['art_studies', 'chinese_language_and_literature', 'high_school_chinese', 'high_school_history',
            'ideological_and_moral_cultivation', 'logic', 'middle_school_history', 'modern_chinese_history',
            'professional_tour_guide'],
    '其他': ['accountant', 'basic_medicine', 'civil_servant', 'clinical_medicine', 'college_economics',
            'environmental_impact_assessment_engineer', 'fire_engineer', 'legal_professional',
            'metrology_engineer', 'physician', 'plant_protection', 'sports_science',
            'tax_accountant', 'urban_and_rural_planner', 'veterinary_medicine'],
}


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
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


def eval_subject(model, tokenizer, subject, split, device, use_chat_template=True):
    """评测单个科目"""
    try:
        dataset = load_dataset('ceval/ceval-exam', subject, split=split)
    except Exception as e:
        print(f"  ⚠️ 加载 {subject}/{split} 失败: {e}")
        return None

    # 获取 A/B/C/D 的 token id
    choice_token_ids = {}
    for c in ['A', 'B', 'C', 'D']:
        ids = tokenizer.encode(c, add_special_tokens=False)
        choice_token_ids[c] = ids[-1]

    correct = 0
    total = 0

    with torch.no_grad():
        for item in dataset:
            question = item['question']
            choices = f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}"
            answer = item['answer']

            if use_chat_template:
                prompt_text = f"以下是关于{subject}的单项选择题，请直接给出正确答案的选项字母。\n\n{question}\n{choices}\n答案："
                conversation = [{"role": "user", "content": prompt_text}]
                text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            else:
                text = f"{question}\n{choices}\n答案："

            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
            outputs = model(input_ids=inputs['input_ids'])
            last_logits = outputs.logits[0, -1, :]

            probs = {c: last_logits[choice_token_ids[c]].item() for c in ['A', 'B', 'C', 'D']}
            predicted = max(probs, key=probs.get)

            if predicted == answer:
                correct += 1
            total += 1

    return {'correct': correct, 'total': total, 'accuracy': round(correct / total, 4) if total > 0 else 0}


def main():
    parser = argparse.ArgumentParser(description="C-Eval 中文学科评测")
    parser.add_argument('--weight', default='full_sft', type=str)
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_mla', default=0, type=int, choices=[0, 1])
    parser.add_argument('--mla_kv_dim', type=int, default=128)
    parser.add_argument('--mla_q_dim', type=int, default=256)
    parser.add_argument('--mla_rope_dim', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--split', default='val', choices=['val', 'test', 'dev'])
    parser.add_argument('--subjects', default='all', type=str,
                        help="评测科目，逗号分隔；all=全部52科")
    parser.add_argument('--no_chat_template', action='store_true',
                        help="不使用 chat template（用于评测预训练模型）")
    args = parser.parse_args()

    setup_seed(2026)
    model, tokenizer = load_model(args)

    arch = "MLA" if args.use_mla else "GQA"
    print(f"\n{'=' * 60}")
    print(f"C-Eval 评测 | 权重: {args.weight} | 架构: {arch} | split: {args.split}")
    print(f"{'=' * 60}")

    if args.subjects == 'all':
        subjects = get_dataset_config_names('ceval/ceval-exam')
    else:
        subjects = [s.strip() for s in args.subjects.split(',')]

    all_results = {}
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct = 0
    total_count = 0

    for i, subject in enumerate(subjects):
        result = eval_subject(model, tokenizer, subject, args.split, args.device,
                              use_chat_template=not args.no_chat_template)
        if result is None:
            continue

        all_results[subject] = result
        total_correct += result['correct']
        total_count += result['total']

        # 归类统计
        for cat, subs in CEVAL_CATEGORIES.items():
            if subject in subs:
                category_stats[cat]['correct'] += result['correct']
                category_stats[cat]['total'] += result['total']

        mark = "✓" if result['accuracy'] > 0.25 else " "
        print(f"  [{i + 1:2d}/{len(subjects)}] {mark} {subject:40s} "
              f"Acc: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    overall_acc = total_correct / total_count if total_count > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"分类结果:")
    for cat, stats in category_stats.items():
        cat_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {cat:10s}: {cat_acc:.2%} ({stats['correct']}/{stats['total']})")

    print(f"\n  总准确率: {overall_acc:.2%} ({total_correct}/{total_count})")
    print(f"  随机基线: 25.00%")
    print(f"{'=' * 60}")

    summary = {
        'weight': args.weight,
        'architecture': arch,
        'hidden_size': args.hidden_size,
        'split': args.split,
        'overall_accuracy': round(overall_acc, 4),
        'total_correct': total_correct,
        'total_count': total_count,
        'category_results': {k: {**v, 'accuracy': round(v['correct'] / v['total'], 4) if v['total'] > 0 else 0}
                             for k, v in category_stats.items()},
        'subject_results': all_results,
    }

    os.makedirs('results', exist_ok=True)
    out_path = f'results/ceval_{args.weight}_{arch}_{args.hidden_size}_{args.split}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n📁 结果已保存: {out_path}")


if __name__ == '__main__':
    main()
