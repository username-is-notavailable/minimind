"""
run_all.py — 一键运行所有评测

用法：
    cd benchmark/

    # GQA Baseline 全部评测
    python run_all.py --weight full_sft

    # MLA 全部评测
    python run_all.py --weight full_sft --use_mla 1

    # 仅运行指定评测
    python run_all.py --weight full_sft --tasks ppl,ceval

    # 预训练模型（仅 PPL）
    python run_all.py --weight pretrain --tasks ppl --pretrain_mode
"""

import os
import sys
import json
import argparse
import subprocess


def run_script(script, args_list, python_exec='python'):
    """运行子脚本"""
    cmd = [python_exec, script] + args_list
    print(f"\n{'─' * 60}")
    print(f"运行: {' '.join(cmd)}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="一键运行所有评测")
    parser.add_argument('--weight', default='full_sft', type=str)
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int)
    parser.add_argument('--use_mla', default=0, type=int)
    parser.add_argument('--mla_kv_dim', type=int, default=128)
    parser.add_argument('--mla_q_dim', type=int, default=256)
    parser.add_argument('--mla_rope_dim', type=int, default=128)
    parser.add_argument('--tasks', default='all', type=str,
                        help="评测任务: ppl,ceval,gen,eff,all")
    parser.add_argument('--pretrain_mode', action='store_true',
                        help="预训练模型模式（仅 PPL）")
    parser.add_argument('--python', default='python', type=str,
                        help="Python 解释器路径")
    args = parser.parse_args()

    if args.tasks == 'all':
        if args.pretrain_mode:
            tasks = ['ppl']
        else:
            tasks = ['ppl', 'ceval', 'gen', 'eff']
    else:
        tasks = [t.strip() for t in args.tasks.split(',')]

    # 公共参数
    common = [
        '--weight', args.weight,
        '--save_dir', args.save_dir,
        '--hidden_size', str(args.hidden_size),
        '--num_hidden_layers', str(args.num_hidden_layers),
        '--use_moe', str(args.use_moe),
        '--use_mla', str(args.use_mla),
        '--mla_kv_dim', str(args.mla_kv_dim),
        '--mla_q_dim', str(args.mla_q_dim),
        '--mla_rope_dim', str(args.mla_rope_dim),
    ]

    arch = "MLA" if args.use_mla else "GQA"
    print(f"{'=' * 60}")
    print(f"MiniMind 全面评测")
    print(f"权重: {args.weight} | 架构: {arch} | 任务: {tasks}")
    print(f"{'=' * 60}")

    results_summary = {}

    if 'ppl' in tasks:
        data_path = '../dataset/pretrain_hq.jsonl' if args.pretrain_mode else '../dataset/sft_mini_512.jsonl'
        ret = run_script('eval_pretrain.py', common + ['--data_path', data_path])
        if ret == 0:
            result_file = f'results/pretrain_{arch}_{args.hidden_size}.json'
            if os.path.exists(result_file):
                results_summary['ppl'] = json.load(open(result_file))

    if 'ceval' in tasks:
        extra = ['--no_chat_template'] if args.pretrain_mode else []
        ret = run_script('eval_ceval.py', common + ['--split', 'val'] + extra)
        if ret == 0:
            result_file = f'results/ceval_{args.weight}_{arch}_{args.hidden_size}_val.json'
            if os.path.exists(result_file):
                results_summary['ceval'] = json.load(open(result_file))

    if 'gen' in tasks:
        ret = run_script('eval_generation.py', common)
        if ret == 0:
            result_file = f'results/generation_{args.weight}_{arch}_{args.hidden_size}.json'
            if os.path.exists(result_file):
                results_summary['generation'] = json.load(open(result_file))

    if 'eff' in tasks:
        ret = run_script('eval_efficiency.py', common)
        if ret == 0:
            result_file = f'results/efficiency_{args.weight}_{arch}_{args.hidden_size}.json'
            if os.path.exists(result_file):
                results_summary['efficiency'] = json.load(open(result_file))

    # 汇总输出
    print(f"\n{'=' * 60}")
    print(f"评测汇总 | {args.weight} | {arch}")
    print(f"{'=' * 60}")

    if 'ppl' in results_summary:
        print(f"  Perplexity:    {results_summary['ppl'].get('perplexity', 'N/A')}")
    if 'ceval' in results_summary:
        print(f"  C-Eval 准确率: {results_summary['ceval'].get('overall_accuracy', 'N/A')}")
    if 'generation' in results_summary:
        print(f"  生成评分:      {results_summary['generation'].get('avg_score', 'N/A')}/100")
        print(f"  生成速度:      {results_summary['generation'].get('avg_speed', 'N/A')} tok/s")
    if 'efficiency' in results_summary:
        eff = results_summary['efficiency']
        if 'model_size' in eff:
            print(f"  模型参数:      {eff['model_size']['total_params_M']}M")

    os.makedirs('results', exist_ok=True)
    summary_path = f'results/summary_{args.weight}_{arch}_{args.hidden_size}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n📁 汇总结果: {summary_path}")


if __name__ == '__main__':
    main()
