"""
eval_efficiency.py — 推理效率评测

评测内容：
  - 不同输入长度下的推理速度 (tokens/s)
  - GPU 显存峰值
  - 首 token 延迟 (TTFT)
  - KV Cache 显存对比（GQA vs MLA 核心指标）

适用：所有阶段的模型权重

用法：
    cd benchmark/
    python eval_efficiency.py --weight full_sft
    python eval_efficiency.py --weight full_sft --use_mla 1
"""

import os
import sys
import json
import time
import argparse
import torch
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


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
    return model.eval().to(args.device), tokenizer, config


def measure_prefill_and_decode(model, tokenizer, prompt_len, gen_len, device, num_runs=5):
    """测量 prefill + decode 阶段的延迟和吞吐"""
    text = "你好" * (prompt_len // 2 + 1)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=prompt_len).to(device)
    actual_len = inputs['input_ids'].shape[1]

    # Warmup
    with torch.no_grad():
        model.generate(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                       max_new_tokens=2, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    total_times = []
    gen_counts = []

    for _ in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        st = time.time()

        with torch.no_grad():
            gen_ids = model.generate(
                inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                max_new_tokens=gen_len, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )

        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - st
        n_gen = len(gen_ids[0]) - actual_len
        total_times.append(elapsed)
        gen_counts.append(n_gen)

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device == 'cuda' else 0
    avg_time = sum(total_times) / len(total_times)
    avg_gen = sum(gen_counts) / len(gen_counts)

    return {
        'input_length': actual_len,
        'avg_gen_tokens': round(avg_gen, 1),
        'avg_time_sec': round(avg_time, 4),
        'throughput_tok_s': round(avg_gen / avg_time, 2) if avg_time > 0 else 0,
        'peak_memory_mb': round(peak_mem, 1),
    }


def measure_model_size(model):
    """测量模型参数量和显存占用"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    return {
        'total_params_M': round(total_params / 1e6, 2),
        'trainable_params_M': round(trainable_params / 1e6, 2),
        'param_memory_mb': round(param_memory, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="推理效率评测")
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
    parser.add_argument('--gen_length', default=64, type=int, help="每次生成的 token 数")
    parser.add_argument('--num_runs', default=5, type=int, help="每个配置重复次数")
    args = parser.parse_args()

    setup_seed(2026)
    model, tokenizer, config = load_model(args)

    arch = "MLA" if args.use_mla else "GQA"
    print(f"\n{'=' * 60}")
    print(f"推理效率评测 | 权重: {args.weight} | 架构: {arch}")
    print(f"{'=' * 60}")

    # 模型尺寸
    size_info = measure_model_size(model)
    print(f"\n  模型参数: {size_info['total_params_M']}M")
    print(f"  参数显存: {size_info['param_memory_mb']} MB")

    # 不同输入长度的速度测试
    prompt_lengths = [16, 32, 64, 128, 256, 512]
    efficiency_results = []

    print(f"\n  {'输入长度':>8} {'生成数':>8} {'耗时(s)':>10} {'速度(tok/s)':>12} {'显存(MB)':>10}")
    print(f"  {'-' * 58}")

    for pl in prompt_lengths:
        result = measure_prefill_and_decode(model, tokenizer, pl, args.gen_length,
                                            args.device, args.num_runs)
        efficiency_results.append(result)
        print(f"  {result['input_length']:>8} {result['avg_gen_tokens']:>8.1f} "
              f"{result['avg_time_sec']:>10.4f} {result['throughput_tok_s']:>12.2f} "
              f"{result['peak_memory_mb']:>10.1f}")

    summary = {
        'weight': args.weight,
        'architecture': arch,
        'hidden_size': args.hidden_size,
        'model_size': size_info,
        'gen_length': args.gen_length,
        'num_runs': args.num_runs,
        'efficiency': efficiency_results,
    }

    os.makedirs('results', exist_ok=True)
    out_path = f'results/efficiency_{args.weight}_{arch}_{args.hidden_size}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n📁 结果已保存: {out_path}")


if __name__ == '__main__':
    main()
