"""
eval_pretrain.py — 预训练阶段评测

评测内容：
  - Perplexity (PPL)：在 pretrain held-out 数据上计算
  - 语言建模 Loss：交叉熵损失

适用模型权重：pretrain_*.pth

用法：
    cd benchmark/
    python eval_pretrain.py --weight pretrain
    python eval_pretrain.py --weight pretrain --use_mla 1
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
    return model.eval().to(args.device), tokenizer


def compute_ppl(model, tokenizer, data_path, max_length, max_samples, device):
    """在 held-out 数据上计算 PPL"""
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 取末尾作为 held-out
    held_out = lines[-max_samples:]
    print(f"  数据: {data_path}")
    print(f"  总行数: {len(lines)}, held-out: {len(held_out)}")

    total_loss = 0.0
    total_tokens = 0
    valid_samples = 0

    with torch.no_grad():
        for i, line in enumerate(held_out):
            data = json.loads(line.strip())
            text = data.get('text', '')
            if len(text) < 20:
                continue

            tokens = tokenizer(text, return_tensors='pt', truncation=True,
                               max_length=max_length, padding=False)
            input_ids = tokens['input_ids'].to(device)
            if input_ids.shape[1] < 10:
                continue

            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)

            num_tokens = input_ids.shape[1] - 1
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
            valid_samples += 1

            if (i + 1) % 200 == 0:
                curr_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
                print(f"  [{i + 1}/{len(held_out)}] PPL: {curr_ppl:.2f}")

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return {
        'perplexity': round(ppl, 4),
        'avg_loss': round(avg_loss, 6),
        'total_tokens': total_tokens,
        'valid_samples': valid_samples,
        'data_path': data_path,
    }


def main():
    parser = argparse.ArgumentParser(description="预训练阶段评测 — Perplexity")
    parser.add_argument('--weight', default='pretrain', type=str)
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_mla', default=0, type=int, choices=[0, 1])
    parser.add_argument('--mla_kv_dim', type=int, default=128)
    parser.add_argument('--mla_q_dim', type=int, default=256)
    parser.add_argument('--mla_rope_dim', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_path', default='../dataset/pretrain_hq.jsonl', type=str)
    parser.add_argument('--max_samples', default=500, type=int)
    parser.add_argument('--max_length', default=512, type=int)
    args = parser.parse_args()

    setup_seed(2026)
    model, tokenizer = load_model(args)

    arch = "MLA" if args.use_mla else "GQA"
    print(f"\n{'=' * 50}")
    print(f"预训练评测 | 权重: {args.weight} | 架构: {arch}")
    print(f"{'=' * 50}")

    result = compute_ppl(model, tokenizer, args.data_path, args.max_length,
                         args.max_samples, args.device)

    print(f"\n  ✅ Perplexity: {result['perplexity']:.4f}")
    print(f"     Avg Loss:  {result['avg_loss']:.6f}")
    print(f"     Tokens:    {result['total_tokens']}")

    result['weight'] = args.weight
    result['architecture'] = arch
    result['hidden_size'] = args.hidden_size

    os.makedirs('results', exist_ok=True)
    out_path = f'results/pretrain_{arch}_{args.hidden_size}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n📁 结果已保存: {out_path}")


if __name__ == '__main__':
    main()
