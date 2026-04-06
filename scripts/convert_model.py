import os
import sys
import json

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore', category=UserWarning)


# MoE/MLA模型需使用此函数转换（自定义架构，加载时需 trust_remote_code=True）
def convert_torch2transformers_minimind(torch_path, transformers_path, lm_config=None, dtype=torch.float16, tokenizer_path='../model'):
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(transformers_path)
    # 兼容transformers-5.0的写法
    config_path = os.path.join(transformers_path, "tokenizer_config.json")
    json.dump({**json.load(open(config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


# LlamaForCausalLM结构兼容第三方生态（仅适用于标准GQA模型，不支持MLA/MoE）
def convert_torch2transformers_llama(torch_path, transformers_path, lm_config=None, dtype=torch.float16, tokenizer_path='../model'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    llama_config = LlamaConfig(
        vocab_size=lm_config.vocab_size,
        hidden_size=lm_config.hidden_size,
        intermediate_size=64 * ((int(lm_config.hidden_size * 8 / 3) + 64 - 1) // 64),
        num_hidden_layers=lm_config.num_hidden_layers,
        num_attention_heads=lm_config.num_attention_heads,
        num_key_value_heads=lm_config.num_key_value_heads,
        max_position_embeddings=lm_config.max_position_embeddings,
        rms_norm_eps=lm_config.rms_norm_eps,
        rope_theta=lm_config.rope_theta,
        tie_word_embeddings=True
    )
    llama_model = LlamaForCausalLM(llama_config)
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model = llama_model.to(dtype)  # 转换模型权重精度
    llama_model.save_pretrained(transformers_path)
    model_params = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(transformers_path)
    # 兼容transformers-5.0的写法
    config_path = os.path.join(transformers_path, "tokenizer_config.json")
    json.dump({**json.load(open(config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-Llama 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式 (half精度): {torch_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="MiniMind 模型格式转换工具")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small, 640=MoE, 768=Base）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否为MoE模型")
    parser.add_argument('--use_mla', default=0, type=int, choices=[0, 1], help="是否为MLA模型")
    parser.add_argument('--mla_kv_dim', type=int, default=128, help="MLA中KV的维度")
    parser.add_argument('--mla_q_dim', type=int, default=256, help="MLA中Q的维度")
    parser.add_argument('--mla_rope_dim', type=int, default=128, help="MLA中RoPE的维度")
    parser.add_argument('--vocab_size', type=int, default=6400, help="词表大小（6400=原始，32000=0.5B新分词器）")
    parser.add_argument('--tokenizer_path', type=str, default='../model', help="分词器路径")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo 等）")
    parser.add_argument('--input_dir', default='../out', type=str, help="输入权重目录")
    parser.add_argument('--output_dir', default=None, type=str, help="输出目录（默认自动生成）")
    parser.add_argument('--direction', default='t2t', choices=['t2t', 't2torch'], help="转换方向：t2t=torch→transformers, t2torch=transformers→torch")
    args = parser.parse_args()

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=args.vocab_size,
        use_moe=bool(args.use_moe),
        use_mla=bool(args.use_mla),
        mla_kv_dim=args.mla_kv_dim,
        mla_q_dim=args.mla_q_dim,
        mla_rope_dim=args.mla_rope_dim,
    )
    tokenizer_path = args.tokenizer_path

    moe_suffix = '_moe' if lm_config.use_moe else ''
    torch_path = f"{args.input_dir}/{args.weight}_{lm_config.hidden_size}{moe_suffix}.pth"

    if args.output_dir is None:
        if lm_config.use_mla:
            args.output_dir = f'../MiniMind2-Small-MLA'
        elif lm_config.use_moe:
            args.output_dir = f'../MiniMind2-MoE'
        else:
            args.output_dir = f'../MiniMind2-Small'

    if args.direction == 't2torch':
        convert_transformers2torch(args.output_dir, torch_path)
    else:
        if lm_config.use_mla or lm_config.use_moe:
            # MLA 和 MoE 模型使用 MiniMind 原生格式（自定义架构，需 trust_remote_code）
            convert_torch2transformers_minimind(torch_path, args.output_dir, lm_config=lm_config, tokenizer_path=tokenizer_path)
        else:
            # 标准 GQA 模型可转为 Llama 格式（兼容第三方生态）
            convert_torch2transformers_llama(torch_path, args.output_dir, lm_config=lm_config, tokenizer_path=tokenizer_path)
