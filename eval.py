"""
eval.py — M3 评测体系

实现 E1-E5:
- E1: Perplexity 计算
- E2: Bits-per-byte (BPB) — 跨分词器可比指标
- E3: 损失曲线追踪
- E4: 生成质量评估（token 多样性）
- E5: 评测结果保存/加载

核心公式:
  perplexity = exp(loss)
  bits_per_byte = loss / ln(2)

Perplexity 的问题:
  - 不同分词器的 vocab_size 不同，ppl 不可比
  - BPB 是标准化指标，可以跨分词器比较
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from llm import ModelConfig, GPTLanguageModel, cross_entropy_loss

logger = logging.getLogger('nanoGPT')


def compute_perplexity(loss: float) -> float:
    """从 cross-entropy 损失计算 perplexity

    Args:
        loss: 平均 cross-entropy 损失（nats）

    Returns:
        perplexity = exp(loss)
    """
    try:
        return math.exp(min(loss, 100))  # 防止 overflow
    except OverflowError:
        return float('inf')


def compute_bpb(loss: float) -> float:
    """从 cross-entropy 损失计算 bits-per-byte

    Bits-per-byte 衡量模型平均每个字节需要多少比特编码。
    这是跨分词器可比的指标（因为 entropy 下限是每字节 8 bits）。

    Args:
        loss: 平均 cross-entropy 损失（nats）

    Returns:
        bits_per_byte = loss / ln(2)
    """
    return loss / math.log(2)


def evaluate_model(
    model: GPTLanguageModel,
    dataset: Dataset,
    batch_size: int = 16,
    max_batches: Optional[int] = None,
    device: str = 'cuda',
) -> Dict[str, float]:
    """在数据集上评估模型

    Args:
        model: GPT 模型
        dataset: 评估数据集
        batch_size: batch 大小
        max_batches: 最大评估 batch 数（None = 全部）
        device: 设备

    Returns:
        评估结果 dict
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    start_time = time.time()

    with torch.no_grad():
        # 手动循环（支持 Dataset）
        n = len(dataset)
        indices = list(range(n))
        for start in range(0, n, batch_size):
            if max_batches and num_batches >= max_batches:
                break

            batch_indices = indices[start:start + batch_size]
            batch_x = torch.stack([dataset[i][0] for i in batch_indices]).to(device)
            batch_y = torch.stack([dataset[i][1] for i in batch_indices]).to(device)

            logits, loss = model(batch_x, batch_y)
            
            # 计算有效 token 数
            n_tokens = batch_y.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            num_batches += 1

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = compute_perplexity(avg_loss)
    bpb = compute_bpb(avg_loss)
    total_tokens_per_sec = total_tokens / max(elapsed, 0.001)

    result = {
        'loss': round(avg_loss, 4),
        'perplexity': round(ppl, 2),
        'bits_per_byte': round(bpb, 4),
        'total_tokens': total_tokens,
        'num_batches': num_batches,
        'elapsed_sec': round(elapsed, 3),
        'tokens_per_sec': round(total_tokens_per_sec, 1),
    }

    model.train()
    return result


def evaluate_generation(
    model: GPTLanguageModel,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    device: str = 'cuda',
    temperature: float = 0.8,
    top_k: int = 40,
) -> Dict[str, any]:
    """评估文本生成质量

    指标:
    - unique_tokens / total_tokens: 多样性比率
    - avg_length: 平均生成长度
    - empty_ratio: 空输出比率

    Args:
        model: GPT 模型
        tokenizer: 分词器
        prompts: 评估 prompt 列表
        max_new_tokens: 最大生成 token 数
        device: 设备

    Returns:
        评估结果 dict
    """
    model.eval()
    generations = []
    unique_ratios = []

    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt)
            if not input_ids:
                continue

            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            output = model.generate(
                input_tensor, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k
            )

            generated_ids = output[0].tolist()[len(input_ids):]
            text = tokenizer.decode(generated_ids)
            generations.append({
                'prompt': prompt,
                'generated': text,
                'tokens': len(generated_ids),
            })

            # 计算 token 多样性
            if len(generated_ids) > 0:
                unique_ratio = len(set(generated_ids)) / len(generated_ids)
                unique_ratios.append(unique_ratio)

    # 汇总
    avg_unique_ratio = sum(unique_ratios) / max(len(unique_ratios), 1)
    avg_length = sum(g['tokens'] for g in generations) / max(len(generations), 1)

    result = {
        'n_prompts': len(prompts),
        'n_generations': len(generations),
        'avg_unique_token_ratio': round(avg_unique_ratio, 4),
        'avg_generation_length': round(avg_length, 1),
        'generations': generations[:5],  # 保存前 5 个示例
    }

    model.train()
    return result


def save_eval_results(results: Dict, path: str):
    """保存评测结果到 JSON"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"评测结果已保存: {path}")


def compare_tokenizers(
    model_char: GPTLanguageModel,
    model_bpe: GPTLanguageModel,
    ds_char: Dataset,
    ds_bpe: Dataset,
    device: str = 'cuda',
) -> Dict:
    """对比字符级和 BPE 分词器的性能

    使用 BPB（而非 perplexity）进行公平比较。

    Args:
        model_char: 字符级模型
        model_bpe: BPE 模型
        ds_char: 字符级数据集
        ds_bpe: BPE 数据集
        device: 设备

    Returns:
        对比结果
    """
    logger.info("评估字符级模型...")
    char_result = evaluate_model(model_char, ds_char, device=device)
    
    logger.info("评估 BPE 模型...")
    bpe_result = evaluate_model(model_bpe, ds_bpe, device=device)

    comparison = {
        'char': {
            'loss': char_result['loss'],
            'perplexity': char_result['perplexity'],
            'bits_per_byte': char_result['bits_per_byte'],
            'seq_len': ds_char.block_size if hasattr(ds_char, 'block_size') else 'N/A',
        },
        'bpe': {
            'loss': bpe_result['loss'],
            'perplexity': bpe_result['perplexity'],
            'bits_per_byte': bpe_result['bits_per_byte'],
            'seq_len': ds_bpe.block_size if hasattr(ds_bpe, 'block_size') else 'N/A',
        },
        'bpb_improvement': round(
            (char_result['bits_per_byte'] - bpe_result['bits_per_byte'])
            / max(char_result['bits_per_byte'], 1e-8) * 100, 2
        ),
    }

    logger.info(f"字符级: BPB={char_result['bits_per_byte']:.4f}, PPL={char_result['perplexity']:.2f}")
    logger.info(f"BPE:    BPB={bpe_result['bits_per_byte']:.4f}, PPL={bpe_result['perplexity']:.2f}")
    logger.info(f"BPB 改善: {comparison['bpb_improvement']:.1f}%")

    return comparison
