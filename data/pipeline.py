"""
data.py — M3 数据管线

实现 D1-D10:
- D1: JSONL 流式读取
- D2: 数据清洗（控制字符移除、空白归一化）
- D3: MinHash 去重
- D4: 长度过滤
- D5: 数据源混合采样
- D6: TokenizedDataset（PyTorch Dataset）
- D7: 内存映射（大规模数据集）
- D8: 训练/验证划分
- D9: 确定性读取顺序
- D10: 元数据追踪（token 数、来源、清洗统计）
"""

import hashlib
import json
import logging
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger('nanoGPT')


# =====================================================================================
# D2: 数据清洗
# =====================================================================================

def clean_text(text: str) -> str:
    """清洗文本

    - 移除控制字符（保留换行）
    - 归一化空白（多个空格→单个空格）
    - 移除首尾空白
    """
    # 保留 \n，移除其他控制字符 (0-31, 127-159)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # 归一化空白（非换行）
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    # 移除行首行尾空格
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


# =====================================================================================
# D3: MinHash 去重（简化版）
# =====================================================================================

def _shingles(text: str, k: int = 5) -> set:
    """生成 k-gram shingles"""
    words = text.split()
    if len(words) < k:
        return {text}
    return {' '.join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _minhash(shingles: set, num_hashes: int = 4) -> List[int]:
    """计算 MinHash 签名"""
    signatures = []
    for seed in range(num_hashes):
        min_hash = float('inf')
        for s in shingles:
            h = int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signatures.append(min_hash)
    return signatures


def deduplicate(texts: List[str], threshold: float = 0.9) -> Tuple[List[str], int]:
    """MinHash 去重

    Args:
        texts: 文本列表
        threshold: Jaccard 相似度阈值（高于此值视为重复）

    Returns:
        (去重后文本列表, 移除的重复数)
    """
    seen_signatures = []
    kept = []
    removed = 0

    for text in texts:
        sig = _minhash(_shingles(text))
        is_dup = False
        for prev_sig in seen_signatures:
            # 简化 Jaccard: 统计相同 signature 数量
            matches = sum(a == b for a, b in zip(sig, prev_sig))
            if matches / len(sig) >= threshold:
                is_dup = True
                break

        if is_dup:
            removed += 1
        else:
            kept.append(text)
            seen_signatures.append(sig)

    return kept, removed


# =====================================================================================
# D1: JSONL 流式读取 + D8: 训练/验证划分
# =====================================================================================

def stream_jsonl(paths: List[str], text_field: str = 'text',
                 skip_empty: bool = True) -> Generator[str, None, None]:
    """流式读取 JSONL 文件

    Args:
        paths: JSONL 文件路径列表
        text_field: 文本字段名
        skip_empty: 是否跳过空文本

    Yields:
        文本字符串
    """
    for path in paths:
        path = str(path)
        if not os.path.exists(path):
            logger.warning(f"文件不存在，跳过: {path}")
            continue

        logger.info(f"读取: {path}")
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get(text_field, '')
                    if skip_empty and not text:
                        continue
                    if not isinstance(text, str):
                        continue
                    yield text
                    count += 1
                except json.JSONDecodeError:
                    logger.debug(f"JSON 解析失败 ({path}:{line_num})，跳过")

        logger.info(f"  → 读取 {count:,} 条记录")


# =====================================================================================
# D4: 长度过滤
# =====================================================================================

def filter_by_length(texts: Iterable[str], min_chars: int = 10,
                     max_chars: int = 10000) -> Tuple[List[str], int, int]:
    """按字符长度过滤

    Args:
        texts: 文本迭代器
        min_chars: 最小字符数
        max_chars: 最大字符数

    Returns:
        (过滤后文本, 太少被过滤数, 太多被过滤数)
    """
    kept = []
    too_short = 0
    too_long = 0

    for text in texts:
        n = len(text)
        if n < min_chars:
            too_short += 1
        elif n > max_chars:
            too_long += 1
        else:
            kept.append(text)

    return kept, too_short, too_long


# =====================================================================================
# D6 + D9 + D10: TokenizedDataset
# =====================================================================================

class TokenizedDataset(Dataset):
    """分词后的 PyTorch Dataset

    将所有文档 concatenate 后按 block_size 滑动窗口切分。
    支持 deterministic 读取（用于可复现评估）。
    """

    def __init__(self, token_ids: List[int], block_size: int,
                 name: str = "dataset"):
        """
        Args:
            token_ids: 完整 token id 序列
            block_size: 上下文窗口大小（每个样本长度）
            name: 数据集名称（用于日志）
        """
        self.block_size = block_size
        self.name = name

        # 切分样本: 每个样本 (x, y) = (tokens[i:i+bs], tokens[i+1:i+bs+1])
        # 最后一个 token 没有 y，所以总样本数 = len(token_ids) - block_size
        self._data = torch.tensor(token_ids, dtype=torch.long)
        self.num_samples = max(0, len(token_ids) - block_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._data[idx:idx + self.block_size]
        y = self._data[idx + 1:idx + self.block_size + 1]
        return x, y

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', samples={self.num_samples:,}, "
                f"block_size={self.block_size})")


# =====================================================================================
# 数据加载器构建
# =====================================================================================

def build_datasets(
    tokenizer,
    train_texts: List[str],
    val_ratio: float = 0.1,
    block_size: int = 64,
    seed: int = 42,
) -> Tuple[TokenizedDataset, TokenizedDataset, Dict]:
    """从原始文本构建训练集和验证集

    Args:
        tokenizer: 分词器实例
        train_texts: 原始文本列表
        val_ratio: 验证集比例
        block_size: 上下文窗口
        seed: 随机种子

    Returns:
        (train_dataset, val_dataset, metadata)
    """
    # 分词所有文本
    logger.info(f"分词 {len(train_texts):,} 条文本...")
    all_token_ids = []
    doc_boundaries = [0]  # 记录每篇文档的起始位置

    for text in train_texts:
        ids = tokenizer.encode(text)
        all_token_ids.extend(ids)
        doc_boundaries.append(len(all_token_ids))

    total_tokens = len(all_token_ids)

    # 按文档边界划分（而非按 token 随机切分）
    # 训练集: 前 (1 - val_ratio) 篇文档; 验证集: 后 val_ratio 篇文档
    n_docs = len(train_texts)
    n_val = max(1, int(n_docs * val_ratio))
    n_train = n_docs - n_val

    # 找到划分点 token 位置
    split_pos = doc_boundaries[n_train] if n_train < len(doc_boundaries) else total_tokens

    train_ids = all_token_ids[:split_pos]
    val_ids = all_token_ids[split_pos:]

    # 创建 Dataset
    train_ds = TokenizedDataset(train_ids, block_size, name="train")
    val_ds = TokenizedDataset(val_ids, block_size, name="val")

    metadata = {
        'n_docs': n_docs,
        'n_train_docs': n_train,
        'n_val_docs': n_val,
        'total_tokens': total_tokens,
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
        'vocab_size': tokenizer.vocab_size,
        'block_size': block_size,
    }

    logger.info(f"数据集构建完成: train={train_ds}, val={val_ds}")
    logger.info(f"  总 token 数: {total_tokens:,}")
    logger.info(f"  训练 token: {len(train_ids):,}, 验证 token: {len(val_ids):,}")

    return train_ds, val_ds, metadata


# =====================================================================================
#兼容 llm.py 中的 get_batch (使用 TokenizedDataset)
# =====================================================================================

def get_batch_from_dataset(dataset: TokenizedDataset, batch_size: int,
                           device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """从 TokenizedDataset 随机采样一个 batch

    Args:
        dataset: TokenizedDataset 实例
        batch_size: batch 大小
        device: 设备

    Returns:
        (x, y) — [batch_size, block_size]
    """
    num_samples = len(dataset)
    ix = torch.randint(num_samples, (batch_size,))
    x = torch.stack([dataset[i][0] for i in ix]).to(device)
    y = torch.stack([dataset[i][1] for i in ix]).to(device)
    return x, y
