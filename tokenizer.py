"""
tokenizer.py — BPE (Byte-Pair Encoding) 分词器 + 字符级分词器

BPE 训练流程:
1. 将文本转为 UTF-8 字节序列（初始词表 = 256 个字节，id 0-255）
2. 统计相邻 pair 频率
3. 合并最高频 pair 为一个新 token（id 从 256 开始）
4. 重复直到词表达到目标大小

API:
    tok = BPETokenizer.train(text, vocab_size=500)
    ids = tok.encode("Hello world")
    text = tok.decode(ids)
    tok.save("tokenizer.json")
    tok = BPETokenizer.load("tokenizer.json")
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """统计相邻 pair 的频率"""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """将指定 pair 替换为新 token id"""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    """BPE 分词器

    支持:
    - 从文本训练 (train)
    - 编码/解码 (encode/decode)
    - 持久化 (save/load)

    属性:
        vocab_size: 词表大小
        merges: pair -> new_id 的合并规则字典
    """

    def __init__(self):
        self.vocab_size: int = 0
        self.merges: Dict[Tuple[int, int], int] = {}  # (id1, id2) -> new_id
        self._trained = False

    @classmethod
    def train(cls, text: str, vocab_size: int = 500, verbose: bool = False) -> 'BPETokenizer':
        """从文本训练 BPE 分词器

        Args:
            text: 训练语料 (str)
            vocab_size: 目标词表大小
            verbose: 是否打印训练进度

        Returns:
            训练好的 BPETokenizer 实例
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size 至少为 256，得到 {vocab_size}")

        tok = cls()
        num_merges = vocab_size - 256

        # 将文本转为 UTF-8 字节 id (0-255)
        ids = list(text.encode('utf-8'))
        merges = {}  # (id1, id2) -> new_id

        for step in range(num_merges):
            # 统计 pair 频率
            stats = _get_stats(ids)
            if not stats:
                break

            # 找到最高频 pair（相同时取字典序最小的，确保确定性）
            pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            freq = stats[pair]

            # 分配新 id（从 256 开始递增）
            new_id = 256 + len(merges)

            # 执行合并
            ids = _merge(ids, pair, new_id)
            merges[pair] = new_id

            if verbose and step % 50 == 0:
                print(f"  BPE merge {step}/{num_merges}: pair={pair} freq={freq} -> id={new_id}")

        tok.merges = merges
        tok.vocab_size = 256 + len(merges)
        tok._trained = True

        if verbose:
            print(f"BPE 训练完成: vocab_size={tok.vocab_size}, merges={len(merges)}")

        return tok

    def encode(self, text: str) -> List[int]:
        """将文本编码为 token id 序列

        使用迭代合并：每轮扫描所有当前 merges，找最高优先级的可合并对。
        按照 merges 的 id 顺序（即训练时的优先级顺序）进行合并。

        Args:
            text: 输入文本

        Returns:
            token id 列表
        """
        if not self._trained:
            raise RuntimeError("分词器未训练，请先调用 train() 或 load()")

        if not text:
            return []

        # 转为字节 ids
        ids = list(text.encode('utf-8'))

        # 迭代合并：每次找第一个可应用的 merge（按 merge id 顺序）
        # merge id 越小 = 训练时优先级越高
        changed = True
        while changed:
            changed = False
            # 按 merge id 从小到大排序，第一个可应用的就是最优选择
            best_pair = None
            best_new_id = None
            best_priority = float('inf')

            for (id1, id2), new_id in self.merges.items():
                if new_id < best_priority and (id1, id2) in zip(ids, ids[1:]):
                    best_pair = (id1, id2)
                    best_new_id = new_id
                    best_priority = new_id

            if best_pair is not None:
                ids = _merge(ids, best_pair, best_new_id)
                changed = True

        return ids

    def decode(self, ids: List[int]) -> str:
        """将 token id 序列解码为文本

        将每个 token id 转为对应字节序列再解码为 UTF-8。

        Args:
            ids: token id 列表

        Returns:
            解码后的文本
        """
        # 构建 id -> bytes 映射
        # 初始: id 0-255 对应单个字节
        id_to_bytes = {i: bytes([i]) for i in range(256)}

        # 根据 merges 构建复合 token 的字节表示
        for (id1, id2), new_id in self.merges.items():
            id_to_bytes[new_id] = id_to_bytes[id1] + id_to_bytes[id2]

        # 拼接所有字节
        result = bytearray()
        for idx in ids:
            if idx in id_to_bytes:
                result.extend(id_to_bytes[idx])
            else:
                result.extend(b'\xef\xbf\xbd')  # 未知 token: 替换字符

        return result.decode('utf-8', errors='replace')

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """批量编码"""
        return [self.encode(t) for t in texts]

    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        """批量解码"""
        return [self.decode(ids) for ids in batch_ids]

    def save(self, path: str):
        """保存分词器到 JSON 文件

        格式:
        {
            "vocab_size": N,
            "merges": [[id1, id2, new_id], ...]
        }
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # 序列化 merges: key 是 tuple，需要转为 list
        merges_list = [[p[0], p[1], new_id] for p, new_id in self.merges.items()]

        data = {
            'vocab_size': self.vocab_size,
            'merges': merges_list,
        }

        tmp_path = path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """从 JSON 文件加载分词器"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tok = cls()
        tok.vocab_size = data['vocab_size']

        # 反序列化 merges
        for id1, id2, new_id in data['merges']:
            tok.merges[(id1, id2)] = new_id

        tok._trained = True
        return tok

    def __repr__(self):
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"


# =====================================================================================
# 字符级分词器（保留用于消融对比）
# =====================================================================================

class CharTokenizer:
    """字符级分词器（原始实现，用于对比实验）"""

    def __init__(self):
        self.vocab_size: int = 0
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._trained = False

    @classmethod
    def train(cls, text: str) -> 'CharTokenizer':
        """从文本构建字符词表"""
        tok = cls()
        chars = sorted(list(set(text)))
        tok.stoi = {ch: i for i, ch in enumerate(chars)}
        tok.itos = {i: ch for i, ch in enumerate(chars)}
        tok.vocab_size = len(chars)
        tok._trained = True
        return tok

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join([self.itos.get(i, '?') for i in ids])

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch_ids]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'vocab_size': self.vocab_size,
            'stoi': self.stoi,
        }
        tmp_path = path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls()
        tok.stoi = data['stoi']
        tok.itos = {int(v): k for k, v in tok.stoi.items()}
        tok.vocab_size = data['vocab_size']
        tok._trained = True
        return tok

    def __repr__(self):
        return f"CharTokenizer(vocab_size={self.vocab_size})"
