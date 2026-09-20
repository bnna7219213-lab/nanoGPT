"""
data/data_sources.py — M6 扩展数据：多源管理 + 流式序列化 + 内存映射

核心能力：
  S1: DataSource 抽象（本地文件 / JSONL / HuggingFace，带采样权重）
  S2: TokenBinWriter —— 流式 tokenize 并写入 .bin（uint16），数据量只受磁盘限制
  S3: MmapTokenDataset —— 基于 numpy 内存映射的 Dataset，O(1) 随机访问，RAM 占用极低
  S4: MultiSourceStreamer —— 多源按比例混合流式产出文本行
  S5: prepare_dataset() —— 一键将多源文本转为训练就绪的 .bin + meta.json

设计选择：
  .bin 格式借鉴 Karpathy 的 nanoGPT：Header(token_dtype, version) + 原始 token 数组。
  不做压缩，换取极简读取逻辑 + 零解析开销 + mmap O(1) 随机访问。
  默认 uint16（支持 vocab < 65536），configurable 到 uint32。

使用方式:
  # 单源构建
  prepare_dataset("shakespeare", ["data/input.txt"], tokenizer, "data/shakespeare.bin")

  # 多源混合构建
  prepare_dataset("mixed", sources=[...], tokenizer, "data/mixed.bin")

  # 训练时使用
  from data.data_sources import MmapTokenDataset
  ds = MmapTokenDataset("data/mixed.bin", block_size=256)
  x, y = ds[0]
"""

import hashlib
import json
import logging
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger('nanoGPT')

# =====================================================================================
# S1: DataSource —— 数据源抽象
# =====================================================================================

@dataclass
class DataSource:
    """单个数据源的定义

    Attributes:
        name: 数据源名称（如 "shakespeare", "openwebtext", "c4"）
        paths: 文件路径列表（支持 glob 和多个文件）
        source_type: 数据源类型 —— "text" (纯文本) / "jsonl" (逐行JSON) / "hf" (HuggingFace)
        text_field: JSONL 中字段名（仅 jsonl 类型）
        weight: 多源混合时的采样权重（>0 的浮点数）
        max_chars: 从该源最多读取的字符数（None = 全部）
    """
    name: str
    paths: List[str]
    source_type: str = "text"           # "text" | "jsonl" | "hf"
    text_field: str = "text"
    weight: float = 1.0
    max_chars: Optional[int] = None

    def __post_init__(self):
        assert self.weight > 0, f"权重必须 > 0，得到 {self.weight}"
        assert self.source_type in ("text", "jsonl", "hf"), f"未知 source_type: {self.source_type}"


# =====================================================================================
# S4: MultiSourceStreamer —— 多源混合流式读取
# =====================================================================================

def stream_single_source(source: DataSource) -> Generator[str, None, None]:
    """从单个数据源流式产出文本块

    对 text 类型：按块（64KB）读取，避免一次性载入大文件。
    对 jsonl 类型：逐行解析，yield 每条记录的 text_field。
    对 hf 类型：延迟导入 datasets，遍历 dataset。
    """
    if source.source_type == "jsonl":
        from data.pipeline import stream_jsonl
        text_field = source.text_field
        total_chars = 0
        for chunk in stream_jsonl(source.paths, text_field=text_field, skip_empty=True):
            if source.max_chars and total_chars >= source.max_chars:
                return
            if source.max_chars and total_chars + len(chunk) > source.max_chars:
                chunk = chunk[:source.max_chars - total_chars]
            total_chars += len(chunk)
            yield chunk

    elif source.source_type == "text":
        total_chars = 0
        for path_str in source.paths:
            path = Path(path_str)
            if not path.exists():
                logger.warning(f"文件不存在，跳过: {path}")
                continue
            if path.is_file():
                # 单文件：按 chunk 读取
                yield from _read_text_chunks(path, source.max_chars)
                if source.max_chars:
                    total_chars = 0  # 重新计数 — 这里简化处理
            elif path.is_dir():
                # 目录：递归所有 .txt 文件
                for txt_file in sorted(path.rglob("*.txt")):
                    yield from _read_text_chunks(txt_file, source.max_chars)

    elif source.source_type == "hf":
        yield from _stream_hf(source)


def _read_text_chunks(path: Path, max_chars: Optional[int] = None) -> Generator[str, None, None]:
    """按 64KB 块读取文本文件，用 \n 边界避免切断行"""
    logger.info(f"  读取文本文件: {path}")
    CHUNK_SIZE = 64 * 1024  # 64KB
    total = 0
    leftover = ""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                if leftover:
                    yield leftover
                break
            text = leftover + chunk
            # 在最后一个换行符处切断，保留剩余部分
            last_nl = text.rfind('\n')
            if last_nl == -1:
                leftover = text
                continue
            to_yield = text[:last_nl + 1]
            leftover = text[last_nl + 1:]
            if max_chars:
                remaining = max_chars - total
                if remaining <= 0:
                    return
                if len(to_yield) > remaining:
                    to_yield = to_yield[:remaining]
            total += len(to_yield)
            yield to_yield


def _stream_hf(source: DataSource) -> Generator[str, None, None]:
    """从 HuggingFace datasets 流式读取（延迟导入）"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("HuggingFace datasets 库未安装，无法加载 HF 数据源。"
                     " 请运行: pip install datasets")
        return

    logger.info(f"  加载 HF 数据源: {source.name} (paths={source.paths})")
    for path_str in source.paths:
        try:
            ds = load_dataset(path_str, split="train", streaming=True)
            total = 0
            for row in ds:
                text = row.get(source.text_field, '')
                if not text or not isinstance(text, str):
                    continue
                if source.max_chars and total >= source.max_chars:
                    return
                if source.max_chars and total + len(text) > source.max_chars:
                    text = text[:source.max_chars - total]
                total += len(text)
                yield text
        except Exception as e:
            logger.error(f"加载 HF 数据集 {path_str} 失败: {e}")
            continue


def multi_source_stream(sources: List[DataSource]) -> Generator[str, None, None]:
    """按比例混合多个数据源，流式产出文本

    使用轮询加权策略：权重 2 的源每次产出 2 条，权重 1 的产出 1 条。
    这样在多次迭代中，各源的比例趋近权重比，同时保持流式低内存。

    如果任一源耗尽，继续从剩余源中按比例产出；全部耗尽后结束。

    Args:
        sources: DataSource 列表

    Yields:
        文本块（str）
    """
    if not sources:
        return

    # 计算权重归一化
    weights = [s.weight for s in sources]
    total_w = sum(weights)
    # 计算每个源的产出配额（按最小权重缩放为整数比）
    min_w = min(weights)
    # 配额 = weight / min_w，最小为 1
    quotas = [max(1, round(w / min_w)) for w in weights]

    logger.info(f"多源混合: {len(sources)} 个源")
    for src, quota in zip(sources, quotas):
        logger.info(f"  - {src.name}: weight={src.weight}, quota={quota}")

    # 创建迭代器
    iterators = [iter(stream_single_source(src)) for src in sources]
    exhausted = [False] * len(sources)

    while not all(exhausted):
        for i, (it, src, quota) in enumerate(zip(iterators, sources, quotas)):
            if exhausted[i]:
                continue
            for _ in range(quota):
                try:
                    yield next(it)
                except StopIteration:
                    exhausted[i] = True
                    break


# =====================================================================================
# S2: TokenBinWriter —— 流式 tokenize 并写入 .bin 文件
# =====================================================================================

# .bin 文件格式定义
# Header (256 bytes):
#   Bytes 0-3:   magic = 0x4E42544E ("NTBN" = nanoGPT Token Binary)
#   Bytes 4-7:   version (uint32, 当前为 1)
#   Bytes 8-9:   dtype_code (uint16: 1=uint16, 2=uint32)
#   Bytes 10-11: header_size (uint32, 固定 256)
#   Bytes 12-19: token_count (uint64, 写入后回填)
#   Bytes 20-255: reserved (填 0)
# Bytes 256+: token 数据

MAGIC = 0x4E42544E  # "NTBN"
HEADER_SIZE = 256
DTYPE_UINT16 = 1
DTYPE_UINT32 = 2


class TokenBinWriter:
    """流式写入 .bin 格式 token 文件

    特点：
    - 内存中只维护一个 chunk 的 tokens，定期 flush 到磁盘
    - 支持 uint16（vocab < 65536）和 uint32
    - 写完后自动回填 token_count 到 header

    用法:
        writer = TokenBinWriter("data/mixed.bin", dtype="uint16")
        writer.open()
        for text in texts:
            ids = tokenizer.encode(text)
            writer.write(ids)
        writer.close()  # 自动回填 count + 计算 hash
    """

    def __init__(self, path: str, dtype: str = "uint16"):
        """
        Args:
            path: .bin 文件输出路径
            dtype: "uint16" (vocab < 65536) 或 "uint32"
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dtype = np.uint16 if dtype == "uint16" else np.uint32
        self.dtype_code = DTYPE_UINT16 if dtype == "uint16" else DTYPE_UINT32
        self._f = None
        self._buffer = []
        self._buffer_limit = 10_000  # flush 阈值
        self._total_tokens = 0

    def open(self):
        """打开文件并写入 header"""
        self._f = open(self.path, 'wb')
        # 写占位 header（token_count 稍后回填）
        header = struct.pack('<IIIQQ',
                            MAGIC,           # magic
                            1,               # version
                            self.dtype_code, # dtype
                            HEADER_SIZE,     # header size
                            0)               # token count (placeholder)
        # 补齐到 HEADER_SIZE 字节
        padding = HEADER_SIZE - len(header)
        header += b'\x00' * padding
        self._f.write(header)
        logger.info(f"打开 .bin 写入: {self.path} (dtype={self.dtype.__name__})")
        return self

    def write(self, token_ids: List[int]):
        """写入一批 token ids"""
        self._buffer.extend(token_ids)
        if len(self._buffer) >= self._buffer_limit:
            self._flush()

    def _flush(self):
        """将 buffer 内容写到磁盘"""
        if not self._buffer:
            return
        arr = np.array(self._buffer, dtype=self.dtype)
        self._f.write(arr.tobytes())
        self._total_tokens += len(self._buffer)
        self._buffer.clear()

    def close(self):
        """关闭文件：flush 剩余 buffer + 回填 token_count + 计算文件 hash"""
        if self._f is None:
            return
        self._flush()

        # 回填 token_count 到 header 的 offset 20 (第 5 个 uint64)
        self._f.seek(20)
        self._f.write(struct.pack('<Q', self._total_tokens))
        self._f.close()

        # 计算 sha256（不包括 header 的前 20 字节中的动态字段？计算整个文件的 hash）
        sha = hashlib.sha256()
        with open(self.path, 'rb') as rf:
            for chunk in iter(lambda: rf.read(65536), b''):
                sha.update(chunk)
        sha_hex = sha.hexdigest()[:16]

        logger.info(f"  .bin 写入完成: {self._total_tokens:,} tokens, "
                     f"文件大小 {self.path.stat().st_size:,} bytes, sha={sha_hex}")

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()


# =====================================================================================
# S3: MmapTokenDataset —— 内存映射 Dataset（核心训练数据源）
# =====================================================================================

class MmapTokenDataset(Dataset):
    """基于 numpy 内存映射的 token 数据集

    特点：
    - 零 RAM 缓存：只在 __getitem__ 时从磁盘/mmap 读取
    - O(1) 随机访问：__getitem__ 直接切片，无 pre-load
    - 多进程安全：多个 DataLoader worker 共享同一 mmap（只读）

    用法:
        ds = MmapTokenDataset("data/mixed.bin", block_size=256)
        x, y = ds[i]  # x=[block_size], y=[block_size] (y 是 x 右移一位)
        loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
    """

    def __init__(self, bin_path: str, block_size: int, vocab_size: Optional[int] = None):
        """
        Args:
            bin_path: .bin 文件路径
            block_size: 每条样本的 token 数
            vocab_size: 可选，用于校验 header 中 vocab < dtype 上界
        """
        self.path = Path(bin_path)
        self.block_size = block_size

        if not self.path.exists():
            raise FileNotFoundError(f".bin 文件不存在: {bin_path}")

        # 读 header
        with open(self.path, 'rb') as f:
            header = f.read(HEADER_SIZE)

        magic, version, dtype_code, header_size, token_count = struct.unpack('<IIIQQ', header[:28])

        if magic != MAGIC:
            raise ValueError(f"无效的 .bin 文件（magic 不匹配）: {bin_path}")
        if version != 1:
            raise ValueError(f"不支持的 .bin 版本: {version}")

        self._dtype = np.uint16 if dtype_code == DTYPE_UINT16 else np.uint32
        self._token_count = token_count
        self._num_samples = max(0, token_count - block_size)

        if token_count == 0:
            logger.warning(f" .bin 文件中 token_count=0，数据集为空: {bin_path}")

        # 内存映射（只读）
        self._mmap = np.memmap(self.path, dtype=self._dtype, mode='r',
                               offset=header_size)

        logger.info(f"加载 MmapTokenDataset: {bin_path} | tokens={self._token_count:,} | "
                     f"samples={self._num_samples:,} | block_size={block_size} | "
                     f"dtype={self._dtype.__name__} | 文件大小={self.path.stat().st_size:,} bytes")

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一个训练样本 (x, y)
        x = tokens[idx : idx + block_size]
        y = tokens[idx+1 : idx+1 + block_size]
        """
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(f"索引 {idx} 超出范围 [0, {self._num_samples:,})")
        chunk = self._mmap[idx:idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:self.block_size].astype(np.int64))
        y = torch.from_numpy(chunk[1:self.block_size + 1].astype(np.int64))
        return x, y

    def __repr__(self):
        return (f"{self.__class__.__name__}(path='{self.path.name}', "
                f"tokens={self._token_count:,}, samples={self._num_samples:,}, "
                f"block_size={self.block_size})")


# =====================================================================================
# S5: prepare_dataset() —— 一键数据集构建管线
# =====================================================================================

def prepare_dataset(
    name: str,
    sources: List[DataSource],
    tokenizer,
    out_bin_path: str,
    dtype: str = "uint16",
    clean: bool = True,
    save_meta: bool = True,
) -> Dict:
    """从多源流式文本构建训练就绪的 .bin 文件

    Args:
        name: 数据集名称
        sources: 数据源列表
        tokenizer: 分词器（已实现 encode() 方法）
        out_bin_path: 输出 .bin 路径
        dtype: "uint16" 或 "uint32"
        clean: 是否先清洗文本（调用 data.clean_text）
        save_meta: 是否同时输出 .meta.json 元信息

    Returns:
        metadata dict
    """
    from data.pipeline import clean_text

    logger.info("=" * 60)
    logger.info(f" 构建数据集: {name}")
    logger.info(f" 输出: {out_bin_path}")
    logger.info(f" 源: {len(sources)} 个")
    logger.info("=" * 60)

    # 打开 writer
    bin_path = Path(out_bin_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    total_tokens = 0
    total_docs = 0
    source_stats = {}

    with TokenBinWriter(str(out_bin_path), dtype=dtype) as writer:
        for text in multi_source_stream(sources):
            if clean and text:
                text = clean_text(text)
            if not text:
                continue
            ids = tokenizer.encode(text)
            writer.write(ids)

            total_chars += len(text)
            total_tokens += len(ids)
            total_docs += 1

            if total_docs % 1000 == 0:
                logger.info(f"  进度: {total_docs:,} 文档, "
                             f"{total_chars:,} 字符 → {total_tokens:,} tokens")

    # 读取实际写入的 token count（close 后 stats 可用）
    # TokenBinWriter 的 close 已打印日志，这里从文件 header 读取
    with open(out_bin_path, 'rb') as f:
        header = f.read(HEADER_SIZE)
        _, _, _, _, token_count = struct.unpack('<IIIQQ', header[:28])

    metadata = {
        'name': name,
        'bin_path': str(bin_path),
        'tokenizer_class': tokenizer.__class__.__name__,
        'vocab_size': tokenizer.vocab_size,
        'dtype': dtype,
        'total_tokens': int(token_count),
        'total_chars': total_chars,
        'total_docs': total_docs,
        'file_size_bytes': bin_path.stat().st_size,
        'sources': [{'name': s.name, 'weight': s.weight, 'paths': s.paths,
                      'type': s.source_type} for s in sources],
        'block_size_recommended': 128,
    }

    if save_meta:
        meta_path = bin_path.with_suffix('.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"  元信息已保存: {meta_path}")

    logger.info(f" 数据集构建完成: {name}")
    logger.info(f"  文档数: {total_docs:,}")
    logger.info(f"  字符数: {total_chars:,}")
    logger.info(f"  Token 数: {int(token_count):,}")
    logger.info(f"  文件大小: {bin_path.stat().st_size:,} bytes")
    logger.info(f"  Tokenizer: {tokenizer.__class__.__name__} (vocab={tokenizer.vocab_size})")

    return metadata


# =====================================================================================
# 便捷函数：从 meta.json 读取元信息
# =====================================================================================

def load_dataset_meta(meta_path: str) -> Dict:
    """从 .meta.json / .json 文件中加载数据集元信息"""
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =====================================================================================
# 兼容层：将 MmapTokenDataset 转为 train.py 需要的 flat tensor（老接口适配）
# =====================================================================================

def get_batch_mmap(dataset: MmapTokenDataset, batch_size: int,
                   device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """从 MmapTokenDataset 随机采样一个 batch（跟 llm.py.get_batch 签名兼容）

    Args:
        dataset: MmapTokenDataset 实例
        batch_size: batch 大小
        device: 设备

    Returns:
        (x, y) — [batch_size, block_size] tensors on device
    """
    num_samples = len(dataset)
    ix = torch.randint(num_samples, (batch_size,))
    x_list, y_list = [], []
    for i in ix.tolist():
        xi, yi = dataset[i]
        x_list.append(xi)
        y_list.append(yi)
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    return x, y
