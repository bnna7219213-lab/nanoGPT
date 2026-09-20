"""
scripts/prepare_data.py — M6 数据构建工具

用法:
  # 1. 从本地文本构建（单源）
  python scripts/prepare_data.py --name shakespeare --input data/input.txt --tokenizer bpe

  # 2. 从本地 JSONL 构建（多文件）
  python scripts/prepare_data.py --name mydata --jsonl data/corpus/*.jsonl --jsonl-field text

  # 3. 从 HuggingFace 数据集构建
  python scripts/prepare_data.py --name openwebtext --hf Skylion007/openwebtext --hf-limit 1000000

  # 4. 多源混合构建
  python scripts/prepare_data.py --name mixed \\
      --source shakespeare:1:data/input.txt \\
      --source openwebtext:2:data/jsonl/*.jsonl:jsonl \\
      --tokenizer bpe

  # 5. 列出已知数据集
  python scripts/prepare_data.py --list

  # 6. 重新对已有文本构建（分词器变更后）
  python scripts/prepare_data.py --name shakespeare --input data/input.txt --tokenizer bpe --force
"""
import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.data_sources import DataSource, prepare_dataset
from tokenizer import BPETokenizer, CharTokenizer


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def collect_sources(args) -> list:
    """根据 CLI 参数构造 DataSource 列表"""
    sources = []

    # 模式 1: 单源 --input (纯文本)
    if args.input:
        for path_str in args.input:
            # 支持 glob 展开
            paths = glob.glob(path_str) or [path_str]
            sources.append(DataSource(
                name=Path(path_str).stem,
                paths=paths,
                source_type="text",
            ))

    # 模式 2: JSONL 文件
    if args.jsonl:
        for path_str in args.jsonl:
            paths = glob.glob(path_str) or [path_str]
            sources.append(DataSource(
                name=Path(path_str).stem,
                paths=paths,
                source_type="jsonl",
                text_field=args.jsonl_field,
            ))

    # 模式 3: HuggingFace 数据集
    if args.hf:
        for hf_name in args.hf:
            sources.append(DataSource(
                name=hf_name.replace('/', '_'),
                paths=[hf_name],
                source_type="hf",
                text_field=args.hf_field,
                max_chars=args.hf_limit,
            ))

    # 模式 4: 多源混合 --source specs
    if args.source:
        for spec in args.source:
            # spec 格式: name:weight:paths[:type]
            parts = spec.split(':')
            name = parts[0]
            weight = float(parts[1]) if len(parts) > 1 else 1.0
            path_str = parts[2]
            src_type = parts[3] if len(parts) > 3 else "text"

            paths = glob.glob(path_str) or [path_str]
            sources.append(DataSource(
                name=name,
                paths=paths,
                source_type=src_type,
                weight=weight,
                text_field=args.jsonl_field if src_type == "jsonl" else "text",
            ))

    return sources


def auto_detect_and_build(args):
    """自动模式：如果没有指定任何源，尝试从 data/ 目录обнаружить文本"""
    data_dir = Path("data")
    if not data_dir.exists():
        print("  data/ 目录不存在")
        return

    # 查找 .txt 和 .jsonl 文件
    txt_files = sorted(data_dir.glob("*.txt"))
    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    sources = []
    for f in txt_files:
        sources.append(DataSource(name=f.stem, paths=[str(f)], source_type="text"))
    for f in jsonl_files:
        sources.append(DataSource(name=f.stem, paths=[str(f)], source_type="jsonl"))

    if not sources:
        print(f"  在 data/ 中未找到 .txt 或 .jsonl 文件")
        print(f"  请将数据文件放入 data/ 目录，或用 --input 显式指定路径")
        return

    print(f"  自动发现 {len(sources)} 个数据文件:")
    for src in sources:
        print(f"    - {src.name}: {src.paths[0]} ({src.source_type})")

    # 在第一个文本文件上训练分词器（全部加载在小语料上不是问题）
    # 对于大语料，build_datasets + sample 更合适
    print(f"\n  训练 {args.tokenizer} 分词器...")
    sample_texts = []
    for src in sources[:3]:  # 最多前3个
        for p in src.paths:
            try:
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    chunk = f.read(2_000_000)  # 最多读 2MB
                    sample_texts.append(chunk)
            except Exception:
                pass

    sample = "\n\n".join(sample_texts)
    if not sample.strip():
        print("  采样文本为空，无法训练分词器")
        return

    tokenizer = train_tokenizer(sample, args)
    build_and_save(sources, tokenizer, args)


def train_tokenizer(sample_text: str, args):
    """根据 args 选项训练分词器"""
    if args.tokenizer == 'bpe':
        print(f"  训练 BPE 分词器 (vocab_size={args.bpe_vocab})...")
        tokenizer = BPETokenizer.train(sample_text, vocab_size=args.bpe_vocab, verbose=False)
        print(f"  完成: vocab={tokenizer.vocab_size}, merges={len(tokenizer.merges)}")
    else:
        tokenizer = CharTokenizer.train(sample_text)
        print(f"  Char 分词器: vocab={tokenizer.vocab_size}")
    return tokenizer


def build_and_save(sources, tokenizer, args):
    """执行构建流程"""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = args.name
    out_bin = str(out_dir / f"{name}.bin")

    # 检查是否已存在
    if Path(out_bin).exists() and not args.force:
        print(f"\n  .bin 文件已存在: {out_bin}")
        print(f"  （使用 --force 重新构建）")
        return

    metadata = prepare_dataset(
        name=name,
        sources=sources,
        tokenizer=tokenizer,
        out_bin_path=out_bin,
        dtype=args.dtype,
        clean=not args.no_clean,
        save_meta=True,
    )

    # 保存 tokenizer 副本
    if args.tokenizer == 'bpe' and hasattr(tokenizer, 'save'):
        tok_path = out_dir / f"{name}_tokenizer.json"
        tokenizer.save(str(tok_path))
        print(f"  分词器已保存: {tok_path}")

    print(f"\n  ✓ 数据集 '{name}' 已构建完成")
    print(f"    .bin:  {out_bin}")
    print(f"    .meta: {Path(out_bin).with_suffix('.json')}")
    print(f"    Tokens: {metadata['total_tokens']:,}")
    print(f"    文件:  {metadata['file_size_bytes']:,} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="M6: 数据集构建工具 —— 从文本/JSONL/HuggingFace 构建 .bin token 数据集"
    )
    parser.add_argument('--name', type=str, default='dataset',
                        help='数据集名称（用于 .bin 和 .meta.json 文件名前缀）')
    parser.add_argument('--input', nargs='*', type=str, default=None,
                        help='本地文本文件路径（可多个，支持 glob）')
    parser.add_argument('--jsonl', nargs='*', type=str, default=None,
                        help='本地 JSONL 文件路径（可多个，支持 glob）')
    parser.add_argument('--jsonl-field', type=str, default='text',
                        help='JSONL 中 text 字段名')
    parser.add_argument('--hf', nargs='*', type=str, default=None,
                        help='HuggingFace 数据集名称（如 Skylion007/openwebtext）')
    parser.add_argument('--hf-field', type=str, default='text',
                        help='HF 数据集中 text 字段名')
    parser.add_argument('--hf-limit', type=int, default=None,
                        help='HF 数据集最大读取字符数')
    parser.add_argument('--source', action='append', type=str, default=None,
                        help='多源混合规格: "name:weight:paths[:type]"')
    parser.add_argument('--tokenizer', type=str, default='bpe',
                        choices=['char', 'bpe'], help='分词器类型')
    parser.add_argument('--bpe-vocab', type=int, default=300, help='BPE 词表大小')
    parser.add_argument('--dtype', type=str, default='uint16',
                        choices=['uint16', 'uint32'], help='token 二进制存储类型')
    parser.add_argument('--out-dir', type=str, default='data/bin',
                        help='输出目录（默认 data/bin/）')
    parser.add_argument('--no-clean', action='store_true', help='禁用文本清洗')
    parser.add_argument('--force', action='store_true', help='强制覆盖已有 .bin')
    parser.add_argument('--list', action='store_true', help='列出已知的示例数据集')

    args = parser.parse_args()
    setup_logger()

    if args.list:
        print("═" * 60)
        print(" M6 已知数据集预设")
        print("═" * 60)
        print()
        print("1. Shakespeare (当前项目已有)")
        print("   python scripts/prepare_data.py --name shakespeare --input data/input.txt")
        print()
        print("2. OpenWebText (HuggingFace, ~4GB text)")
        print("   python scripts/prepare_data.py --name openwebtext --hf Skylion007/openwebtext")
        print()
        print("3. C4 (HuggingFace, 英文, 约 800GB)")
        print("   python scripts/prepare_data.py --name c4 --hf allenai/c4 --hf-limit 100000000")
        print()
        print("4. 多源混合示例")
        print("   python scripts/prepare_data.py \\")
        print("       --name mixed \\")
        print("       --source shakespeare:1:data/input.txt \\")
        print("       --source openwebtext:2:data/owt/*.jsonl:jsonl")
        print()
        print("5. 从 data/ 目录自动发现")
        print("   python scripts/prepare_data.py --name auto")
        return

    sources = collect_sources(args)

    if not sources:
        # 尝试 auto-detect
        print("  未指定数据源，尝试从 data/ 目录自动发现...")
        auto_detect_and_build(args)
        return

    print(f"\n  汇集 {len(sources)} 个数据源:")
    for src in sources:
        print(f"    - {src.name}: {src.source_type}, weight={src.weight}, "
              f"paths={src.paths[:2]}{'...' if len(src.paths) > 2 else ''}")

    # 训练分词器：从每个源采样
    print(f"\n  采样训练 {args.tokenizer} 分词器...")
    sample_parts = []
    for src in sources:
        for p in src.paths[:2]:  # 每个源最多前2个文件
            if src.source_type == "hf":
                # HF 通过流式读一部分
                try:
                    from data.data_sources import _stream_hf
                    ds_src = DataSource(name=src.name, paths=[p],
                                        source_type="hf", text_field=src.text_field,
                                        max_chars=500_000)
                    for text in _stream_hf(ds_src):
                        sample_parts.append(text)
                        if sum(len(t) for t in sample_parts) > 1_000_000:
                            break
                except ImportError:
                    print(f"    HF datasets 未安装，跳过源: {src.name}")
                continue
            else:
                try:
                    with open(p, 'r', encoding='utf-8', errors='replace') as f:
                        sample_parts.append(f.read(500_000))
                except Exception:
                    pass
            if sum(len(t) for t in sample_parts) > 1_000_000:
                break
        if sum(len(t) for t in sample_parts) > 1_000_000:
            break

    sample = "\n\n".join(sample_parts)
    if not sample.strip():
        print("  ERROR: 采样文本为空，无法训练分词器")
        return

    tokenizer = train_tokenizer(sample, args)
    build_and_save(sources, tokenizer, args)


if __name__ == '__main__':
    main()
