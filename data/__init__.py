"""
data — nanoGPT 数据管线 package

M3 层（基础）: data/pipeline.py
  - clean_text, stream_jsonl, deduplicate, filter_by_length
  - TokenizedDataset, build_datasets, get_batch_from_dataset

M6 层（扩展）: data/data_sources.py
  - DataSource, MultiSourceStreamer
  - TokenBinWriter, MmapTokenDataset
  - prepare_dataset
"""
# 导出 M3 层 (pipeline.py) 的公共接口（相对导入）
from .pipeline import (
    clean_text,
    stream_jsonl,
    deduplicate,
    filter_by_length,
    TokenizedDataset,
    build_datasets,
    get_batch_from_dataset,
)

# 导出 M6 层 (data_sources.py) 的公共接口（相对导入）
from .data_sources import (
    DataSource,
    MmapTokenDataset,
    TokenBinWriter,
    multi_source_stream,
    prepare_dataset,
    stream_single_source,
    load_dataset_meta,
    get_batch_mmap,
)
