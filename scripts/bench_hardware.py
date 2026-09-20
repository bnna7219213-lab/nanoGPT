"""Hardware Benchmark — 测量 RTX 4050 的 tok/s 和显存峰值

用法:
    python scripts/bench_hardware.py --duration 30

输出: JSON 格式的硬件基线数据
"""
import argparse
import json
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from llm import ModelConfig, GPTLanguageModel, get_batch


def benchmark_model(config: ModelConfig, device: str, num_batches: int = 50,
                    warmup: int = 5):
    """基准测试单个配置"""
    model = GPTLanguageModel(config).to(device)
    model.train()

    # 模拟数据
    data_size = 10000
    dummy_data = torch.randint(0, config.vocab_size, (data_size,))
    n_train = int(0.9 * len(dummy_data))
    train_data = dummy_data[:n_train].clone()
    val_data = dummy_data[n_train:].clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Warmup
    for _ in range(warmup):
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.block_size), device=device)
        y = torch.randint(0, config.vocab_size, (config.batch_size, config.block_size), device=device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # CUDA 同步
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    start_time = time.time()
    for _ in range(num_batches):
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.block_size), device=device)
        y = torch.randint(0, config.vocab_size, (config.batch_size, config.block_size), device=device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start_time

    # 统计
    total_tokens = config.batch_size * config.block_size * num_batches
    tokens_per_sec = total_tokens / elapsed

    result = {
        'config': config.to_dict(),
        'num_params': model.count_params(),
        'num_batches': num_batches,
        'elapsed_sec': round(elapsed, 3),
        'tokens_per_sec': round(tokens_per_sec, 1),
        'ms_per_batch': round(elapsed / num_batches * 1000, 1),
    }

    if device == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        result['peak_memory_mb'] = round(peak_mem, 1)

    del model, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="nanoGPT 硬件基准测试")
    parser.add_argument('--duration', type=int, default=30,
                        help='单配置测试时长（秒），0 = 固定 batch 数')
    parser.add_argument('--output', type=str, default='benchmark_results.json')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    # 待测配置
    configs = {
        'tiny': ModelConfig(
            vocab_size=65, n_layer=2, n_embd=64, n_head=4,
            block_size=32, batch_size=8, device=device
        ),
        'mini': ModelConfig(
            vocab_size=65, n_layer=4, n_embd=128, n_head=4,
            block_size=64, batch_size=16, device=device
        ),
        'Config-S': ModelConfig(
            vocab_size=65, n_layer=6, n_embd=384, n_head=6,
            n_kv_head=2, block_size=128, batch_size=8, device=device
        ),
        'Config-M': ModelConfig(
            vocab_size=65, n_layer=8, n_embd=640, n_head=10,
            n_kv_head=2, block_size=128, batch_size=4, device=device
        ),
    }

    results = []
    for name, cfg in configs.items():
        params = cfg.estimate_params()
        params_mb = params * 4 / 1024**2  # fp32 bytes -> MB
        params_mb_fp16 = params * 2 / 1024**2
        print(f"\n  [{name}] 估计参数量: {params/1e6:.1f}M "
              f"(fp32={params_mb:.0f}MB, fp16={params_mb_fp16:.0f}MB)")

        try:
            result = benchmark_model(cfg, device, num_batches=max(10, args.duration * 5))
            result['name'] = name
            result['est_params_M'] = round(params / 1e6, 2)
            results.append(result)
            print(f"    tok/s: {result['tokens_per_sec']:.1f}, "
                  f"ms/batch: {result['ms_per_batch']:.1f}, "
                  f"peak_mem: {result.get('peak_memory_mb', 'N/A')} MB")
        except RuntimeError as e:
            print(f"    ❌ OOM 或其他错误: {str(e)[:80]}")
            results.append({
                'name': name,
                'error': str(e)[:200],
                'est_params_M': round(params / 1e6, 2),
            })

    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), '..', args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")

    # 总结
    print("\n" + "=" * 60)
    print(" 硬件基线总结")
    print("=" * 60)
    for r in results:
        if 'error' in r:
            print(f"  {r['name']:12s}: ❌ {r['error'][:50]}")
        else:
            print(f"  {r['name']:12s}: {r['tokens_per_sec']:8.1f} tok/s, "
                  f"peak_mem={r.get('peak_memory_mb', 'N/A')}MB")


if __name__ == '__main__':
    main()
