"""
scripts/pretrain_v2.py — M6 大规模预训练（基于 .bin mmap 数据集）

相比 pretrain.py (M4) 的改进：
  - 支持 MmapTokenDataset（内存映射，O(1) 随机访问，RAM 占用极低）
  - 支持多 epoch 遍历（数据量大时单 epoch 步数不够）
  - 支持从 .bin + .meta.json 自动加载数据集配置
  - 支持混合数据集按比例随机采样
  - 训练进度基于 token 数（而非 step 数），更直观


用法:
  # 先用 prepare_data.py 构建 .bin，再用本脚本训练
  python scripts/prepare_data.py --name mixed --input data/input.txt --tokenizer bpe

  python scripts/pretrain_v2.py --bin data/bin/mixed.bin --epochs 3 --config-medium

  # 加载已有分词器继续训练
  python scripts/pretrain_v2.py --bin data/bin/mixed.bin \
      --tokenizer-bin data/bin/mixed_tokenizer.json --epochs 5

  # 极速验证（50 steps）
  python scripts/pretrain_v2.py --bin data/bin/mixed.bin --iters 50 --eval-interval 10
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch

from llm import ModelConfig, GPTLanguageModel, set_seed, get_batch
from tokenizer import BPETokenizer, CharTokenizer
from eval import compute_perplexity, compute_bpb, evaluate_model, evaluate_generation, save_eval_results
from train import Trainer, CheckpointManager, setup_logger
from data.data_sources import MmapTokenDataset, get_batch_mmap, load_dataset_meta

logger = logging.getLogger('nanoGPT')


def mmap_collate(batch):
    """MmapTokenDataset 返回 (x, y) tuples；这个 collate 将它们 stack 成 batch"""
    x_list, y_list = zip(*batch)
    return torch.stack(x_list), torch.stack(y_list)


def build_mmap_dataloader(dataset: MmapTokenDataset, batch_size: int,
                          num_workers: int = 0, shuffle: bool = True):
    """从 MmapTokenDataset 构建 DataLoader

    num_workers > 0 时，多个 worker 进程共享同一 mmap（只读共享内存，无拷贝开销）
    """
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mmap_collate if not shuffle else None,  # shuffle=True 时 DataLoader 自动 stack
        pin_memory=True,
        drop_last=True,
    )


class MmapTrainer(Trainer):
    """基于 MmapTokenDataset 的训练器

    覆盖 train() 以支持 DataLoader + 多 epoch。
    """

    def __init__(self, model, config, train_dataset: MmapTokenDataset,
                 val_dataset: MmapTokenDataset, decode_fn=None,
                 num_workers: int = 0):
        # 注意：Trainer.__init__ 需要 train_data/val_data 是 tensor，
        # 这里传递 dataset（通过 .dataset 属性兼容）
        # 实际训练中我们覆盖 train()，所以不调用 trainer 中原有的 train()
        self.model = model.to(config.device)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_data = None  # 不使用 tensor 路径
        self.val_data = None
        self.decode_fn = decode_fn
        self.num_workers = num_workers

        self.current_step = 0
        self.best_val_loss = float('inf')

        # 构建优化器
        self.optimizer = self._build_optimizer()

        # 混合精度
        self.scaler = None
        if config.use_amp and config.supports_bf16:
            self.scaler = torch.amp.GradScaler('cuda' if config.device == 'cuda' else 'cpu')

        # 日志
        self.logger = setup_logger('nanoGPT', log_file=str(Path(config.log_dir) / "train.log"))
        self.metrics_writer = __import__('train', fromlist=['MetricsWriter']).MetricsWriter(
            str(Path(config.log_dir) / "metrics.jsonl")
        )
        self.ckpt_manager = CheckpointManager(config.ckpt_dir)

        # autocast 上下文
        self.autocast_ctx = lambda: torch.amp.autocast(
            config.device,
            dtype=torch.bfloat16,
            enabled=config.autocast_enabled and config.use_amp
        )

    def _build_optimizer(self):
        """构建 AdamW 优化器（与 Trainer 相同的分组 weight decay）"""
        cfg = self.config
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'ln' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return torch.optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': cfg.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
        )

    def _get_lr(self, step):
        """warmup + cosine decay"""
        cfg = self.config
        warmup_iters = int(0.1 * cfg.max_iters)
        if step < warmup_iters:
            return cfg.learning_rate * step / max(warmup_iters, 1)
        decay_ratio = min((step - warmup_iters) / max(cfg.max_iters - warmup_iters, 1), 1.0)
        return max(cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_ratio)),
                   cfg.learning_rate * 0.01)

    @torch.no_grad()
    def estimate_loss(self, eval_iters=None):
        """估算训练集 and 验证集 loss，使用 mmap dataset 随机采样"""
        cfg = self.config
        eval_iters = eval_iters or cfg.eval_iters
        out = {}
        self.model.eval()

        for split, ds in [('train', self.train_dataset), ('val', self.val_dataset)]:
            losses = torch.zeros(eval_iters)
            batch_size = cfg.batch_size
            block_size = cfg.block_size
            device = cfg.device

            for k in range(eval_iters):
                num_samples = len(ds)
                ix = torch.randint(num_samples, (batch_size,))
                x_list, y_list = [], []
                for i in ix.tolist():
                    xi, yi = ds[i]
                    x_list.append(xi)
                    y_list.append(yi)
                X = torch.stack(x_list).to(device)
                Y = torch.stack(y_list).to(device)

                with self.autocast_ctx():
                    _, loss = self.model(X, Y)
                losses[k] = loss.item()

            out[split] = losses.mean().item()

        self.model.train()
        return out

    def train(self, start_step=0, num_workers=0):
        """训练主循环，使用 DataLoader 批量加载 mmap 数据"""
        cfg = self.config
        self.current_step = start_step
        device = cfg.device

        self.logger.info(f"开始训练: {cfg.max_iters:,} 步 (起始={start_step})")
        self.logger.info(f"模型: {self.model.count_params() / 1e6:.2f}M 参数")
        self.logger.info(f"训练集: {self.train_dataset}")
        self.logger.info(f"验证集: {self.val_dataset}")
        self.logger.info(f"精度: {'bf16 AMP' if self.scaler else 'fp32'}")
        self.logger.info(f"SDPA: {'ON' if cfg.use_sdpa else 'OFF'}")
        self.logger.info(f"num_workers: {num_workers}")
        self.logger.info("train() 即将开始...")
        sys.stdout.flush(); sys.stderr.flush()

        # 初始评估
        if start_step == 0:
            self.logger.info("开始初始评估 (estimate_loss)...")
            sys.stdout.flush(); sys.stderr.flush()
            losses = self.estimate_loss()
            self.logger.info(f"初始损失 - train: {losses['train']:.4f}, val: {losses['val']:.4f}")
            self.metrics_writer.write(0, {
                'train_loss': losses['train'], 'val_loss': losses['val'], 'lr': cfg.learning_rate
            })
            self.logger.info("初始评估完成")

        # 构建 DataLoader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        train_iter = iter(train_loader)
        total_tokens = 0
        t0 = time.time()

        for step in range(start_step, cfg.max_iters):
            self.current_step = step

            # LR 调度
            lr = self._get_lr(step)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            # 获取一个 batch（DataLoader 耗尽时重新 iter）
            try:
                X, Y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, Y = next(train_iter)
            X = X.to(device)
            Y = Y.to(device)

            # 训练 step
            loss_val = self._train_step(X, Y)
            total_tokens += X.numel()

            # 周期评估
            if step > 0 and step % cfg.eval_interval == 0:
                losses = self.estimate_loss()
                is_best = losses['val'] < self.best_val_loss
                self.best_val_loss = min(self.best_val_loss, losses['val'])

                elapsed = time.time() - t0
                t0 = time.time()
                speed = (cfg.eval_interval * cfg.batch_size * cfg.block_size) / max(elapsed, 0.001)

                self.logger.info(
                    f"step {step:6d}: train {losses['train']:.4f}, val {losses['val']:.4f} "
                    f"{'↓' if is_best else '='} | lr {lr:.2e} | {speed:.0f} tok/s | "
                    f"total {total_tokens:,} tok"
                )
                self.metrics_writer.write(step, {
                    'train_loss': losses['train'], 'val_loss': losses['val'],
                    'lr': lr, 'best_val_loss': self.best_val_loss, 'tok_per_sec': speed,
                })

                if is_best:
                    ckpt = self.ckpt_manager.save(
                        step, self.model, self.optimizer, self.best_val_loss, cfg, None
                    )
                    self.logger.info(f"  → 保存最佳: {ckpt.name}")

            if math.isnan(loss_val) or math.isinf(loss_val):
                self.logger.error(f"⚠ 异常 loss at step {step}: {loss_val}")
                break

        # 最终保存
        self.ckpt_manager.save(
            cfg.max_iters, self.model, self.optimizer, self.best_val_loss, cfg, None
        )
        self.metrics_writer.close()
        self.logger.info(f"训练完成！总处理 token: {total_tokens:,}")
        return self.model


def run_eval(model, tokenizer, val_dataset, config, args):
    """评估 + 生成演示"""
    model.eval()
    result = evaluate_model(model, val_dataset, batch_size=config.batch_size,
                           max_batches=50, device=config.device)
    logger.info(f"\n[评测结果]")
    logger.info(f"  Loss:         {result['loss']:.4f}")
    logger.info(f"  Perplexity:   {result['perplexity']:.2f}")
    logger.info(f"  Bits-per-byte:{result['bits_per_byte']:.4f}")
    logger.info(f"  Tokens/sec:   {result['tokens_per_sec']:.0f}")

    logger.info(f"\n[生成演示]")
    test_prompts = args.eval_prompts or ["The ", "To be, ", "Once upon ", "First "]
    for prompt in test_prompts:
        try:
            ids = tokenizer.encode(prompt)
        except Exception:
            continue
        input_t = torch.tensor([ids], dtype=torch.long, device=config.device)
        with torch.no_grad():
            output = model.generate(input_t, max_new_tokens=150, temperature=0.7, top_k=40)
        gen = tokenizer.decode(output[0].tolist()[len(ids):])
        logger.info(f"\n  >>> {repr(prompt)}")
        for line in gen[:200].split('\n')[:5]:
            logger.info(f"  {line}")

    model.train()
    # 保存eval 结果
    save_eval_results({'metrics': result}, str(Path(config.log_dir) / "eval_results.json"))


def main():
    parser = argparse.ArgumentParser(description="M6: 大规模预训练（mmap .bin）")
    parser.add_argument('--bin', type=str, default=None,
                        help='训练用 .bin 文件路径（如 data/bin/dataset.bin）')
    parser.add_argument('--val-bin', type=str, default=None,
                        help='验证用 .bin（可选，默认从训练集中按 val_ratio 切分）')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='验证集比例（当 --val-bin 未指定时）')
    parser.add_argument('--meta', type=str, default=None,
                        help='数据集 .meta.json 路径（可选，用于读取 tokenizer 信息）')
    parser.add_argument('--tokenizer-bin', type=str, default=None,
                        help='已保存的 BPE 分词器 JSON 路径')
    parser.add_argument('--tokenizer', type=str, default='bpe', choices=['char', 'bpe'])
    parser.add_argument('--bpe-vocab', type=int, default=500)
    parser.add_argument('--config-medium', action='store_true', help='使用 Config-M (~50M)')
    parser.add_argument('--epochs', type=int, default=None, help='训练 epoch 数')
    parser.add_argument('--iters', type=int, default=3000, help='总训练步数（默认覆盖）')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--block-size', type=int, default=None)
    parser.add_argument('--n-layer', type=int, default=None)
    parser.add_argument('--n-embd', type=int, default=None)
    parser.add_argument('--n-head', type=int, default=None)
    parser.add_argument('--n-kv-head', type=int, default=None)
    parser.add_argument('--eval-interval', type=int, default=None)
    parser.add_argument('--eval-iters', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-prompts', nargs='*', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default='runs/pretrain_v2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader worker 数（>0 使用多进程 mmap 加载）')
    args = parser.parse_args()

    # ── 配置 ──
    if args.config_medium:
        config_kwargs = dict(
            n_layer=10, n_embd=640, n_head=10, n_kv_head=5,
            block_size=256, batch_size=8,
            eval_interval=500, eval_iters=30, learning_rate=3e-4,
            use_sdpa=True, use_mixed_precision=not args.no_amp,
            weight_decay=0.1, grad_clip=1.0,
        )
    else:
        config_kwargs = dict(
            n_layer=6, n_embd=384, n_head=6, n_kv_head=2,
            block_size=128, batch_size=16,
            eval_interval=200, eval_iters=20, learning_rate=3e-4,
            use_sdpa=True, use_mixed_precision=not args.no_amp,
            weight_decay=0.1, grad_clip=1.0,
        )

    # CLI 命令行覆盖
    for attr in ['batch_size', 'learning_rate', 'block_size', 'n_layer', 'n_embd',
                 'n_head', 'n_kv_head', 'eval_interval', 'eval_iters']:
        val = getattr(args, attr)
        if val is not None:
            config_kwargs[attr] = val

    config = ModelConfig(**config_kwargs)
    config.max_iters = args.iters
    config.seed = args.seed
    config.ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
    config.log_dir = os.path.join(args.out_dir, 'logs')

    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, deterministic=True)

    # ── 加载数据集 ──
    if args.bin:
        # MmapTokenDataset 路径
        train_ds = MmapTokenDataset(args.bin, config.block_size)
        if args.val_bin:
            val_ds = MmapTokenDataset(args.val_bin, config.block_size)
        else:
            # 按比例从训练集末尾切分
            n_total = len(train_ds._mmap)
            n_val = int(n_total * args.val_ratio)
            n_train = n_total - n_val
            # 使用子集
            val_offset = n_train
            class _OffsetDataset(MmapTokenDataset):
                """子集包装：把偏移量注入"""
                def __init__(self, parent, offset, length):
                    self._mmap = parent._mmap
                    self.block_size = parent.block_size
                    self._dtype = parent._dtype
                    self._token_count = parent._token_count
                    self._offset = offset
                    self._sub_samples = max(0, length - parent.block_size)
                    self.path = parent.path  # 用于 __repr__

                def __len__(self):
                    return self._sub_samples

                def __repr__(self):
                    return (f"_OffsetDataset(path='{self.path.name}', "
                            f"offset={self._offset}, samples={self._sub_samples:,}, "
                            f"block_size={self.block_size})")

                def __getitem__(self, idx):
                    import numpy as np
                    real_idx = self._offset + idx
                    chunk = self._mmap[real_idx:real_idx + self.block_size + 1]
                    x = torch.from_numpy(chunk[:self.block_size].astype(np.int64))
                    y = torch.from_numpy(chunk[1:self.block_size + 1].astype(np.int64))
                    return x, y

            val_ds = _OffsetDataset(train_ds, val_offset, n_val)
            # 训练集限制为前 n_train 个
            train_ds = _OffsetDataset(train_ds, 0, n_train + 0)
            logger.info(f"训练集: {train_ds}")
            logger.info(f"验证集: {val_ds}")

        # tokenizer：从 .bin 伴随的 tokenizer 加载
        if args.tokenizer_bin:
            tokenizer = BPETokenizer.load(args.tokenizer_bin)
            logger.info(f"加载已有分词器: {args.tokenizer_bin}")
        else:
            # 从 meta.json 推断
            meta_path = Path(args.bin).with_suffix('.json')
            if meta_path.exists():
                meta = load_dataset_meta(str(meta_path))
                logger.info(f"数据集元信息: {meta['name']} | tokens={meta['total_tokens']:,}")
                # 尝试从 meta['bin_path'] 同目录加载 tokenizer
                tok_path = Path(args.bin).parent / f"{meta['name']}_tokenizer.json"
                if tok_path.exists():
                    tokenizer = BPETokenizer.load(str(tok_path))
                    logger.info(f"从元信息加载分词器: {tok_path}")
                else:
                    raise RuntimeError(f"找不到分词器文件: {tok_path}")
            else:
                raise RuntimeError(f"找不到数据集元信息: {meta_path}")

        config.vocab_size = tokenizer.vocab_size

    else:
        logger.error("请指定 --bin 路径指向 .bin 数据集文件")
        logger.error("先用 scripts/prepare_data.py 构建 .bin 文件")
        return

    # ── 构建模型 ──
    model = GPTLanguageModel(config)
    logger.info(f"\n{config.summary()}")
    logger.info(f"模型参数: {model.count_params() / 1e6:.2f}M")

    # ── 评测模式 ──
    if args.eval_only:
        load_path = args.resume
        if not load_path:
            latest = Path(config.ckpt_dir) / "latest.pt"
            if latest.exists():
                load_path = str(latest)
        if not load_path or not Path(load_path).exists():
            logger.error(f"未找到检查点用于评测: {load_path}")
            return
        logger.info(f"加载检查点: {load_path}")
        ckpt = torch.load(load_path, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        run_eval(model, tokenizer, val_ds, config, args)
        return

    # ── 构建 Trainer + 训练 ──
    trainer = MmapTrainer(model, config, train_ds, val_ds,
                          decode_fn=lambda ids: tokenizer.decode(ids))
    trainer.logger = logger

    # Resume
    start_step = 0
    if args.resume:
        ckpt = trainer.ckpt_manager.load(args.resume)
        model.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step']
        trainer.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        logger.info(f"恢复: step={start_step}, best_val={trainer.best_val_loss:.4f}")
        trainer.ckpt_manager.restore_rng(ckpt)

    # 按 epoch 自动算 iters
    if args.epochs:
        samples_per_epoch = len(train_ds)
        steps_per_epoch = samples_per_epoch // config.batch_size
        config.max_iters = steps_per_epoch * args.epochs
        logger.info(f"Epochs={args.epochs} → {config.max_iters:,} steps "
                     f"(每 epoch ~{steps_per_epoch:,} steps)")

    assert config.max_iters > 0, f"无效的 max_iters={config.max_iters}"

    trainer.train(start_step=start_step, num_workers=args.num_workers)

    # ── 最终评测 ──
    logger.info("\n" + "=" * 60)
    logger.info(" 训练完成！最终评测...")
    logger.info("=" * 60)
    run_eval(model, tokenizer, val_ds, config, args)
    logger.info("\n✓ M6 完成")


if __name__ == '__main__':
    main()
