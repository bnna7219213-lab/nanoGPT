"""
pretrain.py — M4 能力基线：首次完整预训练

在 Shakespeare 语料上训练 Config-M（~50M 参数），支持多 epoch、自动评测、模型保存。

用法:
    python scripts/pretrain.py --epochs 3 --tokenizer bpe --config-medium
    python scripts/pretrain.py --resume runs/baseline/checkpoints/latest.pt --eval-only
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
from pathlib import Path

import torch

from llm import ModelConfig, GPTLanguageModel, set_seed
from tokenizer import BPETokenizer, CharTokenizer
from eval import compute_perplexity, compute_bpb, evaluate_model, evaluate_generation, save_eval_results
from train import Trainer, CheckpointManager, setup_logger
from data import TokenizedDataset


def main():
    parser = argparse.ArgumentParser(description="M4: 能力基线预训练")
    parser.add_argument('--data', type=str, default='data/input.txt')
    parser.add_argument('--tokenizer', type=str, default='bpe', choices=['char', 'bpe'])
    parser.add_argument('--bpe-vocab', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练 epoch 数 (自动计算 iters)')
    parser.add_argument('--iters', type=int, default=3000, help='总训练步数')
    parser.add_argument('--config-medium', action='store_true',
                        help='使用 Config-M (~50M)')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--eval-only', action='store_true', help='仅评测已保存的模型')
    parser.add_argument('--out-dir', type=str, default='runs/baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', action='store_true', help='禁用混合精度')
    args = parser.parse_args()

    # ══════════════════════════════════════════════════════════════
    # 配置
    # ══════════════════════════════════════════════════════════════
    if args.config_medium:
        config = ModelConfig(
            n_layer=10, n_embd=640, n_head=10, n_kv_head=5,
            block_size=256, batch_size=8, max_iters=args.iters,
            eval_interval=500, learning_rate=3e-4,
            use_sdpa=True, use_mixed_precision=not args.no_amp,
            weight_decay=0.1, grad_clip=1.0,
            ckpt_dir=os.path.join(args.out_dir, 'checkpoints'),
            log_dir=os.path.join(args.out_dir, 'logs'),
            seed=args.seed,
        )
    else:
        config = ModelConfig(
            n_layer=6, n_embd=384, n_head=6, n_kv_head=2,
            block_size=128, batch_size=16, max_iters=args.iters,
            eval_interval=200, learning_rate=3e-4,
            use_sdpa=True, use_mixed_precision=not args.no_amp,
            weight_decay=0.1, grad_clip=1.0,
            ckpt_dir=os.path.join(args.out_dir, 'checkpoints'),
            log_dir=os.path.join(args.out_dir, 'logs'),
            seed=args.seed,
        )

    # 日志
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger('nanoGPT', log_file=str(Path(config.log_dir) / "pretrain.log"))

    logger.info("=" * 60)
    logger.info(" M4 能力基线: 首次完整预训练")
    logger.info("=" * 60)
    logger.info(f"\n{config.summary()}")
    logger.info(f"分词器: {args.tokenizer}")

    # ══════════════════════════════════════════════════════════════
    # 数据加载 + 分词
    # ══════════════════════════════════════════════════════════════
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()
    logger.info(f"语料: {len(text):,} 字符, {len(text.encode('utf-8')):,} 字节")

    if args.tokenizer == 'bpe':
        tokenizer = BPETokenizer.train(text, vocab_size=args.bpe_vocab, verbose=False)
        tokenizer.save(os.path.join(config.ckpt_dir, 'tokenizer.json'))
        logger.info(f"BPE 分词器: vocab={tokenizer.vocab_size}, merges={len(tokenizer.merges)}")
    else:
        tokenizer = CharTokenizer.train(text)
        logger.info(f"字符级分词器: vocab={tokenizer.vocab_size}")

    config.vocab_size = tokenizer.vocab_size

    # 分词 → 数据集
    token_ids = tokenizer.encode(text)
    split = int(0.9 * len(token_ids))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]

    logger.info(f"训练 token: {len(train_ids):,}, 验证 token: {len(val_ids):,}")

    # 按 epoch 自动计算 iters
    steps_per_epoch = len(train_ids) // (config.batch_size * config.block_size)
    if args.epochs:
        config.max_iters = steps_per_epoch * args.epochs
        logger.info(f"Epochs={args.epochs} → {config.max_iters:,} steps "
                     f"(每 epoch ~{steps_per_epoch:,} steps)")
    logger.info(f"实际训练: {config.max_iters:,} steps")

    # 构建 Tensor
    train_data = torch.tensor(train_ids, dtype=torch.long, device=config.device)
    val_data = torch.tensor(val_ids, dtype=torch.long, device=config.device)
    val_dataset = TokenizedDataset(val_ids, config.block_size, name="val")

    # ══════════════════════════════════════════════════════════════
    # 构建模型
    # ══════════════════════════════════════════════════════════════
    model = GPTLanguageModel(config)
    params_M = model.count_params() / 1e6
    logger.info(f"模型参数: {params_M:.2f}M")

    # ────────────────────────────────────────────────────────
    # 评测模式 (eval-only)
    # ────────────────────────────────────────────────────────
    if args.eval_only:
        logger.info("\n" + "=" * 20 + " 评测模式 " + "=" * 20)
        # 加载模型
        load_path = args.resume
        if not load_path:
            # 查找最新 ckpt
            ckpt_dir = Path(config.ckpt_dir)
            latest = ckpt_dir / "latest.pt"
            if latest.exists():
                load_path = str(latest)
        if not load_path or not Path(load_path).exists():
            logger.error(f"没有找到检查点用于评测: {load_path}")
            return

        logger.info(f"加载检查点: {load_path}")
        ckpt = torch.load(load_path, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        run_eval(model, tokenizer, val_ids, config, logger, args)
        return

    # ══════════════════════════════════════════════════════════════
    # Trainer 构建
    # ══════════════════════════════════════════════════════════════
    trainer = Trainer(model, config, train_data, val_data,
                      decode_fn=lambda ids: tokenizer.decode(ids))
    trainer.logger = logger

    # Resume
    start_step = 0
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        ckpt_mgr = CheckpointManager(config.ckpt_dir)
        ckpt = ckpt_mgr.load(args.resume)
        model.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step']
        trainer.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        logger.info(f"  → step={start_step}, best_val_loss={trainer.best_val_loss:.4f}")
        ckpt_mgr.restore_rng(ckpt)

    # ══════════════════════════════════════════════════════════════
    # 训练循环（重写进度显示）
    # ══════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info(f" 训练开始: {config.max_iters:,} 步")
    logger.info(f"{'='*60}\n")

    t_start = time.time()
    ckpt_mgr = CheckpointManager(config.ckpt_dir)

    model.train()
    for step in range(start_step, config.max_iters):
        trainer.current_step = step

        # ETA
        if step > start_step and step % 100 == 0:
            elapsed = time.time() - t_start
            eta = elapsed * (config.max_iters - step) / (step - start_step)
            eta_str = f"{int(eta//3600)}h{int((eta%3600)//60)}m{int(eta%60)}s"
            elapsed_str = f"{int(elapsed//3600)}h{int((elapsed%3600)//60)}m"
        elif step == start_step:
            eta_str, elapsed_str = "N/A", "0m"
            t_start = time.time()

        # LR 调度
        lr = trainer._get_lr(step)
        for pg in trainer.optimizer.param_groups:
            pg['lr'] = lr

        # ── 周期评估 ──
        if step % config.eval_interval == 0:
            losses = trainer.estimate_loss()
            is_best = losses['val'] < trainer.best_val_loss
            trainer.best_val_loss = min(trainer.best_val_loss, losses['val'])

            elapsed_total = time.time() - t_start + (time.time() - t_start) * 0
            total_tokens = step * config.batch_size * config.block_size
            mau = torch.cuda.max_memory_allocated() / 1024**2 if config.device == 'cuda' else 0
            ppl = compute_perplexity(losses['val'])

            logger.info(
                f"[{step:6d}/{config.max_iters}]  "
                f"train={losses['train']:.4f}  val={losses['val']:.4f}  ppl={ppl:.2f}  "
                f"{'↓' if is_best else '='}  "
                f"lr={lr:.2e}  tok={total_tokens:,}  "
                f"mem={mau:.0f}MB"
            )

            trainer.metrics_writer.write(step, {
                'train_loss': losses['train'],
                'val_loss': losses['val'],
                'lr': lr,
                'best_val_loss': trainer.best_val_loss,
            })

            if is_best:
                ckpt_mgr.save(step, model, trainer.optimizer, trainer.best_val_loss, config)
                logger.info(f"  → 最佳模型已保存 (ckpt_step_{step}.pt)")

            if step > 0 and step % 1000 == 0:
                ckpt_mgr.save(step, model, trainer.optimizer, trainer.best_val_loss, config)

        # ── 训练 step ──
        from llm import get_batch
        xb, yb = get_batch('train', trainer.train_data, trainer.val_data, config)
        loss_val = trainer._train_step(xb, yb)

        if step % 500 == 0 and step > start_step:
            logger.info(f"[{step:6d}/{config.max_iters}]  loss={loss_val:.4f}  "
                         f"eta={eta_str}  elapsed={elapsed_str}")

        import math
        if math.isnan(loss_val) or math.isinf(loss_val):
            logger.error(f"!! Loss divergence at step {step}: {loss_val}")
            break

    # ══════════════════════════════════════════════════════════════
    # 保存最终 + 评测
    # ══════════════════════════════════════════════════════════════
    final_step = min(step + 1 if 'step' in dir() else config.max_iters, config.max_iters)
    ckpt_mgr.save(final_step, model, trainer.optimizer, trainer.best_val_loss, config)
    trainer.metrics_writer.close()

    logger.info(f"\n{'='*60}")
    logger.info(" 训练完成！最终评测...")
    logger.info(f"{'='*60}")

    run_eval(model, tokenizer, val_ids, config, logger, args)
    logger.info("\n✓ M4 完成")


def run_eval(model, tokenizer, val_ids, config, logger, args):
    """运行评估 + 生成演示"""
    model.eval()
    val_ds = TokenizedDataset(val_ids, config.block_size, name="val")

    # 主要评测
    result = evaluate_model(model, val_ds, batch_size=config.batch_size,
                           max_batches=50, device=config.device)
    logger.info(f"\n[评测结果]")
    logger.info(f"  Loss:         {result['loss']:.4f}")
    logger.info(f"  Perplexity:   {result['perplexity']:.2f}")
    logger.info(f"  Bits-per-byte:{result['bits_per_byte']:.4f}")
    logger.info(f"  Tokens/sec:   {result['tokens_per_sec']:.0f}")

    # 生成演示
    logger.info(f"\n[生成演示]")
    if args.tokenizer == 'bpe':
        test_prompts = ["ROMEO:\n", "First Citizen:\n", "The ", "To be, "]
    else:
        test_prompts = ["ROMEO", "The ", "First"]

    for prompt in test_prompts:
        try:
            input_ids = tokenizer.encode(prompt)
        except Exception:
            continue
        input_t = torch.tensor([input_ids], dtype=torch.long, device=config.device)
        with torch.no_grad():
            output = model.generate(input_t, max_new_tokens=150,
                                     temperature=0.7, top_k=40)
        generated = tokenizer.decode(output[0].tolist()[len(input_ids):])
        logger.info(f"\n  >>> {repr(prompt)}")
        for line in generated[:200].split('\n')[:5]:
            logger.info(f"  {line}")

    model.train()

    # 保存结果
    results_path = str(Path(config.log_dir) / "eval_results.json")
    save_eval_results({'metrics': result}, results_path)
    logger.info(f"\n评测结果已保存: {results_path}")


if __name__ == '__main__':
    main()
