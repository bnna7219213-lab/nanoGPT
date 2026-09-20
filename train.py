"""
train.py — nanoGPT M2 训练入口

包含：
- CheckpointManager: 检查点保存/加载
- Trainer: 增强训练器（混合精度 bf16, GradScaler, 日志系统）
- CLI: 命令行入口

M2 训练工程化特性：
  - bf16 混合精度（autocast + GradScaler）
  - 检查点保存/恢复（model + optimizer + iter + config）
  - 日志系统（console + file + metrics.jsonl）
  - 完整可复现 seed
  - 分块 CE（通过 config.ce_chunk_size 控制）
  - SDPA 加速（通过 config.use_sdpa 控制）

用法:
  python train.py --config mini --iters 2000
  python train.py --resume checkpoints/ckpt_step_1000.pt
  python train.py --config small --use-sdpa --use-amp --seed 123
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.amp import autocast, GradScaler

# 导入 llm 库层
from llm import (
    ModelConfig, GPTLanguageModel, set_seed,
    load_and_tokenize, get_batch, build_tokenizer, cross_entropy_loss
)
from tokenizer import BPETokenizer, CharTokenizer
from data.pipeline import clean_text, stream_jsonl, deduplicate, filter_by_length, build_datasets, get_batch_from_dataset
from eval import compute_perplexity, compute_bpb, evaluate_model, evaluate_generation, save_eval_results


# =====================================================================================
# 1. 日志系统
# =====================================================================================

def setup_logger(name: str, log_file: Optional[str] = None,
                 level=logging.INFO) -> logging.Logger:
    """配置日志记录器（console + optional file）

    Args:
        name: logger 名称
        log_file: 日志文件路径（None = 仅 console）
        level: 日志级别

    Returns:
        logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # 格式
    fmt = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class MetricsWriter:
    """写入训练指标到 JSONL 文件（便于后续分析/可视化）"""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, 'w', encoding='utf-8')

    def write(self, step: int, metrics: dict):
        """写入一行指标"""
        record = {'step': step, **metrics}
        self.f.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =====================================================================================
# 2. 检查点管理
# =====================================================================================

class CheckpointManager:
    """检查点管理器：保存和恢复训练状态

    检查点结构:
    {
        'step': 当前训练步数,
        'model_state_dict': 模型权重,
        'optimizer_state_dict': 优化器状态,
        'best_val_loss': 最佳验证损失,
        'config': 模型配置 dict,
        'rng_state': 随机数状态（用于可复现恢复）,
        'torch_rng_state': torch 随机数状态,
        'cuda_rng_states': cuda 随机数状态,
    }
    """

    def __init__(self, ckpt_dir: str = "checkpoints"):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, step: int, model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             best_val_loss: float, config: ModelConfig,
             rng_state: Optional[dict] = None):
        """保存检查点"""
        try:
            import random
            py_rng = random.getstate()
        except Exception:
            py_rng = None
        
        ckpt = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config.to_dict(),
            'rng_state': {
                'python': py_rng,
                'torch': torch.get_rng_state(),
                'numpy': self._get_numpy_state(),
            },
        }

        if torch.cuda.is_available():
            ckpt['rng_state']['cuda'] = torch.cuda.get_rng_state_all()

        path = self.ckpt_dir / f"ckpt_step_{step}.pt"
        # 临时文件写入 + 原子 rename
        tmp_path = path.with_suffix('.tmp')
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)

        # 同时保存 latest 软链接 / 副本
        latest_path = self.ckpt_dir / "latest.pt"
        if latest_path.exists():
            latest_path.unlink()
        # Windows 不支持 symlink（需要管理员权限），所以复制
        import shutil
        shutil.copy2(path, latest_path)

        return path

    @staticmethod
    def _get_numpy_state() -> Optional[list]:
        """获取 numpy 随机数状态（可选）"""
        try:
            import numpy as np
            return np.random.get_state()[1].tolist()
        except (ImportError, AttributeError):
            return None

    def load(self, path: str) -> dict:
        """加载检查点

        Returns:
            ckpt dict（包含所有恢复所需信息）
        """
        path = Path(path)
        if not path.exists():
            # 尝试 latest
            latest = self.ckpt_dir / "latest.pt"
            if latest.exists():
                path = latest
            else:
                raise FileNotFoundError(f"检查点不存在: {path}")

        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        return ckpt

    def restore_rng(self, ckpt: dict):
        """恢复随机数状态"""
        rng_state = ckpt.get('rng_state', {})
        # Python random
        if 'python' in rng_state:
            import random
            random.setstate(rng_state['python'])
        # torch
        if 'torch' in rng_state:
            torch.set_rng_state(rng_state['torch'])
        # numpy
        if 'numpy' in rng_state and rng_state['numpy'] is not None:
            try:
                import numpy as np
                # np.random.set_state 需要完整 tuple
                pass  # numpy 状态恢复复杂，暂略
            except ImportError:
                pass
        # cuda
        if 'cuda' in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state['cuda'])

    def list_checkpoints(self) -> list:
        """列出所有检查点"""
        ckpts = sorted(self.ckpt_dir.glob("ckpt_step_*.pt"))
        return [c.name for c in ckpts]


# =====================================================================================
# 3. Trainer — 增强训练器（M2）
# =====================================================================================

class Trainer:
    """增强训练器：混合精度 + 检查点 + 日志系统

    支持：
    - bf16 混合精度（autocast + GradScaler）
    - 检查点定期保存和恢复
    - 结构化日志（console + file + metrics JSONL）
    - 学习率 warmup + cosine decay
    - 梯度裁剪
    - 分组 weight decay
    - 完整可复现
    """

    def __init__(self, model: GPTLanguageModel, config: ModelConfig,
                 train_data: torch.Tensor, val_data: torch.Tensor,
                 decode_fn=None):
        self.model = model.to(config.device)
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.decode_fn = decode_fn

        # 训练状态
        self.current_step = 0
        self.best_val_loss = float('inf')

        # 构建优化器
        self.optimizer = self._build_optimizer()

        # 混合精度
        self.scaler = None
        if config.use_amp and config.supports_bf16:
            self.scaler = GradScaler(
                'cuda' if config.device == 'cuda' else 'cpu',
                enabled=config.grad_scaler_enabled
            )

        # 日志
        self.logger = setup_logger(
            'nanoGPT',
            log_file=str(Path(config.log_dir) / "train.log"),
            level=logging.INFO
        )
        self.metrics_writer = MetricsWriter(
            str(Path(config.log_dir) / "metrics.jsonl")
        )

        # 检查点管理器
        self.ckpt_manager = CheckpointManager(config.ckpt_dir)

        # 自动混合精度上下文
        self.autocast_ctx = lambda: autocast(
            config.device,
            dtype=torch.bfloat16,
            enabled=config.autocast_enabled and config.use_amp
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """构建 AdamW 优化器（分组 weight decay）"""
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

        param_groups = [
            {'params': decay_params, 'weight_decay': cfg.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
        )

    def _get_lr(self, step: int) -> float:
        """学习率调度：warmup + cosine decay"""
        cfg = self.config
        warmup_iters = int(0.1 * cfg.max_iters)

        if step < warmup_iters:
            return cfg.learning_rate * step / max(warmup_iters, 1)

        # cosine decay
        decay_ratio = (step - warmup_iters) / max(cfg.max_iters - warmup_iters, 1)
        decay_ratio = min(decay_ratio, 1.0)
        lr = cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return max(lr, cfg.learning_rate * 0.01)  # 最小 lr 截断

    @torch.no_grad()
    def estimate_loss(self) -> dict:
        """估算训练集和验证集损失"""
        cfg = self.config
        out = {}
        self.model.eval()

        for split in ['train', 'val']:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = get_batch(split, self.train_data, self.val_data, cfg)
                with self.autocast_ctx():
                    _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()

        self.model.train()
        return out

    def _train_step(self, xb: torch.Tensor, yb: torch.Tensor) -> float:
        """执行单步训练（支持混合精度）

        Returns:
            loss 值（float）
        """
        cfg = self.config

        if self.scaler is not None:
            # bf16 混合精度路径
            with self.autocast_ctx():
                logits, loss = self.model(xb, yb)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            # 梯度裁剪前 unscale
            if cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # fp32 路径
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.optimizer.step()

        return loss.item()

    def train(self, start_step: int = 0):
        """运行训练循环

        Args:
            start_step: 起始步数（用于恢复训练）
        """
        cfg = self.config
        self.current_step = start_step

        self.logger.info(f"开始训练: {cfg.max_iters} 步 (起始步={start_step})")
        self.logger.info(f"模型参数量: {self.model.count_params() / 1e6:.2f}M")
        self.logger.info(f"精度: {'bf16 AMP' if self.scaler else 'fp32'}, "
                         f"SDPA: {'ON' if cfg.use_sdpa else 'OFF'}, "
                         f"CE chunk: {cfg.ce_chunk_size or '无'}")

        # 初始评估
        if start_step == 0:
            losses = self.estimate_loss()
            self.logger.info(f"初始损失 - train: {losses['train']:.4f}, "
                             f"val: {losses['val']:.4f}")
            self.metrics_writer.write(0, {
                'train_loss': losses['train'],
                'val_loss': losses['val'],
                'lr': cfg.learning_rate,
            })

        # 训练循环
        t0 = time.time()
        for step in range(start_step, cfg.max_iters):
            self.current_step = step

            # 更新学习率
            lr = self._get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # 周期评估
            if step > 0 and step % cfg.eval_interval == 0:
                losses = self.estimate_loss()
                is_best = losses['val'] < self.best_val_loss
                self.best_val_loss = min(self.best_val_loss, losses['val'])

                elapsed = time.time() - t0
                t0 = time.time()
                speed = (cfg.eval_interval / max(elapsed, 0.001) *
                         cfg.batch_size * cfg.block_size)

                self.logger.info(
                    f"step {step:5d}: train {losses['train']:.4f}, "
                    f"val {losses['val']:.4f} {'↓' if is_best else '='} "
                    f"| lr {lr:.2e} | {speed:.0f} tok/s"
                )

                # 写 metrics
                self.metrics_writer.write(step, {
                    'train_loss': losses['train'],
                    'val_loss': losses['val'],
                    'lr': lr,
                    'best_val_loss': self.best_val_loss,
                    'tok_per_sec': speed,
                })

                # 保存检查点
                if is_best:
                    ckpt_path = self.ckpt_manager.save(
                        step, self.model, self.optimizer,
                        self.best_val_loss, cfg,
                        None  # rng_state 简化处理
                    )
                    self.logger.info(f"  → 保存最佳检查点: {ckpt_path.name}")

            # 定期保存（非最佳但周期性保存）
            if step > 0 and step % cfg.ckpt_interval == 0:
                ckpt_path = self.ckpt_manager.save(
                    step, self.model, self.optimizer,
                    self.best_val_loss, cfg, None
                )
                self.logger.info(f"  → 保存检查点: {ckpt_path.name}")

            # 前向 + 反向
            xb, yb = get_batch('train', self.train_data, self.val_data, cfg)
            loss_val = self._train_step(xb, yb)

            # 检查数值异常
            if math.isnan(loss_val) or math.isinf(loss_val):
                self.logger.error(f"⚠ 异常 loss at step {step}: {loss_val}")
                # 尝试恢复
                ckpts = self.ckpt_manager.list_checkpoints()
                if ckpts:
                    self.logger.info(f"建议恢复: train.py --resume "
                                     f"{cfg.ckpt_dir}/{ckpts[-1]}")
                break

        # 最终检查点
        final_path = self.ckpt_manager.save(
            cfg.max_iters, self.model, self.optimizer,
            self.best_val_loss, cfg, None
        )
        self.logger.info(f"训练完成！最终检查点: {final_path.name}")
        self.metrics_writer.close()

        return self.model

    def generate_sample(self, start_text: str = "The",
                        max_tokens: int = 500):
        """生成文本样本"""
        if not self.decode_fn:
            self.logger.warning("未提供 decode_fn，无法生成样本")
            return

        cfg = self.config
        encode_fn, _, _ = build_tokenizer(
            open(cfg.data_path, 'r', encoding='utf-8').read()
        )

        # 如果 start_text 中有未登录词，用其中存在的字符
        try:
            start_ids = torch.tensor([encode_fn(start_text)], dtype=torch.long,
                                     device=cfg.device)
        except KeyError:
            start_ids = torch.tensor([[0]], dtype=torch.long,
                                     device=cfg.device)

        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                start_ids, max_new_tokens=max_tokens,
                temperature=0.8, top_k=40
            )
        self.model.train()

        text = self.decode_fn(generated[0].tolist())
        self.logger.info(f"生成样本:\n{text}")


# =====================================================================================
# 4. CLI 入口
# =====================================================================================

def main():
    """M2 命令行入口"""
    parser = argparse.ArgumentParser(
        description="nanoGPT v0.5 (M2) — 训练工程化版"
    )
    # 配置
    parser.add_argument('--config', type=str, default='default',
                        choices=['tiny', 'mini', 'small', 'default'],
                        help='预设配置')
    parser.add_argument('--iters', type=int, default=None,
                        help='覆盖 max_iters')
    parser.add_argument('--data', type=str, default=None,
                        help='本地数据文件路径（跳过下载）')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--bs', '--batch-size', type=int, default=None,
                        dest='batch_size')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')

    # M2 工程化选项
    parser.add_argument('--use-sdpa', action='store_true', default=None,
                        help='启用 SDPA（默认自动）')
    parser.add_argument('--no-sdpa', action='store_true', default=False,
                        help='禁用 SDPA')
    parser.add_argument('--use-amp', action='store_true', default=None,
                        help='启用混合精度 bf16')
    parser.add_argument('--no-amp', action='store_true', default=False,
                        help='禁用混合精度')
    parser.add_argument('--ce-chunk', type=int, default=None,
                        dest='ce_chunk_size',
                        help='分块 CE 大小（token 数）')
    parser.add_argument('--grad-clip', type=float, default=None)

    # 消融
    parser.add_argument('--n-layer', type=int, default=None)
    parser.add_argument('--n-embd', type=int, default=None)
    parser.add_argument('--n-head', type=int, default=None)
    parser.add_argument('--block-size', type=int, default=None)

    # M3: 分词器和数据
    parser.add_argument('--tokenizer', type=str, default='char',
                        choices=['char', 'bpe'],
                        help='分词器类型 (char=字符级, bpe=BPE)')
    parser.add_argument('--bpe-vocab', type=int, default=500,
                        help='BPE 词表大小 (仅 BPE 分词器)')
    parser.add_argument('--bpe-save', type=str, default=None,
                        help='保存训练好的 BPE 分词器路径')
    parser.add_argument('--eval-only', action='store_true',
                        help='仅评测模式（不训练）')
    parser.add_argument('--eval-prompts', type=str, default=None,
                        help='评测 generation 的 prompt 文件路径')
    parser.add_argument('--compare-tokenizers', action='store_true',
                        help='对比字符级和 BPE 分词器')

    args = parser.parse_args()

    # ── 预设配置 ──
    presets = {
        'tiny': ModelConfig(
            n_layer=2, n_embd=64, n_head=4, block_size=32,
            batch_size=8, max_iters=500, eval_interval=100, eval_iters=20
        ),
        'mini': ModelConfig(
            n_layer=4, n_embd=128, n_head=4, block_size=64,
            batch_size=16, max_iters=2000, eval_interval=200, eval_iters=30
        ),
        'small': ModelConfig(
            n_layer=6, n_embd=384, n_head=6, n_kv_head=2,
            block_size=128, batch_size=8, max_iters=5000,
            eval_interval=500, eval_iters=30
        ),
    }

    if args.config == 'default':
        config = ModelConfig()
    else:
        config = presets[args.config]

    # ── CLI 覆盖 ──
    if args.iters is not None:
        config.max_iters = args.iters
    if args.data is not None:
        config.data_path = args.data
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.device is not None:
        config.device = args.device
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.seed is not None:
        config.seed = args.seed
        set_seed(config.seed, config.deterministic)
    if args.n_layer is not None:
        config.n_layer = args.n_layer
    if args.n_embd is not None:
        config.n_embd = args.n_embd
    if args.n_head is not None:
        config.n_head = args.n_head
    if args.block_size is not None:
        config.block_size = args.block_size
    if args.grad_clip is not None:
        config.grad_clip = args.grad_clip
    if args.ce_chunk_size is not None:
        config.ce_chunk_size = args.ce_chunk_size
    if args.no_sdpa:
        config.use_sdpa = False
    if args.no_amp:
        config.use_mixed_precision = False
        config.autocast_enabled = False

    # ── 创建目录 ──
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    # ── 保存配置 ──
    config.save(str(Path(config.ckpt_dir) / "config.json"))

    # ── 日志 ──
    logger = setup_logger(
        'nanoGPT',
        log_file=str(Path(config.log_dir) / "train.log")
    )
    logger.info(f"{'='*60}")
    logger.info(f" nanoGPT v0.5 (M3 训练工程化)")
    logger.info(f"{'='*60}")
    logger.info(f"\n{config.summary()}")

    # ── 打印混合精度状态 ──
    amp_status = "bf16 (启用)" if config.use_amp else "fp32 (纯)"
    if config.use_amp and not config.supports_bf16:
        amp_status = "fp32 (GPU 不支持 bf16)"
        config.use_mixed_precision = False
        config.autocast_enabled = False
    logger.info(f"精度: {amp_status}")
    logger.info(f"SDPA: {'ON' if config.use_sdpa else 'OFF'}")
    logger.info(f"分词器: {args.tokenizer}")

    # ── 加载原始文本 ──
    logger.info(f"加载数据: {config.data_path}")
    with open(config.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    logger.info(f"原始文本: {len(text):,} 字符")

    # ── 训练/加载分词器 ──
    if args.tokenizer == 'bpe':
        logger.info(f"训练 BPE 分词器 (vocab_size={args.bpe_vocab})...")
        tokenizer = BPETokenizer.train(text, vocab_size=args.bpe_vocab)
        logger.info(f"BPE 分词器: {tokenizer}")
        if args.bpe_save:
            tokenizer.save(args.bpe_save)
            logger.info(f"  → 保存分词器: {args.bpe_save}")
    else:
        logger.info("使用字符级分词器...")
        tokenizer = CharTokenizer.train(text)
        logger.info(f"字符级分词器: {tokenizer}")

    config.vocab_size = tokenizer.vocab_size

    # ── 构建数据集 ──
    # 文本已经分词，按 90/10 划分
    logger.info("分词并构建数据集...")
    # 将所有文档连成一条 token 序列
    all_token_ids = tokenizer.encode(text)
    
    # 划分训练/验证
    split = int(0.9 * len(all_token_ids))
    train_token_ids = all_token_ids[:split]
    val_token_ids = all_token_ids[split:]
    
    # 转为 Tensor
    train_data = torch.tensor(train_token_ids, dtype=torch.long, device=config.device)
    val_data = torch.tensor(val_token_ids, dtype=torch.long, device=config.device)
    
    logger.info(f"训练 token: {len(train_token_ids):,}, 验证 token: {len(val_token_ids):,}")
    
    # 构建索引化的验证集用于评测
    from data.pipeline import TokenizedDataset
    val_dataset = TokenizedDataset(val_token_ids, config.block_size, name="val")

    # ── 构建模型 ──
    model = GPTLanguageModel(config)
    logger.info(f"实际参数量: {model.count_params() / 1e6:.2f}M")

    # ── 评测模式 ──
    if args.eval_only:
        logger.info(f"\n{'='*60}")
        logger.info(" 评测模式")
        logger.info(f"{'='*60}")
        
        result = evaluate_model(model, val_dataset, batch_size=config.batch_size, 
                               max_batches=20, device=config.device)
        logger.info(f"评测结果: loss={result['loss']:.4f}, "
                    f"ppl={result['perplexity']:.2f}, "
                    f"bpb={result['bits_per_byte']:.4f}")
        
        # 生成测试
        prompts = ["The ", "To be or", "All that"]
        gen_result = evaluate_generation(model, tokenizer, prompts, 
                                         max_new_tokens=50, device=config.device)
        logger.info(f"生成多样性: {gen_result['avg_unique_token_ratio']:.4f}")
        for g in gen_result['generations']:
            logger.info(f"  '{g['prompt']}' → '{g['generated'][:60]}'")
        
        # 保存评测
        eval_path = str(Path(config.log_dir) / "eval_results.json")
        save_eval_results({'metrics': result, 'generation': gen_result}, eval_path)
        return

    # ── 构建 Trainer ──
    trainer = Trainer(model, config, train_data, val_data, decode_fn=lambda ids: tokenizer.decode(ids))
    trainer.logger = logger  # 统一 logger

    # ── 恢复训练 ──
    start_step = 0
    if args.resume:
        ckpt = trainer.ckpt_manager.load(args.resume)
        model.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step']
        trainer.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        logger.info(f"从检查点恢复: step={start_step}, "
                    f"best_val_loss={trainer.best_val_loss:.4f}")
        trainer.ckpt_manager.restore_rng(ckpt)

    # ── 训练 ──
    trainer.train(start_step=start_step)

    # ── 生成示例 ──
    logger.info(f"\n{'='*60}")
    logger.info(" 生成文本示例")
    logger.info(f"{'='*60}")
    trainer.generate_sample(start_text="The ", max_tokens=200)


if __name__ == '__main__':
    main()
