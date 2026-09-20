"""
ablate.py — M5 消融实验

实验矩阵（控制变量法，每次只变一个超参）:
  A. 分词器:  BPE(vocab=300) vs Character
  B. 注意力:  GQA(n_kv=2) vs MHA(n_kv=6)
  C. 深度:    n_layer=4 vs 6 vs 8
  D. 宽度:    n_embd=256 vs 384 vs 512
  E. 学习率:  1e-4 vs 3e-4 vs 1e-3

每组实验固定其他参数，训练相同步数，比较 final val loss / ppl / bpb。
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from llm import ModelConfig, GPTLanguageModel, set_seed
from tokenizer import BPETokenizer, CharTokenizer
from data.pipeline import TokenizedDataset
from eval import compute_perplexity, compute_bpb, evaluate_model, save_eval_results
from train import Trainer, setup_logger
from llm import get_batch


def run_experiment(name, config, train_ids, val_ids, tokenizer, device, decode_fn, log_dir, steps=800):
    """运行单次实验并返回结果"""
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(f'ablate_{name}', log_file=os.path.join(log_dir, f'{name}.log'))

    config.vocab_size = tokenizer.vocab_size
    config.max_iters = steps

    model = GPTLanguageModel(config)
    params_M = model.count_params() / 1e6

    logger.info(f"[{name}] 开始: {params_M:.2f}M 参数, {steps} steps")

    train_data = torch.tensor(train_ids, dtype=torch.long, device=device)
    val_data = torch.tensor(val_ids, dtype=torch.long, device=device)
    val_ds = TokenizedDataset(val_ids, config.block_size, name="val")

    trainer = Trainer(model, config, train_data, val_data, decode_fn=decode_fn)
    trainer.logger = logger

    t0 = time.time()
    model.train()
    eval_interval = max(steps // 5, 50)  # 评估 5 次

    for step in range(steps):
        lr = trainer._get_lr(step)
        for pg in trainer.optimizer.param_groups:
            pg['lr'] = lr

        if step % eval_interval == 0:
            losses = trainer.estimate_loss()
            trainer.best_val_loss = min(trainer.best_val_loss, losses['val'])
            logger.info(f"  step {step:4d}: train={losses['train']:.4f}, val={losses['val']:.4f}")

        xb, yb = get_batch('train', train_data, val_data, config)
        loss_val = trainer._train_step(xb, yb)

    elapsed = time.time() - t0
    final_losses = trainer.estimate_loss()
    ppl = compute_perplexity(final_losses['val'])
    bpb = compute_bpb(final_losses['val'])

    result = {
        'name': name,
        'params_M': round(params_M, 2),
        'train_loss': round(final_losses['train'], 4),
        'val_loss': round(final_losses['val'], 4),
        'best_val_loss': round(trainer.best_val_loss, 4),
        'ppl': round(ppl, 2),
        'bpb': round(bpb, 4),
        'elapsed_sec': round(elapsed, 1),
        'steps': steps,
        'config': {
            'n_layer': config.n_layer,
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_kv_head': config.n_kv_head,
            'block_size': config.block_size,
            'batch_size': config.batch_size,
            'tokenizer': tokenizer.__class__.__name__,
            'vocab_size': tokenizer.vocab_size,
            'learning_rate': config.learning_rate,
        }
    }

    logger.info(f"[{name}] 完成: val_loss={result['val_loss']:.4f}, ppl={result['ppl']:.2f}, bpb={result['bpb']:.4f}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M5: 消融实验")
    parser.add_argument('--data', type=str, default='data/input.txt')
    parser.add_argument('--out-dir', type=str, default='runs/ablations')
    parser.add_argument('--steps', type=int, default=800, help='每组实验步数')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiments', type=str, default='all',
                        help='要运行的实验: tokenizer,gqa,depth,width,lr 或 all')
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    # 加载语料
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()

    # 训练分词器
    bpe_tok = BPETokenizer.train(text, vocab_size=300)
    char_tok = CharTokenizer.train(text)

    # 预计算各分词器的 token 序列
    bpe_ids = bpe_tok.encode(text)
    char_ids = char_tok.encode(text)

    experiments = []

    # ── A. 分词器消融 ──
    if args.experiments in ['all', 'tokenizer']:
        # Char
        cfg = ModelConfig(
            n_layer=6, n_embd=256, n_head=4, n_kv_head=2,
            block_size=128, batch_size=16, max_iters=args.steps,
            eval_interval=200, learning_rate=3e-4,
            use_sdpa=True, use_mixed_precision=True,
            ckpt_dir=os.path.join(args.out_dir, 'checkpoints'),
            log_dir=os.path.join(args.out_dir, 'logs'),
        )
        split = int(0.9 * len(char_ids))
        experiments.append({
            'name': 'tokenizer_char',
            'config': cfg,
            'train_ids': char_ids[:split], 'val_ids': char_ids[split:],
            'tokenizer': char_tok, 'decode_fn': lambda ids: char_tok.decode(ids),
        })
        # BPE
        cfg2 = ModelConfig(
            n_layer=6, n_embd=256, n_head=4, n_kv_head=2,
            block_size=128, batch_size=16, max_iters=args.steps,
            eval_interval=200, learning_rate=3e-4,
            use_sdpa=True, use_mixed_precision=True,
            ckpt_dir=os.path.join(args.out_dir, 'checkpoints'),
            log_dir=os.path.join(args.out_dir, 'logs'),
        )
        split = int(0.9 * len(bpe_ids))
        experiments.append({
            'name': 'tokenizer_bpe300',
            'config': cfg2,
            'train_ids': bpe_ids[:split], 'val_ids': bpe_ids[split:],
            'tokenizer': bpe_tok, 'decode_fn': lambda ids: bpe_tok.decode(ids),
        })

    # ── B. GQA vs MHA ──
    if args.experiments in ['all', 'gqa']:
        for kv_name, n_kv in [('MHA', 4), ('GQA', 2)]:
            cfg = ModelConfig(
                n_layer=6, n_embd=256, n_head=4, n_kv_head=n_kv,
                block_size=128, batch_size=16, max_iters=args.steps,
                eval_interval=200, learning_rate=3e-4,
                use_sdpa=True, use_mixed_precision=True,
                ckpt_dir=os.path.join(args.out_dir, 'checkpoints'),
                log_dir=os.path.join(args.out_dir, 'logs'),
            )
            split = int(0.9 * len(bpe_ids))
            experiments.append({
                'name': f'gqa_{kv_name}',
                'config': cfg,
                'train_ids': bpe_ids[:split], 'val_ids': bpe_ids[split:],
                'tokenizer': bpe_tok, 'decode_fn': lambda ids: bpe_tok.decode(ids),
            })

    # ── C. 深度消融 ──
    if args.experiments in ['all', 'depth']:
        for n_l in [4, 6, 8]:
            cfg = ModelConfig(
                n_layer=n_l, n_embd=256, n_head=4, n_kv_head=2,
                block_size=128, batch_size=16, max_iters=args.steps,
                eval_interval=200, learning_rate=3e-4,
                use_sdpa=True, use_mixed_precision=True,
                ckpt_dir=os.path.join(args.out_dir, 'checkpoints'),
                log_dir=os.path.join(args.out_dir, 'logs'),
            )
            split = int(0.9 * len(bpe_ids))
            experiments.append({
                'name': f'depth_L{n_l}',
                'config': cfg,
                'train_ids': bpe_ids[:split], 'val_ids': bpe_ids[split:],
                'tokenizer': bpe_tok, 'decode_fn': lambda ids: bpe_tok.decode(ids),
            })

    # ── D. 宽度消融 ──
    if args.experiments in ['all', 'width']:
        head_map = {160: 4, 256: 4, 384: 6, 512: 8}
        for n_embd, n_head in head_map.items():
            n_kv = min(2, n_head)
            cfg = ModelConfig(
                n_layer=4, n_embd=n_embd, n_head=n_head, n_kv_head=n_kv,
                block_size=128, batch_size=16, max_iters=args.steps,
                eval_interval=200, learning_rate=3e-4,
                use_sdpa=True, use_mixed_precision=True,
                ckpt_dir=os.path.join(args.out_dir, 'checkpoints'),
                log_dir=os.path.join(args.out_dir, 'logs'),
            )
            split = int(0.9 * len(bpe_ids))
            experiments.append({
                'name': f'width_E{n_embd}',
                'config': cfg,
                'train_ids': bpe_ids[:split], 'val_ids': bpe_ids[split:],
                'tokenizer': bpe_tok, 'decode_fn': lambda ids: bpe_tok.decode(ids),
            })

    # ── E. 学习率消融 ──
    if args.experiments in ['all', 'lr']:
        for lr in [1e-4, 3e-4, 1e-3]:
            cfg = ModelConfig(
                n_layer=6, n_embd=256, n_head=4, n_kv_head=2,
                block_size=128, batch_size=16, max_iters=args.steps,
                eval_interval=200, learning_rate=lr,
                use_sdpa=True, use_mixed_precision=True,
                ckpt_dir=os.path.join(args.out_dir, 'checkpoints'),
                log_dir=os.path.join(args.out_dir, 'logs'),
            )
            split = int(0.9 * len(bpe_ids))
            experiments.append({
                'name': f'lr_{lr}',
                'config': cfg,
                'train_ids': bpe_ids[:split], 'val_ids': bpe_ids[split:],
                'tokenizer': bpe_tok, 'decode_fn': lambda ids: bpe_tok.decode(ids),
            })

    # ── 运行所有实验 ──
    print(f"\n{'='*60}")
    print(f" M5 消融实验: {len(experiments)} 组")
    print(f"{'='*60}\n")

    results = []
    for i, exp in enumerate(experiments):
        print(f"\n{'─'*50}")
        print(f" [{i+1}/{len(experiments)}] {exp['name']}")
        print(f"{'─'*50}")
        result = run_experiment(
            name=exp['name'],
            config=exp['config'],
            train_ids=exp['train_ids'],
            val_ids=exp['val_ids'],
            tokenizer=exp['tokenizer'],
            device=device,
            decode_fn=exp['decode_fn'],
            log_dir=os.path.join(args.out_dir, 'logs'),
            steps=args.steps,
        )
        results.append(result)
        # 清理 GPU 显存
        if device == 'cuda':
            torch.cuda.empty_cache()

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(" 消融实验结果汇总")
    print(f"{'='*60}")
    print(f"{'实验':<20} {'参数(M)':>8} {'Val Loss':>10} {'PPL':>8} {'BPB':>8}")
    print("─" * 60)
    for r in results:
        print(f"{r['name']:<20} {r['params_M']:>8.2f} {r['val_loss']:>10.4f} {r['ppl']:>8.2f} {r['bpb']:>8.4f}")

    # 保存结果
    results_path = os.path.join(args.out_dir, 'ablation_results.json')
    save_eval_results(results, results_path)
    print(f"\n结果已保存: {results_path}")


if __name__ == '__main__':
    main()
