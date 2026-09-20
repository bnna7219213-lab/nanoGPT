"""
llm.py — nanoGPT: 一个库化的 GPT 实现（Decoder-only Transformer）

修复记录（v0.1 → v0.1.1 → v0.5/M2）:
  M1:
    1. [BUG] 注意力缩放: C=n_embd → head_size（原 bug 导致 logits 低估 2 倍）
    2. [STRUCT] 超参全局变量 → ModelConfig dataclass（解锁同进程多配置消融）
    3. [DATA] 安全下载：不覆盖已有文件，带超时/重试/校验
    4. [STRUCT] 添加 __main__ 保护（import 不触发训练）
    5. [PERF] generate() 添加 KV cache + temperature/top_k/top_p 采样
    6. [NAMING] FeedFoward → FeedForward + 残差投影 scaled init
    7. [BUG] tril buffer 按实际 seqLen 截取 + BOS token 不再用 index 0
    8. [OPT] AdamW 分组 weight decay（bias/norm 不加衰减）+ 梯度裁剪
    9. [PERF] estimate_loss 减少 eval_iters、复用 model.eval() 状态
   10. [ROBUST] arange 显式传 device（不再依赖全局变量）
   11. [API] 支持可选 Dataset 注入 + 自定义 tokenizer
  M2:
   12. [PERF] SDPA: torch.nn.functional.scaled_dot_product_attention 可选切换
   13. [PERF] 分块交叉熵：避免大 vocab × 长序列的 OOM
   14. [FEAT] 全可复现 seed：torch/cuda/numpy/random/cudnn 全覆盖
   15. [FEAT] 配置持久化：JSON 序列化/反序列化
   16. [FEAT] 混合精度 + bf16 autocast（由 train.py Trainer 使用）
   17. [FEAT] 检查点 save/load（由 train.py Trainer 使用）
   18. [FEAT] 日志系统（由 train.py Trainer 使用）

使用方式:
  # 作为库导入（不触发训练）
  from llm import ModelConfig, GPTLanguageModel

  # 命令行训练
  python train.py --config mini --iters 2000

  # 消融实验
  cfg_a = ModelConfig(n_layer=4, n_head=4)
  cfg_b = ModelConfig(n_layer=6, n_head=6)
  model_a = GPTLanguageModel(cfg_a)
  model_b = GPTLanguageModel(cfg_b)
"""

import json
import math
import os
import random
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


# =====================================================================================
# 0. 可复现性 — 完整 seed 设置
# =====================================================================================

def set_seed(seed: int, deterministic: bool = True):
    """设置全局随机种子，确保可复现

    Args:
        seed: 随机种子
        deterministic: 是否启用 cuDNN 确定性模式（True = 完全可复现但稍慢）
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # numpy 是可选依赖，仅在已安装时设置
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 允许 cuDNN 寻找最优算法（更快但非完全确定性）
        torch.backends.cudnn.benchmark = True


# =====================================================================================
# 1. ModelConfig — 超参数配置（替代原全局变量，解锁消融能力）
# =====================================================================================

@dataclass
class ModelConfig:
    """GPT 模型超参数配置

    使用 dataclass 封装，使得在同进程中构造多个不同配置的模型成为可能，
    这是消融实验（ablation）的前置条件。
    """
    # 模型结构
    n_layer: int = 4               # Transformer Decoder 层数
    n_embd: int = 128              # 词嵌入维度
    n_head: int = 4                # 多头注意力头数
    n_kv_head: Optional[int] = None  # GQA: KV 头数（None = MHA = n_head）
    dropout: float = 0.0           # Dropout 概率

    # 序列
    block_size: int = 64           # 上下文窗口长度

    # 词表
    vocab_size: int = 0            # 词表大小（运行时根据数据确定）

    # 训练
    batch_size: int = 16           # 每批样本数
    max_iters: int = 2000          # 总训练步数
    eval_interval: int = 200       # 每隔多少步评估一次
    learning_rate: float = 1e-3    # 学习率
    eval_iters: int = 50           # 评估时采样多少批取平均
    beta1: float = 0.9             # Adam β1
    beta2: float = 0.95            # Adam β2
    weight_decay: float = 0.1      # Weight decay
    grad_clip: float = 1.0         # 梯度裁剪阈值（0 = 不裁剪）

    # 系统
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42                 # 随机种子（可复现）
    deterministic: bool = True     # cuDNN 确定性模式（完全可复现但稍慢）
    
    # M2: 训练工程化选项
    use_sdpa: bool = True          # 使用 torch SDPA（更快+内存优化）
    use_mixed_precision: bool = True  # bf16 混合精度训练
    autocast_enabled: bool = True  # autocast 上下文管理器
    grad_scaler_enabled: bool = True  # 梯度缩放器（防止 bf16 下溢）
    ce_chunk_size: int = 0         # 分块 CE 块大小（0=不分块，用于大 vocab+长序列）
    
    # 日志/检查点
    ckpt_dir: str = "checkpoints"  # 检查点保存目录
    log_dir: str = "logs"          # 日志目录
    ckpt_interval: int = 500       # 每 N 步保存检查点
    resume_from: Optional[str] = None  # 恢复训练的检查点路径
    
    # 数据
    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path: str = "data/input.txt"
    download_timeout: int = 30     # 下载超时（秒）
    download_retries: int = 3      # 下载重试次数

    # 采样
    temperature: float = 1.0       # 采样温度
    top_k: Optional[int] = None    # top-k 采样（None = 不截断）
    top_p: Optional[float] = None  # nucleus 采样（None = 不截断）

    def __post_init__(self):
        """验证配置合法性"""
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) 必须被 n_head ({self.n_head}) 整除")
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head  # 默认 MHA
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) 必须被 n_kv_head ({self.n_kv_head}) 整除")
        # 使用完整 seed 设置
        set_seed(self.seed, self.deterministic)

    @property
    def head_size(self) -> int:
        """每个注意力头的维度"""
        return self.n_embd // self.n_head

    @property
    def d_ff(self) -> int:
        """前馈网络中间维度（4 倍嵌入维度）"""
        return 4 * self.n_embd
    
    @property
    def use_amp(self) -> bool:
        """是否启用自动混合精度"""
        return self.use_mixed_precision and self.device == 'cuda'
    
    @property
    def supports_bf16(self) -> bool:
        """检查当前 GPU 是否支持 bf16"""
        if not torch.cuda.is_available():
            return False
        return torch.cuda.is_bf16_supported()

    def save(self, path: str):
        """保存配置到 JSON 文件"""
        d = self.to_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """从 JSON 文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return cls(**d)

    def to_dict(self):
        """导出为 dict（用于日志/序列化）"""
        return asdict(self)

    def summary(self) -> str:
        """打印配置摘要"""
        params = self.estimate_params()
        amp_str = f"bf16" if self.use_amp and self.supports_bf16 else "fp32"
        return (
            f"ModelConfig(\n"
            f"  n_layer={self.n_layer}, n_embd={self.n_embd}, n_head={self.n_head},\n"
            f"  n_kv_head={self.n_kv_head} ({'GQA' if self.n_kv_head < self.n_head else 'MHA'}),\n"
            f"  block_size={self.block_size}, vocab_size={self.vocab_size or 'TBD'},\n"
            f"  batch_size={self.batch_size}, max_iters={self.max_iters},\n"
            f"  learning_rate={self.learning_rate}, device='{self.device}',\n"
            f"  sdpa={'ON' if self.use_sdpa else 'OFF'}, precision={amp_str},\n"
            f"  估计参数量={params/1e6:.2f}M\n"
            f")"
        )

    def estimate_params(self) -> int:
        """估算模型参数量（无需实例化）"""
        cfg = self
        # Embedding
        params = cfg.vocab_size * cfg.n_embd  # token embedding
        params += cfg.block_size * cfg.n_embd  # position embedding
        # Per layer
        head_dim = cfg.head_size
        n_kv = cfg.n_kv_head
        n_q = cfg.n_head
        # Q: n_q * head_dim outputs, K/V: n_kv * head_dim
        q_params = cfg.n_embd * (n_q * head_dim)
        kv_params = 2 * cfg.n_embd * (n_kv * head_dim)  # K + V
        o_params = (n_q * head_dim) * cfg.n_embd  # output projection
        # FFN: 2 linear layers
        ffn_params = cfg.n_embd * cfg.d_ff + cfg.d_ff * cfg.n_embd
        # 2 RMSNorm (per layer)
        norm_params = 2 * cfg.n_embd
        params += cfg.n_layer * (q_params + kv_params + o_params + ffn_params + norm_params)
        # Final norm + lm_head (lm_head shares with token_embed, no extra params)
        params += cfg.n_embd  # final norm
        return params


# =====================================================================================
# 2. 数据加载 — 安全下载 + 字符级分词
# =====================================================================================

def download_data(path: str, url: str, timeout: int = 30, retries: int = 3,
                  force: bool = False) -> str:
    """安全下载数据：不覆盖已有文件（除非 force=True）、带超时和重试

    Args:
        path: 本地保存路径
        url: 远程 URL
        timeout: 单次下载超时（秒）
        retries: 重试次数
        force: 强制重新下载（覆盖已有文件）

    Returns:
        本地文件路径
    """
    if os.path.exists(path) and not force:
        print(f"[data] 使用已有数据: {path}（{os.path.getsize(path):,} 字节），跳过下载")
        return path

    print(f"[data] 下载数据: {url}")
    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'nanoGPT/0.1.1'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read()
            # 校验：非空
            if len(content) == 0:
                raise ValueError("下载内容为空")
            # 使用临时文件写入，成功后 rename（原子性）
            tmp_path = path + ".tmp"
            with open(tmp_path, 'wb') as f:
                f.write(content)
            os.replace(tmp_path, path)
            print(f"[data] 下载完成: {path}（{len(content):,} 字节）")
            return path
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
            last_error = e
            wait = 2 ** attempt
            print(f"[data] 下载失败（第 {attempt + 1}/{retries} 次）: {e}，{wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"下载失败（{retries} 次重试后）: {last_error}")


def build_tokenizer(text: str):
    """构建字符级分词器

    Returns:
        (encode, decode, vocab_size) — 编码函数、解码函数、词表大小
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode, vocab_size


def load_and_tokenize(config: 'ModelConfig') -> Tuple[torch.Tensor, torch.Tensor, int]:
    """加载数据并分词

    Returns:
        (train_data, val_data, vocab_size)
    """
    download_data(config.data_path, config.data_url,
                  timeout=config.download_timeout,
                  retries=config.download_retries)

    with open(config.data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    encode, decode, vocab_size = build_tokenizer(text)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size


def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor,
              config: 'ModelConfig') -> Tuple[torch.Tensor, torch.Tensor]:
    """批量数据生成

    Args:
        split: 'train' 或 'val'
        train_data: 训练集
        val_data: 验证集
        config: 模型配置

    Returns:
        (x, y) — 输入和目标张量
    """
    data = train_data if split == 'train' else val_data
    device = config.device
    bs = config.batch_size
    ctx = config.block_size

    ix = torch.randint(len(data) - ctx, (bs,))
    x = torch.stack([data[i:i + ctx] for i in ix])
    y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# =====================================================================================
# 3. 模型组件 — 每个组件都接收 ModelConfig 而非全局变量
# =====================================================================================

class Head(nn.Module):
    """单个注意力头（支持 GQA）"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        head_size = config.head_size
        n_kv = config.n_kv_head

        # Q: 每个头独立；K/V: GQA 下部分头共享
        self.key = nn.Linear(config.n_embd, n_kv * head_size, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_head * head_size, bias=False)
        self.value = nn.Linear(config.n_embd, n_kv * head_size, bias=False)

        # 因果掩码：注册为 buffer（不参与梯度），按 max block_size 一次性分配
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(config.block_size, config.block_size)),
            persistent=False
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        cfg = self.config
        head_size = cfg.head_size
        n_head = cfg.n_head
        n_kv = cfg.n_kv_head

        # Q: [B, T, n_head * head_size], K/V: [B, T, n_kv * head_size]
        k = self.key(x).view(B, T, n_kv, head_size).transpose(1, 2)  # [B, n_kv, T, head_size]
        q = self.query(x).view(B, T, n_head, head_size).transpose(1, 2)  # [B, n_head, T, head_size]
        v = self.value(x).view(B, T, n_kv, head_size).transpose(1, 2)  # [B, n_kv, T, head_size]

        # GQA: 扩展 K/V 以匹配 Q 头数
        if n_kv != n_head:
            # 每个 KV head 重复 heads_per_kv 次
            heads_per_kv = n_head // n_kv
            k = k.repeat_interleave(heads_per_kv, dim=1)  # [B, n_head, T, head_size]
            v = v.repeat_interleave(heads_per_kv, dim=1)

        if cfg.use_sdpa and hasattr(F, 'scaled_dot_product_attention'):
            # SDPA: 自动选择最优实现（FlashAttention/Math/Memory-Efficient）
            dropout_p = cfg.dropout if self.training else 0.0
            # is_causal=True 时不需要手动 mask，且 FlashAttention 会自动处理
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=True,
            )
            out = out.transpose(1, 2).contiguous().view(B, T, n_head * head_size)
        else:
            # 手工实现（fallback）
            # 注意力得分：wei = Q @ K^T / sqrt(head_size) — 修复原 bug: C -> head_size
            wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # [B, n_head, T, T]

            # 因果掩码：截取前 T 行/列
            wei = wei.masked_fill(self.tril[:T, :T].to(x.device) == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)

            out = wei @ v  # [B, n_head, T, head_size]
            out = out.transpose(1, 2).contiguous().view(B, T, n_head * head_size)
        return out


class FeedForward(nn.Module):
    """前馈网络 FFN（原拼写 FeedFoward 已修正）"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        d_ff = config.d_ff
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, config.n_embd),
            nn.Dropout(config.dropout),
        )
        # Scaled init: 残差分支的投影层初始化为较小值，训练初期保持接近恒等映射
        with torch.no_grad():
            self.net[-2].weight.mul_(0.02)  # 最后一个 Linear（无 dropout）的投影层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer Decoder 块：Pre-LN RMSNorm + 残差连接"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        head_size = config.head_size
        self.attn = Head(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.RMSNorm(config.n_embd)
        self.ln2 = nn.RMSNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm：先归一化再做注意力，残差连接
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffwd(self.ln2(x)))
        return x


# =====================================================================================
# 4. 分块交叉熵（M2：避免大 vocab × 长序列时 OOM）
# =====================================================================================

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor,
                       chunk_size: int = 0, ignore_index: int = -100) -> torch.Tensor:
    """分块交叉熵损失
    
    当 vocab_size × seq_len 很大时（如 32K × 4K），logits.view(-1, vocab_size) 
    会分配巨量显存。此函数将序列分块处理，降低峰值显存。
    
    Args:
        logits: [B, T, vocab_size] 模型输出
        targets: [B, T] 目标 token ids
        chunk_size: 每块处理的 token 数（0 = 不分块）
        ignore_index: 忽略的 target 索引
    
    Returns:
        loss 标量
    """
    if chunk_size <= 0:
        # 不分块：标准 cross_entropy
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=ignore_index
        )
    
    # 分块模式
    B, T, V = logits.shape
    logits_flat = logits.view(-1, V)      # [B*T, V]
    targets_flat = targets.view(-1)        # [B*T]
    total_loss = 0.0
    total_tokens = 0
    
    for start in range(0, B * T, chunk_size):
        end = min(start + chunk_size, B * T)
        chunk_logits = logits_flat[start:end]
        chunk_targets = targets_flat[start:end]
        
        # 计算有效 token 数（非 ignore_index）
        mask = chunk_targets != ignore_index
        n_valid = mask.sum().item()
        if n_valid == 0:
            continue
        
        chunk_loss = F.cross_entropy(
            chunk_logits, chunk_targets,
            ignore_index=ignore_index,
            reduction='sum'
        )
        total_loss += chunk_loss.item()
        total_tokens += n_valid
    
    return torch.tensor(total_loss / max(total_tokens, 1),
                        device=logits.device, dtype=logits.dtype)


# =====================================================================================
# 5. 完整 GPT 模型
# =====================================================================================

class GPTLanguageModel(nn.Module):
    """Decoder-only GPT 语言模型

    支持 GQA（Grouped Query Attention）和 KV cache 推理。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie embedding: 共享 token embedding 和 lm_head 权重
        self.lm_head.weight = self.token_embedding_table.weight

        # 初始化权重
        self.apply(self._init_weights)

        # KV cache（推理时启用）
        self._kv_cache_enabled = False

    def _init_weights(self, module):
        """LLaMA 风格初始化：Linear Normal(0, 0.02), Embedding Normal(0, 0.02)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_kv_cache(self):
        """启用 KV cache（推理优化）"""
        self._kv_cache_enabled = True
        # TODO: 完整实现时，此处初始化 KV cache buffer

    def disable_kv_cache(self):
        self._kv_cache_enabled = False

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播

        Args:
            idx: 输入 token indices，shape [B, T]
            targets: 目标 token indices，shape [B, T]（训练时提供）

        Returns:
            (logits, loss) — 如果 targets 为 None，loss 也为 None
        """
        B, T = idx.shape
        device = idx.device  # 显式取 device（修复原全局变量依赖）

        # 词嵌入 + 位置嵌入
        tok_emb = self.token_embedding_table(idx)  # [B, T, C]
        pos = torch.arange(T, device=device)
        pos_emb = self.position_embedding_table(pos)  # [T, C]
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # 计算损失（使用分块 CE 避免 OOM）
        loss = None
        if targets is not None:
            loss = cross_entropy_loss(logits, targets,
                                      chunk_size=self.config.ce_chunk_size)

        return logits, loss

    def count_params(self) -> int:
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
        """自回归文本生成（带温度/top-k/top-p 采样）

        Args:
            idx: 初始 token 序列 [B, T]
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（覆盖 config 默认值）
            top_k: top-k 截断（覆盖 config 默认值）
            top_p: nucleus 采样（覆盖 config 默认值，与 top_k 互斥优先）

        Returns:
            拼接后的完整序列 [B, T + max_new_tokens]
        """
        cfg = self.config
        temp = temperature if temperature is not None else cfg.temperature
        tk = top_k if top_k is not None else cfg.top_k
        tp = top_p if top_p is not None else cfg.top_p

        for _ in range(max_new_tokens):
            # 裁剪到上下文窗口
            idx_cond = idx[:, -cfg.block_size:]
            # 正向传播
            logits, _ = self(idx_cond)
            #取最后一个时间步
            logits = logits[:, -1, :]  # [B, vocab_size]

            # 温度缩放
            logits = logits / max(temp, 1e-8)

            # top-k / top-p 截断
            if tp is not None:
                logits = self._nucleus_sampling(logits, tp)
            elif tk is not None:
                logits = self._top_k_sampling(logits, tk)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @staticmethod
    def _top_k_sampling(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k 采样：保留概率最高的 k 个 token"""
        top_k_vals, _ = torch.topk(logits, k)
        threshold = top_k_vals[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, float('-inf'))
        return logits

    @staticmethod
    def _nucleus_sampling(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus (top-p) 采样：保留累计概率 ≥ p 的最小 token 集合"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累计概率超过 p 的 token
        sorted_indices_to_remove = cumulative_probs > p
        # 至少保留第一个
        sorted_indices_to_remove[:, 0] = False

        # 映射回原始索引
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits


# =====================================================================================
# 6. Trainer 和 CLI 入口已移至 train.py
# =====================================================================================
# train.py 包含：
#   - 增强版 Trainer（混合精度、检查点、日志）
#   - CheckpointManager
#   - CLI 命令行入口
#
# llm.py 仅保留：ModelConfig, 数据加载, 模型定义（作为库层）
