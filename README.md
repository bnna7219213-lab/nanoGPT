# nanoGPT

从零构建 GPT（Decoder-only Transformer），支持消融实验、EDA 领域适配和多轨演进。

## 项目状态

- **阶段**: M1（缺陷修复与库化已完成）→ M2（训练工程化）→ → M9（全量闭环）
- **硬件目标**: RTX 4050 Laptop 6GB（本地）+ 云短租
- **模型规模**: Config-S (~23M，消融实验) / Config-M (~60M，主线）
- **技术栈**: PyTorch 2.x + CUDA 13.0

## 快速开始

### 1. 安装环境

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
# 或 CPU 版: pip install torch
pip install numpy
```

### 2. 命令行训练

```bash
# 默认配置
python llm.py

# Tiny 配置（快速验证）
python llm.py --config tiny --iters 500

# 自定义参数（消融）
python llm.py --n-layer 6 --n-embed 256 --n-head 8 --iters 2000
```

### 3. 作为库导入（不触发训练）

```python
from llm import ModelConfig, GPTLanguageModel, Trainer

# 消融实验：同进程构造两个不同配置
cfg_a = ModelConfig(n_layer=4, n_head=4)
cfg_b = ModelConfig(n_layer=6, n_head=6)
model_a = GPTLanguageModel(cfg_a)
model_b = GPTLanguageModel(cfg_b)
```

## 项目结构

```
nanoGPT/
├── llm.py              # 核心模块（ModelConfig + GPTLanguageModel + Trainer）
├── configs/            # 模型配置预设
├── scripts/            # 训练/评测脚本
├── tests/              # 单元测试
├── data/               # 训练数据（git 不追踪）
├── checkpoints/        # 模型检查点（git 不追踪）
├── requirements.txt
├── README.md
└── docs/               # 详细设计文档
```

## 模型配置

| 配置名 | n_layer | n_embd | n_head | block_size | 估计参数 | 用途 |
|--------|---------|--------|--------|------------|----------|------|
| tiny   | 2       | 64     | 4      | 32         | ~0.5M    | 快速验证 |
| mini   | 4       | 128    | 4      | 64         | ~3M      | 开发调试 |
| small  | 6       | 384    | 6      | 128        | ~23M     | Config-S 消融 |
| medium | 8       | 640    | 10     | 128        | ~60M     | Config-M 主线 |

## 与 LLMInnovation 的关系

- `llm.py`（本项目）: PyTorch 实现，**可真实训练**，承担 M0→M9 主线
- `LLMInnovation/`（对比项目）: 纯 JS 实现，降级为**架构蓝图与模块契约来源**

路线图详见 `docs/roadmap.md`。
