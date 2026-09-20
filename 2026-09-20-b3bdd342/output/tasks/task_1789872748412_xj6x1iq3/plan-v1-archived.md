# LLM 改进计划：v0.1 → v3.0

> 项目：nanoGPT 自训练 LLM
> 日期：2026-09-20
> 基线：`llm.py`（字符级 nanoGPT 克隆） · `llm_improve v0.1.py`（空壳，仅一行注释）

---

## 0. 代码审计（真实基线）

| 维度 | `llm.py` 现状 | 判定 |
|---|---|---|
| 分词 | 字符级，`set(text)` 建词表，vocab ≈ 65 | ⚠️ 仅能处理英文小语料 |
| 数据 | 硬编码下载 tinyshakespeare；无清洗、无去重、无版本 | ⚠️ 不可换域、不可复现 |
| 模型 | 4 层 / n_embd=128 / 4 头 / ReLU FFN / 可学习位置嵌入 | ✅ 结构正确，但组件陈旧 |
| 注意力 | 手写 `q@k.T`，无 FlashAttention、无 KV cache | ⚠️ 训练与推理都慢 |
| 归一化 | Pre-LN + LayerNorm | ⚠️ 未用 RMSNorm |
| 训练 | 固定 LR=1e-3，AdamW，**无 warmup / 无余弦退火 / 无梯度裁剪** | ❌ 收敛质量不可控 |
| 上下文 | block_size=64 | ⚠️ 太短 |
| 评估 | 仅 train/val loss，无独立评估集、无基准 | ❌ 无能力度量 |
| 工程 | 单文件脚本，超参硬编码在模块级，无 seed、无 checkpoint、无日志 | ❌ 不可复现 |
| 推理 | `generate()` 朴素循环，每步全量前向 | ⚠️ O(n²) 重复计算 |

`llm_improve v0.1.py`：当前仅含一行注释 `#加上学习率调度、梯度裁剪、FlashAttention的进阶版代码`，**尚未实现**——这正是 v0.5 的起点。

**结论**：当前处于真正的 v0.1 状态——管线能跑通、结构正确，但工程化、数据、评估、效率四个维度全部缺失。下面三条方向并行推进，主线为版本阶梯。

---

## 1. 设计原则

1. **一主两辅**：主线（方向 A）是版本阶梯 v0.1→v3.0；方向 B（架构）与方向 C（部署）是并行轨道，不独立占版本号，而是**在每个版本内交付对应增量**。
2. **机器可验证**：每个验收项都能用一条命令或一个数值阈值判定，杜绝"看起来变好了"。
3. **门槛前置**：验收不通过不进入下一版本的算力投入。
4. **单变量消融**：架构改动一次只动一处，配套消融报告。
5. **资产强绑定**：数据集 / 代码 commit / checkpoint / 评估报告四者按版本绑定归档。

---

## 2. 三条方向总览

| | 方向 A：核心语言能力（主线） | 方向 B：架构进化（并行） | 方向 C：应用与部署（并行） |
|---|---|---|---|
| v0.1 | 字符级 baseline 跑通 | vanilla Transformer | 基础文本生成 |
| v0.5 | 工程化收敛（可复现） | + LR 调度/梯度裁剪/FlashAttention | CLI 推理脚本 |
| v1.0 | 能力基线（subword + 基准） | + RoPE / RMSNorm / SwiGLU / GQA | API + 流式输出 |
| v2.0 | 能力跃升（数据扩量） | + 长上下文 / MoE 或混合架构 | RAG + function calling |
| v2.5 | 对齐与安全（SFT→DPO） | + 投机解码 / 蒸馏 | 工具使用 + Agent |
| v3.0 | 生产级交付 | 推理最优架构（量化友好） | 生产级 Agent 平台 |

---

## 3. 方向 A：核心语言能力阶梯（主线）

### v0.1 —— 基线（已完成）
- **状态**：`llm.py` 可跑通，loss 正常下降。
- **验收**：`python llm.py` 在 CPU 上 2000 步内 val loss < 2.5，能输出可读英文片段。

### v0.1 → v0.5 —— 工程化收敛
**关键改动**（落地到 `llm_improve v0.1.py`）：
- 学习率调度：warmup + cosine decay
- 梯度裁剪：`clip_grad_norm_(params, 1.0)`
- FlashAttention：`F.scaled_dot_product_attention(is_causal=True)`
- 配置外置：超参移出模块级，改为 `config.yaml` / dataclass
- 固定 seed + 记录数据哈希 + checkpoint 落盘 + loss 曲线 CSV
- 数据路径参数化（不再硬编码 URL）

**机器可验证验收**：
```bash
# 1) 可复现性：两次 run 的 val loss 偏差 < 1%
python train.py --config configs/v0.5.yaml --seed 42
python train.py --config configs/v0.5.yaml --seed 42
# 2) 调度生效
python -c "import yaml;c=yaml.safe_load(open('configs/v0.5.yaml'));assert c['scheduler']=='cosine'"
# 3) FlashAttention 生效（对比同配置吞吐提升）
python bench_step.py --attn manual  # 基线 tokens/s
python bench_step.py --attn sdpa    # 应提升 ≥ 2x（GPU）
```

### v0.5 → v1.0 —— 能力基线
**关键改动**：
- 分词：字符级 → **subword**（BPE，词表 8k–32k）
- 数据 v1：目标域语料 + 清洗（去重/质量过滤/评估集去污染），token ≈ 20×参数量
- 模型配置冻结（建议 50M–150M 参数，6GB GPU 可用梯度累积训练）
- 建立基准套件（见 §7），v1.0 分数作为此后所有版本的对比锚点

**机器可验证验收**：
```bash
python eval.py --suite bench_v1 --ckpt ckpt/v1.0.pt --report qa/v1.0_eval.json
# 要求：val perplexity 优于字符级基线 ≥ 30%；基准套件全部跑通无异常
python -c "import json;r=json.load(open('qa/v1.0_eval.json'));assert r['ok']"
```

### v1.0 → v2.0 —— 能力跃升
**关键改动**：
- 数据 v2：规模 ×3–10，分层配比（通用/代码/数学/对话）+ 退火段高质量数据
- 架构增量（方向 B）：RoPE、RMSNorm、SwiGLU、GQA，逐项消融
- 上下文 64/512 → 4k
- 引入 **SFT v1**（指令-响应对），获得指令跟随能力

**机器可验证验收**：
```bash
# 每项架构改动一份消融报告
python eval.py --ablation rope --report qa/ablation_rope.json
# 预训练提升阈值（示例）
python -c "import json;r=json.load(open('qa/v2.0_eval.json'));assert r['ppl'] <= 0.85*r['v1_ppl']"
# SFT 抽检
python eval.py --suite ifeval_mini --ckpt ckpt/v2.0_sft.pt  # 通过率 ≥ 70%
```

### v2.0 → v2.5 —— 对齐与安全
**关键改动**：
- 错误分析驱动的 SFT v2（幻觉/拒答/格式/指令丢失四类定向补数据）
- 偏好对齐：**DPO**（成本与稳定性优于 RLHF）
- 红队测试集 → 安全数据 → 复测闭环
- 多轮对话格式 + system prompt 约定

**机器可验证验收**：
```bash
python eval.py --suite redteam --ckpt ckpt/v2.5.pt      # 拒绝率 ≥ 95%
python eval.py --suite bench_v1 --ckpt ckpt/v2.5.pt     # 回归：较 v2.0 下降 < 2%
python judge.py --pair ckpt/v2.0.pt ckpt/v2.5.pt --n 200  # 胜率 ≥ 60%
```

### v2.5 → v3.0 —— 生产级交付
**关键改动**：推理优化（量化/KV cache/投机解码）、服务化、RAG 与工具调用、持续评测回流机制、model card 与回滚预案。

**机器可验证验收**：
```bash
python eval.py --suite bench_v1 --ckpt ckpt/v3.0_int4.pt  # 量化回退 < 1%
python bench_serving.py --url localhost:8000 --qps 10      # 达到 SLO
python pipeline/roundtrip_test.py                          # 回流→迭代→发布闭环跑通
```

---

## 4. 方向 B：架构进化（并行轨道）

从 `llm.py` 的 vanilla 结构出发，按版本逐步现代化。**每项改动独立消融，无效则回滚。**

| 版本 | 架构增量 | 动机 | 验证方式 |
|---|---|---|---|
| v0.1 | vanilla Transformer（ReLU FFN + LayerNorm + 可学习位置嵌入） | 基线 | — |
| v0.5 | `F.scaled_dot_product_attention` 替换手写注意力 | 训练/推理提速 2–4× | 吞吐对比基准 |
| v1.0 | RoPE 位置编码 + RMSNorm + SwiGLU | 支持外推、更稳、更省参数 | 消融报告 |
| v1.0 | GQA（分组查询注意力） | 降 KV cache 显存，推理提速 | 显存/吞吐对比 |
| v2.0 | 长上下文（RoPE scaling）4k→8k | 长文档能力 | 长文本评估集 |
| v2.0 | MoE 或 Mamba-Transformer 混合（探索） | 同等算力下提升容量 | 与同参数量 dense 对比 |
| v2.5 | 投机解码 + 知识蒸馏（大模型→小模型） | 推理加速 + 小模型能力补齐 | 延迟与质量对比 |
| v3.0 | 量化友好结构（量化感知训练） | 部署效率 | INT4 精度回退 < 1% |

**当前缺口（对照 `llm.py`）**：
- `Head.forward` 中 `wei = q @ k.transpose(-2,-1) * C**-0.5` → 应替换为 SDPA
- `FeedFoward` 用 `nn.ReLU()` → 应换 `SwiGLU`
- `nn.LayerNorm` → 应换 `RMSNorm`
- `position_embedding_table`（可学习）→ 应换 RoPE
- `MultiHeadAttention` 每个 head 独立 `nn.Linear` → 应合并为单个 QKV 投影（便于 GQA 与 KV cache）

---

## 5. 方向 C：应用与部署（并行轨道）

| 版本 | 交付物 | 验证方式 |
|---|---|---|
| v0.1 | 基础文本生成（`generate()`） | 能输出连贯片段 |
| v0.5 | CLI 推理脚本（加载 ckpt + 参数化采样） | `python sample.py --ckpt ... --prompt ...` |
| v1.0 | OpenAI 兼容 API + 流式输出 | `curl` 流式返回正常 |
| v2.0 | RAG 检索管线 + function calling 格式 | 检索问答抽检 + 工具调用样例 |
| v2.5 | 工具使用 + 单 Agent 循环 | 多步任务端到端跑通 |
| v3.0 | 生产级 Agent 平台（并发/限流/日志/脱敏） | 压测达 SLO + 灰度发布流程 |

**注意**：方向 C 的每个交付物都依赖方向 A 同版本的能力基线，不得超前——v2.0 的 RAG 必须建立在 v2.0 的模型能力之上。

---

## 6. 进度追踪表

| 阶段 | 方向 A 能力 | 方向 B 架构 | 方向 C 部署 | 状态 |
|---|---|---|---|---|
| v0.1 | 字符级 baseline | vanilla Transformer | 基础生成 | ✅ 已完成 |
| v0.5 | 工程化收敛 | SDPA + 调度 + 裁剪 | CLI 推理 | ⬜ 待开始 |
| v1.0 | subword + 基准 | RoPE/RMSNorm/SwiGLU/GQA | API + 流式 | ⬜ |
| v2.0 | 数据扩量 + SFT | 长上下文 + MoE 探索 | RAG + FC | ⬜ |
| v2.5 | DPO 对齐 + 安全 | 投机解码 + 蒸馏 | 工具 + Agent | ⬜ |
| v3.0 | 生产交付 | 量化友好架构 | Agent 平台 | ⬜ |

**跨版本待修问题清单**（从 `llm.py` 审计得出）：
- [ ] `max_iters`/超参硬编码在模块级 → v0.5 移入配置
- [ ] 数据下载硬编码 URL → v0.5 参数化
- [ ] 无 seed → v0.5 补
- [ ] 无 checkpoint 保存 → v0.5 补
- [ ] 无独立评估集 → v1.0 补
- [ ] 字符级词表 → v1.0 换 subword
- [ ] 无 warmup/余弦退火 → v0.5 补
- [ ] 无梯度裁剪 → v0.5 补
- [ ] 手写注意力 → v0.5 换 SDPA
- [ ] 无 KV cache（推理 O(n²)）→ v1.0 补

---

## 7. 基准套件（bench_v1）

固定版本归档，所有对比必须同套件同 decode 参数复测。

| 子集 | 内容 | 度量 |
|---|---|---|
| `ppl` | 独立验证集困惑度 | perplexity |
| `ifeval_mini` | 20–50 条指令跟随样例 | 通过率 |
| `qa_mini` | 常识/阅读理解小样例 | 准确率 |
| `redteam` | 越狱/有害/隐私诱导 | 拒绝率 |
| `regression` | 历史能力回归集 | 相对基线降幅 |

统一入口：
```bash
python eval.py --suite bench_v1 --ckpt <path> --report qa/<name>_eval.json
```

---

## 8. 工程目录结构（目标态）

```
nanoGPT/
├── configs/            # v0.5.yaml, v1.0.yaml ...
├── data/               # 原始语料（不进 git）
├── src/
│   ├── model/          # attention.py, block.py, gpt.py
│   ├── data/           # tokenizer.py, dataset.py
│   ├── train/          # trainer.py, scheduler.py
│   └── eval/           # suites/, judge.py
├── scripts/            # train.py, sample.py, eval.py, bench_*.py
├── ckpt/               # 按 run_id 归档
├── qa/                 # 评估报告 JSON
├── plan.md             # 本文档
├── llm.py              # v0.1 基线（保留作参考）
└── llm_improve v0.1.py # v0.5 实现（待填充）
```

---

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| 6GB GPU 训不动预训练主阶段 | 本地只做消融/SFT；预训练主阶段云短租 |
| 数据质量不达预期 | 数据 v2 先小规模消融验证再全量投入 |
| 对齐导致能力回退 | `regression` 套件卡门，超 2% 不发布 |
| 架构改动无收益 | 单变量消融，无收益即回滚，不累积技术债 |
| 范围蔓延 | 验收门槛前置定义，新需求进下一版本 backlog |

---

## 10. 建议时间线

| 阶段 | 周期 | 算力 |
|---|---|---|
| v0.1 → v0.5 | 2–4 周 | 本地 6GB |
| v0.5 → v1.0 | 4–8 周 | 本地 + 少量云 |
| v1.0 → v2.0 | 6–12 周 | 云为主（最大投入） |
| v2.0 → v2.5 | 4–6 周 | 云 |
| v2.5 → v3.0 | 4–8 周 | 云 + 部署环境 |

---

## 11. 立即行动项

1. 填充 `llm_improve v0.1.py`：在 `llm.py` 基础上加 warmup+余弦退火、梯度裁剪、SDPA，并把超参外置到 `configs/v0.5.yaml`
2. 加 seed + checkpoint 保存 + loss CSV 落盘，跑两次验证可复现（偏差 < 1%）
3. 把数据路径参数化，确认可切换到自有语料
4. 建 `src/` 目录骨架，为 v1.0 的 subword 与评估套件预留位置
