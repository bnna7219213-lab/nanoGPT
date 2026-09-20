# LLM 升级计划 v2（摘要）— v0.1 → v3.0 与 EDA 工具链 Agent 落地

> 日期：2026-09-20
> **完整主文档**：`C:\Users\bnna7\workspace\git\LLMInnovation\docs\upgrade-plan-v0.1-v3.0.md`
> （已挂入该仓库 `docs/index.md` 导航，标注为「当前有效计划」）
> **本文取代**：本目录原 `plan.md`，已归档为 `plan-v1-archived.md`
> **本文取代**：`LLMInnovation/plan.md`（1273 行）的规模与性能结论，该文件降级为历史文档

---

## 1. 五项已确认决策

| 编号 | 决策项 | 结论 |
|---|---|---|
| D1 | 主线技术栈 | **PyTorch 为真实训练与落地主线**；LLMInnovation（纯 JS）转为架构蓝图与模块契约来源，不再投入新功能 |
| D2 | 存量资产处置 | LLMInnovation 代码/测试/文档全部保留，另加**真实性闸门 G1–G10**，把「形状正确」与「能学习」显式分开 |
| D3 | Agent 场景 | **EDA / 硬件工具链助手**：立项调研、器件选型、规范问答、DFM/DFA 检查、ECAD-MCAD 交换、评审文档生成 |
| D4 | 大脑策略 | **双轨制**：轨道 A 自训练小模型（可控/可离线/窄域）；轨道 B Agent 应用（外部强模型作主脑，不被自训练进度绑架） |
| D5 | 算力预算 | 本地为主（RTX 4050 Laptop 6GB）+ 关键阶段云短租，云支出上限 ¥500 量级 |

---

## 2. 本次审计的实测发现（全部可复现）

### 2.1 LLMInnovation 从未、也无法训练任何模型

| 检查项 | 实测结果 |
|---|---|
| `grep -rn "backward\|gradOf\|computeGrad\|autograd" src/` | **0 匹配** — 无反向传播 |
| `grep -rn "crossEntropy\|computeLoss\|function loss" src/` | **0 匹配** — 无损失函数 |
| 文本模型训练循环 | 无。`trainStep` 仅存在于 diffusion/sds，且用有限差分而非梯度 |
| `node --test test/*.js` | 141/141 通过，**总耗时 384 ms** |
| `CONFIG_3B` 实测参数量 | **2.307 B**（标称 3B，偏差 30%） |
| `CONFIG_3B` 初始化 / 常驻内存 | **83.7 s / RSS 9.19 GB** |
| `CONFIG_3B` 单次前向 | seqLen=4 → 19.92 s；seqLen=8 → 19.65 s ⇒ **≈0.05 token/s** |
| `CONFIG_20B` | 需 80 GB Float32Array（本机 RAM 23.3 GB）⇒ **必然 OOM** |
| `CONFIG_3B/20B/100B` 是否被测试实例化 | **从未**。测试仅 `assert.strictEqual(CONFIG_3B.vocabSize, 32768)` — 在断言常量等于它自己写的常量 |
| `KVCache` 是否接入主模型 | 否。`transformer3b.js:272` `generate()` 每步重跑全量前向 ⇒ 推理优化是死代码 |
| `distributed.js:29` allReduce | `result[i] = localData[i] / numDevices` ⇒ **标量除法，不是 AllReduce，数学错误** |
| `rope.js:40-41` | 频率按 `dModel`(2560) 而非 `head_dim`(80) 计算 ⇒ 差 32 倍尺度 |
| `transformer3b.js:213` | RoPE 施加在 Q/K **投影之前**的整个残差流上；`W_Q·RoPE(x) ≠ RoPE(W_Q·x)` ⇒ 位置编码语义失效 |
| `rope.js:26` NTK | `base * scale^((scale-1)/alpha)`，标准式须含维度项 `scale^(d/(d-2))` |
| `attention.js:114-121` matmul | 文档称 `A(m×k)@B(k×n)`，实现按 `bOff=j*k` 索引 ⇒ 实际算 **A@Bᵀ** |
| `attention.js:28,77` mask | 文档称「-∞ 表示遮盖」，实现按「非零表示遮盖」 |
| `README.md:69` vs `transformer3b.js:247` | README 称因果掩码「预分配一次避免逐层分配」，`_getCausalMask()` 实际**每层每次前向重新分配并拷贝** |

**前向耗时与 seqLen 无关（19.9 s vs 19.7 s）** 说明瓶颈是权重流式读取：每次前向须扫完 9.19 GB Float32Array，有效带宽 ≈466 MB/s。这是纯 JS 手写 matmul 的结构性天花板。

⇒ **「141/141 通过」验证的是模块可构造、维度正确、无 NaN，不是模型能力。** 384 ms 这个数字本身就是反证。

### 2.2 `nanoGPT/llm.py` 的 11 处必修缺陷（能真跑，是主线唯一可用起点）

最关键三处：

- **`llm.py:81` 正确性 bug**：`wei = q @ k.transpose(-2,-1) * C**-0.5`，其中 `C = n_embd = 128`，但应除以 `head_size = 32` ⇒ 注意力 logits 被**低估 2 倍**，softmax 过度平滑，直接损害模型质量。
- **`llm.py:9-19,64-146` 阻断消融**：超参全是模块级全局变量，所有模块直接引用 ⇒ **无法在同进程构造两个不同配置的模型**，原计划要求的「单变量消融」在结构上不可能实现。
- **`llm.py:27-28` 数据破坏**：`urlretrieve(url, "input.txt")` **无条件覆盖**当前目录 `input.txt`（含用户自备语料），无超时/重试/校验，断网即崩；与注释声称的「可替换成自己的 txt」矛盾。

其余：无 `__main__` 保护（import 即训练）、`arange` 用全局 device、残差投影未做 scaled init、`generate()` 无 KV cache 与采样参数、用 vocab index 0（`'\n'`）冒充 BOS、per-head `tril` buffer 越界风险、AdamW 未分组 weight decay、`estimate_loss` 开销过高、`FeedFoward` 拼写错误。

### 2.3 环境现状

| 项 | 实测 |
|---|---|
| GPU | RTX 4050 Laptop，6141 MiB 总量，**桌面进程已占 1236 MiB ⇒ 可用约 4.8 GB**（非 6 GB），95 W，WDDM |
| 驱动 / CUDA | 580.88 / CUDA 13.0 |
| 系统内存 | 23.3 GB |
| Python | 3.14.3（默认）、3.11（备用） |
| **PyTorch** | **两个解释器均未安装** |
| 可用 wheel | torch 2.14.0；`cu126` / `cu130` 通道均有 cp314 Windows 轮子。**PyPI 默认 Windows 轮子是 CPU 版，必须指定 `--index-url`** |
| nanoGPT 目录 | **不是 git 仓库** |

### 2.4 两栈吞吐对比（D1 的定量依据）

| 栈 | 规模 | 实测/推算 | 训练可行性 |
|---|---|---|---|
| LLMInnovation（纯 JS） | 2.307 B | 前向 19.7 s，≈0.05 token/s | ❌ 无 loss/backward；补齐后 1M token 仍需数月 |
| PyTorch + RTX 4050（主线） | 60 M | 推算 5k–15k token/s（M0 实测校准） | ✅ 1B token ≈ 20–55 GPU-h |

**差距约 10⁵–10⁶ 倍** —— 不是偏好问题，是数量级问题。

---

## 3. 原 `plan.md` 的 14 项不合理之处

| # | 级别 | 原文 | 为什么不合理 |
|---|---|---|---|
| P1 | 🔴 阻断 | 全篇假设环境就绪 | torch 根本没装；缺 P0 环境固化阶段，**所有验收命令当前一条都跑不了** |
| P2 | 🔴 技术错误 | 「val perplexity 优于字符级基线 ≥30%」 | **跨分词器比 ppl 在数学上无意义**（vocab 65 vs 32k 量纲不同）。必须换算 **bits-per-byte** |
| P3 | 🔴 阶段错配 | `ifeval_mini`/`redteam` 放进 `bench_v1` 用于 v1.0 | v1.0 是**未经 SFT 的 base model**，不会遵循指令，ifeval 必然 ≈0；base model 无「拒绝率」概念 |
| P4 | 🔴 算力误判 | 「50M–150M，6GB GPU 可用梯度累积训练」 | **梯度累积只降激活显存，不降参数/优化器状态显存**。150M×16B=2.4GB，合计 ≈4.7GB，在**可用 4.8GB** 下触顶无余量 |
| P5 | 🔴 验收不可信 | 「SDPA 应提升 ≥2x」 | `block_size=64` 下 attention 占比极小，实测收益约 1.1–1.3×，CPU 上可能更慢 ⇒ 阈值必然导致误判 |
| P6 | 🟠 结构错误 | 「方向 C 不得超前于方向 A」 | 把 **agent 落地锁死在自训练进度上**；60M 模型无法可靠支撑多步 tool calling ⇒ agent 永远落不了地。**双轨制下必须废除** |
| P7 | 🟠 目标错配 | 把 v2.5/v3.0 的 Agent 平台建立在自训练小模型上 | 小模型的指令跟随率、JSON 合法率、长链路规划会崩。可靠性须靠**工具约束 + 确定性计算 + 人工闸门**换取 |
| P8 | 🟠 瓶颈误判 | 「v1.0→v2.0 云为主（最大投入）」 | 60M/1B token 的 6ND=3.6e17 FLOPs，单张 4090 约 **3–8 GPU-h、¥10–30**。**算力根本不是最大投入，数据工程与人工评测才是** |
| P9 | 🟠 范围失控 | 6GB 笔记本 + 单人项目里排 MoE / Mamba 混合探索 | 与 agent 落地无因果关系，纯消耗 ⇒ 移出主线，降级 backlog |
| P10 | 🟡 缺失阶段 | 无「领域适配」阶段 | 自训练模型在本项目的**唯一真实价值点恰是 EDA 领域适配**（D3），却完全缺失 ⇒ 新增 M5 |
| P11 | 🟡 缺失阶段 | 无环境地基、无缺陷修复阶段 | 直接跳到 v0.5 加功能，而 §2.2 的全局变量缺陷会**阻断 v0.5 自身要求的消融能力** ⇒ 新增 M0/M1 |
| P12 | 🟡 版本语义失衡 | v2.5 一个版本号承担 DPO+安全+红队+蒸馏+投机解码+工具+Agent | 与 v1.0/v2.0 工作量不成比例，无法排期 ⇒ 改里程碑制 M0–M9 |
| P13 | 🟡 合规缺失 | 无数据来源清单、许可审查、PII/出口管制 | v3.0 声称「生产级交付」却无合规项。EDA 领域尤其敏感（ECCN/EAR、datasheet 版权） ⇒ 新增 D1 与许可边界表 |
| P14 | 🟡 无中止条件 | 只有「不达门槛不进入下一阶段」 | 无 kill criteria，容易无限投入 ⇒ 每里程碑补中止条件 |

**LLMInnovation `plan.md` 的 7 项**：Q1「3B CPU 可跑」（实测 9.19GB / 19.7s 前向）、Q2「20B 单卡可推理」（需 80GB，必然 OOM）、Q3 在无 loss/backward 下宣布 Phase 1「3B 基线」完成、Q4 并行训练策略与 `distributed.js` 实现无对应、Q5 声称数据管线但仓库内无任何数据代码、Q6 文本主干无梯度却扩展 5 个模态、Q7 时间线基于「已完成」假前提。

---

## 4. 修正后的路线：双轨 + 里程碑

### 轨道 A（自训练模型，v0.1 → v3.0）

| 里程碑 | 版本 | 名称 | 前置 | 人日 |
|---|---|---|---|---|
| **M0** | v0.0 | 环境地基与硬件实测基线 | — | 1 |
| **M1** | v0.1.1 | `llm.py` 11 处缺陷修复与库化（解锁消融能力） | M0 | 2–3 |
| **M2** | v0.5 | 训练工程化（配置/seed/ckpt/日志/调度/裁剪/SDPA/bf16/**分块 CE**） | M1 | 3–5 |
| **M3** | v0.8 | 分词器 + 数据管线 D1–D10 + 评测体系 E1–E5 | M2 | 8–12 |
| **M4** | v1.0 | 能力基线：首次真实预训练（Config-M 60M / 0.3–1B token） | M3 | 2–4 |
| **M5** | v1.5 | **EDA 领域适配**（原计划缺失；含 1k→4k 上下文扩展） | M4 | 7–10 |
| **M6** | v2.0 | 指令化 + tool-call JSON 格式 | M5 | 10–15 |
| **M7** | v2.5 | DPO 对齐 + 安全 + 领域幻觉拦截 | M6 | 10–14 |
| **M8** | v3.0-rc | KV cache + 量化 + OpenAI 兼容 API + 并发限流 | M7 | 10–15 |
| **M9** | v3.0 | 回流→迭代→发布闭环 + model card + 回滚演练 | M8 | 7–10 |

### 轨道 B（Agent 落地，**从 M2 起并行，不等轨道 A**）

| 里程碑 | 名称 | 前置 | 人日 |
|---|---|---|---|
| **B0** | Agent 骨架 + 工具契约（JSON Schema）+ 审计 | **M2** | 5–8 |
| **B1** | RAG 检索管线（EDA 语料） | B0 | 5–8 |
| **B2** | 工具集 T1–T9 实现 | B0 | 10–15 |
| **B3** | 护栏与人工闸门 | B2 | 5–8 |
| **B4** | Agent 评测集（30–50 golden task）+ success rate 基线 | B2 | 8–12 |
| **B5** | 双后端路由（可离线子任务切自训练模型） | B3 + M5/M6 | 5–8 |
| **B6** | 部署与运维（并发/监控/灰度/回滚） | B4 + M8 | 7–10 |

**关键路径**：Agent MVP = B0→B4 ≈ **4–6 周**，从 M2 结束（第 2 周）即可启动 ⇒ **第 6–8 周有可用 agent**（原计划 20+ 周）。v3.0 全量 ≈ **16–20 周**（原计划 20–38 周）。

### 冻结的模型配置（显存账本已验证，可用显存按 4.8 GB 计）

| 参数 | Config-S（消融） | Config-M（主线） |
|---|---|---|
| `n_layer` / `n_embd` | 6 / 384 | 8 / 640 |
| `n_head` / `n_kv_head` | 6 / 2 | 10 / 2 |
| `head_dim` | 64 | 64 |
| `d_ff`（SwiGLU） | 1024 | 1728 |
| `vocab_size` | 32768（M3 对 `{16k,32k}` 消融后冻结） | 同 |
| `ctx` | 512 | 1024 → 4096（M5 后） |
| 实测参数量 | ≈23 M | ≈60 M |
| 显存合计 | ≈1.2 GB ✅ | ≈2.7–3.1 GB ✅ |

> `logits = micro_bs × ctx × vocab` 是隐藏杀手：`4×1024×32k` 在 CE 内 fp32 上取后约 **0.8 GB** ⇒ **分块交叉熵是 M2 的硬要求，不是优化项**。
> **强制约定**：`head_dim` 恒为 64，RoPE 按 head_dim 逐头在 Q/K **投影之后**施加；命名一律用实测参数量，禁止未经验证的标称值。

---

## 5. Agent 落地要点（详见主文档第八部分）

**核心设计原则（修正 P7）**：**数值与合规判定必须由确定性代码执行，LLM 只负责路由、抽取、解释、成文。** 这是让 60M 小模型也能可靠参与 agent 的唯一途径。

**工具集 T1–T9**：`search_spec`（检索）、`part_lookup`（器件库）、`calc_derating`（**确定性降额计算**）、`stackup_query`（叠层/阻抗）、`dfm_check`（规则引擎）、`doc_render`（模板填充，非自由生成）、`bom_normalize`、`unit_convert`/`tolerance_stack`、`handoff_ecad_mcad`（**写操作，强制人工闸门**）。

**明确不做的边界**：❌ 生成可制造 PCB 版图 ❌ 直接驱动 EDA 工具布局布线 ❌ 输出无工具引用的数值结论 ❌ 替代认证/合规判定 ❌ 出口管制器件的最终采购决策。

**编排**：ReAct + **受限状态机** `INTAKE→PLAN→ACT→OBSERVE→REPLAN→GATE→SYNTH→AUDIT`，`max_steps=6`、`max_wall_time=120s`、写操作带幂等键、全轨迹落 SQLite 可重放。

**Agent 评测（原计划完全缺失）**：`task_success_rate ≥70%`、`tool_select_acc ≥90%`、`arg_valid_rate ≥95%`、`step_efficiency ≤1.5`、`grounded_faithfulness =100%`、`fabrication_rate =0`、`hazard_rate =0`（硬性）、`cost_per_task`、`latency_p95 <60s`、`offline_capable_rate ≥40%`。

**落地判定 6 条**：指标达标 + 连续 2 周真实使用 ≥20 次/周无 P0 + 写操作闸门日志可查 + 断云降级实测可用 + 数据许可合规通过 + model card 与回滚演练齐备。

**真实性闸门 G1–G10（D2 侧）**：保留 LLMInnovation 全部资产，但公开标注 G4–G10 为 🔴（无梯度、无 loss、allReduce 数学错、KVCache 死代码、matmul/mask 契约不符、RoPE 三处错、配置断言无效），并在 README 顶部加状态声明，防止「141/141」再被误读为模型可用。

---

## 6. 最高优先级风险

| ID | 风险 | 触发信号 | 对策 |
|---|---|---|---|
| **R4** | EDA 语料许可审查后可用量不足 ⇒ **轨道 A 价值归零（致命）** | D1 审查后 <100MB 可用 | 轨道 A 止于 v1.0，全力轨道 B（外部主脑 + RAG） |
| R5 | 厂商 datasheet 版权风险 | 书面结论未明 | 标为高风险源暂不入训，仅用自有文档 + 开源项目文档 |
| R3 | 笔记本长训热降频 / WDDM 驱动重置 | M0 的 30 分钟降频曲线掉 >20% | 全量训练改云短租，本地只跑 ≤2h 消融 |
| R6 | 60M 模型 tool-call JSON 合法率 <95% | M6 验收失败 | 该能力永久由外部主脑承担，自训练只做分类/抽取/rerank |
| R8 | Agent 编造器件参数 ⇒ 工程安全事故 | B4 `fabrication_rate >0` | 引用强制 + 逐字段匹配拦截，不达标不发布 |
| R1 | torch CUDA 在 Python 3.14 + Windows 兼容问题 | M0 安装失败 | 退 `cu126` → 退 Python 3.11 → 退 WSL2 |

---

## 7. 本周立即行动项

1. **【M0，30 分钟】装环境**
   ```bash
   cd C:/Users/bnna7/RaccoonWork/nanoGPT
   py -3.14 -m venv .venv && source .venv/Scripts/activate
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   pip freeze > requirements.lock.txt
   ```
2. **【M0，1 小时】git init + 仓库骨架**（按主文档 §9.1）
3. **【M0，半天】硬件实测** → `scripts/bench_hardware.py`，测 tok/s、显存峰值、30 分钟降频曲线；**用实测值重算主文档 §2.3 与 §11.2**，这一步决定 M4 走本地还是云
4. **【M1，1 天】先修最痛两处**：`llm.py:81` 注意力缩放（单独 commit，记录修复前后 val loss）+ 超参全局变量 → `ModelConfig`（消融能力前置）
5. **【D2，1 小时】给 LLMInnovation 打状态声明**：README 顶部注明「形状级验证原型，未训练任何模型」+ 实测数字（2.307B / 19.7s / 9.19GB），链接 `docs/reality-gates.md` 与主文档
6. **【D1，需你决策】EDA 语料许可预审** —— 对「自有文档 / 开源 EDA 项目 / 厂商 datasheet / 行业标准」四类给出可用性初判。**这是 R4（致命风险）的唯一解法，也是全计划最早需要人工判断的一项。**

---

## 8. 预算汇总

| 项 | 数值 |
|---|---|
| 人力（轨道 A，M0–M9） | ≈89 人日 ≈ 18 周 @5 人日/周 |
| 人力（轨道 B，B0–B6） | 40–66 人日，与 A 并行 |
| **Agent MVP** | **28–43 人日 ≈ 4–6 周** |
| 云算力 | M4 预训练 3–8 GPU-h（¥10–30）+ M5 领域适配 5–15 GPU-h（¥15–50）+ 备用与不可预见 ⇒ **合计 ≤ ¥500** |
| M6/M7/M8 | **本地即可**（60M 全参 SFT/DPO <3GB 显存） |

> 云单价为量级参考（4090 约 ¥1.5–3/GPU-h），下单前须实时核价。排期以 **GPU-hour** 为主单位（可复算），成本为副单位。若 R3 触发，云预算上浮至 ¥800，需重新确认。

---

## 9. 与两份旧计划的差异速查

| 维度 | 原 task/plan.md | 原 LLMInnovation/plan.md | 本计划 |
|---|---|---|---|
| 技术栈 | PyTorch（未验证可用） | 纯 JS（无 backward） | **PyTorch 主线 + JS 转蓝图** |
| 环境阶段 | ❌ 无 | ❌ 无 | ✅ M0，含实测 |
| 代码缺陷审计 | 部分（仅"组件陈旧"） | ❌ 无 | ✅ 11 项（llm.py）+ 4 类（JS），带行号 |
| 模型规模 | 50–150M（150M 不可行） | 3B/20B/100B（纸面） | **23M / 60M（显存账本验证）** |
| 跨分词器评估 | ppl 直接比（错误） | ppl | **BPB** |
| 评测阶段匹配 | ifeval/redteam 用于 v1.0（错配） | — | ✅ base / instruct 分离 |
| 领域适配阶段 | ❌ 无 | ❌ 无 | ✅ M5（价值核心） |
| 数据工序 | 一句括号 | 一节但无代码 | ✅ D1–D10，含许可边界表 |
| Agent 与模型进度 | 强绑定（不得超前） | — | **解绑双轨** |
| Agent 详规 | 每版本一行 | ❌ 无 | ✅ 边界/用户故事/T1–T9/状态机/护栏/路由/10 项指标/SLO/6 条落地判定 |
| 算力认知 | "最大投入" | "CPU 可跑 3B" | **实测：算力极小，数据与评测才是瓶颈** |
| 中止条件 | ❌ 无 | ❌ 无 | ✅ 每里程碑 kill criteria |
| 时间线 | 20–38 周 | 35 周（假前提） | **agent MVP 6–8 周；v3.0 全量 16–20 周** |
