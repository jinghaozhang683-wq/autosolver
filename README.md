---
title: AutoSolver Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AutoSolver — 基于多臂老虎机的自适应调度求解 Agent

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![scipy](https://img.shields.io/badge/scipy-%E2%89%A51.9-green)](https://scipy.org)
[![OR-Tools](https://img.shields.io/badge/OR--Tools-%E2%89%A59.0-orange)](https://developers.google.com/optimization)
[![Flask](https://img.shields.io/badge/Flask-%E2%89%A53.0-lightgrey)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**美团首届 AI Hackathon 赛道 4 参赛项目**

---

## 项目概述

### 比赛背景

外卖配送平台需将一组**配送任务**实时分配给可用**骑手**。每个「任务-骑手」组合涉及合单配送、骑手接单概率、多骑手冗余分配等复杂因素。赛道 4 要求设计 **AI Agent 系统**，在 10 秒/用例内自主求解该分配优化问题。

### 核心挑战

| 挑战 | 说明 |
|------|------|
| **时间约束** | 每次求解需在 **10 秒内** 返回结果 |
| **非线性目标** | 多骑手备份下的期望成本为非线性函数 |
| **组合爆炸** | n 个任务 × m 个骑手的集合划分，搜索空间指数级 |
| **不确定性** | 骑手接单概率（willingness）∈ [0,1]，影响冗余分配策略 |
| **规模差异** | 实例从小（6 任务）到大（40 任务 × 33780 候选）跨度极大 |

---

## AutoSolver 架构

```
                         ┌───────── AutoSolverAgent (核心控制器) ──────────┐
         实例 ──► 画像分析 ─► UCB 多臂老虎机 ─► 评估(真实目标) ─► 保留最优
                         │         ▲                    │            │
                         │         └── 在线信用分配 ◄────┘            │
                         │         └── 跨实验记忆 ◄───── 学习 ◄──────┘
                         ├────────────────────────────────────────────┤
                         │                                            │
                         │  调度 7 类求解策略                          │
                         │  ┌──────┐ ┌──────┐ ┌─────┐ ┌──────────┐  │
                         │  │Greedy│ │ MIP  │ │CP-SAT│ │Col. Gen. │  │
                         │  └──────┘ └──────┘ └─────┘ └──────────┘  │
                         │  ┌──────┐ ┌──────┐ ┌─────┐                │
                         │  │  LP  │ │Heuri.│ │ LLM │                │
                         │  └──────┘ └──────┘ └─────┘                │
                         └────────────────┬───────────────────────────┘
                                          ▼
                            ┌─────────────────────────┐
                            │       Evaluator          │
                            │   (Solution.evaluate)    │
                            └─────────────────────────┘
                            ┌─────────────────────────┐
                            │        Memory            │
                            │   (trained_memory.json)  │
                            └─────────────────────────┘
```

---

## 核心设计亮点

### 1. 自适应多策略探索（UCB 多臂老虎机）

```python
# Agent 自主决定下一步试什么，无需人工指定
moves = {
    "greedy":      0.05,   # 贪心基线
    "heuristic":   0.92,   # 纯 Python 启发式（大实例主力）
    "exact":       0.80,   # 精确 MILP（小实例可达最优）
    "colgen":      0.75,   # 列生成（密集/低意愿场景）
    "lp":          0.25,   # LP 松弛 + 取整
    "perturb":     0.30,   # 扰动精修（迭代改善）
    "llm":         0.35,   # LLM 推理（多样性来源）
}
# 选臂: argmax[value + c·√(ln N / (n+1))]
```

### 2. 10 秒 Anytime 预算

```python
while time.time() - t0 < total_budget - 0.4:
    m = ucb_select(available_moves)
    sol = run_solver(m, time_slice)
    if sol.better_than(best):
        best = sol
    if sol.proves_optimal():    # 精确法证明最优——立即返回
        break
    if m.stalled_twice():       # 连续无改进——策略转向
        m.exhausted = True
```

### 3. 在线信用分配与策略转向

- 改进收益归一化后**除以耗时**，奖励快且好的策略
- 可重复策略连续 4 次零改进即**自动弃用（pivot）**
- 扰动精修可迭代运行，不同随机种子探索不同方向

### 4. 精确求解（scipy / OR-Tools）

- **MIP 求解器**：基于 HiGHS 的分支定界，中小实例**证明最优**
- **CP-SAT 求解器**：OR-Tools 约束规划，备选精确引擎
- **列生成**：LP 对偶定价按需生成列，克服静态列池爆炸

### 5. 跨实验学习

- 实例画像（规模 × 密度 × 意愿）→ 离散桶
- 每次求解后更新 `trained_memory.json`
- 新实例启动时引用历史统计**加权初始化**策略先验
- 通过预训练脚本（`pretrain.py`）覆盖 21 种画像配置

---

## 环境配置与安装

### 依赖

```
Python >= 3.8
scipy >= 1.9.0
numpy
ortools >= 9.0 (可选，--cpsat)
flask >= 3.0
anthropic >= 0.40 (可选，--llm)
```

### 安装

```bash
# 克隆项目
git clone https://github.com/jinghaozhang683-wq/autosolver.git
cd autosolver

# 安装核心依赖
pip install scipy numpy flask

# 可选：安装 OR-Tools（CP-SAT 引擎）和 LLM 支持
pip install ortools anthropic
```

### 网页 UI

```bash
python -m autosolver.web.server      # http://127.0.0.1:5000
```

浏览器中实时观看 Agent 逐步探索、评估、转向、学习。

---

## 使用方法

### 命令行

```bash
# 单用例
python -m autosolver.runner datasets/synth_medium20.txt --budget 10

# 批量
python -m autosolver.runner datasets/*.txt --budget 10

# 使用 CP-SAT 替代 MILP
python -m autosolver.runner case.txt --cpsat

# 启用 LLM 推理策略
python -m autosolver.runner case.txt --llm

# 自演展示
python -m autosolver.demo

# 重新预训练策略先验
python -m autosolver.pretrain
```

### Python API

```python
from autosolver import Problem
from autosolver.agent import AutoSolverAgent

# 加载数据
problem = Problem.parse(open("case.txt").read())

# 求解
agent = AutoSolverAgent(use_cpsat=False, use_llm=False, verbose=True)
solution = agent.solve(problem, total_budget=10.0)

# 结果
print(solution.score)            # 总成本（越低越好）
print(solution.assigned_count)   # 已分配任务数
print(solution.to_output())      # 分配方案

# 查看 Agent 决策轨迹
for attempt in agent.log:
    print(attempt)
```

---

## 算法说明

### 求解器对比

| 求解器 | 适用场景 | 原理 | 求解质量 |
|--------|----------|------|:--:|
| Greedy | 快速基线 | 5 种排序优先级 + 备份填充 | 较优 |
| LP Relaxation | 快速可行解 + 下界 | HiGHS 线性松弛 + 取整 | 近似 |
| Exact MIP | 中小实例 (< 32k 列) | scipy HiGHS 分支定界 | **最优（可证明）** |
| CP-SAT | 备选精确引擎 | OR-Tools 约束规划 | **最优（可证明）** |
| Column Generation | 密集/低意愿（列池爆炸） | LP 对偶定价按需生成列 | 较优–近似 |
| Heuristic | 超大实例兜底 | 纯 Python beam search + 局部搜索 | 较优 |
| LLM | 小实例、多样性 | LongCat 大模型直接推理 | 较优 |
| Perturb | 迭代改善 | Ruin-and-recreate 扰动精修 | 安全改善 |

### 目标函数

一个任务束（n 个任务）指派给一组骑手 S，期望成本：

```
fail_prob      = ∏(1 − wᵢ)
p_complete     = 1 − fail_prob
expected_score = Σ(wᵢ·sᵢ) / Σ(wᵢ)
group_cost(S)  = p_complete · expected_score + fail_prob · 100 · n
```

未覆盖任务惩罚 = 100 × 未覆盖数。总成本 = Σ 各组成本 + 未覆盖惩罚。

### 列生成（创新核心）

静态列池在密集实例上爆炸（每个任务束 × 骑手子集）。列生成按 LP 对偶价格按需定价：

```
循环：
  解 LP 松弛 → 得对偶价格 (λ_骑手, μ_任务)
  对每个任务束：找 reduced cost < 0 的骑手子集
     rc(S) = cost(S) − Σ λ_c − Σ μ_t
  加入池 → 重解 LP
直到无改进列 → 整数求解
```

---

## 实验结果

### 合成基准

| 实例 | 任务 | 候选数 | Agent 得分 | 胜出策略 | 用时 |
|------|:--:|------:|-----------:|----------|:--:|
| synth_tiny6 | 6 | 57 | 109.27（最优） | Exact MIP | < 1s |
| synth_small12 | 12 | 181 | 214.65（最优） | Exact MIP | ~ 1s |
| synth_medium20 | 20 | 497 | 290.75 | MIP + Perturb | ~ 3s |
| synth_dense28 | 28 | 1.8k | 371.31 | MIP + Perturb | ~ 1.5s |
| synth_lowdense22 | 22 | 24k 列 | 785.39 | ColGen + Heuristic | ~ 9s |
| synth_xdense35 | 35 | 13.6k | 326.46 | Greedy + Perturb | ~ 9s |

### Anytime 自终止行为

| 实例 | 实际耗时 | 终止原因 |
|------|:--:|------|
| tiny / small / medium | < 1.5s | 精确法证明最优 → 提前停 |
| dense / low-dense | 4–9s | 无法提前证明 → 用满预算搜索 |
| xdense / large | 9–10s | 大实例 → 耗尽预算 |

---

## 项目结构

```
autosolver/
├── autosolver/                    # Agent 核心包
│   ├── agent.py                   # UCB 自适应控制器（核心）
│   ├── problem.py                 # 问题建模（Candidate, group_cost）
│   ├── solution.py                # 解表示与目标函数评估
│   ├── columns.py                 # 列枚举与稀疏矩阵构建
│   ├── profile.py                 # 实例画像与特征工程
│   ├── memory.py                  # 跨实验学习记忆（JSON 持久化）
│   ├── pretrain.py                # 预训练策略先验
│   ├── runner.py                  # CLI 命令行入口
│   ├── demo.py                    # 自演展示脚本
│   ├── solvers/                   # 求解器库
│   │   ├── greedy.py              # 贪心构造（5 种排序 + 备份填充）
│   │   ├── milp.py                # 精确 MIP（HiGHS）
│   │   ├── cpsat.py               # 精确 CP-SAT（OR-Tools）
│   │   ├── colgen.py              # 列生成
│   │   ├── lp.py                  # LP 松弛
│   │   ├── heuristic.py           # 纯 Python 启发式（复用判题引擎）
│   │   └── llm.py                 # LLM 直接推理
│   ├── learning/                  # 学习子系统
│   │   ├── policy_model.py        # 上下文 Bandit 策略模型
│   │   ├── column_model.py        # 列质量评分模型
│   │   ├── backup_model.py        # 备份骑手评分模型
│   │   ├── operator_model.py      # 算子选择模型
│   │   ├── llm_advisor.py         # LongCat 参数顾问
│   │   ├── features.py            # 状态特征提取
│   │   ├── logger.py              # 学习日志记录
│   │   └── train.py               # 策略模型训练
│   └── web/                       # Web UI
│       ├── server.py              # Flask + SSE 实时流
│       └── static/index.html      # 单页应用前端
├── solver.py                      # 判题引擎（纯标准库，无第三方依赖）
├── datasets/                      # 测试用例
│   ├── synth_tiny6.txt
│   ├── synth_small12.txt
│   ├── synth_medium20.txt
│   ├── synth_dense28.txt
│   ├── synth_lowdense22.txt
│   ├── synth_xdense35.txt
│   ├── large_seed301.txt          # 赛方公开大用例
│   └── synthetic/                 # 20 个合成用例（多种画像）
├── Dockerfile                     # Docker 部署
├── DEPLOY.md                      # 部署说明（HF Spaces / 阿里云）
├── 技术报告.md                     # 完整技术报告
└── requirements.txt
```

---

## 创新点

1. **把"选算法"变成强化学习问题**：多数求解器是固定流水线，AutoSolver 用 **UCB 多臂老虎机 + 在线信用分配**让 Agent 自主决定下一步试什么，对零收益的策略**自动转向**

2. **跨实验迁移学习**：以「规模 × 密度 × 意愿」画像为键，历史策略表现持久化。配合预训练，Agent**出厂即带经验**，且每次运行都在学习

3. **列生成填补精确法盲区**：静态列池在密集/低意愿实例上爆炸，列生成按 LP 对偶价格按需生成列。实测将静态 MIP 的失败（2200）救回 789

4. **收尾精修修补列池漏洞**：受限列池上的"精确最优"并非全局最优，收尾精修用真实成本局部搜索补回遗漏的备份骑手，**安全（只改进不变坏）**

5. **Anytime + 提前最优证明**：能证明最优则秒返回、不浪费算力；不能则用满预算——兼顾速度与质量

6. **双交付物一体化**：判题环境用纯标准库 `solver.py`，完整 Agent 系统额外用 scipy/ortools/LLM，且前者被后者**当作一个策略复用**

7. **全程可观测**：Flask + SSE 把 Agent 的决策轨迹——UCB 选择、策略价值、Pivot 转向、学习记忆——**实时流到网页**，让 Agent 行为可见、可演示
