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

**美团首届 AI Hackathon 赛道 4 参赛项目**

> 一个能自主探索策略、自动评估筛选、跨实验持续学习的 AI Agent——不是单一算法，而是会"自己选算法"的智能体。

---

## 问题背景

美团每天需要处理海量配送订单。每一次配送都是一次实时的「订单→骑手」匹配决策。赛道 4 要求设计一个 **AI Agent 系统**，面对任意给定的配送实例，在 **10 秒内** 自主完成求解。

问题本身是一个**带非线性期望成本的集合划分（Set Partitioning）问题**：

- 订单可以**合单配送**（bundle），骑手可以选择**是否接单**（willingness ∈ [0,1]）
- 一个任务可指派多名骑手做备份——第一个接单的获得订单
- 不分配的任务每个罚 100 分
- 每个任务至多被覆盖一次，每个骑手至多被使用一次

目标：**最大化接单数，最小化总成本，10 秒内返回结果。**

---

## 核心挑战

| 挑战 | 说明 |
|------|------|
| **非线性目标** | 多骑手备份下的期望成本为 `p_complete × 期望分数 + fail_prob × 100 × n`，不能直接用线性模型 |
| **组合爆炸** | 枚举所有 `(任务束, 骑手子集)` 的列，搜索空间指数级 |
| **规模差异巨大** | 从 6 个任务 × 57 条候选 → 40 个任务 × 33,780 条候选，同一套策略无法通吃 |
| **10 秒硬约束** | 决策时间极度紧张，必须在精度与速度间实时权衡 |

---

## AutoSolver 架构

```
                         ┌─────────── AutoSolverAgent (核心) ────────────┐
                         │                                              │
  实例 ──► 画像分析 ──► UCB 多臂老虎机选策略 ──► 真实目标评估 ──► 保留最优 │
                         │        ▲              │             │        │
                         │        └─ 信用分配 ◄──┘             │        │
                         │        └─ 跨实验记忆 ◄─── 学习 ◄────┘        │
                         │                                              │
                         │  ┌──────────────────────────────────────┐    │
                         │  │           7 类求解策略                 │    │
                         │  │  Greedy │ MIP │ CP-SAT │ ColGen      │    │
                         │  │  LP     │ Heuristic  │ LLM | Perturb │    │
                         │  └──────────────────────────────────────┘    │
                         └─────────────────────────────────────────────┘
```

**一句话**：Agent 自动分析实例特征 → 用多臂老虎机选择该试哪个策略 → 跑出结果按真实目标评分 → 好的保留，差的转向 → 写进记忆，下次同类实例直接从经验出发。

---

## 快速开始

```bash
# 安装
git clone https://github.com/jinghaozhang683-wq/autosolver.git
cd autosolver
pip install scipy numpy flask

# 启动网页（推荐）
python -m autosolver.web.server          # → http://127.0.0.1:5000

# 命令行
python -m autosolver.runner datasets/synth_medium20.txt --budget 10
python -m autosolver.demo                # 自带演示
```

网页中可实时观看 Agent 的**完整决策轨迹**——选了什么策略、得了多少分、为什么转向、学到了什么。

---

## 核心设计

### 1. 不是固定流水线，而是自主决策

大多数求解器是固定流程（先 A 后 B 再 C）。AutoSolver 不同——它把"选算法"本身变成了一个**在线学习问题**。

```python
# 7 个策略臂，每个有自己的价值估计
moves = {
    "greedy":      0.05,    # 贪心基线
    "heuristic":   0.92,    # 纯 Python 启发式
    "exact":       0.80,    # 精确 MIP（可达最优）
    "colgen":      0.75,    # 列生成
    "lp":          0.25,    # LP 松弛
    "perturb":     0.30,    # 扰动精修
    "llm":         0.35,    # LLM 推理
}

# UCB 选臂：价值 + 探索奖励 → 平衡"用好的"和"试新的"
m = argmax(value + c · √(ln N / (n+1)))
```

### 2. 策略在线信用分配与自动转向

```python
# 改进除以耗时——快速有效的策略获得更高奖励
reward = (old_best - new_best) / baseline / max(0.1, time_spent)

# 连续多次零改进 → 自动弃用（pivot），转向其他策略
if m.stalled_consecutive >= 4:
    m.exhausted = True   # "这个方向没前途，不试了"
```

### 3. Anytime 预算 + 提前停止

```python
while time_left > 0.4:
    run_best_strategy()
    if proved_optimal:      # 精确法证最优 → 提前停
        break
return best_so_far          # 随时可中断，返回当前最优
```

| 实例 | 实际耗时 | 为什么停了 |
|------|:--:|------|
| tiny (6 任务) | 0.7s | 策略全探索完 |
| small/medium (12–20 任务) | 1.5–4s | **MIP 证明最优** → 提前停 |
| xdense (35 任务) | 9.2–9.9s | 无法提前证明 → 用满预算 |

---

## 7 类求解策略

| 策略 | 原理 | 擅长场景 | 时长 |
|------|------|----------|:--:|
| **Greedy** | 5 种排序优先级 + 备份骑手，亚秒级 | 快速基线、简单实例 | < 0.1s |
| **LP 松弛** | HiGHS 线性松弛 + 取整 | 快速可行解 + 下界 | < 0.2s |
| **Exact MIP** | scipy HiGHS 分支定界，可证最优 | 中小实例（< 32k 列）| 1–5s |
| **CP-SAT** | OR-Tools 约束规划 | 备选精确引擎 | 1–5s |
| **列生成** | LP 对偶定价**按需生成列** | 密集/低意愿（列池爆炸区间）| 3–8s |
| **Heuristic** | 纯 Python beam search + 局部搜索 | 超大实例兜底 | 0.5–9s |
| **Perturb** | Ruin-and-recreate 扰动精修 | 安全改善（只改进不变坏）| 0.6s |

### 列生成 —— 精确法的"救火队长"

静态列池枚举所有「任务束 × 骑手子集」。在密集实例上，列数爆炸但 MIP 求解器处理不了。列生成换一种思路：

```
循环：
  解 LP 松弛 → 拿对偶价格 (λ_骑手 ≤ 0, μ_任务 自由)
  为每个任务束找"划算的"骑手子集：
    reduced_cost = group_cost − Σ λ − Σ μ < 0  → 加入池
  重解 LP
直到无新列 → 对生成的列做整数求解
```

**实测**：在一个 24,000 列的密集实例上，静态 MIP 返回失败（2200），列生成救回到 789。

---

## 跨实验学习

```
实例 → 画像键（规模/密度/意愿）→ 查历史 → 初始化策略价值 → 求解 → 写回
```

- 每次求解后自动更新 `trained_memory.json`
- 新实例启动时，引用同画像下的**历史统计加权初始化**策略先验
- `pretrain.py` 在 **21 种画像 × 多轮合成实例**上预训练，Agent **出厂即带经验**

学到的策略分工（示例）：

| 画像 | 最优策略 | 为什么 |
|------|----------|------|
| tiny / small / sparse | Exact MIP | 小实例可证最优 |
| dense / lowwill | Column Generation | 静态列池爆炸 |
| xlarge / dense | Heuristic | 唯一的可解策略 |

---

## 实验结果

### 合成基准

| 实例 | 任务 | 候选 | 得分 | 策略 | 用时 |
|------|:--:|------:|-----:|------|:--:|
| synth_tiny6 | 6 | 57 | **109.27** | Exact MIP | < 1s |
| synth_small12 | 12 | 181 | **214.65** | Exact MIP | ~ 1s |
| synth_medium20 | 20 | 497 | **290.75** | MIP + Perturb | ~ 3s |
| synth_dense28 | 28 | 1,800 | **371.31** | MIP + Perturb | ~ 1.5s |
| synth_lowdense22 | 22 | 24,000 列 | **785.39** | ColGen | ~ 9s |
| synth_xdense35 | 35 | 13,641 | **326.46** | Greedy + Perturb | ~ 9s |
| large_seed301 | 40 | 33,780 | **658.64** | Heuristic | ~ 10s |

### 策略间效果对比

| 场景 | 策略 A | 策略 B | 差距 |
|------|--------|--------|:--:|
| 密集实例 (24k 列) | 静态 MIP: 2200（失败） | 列生成: 789 | **2.8×** |
| 密集实例 (13.6k 候选) | 启发式: 513 | 列生成: 329 | **1.6×** |
| medium 精修前后 | MIP 291.35 | +Perturb 290.75 | 0.6 分改善 |
| dense 精修前后 | MIP 374.34 | +Perturb 371.31 | 3.0 分改善 |

> Perturb 的改善虽小但**只改进不变坏**（strictly non-worsening），在竞赛中每一分都算数。

### 判题引擎

`solver.py` 是纯标准库（无第三方依赖）的单文件引擎，可直接提交判题系统。在 hidden test 上的历史最好成绩约为 **711–713 分**级别。

---

## 项目结构

```
autosolver/
├── autosolver/                    # Agent 核心包
│   ├── agent.py                   # UCB 自适应控制器（核心，~430 行）
│   ├── problem.py                 # 问题建模、输入解析
│   ├── solution.py                # 解表示与目标函数评估
│   ├── columns.py                 # 列枚举与稀疏矩阵
│   ├── profile.py                 # 实例画像（规模/密度/意愿 + 10+ 特征）
│   ├── memory.py                  # 跨实验学习（JSON 持久化）
│   ├── pretrain.py                # 预训练脚本
│   │
│   ├── solvers/                   # 求解器库
│   │   ├── greedy.py              # 贪心构造
│   │   ├── milp.py                # 精确 MIP (scipy HiGHS)
│   │   ├── cpsat.py               # 精确 CP-SAT (OR-Tools)
│   │   ├── colgen.py              # 列生成
│   │   ├── lp.py                  # LP 松弛
│   │   ├── heuristic.py           # 纯 Python 启发式
│   │   └── llm.py                 # LLM 推理
│   │
│   ├── learning/                  # 学习增强子系统
│   │   ├── policy_model.py        # 上下文 Bandit 策略预测
│   │   ├── column_model.py        # 列质量评分
│   │   ├── backup_model.py        # 备份骑手评分
│   │   ├── operator_model.py      # 算子选择
│   │   ├── llm_advisor.py         # LLM 参数顾问
│   │   └── train.py               # 离线训练
│   │
│   └── web/                       # Web UI
│       ├── server.py              # Flask + SSE 实时决策流
│       └── static/index.html      # 单页应用
│
├── solver.py                      # 判题引擎（纯标准库，单文件）
├── datasets/                      # 测试用例（8 个内置 + 20 个合成）
├── Dockerfile                     # Docker 部署
└── 技术报告.md                     # 完整技术报告
```

---

## Python API

```python
from autosolver import Problem
from autosolver.agent import AutoSolverAgent

problem = Problem.parse(open("case.txt").read())
agent = AutoSolverAgent(use_cpsat=False, use_llm=False, verbose=True)
solution = agent.solve(problem, total_budget=10.0)

print(solution.score)           # 总成本
print(solution.to_output())     # [(task_key, [courier_ids]), ...]

# 查看 Agent 每一步决策
for entry in agent.log:
    print(entry)
```

---

## 创新点

1. **把"选算法"变成在线学习问题**：UCB 多臂老虎机自主决定下一步试什么，在线信用分配让快且有效的策略获得更高奖励——"自主探索"不是口号，是可观测的决策行为

2. **跨实验迁移学习**：按画像索引历史表现，新实例引用历史先验。配合预训练，Agent 出厂即带经验，且每次运行都在更新

3. **列生成填补精确法盲区**：静态列池在大密集实例上爆炸，按 LP 对偶价格按需定价生成列，把静态 MIP 的"失败"救回可用的解

4. **收尾精修修补列池漏洞**：受限于列池规模的 MIP"最优"并非全局最优——扰动精修用真实成本下的增减移换局部搜索，安全地（只改进不变坏）补回遗漏

5. **Anytime + 提前最优证明**：中小实例证最优后秒返回、不浪费算力；大实例用满预算搜索

6. **双交付物一体化**：纯标准库 `solver.py` 可独立提交判题系统，同时被完整 Agent 当作一个策略复用

7. **全程可观测**：Flask + SSE 实时流将 Agent 的决策轨迹、策略价值、转向逻辑、学习记忆展现在网页上——让"agent 味"可见、可交互

---

## 部署

支持 Docker 一键部署。详见 [`DEPLOY.md`](DEPLOY.md)。

```bash
docker build -t autosolver .
docker run -p 7860:7860 autosolver    # → http://localhost:7860
```

在线演示：`http://116.62.44.72:5000`（阿里云 ECS 部署）
