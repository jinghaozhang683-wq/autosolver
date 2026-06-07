---
title: AutoSolver Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AutoSolver Agent

一个自主、可学习的多策略求解 Agent，用于骑手调度优化（任务→快递员分配，
最大化接单、最小化总成本，10 秒/用例内）。

打开页面后选一个用例、点 **运行 Agent**，即可实时看到它：
**自主探索多种策略（贪心 / LP / 精确 MILP / CP-SAT / 列生成 / 启发式 / LLM）→
自动评估筛选 → 对表现差的策略转向 → 跨实验从历史中学习**。

- 算法与架构说明见 [`autosolver/README.md`](autosolver/README.md)
- 部署说明见 [`DEPLOY.md`](DEPLOY.md)

## 本地运行

```bash
pip install -r requirements.txt
python -m autosolver.web.server      # http://127.0.0.1:5000
```

## 命令行

```bash
python -m autosolver.runner CASE.txt --budget 10
python -m autosolver.demo            # 自带演示
python -m autosolver.pretrain        # 预训练策略先验
```
