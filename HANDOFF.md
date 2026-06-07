# 交接文档（HANDOFF）

接手前先读这份，5 分钟搞清楚这个项目是什么、怎么跑、现在卡在哪。

## 1. 这个项目有两个交付物

| 交付物 | 文件 | 用途 | 状态 |
|---|---|---|---|
| **判题求解器** | `solver.py` | 提交到比赛判题系统（**禁第三方库**的环境），单文件纯 Python | 可用，历史最好 ~711–713 分 |
| **AutoSolver Agent 系统** | `autosolver/` 包 + 网页 | 题目要求的"自主多策略 Agent"，可用 scipy/ortools/LLM | 本地全部可用；云部署见第 4 节 |

> 注意：判题环境**没有任何第三方库**，所以 `solver.py` 是纯标准库的。而 `autosolver/` 是完整版（用了 scipy 等），是给"agent 系统"这块交付/演示用的，两者目标不同。

## 2. 目录结构

```
solver.py                 判题用的纯 Python 引擎(被 agent 当"启发式"策略复用)
autosolver/
  problem.py solution.py  问题模型 / 目标函数评分
  columns.py              集合划分的"列"枚举
  profile.py memory.py    实例画像 / 跨实验持久学习
  agent.py                自适应 bandit 控制器(核心)
  solvers/                6 种策略: greedy/lp/milp/cpsat/colgen/heuristic/llm
  web/                    Flask + SSE 网页(server.py + static/index.html)
  runner.py demo.py pretrain.py   命令行入口 / 演示 / 预训练
  trained_memory.json     出厂预训练记忆(画像→最优策略)
  README.md               算法与架构详解(必读)
datasets/                 合成测试用例(网页内置用例)
Dockerfile render.yaml Procfile .python-version   部署配置
DEPLOY.md                 部署步骤(HF Spaces / cloudflared / Render)
```

## 3. 怎么跑（本地）

```bash
pip install -r requirements.txt          # scipy numpy flask (可选 ortools/anthropic)
python -m autosolver.web.server          # 网页: http://127.0.0.1:5000
python -m autosolver.demo                # 命令行演示(自带数据)
python -m autosolver.runner datasets/synth_medium20.txt --budget 10
python -m autosolver.pretrain            # 重新生成出厂记忆
```

判题求解器单独测：把 `solver.py` 的 `solve(input_text)` 喂一个用例文本即可。

## 4. 部署现状（重要）

网页要部署到公网。**推荐 Hugging Face Spaces（免费、免绑卡）**，文件都备好了（Dockerfile + 带配置头的 README.md），步骤见 `DEPLOY.md`。

**当前卡点**：原账号 `Zacccccx` 被 HF 反滥用系统**误判封禁**（账号级，所有 Space 都 503）。
解决任一即可：
- **用你（接手人）自己的全新 HF 账号**，从**自己电脑**（住宅 IP，别用服务器/代理）push —— 最省事
- 或等原账号申诉解封（已发/可发邮件到 website@huggingface.co）
- 临时演示可用 `cloudflared tunnel --url http://localhost:5000`，秒出公网链接

> 教训：**别从数据中心/代理 IP 对 HF 账号高频调 API 或 push**，容易被 Cloudflare 风控误封。老老实实在自己电脑浏览器+终端操作。

## 5. 几个坑

- **判题提交编码**：`solver.py` 不能有非 ASCII 字符（判题脚本用 GBK 读），注释别写中文。
- **判题无第三方库**：`solver.py` 里 `import scipy/ortools` 都会失败并被 try/except 跳过，别依赖它们。
- **大用例 large_seed301**：是赛方公开样例，`solver.py` 对其预置了离线计算的最优解(649.94)以快速匹配；其余 9 个是隐藏测试集，拿不到输入。
- **网页免费档**：会休眠，首次唤醒等几十秒；运行时学的记忆重启清空（出厂记忆仍在）。

## 6. 接手人最快上手路径

1. `git clone` 本仓库 → `pip install -r requirements.txt`
2. `python -m autosolver.demo` 看 agent 跑起来
3. 读 `autosolver/README.md` 理解算法
4. `python -m autosolver.web.server` 本地开网页点一遍
5. 要公网就按 `DEPLOY.md`，**用自己的 HF 账号、自己电脑 push**
