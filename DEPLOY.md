# 部署 AutoSolver Agent 网页

## 方案对比（国内优先看"免绑卡"）

| 平台 | 免绑卡 | 中国可用 | 公网URL | 说明 |
|---|---|---|---|---|
| **Hugging Face Spaces** | ✅ 不要卡 | ✅ 邮箱注册 | ✅ 永久 | **推荐**，Docker 跑 Flask，免费 2核16G |
| cloudflared 隧道 | ✅ 无需账号 | ✅ | 临时 | 1 分钟，但你电脑要一直开着 |
| Render | ❌ 要绑卡(Stripe) | ❌ 银联基本不行 | ✅ | 国内绑卡难 |
| Railway | ❌ 要验证 | ⚠️ | ✅ | 同上 |

---

## 推荐：Hugging Face Spaces（免费、免绑卡）

仓库里已备好 `Dockerfile` 和带配置头的根 `README.md`（`sdk: docker, app_port: 7860`）。

1. 打开 https://huggingface.co → 邮箱注册/登录（国内可直接注册）。
2. 右上头像 → **New Space**。
3. 填：
   - **Space name**：`autosolver-agent`（随意）
   - **SDK**：选 **Docker** → **Blank**
   - 可见性：Public
4. 创建后，把本项目文件推到这个 Space 的 git 仓库（HF 会给你仓库地址）：
   ```
   git init && git add -A && git commit -m "autosolver agent"
   git remote add hf https://huggingface.co/spaces/<你的用户名>/autosolver-agent
   git push hf main
   ```
   （首次 push 要输 HF 用户名 + Access Token，在 HF 设置里生成。）
5. HF 自动读 `Dockerfile` 构建，几分钟后页面就上线，网址形如
   `https://<你的用户名>-autosolver-agent.hf.space`。

> 也可以网页直接拖拽上传文件，不用命令行。

---

## 最快：cloudflared 临时隧道（无需任何账号）

适合"现在就要个链接给人看"，但你的电脑要一直开着、关了就失效。

1. 下载 cloudflared（https://github.com/cloudflare/cloudflared/releases ，Windows 选 `.exe`）。
2. 本地起服务 + 起隧道：
   ```
   python -m autosolver.web.server                       # 一个终端
   cloudflared.exe tunnel --url http://localhost:5000     # 另一个终端
   ```
3. 它会打印一个 `https://xxx.trycloudflare.com` 公网网址，发给谁都能开。

---

## Render / Railway（要绑卡，国内不便）

仓库也带了 `render.yaml` / `Procfile` / `.python-version`，若你有可用的境外
信用卡：Render.com → New + → Blueprint → 选仓库即可。详见各平台文档。

---

## 本地 / 局域网

```
python -m autosolver.web.server                 # http://127.0.0.1:5000
python -m autosolver.web.server --host 0.0.0.0   # 局域网: http://<本机IP>:5000
```

## 免费档通用注意

- **休眠**：HF Spaces free 长时间无访问会休眠，首次唤醒等几十秒。
- **临时文件系统**：运行时学到的记忆重启清空——但仓库自带**出厂预训练记忆**
  `autosolver/trained_memory.json`，画像先验始终在。
- **依赖**：部署用精简依赖（scipy+numpy+flask），默认 agent（MILP+列生成）已够强；
  开 `--cpsat` 需 OR-Tools，体积大，免费档可不装。
