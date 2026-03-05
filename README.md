# 知识库 GPT 聊天 Demo

一个基于 **FastAPI + LangGraph + DeepSeek + 智谱 Web Search + 本地知识库 RAG** 的简单聊天 Demo。

## 目录结构

```text
backend/
  app/
    api/chat.py        # 聊天 API
    config.py          # 配置与环境变量
    graph/             # LangGraph 流程
    rag/               # 知识库 RAG 组件
    main.py            # FastAPI 入口
  scripts/ingest.py    # 知识库导入脚本
  data/                # 放置你的 txt/md 文档
frontend/
  index.html           # 简易聊天页面
  app.js
  style.css
```

## 环境准备

```bash
cd backend
cp .env.example .env  # 填写 DEEPSEEK_API_KEY / ZHIPUAI_API_KEY
pip install -r requirements.txt
```

## 向量数据库（Chroma）Docker 部署

向量库默认通过 Docker 运行，需先启动 Chroma 容器：

```bash
docker compose up -d chroma
```

Chroma 将监听 `localhost:8001`。在 `backend/.env` 中已配置：

```
CHROMA_HOST=localhost
CHROMA_PORT=8001
```

若未设置 `CHROMA_HOST`，则使用本地目录 `CHROMA_PERSIST_DIR` 作为嵌入式存储。

## 导入知识库

将 `.txt` / `.md` 文档放入 `backend/data/`，然后：

```bash
cd backend
python -m scripts.ingest
```

（需确保 Chroma 已启动，或使用本地模式）

## 启动后端

**方式一：使用启动脚本（推荐）**

```bash
# 1. 复制并填写环境变量
cp backend/.env.example backend/.env
# 编辑 backend/.env，填入 DEEPSEEK_API_KEY 和 ZHIPUAI_API_KEY

# 2. 运行脚本（会自动加载 .env 并启动）
./scripts/start.sh
```

**方式二：手动启动**

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 打开前端

直接用浏览器打开 `frontend/index.html`，或用任意静态服务器（如 VSCode Live Server）打开该目录。

