#!/bin/bash
# 启动 Chroma 向量数据库 Docker 容器
# 用法: ./scripts/start-chroma.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "启动 Chroma 容器..."
docker compose up -d chroma

echo "Chroma 已启动，监听 localhost:8001"
echo "确保 backend/.env 中已设置: CHROMA_HOST=localhost CHROMA_PORT=8001"
