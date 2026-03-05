#!/bin/bash
# 加载 .env 并启动后端服务
# 用法: ./scripts/start.sh  或  bash scripts/start.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

if [ ! -f .env ]; then
  echo "未找到 .env 文件"
  echo "请执行: cp backend/.env.example backend/.env"
  echo "然后编辑 backend/.env，填入 DEEPSEEK_API_KEY 和 ZHIPUAI_API_KEY"
  exit 1
fi

# 加载 .env 到当前 shell（兼容含空格的值）
set -a
. ./.env
set +a

echo "已加载环境变量，启动后端..."
exec uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
