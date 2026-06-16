#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip

if ! python -c "import flask, PIL" >/dev/null 2>&1; then
  pip install -r requirements.txt
fi

if ! python -c "import paddleocr" >/dev/null 2>&1; then
  echo "[提示] 还没安装 PaddleOCR，请先安装 PaddlePaddle 和 PaddleOCR："
  echo "python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/"
  echo "pip install paddleocr"
  read -p "按回车退出..."
  exit 1
fi

echo "正在启动应用..."
python app.py
