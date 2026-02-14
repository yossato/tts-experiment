#!/bin/bash
# Simple Batch TTS Server 起動スクリプト

cd "$(dirname "$0")"

echo "🔧 仮想環境をアクティベート中..."
source venv/bin/activate

echo "🚀 TTS サーバーを起動中..."
echo "   URL: http://localhost:8000"
echo "   停止: Ctrl+C"
echo ""

python app.py
