#!/bin/bash
# Audio Library TTS Server 起動スクリプト

cd "$(dirname "$0")"

echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting Audio Library Server..."
echo "  URL: http://0.0.0.0:8001"
echo "  Stop: Ctrl+C"
echo ""

python audio_library_server.py
