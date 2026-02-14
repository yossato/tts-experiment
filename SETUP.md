# セットアップガイド

## クイックスタート（5分で開始）

### 1. リポジトリのクローン
```bash
git clone --recursive git@github.com:yossato/tts-experiment.git
cd tts-experiment
```

### 2. 仮想環境の作成とパッケージインストール
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. システムライブラリのインストール（音声再生用）
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
```

### 4. 実行テスト
```bash
# シンプルバッチ処理版
python simple_batch_tts.py

# Webサーバー版
./run_server.sh
# ブラウザで http://localhost:8000 にアクセス

# ストリーミング版
python streaming_tts.py
```

## トラブルシューティング

### GPU メモリ不足
```python
# streaming_tts.py の batch_size を調整
generator = StreamingTTSGenerator(batch_size=5)  # 10→5
```

### submodule のエラー
```bash
git submodule update --init --recursive
```

### CUDA が見つからない
```bash
# PyTorch の CUDA バージョンを確認
python -c "import torch; print(torch.cuda.is_available())"

# 必要に応じて PyTorch を再インストール
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 次のステップ

- [README.md](README.md) で詳細な使い方を確認
- サンプルテキストを編集してカスタマイズ
- Web UI で様々な話者を試す
