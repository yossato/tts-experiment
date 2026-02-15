# TTS実験プロジェクト

Qwen3-TTS-12Hz-1.7B-CustomVoiceを使った高速音声合成の実験プロジェクト

## 🎯 プロジェクト概要

長文テキストの音声合成において、メモリ効率と生成速度を最適化する手法を検証。
テキスト分割 + バッチ処理により、RTF < 1.0 かつメモリ枯渇を防ぐ実装を実現。

## 📊 主な成果

### 1. シンプルバッチ処理版 (`simple_batch_tts.py`)
- **RTF: 0.37** - リアルタイムより2.7倍高速
- Qwen3-TTS公式バッチ処理のみを使用
- asyncio/Semaphore等の複雑な制御なし
- GPU利用率: ~100%

### 2. FastAPI Webサーバー版 (`app.py`)
- REST API + Web UIを提供
- ブラウザから簡単に音声生成
- 処理統計をリアルタイム表示
- 複数話者対応（日本語・英語・中国語）

### 3. ストリーミング風生成＆再生版 (`streaming_tts.py`)
- **RTF: 0.42** - リアルタイムより2.4倍高速
- メモリ枯渇防止：10チャンクずつバッチ処理
- 生成しながら順次再生（自転車操業方式）
- マルチスレッド：生成と再生が並行動作
- 長文対応：夏目漱石「吾輩は猫である」で検証成功

## 🛠️ 環境構築

### システム要件
- Ubuntu 22.04 / 24.04
- Python 3.10+
- NVIDIA GPU (CUDA対応)
  - 推奨: RTX 3080 10GB以上
  - FlashAttention 2対応GPU
- Git

### 1. リポジトリのクローン

```bash
git clone --recursive git@github.com:yossato/tts-experiment.git
cd tts-experiment
```

### 2. Python仮想環境の作成

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 依存パッケージのインストール

#### Python パッケージ
```bash
pip install -r requirements.txt
```

**requirements.txt の内容:**
```
fastapi
uvicorn[standard]
numpy
torch
soundfile
qwen-tts
sounddevice  # 音声再生用（オプション）
requests     # APIクライアント用
```

#### システムライブラリ（音声再生用）
```bash
# PortAudio（sounddeviceの依存）
sudo apt-get update
sudo apt-get install -y portaudio19-dev

# 音声再生パッケージ
pip install sounddevice
```

## 🚀 使い方

### 1. シンプルバッチ処理（コマンドライン）

```bash
python simple_batch_tts.py
```

**特徴:**
- 最もシンプルな実装
- 高速処理（RTF 0.37）
- スクリプト内のテキストを編集して使用

### 2. Webサーバー（ブラウザUI）

```bash
# サーバー起動
./run_server.sh

# ブラウザでアクセス
# http://localhost:8000
```

**特徴:**
- ブラウザから操作可能
- 複数話者選択
- 処理統計のリアルタイム表示
- 生成音声をその場で再生

**API経由での使用:**
```bash
# Pythonクライアント
python api_client.py "こんにちは、今日は良い天気ですね。" output.wav

# curlコマンド
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "こんにちは", "speaker": "Ono_Anna", "language": "Japanese"}' \
  --output output.wav
```

### 3. ストリーミング風生成＆再生（長文対応）

```bash
python streaming_tts.py
```

**特徴:**
- 長文でもメモリ枯渇しない
- 生成しながら順次再生
- 待ち時間を最小化
- 10チャンクずつバッチ処理

**カスタマイズ:**
```python
generator = StreamingTTSGenerator(
    batch_size=10,  # 一度に処理するチャンク数（メモリに応じて調整）
)

generator.generate_and_play(
    text="長文テキスト...",
    max_chars=50,        # チャンク分割サイズ
    enable_playback=True # リアルタイム再生
)
```

### 4. MCP統合版（VS Code + macOSメニューバー）

**構成:**
```
VS Code (MCP Client)
  ↓
mcp_server.py (FastMCP)
  ↓ subprocess
tts_player.py (rumps - macOS Menu Bar App)
  ↓ SSE Stream
audio_library_server.py (FastAPI on Ubuntu GPU Server)
```

**サーバー起動（Ubuntu）:**
```bash
./run_library_server.sh
```

**VS Code MCP設定（`~/.config/Code/User/mcp.json`）:**
```json
{
  "mcpServers": {
    "tts": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "TTS_SERVER_URL": "http://192.168.1.99:8001"
      }
    }
  }
}
```

**機能:**
- ✅ VS Code GitHub Copilotから直接TTS呼び出し
- ✅ macOSメニューバー常駐プレイヤー
- ✅ ストリーミング再生（生成中に再生開始）
- ✅ Pause/Resume/Stop制御
- ✅ リアルタイム速度変更（0.5x～2.0x）
- ✅ ピッチ保持（音程は変わらない）
- ✅ マルチチャンク対応（18チャンク検証済み）

**使い方:**
1. VS Code で GitHub Copilot Chat を開く
2. `@tts` を入力してツールを呼び出す
3. テキストを入力
4. メニューバーにTTSアイコンが表示され、音声が再生される
5. メニューから速度変更・Pause/Resume/Stopが可能

## 📁 ファイル構成

```
.
├── simple_batch_tts.py         # シンプルバッチ処理版
├── streaming_tts.py            # ストリーミング風生成＆再生版
├── app.py                      # FastAPI Webサーバー（基本版）
├── audio_library_server.py     # FastAPI Webサーバー（MCP統合版）
├── mcp_server.py               # MCP Server（FastMCP）
├── tts_player.py               # macOSメニューバープレイヤー（rumps）
├── api_client.py               # APIクライアントサンプル
├── run_server.sh               # サーバー起動スクリプト（基本版）
├── run_library_server.sh       # サーバー起動スクリプト（MCP統合版）
├── requirements.txt            # Python依存パッケージ
├── EXPERIMENTS.md              # 実験ログ・技術詳細
├── SETUP.md                    # セットアップガイド
├── .gitignore                  # Git除外設定
└── Qwen3-TTS/                  # Qwen3-TTS サブモジュール
```

## 🧪 実験結果

### テスト環境
- GPU: NVIDIA GeForce RTX 3080 (10GB)
- OS: Ubuntu 24.04
- Python: 3.12
- CUDA: 12.x
- モデル: Qwen3-TTS-12Hz-1.7B-CustomVoice

### 性能比較

| 方式 | RTF | GPU利用率 | メモリ効率 | 待ち時間 |
|------|-----|-----------|------------|----------|
| シンプルバッチ | 0.37 | ~100% | 高（一括処理） | 全生成完了まで待機 |
| Webサーバー | 0.76 | ~60% | 中（リクエスト単位） | リクエストごと |
| ストリーミング | 0.42 | ~100% | 高（分割バッチ） | 最小（順次再生） |

### 長文テスト結果（夏目漱石「吾輩は猫である」冒頭493文字）

**ストリーミング版:**
- 総チャンク数: 13
- バッチ数: 2（10 + 3チャンク）
- 生成時間: 38.28秒
- 音声長: 91.04秒
- **RTF: 0.42**
- スループット: 12.9文字/秒
- **メモリエラーなし**

## 💡 技術的知見

### 1. バッチ処理の重要性
Qwen3-TTSのネイティブバッチ処理（`text=[複数]`で渡す）が、個別処理より圧倒的に高速。
- GPU利用率が100%近くに達する
- メモリコピーのオーバーヘッドが最小化

### 2. メモリ管理戦略
長文（1000文字以上）では一度に全チャンクをバッチ処理すると OOM エラー。
→ 10チャンクずつに分割してバッチ処理することで解決

### 3. 自転車操業方式の有効性
生成スレッドと再生スレッドを分離し、キューで連携。
- RTF < 1.0 なら、再生中に次のバッチを生成できる
- 100秒の音声でも、最初の音は20秒以内に聞こえ始める

### 4. ストリーミング機能の現状
Qwen3-TTSモデル自体はストリーミングをサポート（97msレイテンシー）しているが、
現在のPython APIでは真のストリーミング出力は未実装。
- `non_streaming_mode=False` は「シミュレート」のみ
- 真のストリーミングはDashScope API（商用）またはvLLM-Omni（今後対応）で可能

### 5. TCP Buffer Saturationの回避
SSEで大きなデータ（Base64エンコード音声 ~500KB）を送信すると、TCPバッファが満杯になり`yield`が長時間ブロックする。
- **問題**: 2チャンク目以降のyieldが55秒以上ブロック
- **解決**: 音声を一時ファイルに保存し、SSEではURL（数百バイト）のみ送信
- **効果**: Web UIフリーズ解消、マルチチャンク安定配信

### 6. GUIスレッド制約への対処
macOSのUIフレームワーク（rumps）ではバックグラウンドスレッドからのUI更新は禁止。
- **問題**: `self.title`をバックグラウンドスレッドから更新するとクラッシュ
- **解決**: メインスレッドのタイマー（0.2秒間隔）で間接的に更新
- **実装**: `_pending_title`変数 + `rumps.Timer`

### 7. 速度変更のタイミング
ストリーミングではダウンロードが再生より先に進むため、速度変更は「再生時」に適用する必要がある。
- **ダウンロード時適用**: 先読みバッファのため変更が反映されない
- **再生時適用**: リアルタイムで速度変更が反映

### 8. ピッチ保持の実装
librosaの`time_stretch`（フェーズボコーダーベース）を使用することで、速度変更時にピッチ（音程）を保持。
- **scipy.signal.resample**: 速度変更できるがピッチも変わる
- **librosa.effects.time_stretch**: ピッチ保持で速度変更（推奨）

## 🔧 トラブルシューティング

### CUDA Out of Memory エラー
```python
# streaming_tts.py の batch_size を減らす
generator = StreamingTTSGenerator(
    batch_size=5,  # 10 → 5 に削減
)
```

### 音声が再生されない
```bash
# システムの音声デバイスを確認
python -m sounddevice

# デフォルトデバイスを変更（必要に応じて）
export SDL_AUDIODRIVER=pulseaudio
```

### FlashAttention警告
```
You are attempting to use Flash Attention 2 without specifying a torch dtype.
```
→ 警告のみで動作に問題なし。bfloat16で正しく動作している。

### Web UIの「Streaming」ボタンで音声が聞こえない
**現状の動作**:
- 「Streaming」ボタンは生成進捗をリアルタイム表示するが、音声の自動再生は行わない
- 音声はライブラリに保存され、完了後に手動で再生する必要がある

**回避策**:
- **方法1**: 生成完了後、ライブラリ一覧から手動で再生
- **方法2**: リアルタイム再生が必要な場合は、MCP統合版（`tts_player.py`）を使用

詳細は[EXPERIMENTS.md](EXPERIMENTS.md#既知の制約課題)を参照。

## 📝 今後の展望

- [ ] vLLM-Omni統合（真のストリーミング対応）
- [ ] バッチサイズの動的調整（GPU空きメモリに応じて）
- [ ] 複数GPUサポート
- [ ] 音声品質の詳細評価
- [ ] 他言語での性能検証
- [ ] Web UIストリーミング再生機能の実装

## 📚 参考資料

- [Qwen3-TTS公式リポジトリ](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS技術ブログ](https://qwen.ai/blog?id=qwen3tts-0115)
- [Qwen3-TTS論文](https://arxiv.org/abs/2601.15621)
- [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
- [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [rumps - macOS Menu Bar Apps](https://github.com/jaredks/rumps)
- [librosa - Audio Analysis](https://librosa.org/doc/latest/index.html)
- [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

## 📄 ライセンス

このプロジェクトのコードはMITライセンス。
Qwen3-TTSモデルは[公式ライセンス](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE)に従います。

## 👤 著者

Yoshiaki Sato

## 🙏 謝辞

- Qwen Team による素晴らしいTTSモデルの提供
- FlashAttention の開発チーム
