# Audio Library TTS System

MCP連携対応のTTS音声ライブラリシステム。Claude Code等のAIツールからテキストを音声化し、即座に再生したり、サーバーに保存して後から聴き直すことができる。

## アーキテクチャ

```
[MacBook]
Claude Code ──stdio──> mcp_server.py ──HTTP──> [Ubuntu] audio_library_server.py:8001
                            │                              │
                            └── tts_player.py              ├── TTS Model (GPU)
                                (メニューバー再生)          ├── Web UI (ライブラリ)
                                                           └── audio_library/ (永続保存)

[スマホ/PC] ──ブラウザ──> [Ubuntu] http://<ip>:8001
```

| コンポーネント | 動作環境 | 役割 |
|---|---|---|
| `audio_library_server.py` | Ubuntu (GPU) | TTS音声生成、ライブラリ保存、Web UI |
| `mcp_server.py` | MacBook | MCPツール提供 (`read_aloud` / `generate_audio`) |
| `tts_player.py` | MacBook | メニューバー常駐の音声プレイヤー |

## ユースケース

**リアルタイム読み上げ** — Claude Codeで作業中にテキストを即座に読み上げ。メニューバーから一時停止/停止が可能。

**保存して後で聴く** — ディープリサーチの結果やWebページの内容を音声化してサーバーに保存。後からスマホでアクセスして再生・ダウンロード。

## セットアップ

### Ubuntu側 (音声生成サーバー)

```bash
cd /path/to/asr-tts
./run_library_server.sh
```

ポート8001で起動する。既存の `app.py` (ポート8000) とは独立しており、GPUメモリの都合上同時起動は不可。

### MacBook側 (MCPクライアント)

1. `mcp_server.py` と `tts_player.py` をMacBookにコピー

2. 依存パッケージをインストール
```bash
pip install "mcp[cli]" rumps sounddevice soundfile httpx numpy
```

3. Claude Codeの設定ファイルにMCPサーバーを追加

`~/.claude/claude_code_config.json`:
```json
{
  "mcpServers": {
    "tts": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "TTS_SERVER_URL": "http://<ubuntu-ip>:8001"
      }
    }
  }
}
```

## MCPツール

### `read_aloud` — 今すぐ読み上げ

テキストを音声化し、MacBookで即座に再生する。メニューバーにプレイヤーが表示され、一時停止/停止が可能。音声はサーバーのライブラリにも保存される。

| パラメータ | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `text` | Yes | — | 読み上げるテキスト |
| `speaker` | No | `Ono_Anna` | 話者 (`Ono_Anna`, `Aiden`, `Vivian`) |
| `language` | No | `Japanese` | 言語 (`Japanese`, `English`, `Chinese`) |
| `title` | No | — | ライブラリでのタイトル |

### `generate_audio` — 音声生成・保存

テキストを音声化してサーバーのライブラリに保存する。再生はせず、Web UIから後で聴ける。

| パラメータ | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `text` | Yes | — | 音声化するテキスト |
| `title` | No | テキスト先頭50文字 | タイトル |
| `speaker` | No | `Ono_Anna` | 話者 |
| `language` | No | `Japanese` | 言語 |

## Web UI

ブラウザで `http://<ubuntu-ip>:8001` にアクセス。スマホ対応。

- 保存済み音声のリスト表示 (タイトル、話者、再生時間、日時)
- 各音声の再生 (HTML5 audio) とダウンロード
- 新規テキストからの音声生成フォーム
- 音声の削除

## REST API

| Endpoint | Method | 説明 |
|---|---|---|
| `/api/generate` | POST | 音声生成 → ライブラリ保存 → WAV返却 |
| `/api/generate/streaming` | GET | SSEストリーミング生成 → ライブラリ保存 |
| `/api/library` | GET | 保存済み音声一覧 (JSON) |
| `/api/library/{id}/audio` | GET | 音声ファイル取得 (WAV) |
| `/api/library/{id}` | DELETE | 音声削除 |
| `/health` | GET | ヘルスチェック |

### `POST /api/generate` リクエスト例

```bash
curl -X POST http://localhost:8001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "読み上げるテキスト", "title": "タイトル", "speaker": "Ono_Anna"}' \
  -o output.wav
```

### `GET /api/generate/streaming` クエリパラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `text` | (必須) | テキスト |
| `title` | — | タイトル |
| `speaker` | `Ono_Anna` | 話者 |
| `language` | `Japanese` | 言語 |
| `batch_size` | `8` | バッチサイズ |
| `save` | `true` | ライブラリに保存するか |

SSEイベント形式:
- `init` — `{type, total_chunks, sample_rate, entry_id}`
- `chunk` — `{type, index, total, text, duration, audio (base64 WAV), end_type}`
- `complete` — `{type, entry_id}`
- `error` — `{type, message}`

## メニューバープレイヤー (tts_player.py)

MCPの `read_aloud` ツール経由で自動起動する。直接起動も可能:

```bash
python tts_player.py --server http://<ubuntu-ip>:8001 --text "読み上げるテキスト"
```

| オプション | 説明 |
|---|---|
| `--server` | サーバーURL (必須) |
| `--text` | 読み上げテキスト (必須) |
| `--speaker` | 話者 (デフォルト: `Ono_Anna`) |
| `--language` | 言語 (デフォルト: `Japanese`) |
| `--title` | タイトル |
| `--no-save` | ライブラリに保存しない |

メニューバー操作:
- **Pause / Resume** — 一時停止・再開
- **Stop** — 再生中断、アプリ終了

再生完了後は自動的にメニューバーから消える。

## ファイル構成

```
asr-tts/
├── audio_library_server.py   # Web/APIサーバー (Ubuntu)
├── mcp_server.py             # MCPサーバー (MacBook)
├── tts_player.py             # メニューバープレイヤー (MacBook)
├── run_library_server.sh     # サーバー起動スクリプト
├── audio_library/            # 生成音声の永続ストレージ
│   ├── metadata.json         # メタデータ一覧
│   └── *.wav                 # 音声ファイル
├── app.py                    # 既存ベースラインサーバー (変更なし)
└── AUDIO_LIBRARY.md          # このドキュメント
```

## 既存システムとの関係

`app.py` (ポート8000) は従来のTTSサーバーで、Web UIからの通常生成・ストリーミング生成に対応する。ベースラインとして変更せず残してある。

`audio_library_server.py` (ポート8001) は音声ライブラリ機能を追加した新しいサーバー。同じTTSモデル・同じ `split_text()` 関数を使用しており、音声品質は同等。
