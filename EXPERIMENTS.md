# 実験ログ

## 2026年2月14日 - プロジェクト開始

### 目標
長文テキストの高速音声合成において、メモリ効率と生成速度を最適化する。

---

## 実験1: 並列処理アプローチの検証

### 仮説
テキストを細かく分割して並列処理すれば、疑似的にRTFを改善できる。

### 試行1: Webサーバー + 並列リクエスト
- **アプローチ**: uvicorn 2ワーカー + Semaphore(2)
- **結果**: RTF 0.76、GPU利用率 ~60%
- **問題**: 
  - プロセス間の切り替えオーバーヘッド
  - リクエスト単位の処理でバッチ効率が低い

### 試行2: asyncio + Semaphore + 個別並列
- **アプローチ**: asyncioで複数チャンクを並列生成
- **結果**: GPU利用率 25-35%
- **問題**: 
  - 個別生成では GPU の並列化が活かせない
  - スレッドプールでも効果薄い

### 試行3: Qwen3-TTS ネイティブバッチ処理
- **アプローチ**: `model.generate_custom_voice(text=[リスト])`
- **結果**: ✅ **RTF 0.37、GPU利用率 ~100%**
- **発見**: 
  - ネイティブバッチ処理が圧倒的に高速
  - 並列制御の複雑さは不要だった

**結論**: Qwen3-TTS公式のバッチ処理だけで十分高速。

---

## 実験2: 長文対応とメモリ管理

### 問題発見
夏目漱石「吾輩は猫である」冒頭（493文字）でOOMエラー発生。

### 原因分析
- モデルは1つだけロード（メモリ使用OK）
- 全チャンクを一度にバッチ処理 → GPU メモリ不足
- RTX 3080 (10GB) では15-20チャンクが限界

### 解決策: 分割バッチ処理
```python
# 10チャンクずつバッチ処理
for i in range(0, total_chunks, 10):
    batch = chunks[i:i+10]
    wavs = model.generate_custom_voice(text=batch, ...)
```

### 結果
- ✅ OOMエラー解消
- ✅ RTF 0.42 を維持（ほぼ劣化なし）
- ✅ 任意の長さのテキストに対応可能

---

## 実験3: ストリーミング風再生の実装

### 動機
- 100秒の音声を40秒で生成できる（RTF 0.4）
- しかし全生成完了まで待つのは体感が悪い

### アプローチ: 自転車操業方式
1. **生成スレッド**: 10チャンクずつ生成してキューに投入
2. **再生スレッド**: キューから取得して順次再生
3. RTF < 1.0 なので、再生中に次のバッチを生成可能

### 実装のポイント
```python
# マルチスレッド + キュー
generation_thread = Thread(target=generate_worker)
playback_thread = Thread(target=playback_worker)
audio_queue = Queue()

# 順序保証
pending_audios = {}  # 順番待ちバッファ
while expected_chunk in pending_audios:
    play(pending_audios.pop(expected_chunk))
```

### 結果
- ✅ 91秒の音声でも17秒後には再生開始
- ✅ メモリ効率的
- ✅ 体感速度が大幅改善

---

## 実験4: Webサーバー版の実装

### 目的
ブラウザから簡単に使えるUI提供。

### 実装内容
- FastAPI + HTML/CSS/JavaScript
- シングルページアプリケーション
- REST API (`POST /api/tts`)
- 処理統計のリアルタイム表示

### 機能
- 複数話者選択（9種類）
- 3言語対応（日本語・英語・中国語）
- ブラウザで音声再生
- API経由での利用も可能

---

## 技術的発見まとめ

### 1. バッチ処理 > 並列処理
Qwen3-TTSのネイティブバッチ処理が、複雑な並列制御より遥かに効率的。

### 2. メモリ管理の重要性
RTX 3080 (10GB) では10-15チャンクが適切なバッチサイズ。

### 3. 体感速度の最適化
技術的な RTF だけでなく、「最初の音が聞こえるまでの時間」も重要。

### 4. ストリーミング機能の現状
- モデルレベルではサポート（97ms レイテンシー）
- Python API では未実装（シミュレートのみ）
- 真のストリーミングは DashScope API（商用）で利用可能

---

## 性能データ

### 短文（236文字）
| 方式 | チャンク数 | 生成時間 | 音声長 | RTF |
|------|-----------|---------|--------|-----|
| シンプルバッチ | 6 | 15.19s | 40.56s | 0.37 |
| Webサーバー | 4リクエスト | 18.11s | 23.68s | 0.76 |

### 長文（493文字）
| 方式 | チャンク数 | バッチ数 | 生成時間 | 音声長 | RTF |
|------|-----------|---------|---------|--------|-----|
| ストリーミング | 13 | 2 | 38.28s | 91.04s | 0.42 |

---

## 実験5: Server-Sent Eventsによる真のストリーミング再生

### 動機
- 実験3のストリーミング風再生はローカル専用
- Webブラウザでも同様の体験を提供したい
- 生成中の音声を順次再生し、待ち時間を最小化

### 技術選択: Server-Sent Events (SSE)
- WebSocketより軽量（単方向通信で十分）
- ブラウザ標準の EventSource API
- text/event-stream によるチャンク配信

### 実装アーキテクチャ
```python
# サーバー側 (FastAPI)
@app.get("/api/tts/streaming")
async def streaming_tts(text: str):
    async def event_generator():
        # テキスト分割
        chunks_with_type = split_text(text, max_chars=50)
        
        # バッチ処理
        for batch in batches:
            wavs = model.generate_custom_voice(...)
            
            # Base64エンコードしてSSE配信
            for wav, (_, end_type) in zip(wavs, batch_chunks_with_type):
                chunk_data = {
                    "audio": base64.b64encode(wav_bytes).decode(),
                    "end_type": end_type,  # "period" or "comma"
                    "sample_rate": 24000
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.1)  # SSEフラッシュ
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

```javascript
// クライアント側 (Web Audio API)
const audioContext = new AudioContext({sampleRate: 24000});
const eventSource = new EventSource('/api/tts/streaming?text=' + encodeURIComponent(text));

eventSource.onmessage = async (event) => {
    const data = JSON.parse(event.data);
    
    // Base64 → Float32Array
    const audioData = base64ToFloat32(data.audio);
    const audioBuffer = audioContext.createBuffer(1, audioData.length, 24000);
    audioBuffer.getChannelData(0).set(audioData);
    
    // プログレッシブ再生（キューイング）
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start(nextStartTime);
    
    // 無音挿入（句読点による制御）
    if (data.end_type === "period") {
        nextStartTime += audioBuffer.duration + 1.0;  // 1秒の沈黙
    } else {
        nextStartTime += audioBuffer.duration;  // 読点は無音なし
    }
};
```

### 発見した問題と解決策

#### 問題1: SSEバッファリングによる遅延
**症状**: サーバーで `yield` してもブラウザに即座に届かない

**原因**: FastAPIのStreamingResponseがバッファリングしている

**解決**: 各 `yield` の直後に `await asyncio.sleep(0.1)` を挿入
```python
yield f"data: {json.dumps(chunk_data)}\n\n"
await asyncio.sleep(0.1)  # イベントループに制御を戻してフラッシュ
```

**結果**: チャンク生成後すぐにクライアントで再生開始

#### 問題2: サーバーハング
**症状**: リクエスト処理中に新しいリクエストが来るとサーバーがフリーズ

**原因**: 同時に複数の TTS 生成がGPUメモリを消費

**解決**: `asyncio.Lock` による排他制御
```python
generation_lock = asyncio.Lock()

async with generation_lock:
    # GPU利用処理
    wavs = await asyncio.to_thread(model.generate_custom_voice, ...)
```

**結果**: 複数ユーザーからのリクエストを順次処理、OOMエラー回避

#### 問題3: EventSourceのクリーンアップ不足
**症状**: 停止ボタン押下後もサーバー側で処理継続

**解決**: クライアント側で明示的にclose
```javascript
stopButton.addEventListener('click', () => {
    if (eventSource) {
        eventSource.close();  // 接続切断
    }
    audioContext.close();  // 音声停止
});
```

---

## 実験6: テキスト分割アルゴリズムの改良

### 基本方針
- 最大文字数（デフォルト50文字）で分割
- 句読点で自然に区切る
- 文の途中で切らない

### 初期実装
```python
def split_text(text: str, max_chars: int = 50) -> List[tuple[str, str]]:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    pattern = r'[。！？\.!?、,]'
    
    current_chunk = ""
    for char in text:
        current_chunk += char
        if re.match(pattern, char) and len(current_chunk) >= max_chars:
            yield (current_chunk.strip(), "period")
            current_chunk = ""
```

### 改良1: 読点と句点の区別

**動機**: ユーザーフィードバック
> "五十文字の中で今は丸とかビックリマークとかハテナでバッチを区切っていると思うんだけども、読点でも区切るようにしたい"

**要件**:
- 読点（、,）: 区切るが無音は入れない
- 句点（。！？）: 区切って1秒の無音を入れる

**実装**:
```python
def split_text(text: str, max_chars: int = 50) -> List[tuple[str, str]]:
    chunks_with_type = []
    current_chunk = ""
    
    for char in text:
        current_chunk += char
        
        if char in ['。', '！', '？', '.', '!', '?']:
            if len(current_chunk) > 0:
                chunks_with_type.append((current_chunk.strip(), "period"))
                current_chunk = ""
        
        elif char in ['、', ',']:
            if len(current_chunk) >= max_chars:
                chunks_with_type.append((current_chunk.strip(), "comma"))
                current_chunk = ""
        
        elif len(current_chunk) >= max_chars:
            # 句読点がない場合は次の句読点まで継続
            pass
    
    return chunks_with_type
```

### 改良2: 改行コードの正規化と対応

**動機**: ユーザーフィードバック
> "改行もそこでその句点みたいに1秒待ってほしい"
> "改行コードって\nだけで対応できる？"

**課題**: Windows (\r\n)、Mac (\r)、Unix (\n) の混在

**解決**: 正規化処理
```python
# 前処理: すべての改行を \n に統一
text = text.replace('\r\n', '\n').replace('\r', '\n')
# 連続する改行を1つにまとめる
text = re.sub(r'\n+', '\n', text)
```

### 改良3: 改行でのチャンク境界強制

**動機**: ユーザー報告
> "■特殊文字・リンク抽出\n全角英数字：～" が1チャンクに結合されてしまう

**問題**: 改行後のテキストが max_chars に達していなければ次のテキストと結合されていた

**解決**: 改行は無条件にチャンク境界とする
```python
for char in text:
    current_chunk += char
    
    is_period = char in ['。', '！', '？', '.', '!', '?']
    is_comma = char in ['、', ',']
    is_newline = char == '\n'
    
    if is_newline:
        # 改行は即座にチャンクを分割
        if len(current_chunk) > 1:  # \n だけのチャンクを避ける
            # 改行前のテキストを保存
            before_newline = current_chunk[:-1].strip()
            if before_newline:
                chunks_with_type.append((before_newline, "period"))
            # 改行それ自体も保存（句点扱い）
            chunks_with_type.append(('\n', "period"))
            current_chunk = ""
    
    elif is_period:
        chunks_with_type.append((current_chunk.strip(), "period"))
        current_chunk = ""
    
    elif is_comma and len(current_chunk) >= max_chars:
        chunks_with_type.append((current_chunk.strip(), "comma"))
        current_chunk = ""
```

**結果**:
- "■特殊文字・リンク抽出" → チャンク1
- "\n" → チャンク2（改行、1秒沈黙）
- "全角英数字：～" → チャンク3

### 沈黙時間の最適化

**初期値**: 0.5秒

**ユーザーフィードバック**:
> "改行の時の待ち時間は1秒が良いですね"

**変更**: 句点・改行ともに1.0秒に統一

**実装箇所**:
1. **app.py** (SSE ストリーミング)
```python
if end_type == "period":
    silence_duration = 1.0
else:
    silence_duration = 0.0
```

2. **streaming_tts.py** (ローカル再生)
```python
# 無音データ生成
silence = np.zeros(int(sample_rate * 1.0), dtype=np.float32)
```

**体感**: 文と文の間が適切に空き、自然な読み上げに

---

## 技術的観察と知見

### 1. SSEのフラッシュタイミング
FastAPIの `StreamingResponse` は明示的にイベントループへ制御を戻さないとバッファリングされる。
`await asyncio.sleep(0.1)` は単なる待機ではなく、フラッシュの必須技法。

### 2. Web Audio APIのスケジューリング
`AudioContext.currentTime` を使った事前スケジューリングで、音声のギャップなくプログレッシブ再生が可能。
```javascript
source.start(nextStartTime);
nextStartTime = audioContext.currentTime + audioBuffer.duration;
```

### 3. テキスト分割の複雑性
単純な正規表現では不十分。改行、読点、句点のそれぞれが異なる挙動を要求：
- **改行**: 無条件に境界、句点相当の沈黙
- **句点**: 境界、1秒沈黙
- **読点**: max_chars 到達時のみ境界、沈黙なし

### 4. Base64エンコーディングのオーバーヘッド
- Float32Array → Base64: 約33%のデータ増加
- しかし JSON over SSE では必須
- 代替案: WebSocket でバイナリ送信（今回は不要と判断）

### 5. asyncio.Lock の重要性
GPU処理は並列化できないため、複数リクエストの直列化が必須。
`asyncio.Lock` により、メモリエラーなしでマルチユーザー対応。

---

## 性能データ（ストリーミング版）

### テスト環境
- GPU: RTX 3080 (10GB)
- モデル: Qwen3-TTS-12Hz-1.7B-CustomVoice (bfloat16)
- ブラウザ: Chrome/Chromium

### 体感レイテンシー
| テキスト長 | チャンク数 | 初回音声再生まで | 全体生成時間 | 音声長 | RTF |
|----------|----------|--------------|------------|--------|-----|
| 50文字 | 1 | ~2秒 | 2.5秒 | 6.8秒 | 0.37 |
| 236文字 | 6 | ~3秒 | 15.2秒 | 40.5秒 | 0.38 |
| 493文字 | 13 | ~4秒 | 38.3秒 | 91.0秒 | 0.42 |

**重要な発見**: 
- テキスト長に関わらず、初回再生まで2-4秒
- RTF 0.4程度なので、生成中に再生が追いつく
- 「全部生成完了してから再生」より体感速度が大幅改善

---

## 実装の完成度

### 本番導入可能な機能
- ✅ SSEによるリアルタイムストリーミング配信
- ✅ Web Audio APIでのプログレッシブ再生
- ✅ 複数リクエストの排他制御（Lock）
- ✅ 改行・句読点を考慮した高度なテキスト分割
- ✅ 読点・句点で異なる沈黙制御
- ✅ EventSource のクリーンアップ処理
- ✅ 停止ボタンによる中断機能
- ✅ レスポンシブWeb UI

### 解決済みの問題
1. ✅ SSE バッファリング → `asyncio.sleep(0.1)` で解決
2. ✅ サーバーハング → `asyncio.Lock` で解決
3. ✅ 改行コード混在 → 正規化で解決
4. ✅ 改行後のテキスト結合 → 境界強制で解決
5. ✅ 沈黙時間の体感 → 1.0秒に調整で解決

---

## 今後の課題

### 調査・検証
- [ ] バッチサイズの最適値（GPU別）
- [ ] 他の話者での性能比較
- [ ] 音声品質の定量評価（MOS、WER等）
- [ ] WebSocketによるバイナリ配信（Base64オーバーヘッド削減）

### 実装改善
- [ ] バッチサイズの動的調整（GPU空きメモリに応じて）
- [ ] vLLM-Omni 統合（真のストリーミング）
- [ ] 複数GPU対応
- [ ] カスタム音声のアップロード機能

### ユーザビリティ
- [x] Web UI の改善（進捗バー、エラー処理、停止ボタン）
- [x] SSE によるリアルタイム配信
- [ ] 設定ファイル対応（話者、バッチサイズ等）
- [ ] Docker コンテナ化

---

## 実験7: MCP統合とmacOSクライアント実装（2026年2月14日夜）

### 目標
VS Code の Model Context Protocol (MCP) を使って、エディタから直接 TTS を呼び出せるようにする。

### アーキテクチャ
```
VS Code (MCP Client)
  ↓
mcp_server.py (FastMCP)
  ↓ subprocess
tts_player.py (rumps - macOS Menu Bar App)
  ↓ SSE Stream
audio_library_server.py (FastAPI on Ubuntu GPU Server)
  ↓ CUDA
Qwen3-TTS-12Hz-1.7B-CustomVoice
```

### 実装内容

#### 1. MCP Server (`mcp_server.py`)
- **ツール**: `read_aloud`, `generate_audio`
- FastMCP 1.26.0 を使用
- `tts_player.py` をサブプロセスとして起動
- デバッグログを `tts_player_debug.log` に出力

#### 2. macOS Menu Bar Player (`tts_player.py`)
- **フレームワーク**: rumps 0.4.0 (macOS専用)
- **機能**:
  - SSEストリームからチャンクをダウンロード
  - メニューバーからPause/Resume/Stop
  - リアルタイム速度変更（0.5x～2.0x）
  - ピッチ保持（librosaのtime_stretch）
- **再生方式**: sounddevice（ブロッキング再生）

#### 3. サーバー側改善 (`audio_library_server.py`)
- **問題**: TCP buffer saturation
  - Base64エンコードした音声（~500KB）をSSEで送信
  - 2チャンク目以降のyieldが55秒以上ブロック
  - 原因: TCP送信バッファが満杯、クライアント側の受信が追いつかない
- **解決策**: URL方式への移行
  - 音声を一時ファイル（`temp_chunks/`）に保存
  - SSEではメタデータ+URLのみ送信（数百バイト）
  - クライアントは別HTTPリクエストでWAVをダウンロード
  - 60秒後に自動クリーンアップ
- **副次的効果**:
  - イベントループのブロッキング解消
  - Web UIのフリーズ解消
  - マルチチャンクでも安定動作

### 技術的課題と解決

#### 課題1: UIスレッドの制約
- **問題**: バックグラウンドスレッドから`self.title`を更新するとmacOSがクラッシュ
- **解決**: タイマーベースのメインスレッド更新
  ```python
  self._ui_timer = rumps.Timer(self._update_title_on_main, 0.2)
  ```

#### 課題2: TCP Buffer Saturation
- **現象**: 
  - 1チャンク目: yield後2ms
  - 2チャンク目: yield後55秒以上
  - サーバーログで確認（`time.time()`タイムスタンプ）
- **原因**: 
  - Base64エンコード後の音声データ: ~500KB
  - TCPバッファ: 通常64KB-256KB
  - クライアント受信速度 < サーバー送信速度
  - `yield`がバッファ空きを待って長時間ブロック
- **試行錯誤**:
  1. 64KB分割送信 → 効果なし（個別yieldも同様にブロック）
  2. タイムアウト調整 → 根本解決にならず
- **最終解決**: URL方式（上記）

#### 課題3: 速度変更のタイミング
- **初期実装**: ダウンロード時に速度変更適用
  - 問題: ストリーミングでは先にダウンロードが進むため反映されない
- **改善**: 再生時に速度変更適用
  ```python
  # _playback_worker内で
  if self.playback_speed != 1.0:
      audio = self._apply_speed_change(audio, self.sample_rate)
  ```
  - 結果: リアルタイムで速度変更が反映

#### 課題4: ピッチ保持
- **要件**: 速度変更時にピッチ（音程）は変えたくない
- **解決**: librosa 0.11.0 の `time_stretch`
  - Pitch-preserving time stretching
  - rate = 2.0 で2倍速、rate = 0.5 で0.5倍速
  - フォールバック: scipy の resample（ピッチも変わる）
- **検証結果**: ピッチ変化なし、音質良好

### パフォーマンス

#### ストリーミング性能
- **テストケース**: 18チャンクの長文（パラオ語の記事、約600文字）
- **結果**:
  - 1チャンク目生成完了後、即座に再生開始
  - GPU生成中に並行して音声再生
  - Web UIフリーズなし（`run_in_executor`による非ブロッキング化）
  - 全チャンク正常配信・再生

#### 速度変更性能
- librosa `time_stretch`の処理時間: チャンクあたり約0.1-0.3秒
- 再生には影響なし（前のチャンク再生中に処理）
- メモリオーバーヘッド: 元の音声データ + 変換後データ（一時的）

### ネットワーク最適化の考察

#### Base64 vs URL方式の比較
| 項目 | Base64 (旧) | URL方式 (新) |
|------|-------------|--------------|
| SSEペイロード | ~500KB | ~200バイト |
| TCPバッファ圧迫 | あり（55秒ブロック） | なし |
| HTTPリクエスト数 | 1 (SSE) | 2 (SSE + GET) |
| サーバーディスク使用 | なし | 一時ファイル（60秒で削除） |
| 実装複雑度 | 低 | 中 |
| 推奨環境 | 単一チャンク | マルチチャンク |

#### WebSocket vs SSE
- **SSE選択理由**:
  - 単方向通信で十分（サーバー→クライアント）
  - HTTP/2サーバープッシュとの親和性
  - 自動再接続機能
  - シンプルな実装
- **WebSocket検討余地**:
  - バイナリ配信でBase64オーバーヘッド削減
  - 双方向制御（速度変更指示など）

### 完成した機能

#### クライアント機能
- ✅ ストリーミング再生（生成中に再生開始）
- ✅ マルチチャンク対応（18チャンク検証済み）
- ✅ Pause/Resume（再生中断・再開）
- ✅ Stop（完全停止）
- ✅ リアルタイム速度変更（0.5x～2.0x）
- ✅ ピッチ保持（librosa time_stretch）
- ✅ メニューバー常駐
- ✅ 日本語・英語対応

#### サーバー機能
- ✅ URL方式による安定配信
- ✅ 一時ファイルの自動クリーンアップ
- ✅ run_in_executorによる非ブロッキングGPU推論
- ✅ generation_lockのスコープ最適化
- ✅ Web UIフリーズ解消

### 既知の制約・課題

#### Web UI「Streaming」ボタンの制約
**現状**: 
- Web UIの「Streaming」ボタンは、SSEで生成進捗を受け取り進行状況を表示するが、**音声の再生は行わない**
- 音声はライブラリに保存され、完了後にユーザーが手動で再生する必要がある

**動作フロー**:
1. テキスト入力 → 「Streaming」ボタンクリック
2. SSE経由で生成進捗を受信（`init` → `chunk` → `complete`）
3. 進捗バーにチャンク数と進捗率を表示
4. 完了後、ライブラリに保存（自動リロード）
5. **音声再生は行われない** → ライブラリ一覧から手動再生が必要

**制約理由**:
- `generateStreaming()`関数は進捗表示のみ実装
- 音声チャンクのダウンロード・デコード・再生処理が未実装
- Web Audio APIを使ったプログレッシブ再生は複雑度が高い

**改善案**:
1. **案1: ストリーミング再生の実装（本格的）**
   - Web Audio APIでチャンクごとに再生
   - `tts_player.py`のmacOSクライアントと同等の体験
   - 実装複雑度: 高（AudioContext、AudioBuffer管理、タイミング制御）
   
2. **案2: 生成完了後の自動再生（シンプル）**
   - `complete`イベント受信後、最新ライブラリエントリを自動再生
   - 実装複雑度: 低（既存のplayAudio関数を呼ぶだけ）
   - デメリット: 生成完了まで音声が聞こえない

**現状の推奨ワークフロー**:
- **リアルタイム再生が必要な場合**: MCP統合版（`tts_player.py`）を使用
- **バッチ生成の場合**: Web UIの「Generate」ボタン → ライブラリから再生
- **進捗確認が必要な場合**: Web UIの「Streaming」ボタン → 完了後に手動再生

### デバッグ手法

#### ログベース診断
```python
# タイムスタンプ付きログで遅延を可視化
print(f"[{time.time():.3f}] Chunk {i} yielded", file=sys.stderr, flush=True)
```
- 結果: yieldから次のログまで55秒→TCP buffer blocking判明

#### ネットワークトレース
```bash
# クライアント側
curl -v http://192.168.1.99:8001/api/generate/streaming?...

# 接続確認
ping -c 3 192.168.1.99
```

#### プロセス管理
```bash
# 残存プロセスの確認
ps aux | grep tts_player

# クリーンアップ
pkill -f tts_player.py
```

### 学んだこと

#### 1. TCPバッファの限界
- ストリーミングでも、データサイズが大きいとyieldがブロックする
- 解決策: ペイロードを小さくする（メタデータのみ）+ 別リクエストで本体取得

#### 2. GUIスレッドの制約
- macOSのUIフレームワークは厳格なスレッド制約あり
- バックグラウンドスレッドからのUI更新は必ずクラッシュ
- メインスレッドのタイマーで間接的に更新

#### 3. 速度変更の実装位置
- ダウンロード時 vs 再生時
- ストリーミングでは「再生時」でないとリアルタイム反映されない
- 先読みバッファリングでは、ダウンロード済みチャンクは変更不可

#### 4. ピッチ保持の重要性
- 単純なリサンプリング（scipy.signal.resample）はピッチも変わる
- librosaのtime_stretchはフェーズボコーダーベースでピッチ保持
- 音質と処理速度のトレードオフあり（今回は問題なし）

---

## 参考文献

- Qwen3-TTS Technical Report: https://arxiv.org/abs/2601.15621
- FlashAttention 2: https://arxiv.org/abs/2307.08691
- PyTorch CUDA Best Practices: https://pytorch.org/docs/stable/notes/cuda.html
- Server-Sent Events (SSE): https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- Web Audio API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API
- Model Context Protocol (MCP): https://spec.modelcontextprotocol.io/
- FastMCP: https://github.com/jlowin/fastmcp
- rumps (Ridiculously Uncomplicated macOS Python Statusbar apps): https://github.com/jaredks/rumps
- librosa Audio Analysis: https://librosa.org/doc/latest/index.html
