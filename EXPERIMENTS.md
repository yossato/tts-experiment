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

## 参考文献

- Qwen3-TTS Technical Report: https://arxiv.org/abs/2601.15621
- FlashAttention 2: https://arxiv.org/abs/2307.08691
- PyTorch CUDA Best Practices: https://pytorch.org/docs/stable/notes/cuda.html
- Server-Sent Events (SSE): https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- Web Audio API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API
