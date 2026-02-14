#!/usr/bin/env python3
"""
Simple Batch TTS Server with FastAPI and Web UI

シンプルバッチTTSのWebサーバー版
- FastAPIで簡単なREST API
- シンプルなWeb UIを提供
- simple_batch_tts.pyのバッチ処理ロジックを使用
"""

import io
import time
import re
import base64
import asyncio
import json
from typing import List
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel
from streaming_tts import split_text


# リクエストモデル
class TTSRequest(BaseModel):
    text: str
    speaker: str = "Ono_Anna"
    language: str = "Japanese"
    max_chars: int = 50


class TTSStreamingRequest(BaseModel):
    text: str
    speaker: str = "Ono_Anna"
    language: str = "Japanese"
    max_chars: int = 50
    batch_size: int = 8


# グローバル変数でモデルを保持
model = None


def generate_speech(
    text: str,
    speaker: str = "Ono_Anna",
    language: str = "Japanese",
    max_chars: int = 50
) -> tuple[np.ndarray, int, dict]:
    """
    テキストから音声を生成（バッチ処理版）
    
    Args:
        text: 生成するテキスト
        speaker: スピーカー名
        language: 言語
        max_chars: チャンク分割の目安文字数
    
    Returns:
        (audio_data, sample_rate, stats)
    """
    if model is None:
        raise RuntimeError("モデルが初期化されていません")
    
    start_time = time.time()
    
    # テキスト分割
    chunks_with_type = split_text(text, max_chars=max_chars)
    chunks = [text for text, _ in chunks_with_type]  # テキストのみ抽出
    
    # バッチ音声生成
    generation_start = time.time()
    wavs, sr = model.generate_custom_voice(
        text=chunks,
        language=[language] * len(chunks),
        speaker=[speaker] * len(chunks),
    )
    generation_time = time.time() - generation_start
    
    # 音声結合
    combined_audio = np.concatenate(wavs)
    
    total_time = time.time() - start_time
    audio_duration = len(combined_audio) / sr
    
    # 統計情報
    stats = {
        "text_length": len(text),
        "chunks": len(chunks),
        "generation_time": generation_time,
        "total_time": total_time,
        "audio_duration": audio_duration,
        "rtf": total_time / audio_duration if audio_duration > 0 else 0,
        "throughput": len(text) / total_time if total_time > 0 else 0
    }
    
    return combined_audio, sr, stats


# FastAPIアプリケーション
app = FastAPI(title="Simple Batch TTS Server", version="1.0.0")

# モデル生成のロック（同時実行を防ぐ）
generation_lock = asyncio.Lock()

# 静的ファイル（CSS/JS）を提供
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def startup_event():
    """サーバー起動時にモデルをロード"""
    global model
    print("🔧 モデルをロード中...")
    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    elapsed = time.time() - start
    print(f"✅ モデルロード完了 ({elapsed:.2f}秒)")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Web UIのメインページ"""
    html_content = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Batch TTS</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 800px;
            width: 100%;
            padding: 40px;
        }
        
        h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 600;
        }
        
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            background: white;
            cursor: pointer;
            transition: border-color 0.3s;
        }
        
        select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .audio-player {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
            display: none;
        }
        
        .audio-player.show {
            display: block;
        }
        
        audio {
            width: 100%;
            margin-top: 10px;
        }
        
        .stats {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        
        .stat-item {
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
            text-align: center;
        }
        
        .stat-label {
            color: #666;
            font-size: 12px;
            margin-bottom: 4px;
        }
        
        .stat-value {
            color: #333;
            font-size: 18px;
            font-weight: 600;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
        }
        
        .error.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Simple Batch TTS</h1>
        <p class="subtitle">Qwen3-TTS バッチ処理による高速音声合成</p>
        
        <form id="ttsForm">
            <div class="form-group">
                <label for="text">テキスト入力</label>
                <textarea id="text" rows="6" placeholder="ここに音声化したいテキストを入力してください..." required></textarea>
            </div>
            
            <div class="form-group">
                <label for="speaker">話者選択</label>
                <select id="speaker">
                    <option value="Ono_Anna">Ono Anna (日本語・女性)</option>
                    <option value="Aiden">Aiden (英語・男性)</option>
                    <option value="Vivian">Vivian (中国語・女性)</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="language">言語</label>
                <select id="language">
                    <option value="Japanese">日本語</option>
                    <option value="English">英語</option>
                    <option value="Chinese">中国語</option>
                </select>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <button type="submit" id="submitBtn">通常生成</button>
                <button type="button" id="streamBtn" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">ストリーミング生成</button>
            </div>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 15px; color: #666;">音声を生成中...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="audio-player" id="audioPlayer">
            <h3 style="color: #333; margin-bottom: 10px;">生成された音声</h3>
            <audio controls id="audioElement"></audio>
            
            <div class="stats">
                <h4 style="color: #555; margin-bottom: 10px;">📊 処理統計</h4>
                <div class="stats-grid" id="statsGrid"></div>
            </div>
        </div>
        
        <div class="audio-player" id="streamingPlayer" style="display: none;">
            <h3 style="color: #333; margin-bottom: 10px;">🔄 ストリーミング再生</h3>
            <div id="streamProgress" style="margin-bottom: 15px;">
                <div style="background: #e0e0e0; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div id="progressBar" style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; width: 0%; transition: width 0.3s;"></div>
                </div>
                <p id="progressText" style="margin-top: 8px; color: #666; font-size: 14px;">準備中...</p>
            </div>
            <div id="chunkList" style="max-height: 200px; overflow-y: auto; background: white; padding: 10px; border-radius: 8px; margin-bottom: 15px;"></div>
            <button id="stopStreamBtn" style="background: #dc3545;" disabled>ストリーミング停止</button>
        </div>
    </div>
    
    <script>
        const form = document.getElementById('ttsForm');
        const submitBtn = document.getElementById('submitBtn');
        const streamBtn = document.getElementById('streamBtn');
        const loading = document.getElementById('loading');
        const audioPlayer = document.getElementById('audioPlayer');
        const audioElement = document.getElementById('audioElement');
        const statsGrid = document.getElementById('statsGrid');
        const errorDiv = document.getElementById('error');
        const streamingPlayer = document.getElementById('streamingPlayer');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const chunkList = document.getElementById('chunkList');
        const stopStreamBtn = document.getElementById('stopStreamBtn');
        
        let audioContext = null;
        let currentSource = null;
        let audioQueue = [];
        let isPlaying = false;
        let eventSource = null;
        
        // Web Audio API初期化
        function initAudioContext() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
        }
        
        // 音声チャンクをキューに追加して再生
        async function playAudioChunk(base64Audio) {
            initAudioContext();
            
            // Base64デコード
            const binaryString = atob(base64Audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            // AudioBufferにデコード
            const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
            audioQueue.push(audioBuffer);
            
            // 再生中でなければ再生開始
            if (!isPlaying) {
                playNext();
            }
        }
        
        // キューから次の音声を再生
        function playNext() {
            if (audioQueue.length === 0) {
                isPlaying = false;
                return;
            }
            
            isPlaying = true;
            const audioBuffer = audioQueue.shift();
            
            currentSource = audioContext.createBufferSource();
            currentSource.buffer = audioBuffer;
            currentSource.connect(audioContext.destination);
            
            currentSource.onended = () => {
                playNext();
            };
            
            currentSource.start(0);
        }
        
        // ストリーミング停止
        function stopStreaming() {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            if (currentSource) {
                currentSource.stop();
                currentSource = null;
            }
            audioQueue = [];
            isPlaying = false;
            streamBtn.disabled = false;
            stopStreamBtn.disabled = true;
        }
        
        // 通常の音声生成
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const text = document.getElementById('text').value;
            const speaker = document.getElementById('speaker').value;
            const language = document.getElementById('language').value;
            
            // UIリセット
            submitBtn.disabled = true;
            loading.classList.add('show');
            audioPlayer.classList.remove('show');
            streamingPlayer.style.display = 'none';
            errorDiv.classList.remove('show');
            
            try {
                const response = await fetch('/api/tts', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        speaker: speaker,
                        language: language,
                        max_chars: 50
                    })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || '音声生成に失敗しました');
                }
                
                // 統計情報を取得
                const stats = JSON.parse(response.headers.get('X-TTS-Stats'));
                
                // 音声データを取得
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                
                // 音声プレイヤーを表示
                audioElement.src = url;
                audioPlayer.classList.add('show');
                
                // 統計情報を表示
                statsGrid.innerHTML = `
                    <div class="stat-item">
                        <div class="stat-label">文字数</div>
                        <div class="stat-value">${stats.text_length}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">チャンク数</div>
                        <div class="stat-value">${stats.chunks}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">処理時間</div>
                        <div class="stat-value">${stats.total_time.toFixed(2)}秒</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">音声長</div>
                        <div class="stat-value">${stats.audio_duration.toFixed(2)}秒</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">RTF</div>
                        <div class="stat-value">${stats.rtf.toFixed(2)}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">スループット</div>
                        <div class="stat-value">${stats.throughput.toFixed(1)} 字/秒</div>
                    </div>
                `;
                
            } catch (error) {
                errorDiv.textContent = `エラー: ${error.message}`;
                errorDiv.classList.add('show');
            } finally {
                submitBtn.disabled = false;
                loading.classList.remove('show');
            }
        });
        
        // ストリーミング生成
        streamBtn.addEventListener('click', async () => {
            const text = document.getElementById('text').value;
            const speaker = document.getElementById('speaker').value;
            const language = document.getElementById('language').value;
            
            if (!text) {
                errorDiv.textContent = 'テキストを入力してください';
                errorDiv.classList.add('show');
                return;
            }
            
            // 既存の接続をクリーンアップ
            if (eventSource) {
                console.log('既存のEventSourceを閉じます');
                eventSource.close();
                eventSource = null;
            }
            
            // UIリセット
            streamBtn.disabled = true;
            stopStreamBtn.disabled = false;
            audioPlayer.classList.remove('show');
            streamingPlayer.style.display = 'block';
            errorDiv.classList.remove('show');
            chunkList.innerHTML = '';
            progressBar.style.width = '0%';
            progressText.textContent = '準備中...';
            
            // AudioContext初期化
            initAudioContext();
            audioQueue = [];
            isPlaying = false;
            
            let totalChunks = 0;
            let processedChunks = 0;
            let hasReceivedData = false;
            
            try {
                // Server-Sent Events接続
                const params = new URLSearchParams({
                    text: text,
                    speaker: speaker,
                    language: language,
                    max_chars: '50',
                    batch_size: '10'
                });
                console.log('EventSource接続開始:', '/api/tts/streaming?' + params.toString());
                eventSource = new EventSource('/api/tts/streaming?' + params.toString());
                
                eventSource.onopen = (event) => {
                    console.log('EventSource接続確立');
                };
                
                eventSource.onmessage = async (event) => {
                    hasReceivedData = true;
                    console.log('SSEメッセージ受信:', event.data.substring(0, 100) + '...');
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'init') {
                        totalChunks = data.total_chunks;
                        progressText.textContent = `合計 ${totalChunks} チャンク`;
                        console.log(`初期化: ${totalChunks}チャンク`);
                    } else if (data.type === 'chunk') {
                        processedChunks++;
                        const progress = (processedChunks / totalChunks) * 100;
                        progressBar.style.width = `${progress}%`;
                        progressText.textContent = `${processedChunks} / ${totalChunks} チャンク (${progress.toFixed(0)}%)`;
                        
                        // チャンクリストに追加
                        const chunkDiv = document.createElement('div');
                        chunkDiv.style.padding = '5px';
                        chunkDiv.style.marginBottom = '3px';
                        chunkDiv.style.background = '#f8f9fa';
                        chunkDiv.style.borderRadius = '4px';
                        chunkDiv.style.fontSize = '13px';
                        chunkDiv.textContent = `${processedChunks}. ${data.text} (${data.duration.toFixed(2)}秒)`;
                        chunkList.appendChild(chunkDiv);
                        chunkList.scrollTop = chunkList.scrollHeight;
                        
                        // 音声を再生
                        try {
                            await playAudioChunk(data.audio);
                        } catch (e) {
                            console.error('音声再生エラー:', e);
                        }
                    } else if (data.type === 'complete') {
                        console.log('ストリーミング完了');
                        progressText.textContent = '完了しました!';
                        // 完了時は自動的にクリーンアップ
                        if (eventSource) {
                            eventSource.close();
                            eventSource = null;
                        }
                        streamBtn.disabled = false;
                        stopStreamBtn.disabled = true;
                    } else if (data.type === 'error') {
                        throw new Error(data.message);
                    }
                };
                
                eventSource.onerror = (error) => {
                    console.error('EventSource エラー:', error);
                    // データを受信していない場合のみエラー表示
                    if (!hasReceivedData) {
                        errorDiv.textContent = 'ストリーミング接続エラーが発生しました';
                        errorDiv.classList.add('show');
                    } else {
                        console.log('ストリーミング終了（データ受信後）');
                    }
                    stopStreaming();
                };
                
            } catch (error) {
                console.error('ストリーミング開始エラー:', error);
                errorDiv.textContent = `エラー: ${error.message}`;
                errorDiv.classList.add('show');
                stopStreaming();
            }
        });
        
        // ストリーミング停止ボタン
        stopStreamBtn.addEventListener('click', stopStreaming);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """
    テキストから音声を生成するAPI
    
    Args:
        request: TTSリクエスト（text, speaker, language, max_chars）
    
    Returns:
        WAV音声データ（StreamingResponse）
    """
    try:
        # 音声生成
        audio_data, sr, stats = generate_speech(
            text=request.text,
            speaker=request.speaker,
            language=request.language,
            max_chars=request.max_chars
        )
        
        # WAVファイルをメモリに書き込み
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format='WAV')
        buffer.seek(0)
        
        # 統計情報をヘッダーに追加
        headers = {
            "X-TTS-Stats": str(stats).replace("'", '"')
        }
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers=headers
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tts/streaming")
async def text_to_speech_streaming(
    text: str,
    speaker: str = "Ono_Anna",
    language: str = "Japanese",
    max_chars: int = 50,
    batch_size: int = 10
):
    """
    テキストから音声をストリーミング生成するAPI
    
    Args:
        text: 生成するテキスト
        speaker: 話者名
        language: 言語
        max_chars: 最大文字数
        batch_size: バッチサイズ
    
    Returns:
        Server-Sent Events形式でチャンクごとの音声データ
    """
    print(f"🎵 ストリーミングリクエスト受信: {len(text)}文字, speaker={speaker}, lang={language}")
    
    async def generate_stream():
        # ロック取得を試みる（既に他の処理中なら待機）
        print("🔒 ロック取得を試みています...")
        async with generation_lock:
            print("✅ ロック取得成功、生成開始")
            try:
                # テキスト分割（文末タイプ付き）
                chunks_with_type = split_text(text, max_chars=max_chars)
                total_chunks = len(chunks_with_type)
                print(f"📝 分割完了: {total_chunks}チャンク")
                
                # 初期情報を送信
                init_data = {'type': 'init', 'total_chunks': total_chunks, 'sample_rate': 24000}
                yield f"data: {json.dumps(init_data)}\n\n"
                await asyncio.sleep(0.1)  # データをフラッシュ
                
                # バッチごとに処理
                for i in range(0, total_chunks, batch_size):
                    batch_chunks_with_type = chunks_with_type[i:i + batch_size]
                    batch_texts = [text for text, _ in batch_chunks_with_type]
                    batch_num = i // batch_size + 1
                    print(f"🎤 バッチ {batch_num} 生成中...")
                    
                    # バッチ生成
                    wavs, sr = model.generate_custom_voice(
                        text=batch_texts,
                        language=[language] * len(batch_texts),
                        speaker=[speaker] * len(batch_texts),
                    )
                    print(f"✓ バッチ {batch_num} 生成完了")
                    
                    # 各チャンクを送信
                    for j, wav in enumerate(wavs):
                        chunk_idx = i + j
                        chunk_text, end_type = batch_chunks_with_type[j]
                        duration = len(wav) / sr
                        
                        # 文末タイプに応じて無音の長さを変える
                        if end_type == "period":
                            # 句点・改行: 1秒の無音
                            silence_duration = 1.0
                        else:
                            # 読点: 無音なし
                            silence_duration = 0.0
                        
                        silence = np.zeros(int(sr * silence_duration), dtype=wav.dtype)
                        wav_with_silence = np.concatenate([wav, silence])
                        
                        # WAVファイルとしてエンコード
                        buffer = io.BytesIO()
                        sf.write(buffer, wav_with_silence, sr, format='WAV')
                        audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        # チャンクデータを送信
                        chunk_data = {
                            'type': 'chunk',
                            'index': chunk_idx,
                            'total': total_chunks,
                            'text': chunk_text,
                            'duration': duration,
                            'audio': audio_base64,
                            'end_type': end_type
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        
                        # データをフラッシュしてクライアント側の処理を待機
                        await asyncio.sleep(0.1)
                
                # 完了通知
                complete_data = {'type': 'complete'}
                yield f"data: {json.dumps(complete_data)}\n\n"
                print("✅ ストリーミング完了")
                
            except asyncio.CancelledError:
                # クライアント切断時: リソースクリーンアップして静かに終了
                print(f"⚠️  ストリーミング中断: クライアント切断を検知")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # CancelledErrorは再送出しない（接続終了を正常に処理）
            except Exception as e:
                print(f"❌ ストリーミングエラー: {e}")
                import traceback
                traceback.print_exc()
                error_data = {'type': 'error', 'message': str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"
            finally:
                # 必ずGPUキャッシュをクリア
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("🔓 ロック解放")
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )



@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
