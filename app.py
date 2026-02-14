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


# リクエストモデル
class TTSRequest(BaseModel):
    text: str
    speaker: str = "Ono_Anna"
    language: str = "Japanese"
    max_chars: int = 50


# グローバル変数でモデルを保持
model = None


def split_text(text: str, max_chars: int = 50) -> List[str]:
    """
    テキストを句点位置で分割（simple_batch_tts.pyから流用）
    
    Args:
        text: 分割するテキスト
        max_chars: 1チャンクの目安文字数
    
    Returns:
        分割されたテキストのリスト
    """
    # 句点パターン（日本語と英語の句読点）
    sentence_end_pattern = r'[。！？\.!?]'
    
    chunks = []
    current_chunk = ""
    
    # 文単位で分割
    sentences = re.split(f'({sentence_end_pattern})', text)
    
    # 句読点を前の文に結合
    merged_sentences = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            merged_sentences.append(sentences[i] + sentences[i + 1])
        elif sentences[i].strip():
            merged_sentences.append(sentences[i])
    
    # max_chars前後でチャンク化
    for sentence in merged_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # 現在のチャンクに追加できるか
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence
        else:
            # 現在のチャンクを保存して新しいチャンクを開始
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    # 最後のチャンク
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


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
    chunks = split_text(text, max_chars=max_chars)
    
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
            
            <button type="submit" id="submitBtn">音声生成</button>
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
    </div>
    
    <script>
        const form = document.getElementById('ttsForm');
        const submitBtn = document.getElementById('submitBtn');
        const loading = document.getElementById('loading');
        const audioPlayer = document.getElementById('audioPlayer');
        const audioElement = document.getElementById('audioElement');
        const statsGrid = document.getElementById('statsGrid');
        const errorDiv = document.getElementById('error');
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const text = document.getElementById('text').value;
            const speaker = document.getElementById('speaker').value;
            const language = document.getElementById('language').value;
            
            // UIリセット
            submitBtn.disabled = true;
            loading.classList.add('show');
            audioPlayer.classList.remove('show');
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
