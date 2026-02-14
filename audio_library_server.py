#!/usr/bin/env python3
"""
Audio Library TTS Server

MCP連携用のTTSサーバー。音声生成・保存・Web UIでの再生を提供。
- FastAPIでREST API
- 生成した音声をライブラリに永続保存
- オーディオブック風Web UIで再生・ダウンロード
- SSEストリーミング生成対応
"""

import io
import time
import uuid
import base64
import asyncio
import json
import functools
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel
from streaming_tts import split_text


# --- モデル・設定 ---

LIBRARY_DIR = Path(__file__).parent / "audio_library"
TEMP_DIR = Path(__file__).parent / "temp_chunks"
METADATA_FILE = LIBRARY_DIR / "metadata.json"
SAMPLE_RATE = 24000

model = None
generation_lock = asyncio.Lock()


# --- データモデル ---

class GenerateRequest(BaseModel):
    text: str
    title: Optional[str] = None
    speaker: str = "Ono_Anna"
    language: str = "Japanese"
    max_chars: int = 50
    batch_size: int = 8


# --- ライブラリ管理 ---

def load_metadata() -> list[dict]:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return []


def save_metadata(entries: list[dict]):
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def add_entry(entry: dict):
    entries = load_metadata()
    entries.insert(0, entry)  # 新しいものが上
    save_metadata(entries)


def delete_entry(entry_id: str) -> bool:
    entries = load_metadata()
    new_entries = [e for e in entries if e["id"] != entry_id]
    if len(new_entries) == len(entries):
        return False
    save_metadata(new_entries)
    audio_path = LIBRARY_DIR / f"{entry_id}.wav"
    if audio_path.exists():
        audio_path.unlink()
    return True


def save_audio(entry_id: str, audio_data: np.ndarray, sr: int):
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(LIBRARY_DIR / f"{entry_id}.wav"), audio_data, sr, format="WAV")


# --- FastAPIアプリ ---

app = FastAPI(title="Audio Library TTS Server", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    global model
    print("Loading TTS model...")
    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.2f}s")


# --- APIエンドポイント ---

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """音声生成 → ライブラリに保存 → WAV返却"""
    async with generation_lock:
        try:
            chunks_with_type = split_text(request.text, max_chars=request.max_chars)
            chunks = [t for t, _ in chunks_with_type]

            start_time = time.time()
            all_wavs = []

            for i in range(0, len(chunks), request.batch_size):
                batch = chunks[i:i + request.batch_size]
                wavs, sr = model.generate_custom_voice(
                    text=batch,
                    language=[request.language] * len(batch),
                    speaker=[request.speaker] * len(batch),
                )
                # 文末タイプに応じた無音を追加
                for j, wav in enumerate(wavs):
                    _, end_type = chunks_with_type[i + j]
                    if end_type == "period":
                        silence = np.zeros(int(sr * 1.0), dtype=wav.dtype)
                        wav = np.concatenate([wav, silence])
                    all_wavs.append(wav)

            combined = np.concatenate(all_wavs)
            gen_time = time.time() - start_time
            duration = len(combined) / sr

            # ライブラリに保存
            entry_id = str(uuid.uuid4())[:8]
            save_audio(entry_id, combined, sr)

            title = request.title or request.text[:50]
            add_entry({
                "id": entry_id,
                "title": title,
                "text": request.text,
                "speaker": request.speaker,
                "language": request.language,
                "duration": round(duration, 1),
                "created_at": datetime.now().isoformat(),
                "generation_time": round(gen_time, 2),
            })

            # WAVをレスポンスとして返却
            buffer = io.BytesIO()
            sf.write(buffer, combined, sr, format="WAV")
            buffer.seek(0)

            from urllib.parse import quote
            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    "X-Entry-Id": entry_id,
                    "X-Duration": str(round(duration, 1)),
                    "X-Title": quote(title, safe=""),
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/generate/streaming")
async def generate_streaming(
    request: Request,
    text: str,
    title: Optional[str] = None,
    speaker: str = "Ono_Anna",
    language: str = "Japanese",
    max_chars: int = 50,
    batch_size: int = 8,
    save: bool = True,
):
    """SSEストリーミング生成 → ライブラリに保存"""
    entry_id = str(uuid.uuid4())[:8] if save else None

    # ストリーム用の一時ディレクトリ
    stream_id = str(uuid.uuid4())[:8]
    stream_dir = TEMP_DIR / stream_id
    stream_dir.mkdir(parents=True, exist_ok=True)

    async def generate_stream():
        try:
            disconnected = False
            chunks_with_type = split_text(text, max_chars=max_chars)
            total_chunks = len(chunks_with_type)

            init_data = {"type": "init", "total_chunks": total_chunks, "sample_rate": SAMPLE_RATE}
            if entry_id:
                init_data["entry_id"] = entry_id
            yield f"data: {json.dumps(init_data)}\n\n"

            all_wavs = []

            for i in range(0, total_chunks, batch_size):
                is_disc = await request.is_disconnected()
                print(f"[DEBUG] Batch {i}, is_disconnected={is_disc}")
                if is_disc:
                    print(f"[WARNING] Client disconnected, stopping early")
                    disconnected = True
                    break

                batch_chunks = chunks_with_type[i:i + batch_size]
                batch_texts = [t for t, _ in batch_chunks]

                t0 = datetime.now()
                print(f"[DEBUG] [{t0.strftime('%H:%M:%S.%f')[:-3]}] Generating {len(batch_texts)} chunks...")
                loop = asyncio.get_event_loop()
                async with generation_lock:
                    wavs, sr = await loop.run_in_executor(
                        None,
                        functools.partial(
                            model.generate_custom_voice,
                            text=batch_texts,
                            language=[language] * len(batch_texts),
                            speaker=[speaker] * len(batch_texts),
                        )
                    )
                t1 = datetime.now()
                print(f"[DEBUG] [{t1.strftime('%H:%M:%S.%f')[:-3]}] Generated {len(wavs)} chunks (took {(t1-t0).total_seconds():.2f}s)")

                for j, wav in enumerate(wavs):
                    is_disc = await request.is_disconnected()
                    if is_disc:
                        print(f"[WARNING] Client disconnected, stopping stream")
                        disconnected = True
                        break

                    chunk_idx = i + j
                    chunk_text, end_type = batch_chunks[j]
                    duration = len(wav) / sr

                    if end_type == "period":
                        silence = np.zeros(int(sr * 1.0), dtype=wav.dtype)
                        wav_with_silence = np.concatenate([wav, silence])
                    else:
                        wav_with_silence = wav

                    if save:
                        all_wavs.append(wav_with_silence)

                    # 一時ファイルとして保存
                    chunk_file = stream_dir / f"{chunk_idx}.wav"
                    sf.write(str(chunk_file), wav_with_silence, sr, format="WAV")
                    audio_url = f"/api/temp/{stream_id}/{chunk_idx}.wav"

                    # SSEでは軽量なメタデータ+URLのみ送信
                    chunk_data = {
                        "type": "chunk",
                        "index": chunk_idx,
                        "total": total_chunks,
                        "text": chunk_text,
                        "duration": duration,
                        "end_type": end_type,
                        "audio_url": audio_url,
                    }
                    t_yield = datetime.now()
                    print(f"[DEBUG] [{t_yield.strftime('%H:%M:%S.%f')[:-3]}] Yielding chunk {chunk_idx} (URL: {audio_url})")
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    t_done = datetime.now()
                    print(f"[DEBUG] [{t_done.strftime('%H:%M:%S.%f')[:-3]}] Chunk {chunk_idx} sent")

                if disconnected:
                    break

            # ライブラリに保存
            if save and all_wavs and not disconnected:
                combined = np.concatenate(all_wavs)
                total_duration = len(combined) / sr
                save_audio(entry_id, combined, sr)
                add_entry({
                    "id": entry_id,
                    "title": title or text[:50],
                    "text": text,
                    "speaker": speaker,
                    "language": language,
                    "duration": round(total_duration, 1),
                    "created_at": datetime.now().isoformat(),
                })

            if not disconnected:
                complete_data = {"type": "complete"}
                if entry_id:
                    complete_data["entry_id"] = entry_id
                yield f"data: {json.dumps(complete_data)}\n\n"
            else:
                print(f"[INFO] Stream terminated due to client disconnect")

        except asyncio.CancelledError:
            print(f"[WARNING] Task cancelled, cleaning up")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"[ERROR] Exception in generate_stream: {e}")
            import traceback
            traceback.print_exc()
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[INFO] GPU resources cleaned up")
            # 一時ファイルを遅延削除（クライアントがダウンロードする時間を確保）
            asyncio.get_event_loop().call_later(60, _cleanup_temp, stream_dir)

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _cleanup_temp(path: Path):
    """一時ファイルのクリーンアップ"""
    import shutil
    try:
        if path.exists():
            shutil.rmtree(path)
            print(f"[INFO] Cleaned up temp dir: {path}")
    except Exception as e:
        print(f"[WARNING] Failed to cleanup {path}: {e}")


@app.get("/api/temp/{stream_id}/{filename}")
async def get_temp_audio(stream_id: str, filename: str):
    """一時チャンクファイルの配信"""
    path = TEMP_DIR / stream_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Temp audio not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/library")
async def list_library():
    """保存済み音声一覧"""
    return JSONResponse(load_metadata())


@app.get("/api/library/{entry_id}/audio")
async def get_audio(entry_id: str):
    """音声ファイル取得"""
    path = LIBRARY_DIR / f"{entry_id}.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav", filename=f"{entry_id}.wav")


@app.delete("/api/library/{entry_id}")
async def remove_entry(entry_id: str):
    """音声削除"""
    if not delete_entry(entry_id):
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "deleted"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}


# --- Web UI ---

@app.get("/", response_class=HTMLResponse)
async def index():
    html = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Audio Library</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f5f5f7;
    color: #1d1d1f;
    min-height: 100vh;
}
.header {
    background: #1d1d1f;
    color: white;
    padding: 20px 24px;
    position: sticky;
    top: 0;
    z-index: 100;
}
.header h1 { font-size: 1.4em; font-weight: 600; }
.header p { color: #86868b; font-size: 0.85em; margin-top: 4px; }
.main { max-width: 800px; margin: 0 auto; padding: 20px; }

/* 生成フォーム */
.generate-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.generate-card h2 { font-size: 1.1em; margin-bottom: 16px; }
.generate-card textarea {
    width: 100%;
    min-height: 100px;
    padding: 12px;
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    font-size: 15px;
    font-family: inherit;
    resize: vertical;
    margin-bottom: 12px;
}
.generate-card textarea:focus { outline: none; border-color: #0071e3; }
.form-row {
    display: flex;
    gap: 12px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}
.form-row select, .form-row input {
    flex: 1;
    min-width: 120px;
    padding: 10px 12px;
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    font-size: 14px;
    background: white;
}
.btn {
    display: inline-block;
    padding: 10px 24px;
    border: none;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.2s;
}
.btn:hover { opacity: 0.85; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-primary { background: #0071e3; color: white; }
.btn-stream { background: #30d158; color: white; }
.btn-danger { background: #ff3b30; color: white; font-size: 13px; padding: 6px 14px; }
.btn-row { display: flex; gap: 10px; }

/* 進捗 */
.progress-bar {
    display: none;
    margin-top: 12px;
    background: #e5e5ea;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.progress-bar .fill {
    height: 100%;
    background: #0071e3;
    width: 0%;
    transition: width 0.3s;
}
.status-text {
    display: none;
    margin-top: 8px;
    font-size: 13px;
    color: #86868b;
}

/* ライブラリ */
.library-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}
.library-header h2 { font-size: 1.1em; }
.count { color: #86868b; font-size: 0.85em; }

.audio-card {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    transition: box-shadow 0.2s;
}
.audio-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
}
.card-title {
    font-weight: 600;
    font-size: 0.95em;
    flex: 1;
    margin-right: 12px;
    word-break: break-word;
}
.card-meta {
    font-size: 0.8em;
    color: #86868b;
    white-space: nowrap;
}
.card-text {
    font-size: 0.85em;
    color: #6e6e73;
    margin-bottom: 10px;
    line-height: 1.5;
    max-height: 60px;
    overflow: hidden;
    word-break: break-word;
}
.card-bottom {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.card-bottom audio {
    flex: 1;
    min-width: 200px;
    height: 36px;
}
.card-actions {
    display: flex;
    gap: 8px;
    align-items: center;
}
.card-actions a, .card-actions button {
    font-size: 13px;
    color: #0071e3;
    text-decoration: none;
    background: none;
    border: none;
    cursor: pointer;
    padding: 4px 8px;
}
.card-actions button.delete { color: #ff3b30; }
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #86868b;
}
</style>
</head>
<body>
<div class="header">
    <h1>Audio Library</h1>
    <p>TTS Audio Collection</p>
</div>
<div class="main">
    <div class="generate-card">
        <h2>New Generation</h2>
        <textarea id="text" placeholder="Enter text to generate speech..."></textarea>
        <div class="form-row">
            <input type="text" id="title" placeholder="Title (optional)">
            <select id="speaker">
                <option value="Ono_Anna">Ono Anna</option>
                <option value="Aiden">Aiden</option>
                <option value="Vivian">Vivian</option>
            </select>
            <select id="language">
                <option value="Japanese">Japanese</option>
                <option value="English">English</option>
                <option value="Chinese">Chinese</option>
            </select>
        </div>
        <div class="btn-row">
            <button class="btn btn-primary" onclick="generateNormal()">Generate</button>
            <button class="btn btn-stream" onclick="generateStreaming()">Streaming</button>
        </div>
        <div class="progress-bar" id="progress"><div class="fill" id="progressFill"></div></div>
        <div class="status-text" id="status"></div>
    </div>
    <div class="library-header">
        <h2>Library</h2>
        <span class="count" id="count"></span>
    </div>
    <div id="library"></div>
</div>
<script>
const API = '';  // same origin

async function loadLibrary() {
    const res = await fetch(API + '/api/library');
    const entries = await res.json();
    const el = document.getElementById('library');
    const countEl = document.getElementById('count');
    countEl.textContent = entries.length + ' items';
    if (entries.length === 0) {
        el.innerHTML = '<div class="empty-state">No audio yet. Generate your first one above.</div>';
        return;
    }
    el.innerHTML = entries.map(e => `
        <div class="audio-card" id="card-${e.id}">
            <div class="card-top">
                <div class="card-title">${esc(e.title)}</div>
                <div class="card-meta">${e.speaker} / ${fmtDur(e.duration)} / ${fmtDate(e.created_at)}</div>
            </div>
            <div class="card-text">${esc(e.text)}</div>
            <div class="card-bottom">
                <audio controls preload="none" src="${API}/api/library/${e.id}/audio"></audio>
                <div class="card-actions">
                    <a href="${API}/api/library/${e.id}/audio" download="${e.id}.wav">Download</a>
                    <button class="delete" onclick="deleteEntry('${e.id}')">Delete</button>
                </div>
            </div>
        </div>
    `).join('');
}

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s || '';
    return d.innerHTML;
}

function fmtDur(sec) {
    if (!sec) return '--';
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return m > 0 ? m + 'm' + s + 's' : s + 's';
}

function fmtDate(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    return (d.getMonth()+1) + '/' + d.getDate() + ' ' + d.getHours() + ':' + String(d.getMinutes()).padStart(2,'0');
}

function showProgress(show) {
    document.getElementById('progress').style.display = show ? 'block' : 'none';
    document.getElementById('status').style.display = show ? 'block' : 'none';
}
function setProgress(pct, msg) {
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('status').textContent = msg;
}

async function generateNormal() {
    const text = document.getElementById('text').value.trim();
    if (!text) return;
    const btns = document.querySelectorAll('.btn');
    btns.forEach(b => b.disabled = true);
    showProgress(true);
    setProgress(10, 'Generating...');
    try {
        const res = await fetch(API + '/api/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                text,
                title: document.getElementById('title').value || null,
                speaker: document.getElementById('speaker').value,
                language: document.getElementById('language').value,
            })
        });
        if (!res.ok) throw new Error(await res.text());
        setProgress(100, 'Done!');
        document.getElementById('text').value = '';
        document.getElementById('title').value = '';
        await loadLibrary();
    } catch(e) {
        setProgress(0, 'Error: ' + e.message);
    } finally {
        btns.forEach(b => b.disabled = false);
        setTimeout(() => showProgress(false), 3000);
    }
}

async function generateStreaming() {
    const text = document.getElementById('text').value.trim();
    if (!text) return;
    const btns = document.querySelectorAll('.btn');
    btns.forEach(b => b.disabled = true);
    showProgress(true);
    setProgress(0, 'Connecting...');

    const params = new URLSearchParams({
        text,
        title: document.getElementById('title').value || '',
        speaker: document.getElementById('speaker').value,
        language: document.getElementById('language').value,
    });

    const evtSource = new EventSource(API + '/api/generate/streaming?' + params);
    evtSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === 'init') {
            setProgress(5, 'Generating 0/' + data.total_chunks + ' chunks...');
        } else if (data.type === 'chunk') {
            const pct = Math.round(((data.index + 1) / data.total) * 100);
            setProgress(pct, 'Generating ' + (data.index+1) + '/' + data.total + ' chunks...');
        } else if (data.type === 'complete') {
            setProgress(100, 'Done!');
            evtSource.close();
            document.getElementById('text').value = '';
            document.getElementById('title').value = '';
            loadLibrary();
            btns.forEach(b => b.disabled = false);
            setTimeout(() => showProgress(false), 3000);
        } else if (data.type === 'error') {
            setProgress(0, 'Error: ' + data.message);
            evtSource.close();
            btns.forEach(b => b.disabled = false);
        }
    };
    evtSource.onerror = function() {
        setProgress(0, 'Connection error');
        evtSource.close();
        btns.forEach(b => b.disabled = false);
    };
}

async function deleteEntry(id) {
    if (!confirm('Delete this audio?')) return;
    await fetch(API + '/api/library/' + id, {method: 'DELETE'});
    const card = document.getElementById('card-' + id);
    if (card) card.remove();
    loadLibrary();
}

loadLibrary();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
