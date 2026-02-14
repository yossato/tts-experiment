#!/usr/bin/env python3
"""
TTS MCP Server

Claude Code等のMCPクライアントから呼び出せるTTSツールを提供。
audio_library_serverのAPIを呼び出して音声生成を行う。

ツール:
  - read_aloud: テキストを即座に読み上げ（メニューバープレイヤーで再生）
  - generate_audio: テキストを音声化してサーバーに保存（後から再生）

設定例 (.mcp.json):
{
  "mcpServers": {
    "tts": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "TTS_SERVER_URL": "http://<ubuntu-ip>:8001"
      }
    }
  }
}
"""

import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx
from mcp.server.fastmcp import FastMCP


def resolve_server_url(url: str) -> str:
    """URLのホスト名をIPアドレスに解決する。

    mDNS (.local) などMCPサブプロセスから解決できないホスト名を
    起動時にIPアドレスへ変換しておく。
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return url

    try:
        ip = socket.getaddrinfo(hostname, None, socket.AF_INET)[0][4][0]
    except socket.gaierror:
        # 解決できなければ元のURLをそのまま返す
        return url

    # ホスト名をIPに置換（ポート付きの場合も考慮）
    if parsed.port:
        new_netloc = f"{ip}:{parsed.port}"
    else:
        new_netloc = ip

    resolved = urlunparse(parsed._replace(netloc=new_netloc))
    if resolved != url:
        print(f"Resolved {hostname} -> {ip}", file=sys.stderr)
    return resolved


# サーバー設定
TTS_SERVER_URL = resolve_server_url(
    os.environ.get("TTS_SERVER_URL", "http://localhost:8001")
)
PLAYER_SCRIPT = str(Path(__file__).parent / "tts_player.py")

mcp = FastMCP("tts")


@mcp.tool()
async def read_aloud(
    text: str,
    speaker: str = "Ono_Anna",
    language: str = "Japanese",
    title: str = "",
) -> str:
    """Read text aloud immediately using TTS. Plays audio on this machine via menu bar player.

    Args:
        text: Text to read aloud
        speaker: Speaker voice (Ono_Anna, Aiden, Vivian)
        language: Language (Japanese, English, Chinese)
        title: Optional title for the audio entry
    """
    # デバッグ用ログファイル
    log_file = Path(__file__).parent / "tts_player_debug.log"
    
    cmd = [
        sys.executable, PLAYER_SCRIPT,
        "--server", TTS_SERVER_URL,
        "--text", text,
        "--speaker", speaker,
        "--language", language,
    ]
    if title:
        cmd.extend(["--title", title])

    try:
        # デバッグ: エラーログをファイルに出力
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        
        # プロセスが即座に終了していないか確認
        import time
        time.sleep(0.5)
        if process.poll() is not None:
            # プロセスが終了している場合、ログを読む
            with open(log_file, "r") as log:
                error_msg = log.read()
            return f"Player process exited immediately (code: {process.returncode}).\nLog:\n{error_msg}\nCheck {log_file} for details."
    except Exception as e:
        return f"Failed to start player: {e}\nCommand: {' '.join(cmd)}"

    preview = text[:80] + "..." if len(text) > 80 else text
    return (
        f"Playback started. Control via menu bar icon.\n"
        f"Text: {preview}\n"
        f"Speaker: {speaker}, Language: {language}\n"
        f"Debug log: {log_file}"
    )


@mcp.tool()
async def generate_audio(
    text: str,
    title: str = "",
    speaker: str = "Ono_Anna",
    language: str = "Japanese",
) -> str:
    """Generate TTS audio and save to the audio library for later playback.

    Args:
        text: Text to convert to speech
        title: Title for the audio entry (defaults to first 50 chars of text)
        speaker: Speaker voice (Ono_Anna, Aiden, Vivian)
        language: Language (Japanese, English, Chinese)
    """
    url = f"{TTS_SERVER_URL}/api/generate"
    payload = {
        "text": text,
        "title": title or None,
        "speaker": speaker,
        "language": language,
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
    except httpx.ConnectError:
        return f"Cannot connect to TTS server at {TTS_SERVER_URL}. Is it running?"
    except httpx.HTTPStatusError as e:
        return f"Server error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error: {e}"

    entry_id = response.headers.get("X-Entry-Id", "unknown")
    duration = response.headers.get("X-Duration", "?")
    from urllib.parse import unquote
    saved_title = unquote(response.headers.get("X-Title", title or text[:50]))

    return (
        f"Audio saved to library.\n"
        f"Title: {saved_title}\n"
        f"Duration: {duration}s\n"
        f"Speaker: {speaker}, Language: {language}\n"
        f"View at: {TTS_SERVER_URL}"
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
