#!/usr/bin/env python3
"""
TTS Menu Bar Player for macOS

macOSメニューバーに常駐する音声プレイヤー。
Ubuntu上のaudio_library_serverからSSEストリームで音声を受信し、
チャンクごとにリアルタイム再生する。

依存: rumps, sounddevice, numpy, soundfile, httpx

使い方:
    python tts_player.py --server http://<ubuntu-ip>:8001 --text "読み上げるテキスト"
    python tts_player.py --server http://192.168.1.100:8001 --text "テスト" --speaker Ono_Anna
"""

import argparse
import base64
import io
import json
import sys
import threading
import time
from urllib.parse import urlencode

import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    import rumps
except ImportError:
    print("Error: rumps is required. Install with: pip install rumps")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("Error: httpx is required. Install with: pip install httpx")
    sys.exit(1)


class TTSPlayerApp(rumps.App):
    def __init__(self, server_url: str, text: str, speaker: str = "Ono_Anna",
                 language: str = "Japanese", title: str | None = None, save: bool = True):
        display_text = text[:30] + "..." if len(text) > 30 else text
        super().__init__(
            name="TTS Player",
            icon=None,
            title="TTS...",
            quit_button=None,
        )

        self.server_url = server_url.rstrip("/")
        self.tts_text = text
        self.speaker = speaker
        self.language = language
        self.tts_title = title
        self.save = save

        # 再生状態
        self.is_paused = False
        self.is_stopped = False
        self.audio_queue: list[np.ndarray] = []
        self.queue_lock = threading.Lock()
        self.current_chunk = 0
        self.total_chunks = 0
        self.sample_rate = 24000

        # sounddeviceストリーム
        self.stream: sd.OutputStream | None = None
        self.play_position = 0
        self.current_audio: np.ndarray | None = None
        self.playback_event = threading.Event()

        # メニュー構成
        self.text_item = rumps.MenuItem(display_text, callback=None)
        self.text_item.set_callback(None)
        self.pause_item = rumps.MenuItem("Pause", callback=self.toggle_pause)
        self.stop_item = rumps.MenuItem("Stop", callback=self.stop_playback)
        self.menu = [self.text_item, None, self.pause_item, self.stop_item]

    def ready(self):
        """アプリ起動後にSSE受信・再生スレッドを開始"""
        self.sse_thread = threading.Thread(target=self._sse_worker, daemon=True)
        self.play_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.sse_thread.start()
        self.play_thread.start()

    def toggle_pause(self, _):
        if self.is_paused:
            self.is_paused = False
            self.pause_item.title = "Pause"
            self.title = f"TTS {self.current_chunk}/{self.total_chunks}"
            if self.stream:
                self.stream.start()
        else:
            self.is_paused = True
            self.pause_item.title = "Resume"
            self.title = "TTS (paused)"
            if self.stream:
                self.stream.stop()

    def stop_playback(self, _):
        self.is_stopped = True
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.title = "TTS (stopped)"
        # 少し待ってからアプリ終了
        threading.Timer(1.0, self._quit).start()

    def _quit(self):
        rumps.quit_application()

    def _sse_worker(self):
        """SSEストリームからチャンクを受信してキューに追加"""
        params = urlencode({
            "text": self.tts_text,
            "speaker": self.speaker,
            "language": self.language,
            "title": self.tts_title or "",
            "save": str(self.save).lower(),
        })
        url = f"{self.server_url}/api/generate/streaming?{params}"

        try:
            with httpx.Client(timeout=None) as client:
                with client.stream("GET", url) as response:
                    buffer = ""
                    for chunk in response.iter_text():
                        if self.is_stopped:
                            return
                        buffer += chunk
                        while "\n\n" in buffer:
                            message, buffer = buffer.split("\n\n", 1)
                            for line in message.strip().split("\n"):
                                if line.startswith("data: "):
                                    data = json.loads(line[6:])
                                    self._handle_sse(data)
        except Exception as e:
            print(f"SSE error: {e}", file=sys.stderr)
            self.title = "TTS (error)"
            threading.Timer(3.0, self._quit).start()

    def _handle_sse(self, data: dict):
        if data["type"] == "init":
            self.total_chunks = data["total_chunks"]
            self.sample_rate = data.get("sample_rate", 24000)
            rumps.notification("TTS Player", "", f"Generating {self.total_chunks} chunks...")
        elif data["type"] == "chunk":
            # Base64 WAVをデコード
            audio_bytes = base64.b64decode(data["audio"])
            audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            with self.queue_lock:
                self.audio_queue.append(audio_data)
            self.playback_event.set()  # 再生スレッドに通知
            self.current_chunk = data["index"] + 1
            self.title = f"TTS {self.current_chunk}/{self.total_chunks}"
        elif data["type"] == "complete":
            pass  # 再生完了は playback_worker が処理
        elif data["type"] == "error":
            print(f"Server error: {data['message']}", file=sys.stderr)
            self.title = "TTS (error)"
            threading.Timer(3.0, self._quit).start()

    def _playback_worker(self):
        """キューから音声を取り出して順番に再生"""
        played = 0
        while not self.is_stopped:
            self.playback_event.wait(timeout=1.0)
            self.playback_event.clear()

            while not self.is_stopped:
                with self.queue_lock:
                    if played >= len(self.audio_queue):
                        break
                    audio = self.audio_queue[played]

                # ブロッキング再生
                try:
                    sd.play(audio, samplerate=self.sample_rate)
                    # 一時停止対応のためポーリングで待機
                    while sd.get_stream().active:
                        if self.is_stopped:
                            sd.stop()
                            return
                        time.sleep(0.05)
                    sd.wait()
                except Exception as e:
                    print(f"Playback error: {e}", file=sys.stderr)

                played += 1

            # 全チャンク受信済みかつ再生完了なら終了
            if self.total_chunks > 0 and played >= self.total_chunks:
                self.title = "TTS (done)"
                rumps.notification("TTS Player", "", "Playback complete")
                threading.Timer(3.0, self._quit).start()
                return


def main():
    parser = argparse.ArgumentParser(description="TTS Menu Bar Player")
    parser.add_argument("--server", required=True, help="Audio library server URL")
    parser.add_argument("--text", required=True, help="Text to read aloud")
    parser.add_argument("--speaker", default="Ono_Anna", help="Speaker name")
    parser.add_argument("--language", default="Japanese", help="Language")
    parser.add_argument("--title", default=None, help="Title for the audio entry")
    parser.add_argument("--no-save", action="store_true", help="Don't save to library")
    args = parser.parse_args()

    app = TTSPlayerApp(
        server_url=args.server,
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        title=args.title,
        save=not args.no_save,
    )
    app.run()


if __name__ == "__main__":
    main()
