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

        # メインスレッドUI更新用（バックグラウンドスレッドから直接self.titleを変更するとクラッシュする）
        self._pending_title: str | None = None
        self._title_lock = threading.Lock()

        # メニュー構成
        self.text_item = rumps.MenuItem(display_text, callback=None)
        self.text_item.set_callback(None)
        self.pause_item = rumps.MenuItem("Pause", callback=self.toggle_pause)
        self.stop_item = rumps.MenuItem("Stop", callback=self.stop_playback)
        self.menu = [self.text_item, None, self.pause_item, self.stop_item]

        # SSE受信・再生スレッドを__init__で開始（ready()はサブプロセスから呼ばれない場合がある）
        self.sse_done = threading.Event()
        print(f"[DEBUG] Starting SSE thread...", file=sys.stderr, flush=True)
        self.sse_thread = threading.Thread(target=self._sse_worker, daemon=True)
        print(f"[DEBUG] Starting playback thread...", file=sys.stderr, flush=True)
        self.play_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.sse_thread.start()
        self.play_thread.start()
        print(f"[DEBUG] Threads started", file=sys.stderr, flush=True)

        # メインスレッドでタイトルを更新するタイマー（0.2秒間隔）
        self._ui_timer = rumps.Timer(self._update_title_on_main, 0.2)
        self._ui_timer.start()

    def _set_title_safe(self, new_title: str):
        """スレッドセーフにタイトルを設定（メインスレッドのタイマーが反映）"""
        with self._title_lock:
            self._pending_title = new_title

    def _update_title_on_main(self, _):
        """メインスレッドで呼ばれるタイマーコールバック"""
        with self._title_lock:
            if self._pending_title is not None:
                self.title = self._pending_title
                self._pending_title = None

    def toggle_pause(self, _):
        if self.is_paused:
            self.is_paused = False
            self.pause_item.title = "Pause"
            self._set_title_safe(f"TTS {self.current_chunk}/{self.total_chunks}")
            if self.stream:
                self.stream.start()
        else:
            self.is_paused = True
            self.pause_item.title = "Resume"
            self._set_title_safe("TTS (paused)")
            if self.stream:
                self.stream.stop()

    def stop_playback(self, _):
        self.is_stopped = True
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self._set_title_safe("TTS (stopped)")
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
        print(f"[DEBUG] Connecting to: {url}", file=sys.stderr, flush=True)

        try:
            # タイムアウト設定: 接続30秒、読み取り無制限
            with httpx.Client(timeout=httpx.Timeout(30.0, read=None)) as client:
                print(f"[DEBUG] Opening SSE stream...", file=sys.stderr, flush=True)
                with client.stream("GET", url) as response:
                    print(f"[DEBUG] Stream opened, status: {response.status_code}", file=sys.stderr, flush=True)
                    buffer = ""
                    chunk_count = 0
                    for chunk in response.iter_text():
                        if self.is_stopped:
                            print(f"[DEBUG] Stopped by user", file=sys.stderr, flush=True)
                            return
                        buffer += chunk
                        while "\n\n" in buffer:
                            message, buffer = buffer.split("\n\n", 1)
                            for line in message.strip().split("\n"):
                                if line.startswith("data: "):
                                    chunk_count += 1
                                    print(f"[DEBUG] Received chunk #{chunk_count}", file=sys.stderr, flush=True)
                                    data = json.loads(line[6:])
                                    self._handle_sse(data)
        except Exception as e:
            print(f"[ERROR] SSE error: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            self._set_title_safe("TTS (error)")
            threading.Timer(1.0, self._quit).start()
            return
        finally:
            print(f"[DEBUG] SSE stream closed", file=sys.stderr, flush=True)
            self.sse_done.set()

    def _handle_sse(self, data: dict):
        event_type = data.get('type')
        print(f"[DEBUG] SSE event: {event_type}", file=sys.stderr, flush=True)
        if event_type == "init":
            self.total_chunks = data["total_chunks"]
            self.sample_rate = data.get("sample_rate", 24000)
            self._set_title_safe(f"TTS 0/{self.total_chunks}")
            print(f"[DEBUG] Init: {self.total_chunks} chunks, sample_rate: {self.sample_rate}", file=sys.stderr, flush=True)
        elif event_type == "chunk":
            if "audio" in data:
                # 旧形式: audioフィールドが直接含まれる（後方互換）
                self._decode_and_queue_b64(data["audio"], data["index"])
            elif "audio_url" in data:
                # 新形式: URLから音声をダウンロード
                audio_url = self.server_url + data["audio_url"]
                idx = data["index"]
                print(f"[DEBUG] Downloading chunk {idx} from {audio_url}", file=sys.stderr, flush=True)
                try:
                    resp = httpx.get(audio_url, timeout=30.0)
                    resp.raise_for_status()
                    audio_data, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
                    print(f"[DEBUG] Downloaded chunk {idx}: {len(resp.content)} bytes -> shape {audio_data.shape}", file=sys.stderr, flush=True)
                    with self.queue_lock:
                        self.audio_queue.append(audio_data)
                    self.playback_event.set()
                    self.current_chunk = idx + 1
                    self._set_title_safe(f"TTS {self.current_chunk}/{self.total_chunks}")
                except Exception as e:
                    print(f"[ERROR] Failed to download chunk {idx}: {e}", file=sys.stderr, flush=True)
        elif event_type == "complete":
            print(f"[DEBUG] Stream complete", file=sys.stderr, flush=True)
            self.sse_done.set()
        elif event_type == "error":
            print(f"[ERROR] Server error: {data['message']}", file=sys.stderr, flush=True)
            self._set_title_safe("TTS (error)")
            threading.Timer(1.0, self._quit).start()

    def _decode_and_queue_b64(self, audio_b64: str, index: int):
        """Base64音声データをデコードしてキューに追加（後方互換）"""
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            print(f"[DEBUG] Decoded audio: {len(audio_bytes)} bytes -> shape {audio_data.shape}, sr={sr}", file=sys.stderr, flush=True)
            with self.queue_lock:
                self.audio_queue.append(audio_data)
            self.playback_event.set()
            self.current_chunk = index + 1
            self._set_title_safe(f"TTS {self.current_chunk}/{self.total_chunks}")
        except Exception as e:
            print(f"[ERROR] Failed to decode audio: {e}", file=sys.stderr, flush=True)

    def _playback_worker(self):
        """キューから音声を取り出して順番に再生"""
        print(f"[DEBUG] Playback worker started", file=sys.stderr, flush=True)
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
                    print(f"[DEBUG] Playing chunk #{played + 1}, shape: {audio.shape}", file=sys.stderr, flush=True)
                    sd.play(audio, samplerate=self.sample_rate)
                    while sd.get_stream().active:
                        if self.is_stopped:
                            print(f"[DEBUG] Playback stopped", file=sys.stderr, flush=True)
                            sd.stop()
                            return
                        time.sleep(0.05)
                    sd.wait()
                    print(f"[DEBUG] Chunk #{played + 1} finished", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[ERROR] Playback error: {e}", file=sys.stderr, flush=True)
                    import traceback
                    traceback.print_exc(file=sys.stderr)

                played += 1

            # SSE完了済み かつ 全チャンク再生完了なら自動終了
            if self.sse_done.is_set() and played >= len(self.audio_queue):
                print(f"[DEBUG] All chunks played, exiting", file=sys.stderr, flush=True)
                self._set_title_safe("TTS (done)")
                threading.Timer(1.0, self._quit).start()
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

    print(f"[DEBUG] Starting TTS Player", file=sys.stderr, flush=True)
    print(f"[DEBUG] Server: {args.server}", file=sys.stderr, flush=True)
    print(f"[DEBUG] Text: {args.text[:50]}...", file=sys.stderr, flush=True)
    print(f"[DEBUG] Speaker: {args.speaker}, Language: {args.language}", file=sys.stderr, flush=True)

    try:
        app = TTSPlayerApp(
            server_url=args.server,
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            title=args.title,
            save=not args.no_save,
        )
        print(f"[DEBUG] App created, starting run loop", file=sys.stderr, flush=True)
        app.run()
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
