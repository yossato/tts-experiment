#!/usr/bin/env python3
"""
ストリーミング風TTS：メモリ効率的な生成と同時再生

長文を小さなバッチに分割し、生成完了分から順次再生。
メモリ枯渇を防ぎながら、待ち時間を最小化。
"""

import re
import time
import threading
import queue
from typing import List
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

try:
    import sounddevice as sd
    AUDIO_PLAYBACK = True
except ImportError:
    AUDIO_PLAYBACK = False
    print("⚠️ sounddeviceが未インストール: 音声再生は無効")
    print("   インストール: pip install sounddevice")


def split_text(text: str, max_chars: int = 50) -> List[str]:
    """
    テキストを句点位置で分割
    
    Args:
        text: 分割するテキスト
        max_chars: 1チャンクの目安文字数
    
    Returns:
        分割されたテキストのリスト
    """
    sentence_end_pattern = r'[。！？\.!?]'
    
    chunks = []
    current_chunk = ""
    
    sentences = re.split(f'({sentence_end_pattern})', text)
    
    merged_sentences = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            merged_sentences.append(sentences[i] + sentences[i + 1])
        elif sentences[i].strip():
            merged_sentences.append(sentences[i])
    
    for sentence in merged_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


class StreamingTTSGenerator:
    """
    メモリ効率的なストリーミング風TTS生成器
    
    - 小バッチ（例：10チャンク）ずつ生成
    - 生成完了分から順次再生
    - 生成スレッドと再生スレッドが並行動作
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: str = "cuda:0",
        batch_size: int = 10,  # 一度に処理するチャンク数
    ):
        """
        Args:
            model_name: モデル名
            device: デバイス
            batch_size: 一度に処理する最大チャンク数（メモリ制約）
        """
        print(f"🔧 モデルをロード中: {model_name}")
        print(f"   デバイス: {device}")
        print(f"   バッチサイズ: {batch_size}チャンクずつ処理")
        
        start = time.time()
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        elapsed = time.time() - start
        print(f"✅ モデルロード完了 ({elapsed:.2f}秒)\n")
        
        self.batch_size = batch_size
        self.sample_rate = 24000
        self.audio_queue = queue.Queue()
        self.generation_complete = threading.Event()
        self.stats = {
            "total_chunks": 0,
            "batches_processed": 0,
            "generation_time": 0,
            "total_audio_duration": 0
        }
    
    def generate_batch(
        self,
        chunks: List[str],
        speaker: str = "Ono_Anna",
        language: str = "Japanese"
    ) -> tuple[List[np.ndarray], int]:
        """
        チャンクのバッチを生成
        
        Args:
            chunks: テキストチャンクのリスト
            speaker: 話者名
            language: 言語
        
        Returns:
            (音声データリスト, サンプリングレート)
        """
        wavs, sr = self.model.generate_custom_voice(
            text=chunks,
            language=[language] * len(chunks),
            speaker=[speaker] * len(chunks),
        )
        return wavs, sr
    
    def generation_worker(
        self,
        chunks: List[str],
        speaker: str,
        language: str
    ):
        """
        音声生成ワーカースレッド
        
        バッチサイズごとに分割して生成し、完了分をキューに追加
        """
        total_chunks = len(chunks)
        self.stats["total_chunks"] = total_chunks
        
        print(f"📊 生成計画:")
        print(f"   総チャンク数: {total_chunks}")
        print(f"   バッチ数: {(total_chunks + self.batch_size - 1) // self.batch_size}")
        print(f"   バッチサイズ: {self.batch_size}チャンクずつ\n")
        
        start_time = time.time()
        
        # バッチごとに処理
        for i in range(0, total_chunks, self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
            
            print(f"🎵 バッチ {batch_num}/{total_batches} 生成中 ({len(batch_chunks)}チャンク)...")
            batch_start = time.time()
            
            try:
                wavs, sr = self.generate_batch(batch_chunks, speaker, language)
                batch_time = time.time() - batch_start
                
                # 各音声をキューに追加
                for j, wav in enumerate(wavs):
                    chunk_idx = i + j
                    chunk_duration = len(wav) / sr
                    self.stats["total_audio_duration"] += chunk_duration
                    
                    self.audio_queue.put((chunk_idx, wav, sr))
                    print(f"   ✓ チャンク {chunk_idx + 1}/{total_chunks}: "
                          f"{len(batch_chunks[j])}文字 → {chunk_duration:.2f}秒音声")
                
                self.stats["batches_processed"] += 1
                print(f"   バッチ処理時間: {batch_time:.2f}秒\n")
                
            except Exception as e:
                print(f"❌ バッチ {batch_num} 生成エラー: {e}")
                break
        
        self.stats["generation_time"] = time.time() - start_time
        self.generation_complete.set()
        print("✅ 全バッチの生成完了\n")
    
    def playback_worker(self, save_path: str = None):
        """
        音声再生ワーカースレッド
        
        キューから音声を取得して順次再生・保存
        """
        all_audios = []
        expected_chunk = 0
        pending_audios = {}  # 順序待ちの音声バッファ
        
        print("🔊 再生スレッド開始（生成完了を待機中...）\n")
        
        while True:
            try:
                # タイムアウト付きでキューから取得
                chunk_idx, wav, sr = self.audio_queue.get(timeout=1)
                
                # 順序が正しければすぐに処理、そうでなければバッファに保存
                pending_audios[chunk_idx] = (wav, sr)
                
                # 順序通りに処理
                while expected_chunk in pending_audios:
                    wav, sr = pending_audios.pop(expected_chunk)
                    duration = len(wav) / sr
                    
                    # 音声再生（オプション）
                    if AUDIO_PLAYBACK:
                        print(f"▶️  チャンク {expected_chunk + 1} 再生中 ({duration:.2f}秒)...")
                        sd.play(wav, sr)
                        sd.wait()
                        # チャンク間に0.5秒の無音を挿入（再生時）
                        time.sleep(0.5)
                    
                    # 保存用にバッファ（チャンク間に0.5秒の無音を追加）
                    all_audios.append(wav)
                    # 0.5秒分の無音を追加（サンプリングレート24kHz × 0.5秒）
                    silence = np.zeros(int(sr * 0.5), dtype=wav.dtype)
                    all_audios.append(silence)
                    expected_chunk += 1
                
            except queue.Empty:
                # キューが空 & 生成完了なら終了
                if self.generation_complete.is_set() and self.audio_queue.empty():
                    break
                continue
        
        # 保存処理
        if save_path and all_audios:
            print(f"\n💾 最終音声を保存中: {save_path}")
            combined = np.concatenate(all_audios)
            sf.write(save_path, combined, sr)
            print(f"✅ 保存完了: {len(combined) / sr:.2f}秒の音声\n")
    
    def generate_and_play(
        self,
        text: str,
        speaker: str = "Ono_Anna",
        language: str = "Japanese",
        max_chars: int = 50,
        save_path: str = "streaming_output.wav",
        enable_playback: bool = True
    ):
        """
        テキストから音声を生成し、同時再生
        
        Args:
            text: 生成するテキスト
            speaker: 話者名
            language: 言語
            max_chars: チャンク分割の目安文字数
            save_path: 保存先パス
            enable_playback: リアルタイム再生を有効化
        """
        print("=" * 70)
        print("🚀 ストリーミング風TTS生成＆再生")
        print("=" * 70)
        
        # テキスト分割
        print(f"\n📝 テキスト分割中...")
        chunks = split_text(text, max_chars=max_chars)
        print(f"   全体: {len(text)}文字 → {len(chunks)}チャンクに分割")
        print(f"   バッチ処理: {self.batch_size}チャンクずつ生成\n")
        
        overall_start = time.time()
        
        # 生成スレッドを起動
        gen_thread = threading.Thread(
            target=self.generation_worker,
            args=(chunks, speaker, language)
        )
        
        # 再生スレッドを起動
        playback_thread = threading.Thread(
            target=self.playback_worker,
            args=(save_path if save_path else None,)
        )
        
        gen_thread.start()
        
        if enable_playback and AUDIO_PLAYBACK:
            playback_thread.start()
        
        # 両スレッドの完了を待機
        gen_thread.join()
        
        if enable_playback and AUDIO_PLAYBACK:
            playback_thread.join()
        else:
            # 再生なしの場合は保存のみ（チャンク間に無音を挿入）
            all_audios = []
            while not self.audio_queue.empty():
                _, wav, sr = self.audio_queue.get()
                all_audios.append(wav)
                # チャンク間に0.5秒の無音を追加
                silence = np.zeros(int(sr * 0.5), dtype=wav.dtype)
                all_audios.append(silence)
            
            if save_path and all_audios:
                combined = np.concatenate(all_audios)
                sf.write(save_path, combined, sr)
        
        overall_time = time.time() - overall_start
        
        # 統計表示
        print("=" * 70)
        print("📊 処理統計")
        print("=" * 70)
        print(f"総文字数:       {len(text)} 文字")
        print(f"総チャンク数:   {self.stats['total_chunks']}")
        print(f"処理バッチ数:   {self.stats['batches_processed']}")
        print(f"バッチサイズ:   {self.batch_size} チャンク")
        print(f"生成時間:       {self.stats['generation_time']:.2f} 秒")
        print(f"音声長:         {self.stats['total_audio_duration']:.2f} 秒")
        print(f"総処理時間:     {overall_time:.2f} 秒")
        
        if self.stats['total_audio_duration'] > 0:
            rtf = self.stats['generation_time'] / self.stats['total_audio_duration']
            print(f"RTF:            {rtf:.2f}")
            print(f"スループット:   {len(text) / self.stats['generation_time']:.1f} 文字/秒")
        
        print("=" * 70)


def main():
    # サンプルテキスト（夏目漱石「吾輩は猫である」冒頭）
    sample_text = """
    吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。
    何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
    吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
    この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
    ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
    掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。
    この時妙なものだと思った感じが今でも残っている。第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。
    その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。のみならず顔の真中があまりに突起している。
    そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。
    これが人間の飲む煙草というものである事はようやくこの頃知った。
    """
    sample_text = sample_text.strip()
    
    # 生成器を初期化
    generator = StreamingTTSGenerator(
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device="cuda:0",
        batch_size=10  # 10チャンクずつ処理（メモリ制約）
    )
    
    # 生成＆再生
    generator.generate_and_play(
        text=sample_text,
        speaker="Ono_Anna",
        language="Japanese",
        max_chars=50,
        save_path="natsume_soseki_output.wav",
        enable_playback=True  # リアルタイム再生有効
    )


if __name__ == "__main__":
    main()
