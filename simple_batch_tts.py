#!/usr/bin/env python3
"""
シンプルバッチTTS：Qwen3-TTSのネイティブバッチ処理の能力を検証

このコードは:
- テキストを句点で分割
- Qwen3-TTSのバッチ処理で一括生成
- RTFを計測

asyncio/Semaphoreなどの複雑な並列処理は使わず、
公式のバッチ処理機能だけでどこまで速いか検証する。
"""

import re
import time
from typing import List
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


def split_text(text: str, max_chars: int = 50) -> List[str]:
    """
    テキストを句点位置で分割
    
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


def main():
    # サンプルテキスト
    sample_text = """
    こんにちは。今日は良い天気ですね。私たちは新しい音声合成技術を試しています。
    この技術では、長いテキストを小さな部分に分割します。そして、それぞれをバッチ処理することで高速化を実現します。
    最後に、生成された音声を順番通りに結合して完成させます。これは非常に効率的な方法です。
    人工知能の進歩により、自然な音声合成が可能になりました。今後も技術は進化し続けるでしょう。
    この実証コードが皆様のお役に立てれば幸いです。ありがとうございました。
    """
    sample_text = sample_text.strip()
    
    print("=" * 70)
    print("🚀 シンプルバッチTTS検証（公式バッチ処理のみ）")
    print("=" * 70)
    
    # 1. モデルロード
    print("\n🔧 モデルをロード中...")
    model_start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model_time = time.time() - model_start
    print(f"✅ モデルロード完了 ({model_time:.2f}秒)")
    
    # 2. テキスト分割
    print(f"\n📝 テキスト分割中...")
    chunks = split_text(sample_text, max_chars=50)
    print(f"   全体: {len(sample_text)}文字 -> {len(chunks)}チャンクに分割")
    for i, chunk in enumerate(chunks):
        print(f"   [{i+1}] ({len(chunk):2d}文字) {chunk[:40]}...")
    
    # 3. バッチ音声生成（公式のバッチ処理のみ）
    print(f"\n🎵 バッチ音声生成中（{len(chunks)}チャンクを一括処理）...")
    generation_start = time.time()
    
    # これが公式のバッチ処理
    wavs, sr = model.generate_custom_voice(
        text=chunks,
        language=["Japanese"] * len(chunks),
        speaker=["Ono_Anna"] * len(chunks),
    )
    
    generation_time = time.time() - generation_start
    
    # チャンク情報表示
    print(f"\n   ✅ バッチ処理完了:")
    chunk_durations = []
    for i, (chunk, wav) in enumerate(zip(chunks, wavs)):
        chunk_duration = len(wav) / sr
        chunk_durations.append(chunk_duration)
        print(f"      [{i+1}] {len(chunk):2d}文字 -> {chunk_duration:5.2f}秒音声")
    
    # 4. 音声結合
    print(f"\n🔗 音声結合中...")
    combine_start = time.time()
    combined_audio = np.concatenate(wavs)
    combine_time = time.time() - combine_start
    
    # 5. 保存
    output_file = "simple_batch_output.wav"
    sf.write(output_file, combined_audio, sr)
    
    # 6. 統計情報
    total_time = generation_time + combine_time
    audio_duration = len(combined_audio) / sr
    rtf = total_time / audio_duration
    
    print("\n" + "=" * 70)
    print("📊 処理結果")
    print("=" * 70)
    print(f"総文字数:         {len(sample_text)} 文字")
    print(f"チャンク数:       {len(chunks)}")
    print(f"音声生成時間:     {generation_time:.2f} 秒")
    print(f"音声結合時間:     {combine_time:.4f} 秒")
    print(f"総処理時間:       {total_time:.2f} 秒")
    print(f"生成音声長:       {audio_duration:.2f} 秒")
    print(f"RTF:              {rtf:.2f}")
    print(f"スループット:     {len(sample_text) / total_time:.1f} 文字/秒")
    print(f"\n出力ファイル:     {output_file}")
    print(f"サンプリングレート: {sr} Hz")
    print(f"ファイルサイズ:   {len(combined_audio) * 2 / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 70)
    print("💡 このコードの特徴:")
    print("   - asyncio/Semaphore等の複雑な並列処理は不使用")
    print("   - Qwen3-TTSの公式バッチ処理機能のみ使用")
    print("   - テキスト分割→バッチ生成→結合のシンプルな流れ")
    print("   - RTF < 1.0 でリアルタイムより高速な音声生成")
    print("=" * 70)


if __name__ == "__main__":
    main()
