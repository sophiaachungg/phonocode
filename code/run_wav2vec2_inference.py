import os
import re
import csv
from pathlib import Path

import torch
import librosa
import pandas as pd
from transformers import pipeline

# ---------- CONFIG ----------
ROOT_DIR = Path("data_processed/phoneme_reversal")
MODEL_NAME = "facebook/wav2vec2-base-960h"
OUTPUT_CSV = "scoring/wav2vec2_results.csv"
TARGET_SR = 16000
# -----------------------------


def parse_filename(path: Path):
    """
    Expect filenames like:
      ReXa_149_01_an.wav
    Format:
      [participant_id]_[audio_num]_[target_word].wav
    Where participant_id itself may contain underscores (ReXa_149).

    Strategy:
      Split on "_" from the right:
        parts[-1] = target_word
        parts[-2] = audio_num
        parts[0..-3] = participant_id (joined with "_")
    """
    stem = path.stem  # e.g. "ReXa_149_01_an"
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {path}")

    target_word = parts[-1]
    audio_num = parts[-2]
    participant_id = "_".join(parts[:-2])

    return participant_id, audio_num, target_word


def normalize_text(s: str) -> str:
    """
    Simple normalization for comparing ASR output to target.
    Tweak this as your regex/label rule evolves.
    """
    s = s.lower().strip()
    # keep letters and apostrophes, collapse everything else to space
    s = re.sub(r"[^a-z']+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def main():
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

    asr = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        device=device,
    )

    wav_files = sorted(ROOT_DIR.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No .wav files found under {ROOT_DIR}")

    print(f"Found {len(wav_files)} wav files")

    results = []

    for i, wav_path in enumerate(wav_files, 1):
        try:
            participant_id, audio_num, target_word = parse_filename(wav_path)
        except ValueError as e:
            print(f"[SKIP] {e}")
            continue

        # load and resample to 16k mono
        audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        # librosa returns float32 numpy array in [-1, 1]

        # run ASR
        out = asr(audio, sampling_rate=TARGET_SR)
        # transformers >=4.26 returns dict with "text"
        transcript = out["text"].strip()

        norm_target = normalize_text(target_word)
        norm_transcript = normalize_text(transcript)

        is_correct = int(norm_target == norm_transcript)

        results.append(
            {
                "participant_id": participant_id,
                "audio_num": audio_num,
                "target_word": target_word,
                "transcript": transcript,
                "normalized_target": norm_target,
                "normalized_transcript": norm_transcript,
                "is_correct": is_correct,
                "file_path": str(wav_path),
            }
        )

        if i % 20 == 0:
            print(f"Processed {i}/{len(wav_files)} files...")

    if not results:
        raise RuntimeError("No valid results produced. Check your folder structure / filenames.")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")

    # Quick sanity check: overall accuracy
    acc = df["is_correct"].mean()
    print(f"Overall exact-match accuracy (normalized): {acc:.3f}")


if __name__ == "__main__":
    main()
