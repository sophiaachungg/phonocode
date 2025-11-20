import re
from pathlib import Path

import torch
import librosa
import pandas as pd
from transformers import AutoProcessor, AutoModelForCTC

# ---------- CONFIG ----------
ROOT_DIR = Path("data_processed/phoneme_reversal")

# WavLM ASR checkpoint
MODEL_ID = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"

OUTPUT_CSV = "scoring/wavlm_results.csv"
TARGET_SR = 16000
# -----------------------------


def parse_filename(path: Path):
    """
    Expect filenames like:
      ReXa_149_01_an.wav

    Format:
      [participant_id]_[audio_num]_[target_word].wav

    Where participant_id itself may contain underscores.
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {path}")

    target_word = parts[-1]
    audio_num = parts[-2]
    participant_id = "_".join(parts[:-2])
    return participant_id, audio_num, target_word


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z']+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def main():
    # --- device selection, CPU vs MPS ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    print(f"Loading model and processor from: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

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

        # Load and resample to 16k mono
        audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

        # Processor -> tensors
        inputs = processor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding="longest",
        )

        input_values = inputs.input_values.to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_values=input_values, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1)

        transcript = processor.batch_decode(pred_ids)[0].strip()

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
        raise RuntimeError("No valid results produced. Check folder structure / MODEL_ID.")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")

    acc = df["is_correct"].mean()
    print(f"Overall exact-match accuracy (normalized): {acc:.3f}")


if __name__ == "__main__":
    main()
