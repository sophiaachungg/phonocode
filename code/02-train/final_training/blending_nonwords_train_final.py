"""
bn_final_lr_production.py
--------------------------
Final production training for blending nonwords scoring.

Trains a Logistic Regression on WavLM similarity features using ALL available
labeled participants — no held-out fold. This is the correct final step after
CV has established that:
  1. WavLM similarity features outperform wav2vec2 frozen features
  2. LR matches or exceeds the MLP on these features (0.845 vs 0.842 acc)
  3. The MLP shows meaningful train/val gap indicating overfitting on this N

The CV run gave you the honest performance estimate. This run gives you the
model that actually goes to production.

What this script does
---------------------
  1. Loads WavLM-Large (eval mode, frozen — same as CV run).
  2. Pre-encodes all 24 reference recordings.
  3. Encodes all participant responses and builds similarity features [A,B,A-B,A*B].
  4. Fits a single LR on ALL samples (class_weight='balanced').
  5. Reports training-set metrics (for sanity check — NOT a valid performance estimate;
     use the CV results from stage 2 for that).
  6. Saves the fitted LR to pickle and the feature matrix to numpy for inspection.

Outputs
-------
  results_bn_final_lr/
      lr_production_model.pkl      -- fitted LR, ready for inference
      feature_matrix.npy           -- X_all [N, 8193] for inspection
      labels.npy                   -- y_all [N]
      participant_ids.json         -- ordered list of participant IDs
      stimulus_list.json           -- ordered list of stimulus names per trial
      training_summary.txt         -- metadata, class balance, training-set metrics
      skipped_trials.csv           -- log of any trials skipped during extraction

Usage
-----
  python bn_final_lr_production.py

SLURM: copy the Stage 2 SLURM script and change the python call to:
  python bn_final_lr_production.py
  --time=02:00:00 is sufficient (extraction only, no training loop)
"""

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
)
from transformers import AutoFeatureExtractor, AutoModel

# ---------------------------------------------------------------------------
# CONFIG — must match Stage 2 exactly so features are comparable
# ---------------------------------------------------------------------------
AUDIO_ROOT       = Path("../data/processed/blending_nonwords")
REF_ROOT         = Path("../data/reference_recordings/blending_nonwords")
GROUND_TRUTH_CSV = Path("../scoring/blending_nonwords_ground_truth.csv")
OUTPUT_DIR       = Path("../results_bn_final_lr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID    = "microsoft/wavlm-large"
TARGET_SR   = 16000
RANDOM_SEED = 42

DURATION_THRESHOLD_MS = 8000
MIN_DURATION_MS       = 100

# Difficulty tiers derived from ground truth base-rate correct (n=141 participants)
# Tier boundaries at 33rd (0.397) and 67th (0.820) percentiles
WORD_DIFFICULTY_WEIGHT: Dict[str, float] = {
    # Easy (base-rate >= 0.820)
    "lander": 0.5, "nimby": 0.5, "mog": 0.5, "ko": 0.5,
    "teb": 0.5, "shawbo": 0.5, "tigu": 0.5, "motabe": 0.5,
    # Medium (0.397 <= base-rate < 0.820)
    "vope": 1.0, "shib": 1.0, "het": 1.0, "basp": 1.0,
    "nass": 1.0, "jad": 1.0, "heckobi": 1.0, "ghite": 1.0, "jop": 1.0,
    # Hard (base-rate < 0.397)
    "zigopple": 2.0, "nemowk": 2.0, "koomayg": 2.0, "shyvitch": 2.0,
    "tastains": 2.0, "suhnypogh": 2.0, "nysheeboki": 2.0,
}

STIMULI = [
    "lander", "jad", "mog", "het", "ko", "nimby", "teb", "shawbo",
    "ghite", "zigopple", "shib", "motabe", "heckobi", "tastains",
    "nysheeboki", "jop", "nass", "vope", "suhnypogh", "nemowk",
    "shyvitch", "basp", "tigu", "koomayg",
]

# ---------------------------------------------------------------------------
# Utilities (identical to Stage 2)
# ---------------------------------------------------------------------------

def parse_filename(path: Path) -> Tuple[str, str, str]:
    stem  = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename: {path}")
    return "_".join(parts[:-2]), parts[-2], parts[-1]


def normalize_word(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z']+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_label_map(gt_path: Path) -> Dict[Tuple[str, str], int]:
    gt = pd.read_csv(gt_path, encoding="utf-8-sig")
    non_word_cols = ["participant_id", "RA", "score_source"]
    word_cols = [c for c in gt.columns if c not in non_word_cols]
    gt_long = gt.melt(
        id_vars=["participant_id"], value_vars=word_cols,
        var_name="target_word", value_name="label",
    ).dropna(subset=["label"])
    gt_long["label"]     = gt_long["label"].astype(int)
    gt_long["word_norm"] = gt_long["target_word"].apply(normalize_word)
    return {
        (str(r["participant_id"]), r["word_norm"]): r["label"]
        for _, r in gt_long.iterrows()
    }


def encode_audio(audio: np.ndarray, processor, model, device) -> np.ndarray:
    """Mean+max pooled WavLM embedding, shape [2*D]."""
    inputs = processor(
        audio, sampling_rate=TARGET_SR, return_tensors="pt", padding="longest"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state
    mean_emb = hidden.mean(dim=1).squeeze(0)
    max_emb  = hidden.max(dim=1).values.squeeze(0)
    return torch.cat([mean_emb, max_emb], dim=0).cpu().numpy()


def similarity_features(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """[A, B, A-B, A*B] — standard sentence similarity feature vector."""
    return np.concatenate([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PHONOCODE BN — Final LR Production Training")
    print(f"  Model: {MODEL_ID}")
    print(f"  Training on ALL labeled participants (no held-out fold)")
    print("=" * 60)
    sys.stdout.flush()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")
    sys.stdout.flush()

    # ---- Load WavLM (frozen, eval mode throughout) ----
    print(f"\nLoading WavLM: {MODEL_ID}")
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    wavlm     = AutoModel.from_pretrained(MODEL_ID).to(device)
    wavlm.eval()
    for param in wavlm.parameters():
        param.requires_grad = False
    print(f"WavLM loaded and frozen ({sum(p.numel() for p in wavlm.parameters()):,} params)")
    sys.stdout.flush()

    # ---- Pre-encode reference recordings ----
    print("\nPre-encoding reference recordings...")
    ref_embeddings: Dict[str, np.ndarray] = {}
    for stim in STIMULI:
        ref_path = REF_ROOT / f"{stim}.wav"
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference not found: {ref_path}\n"
                f"Ensure all 24 WAV references are in {REF_ROOT}/"
            )
        audio, _ = librosa.load(ref_path, sr=TARGET_SR, mono=True)
        ref_embeddings[stim] = encode_audio(audio, processor, wavlm, device)
    print(f"  Done. Embedding dim per recording: {next(iter(ref_embeddings.values())).shape[0]}")
    sys.stdout.flush()

    # ---- Build label map ----
    label_map = build_label_map(GROUND_TRUTH_CSV)
    print(f"\nLabel map: {len(label_map)} entries")
    sys.stdout.flush()

    # ---- Extract similarity features for all participants ----
    print("\nExtracting WavLM similarity features (all participants)...")
    wav_files = sorted(AUDIO_ROOT.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No WAV files found under {AUDIO_ROOT}")
    print(f"Found {len(wav_files)} WAV files")
    sys.stdout.flush()

    all_X:    List[np.ndarray] = []
    all_y:    List[int]        = []
    all_pids: List[str]        = []
    all_stims: List[str]       = []
    skipped_rows = []

    for i, wav_path in enumerate(wav_files, 1):
        try:
            participant_id, _, target_word = parse_filename(wav_path)
        except ValueError as e:
            skipped_rows.append({"file": str(wav_path), "reason": str(e)})
            continue

        word_norm = normalize_word(target_word)
        key = (participant_id, word_norm)

        if key not in label_map:
            skipped_rows.append({
                "file": str(wav_path),
                "reason": "no label in ground truth",
            })
            continue

        if word_norm not in ref_embeddings:
            skipped_rows.append({
                "file": str(wav_path),
                "reason": f"no reference recording for '{word_norm}'",
            })
            continue

        audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        duration_ms = len(audio) / sr * 1000.0

        if duration_ms > DURATION_THRESHOLD_MS:
            skipped_rows.append({
                "file": str(wav_path),
                "reason": f"duration {duration_ms:.0f}ms > {DURATION_THRESHOLD_MS}ms",
            })
            continue

        if duration_ms < MIN_DURATION_MS:
            skipped_rows.append({
                "file": str(wav_path),
                "reason": f"duration {duration_ms:.0f}ms < {MIN_DURATION_MS}ms",
            })
            continue

        emb_resp = encode_audio(audio, processor, wavlm, device)
        emb_ref  = ref_embeddings[word_norm]
        sim_feat = similarity_features(emb_resp, emb_ref)

        diff_w    = WORD_DIFFICULTY_WEIGHT.get(word_norm, 1.0)
        diff_feat = np.array([{0.5: 0.0, 1.0: 0.5, 2.0: 1.0}.get(diff_w, 0.5)],
                             dtype=np.float32)
        feat = np.concatenate([sim_feat, diff_feat])

        all_X.append(feat)
        all_y.append(label_map[key])
        all_pids.append(participant_id)
        all_stims.append(word_norm)

        if i % 200 == 0:
            print(f"  {i}/{len(wav_files)} files processed ({len(all_X)} trials)...")
            sys.stdout.flush()

    X_all = np.stack(all_X, axis=0)
    y_all = np.array(all_y, dtype=np.int64)
    n_total = len(X_all)
    n_participants = len(set(all_pids))

    lc = np.bincount(y_all)
    print(f"\nExtraction complete:")
    print(f"  Trials:       {n_total}")
    print(f"  Participants: {n_participants}")
    print(f"  Feature dim:  {X_all.shape[1]}")
    print(f"  Class 0:      {lc[0]} ({lc[0]/n_total*100:.1f}%)")
    print(f"  Class 1:      {lc[1]} ({lc[1]/n_total*100:.1f}%)")
    print(f"  Skipped:      {len(skipped_rows)}")
    sys.stdout.flush()

    # Save feature matrix and metadata
    np.save(OUTPUT_DIR / "feature_matrix.npy", X_all)
    np.save(OUTPUT_DIR / "labels.npy", y_all)
    with open(OUTPUT_DIR / "participant_ids.json", "w") as f:
        json.dump(all_pids, f)
    with open(OUTPUT_DIR / "stimulus_list.json", "w") as f:
        json.dump(all_stims, f)
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(OUTPUT_DIR / "skipped_trials.csv", index=False)
        print(f"  Skipped trial log: {OUTPUT_DIR}/skipped_trials.csv")

    # ---- Fit LR on all data ----
    print("\nFitting Logistic Regression on all data...")
    sys.stdout.flush()

    lr = LogisticRegression(
        max_iter=2000,          # more iterations since we have more data than any single fold
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    lr.fit(X_all, y_all)
    print("  Fit complete.")
    sys.stdout.flush()

    # ---- Training-set metrics (sanity check only — not a valid performance estimate) ----
    # These will be optimistic. Use CV results from Stage 2 for actual performance.
    train_preds = lr.predict(X_all)
    train_acc   = accuracy_score(y_all, train_preds)
    train_bac   = balanced_accuracy_score(y_all, train_preds)
    train_f1    = f1_score(y_all, train_preds, average="macro")
    train_cm    = confusion_matrix(y_all, train_preds)
    train_rpt   = classification_report(y_all, train_preds, digits=3)

    print(f"\nTraining-set metrics (OPTIMISTIC — use CV results for true performance):")
    print(f"  Accuracy:     {train_acc:.3f}")
    print(f"  Balanced Acc: {train_bac:.3f}")
    print(f"  Macro F1:     {train_f1:.3f}")
    print(f"  CM:\n{train_cm}")
    sys.stdout.flush()

    # ---- Save model ----
    model_path = OUTPUT_DIR / "lr_production_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(lr, f)
    print(f"\nModel saved to: {model_path}")

    # ---- Save summary ----
    with open(OUTPUT_DIR / "training_summary.txt", "w") as f:
        f.write("PHONOCODE BN — Final LR Production Model\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Encoder:          {MODEL_ID} (frozen, eval mode)\n")
        f.write(f"Feature:          [A, B, A-B, A*B] + difficulty scalar\n")
        f.write(f"Feature dim:      {X_all.shape[1]}\n")
        f.write(f"Classifier:       LogisticRegression(class_weight='balanced')\n")
        f.write(f"Training trials:  {n_total}\n")
        f.write(f"Participants:     {n_participants}\n")
        f.write(f"Class 0:          {lc[0]} ({lc[0]/n_total*100:.1f}%)\n")
        f.write(f"Class 1:          {lc[1]} ({lc[1]/n_total*100:.1f}%)\n")
        f.write(f"Skipped trials:   {len(skipped_rows)}\n\n")
        f.write("CV performance estimate (from Stage 2 — use these numbers):\n")
        f.write("  Accuracy:        0.845 ± 0.026\n")
        f.write("  Balanced Acc:    0.839 ± 0.027  (primary metric)\n")
        f.write("  Macro F1:        0.837 ± 0.028\n")
        f.write("  Class-0 Recall:  0.816 ± 0.042\n")
        f.write("  Class-1 Recall:  0.861 ± 0.049\n\n")
        f.write("Training-set metrics (optimistic — for sanity check only):\n")
        f.write(f"  Accuracy:        {train_acc:.3f}\n")
        f.write(f"  Balanced Acc:    {train_bac:.3f}\n")
        f.write(f"  Macro F1:        {train_f1:.3f}\n\n")
        f.write(f"Confusion matrix (training set):\n{train_cm}\n\n")
        f.write(f"Classification report (training set):\n{train_rpt}\n")

    print("\n" + "=" * 60)
    print("PRODUCTION MODEL READY")
    print("=" * 60)
    print(f"  Model:   {model_path}")
    print(f"  Summary: {OUTPUT_DIR}/training_summary.txt")
    print()
    print("  Reported performance (CV estimates from Stage 2):")
    print("    Balanced Acc:   0.839 ± 0.027")
    print("    Macro F1:       0.837 ± 0.028")
    print("    Class-0 Recall: 0.816 ± 0.042")
    print()
    print("  To use in inference:")
    print("    import pickle, numpy as np")
    print("    with open('lr_production_model.pkl', 'rb') as f:")
    print("        lr = pickle.load(f)")
    print("    pred  = lr.predict(X_new)         # 0 or 1")
    print("    proba = lr.predict_proba(X_new)   # [p_incorrect, p_correct]")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
