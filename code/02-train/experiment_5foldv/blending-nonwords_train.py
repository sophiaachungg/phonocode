"""
bn_stage2_wavlm_similarity.py
-------------------------------
Stage 2: WavLM similarity-based learned classifier for blending nonwords.

For each participant trial, encodes BOTH the participant response and the
reference recording with WavLM-Large (top 2 transformer layers unfrozen).
Builds a 4-part similarity feature vector:
    [A, B, A-B, A*B]   where A = participant embedding, B = reference embedding

This is the standard sentence similarity feature vector from the NLP literature.
Cosine similarity alone would lose directional information; the concatenation
preserves it and gives the MLP more to work with.

Key differences from wav2vec2 frozen (stage 3):
  - WavLM-Large (not base): richer representations, trained with masked
    speech denoising, better for out-of-distribution phonology.
  - Top 2 layers unfrozen: allows fine-tuning to pseudoword acoustics.
  - Similarity feature vector (not single-embedding): comparative signal
    between response and canonical reference.

Duration gate: participant files > DURATION_THRESHOLD_MS are skipped.
Reference files are always used as-is (no gate).

Outputs
-------
  results_bn_stage2_wavlm/
      cv_fold_results.csv
      cv_aggregate_results.txt
      cv_summary_plot.png
      fold{N}/
          mlp_best_model.pt
          test_results.txt
          test_predictions.csv
          training_history.csv
          learning_curves.png

Usage
-----
  python bn_stage2_wavlm_similarity.py

SLURM: see bn_stage2_wavlm_similarity.sh
"""

import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import librosa
import torchaudio.functional as TAF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, classification_report, f1_score,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
AUDIO_ROOT       = Path("../data_processed/blending_nonwords")
REF_ROOT         = Path("../reference_recordings/blending_nonwords")
GROUND_TRUTH_CSV = Path("../scoring/blending_nonwords_ground_truth.csv")
OUTPUT_DIR       = Path("../results_bn_stage2_wavlm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID    = "microsoft/wavlm-large"
TARGET_SR   = 16000
RANDOM_SEED = 42
N_FOLDS     = 5

# WavLM: freeze all but top N transformer layers
N_UNFROZEN_LAYERS = 2

# Duration gate — pseudoword responses should never be this long
DURATION_THRESHOLD_MS = 4000

# Training hyperparameters
MLP_LR         = 1e-4
WEIGHT_DECAY   = 2e-4
N_AUGMENT_COPIES = 2
EARLY_STOPPING_PATIENCE  = 10
EARLY_STOPPING_MIN_DELTA = 1e-4
THRESHOLD_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50]

# Augmentation (same as phoneme reversal)
AUG_NOISE_PROB    = 0.5
AUG_NOISE_SNR_DB  = (10, 30)
AUG_PITCH_PROB    = 0.5
AUG_PITCH_STEPS   = (-2, 2)
AUG_STRETCH_PROB  = 0.25
AUG_STRETCH_RANGE = (0.9, 1.1)

# Difficulty tiers for blending nonwords.
# Derived from ground truth base-rate correct (low base-rate = high difficulty).
# Three tiers, matching the structure used in phoneme reversal.
WORD_DIFFICULTY_WEIGHT: Dict[str, float] = {
    # Easy (base-rate correct >= 0.70)
    "lander": 0.5, "mog": 0.5, "het": 0.5, "ko": 0.5,
    "nimby": 0.5, "teb": 0.5, "shib": 0.5, "jop": 0.5,
    "nass": 0.5, "basp": 0.5, "tigu": 0.5,
    # Medium (base-rate 0.40–0.69)
    "jad": 1.0, "shawbo": 1.0, "motabe": 1.0, "vope": 1.0,
    "nemowk": 1.0, "shyvitch": 1.0,
    # Hard (base-rate < 0.40)
    "ghite": 2.0, "zigopple": 2.0, "heckobi": 2.0, "tastains": 2.0,
    "nysheeboki": 2.0, "suhnypogh": 2.0, "koomayg": 2.0,
}

DURATION_WEIGHT_FLOOR = 0.25

STIMULI = [
    "lander", "jad", "mog", "het", "ko", "nimby", "teb", "shawbo",
    "ghite", "zigopple", "shib", "motabe", "heckobi", "tastains",
    "nysheeboki", "jop", "nass", "vope", "suhnypogh", "nemowk",
    "shyvitch", "basp", "tigu", "koomayg",
]

# ---------------------------------------------------------------------------
# Utilities (shared with other BN scripts)
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


def duration_weight(duration_ms: float) -> float:
    if duration_ms <= DURATION_THRESHOLD_MS:
        return 1.0
    excess = (duration_ms - DURATION_THRESHOLD_MS) / DURATION_THRESHOLD_MS
    return max(1.0 - (1.0 - DURATION_WEIGHT_FLOOR) * min(excess, 1.0), DURATION_WEIGHT_FLOOR)


def augment_audio(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    audio = audio.copy().astype(np.float32)
    if rng.random() < AUG_NOISE_PROB:
        snr_db   = rng.uniform(*AUG_NOISE_SNR_DB)
        sig_rms  = np.sqrt(np.mean(audio ** 2)) + 1e-9
        noise    = rng.standard_normal(len(audio)).astype(np.float32) * (sig_rms / (10 ** (snr_db / 20)))
        audio    = audio + noise
    if rng.random() < AUG_PITCH_PROB:
        n_steps = rng.uniform(*AUG_PITCH_STEPS)
        wav_t   = torch.from_numpy(audio).unsqueeze(0)
        audio   = TAF.pitch_shift(wav_t, sr, n_steps=float(n_steps)).squeeze(0).numpy()
    if rng.random() < AUG_STRETCH_PROB:
        rate  = rng.uniform(*AUG_STRETCH_RANGE)
        wav_t = torch.from_numpy(audio).unsqueeze(0)
        audio = TAF.resample(wav_t, orig_freq=int(sr * rate), new_freq=sr).squeeze(0).numpy()
    return np.clip(audio, -1.0, 1.0)


def encode_audio(audio: np.ndarray, processor, model, device) -> np.ndarray:
    """
    Encode a single audio clip with WavLM.
    Returns mean+max pooled embedding of shape [2*D].
    """
    inputs = processor(
        audio, sampling_rate=TARGET_SR, return_tensors="pt", padding="longest"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state   # [1, T, D]
    mean_emb = hidden.mean(dim=1).squeeze(0)
    max_emb  = hidden.max(dim=1).values.squeeze(0)
    return torch.cat([mean_emb, max_emb], dim=0).cpu().numpy()   # [2*D]


def similarity_features(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Build [A, B, A-B, A*B] similarity feature vector.
    This is the standard representation from sentence similarity literature.
    Captures both absolute representations and their difference/interaction.
    """
    return np.concatenate([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b])


def participant_kfold(participants, n_splits=5, seed=42):
    rng         = np.random.default_rng(seed)
    participants = np.array(participants)
    unique_pids = np.array(sorted(set(participants)))
    rng.shuffle(unique_pids)
    kf    = KFold(n_splits=n_splits, shuffle=False)
    folds = []
    for pid_train_idx, pid_test_idx in kf.split(unique_pids):
        train_pids = set(unique_pids[pid_train_idx])
        test_pids  = set(unique_pids[pid_test_idx])
        folds.append((
            np.array([i for i, p in enumerate(participants) if p in train_pids]),
            np.array([i for i, p in enumerate(participants) if p in test_pids]),
        ))
    return folds, unique_pids


def participant_holdout_split(participants, val_fraction=0.2, seed=42):
    participants = np.array(participants)
    unique_pids  = np.array(sorted(set(participants)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pids)
    n_val = max(1, min(int(round(len(unique_pids) * val_fraction)), len(unique_pids) - 1))
    val_pids   = set(unique_pids[:n_val])
    train_pids = set(unique_pids[n_val:])
    return (
        np.array([i for i, p in enumerate(participants) if p in train_pids]),
        np.array([i for i, p in enumerate(participants) if p in val_pids]),
        sorted(train_pids), sorted(val_pids),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SimilarityDataset(data.Dataset):
    def __init__(self, X, y, sample_weights, augment=False, jitter_std=0.01, seed=42):
        self.X              = torch.tensor(X, dtype=torch.float32)
        self.y              = torch.tensor(y, dtype=torch.long)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        self.augment        = augment
        self.jitter_std     = jitter_std
        self.rng            = torch.Generator().manual_seed(seed)

    def __len__(self):   return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and self.jitter_std > 0:
            x = x + torch.randn(x.shape, generator=self.rng) * self.jitter_std
        return x, self.y[idx], self.sample_weights[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PHONOCODE BN — Stage 2: WavLM Similarity Classifier")
    print(f"  Model: {MODEL_ID}  |  Unfrozen layers: {N_UNFROZEN_LAYERS}")
    print(f"  Feature: [A, B, A-B, A*B]  |  N_FOLDS={N_FOLDS}")
    print("=" * 60)
    sys.stdout.flush()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")
    sys.stdout.flush()

    # ---- Load WavLM ----
    print(f"\nLoading WavLM: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    wavlm     = AutoModel.from_pretrained(MODEL_ID).to(device)

    # Freeze all parameters, then unfreeze top N transformer layers
    for param in wavlm.parameters():
        param.requires_grad = False

    # WavLM-Large has encoder.layers; unfreeze the last N
    encoder_layers = wavlm.encoder.layers
    n_layers = len(encoder_layers)
    for layer in encoder_layers[n_layers - N_UNFROZEN_LAYERS:]:
        for param in layer.parameters():
            param.requires_grad = True

    n_trainable = sum(p.numel() for p in wavlm.parameters() if p.requires_grad)
    n_total_p   = sum(p.numel() for p in wavlm.parameters())
    print(f"WavLM loaded: {n_total_p:,} total params, {n_trainable:,} trainable "
          f"(top {N_UNFROZEN_LAYERS} encoder layers)")
    sys.stdout.flush()

    # ---- Pre-encode reference recordings (clean, no augmentation) ----
    print("\nPre-encoding reference recordings...")
    wavlm.eval()
    ref_embeddings: Dict[str, np.ndarray] = {}
    for stim in STIMULI:
        ref_path = REF_ROOT / f"{stim}.wav"
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference not found: {ref_path}")
        audio, _ = librosa.load(ref_path, sr=TARGET_SR, mono=True)
        ref_embeddings[stim] = encode_audio(audio, processor, wavlm, device)
        print(f"  {stim}: dim={ref_embeddings[stim].shape[0]}")
    ref_dim = next(iter(ref_embeddings.values())).shape[0]
    sim_dim = ref_dim * 4   # [A, B, A-B, A*B]
    print(f"Reference dim={ref_dim}, similarity feature dim={sim_dim}")
    sys.stdout.flush()

    # ---- Build label map ----
    label_map = build_label_map(GROUND_TRUTH_CSV)
    print(f"\nLabel map: {len(label_map)} entries")

    # ---- Extract CLEAN similarity embeddings for all participants ----
    print("\nExtracting clean similarity features (all participants)...")
    wav_files = sorted(AUDIO_ROOT.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No WAV files found under {AUDIO_ROOT}")
    print(f"Found {len(wav_files)} WAV files")
    sys.stdout.flush()

    all_X:    List[np.ndarray] = []
    all_y:    List[int]        = []
    all_pids: List[str]        = []
    all_words: List[str]       = []
    all_durations: List[float] = []
    skipped_label = skipped_duration = 0

    wavlm.eval()
    for i, wav_path in enumerate(wav_files, 1):
        try:
            participant_id, _, target_word = parse_filename(wav_path)
        except ValueError:
            continue

        word_norm = normalize_word(target_word)
        key = (participant_id, word_norm)
        if key not in label_map:
            skipped_label += 1
            continue

        if word_norm not in ref_embeddings:
            continue

        audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        duration_ms = len(audio) / sr * 1000.0
        if duration_ms > DURATION_THRESHOLD_MS:
            skipped_duration += 1
            continue

        emb_resp = encode_audio(audio, processor, wavlm, device)
        emb_ref  = ref_embeddings[word_norm]
        sim_feat = similarity_features(emb_resp, emb_ref)

        diff_w    = WORD_DIFFICULTY_WEIGHT.get(word_norm, 1.0)
        diff_map  = {0.5: 0.0, 1.0: 0.5, 2.0: 1.0}
        diff_feat = np.array([diff_map.get(diff_w, 0.5)], dtype=np.float32)
        feat      = np.concatenate([sim_feat, diff_feat])   # [sim_dim + 1]

        all_X.append(feat)
        all_y.append(label_map[key])
        all_pids.append(participant_id)
        all_words.append(word_norm)
        all_durations.append(duration_ms)

        if i % 100 == 0:
            print(f"  {i}/{len(wav_files)} files processed ({len(all_X)} embeddings)...")
            sys.stdout.flush()

    X_all = np.stack(all_X, axis=0)
    y_all = np.array(all_y, dtype=np.int64)
    n_total = len(X_all)
    print(f"\nClean extraction done: {n_total} samples, dim={X_all.shape[1]}")
    print(f"  Skipped (no label): {skipped_label}  |  Skipped (duration): {skipped_duration}")
    lc = np.bincount(y_all)
    print(f"  Class 0: {lc[0]} ({lc[0]/n_total*100:.1f}%)  Class 1: {lc[1]} ({lc[1]/n_total*100:.1f}%)")
    sys.stdout.flush()

    input_dim = X_all.shape[1]

    # ---- LR baseline ----
    print("\n" + "=" * 60)
    print("Logistic Regression baseline")
    print("=" * 60)
    folds, unique_pids = participant_kfold(all_pids, N_FOLDS, RANDOM_SEED)
    lr_accs, lr_f1s = [], []
    for train_idx, test_idx in folds:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 n_jobs=-1, random_state=RANDOM_SEED)
        clf.fit(X_all[train_idx], y_all[train_idx])
        preds = clf.predict(X_all[test_idx])
        lr_accs.append(accuracy_score(y_all[test_idx], preds))
        lr_f1s.append(f1_score(y_all[test_idx], preds, average="macro"))
    lr_mean_acc, lr_std_acc = np.mean(lr_accs), np.std(lr_accs)
    lr_mean_f1,  lr_std_f1  = np.mean(lr_f1s),  np.std(lr_f1s)
    print(f"LR Accuracy: {lr_mean_acc:.3f} ± {lr_std_acc:.3f}  "
          f"Macro F1: {lr_mean_f1:.3f} ± {lr_std_f1:.3f}")
    sys.stdout.flush()

    # ---- MLP ----
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
            )
        def forward(self, x): return self.net(x)

    criterion = nn.CrossEntropyLoss(reduction='none')

    def eval_epoch(model, loader, threshold=0.5):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for xb, yb, wb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                losses = criterion(logits, yb)
                total_loss += losses.mean().item() * xb.size(0)
                probs  = torch.softmax(logits, dim=-1)
                preds  = (probs[:, 1] >= threshold).long()
                correct += (preds == yb).sum().item()
                total   += xb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        avg_loss = total_loss / total if total > 0 else 0.0
        acc      = correct / total if total > 0 else 0.0
        preds_np = np.array(all_preds); labels_np = np.array(all_labels)
        bal_acc  = balanced_accuracy_score(labels_np, preds_np)
        return avg_loss, acc, preds_np, labels_np, np.array(all_probs), bal_acc

    # ---- 5-fold CV ----
    print("\n" + "=" * 60)
    print(f"MLP 5-FOLD CV")
    print("=" * 60)

    fold_results     = []
    all_test_preds   = np.zeros(n_total, dtype=np.int64)
    all_test_labels  = np.zeros(n_total, dtype=np.int64)
    all_test_covered = np.zeros(n_total, dtype=bool)

    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        fold_dir = OUTPUT_DIR / f"fold{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        outer_train_pids = set(np.array(all_pids)[train_idx])
        test_pids_fold   = set(np.array(all_pids)[test_idx])

        inner_train_rel, inner_val_rel, inner_train_pids_list, inner_val_pids_list = \
            participant_holdout_split(
                list(np.array(all_pids)[train_idx]),
                val_fraction=0.2,
                seed=RANDOM_SEED + 1000 + fold_idx,
            )
        inner_train_idx = train_idx[inner_train_rel]
        inner_val_idx   = train_idx[inner_val_rel]

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{N_FOLDS}  |  "
              f"train pids: {len(set(np.array(all_pids)[inner_train_idx]))}  "
              f"val pids: {len(set(np.array(all_pids)[inner_val_idx]))}  "
              f"test pids: {len(test_pids_fold)}")
        sys.stdout.flush()

        # ---- Augmented training embeddings ----
        # Re-walk audio; generate augmented versions for inner-train participants.
        # Each augmented version gets a fresh participant response encoding,
        # but the reference encoding is always clean (reference is fixed).
        fold_rng = np.random.default_rng(RANDOM_SEED + fold_idx)
        train_pids_fold = set(np.array(all_pids)[inner_train_idx])

        aug_X: List[np.ndarray] = []
        aug_y: List[int]        = []
        aug_words_fold: List[str]   = []
        aug_durations_fold: List[float] = []

        wavlm.train()   # allow gradient flow through unfrozen layers
        for wav_path in wav_files:
            try:
                participant_id, _, target_word = parse_filename(wav_path)
            except ValueError:
                continue
            if participant_id not in train_pids_fold:
                continue
            word_norm = normalize_word(target_word)
            key = (participant_id, word_norm)
            if key not in label_map or word_norm not in ref_embeddings:
                continue

            audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
            duration_ms = len(audio) / sr * 1000.0
            if duration_ms > DURATION_THRESHOLD_MS:
                continue

            label     = label_map[key]
            emb_ref   = ref_embeddings[word_norm]
            diff_w    = WORD_DIFFICULTY_WEIGHT.get(word_norm, 1.0)
            diff_feat = np.array([{0.5: 0.0, 1.0: 0.5, 2.0: 1.0}.get(diff_w, 0.5)], dtype=np.float32)

            audio_versions = [audio] + [augment_audio(audio, sr, fold_rng) for _ in range(N_AUGMENT_COPIES)]

            for audio_v in audio_versions:
                with torch.no_grad():
                    emb_resp = encode_audio(audio_v, processor, wavlm, device)
                feat = np.concatenate([similarity_features(emb_resp, emb_ref), diff_feat])
                aug_X.append(feat)
                aug_y.append(label)
                aug_words_fold.append(word_norm)
                aug_durations_fold.append(duration_ms)

        wavlm.eval()

        X_train = np.stack(aug_X, axis=0)
        y_train = np.array(aug_y, dtype=np.int64)
        X_val   = X_all[inner_val_idx]
        y_val   = y_all[inner_val_idx]
        X_test  = X_all[test_idx]
        y_test  = y_all[test_idx]

        print(f"  Aug train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
        sys.stdout.flush()

        # Sample weights
        train_counts  = np.bincount(y_train, minlength=2).astype(np.float32)
        class_weights = train_counts.sum() / (2.0 * train_counts)
        diff_weights  = np.array([WORD_DIFFICULTY_WEIGHT.get(w, 1.0) for w in aug_words_fold], dtype=np.float32)
        dur_weights   = np.array([duration_weight(d) for d in aug_durations_fold], dtype=np.float32)
        sw = np.array([class_weights[y] * diff_weights[i] * dur_weights[i] for i, y in enumerate(y_train)], dtype=np.float32)
        sw = sw / sw.mean()

        train_ds = SimilarityDataset(X_train, y_train, sw, augment=True, jitter_std=0.01, seed=RANDOM_SEED + fold_idx)
        val_ds   = SimilarityDataset(X_val, y_val, np.ones(len(y_val), dtype=np.float32), augment=False)
        test_ds  = SimilarityDataset(X_test, y_test, np.ones(len(y_test), dtype=np.float32), augment=False)
        train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = data.DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader  = data.DataLoader(test_ds, batch_size=64, shuffle=False)

        torch.manual_seed(RANDOM_SEED + fold_idx)
        mlp       = MLP().to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=MLP_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6)

        num_epochs = 100
        best_val_bal_acc  = -1.0
        best_val_loss     = float("inf")
        best_val_acc      = 0.0
        best_epoch        = 0
        epochs_no_improve = 0
        history = {"train_loss": [], "train_loss_unweighted": [], "val_loss": [],
                   "train_acc": [], "val_acc": [], "val_bal_acc": []}

        for epoch in range(1, num_epochs + 1):
            mlp.train()
            run_loss_w, run_loss_uw, correct, total = 0.0, 0.0, 0, 0
            for xb, yb, wb in train_loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                optimizer.zero_grad()
                logits = mlp(xb)
                per_sample = criterion(logits, yb)
                loss = (per_sample * wb).mean()
                loss.backward()
                optimizer.step()
                run_loss_w  += loss.item() * xb.size(0)
                run_loss_uw += per_sample.mean().item() * xb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total   += xb.size(0)

            train_loss_w  = run_loss_w  / total
            train_loss_uw = run_loss_uw / total
            train_acc     = correct / total

            val_loss, val_acc, _, _, _, val_bal_acc = eval_epoch(mlp, val_loader)
            scheduler.step(val_loss)

            for k, v in zip(["train_loss","train_loss_unweighted","val_loss","train_acc","val_acc","val_bal_acc"],
                             [train_loss_w, train_loss_uw, val_loss, train_acc, val_acc, val_bal_acc]):
                history[k].append(v)

            improved = val_bal_acc > best_val_bal_acc + EARLY_STOPPING_MIN_DELTA
            if improved:
                best_val_bal_acc  = val_bal_acc
                best_val_loss     = val_loss
                best_val_acc      = val_acc
                best_epoch        = epoch
                epochs_no_improve = 0
                torch.save({'epoch': epoch, 'model_state_dict': mlp.state_dict(),
                            'val_bal_acc': val_bal_acc, 'fold': fold_idx},
                           fold_dir / "mlp_best_model.pt")
            else:
                epochs_no_improve += 1

            lr_now = optimizer.param_groups[0]['lr']
            tag = "  [✓]" if improved else f"  [{epochs_no_improve}/{EARLY_STOPPING_PATIENCE}]"
            print(f"  Ep {epoch:03d}  train_uw={train_loss_uw:.4f} acc={train_acc:.3f}  "
                  f"val={val_loss:.4f} acc={val_acc:.3f} bal={val_bal_acc:.3f}  lr={lr_now:.2e}{tag}")
            sys.stdout.flush()

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stop at epoch {epoch}.")
                break

        actual_epochs = len(history["train_loss"])
        train_loss_at_best = history["train_loss_unweighted"][best_epoch - 1]
        tv_gap = best_val_loss - train_loss_at_best

        ckpt = torch.load(fold_dir / "mlp_best_model.pt", weights_only=True)
        mlp.load_state_dict(ckpt['model_state_dict'])

        best_threshold, best_thresh_bac = 0.5, -1.0
        for thr in THRESHOLD_CANDIDATES:
            _, _, _, _, _, thr_bac = eval_epoch(mlp, val_loader, threshold=thr)
            if thr_bac > best_thresh_bac:
                best_thresh_bac, best_threshold = thr_bac, thr

        _, _, fold_preds, fold_labels, fold_probs, fold_bal_acc = eval_epoch(mlp, test_loader, threshold=best_threshold)
        fold_cm   = confusion_matrix(fold_labels, fold_preds)
        fold_rpt  = classification_report(fold_labels, fold_preds, digits=3, output_dict=False)
        fold_dict = classification_report(fold_labels, fold_preds, digits=3, output_dict=True)
        fold_acc      = fold_dict['accuracy']
        fold_macro_f1 = fold_dict['macro avg']['f1-score']
        fold_cl0_rec  = fold_dict['0']['recall']
        fold_cl1_rec  = fold_dict['1']['recall']

        print(f"\n  [FOLD {fold_idx} TEST]  acc={fold_acc:.3f}  bal_acc={fold_bal_acc:.3f}  "
              f"macro_f1={fold_macro_f1:.3f}  cl0={fold_cl0_rec:.3f}  cl1={fold_cl1_rec:.3f}  thr={best_threshold:.2f}")
        print(f"  CM:\n{fold_cm}")
        sys.stdout.flush()

        all_test_preds[test_idx]   = fold_preds
        all_test_labels[test_idx]  = fold_labels
        all_test_covered[test_idx] = True

        # Save fold outputs
        with open(fold_dir / "test_results.txt", "w") as f:
            f.write(f"Fold: {fold_idx}\n")
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Stopped at: {actual_epochs}\n")
            f.write(f"TV-gap: {tv_gap:.4f}\n")
            f.write(f"Threshold: {best_threshold:.2f}\n")
            f.write(f"Test Acc: {fold_acc:.3f}\n")
            f.write(f"Test Bal Acc: {fold_bal_acc:.3f}\n")
            f.write(f"Macro F1: {fold_macro_f1:.3f}\n")
            f.write(f"Cl0 Recall: {fold_cl0_rec:.3f}\n")
            f.write(f"Cl1 Recall: {fold_cl1_rec:.3f}\n")
            f.write(f"\nCM:\n{fold_cm}\n\n{fold_rpt}\n")

        pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
        pd.DataFrame({'true': fold_labels, 'pred': fold_preds, 'prob1': fold_probs}).to_csv(
            fold_dir / "test_predictions.csv", index=False)

        epochs_r = range(1, actual_epochs + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_r, history["train_loss_unweighted"], label="Train loss")
        plt.plot(epochs_r, history["val_loss"], label="Val loss")
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
        plt.title(f"Loss — Fold {fold_idx}"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(epochs_r, history["train_acc"], label="Train acc")
        plt.plot(epochs_r, history["val_acc"], label="Val acc")
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
        plt.title(f"Acc — Fold {fold_idx}"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fold_dir / "learning_curves.png", dpi=150, bbox_inches='tight')
        plt.close()

        fold_results.append({
            'fold': fold_idx, 'best_epoch': best_epoch, 'stopped_epoch': actual_epochs,
            'tv_gap': tv_gap, 'threshold': best_threshold,
            'test_acc': fold_acc, 'test_bal_acc': fold_bal_acc,
            'macro_f1': fold_macro_f1, 'cl0_recall': fold_cl0_rec, 'cl1_recall': fold_cl1_rec,
        })

    # ---- Aggregate ----
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(OUTPUT_DIR / "cv_fold_results.csv", index=False)

    mean_acc  = results_df['test_acc'].mean();       std_acc  = results_df['test_acc'].std()
    mean_bac  = results_df['test_bal_acc'].mean();   std_bac  = results_df['test_bal_acc'].std()
    mean_f1   = results_df['macro_f1'].mean();       std_f1   = results_df['macro_f1'].std()
    mean_cl0  = results_df['cl0_recall'].mean();     std_cl0  = results_df['cl0_recall'].std()
    mean_cl1  = results_df['cl1_recall'].mean();     std_cl1  = results_df['cl1_recall'].std()
    mean_gap  = results_df['tv_gap'].mean()

    agg_cm   = confusion_matrix(all_test_labels, all_test_preds)
    agg_bac  = balanced_accuracy_score(all_test_labels, all_test_preds)
    agg_rpt  = classification_report(all_test_labels, all_test_preds, digits=3)
    agg_dict = classification_report(all_test_labels, all_test_preds, output_dict=True, digits=3)

    with open(OUTPUT_DIR / "cv_aggregate_results.txt", "w") as f:
        f.write("Stage 2: WavLM Similarity Classifier\n")
        f.write(f"Model: {MODEL_ID}  |  Unfrozen layers: {N_UNFROZEN_LAYERS}\n\n")
        f.write(results_df[['fold','test_acc','test_bal_acc','macro_f1','cl0_recall','cl1_recall',
                             'threshold','tv_gap','best_epoch','stopped_epoch']].to_string(index=False))
        f.write(f"\n\nMean ± Std over {N_FOLDS} folds:\n")
        f.write(f"  Accuracy:       {mean_acc:.3f} ± {std_acc:.3f}\n")
        f.write(f"  Balanced Acc:   {mean_bac:.3f} ± {std_bac:.3f}\n")
        f.write(f"  Macro F1:       {mean_f1:.3f} ± {std_f1:.3f}\n")
        f.write(f"  Class-0 Recall: {mean_cl0:.3f} ± {std_cl0:.3f}\n")
        f.write(f"  Class-1 Recall: {mean_cl1:.3f} ± {std_cl1:.3f}\n")
        f.write(f"  Mean TV-gap:    {mean_gap:.4f}\n\n")
        f.write(f"Aggregate CM:\n{agg_cm}\n\nAggregate report:\n{agg_rpt}\n")
        f.write(f"\nLR baseline: acc={lr_mean_acc:.3f}±{lr_std_acc:.3f}  f1={lr_mean_f1:.3f}±{lr_std_f1:.3f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(agg_cm, cmap='Blues')
    axes[0].set_title(f"WavLM Sim [Stage 2]\nAcc={agg_dict['accuracy']:.3f}  "
                      f"BalAcc={agg_bac:.3f}  F1={agg_dict['macro avg']['f1-score']:.3f}")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    for ii in range(2):
        for jj in range(2):
            axes[0].text(jj, ii, str(agg_cm[ii, jj]), ha='center', va='center', fontsize=16)
    plt.colorbar(im, ax=axes[0])
    axes[1].bar(range(1, N_FOLDS+1), results_df['macro_f1'], color='steelblue', alpha=0.8)
    axes[1].axhline(mean_f1, color='r', linestyle='--', linewidth=2, label=f'Mean={mean_f1:.3f}')
    axes[1].set_xlabel("Fold"); axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Macro F1 per Fold [WavLM Sim]")
    axes[1].set_xticks(range(1, N_FOLDS+1)); axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y'); axes[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cv_summary_plot.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 60)
    print("STAGE 2 SUMMARY")
    print("=" * 60)
    print(f"  Accuracy:       {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  Balanced Acc:   {mean_bac:.3f} ± {std_bac:.3f}")
    print(f"  Macro F1:       {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"  Class-0 Recall: {mean_cl0:.3f} ± {std_cl0:.3f}")
    print(f"  Class-1 Recall: {mean_cl1:.3f} ± {std_cl1:.3f}")
    print(f"\nAggregate CM:\n{agg_cm}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
