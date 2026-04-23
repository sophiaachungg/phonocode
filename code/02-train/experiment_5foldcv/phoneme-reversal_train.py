import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import sys
import json

import numpy as np
import pandas as pd
import torch
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as data
from transformers import AutoProcessor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, f1_score,
                              balanced_accuracy_score)

import torchaudio
import torchaudio.functional as TAF

# ---------- CONFIG ----------
AUDIO_ROOT       = Path("../data_processed/phoneme_reversal")
GROUND_TRUTH_CSV = Path("../scoring/phoneme_reversal_ground_truth.csv")

MODEL_ID    = "facebook/wav2vec2-base-960h"
TARGET_SR   = 16000
RANDOM_SEED = 42

OUTPUT_DIR = Path("../results_phoneme-reversal")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- word to skip ----
SKIP_WORDS = {"an"}

# ---- difficulty tiers & weights ----
WORD_DIFFICULTY_WEIGHT: Dict[str, float] = {
    "sit": 0.5, "be": 0.5, "pet": 0.5, "sun": 0.5,
    "to":  0.5, "do": 0.5, "speed": 0.5, "in": 0.5,
    "at":  1.0, "see": 1.0, "seven": 1.0, "spoon": 1.0,
    "dime": 1.0, "pile": 1.0, "cheek": 1.0,
    "state": 2.0, "boots": 2.0, "system": 2.0,
    "midnight": 2.0, "baseball": 2.0, "sometimes": 2.0,
}

# ---- early stopping ----
EARLY_STOPPING_PATIENCE  = 10
EARLY_STOPPING_MIN_DELTA = 1e-4

# ---- duration penalty weight ----
# Recordings at or below threshold get weight 1.0.
# Weight decays linearly to a floor as duration increases beyond threshold.
# This doesn't change what the model sees — it changes how much each
# sample contributes to the loss, so noisy long recordings have less
# influence on the decision boundary.
DURATION_THRESHOLD_MS  = 8000   # tune based on outlier analysis
DURATION_WEIGHT_FLOOR  = 0.25   # minimum weight for the longest recordings

# ---- augmentation ----
# AUG_STRETCH_PROB reduced to 0.25 (from v4_1 onward)
AUG_NOISE_PROB   = 0.5
AUG_NOISE_SNR_DB = (10, 30)
AUG_PITCH_PROB   = 0.5
AUG_PITCH_STEPS  = (-2, 2)
AUG_STRETCH_PROB = 0.25
AUG_STRETCH_RANGE = (0.9, 1.1)

# ---- cv1: locked hyperparameters (winners from v4_1 / v4_2 sweeps) ----
MLP_LR       = 1e-4
WEIGHT_DECAY = 2e-4

# ---- cv1: cross-validation ----
# Participant-grouped 5-fold CV. Each fold trains on ~80% of participants
# and evaluates on ~20%, ensuring no participant appears in both train and
# test within a fold. This is the correct unit of independence — samples
# from the same participant are correlated so they must not be split across
# train/test within a fold.
#
# Augmentation is applied to each fold's training participants only,
# exactly as in the single-split runs. The test fold always uses clean
# embeddings so per-fold metrics are directly comparable.
#
# n_augment_copies=2 kept from v4 (3× train size per fold).
N_FOLDS          = 5
N_AUGMENT_COPIES = 2
# -----------------------------


# ---------------------------------------------------------------------------
# Acoustic augmentation helpers
# ---------------------------------------------------------------------------

def duration_weight(duration_ms: float) -> float:
    """
    Return a loss weight in [DURATION_WEIGHT_FLOOR, 1.0] for a recording.

    Recordings at or below DURATION_THRESHOLD_MS get weight 1.0.
    Weight decays linearly to DURATION_WEIGHT_FLOOR over the next
    DURATION_THRESHOLD_MS of excess length, then stays at the floor.
    Augmented copies inherit the weight of their source audio.
    """
    if duration_ms <= DURATION_THRESHOLD_MS:
        return 1.0
    excess = (duration_ms - DURATION_THRESHOLD_MS) / DURATION_THRESHOLD_MS
    w = 1.0 - (1.0 - DURATION_WEIGHT_FLOOR) * min(excess, 1.0)
    return max(w, DURATION_WEIGHT_FLOOR)


def augment_audio(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """
    Apply stochastic acoustic augmentation to a raw waveform.

    All three transforms are applied independently, each with its own
    probability, so the model sees: clean, noise-only, pitch-only,
    stretch-only, and all combinations during training.

    Parameters
    ----------
    audio : np.ndarray  shape [T], float32, normalised to [-1, 1]
    sr    : int         sample rate (should be TARGET_SR = 16000)
    rng   : np.random.Generator  seeded RNG for reproducibility

    Returns
    -------
    np.ndarray  augmented waveform, same shape and dtype as input
    """
    audio = audio.copy().astype(np.float32)

    # 1. Background noise injection
    #    Gaussian white noise at a random SNR within AUG_NOISE_SNR_DB.
    #    White noise is a reasonable stand-in for room ambience / mic noise
    #    and doesn't require a real noise corpus.
    if rng.random() < AUG_NOISE_PROB:
        snr_db  = rng.uniform(*AUG_NOISE_SNR_DB)
        sig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
        noise_rms = sig_rms / (10 ** (snr_db / 20))
        noise = rng.standard_normal(len(audio)).astype(np.float32) * noise_rms
        audio = audio + noise

    # 2. Pitch shifting  (via torchaudio)
    #    Shifts fundamental frequency without changing duration.
    #    ±2 semitones keeps the speech recognisably the same phonemes
    #    while breaking the model's ability to use speaker pitch as a cue.
    if rng.random() < AUG_PITCH_PROB:
        n_steps = rng.uniform(*AUG_PITCH_STEPS)
        wav_t   = torch.from_numpy(audio).unsqueeze(0)   # [1, T]
        wav_t   = TAF.pitch_shift(wav_t, sr, n_steps=float(n_steps))
        audio   = wav_t.squeeze(0).numpy()

    # 3. Time stretching  (via torchaudio phase vocoder)
    #    Stretches or compresses duration without changing pitch.
    #    ±10% keeps phoneme boundaries intact while varying speaking rate.
    if rng.random() < AUG_STRETCH_PROB:
        rate  = rng.uniform(*AUG_STRETCH_RANGE)
        wav_t = torch.from_numpy(audio).unsqueeze(0)   # [1, T]
        # torchaudio.functional.resample achieves a speed change;
        # we resample to rate*sr then back to sr (equivalent to time stretch)
        new_sr  = int(sr * rate)
        wav_t   = TAF.resample(wav_t, orig_freq=new_sr, new_freq=sr)
        audio   = wav_t.squeeze(0).numpy()

    # Clip to [-1, 1] to prevent clipping artefacts from chained transforms
    audio = np.clip(audio, -1.0, 1.0)
    return audio


# ---------------------------------------------------------------------------
# Filename / label utilities
# ---------------------------------------------------------------------------

def parse_filename(path: Path) -> Tuple[str, str, str]:
    """
    Expect filenames like: ReXa_149_01_an.wav
    Returns (participant_id, audio_num, target_word).
    participant_id may itself contain underscores.
    """
    stem  = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {path}")
    target_word    = parts[-1]
    audio_num      = parts[-2]
    participant_id = "_".join(parts[:-2])
    return participant_id, audio_num, target_word


def normalize_word(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z']+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_label_map(gt_path: Path) -> Dict[Tuple[str, str], int]:
    """Returns dict mapping (participant_id, target_word) -> label (0/1)."""
    gt = pd.read_csv(gt_path)

    id_col       = "participant_id"
    non_word_cols = ["RA_preversal"]
    word_cols    = [c for c in gt.columns if c not in [id_col] + non_word_cols]

    gt_long = gt.melt(
        id_vars=[id_col],
        value_vars=word_cols,
        var_name="target_word",
        value_name="ground_truth_label",
    )
    gt_long["target_word_norm"] = gt_long["target_word"].apply(normalize_word)

    print(f"Total entries before dropping NaN: {len(gt_long)}")
    gt_long = gt_long.dropna(subset=["ground_truth_label"])
    print(f"Total entries after dropping NaN: {len(gt_long)}")
    sys.stdout.flush()

    gt_long["ground_truth_label"] = gt_long["ground_truth_label"].astype(int)

    label_map: Dict[Tuple[str, str], int] = {}
    for _, row in gt_long.iterrows():
        key = (str(row[id_col]), row["target_word_norm"])
        label_map[key] = int(row["ground_truth_label"])
    return label_map


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    audio_root: Path,
    processor,
    model,
    device,
    label_map: Dict[Tuple[str, str], int],
    augment: bool = False,
    n_augment_copies: int = 2,
    rng: Optional[np.random.Generator] = None,
):
    """
    Iterate over wav files, extract frozen Wav2Vec2 embeddings + labels.

    v4 changes
    ----------
    SKIP_WORDS : 'an' is skipped unconditionally (audio files untouched).

    HYBRID POOLING (from v3): mean and max hidden states are concatenated
    along the feature axis, giving embedding dim 1536 (= 2 × 768).

    DIFFICULTY FEATURE: a single scalar representing the word's phonological
    difficulty tier (0.0=easy, 0.5=medium, 1.0=hard) is appended to every
    embedding. This gives the MLP an explicit context signal so it can learn
    a difficulty-conditional decision boundary. Final dim = 1537.

    ACOUSTIC AUGMENTATION (training only): when augment=True, each training
    audio clip is run through augment_audio() n_augment_copies additional
    times, each with a fresh random transform. The original clean embedding
    is always included. This means each training sample appears
    (1 + n_augment_copies) times with different acoustic surface forms,
    forcing the encoder to focus on phonological structure rather than
    speaker identity or recording conditions.

    Returns
    -------
    X            : np.ndarray  [N, 1537]
    y            : np.ndarray  [N]
    participants : List[str]   length N
    words        : List[str]   length N  (for difficulty weight lookup)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    wav_files = sorted(audio_root.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No .wav files found under {audio_root}")
    print(f"Found {len(wav_files)} wav files")
    sys.stdout.flush()

    embeddings:   List[np.ndarray] = []
    labels:       List[int]        = []
    participants: List[str]        = []
    words:        List[str]        = []

    skipped_word  = 0
    skipped_label = 0

    for i, wav_path in enumerate(wav_files, 1):
        try:
            participant_id, audio_num, target_word = parse_filename(wav_path)
        except ValueError as e:
            print(f"[SKIP] {e}")
            continue

        word_norm = normalize_word(target_word)

        # ---- v4: skip 'an' (and any future words added to SKIP_WORDS) ----
        if word_norm in SKIP_WORDS:
            skipped_word += 1
            continue

        key = (participant_id, word_norm)
        if key not in label_map:
            skipped_label += 1
            continue
        label = label_map[key]

        # Load raw audio once
        audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

        # Build list of audio versions: always start with clean original
        audio_versions = [audio]
        if augment:
            for _ in range(n_augment_copies):
                audio_versions.append(augment_audio(audio, sr, rng))

        # Encode each version and collect embeddings
        for audio_v in audio_versions:
            inputs = processor(
                audio_v,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding="longest",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                hidden  = outputs.last_hidden_state   # [1, T, D]

                # Hybrid pooling (v3): mean + max concatenated -> [2*D]
                mean_emb = hidden.mean(dim=1).squeeze(0)
                max_emb  = hidden.max(dim=1).values.squeeze(0)
                pooled   = torch.cat([mean_emb, max_emb], dim=0)  # [1536]

            # ---- v4: difficulty feature ----
            # Map difficulty weight -> normalised scalar in [0, 1]
            #   easy   (0.5) -> 0.0
            #   medium (1.0) -> 0.5
            #   hard   (2.0) -> 1.0
            diff_w     = WORD_DIFFICULTY_WEIGHT.get(word_norm, 1.0)
            diff_map   = {0.5: 0.0, 1.0: 0.5, 2.0: 1.0}
            diff_feat  = np.array([diff_map.get(diff_w, 0.5)], dtype=np.float32)

            emb = np.concatenate([pooled.cpu().numpy(), diff_feat])  # [1537]

            embeddings.append(emb)
            labels.append(label)
            participants.append(participant_id)
            words.append(word_norm)

        if i % 50 == 0:
            print(f"Processed {i}/{len(wav_files)} files "
                  f"({len(embeddings)} embeddings so far)...")
            sys.stdout.flush()

    if not embeddings:
        raise RuntimeError("No embeddings extracted. Check label_map / filenames.")

    print(f"Skipped {skipped_word} files matching SKIP_WORDS {SKIP_WORDS}.")
    if skipped_label > 0:
        print(f"Skipped {skipped_label} files with no matching label.")

    X = np.stack(embeddings, axis=0)
    y = np.array(labels, dtype=np.int64)
    return X, y, participants, words


# ---------------------------------------------------------------------------
# Participant-grouped split
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Participant-grouped K-Fold splitter
# ---------------------------------------------------------------------------

def participant_kfold(
    participants: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, test_idx) arrays for each fold such that all samples
    from a given participant land entirely in train or entirely in test.

    This is the correct CV strategy for this dataset: samples from the same
    participant are correlated (same speaker, same session), so mixing them
    across train/test within a fold would constitute data leakage and inflate
    performance estimates.

    Process
    -------
    1. Collect the unique participant IDs and shuffle them with a fixed seed.
    2. Apply sklearn KFold to the *participant* list (not the sample list).
    3. Map participant-level fold assignments back to sample indices.

    Returns a list of (train_idx, test_idx) tuples, one per fold.
    """
    rng          = np.random.default_rng(seed)
    participants = np.array(participants)
    unique_pids  = np.array(sorted(set(participants)))
    rng.shuffle(unique_pids)

    kf     = KFold(n_splits=n_splits, shuffle=False)  # shuffle already done above
    folds  = []

    for pid_train_idx, pid_test_idx in kf.split(unique_pids):
        train_pids = set(unique_pids[pid_train_idx])
        test_pids  = set(unique_pids[pid_test_idx])

        train_idx = np.array([i for i, p in enumerate(participants) if p in train_pids], dtype=np.int64)
        test_idx  = np.array([i for i, p in enumerate(participants) if p in test_pids],  dtype=np.int64)

        folds.append((train_idx, test_idx))

    return folds, unique_pids


def participant_holdout_split(
    participants: List[str],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Split a participant list into disjoint train/validation partitions.

    The split is performed at the participant level so that no speaker can
    appear in both subsets. This is used inside each outer CV fold to create a
    clean early-stopping validation set without touching the outer test fold.
    """
    participants = np.array(participants)
    unique_pids = np.array(sorted(set(participants)))
    if len(unique_pids) < 2:
        raise ValueError("Need at least 2 participants to make a train/val split")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pids)

    n_val = int(round(len(unique_pids) * val_fraction))
    n_val = max(1, n_val)
    n_val = min(n_val, len(unique_pids) - 1)

    val_pids = set(unique_pids[:n_val])
    train_pids = set(unique_pids[n_val:])

    train_idx = np.array([i for i, p in enumerate(participants) if p in train_pids], dtype=np.int64)
    val_idx = np.array([i for i, p in enumerate(participants) if p in val_pids], dtype=np.int64)

    return train_idx, val_idx, sorted(train_pids), sorted(val_pids)


# ---------------------------------------------------------------------------
# Augmented Dataset wrapper
# ---------------------------------------------------------------------------

class AugmentedEmbeddingDataset(data.Dataset):
    """
    Wraps pre-extracted (X, y, difficulty_weights) tensors.

    During training, a small amount of Gaussian jitter is applied directly
    to the embedding vector as a second-pass regulariser. This is separate
    from the audio-level augmentation above: the audio transforms create
    acoustically distinct embeddings; the embedding jitter adds fine-grained
    perturbation in representation space.

    Jitter std is kept very small (0.01) so it perturbs without distorting.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        augment: bool = False,
        jitter_std: float = 0.01,
        seed: int = 42,
    ):
        self.X              = torch.tensor(X, dtype=torch.float32)
        self.y              = torch.tensor(y, dtype=torch.long)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        self.augment        = augment
        self.jitter_std     = jitter_std
        self.rng            = torch.Generator()
        self.rng.manual_seed(seed)

    def __len__(self):
        return len(self.y)

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
    print("PHONOCODE: Wav2Vec2 Frozen  [cv1]  5-Fold Cross-Validation")
    print(f"  + {N_FOLDS}-fold participant-grouped CV")
    print(f"  + LR={MLP_LR:.1e}  WD={WEIGHT_DECAY:.1e}  (locked from v4_1/v4_2)")
    print(f"  + AUG_STRETCH_PROB={AUG_STRETCH_PROB}")
    print(f"  + Duration penalty: threshold={DURATION_THRESHOLD_MS}ms  floor={DURATION_WEIGHT_FLOOR}")
    print("  + Embeddings extracted once from all participants")
    print("  + Augmentation applied to inner-train split only")
    print("  + Selection criterion: val_loss on inner validation split")
    print("  + Final report: mean ± std over outer test folds (macro F1, acc, recall)")
    print("=" * 60)
    sys.stdout.flush()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
    sys.stdout.flush()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # ---- Load encoder (frozen, shared across all folds) ----
    print(f"\nLoading Wav2Vec2 encoder: {MODEL_ID}")
    sys.stdout.flush()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    wav2vec   = AutoModel.from_pretrained(MODEL_ID)
    wav2vec.to(device)
    wav2vec.eval()
    for param in wav2vec.parameters():
        param.requires_grad = False
    print("Encoder loaded and frozen.")
    sys.stdout.flush()

    # ---- Build label map ----
    print(f"\nBuilding label map from {GROUND_TRUTH_CSV} ...")
    sys.stdout.flush()
    label_map = build_label_map(GROUND_TRUTH_CSV)
    print(f"Label map entries: {len(label_map)}")
    sys.stdout.flush()

    # ---- Extract CLEAN embeddings for all participants once ----
    # Augmented copies are generated per-fold inside the loop so each fold's
    # augmented training set is independent. Clean embeddings are used for
    # the test split of every fold (and for the LR baseline).
    print(f"\nExtracting CLEAN embeddings (all participants, no augmentation)...")
    print(f"Skipping words: {SKIP_WORDS}")
    sys.stdout.flush()

    X_all, y_all, participants_all, words_all = extract_embeddings(
        AUDIO_ROOT, processor, wav2vec, device, label_map,
        augment=False, rng=rng,
    )
    n_total = len(X_all)
    print(f"Clean extraction complete: {n_total} samples, dim={X_all.shape[1]}")
    label_counts = np.bincount(y_all)
    print(f"Label distribution: class 0={label_counts[0]} ({label_counts[0]/n_total*100:.1f}%)  "
          f"class 1={label_counts[1]} ({label_counts[1]/n_total*100:.1f}%)")
    sys.stdout.flush()

    # ---- LR baseline (clean embeddings, all data, leave-one-fold-out) ----
    # Run LR CV using the same folds so the comparison is fair.
    print("\n" + "=" * 60)
    print("Logistic Regression baseline (participant-grouped CV)")
    print("=" * 60)
    sys.stdout.flush()

    folds, unique_pids = participant_kfold(participants_all, N_FOLDS, RANDOM_SEED)
    print(f"Participants: {len(unique_pids)} total, {N_FOLDS} folds "
          f"(~{len(unique_pids)//N_FOLDS} test participants per fold)")
    sys.stdout.flush()

    lr_fold_accs, lr_fold_f1s = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        clf = LogisticRegression(
            max_iter=1000, solver="lbfgs",
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED,
        )
        clf.fit(X_all[train_idx], y_all[train_idx])
        preds = clf.predict(X_all[test_idx])
        lr_fold_accs.append(accuracy_score(y_all[test_idx], preds))
        lr_fold_f1s.append(f1_score(y_all[test_idx], preds, average='macro'))

    lr_mean_acc = np.mean(lr_fold_accs)
    lr_std_acc  = np.std(lr_fold_accs)
    lr_mean_f1  = np.mean(lr_fold_f1s)
    lr_std_f1   = np.std(lr_fold_f1s)
    print(f"LR CV Accuracy:  {lr_mean_acc:.3f} ± {lr_std_acc:.3f}")
    print(f"LR CV Macro F1:  {lr_mean_f1:.3f} ± {lr_std_f1:.3f}")
    sys.stdout.flush()

    # ---- MLP definition ----
    input_dim   = X_all.shape[1]   # 1537
    num_classes = 2

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )
        def forward(self, x):
            return self.net(x)

    criterion = nn.CrossEntropyLoss(reduction='none')

    def eval_epoch(model, loader, threshold: float = 0.5):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for xb, yb, wb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                losses  = criterion(logits, yb)
                total_loss += losses.mean().item() * xb.size(0)
                probs   = torch.softmax(logits, dim=-1)
                # Apply threshold: predict class 1 only if p(class1) >= threshold
                preds   = (probs[:, 1] >= threshold).long()
                correct += (preds == yb).sum().item()
                total   += xb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        avg_loss      = total_loss / total if total > 0 else 0.0
        acc           = correct / total if total > 0 else 0.0
        all_preds_np  = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        bal_acc       = balanced_accuracy_score(all_labels_np, all_preds_np)
        return avg_loss, acc, all_preds_np, all_labels_np, np.array(all_probs), bal_acc

    # ---- 5-fold CV ----
    print("\n" + "=" * 60)
    print(f"MLP 5-FOLD CV  (LR={MLP_LR:.1e}, WD={WEIGHT_DECAY:.1e})")
    print("=" * 60)
    sys.stdout.flush()

    fold_results  = []   # one dict per outer fold
    all_test_preds   = np.zeros(n_total, dtype=np.int64)
    all_test_labels  = np.zeros(n_total, dtype=np.int64)
    all_test_covered = np.zeros(n_total, dtype=bool)

    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        fold_dir = OUTPUT_DIR / f"fold{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        # Outer-fold participant sets
        outer_train_pids = set(np.array(participants_all)[train_idx])
        test_pids_fold   = set(np.array(participants_all)[test_idx])

        # Inner split: hold out a participant-level validation set from the
        # outer training participants only. This keeps the outer test fold
        # completely untouched until the final evaluation step.
        inner_train_rel_idx, inner_val_rel_idx, inner_train_pids_list, inner_val_pids_list = participant_holdout_split(
            list(np.array(participants_all)[train_idx]),
            val_fraction=0.2,
            seed=RANDOM_SEED + 1000 + fold_idx,
        )
        inner_train_idx = train_idx[inner_train_rel_idx]
        inner_val_idx   = train_idx[inner_val_rel_idx]

        train_pids_fold = set(np.array(participants_all)[inner_train_idx])
        val_pids_fold   = set(np.array(participants_all)[inner_val_idx])

        n_train_clean = len(inner_train_idx)
        n_val         = len(inner_val_idx)
        n_test        = len(test_idx)
        n_outer_train = len(train_idx)

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{N_FOLDS}  —  "
              f"outer train: {len(outer_train_pids)} pids ({n_outer_train} samples)  |  "
              f"inner train: {len(train_pids_fold)} pids ({n_train_clean} samples)  |  "
              f"val: {len(val_pids_fold)} pids ({n_val} samples)  |  "
              f"test: {len(test_pids_fold)} pids ({n_test} samples)")
        print(f"  Inner-train participants: {sorted(train_pids_fold)}")
        print(f"  Inner-val participants:   {sorted(val_pids_fold)}")
        print(f"  Test participants:        {sorted(test_pids_fold)}")
        print(f"{'='*60}")
        sys.stdout.flush()

        # Save fold split info
        with open(fold_dir / "fold_split_info.json", "w") as f:
            json.dump({
                'fold': fold_idx,
                'outer_train_participants': sorted(outer_train_pids),
                'inner_train_participants': sorted(train_pids_fold),
                'inner_val_participants': sorted(val_pids_fold),
                'test_participants': sorted(test_pids_fold),
                'outer_train_samples': n_outer_train,
                'inner_train_samples_clean': n_train_clean,
                'val_samples': n_val,
                'test_samples': n_test,
            }, f, indent=2)

        # ---- Augmented train embeddings for this fold ----
        # We re-walk audio files, but only generate augmented copies for the
        # inner-train participants. The validation and test splits remain clean.
        fold_rng = np.random.default_rng(RANDOM_SEED + fold_idx)

        print(f"  Extracting augmented inner-train embeddings ({N_AUGMENT_COPIES} copies)...")
        sys.stdout.flush()

        wav_files = sorted(AUDIO_ROOT.rglob("*.wav"))
        aug_embeddings: List[np.ndarray] = []
        aug_labels:     List[int]        = []
        aug_words:      List[str]        = []
        aug_durations:  List[float]      = []   # ms; augmented copies inherit source duration

        for wav_path in wav_files:
            try:
                participant_id, _, target_word = parse_filename(wav_path)
            except ValueError:
                continue

            if participant_id not in train_pids_fold:
                continue

            word_norm = normalize_word(target_word)
            if word_norm in SKIP_WORDS:
                continue

            key = (participant_id, word_norm)
            if key not in label_map:
                continue
            label = label_map[key]

            audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
            duration_ms = len(audio) / sr * 1000.0   # computed once from clean audio

            audio_versions = [audio] + [
                augment_audio(audio, sr, fold_rng) for _ in range(N_AUGMENT_COPIES)
            ]

            for audio_v in audio_versions:
                inputs = processor(
                    audio_v, sampling_rate=TARGET_SR,
                    return_tensors="pt", padding="longest",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs  = wav2vec(**inputs)
                    hidden   = outputs.last_hidden_state
                    mean_emb = hidden.mean(dim=1).squeeze(0)
                    max_emb  = hidden.max(dim=1).values.squeeze(0)
                    pooled   = torch.cat([mean_emb, max_emb], dim=0)

                diff_w    = WORD_DIFFICULTY_WEIGHT.get(word_norm, 1.0)
                diff_map  = {0.5: 0.0, 1.0: 0.5, 2.0: 1.0}
                diff_feat = np.array([diff_map.get(diff_w, 0.5)], dtype=np.float32)
                emb = np.concatenate([pooled.cpu().numpy(), diff_feat])

                aug_embeddings.append(emb)
                aug_labels.append(label)
                aug_words.append(word_norm)
                aug_durations.append(duration_ms)   # same duration for all augmented copies

        if not aug_embeddings:
            raise RuntimeError(f"No augmented training embeddings extracted for fold {fold_idx}.")

        X_train = np.stack(aug_embeddings, axis=0)
        y_train = np.array(aug_labels, dtype=np.int64)

        # Clean validation and test sets (no augmentation)
        X_val = X_all[inner_val_idx]
        y_val = y_all[inner_val_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]

        print(f"  Augmented inner-train: {len(X_train)} embeddings  |  "
              f"Clean val: {len(X_val)} samples  |  Clean test: {len(X_test)} samples")
        sys.stdout.flush()

        # ---- Per-sample weights (class × difficulty × duration) ----
        # Fold-specific inverse-frequency weighting: data-driven, no manual tuning.
        # Formula: n_total / (n_classes * n_samples_per_class)
        train_counts  = np.bincount(y_train, minlength=2).astype(np.float32)
        class_weights = train_counts.sum() / (2.0 * train_counts)  # [w0, w1]
        diff_weights  = np.array(
            [WORD_DIFFICULTY_WEIGHT.get(w, 1.0) for w in aug_words],
            dtype=np.float32,
        )
        dur_weights = np.array(
            [duration_weight(d) for d in aug_durations],
            dtype=np.float32,
        )
        sample_weights = np.array(
            [class_weights[y] * diff_weights[i] * dur_weights[i]
             for i, y in enumerate(y_train)],
            dtype=np.float32,
        )
        sample_weights = sample_weights / sample_weights.mean()

        val_sw  = np.ones(len(y_val), dtype=np.float32)
        test_sw = np.ones(len(y_test), dtype=np.float32)

        train_ds = AugmentedEmbeddingDataset(
            X_train, y_train, sample_weights, augment=True, jitter_std=0.01, seed=RANDOM_SEED + fold_idx
        )
        val_ds = AugmentedEmbeddingDataset(
            X_val, y_val, val_sw, augment=False, jitter_std=0.0
        )
        test_ds = AugmentedEmbeddingDataset(
            X_test, y_test, test_sw, augment=False, jitter_std=0.0
        )

        train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = data.DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader  = data.DataLoader(test_ds, batch_size=64, shuffle=False)

        # ---- Train MLP ----
        torch.manual_seed(RANDOM_SEED + fold_idx)
        mlp       = MLP().to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=MLP_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6,
        )

        num_epochs         = 100
        best_val_bal_acc   = -1.0          # primary checkpoint criterion
        best_val_loss      = float("inf")  # tracked for logging / TV-gap only
        best_val_acc       = 0.0
        best_epoch         = 0
        epochs_no_improve  = 0
        history = {"train_loss": [], "train_loss_unweighted": [], "val_loss": [],
                   "train_acc": [], "val_acc": [], "val_bal_acc": []}

        # Threshold candidates to sweep on the validation set after training.
        # We pick the threshold that maximises balanced accuracy on val, then
        # apply it at test time. This is tuned per-fold so it sees no test data.
        THRESHOLD_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50]
        print(f"\n  Training MLP (fold {fold_idx})...")
        sys.stdout.flush()

        for epoch in range(1, num_epochs + 1):
            mlp.train()
            running_loss_weighted = 0.0
            running_loss_unweighted = 0.0
            correct, total = 0, 0

            for xb, yb, wb in train_loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                optimizer.zero_grad()
                logits          = mlp(xb)
                per_sample_loss = criterion(logits, yb)
                loss            = (per_sample_loss * wb).mean()
                loss.backward()
                optimizer.step()

                running_loss_weighted += loss.item() * xb.size(0)
                running_loss_unweighted += per_sample_loss.mean().item() * xb.size(0)
                preds    = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += xb.size(0)

            train_loss_weighted = running_loss_weighted / total if total > 0 else 0.0
            train_loss_unweighted = running_loss_unweighted / total if total > 0 else 0.0
            train_acc  = correct / total if total > 0 else 0.0

            val_loss, val_acc, _, _, _, val_bal_acc = eval_epoch(mlp, val_loader)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss_weighted)
            history["train_loss_unweighted"].append(train_loss_unweighted)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_bal_acc"].append(val_bal_acc)

            # Checkpoint on balanced accuracy (primary criterion — aligns with
            # clinical goal of catching class-0 errors without ignoring class-1).
            improved = val_bal_acc > best_val_bal_acc + EARLY_STOPPING_MIN_DELTA
            if improved:
                best_val_bal_acc  = val_bal_acc
                best_val_loss     = val_loss
                best_val_acc      = val_acc
                best_epoch        = epoch
                epochs_no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': mlp.state_dict(),
                    'val_loss': val_loss,
                    'val_acc':  val_acc,
                    'val_bal_acc': val_bal_acc,
                    'fold':     fold_idx,
                }, fold_dir / "mlp_best_model.pt")
            else:
                epochs_no_improve += 1

            current_lr = optimizer.param_groups[0]['lr']
            es_tag = "  [✓ best]" if improved else f"  [no imp {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}]"
            print(
                f"  Epoch {epoch:03d}/{num_epochs}  "
                f"train loss weighted={train_loss_weighted:.4f} train loss unweighted={train_loss_unweighted:.4f} acc={train_acc:.3f}  |  "
                f"val loss={val_loss:.4f} acc={val_acc:.3f} bal_acc={val_bal_acc:.3f}  "
                f"lr={current_lr:.2e}"
                f"{es_tag}"
            )
            sys.stdout.flush()

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}.")
                sys.stdout.flush()
                break

        actual_epochs = len(history["train_loss"])
        train_loss_at_best = history["train_loss_unweighted"][best_epoch - 1]
        tv_gap = best_val_loss - train_loss_at_best

        # Load best checkpoint for threshold tuning and final fold evaluation
        ckpt = torch.load(fold_dir / "mlp_best_model.pt", weights_only=True)
        mlp.load_state_dict(ckpt['model_state_dict'])

        # ---- Threshold tuning on validation set ----
        # Sweep candidate thresholds and pick the one that maximises balanced
        # accuracy on the val fold. The test set is never seen during this step.
        best_threshold   = 0.5
        best_thresh_bac  = -1.0
        print(f"\n  Threshold sweep (val set):")
        for thr in THRESHOLD_CANDIDATES:
            _, _, _, thr_labels, thr_probs, thr_bal_acc = eval_epoch(mlp, val_loader, threshold=thr)
            print(f"    threshold={thr:.2f}  val_bal_acc={thr_bal_acc:.3f}")
            if thr_bal_acc > best_thresh_bac:
                best_thresh_bac = thr_bal_acc
                best_threshold  = thr
        print(f"  → Selected threshold: {best_threshold:.2f}  (val_bal_acc={best_thresh_bac:.3f})")

        _, _, fold_preds, fold_labels, fold_probs, fold_bal_acc = eval_epoch(
            mlp, test_loader, threshold=best_threshold
        )
        fold_cm   = confusion_matrix(fold_labels, fold_preds)
        fold_rpt  = classification_report(fold_labels, fold_preds, digits=3, output_dict=False)
        fold_dict = classification_report(fold_labels, fold_preds, digits=3, output_dict=True)

        fold_acc      = fold_dict['accuracy']
        fold_macro_f1 = fold_dict['macro avg']['f1-score']
        fold_cl0_rec  = fold_dict['0']['recall']
        fold_cl1_rec  = fold_dict['1']['recall']

        print(f"\n  [FOLD {fold_idx} TEST]  "
              f"acc={fold_acc:.3f}  bal_acc={fold_bal_acc:.3f}  macro_f1={fold_macro_f1:.3f}  "
              f"cl0_recall={fold_cl0_rec:.3f}  cl1_recall={fold_cl1_rec:.3f}  "
              f"threshold={best_threshold:.2f}")
        print(f"  Confusion matrix:\n{fold_cm}")
        print(f"  Classification report:\n{fold_rpt}")
        sys.stdout.flush()

        # Accumulate predictions for the aggregate confusion matrix
        all_test_preds[test_idx]   = fold_preds
        all_test_labels[test_idx]  = fold_labels
        all_test_covered[test_idx] = True

        # Save per-fold outputs
        with open(fold_dir / "test_results.txt", "w") as f:
            f.write(f"Fold:             {fold_idx}\n")
            f.write(f"Outer train participants:{sorted(outer_train_pids)}\n")
            f.write(f"Inner train participants:{sorted(train_pids_fold)}\n")
            f.write(f"Validation participants:{sorted(val_pids_fold)}\n")
            f.write(f"Test participants:{sorted(test_pids_fold)}\n")
            f.write(f"Best epoch:       {best_epoch}\n")
            f.write(f"Stopped at epoch: {actual_epochs}\n")
            f.write(f"Best val bal_acc: {best_val_bal_acc:.4f}  (acc={best_val_acc:.3f})\n")
            f.write(f"Best val loss:    {best_val_loss:.4f}\n")
            f.write(f"Train loss @ best:{train_loss_at_best:.4f}\n")
            f.write(f"Train/val gap:    {tv_gap:.4f}\n")
            f.write(f"Tuned threshold:  {best_threshold:.2f}  (val_bal_acc={best_thresh_bac:.3f})\n")
            f.write(f"Test Accuracy:    {fold_acc:.3f}\n")
            f.write(f"Test Bal Acc:     {fold_bal_acc:.3f}\n")
            f.write(f"Macro F1:         {fold_macro_f1:.3f}\n")
            f.write(f"Class-0 Recall:   {fold_cl0_rec:.3f}\n")
            f.write(f"Class-1 Recall:   {fold_cl1_rec:.3f}\n\n")
            f.write(f"Confusion Matrix:\n{fold_cm}\n\n")
            f.write(f"Classification Report:\n{fold_rpt}\n")

        pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
        pd.DataFrame({
            'true_label': fold_labels, 'predicted_label': fold_preds, 'prob_class_1': fold_probs
        }).to_csv(fold_dir / "test_predictions.csv", index=False)

        # Learning curves per fold
        epochs_range = range(1, actual_epochs + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history["train_loss_unweighted"], label="Train loss unweighted", linewidth=2)
        plt.plot(epochs_range, history["val_loss"],   label="Val loss",  linewidth=2)
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch})')
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"Loss — Fold {fold_idx}", fontsize=14)
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history["train_acc"], label="Train acc", linewidth=2)
        plt.plot(epochs_range, history["val_acc"],   label="Val acc",  linewidth=2)
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch})')
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.title(f"Accuracy — Fold {fold_idx}", fontsize=14)
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fold_dir / "learning_curves.png", dpi=150, bbox_inches='tight')
        plt.close()

        fold_results.append({
            'fold':           fold_idx,
            'outer_train_pids': sorted(outer_train_pids),
            'train_pids':     sorted(train_pids_fold),
            'val_pids':       sorted(val_pids_fold),
            'test_pids':      sorted(test_pids_fold),
            'n_outer_train':  len(train_idx),
            'n_train':        len(X_train),
            'n_val':          len(X_val),
            'n_test':         n_test,
            'best_epoch':     best_epoch,
            'stopped_epoch':  actual_epochs,
            'tv_gap':         tv_gap,
            'val_bal_acc':    best_val_bal_acc,
            'val_loss':       best_val_loss,
            'threshold':      best_threshold,
            'test_acc':       fold_acc,
            'test_bal_acc':   fold_bal_acc,
            'macro_f1':       fold_macro_f1,
            'cl0_recall':     fold_cl0_rec,
            'cl1_recall':     fold_cl1_rec,
        })

    # ---- Aggregate results ----
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(OUTPUT_DIR / "cv_fold_results.csv", index=False)

    # Aggregate confusion matrix (stitches fold predictions back together)
    agg_cm     = confusion_matrix(all_test_labels, all_test_preds)
    agg_rpt    = classification_report(all_test_labels, all_test_preds, digits=3)
    agg_dict   = classification_report(all_test_labels, all_test_preds, digits=3, output_dict=True)

    mean_acc      = results_df['test_acc'].mean()
    std_acc       = results_df['test_acc'].std()
    mean_bal_acc  = results_df['test_bal_acc'].mean()
    std_bal_acc   = results_df['test_bal_acc'].std()
    mean_f1       = results_df['macro_f1'].mean()
    std_f1        = results_df['macro_f1'].std()
    mean_cl0      = results_df['cl0_recall'].mean()
    std_cl0       = results_df['cl0_recall'].std()
    mean_cl1      = results_df['cl1_recall'].mean()
    std_cl1       = results_df['cl1_recall'].std()
    mean_gap      = results_df['tv_gap'].mean()

    agg_bal_acc = balanced_accuracy_score(all_test_labels, all_test_preds)

    # Save aggregate confusion matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title in [
        (axes[0], agg_cm,
         f"MLP [cv1]  Aggregate\nAcc={agg_dict['accuracy']:.3f}  BalAcc={agg_bal_acc:.3f}  F1={agg_dict['macro avg']['f1-score']:.3f}"),
    ]:
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, str(cm[ii, jj]), ha='center', va='center', fontsize=16)
        plt.colorbar(im, ax=ax)

    # Per-fold bar chart of macro F1
    axes[1].bar(range(1, N_FOLDS + 1), results_df['macro_f1'], color='steelblue', alpha=0.8)
    axes[1].axhline(mean_f1, color='r', linestyle='--', linewidth=2, label=f'Mean={mean_f1:.3f}')
    axes[1].set_xlabel('Fold'); axes[1].set_ylabel('Macro F1')
    axes[1].set_title('Macro F1 per Fold  [cv1]', fontsize=13)
    axes[1].set_xticks(range(1, N_FOLDS + 1))
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cv_summary_plot.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save aggregate results
    with open(OUTPUT_DIR / "cv_aggregate_results.txt", "w") as f:
        f.write(f"cv1: 5-Fold Participant-Grouped Cross-Validation (nested validation)\n")
        f.write(f"LR={MLP_LR:.1e}  WD={WEIGHT_DECAY:.1e}\n")
        f.write(f"Checkpoint criterion: val balanced accuracy\n")
        f.write(f"Threshold: tuned per-fold on val set from {THRESHOLD_CANDIDATES}\n\n")
        f.write(f"Per-fold results:\n")
        f.write(results_df[['fold','test_acc','test_bal_acc','macro_f1','cl0_recall','cl1_recall',
                             'threshold','tv_gap','best_epoch','stopped_epoch']].to_string(index=False))
        f.write(f"\n\nMean ± Std over {N_FOLDS} folds:\n")
        f.write(f"  Accuracy:        {mean_acc:.3f} ± {std_acc:.3f}\n")
        f.write(f"  Balanced Acc:    {mean_bal_acc:.3f} ± {std_bal_acc:.3f}\n")
        f.write(f"  Macro F1:        {mean_f1:.3f} ± {std_f1:.3f}\n")
        f.write(f"  Class-0 Recall:  {mean_cl0:.3f} ± {std_cl0:.3f}\n")
        f.write(f"  Class-1 Recall:  {mean_cl1:.3f} ± {std_cl1:.3f}\n")
        f.write(f"  Mean TV-gap:     {mean_gap:.4f}\n\n")
        f.write(f"Aggregate confusion matrix (all folds stitched):\n{agg_cm}\n\n")
        f.write(f"Aggregate classification report:\n{agg_rpt}\n")
        f.write(f"\nLR CV baseline:  acc={lr_mean_acc:.3f} ± {lr_std_acc:.3f}  "
                f"f1={lr_mean_f1:.3f} ± {lr_std_f1:.3f}\n")

    # ---- Final summary to stdout ----
    print("\n" + "=" * 60)
    print("CV SUMMARY  [cv1]")
    print("=" * 60)
    print(f"\nPer-fold results:")
    print(f"  {'Fold':<6}  {'Acc':>6}  {'BalAcc':>7}  {'MacroF1':>8}  {'Cl0Rec':>7}  {'Cl1Rec':>7}  {'Thr':>5}  {'TV-gap':>7}  {'BestEp':>7}")
    print(f"  {'-'*68}")
    for _, row in results_df.iterrows():
        print(f"  {int(row['fold']):<6}  {row['test_acc']:>6.3f}  {row['test_bal_acc']:>7.3f}  {row['macro_f1']:>8.3f}  "
              f"{row['cl0_recall']:>7.3f}  {row['cl1_recall']:>7.3f}  {row['threshold']:>5.2f}  "
              f"{row['tv_gap']:>7.4f}  {int(row['best_epoch']):>7}")
    print(f"  {'-'*68}")
    print(f"  {'Mean':<6}  {mean_acc:>6.3f}  {mean_bal_acc:>7.3f}  {mean_f1:>8.3f}  "
          f"{mean_cl0:>7.3f}  {mean_cl1:>7.3f}")
    print(f"  {'Std':<6}  {std_acc:>6.3f}  {std_bal_acc:>7.3f}  {std_f1:>8.3f}  "
          f"{std_cl0:>7.3f}  {std_cl1:>7.3f}")
    print(f"\nAggregate confusion matrix (all {n_total} samples, each predicted once):")
    print(agg_cm)
    print(f"\nAggregate classification report:\n{agg_rpt}")
    print(f"\nLR CV baseline:  acc={lr_mean_acc:.3f} ± {lr_std_acc:.3f}  "
          f"macro_f1={lr_mean_f1:.3f} ± {lr_std_f1:.3f}")
    print("=" * 60)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

