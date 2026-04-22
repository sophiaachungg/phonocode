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
AUDIO_ROOT       = Path("../data_processed/naart")
GROUND_TRUTH_CSV = Path("../scoring/naart_ground_truth.csv")

MODEL_ID    = "facebook/wav2vec2-base-960h"
TARGET_SR   = 16000
RANDOM_SEED = 42

OUTPUT_DIR = Path("../results_naart")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- words to skip entirely (not included in training or evaluation) ----
# gouge and placebo are skipped: they are the practice/warm-up items and
# have artificially high correct rates that would distort the model.
SKIP_WORDS = {"gouge", "placebo"}

# ---- filename stem → ground truth column name ----
# Wav files use filesystem-safe stems; ground truth CSV uses canonical names.
# Any stem not listed here is assumed to match the column name directly.
FILESTEM_TO_COLUMN: Dict[str, str] = {
    "hors-doeuvre": "hors d'oeuvre",
    # accented variants that may appear due to latin-1/UTF-8 mojibake
    "dÃ©tente":     "detente",
    "faÃ§ade":      "facade",
    # unaccented forms already match the column names, listed for explicitness
    "detente":      "detente",
    "facade":       "facade",
    "ci-devant":    "ci-devant",
}

# ---- difficulty tiers derived from per-stimulus base rate correct ----
# Tiers are assigned from ground_truth_naart.csv base rates (N=110 participants):
#   hard   (2.0): base rate < 0.33  — 11 words, model errors here matter most
#   medium (1.0): base rate 0.33–0.70 — 16 words
#   easy   (0.5): base rate > 0.70  — 34 words (excluding gouge/placebo)
#
# The tier feeds two roles:
#   1. A normalised scalar feature appended to every embedding [0.0/0.5/1.0]
#      so the MLP can learn a difficulty-conditional decision boundary.
#   2. A loss multiplier so errors on hard items contribute more to training.
WORD_DIFFICULTY_WEIGHT: Dict[str, float] = {
    # --- hard (2.0) ---
    "ci-devant":   2.0,
    "drachm":      2.0,
    "talipes":     2.0,
    "gaoled":      2.0,
    "demesne":     2.0,
    "vivace":      2.0,
    "sidereal":    2.0,
    "beatify":     2.0,
    "synecdoche":  2.0,
    "capon":       2.0,
    "syncope":     2.0,
    # --- medium (1.0) ---
    "abstemious":  1.0,
    "ennui":       1.0,
    "detente":     1.0,
    "assignate":   1.0,
    "sieve":       1.0,
    "epergne":     1.0,
    "aeon":        1.0,
    "gauche":      1.0,
    "reify":       1.0,
    "radix":       1.0,
    "indices":     1.0,
    "leviathan":   1.0,
    "superfluous": 1.0,
    "gauge":       1.0,
    "cellist":     1.0,
    "banal":       1.0,
    # --- easy (0.5) ---
    "psalm":          0.5,
    "depot":          0.5,
    "equivocal":      0.5,
    "bouquet":        0.5,
    "indict":         0.5,
    "caveat":         0.5,
    "hors d'oeuvre":  0.5,
    "paradigm":       0.5,
    "corps":          0.5,
    "impugn":         0.5,
    "recipe":         0.5,
    "aisle":          0.5,
    "subtle":         0.5,
    "epergne":        0.5,   # also appears in medium — medium takes precedence via dict ordering
    "quadrupled":     0.5,
    "simile":         0.5,
    "heir":           0.5,
    "topiary":        0.5,
    "zealot":         0.5,
    "epitome":        0.5,
    "colonel":        0.5,
    "lingerie":       0.5,
    "debt":           0.5,
    "facade":         0.5,
    "hiatus":         0.5,
    "catacomb":       0.5,
    "rarefy":         0.5,
    "prelate":        0.5,
    "procreate":      0.5,
    "reign":          0.5,
    "gist":           0.5,
    "subpoena":       0.5,
    "debris":         0.5,
}
# Resolve the duplicate epergne key: epergne is medium (1.0).
# Python dicts keep last value for duplicate keys, so explicitly overwrite.
WORD_DIFFICULTY_WEIGHT["epergne"] = 1.0

# ---- early stopping ----
EARLY_STOPPING_PATIENCE  = 10
EARLY_STOPPING_MIN_DELTA = 1e-4

# ---- duration penalty ----
# NAART items are single read-aloud words; typical duration is well under 3s.
# Recordings over DURATION_THRESHOLD_MS are almost certainly long silences,
# double-recordings, or technical artefacts — downweight them in the loss.
# Files that are 0ms or infinite are treated as label=0 and never loaded
# (they are already scored 0 in ground_truth_naart.csv).
DURATION_THRESHOLD_MS  = 3000   # 3 s — anything longer is suspicious for NAART
DURATION_WEIGHT_FLOOR  = 0.25

# ---- augmentation ----
AUG_NOISE_PROB    = 0.5
AUG_NOISE_SNR_DB  = (10, 30)
AUG_PITCH_PROB    = 0.5
AUG_PITCH_STEPS   = (-2, 2)
AUG_STRETCH_PROB  = 0.25
AUG_STRETCH_RANGE = (0.9, 1.1)

# ---- locked hyperparameters ----
MLP_LR       = 1e-4
WEIGHT_DECAY = 2e-4

# ---- cross-validation ----
N_FOLDS          = 5
N_AUGMENT_COPIES = 2
# ---------- END CONFIG ----------


# ---------------------------------------------------------------------------
# Acoustic augmentation helpers
# ---------------------------------------------------------------------------

def duration_weight(duration_ms: float) -> float:
    """
    Return a loss weight in [DURATION_WEIGHT_FLOOR, 1.0].
    Recordings at or under threshold get 1.0; weight decays linearly to the
    floor over the next DURATION_THRESHOLD_MS of excess, then stays at floor.
    """
    if duration_ms <= DURATION_THRESHOLD_MS:
        return 1.0
    excess = (duration_ms - DURATION_THRESHOLD_MS) / DURATION_THRESHOLD_MS
    w = 1.0 - (1.0 - DURATION_WEIGHT_FLOOR) * min(excess, 1.0)
    return max(w, DURATION_WEIGHT_FLOOR)


def augment_audio(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """
    Apply stochastic acoustic augmentation to a raw waveform.
    All three transforms are applied independently.
    """
    audio = audio.copy().astype(np.float32)

    if rng.random() < AUG_NOISE_PROB:
        snr_db    = rng.uniform(*AUG_NOISE_SNR_DB)
        sig_rms   = np.sqrt(np.mean(audio ** 2)) + 1e-9
        noise_rms = sig_rms / (10 ** (snr_db / 20))
        noise     = rng.standard_normal(len(audio)).astype(np.float32) * noise_rms
        audio     = audio + noise

    if rng.random() < AUG_PITCH_PROB:
        n_steps = rng.uniform(*AUG_PITCH_STEPS)
        wav_t   = torch.from_numpy(audio).unsqueeze(0)
        wav_t   = TAF.pitch_shift(wav_t, sr, n_steps=float(n_steps))
        audio   = wav_t.squeeze(0).numpy()

    if rng.random() < AUG_STRETCH_PROB:
        rate    = rng.uniform(*AUG_STRETCH_RANGE)
        wav_t   = torch.from_numpy(audio).unsqueeze(0)
        new_sr  = int(sr * rate)
        wav_t   = TAF.resample(wav_t, orig_freq=new_sr, new_freq=sr)
        audio   = wav_t.squeeze(0).numpy()

    return np.clip(audio, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Filename / label utilities
# ---------------------------------------------------------------------------

def parse_filename(path: Path) -> Tuple[str, str, str]:
    """
    Expect filenames like: ReXa_034_13_hors-doeuvre.wav
    Returns (participant_id, item_no, filestem_word).
    participant_id may contain underscores; item_no is zero-padded two digits.
    The filestem_word is the raw last segment — call filestem_to_column() to
    resolve it to a ground-truth column name.
    """
    stem  = path.stem                  # e.g. "ReXa_034_13_hors-doeuvre"
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {path.name!r} "
                         f"(need at least 4 underscore-separated parts)")
    filestem_word  = parts[-1]         # "hors-doeuvre"
    item_no        = parts[-2]         # "13"
    participant_id = "_".join(parts[:-2])  # "ReXa_034"
    return participant_id, item_no, filestem_word


def filestem_to_column(raw_stem: str) -> str:
    """
    Map a wav filestem word to its ground-truth column name.
    Handles filesystem-safe substitutions and latin-1/UTF-8 mojibake.
    Falls back to the raw stem unchanged if no mapping is defined.
    """
    return FILESTEM_TO_COLUMN.get(raw_stem.strip(), raw_stem.strip())


def normalize_word(s: str) -> str:
    """
    Lowercase + transliterate accented/special characters + collapse whitespace.
    Keeps only ASCII letters, hyphens, and apostrophes after transliteration.

    Applied to both ground-truth column names and wav filestem words so that
    mojibake, accents, and punctuation variations never cause missed lookups.

    Examples
    --------
    "hors d'oeuvre"  ->  "hors d'oeuvre"
    "detente"        ->  "detente"
    "facade"         ->  "facade"
    "ci-devant"      ->  "ci-devant"
    "dÃ©tente"       ->  "detente"   (mojibake round-trip via latin-1)
    """
    # Fix mojibake: bytes encoded as latin-1 then decoded as UTF-8
    try:
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass  # already valid UTF-8

    # Transliterate accented characters used in NAART stimuli
    _ACCENT_MAP = str.maketrans({
        "\xe9": "e", "\xe8": "e", "\xea": "e", "\xeb": "e",  # é è ê ë
        "\xe0": "a", "\xe2": "a", "\xe4": "a",                # à â ä
        "\xe7": "c",                                           # ç
        "\xee": "i", "\xef": "i",                             # î ï
        "\xf4": "o", "\xf6": "o",                             # ô ö
        "\xfb": "u", "\xfc": "u", "\xf9": "u",               # û ü ù
        "\u2019": "'",   # right single quotation mark
        "\u2018": "'",   # left single quotation mark
    })
    s = s.translate(_ACCENT_MAP)
    s = s.lower().strip()
    s = re.sub(r"[^a-z'\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_label_map(gt_path: Path) -> Dict[Tuple[str, str], int]:
    """
    Returns dict mapping (participant_id, normalized_column_name) → label (0/1).

    Non-word columns ('RA', 'score_source') are excluded automatically.
    NaN cells are dropped — they represent genuinely missing scores, not zeros.
    Items that were skipped by the participant (fully missed audio) are already
    scored 0 in the ground truth and will be present as label=0 here.
    """
    gt = pd.read_csv(gt_path)

    id_col        = "participant_id"
    non_word_cols = ["RA", "score_source"]
    word_cols     = [c for c in gt.columns if c not in [id_col] + non_word_cols]

    gt_long = gt.melt(
        id_vars=[id_col],
        value_vars=word_cols,
        var_name="column_name",
        value_name="ground_truth_label",
    )
    gt_long["word_norm"] = gt_long["column_name"].apply(normalize_word)

    print(f"Total entries before dropping NaN: {len(gt_long)}")
    gt_long = gt_long.dropna(subset=["ground_truth_label"])
    print(f"Total entries after dropping NaN:  {len(gt_long)}")
    sys.stdout.flush()

    gt_long["ground_truth_label"] = gt_long["ground_truth_label"].astype(int)

    label_map: Dict[Tuple[str, str], int] = {}
    for _, row in gt_long.iterrows():
        key = (str(row[id_col]), row["word_norm"])
        label_map[key] = int(row["ground_truth_label"])
    return label_map


# ---------------------------------------------------------------------------
# Missing-item handling
# ---------------------------------------------------------------------------

def expected_items_for_participant(
    pid: str,
    label_map: Dict[Tuple[str, str], int],
) -> Dict[str, int]:
    """
    Return {normalized_word: label} for every scored item for this participant.
    Used to detect missing wav files and inject synthetic zero-embeddings.
    """
    return {
        word_norm: label
        for (p, word_norm), label in label_map.items()
        if p == pid and word_norm not in {normalize_word(w) for w in SKIP_WORDS}
    }


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

ZERO_AUDIO = np.zeros(TARGET_SR // 4, dtype=np.float32)  # 250 ms of silence


def embed_audio(audio: np.ndarray, processor, model, device) -> np.ndarray:
    """Extract a single hybrid-pooled embedding [1536] from a waveform."""
    inputs = processor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding="longest",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        hidden   = model(**inputs).last_hidden_state   # [1, T, D]
        mean_emb = hidden.mean(dim=1).squeeze(0)       # [D]
        max_emb  = hidden.max(dim=1).values.squeeze(0) # [D]
        pooled   = torch.cat([mean_emb, max_emb], dim=0)  # [2D = 1536]
    return pooled.cpu().numpy()


def extract_embeddings(
    audio_root: Path,
    processor,
    model,
    device,
    label_map: Dict[Tuple[str, str], int],
    augment: bool = False,
    n_augment_copies: int = 2,
    rng: Optional[np.random.Generator] = None,
    train_pids: Optional[set] = None,
):
    """
    Iterate over participant wav directories, extract frozen Wav2Vec2 embeddings.

    NAART-specific behaviour
    ------------------------
    SKIP_WORDS : gouge and placebo are excluded unconditionally.

    MISSING FILES : if a participant has a scored item in the ground truth but
    no corresponding wav file (participant double-clicked and skipped the item),
    a synthetic silence embedding is injected.  The ground truth already records
    these as label=0, so the model learns that silence → incorrect.

    EMPTY / INFINITE FILES : files that librosa cannot read, or that contain
    zero frames, are treated the same as missing files (silence embedding,
    label from ground truth which is 0).

    DURATION PENALTY : files over DURATION_THRESHOLD_MS are downweighted in the
    loss.  Silent/missing items get duration 0 ms (weight = 1.0).

    HYBRID POOLING : mean and max of last_hidden_state are concatenated → 1536.

    DIFFICULTY FEATURE : a scalar in {0.0, 0.5, 1.0} (easy/medium/hard) is
    appended → final dim 1537.

    AUGMENTATION : when augment=True, n_augment_copies additional perturbed
    versions are created for each training audio.  Silent embeddings are never
    augmented (there is nothing to perturb).

    Parameters
    ----------
    train_pids : if not None, only participants in this set are processed.

    Returns
    -------
    X            : np.ndarray  [N, 1537]
    y            : np.ndarray  [N]
    participants : List[str]
    words        : List[str]   normalised word (for difficulty weight lookup)
    durations_ms : List[float] source audio duration (0.0 for silent/missing)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    skip_norms = {normalize_word(w) for w in SKIP_WORDS}
    diff_map   = {0.5: 0.0, 1.0: 0.5, 2.0: 1.0}

    embeddings:   List[np.ndarray] = []
    labels:       List[int]        = []
    participants: List[str]        = []
    words:        List[str]        = []
    durations_ms: List[float]      = []

    skipped_word  = 0
    skipped_label = 0
    injected_silence = 0
    bad_audio     = 0

    # Collect all participant directories
    pid_dirs = sorted([d for d in audio_root.iterdir() if d.is_dir()])
    if not pid_dirs:
        raise RuntimeError(f"No participant directories found under {audio_root}")

    print(f"Found {len(pid_dirs)} participant directories")
    sys.stdout.flush()

    for pid_dir in pid_dirs:
        pid = pid_dir.name

        if train_pids is not None and pid not in train_pids:
            continue

        # Build the set of items this participant should have
        expected = expected_items_for_participant(pid, label_map)
        if not expected:
            continue  # participant not in ground truth at all

        # Index available wav files by normalized word
        wav_by_word: Dict[str, Path] = {}
        for wav_path in sorted(pid_dir.glob("*.wav")):
            try:
                _, _, raw_stem = parse_filename(wav_path)
            except ValueError:
                continue
            col_name  = filestem_to_column(raw_stem)
            word_norm = normalize_word(col_name)
            if word_norm in skip_norms:
                skipped_word += 1
                continue
            wav_by_word[word_norm] = wav_path

        # Process every scored item; inject silence for missing files
        for word_norm, label in expected.items():
            if word_norm in skip_norms:
                continue

            key = (pid, word_norm)
            if key not in label_map:
                skipped_label += 1
                continue

            wav_path = wav_by_word.get(word_norm)

            # Try to load audio; fall back to silence on any failure
            audio       = None
            duration    = 0.0
            is_silent   = False

            if wav_path is not None:
                try:
                    audio_raw, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
                    if len(audio_raw) == 0 or not np.isfinite(audio_raw).all():
                        raise ValueError("empty or non-finite audio")
                    duration = len(audio_raw) / TARGET_SR * 1000.0
                    audio    = audio_raw
                except Exception as e:
                    bad_audio += 1
                    print(f"  [BAD AUDIO] {wav_path.name}: {e} — injecting silence")
                    is_silent = True
            else:
                # File missing: participant skipped this item
                injected_silence += 1
                is_silent = True

            if is_silent:
                audio    = ZERO_AUDIO
                duration = 0.0

            # Difficulty feature
            diff_w    = WORD_DIFFICULTY_WEIGHT.get(word_norm, 1.0)
            diff_feat = np.array([diff_map.get(diff_w, 0.5)], dtype=np.float32)

            # Build list of audio versions (original + augmented copies)
            audio_versions = [audio]
            if augment and not is_silent:
                for _ in range(n_augment_copies):
                    audio_versions.append(augment_audio(audio, sr if wav_path else TARGET_SR, rng))

            for audio_v in audio_versions:
                emb = embed_audio(audio_v, processor, model, device)
                emb = np.concatenate([emb, diff_feat])   # [1537]

                embeddings.append(emb)
                labels.append(label)
                participants.append(pid)
                words.append(word_norm)
                durations_ms.append(duration)

    if not embeddings:
        raise RuntimeError("No embeddings extracted. Check label_map / audio paths.")

    print(f"Skipped {skipped_word} files in SKIP_WORDS {SKIP_WORDS}.")
    print(f"Injected {injected_silence} silence embeddings for missing files.")
    if bad_audio:
        print(f"Replaced {bad_audio} bad/empty audio file(s) with silence.")
    if skipped_label > 0:
        print(f"Skipped {skipped_label} items with no ground truth label.")

    X = np.stack(embeddings, axis=0)
    y = np.array(labels, dtype=np.int64)
    return X, y, participants, words, durations_ms


# ---------------------------------------------------------------------------
# Participant-grouped K-Fold splitter
# ---------------------------------------------------------------------------

def participant_kfold(
    participants: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Return (train_idx, test_idx) tuples such that all samples from a given
    participant land entirely in train or entirely in test within each fold.
    """
    rng          = np.random.default_rng(seed)
    participants = np.array(participants)
    unique_pids  = np.array(sorted(set(participants)))
    rng.shuffle(unique_pids)

    kf    = KFold(n_splits=n_splits, shuffle=False)
    folds = []

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
    """Split a participant list into disjoint train/val partitions."""
    participants = np.array(participants)
    unique_pids  = np.array(sorted(set(participants)))
    if len(unique_pids) < 2:
        raise ValueError("Need at least 2 participants to make a train/val split")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pids)

    n_val      = max(1, min(int(round(len(unique_pids) * val_fraction)), len(unique_pids) - 1))
    val_pids   = set(unique_pids[:n_val])
    train_pids = set(unique_pids[n_val:])

    train_idx = np.array([i for i, p in enumerate(participants) if p in train_pids], dtype=np.int64)
    val_idx   = np.array([i for i, p in enumerate(participants) if p in val_pids],   dtype=np.int64)

    return train_idx, val_idx, sorted(train_pids), sorted(val_pids)


# ---------------------------------------------------------------------------
# Augmented embedding dataset
# ---------------------------------------------------------------------------

class AugmentedEmbeddingDataset(data.Dataset):
    """
    Wraps (X, y, sample_weights) tensors.
    During training, small Gaussian jitter is applied in embedding space as a
    second-pass regulariser (distinct from audio-level augmentation).
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
    print("NAART: Wav2Vec2 Frozen  [cv1]  5-Fold Cross-Validation")
    print(f"  + {N_FOLDS}-fold participant-grouped CV")
    print(f"  + LR={MLP_LR:.1e}  WD={WEIGHT_DECAY:.1e}")
    print(f"  + AUG_STRETCH_PROB={AUG_STRETCH_PROB}")
    print(f"  + Duration penalty: threshold={DURATION_THRESHOLD_MS}ms  floor={DURATION_WEIGHT_FLOOR}")
    print(f"  + SKIP_WORDS={SKIP_WORDS}")
    print("  + Missing/empty audio → silence embedding (label from ground truth)")
    print("  + Augmentation applied to inner-train split only")
    print("  + Checkpoint criterion: val balanced accuracy")
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
    print(f"\nExtracting CLEAN embeddings (all participants, no augmentation)...")
    sys.stdout.flush()

    X_all, y_all, participants_all, words_all, durations_all = extract_embeddings(
        AUDIO_ROOT, processor, wav2vec, device, label_map,
        augment=False, rng=rng,
    )
    n_total = len(X_all)
    print(f"Clean extraction complete: {n_total} samples, dim={X_all.shape[1]}")
    label_counts = np.bincount(y_all)
    print(f"Label distribution: class 0={label_counts[0]} ({label_counts[0]/n_total*100:.1f}%)  "
          f"class 1={label_counts[1]} ({label_counts[1]/n_total*100:.1f}%)")
    sys.stdout.flush()

    # ---- LR baseline ----
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

    THRESHOLD_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50]

    fold_results     = []
    all_test_preds   = np.zeros(n_total, dtype=np.int64)
    all_test_labels  = np.zeros(n_total, dtype=np.int64)
    all_test_covered = np.zeros(n_total, dtype=bool)

    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        fold_dir = OUTPUT_DIR / f"fold{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        outer_train_pids = set(np.array(participants_all)[train_idx])
        test_pids_fold   = set(np.array(participants_all)[test_idx])

        inner_train_rel_idx, inner_val_rel_idx, inner_train_pids_list, inner_val_pids_list = \
            participant_holdout_split(
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

        with open(fold_dir / "fold_split_info.json", "w") as f:
            json.dump({
                'fold': fold_idx,
                'outer_train_participants': sorted(outer_train_pids),
                'inner_train_participants': sorted(train_pids_fold),
                'inner_val_participants':   sorted(val_pids_fold),
                'test_participants':        sorted(test_pids_fold),
                'outer_train_samples':      n_outer_train,
                'inner_train_samples_clean': n_train_clean,
                'val_samples':              n_val,
                'test_samples':             n_test,
            }, f, indent=2)

        # ---- Augmented train embeddings for this fold ----
        fold_rng = np.random.default_rng(RANDOM_SEED + fold_idx)

        print(f"  Extracting augmented inner-train embeddings ({N_AUGMENT_COPIES} copies)...")
        sys.stdout.flush()

        X_aug, y_aug, _, aug_words, aug_durations = extract_embeddings(
            AUDIO_ROOT, processor, wav2vec, device, label_map,
            augment=True, n_augment_copies=N_AUGMENT_COPIES,
            rng=fold_rng, train_pids=train_pids_fold,
        )

        if len(X_aug) == 0:
            raise RuntimeError(f"No augmented training embeddings extracted for fold {fold_idx}.")

        X_train = X_aug
        y_train = y_aug
        X_val   = X_all[inner_val_idx]
        y_val   = y_all[inner_val_idx]
        X_test  = X_all[test_idx]
        y_test  = y_all[test_idx]

        print(f"  Augmented inner-train: {len(X_train)} embeddings  |  "
              f"Clean val: {len(X_val)} samples  |  Clean test: {len(X_test)} samples")
        sys.stdout.flush()

        # ---- Per-sample weights: class × difficulty × duration ----
        train_counts  = np.bincount(y_train, minlength=2).astype(np.float32)
        class_weights = train_counts.sum() / (2.0 * train_counts)
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

        val_sw  = np.ones(len(y_val),  dtype=np.float32)
        test_sw = np.ones(len(y_test), dtype=np.float32)

        train_ds = AugmentedEmbeddingDataset(
            X_train, y_train, sample_weights, augment=True, jitter_std=0.01,
            seed=RANDOM_SEED + fold_idx,
        )
        val_ds  = AugmentedEmbeddingDataset(X_val,  y_val,  val_sw,  augment=False, jitter_std=0.0)
        test_ds = AugmentedEmbeddingDataset(X_test, y_test, test_sw, augment=False, jitter_std=0.0)

        train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = data.DataLoader(val_ds,   batch_size=64, shuffle=False)
        test_loader  = data.DataLoader(test_ds,  batch_size=64, shuffle=False)

        # ---- Train MLP ----
        torch.manual_seed(RANDOM_SEED + fold_idx)
        mlp       = MLP().to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=MLP_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6,
        )

        num_epochs        = 100
        best_val_bal_acc  = -1.0
        best_val_loss     = float("inf")
        best_val_acc      = 0.0
        best_epoch        = 0
        epochs_no_improve = 0
        history = {
            "train_loss": [], "train_loss_unweighted": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "val_bal_acc": [],
        }

        print(f"\n  Training MLP (fold {fold_idx})...")
        sys.stdout.flush()

        for epoch in range(1, num_epochs + 1):
            mlp.train()
            running_loss_weighted   = 0.0
            running_loss_unweighted = 0.0
            correct, total          = 0, 0

            for xb, yb, wb in train_loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                optimizer.zero_grad()
                logits          = mlp(xb)
                per_sample_loss = criterion(logits, yb)
                loss            = (per_sample_loss * wb).mean()
                loss.backward()
                optimizer.step()

                running_loss_weighted   += loss.item() * xb.size(0)
                running_loss_unweighted += per_sample_loss.mean().item() * xb.size(0)
                preds    = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += xb.size(0)

            train_loss_weighted   = running_loss_weighted   / total if total > 0 else 0.0
            train_loss_unweighted = running_loss_unweighted / total if total > 0 else 0.0
            train_acc             = correct / total if total > 0 else 0.0

            val_loss, val_acc, _, _, _, val_bal_acc = eval_epoch(mlp, val_loader)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss_weighted)
            history["train_loss_unweighted"].append(train_loss_unweighted)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_bal_acc"].append(val_bal_acc)

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
                f"train loss weighted={train_loss_weighted:.4f} "
                f"train loss unweighted={train_loss_unweighted:.4f} acc={train_acc:.3f}  |  "
                f"val loss={val_loss:.4f} acc={val_acc:.3f} bal_acc={val_bal_acc:.3f}  "
                f"lr={current_lr:.2e}{es_tag}"
            )
            sys.stdout.flush()

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}.")
                sys.stdout.flush()
                break

        actual_epochs      = len(history["train_loss"])
        train_loss_at_best = history["train_loss_unweighted"][best_epoch - 1]
        tv_gap             = best_val_loss - train_loss_at_best

        ckpt = torch.load(fold_dir / "mlp_best_model.pt", weights_only=True)
        mlp.load_state_dict(ckpt['model_state_dict'])

        # ---- Threshold tuning on val set ----
        best_threshold  = 0.5
        best_thresh_bac = -1.0
        print(f"\n  Threshold sweep (val set):")
        for thr in THRESHOLD_CANDIDATES:
            _, _, _, thr_labels, thr_probs, thr_bal_acc = eval_epoch(mlp, val_loader, threshold=thr)
            print(f"    threshold={thr:.2f}  val_bal_acc={thr_bal_acc:.3f}")
            if thr_bal_acc > best_thresh_bac:
                best_thresh_bac = thr_bal_acc
                best_threshold  = thr
        print(f"  → Selected threshold: {best_threshold:.2f}  (val_bal_acc={best_thresh_bac:.3f})")

        _, _, fold_preds, fold_labels, fold_probs, fold_bal_acc = eval_epoch(
            mlp, test_loader, threshold=best_threshold,
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
            'true_label': fold_labels,
            'predicted_label': fold_preds,
            'prob_class_1': fold_probs,
        }).to_csv(fold_dir / "test_predictions.csv", index=False)

        # Learning curves
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
            'fold':             fold_idx,
            'outer_train_pids': sorted(outer_train_pids),
            'train_pids':       sorted(train_pids_fold),
            'val_pids':         sorted(val_pids_fold),
            'test_pids':        sorted(test_pids_fold),
            'n_outer_train':    len(train_idx),
            'n_train':          len(X_train),
            'n_val':            len(X_val),
            'n_test':           n_test,
            'best_epoch':       best_epoch,
            'stopped_epoch':    actual_epochs,
            'tv_gap':           tv_gap,
            'val_bal_acc':      best_val_bal_acc,
            'val_loss':         best_val_loss,
            'threshold':        best_threshold,
            'test_acc':         fold_acc,
            'test_bal_acc':     fold_bal_acc,
            'macro_f1':         fold_macro_f1,
            'cl0_recall':       fold_cl0_rec,
            'cl1_recall':       fold_cl1_rec,
        })

    # ---- Aggregate results ----
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(OUTPUT_DIR / "cv_fold_results.csv", index=False)

    agg_cm   = confusion_matrix(all_test_labels, all_test_preds)
    agg_rpt  = classification_report(all_test_labels, all_test_preds, digits=3)
    agg_dict = classification_report(all_test_labels, all_test_preds, digits=3, output_dict=True)

    mean_acc     = results_df['test_acc'].mean()
    std_acc      = results_df['test_acc'].std()
    mean_bal_acc = results_df['test_bal_acc'].mean()
    std_bal_acc  = results_df['test_bal_acc'].std()
    mean_f1      = results_df['macro_f1'].mean()
    std_f1       = results_df['macro_f1'].std()
    mean_cl0     = results_df['cl0_recall'].mean()
    std_cl0      = results_df['cl0_recall'].std()
    mean_cl1     = results_df['cl1_recall'].mean()
    std_cl1      = results_df['cl1_recall'].std()
    mean_gap     = results_df['tv_gap'].mean()

    agg_bal_acc = balanced_accuracy_score(all_test_labels, all_test_preds)

    # Aggregate plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_title = (f"MLP [cv1]  Aggregate\n"
                f"Acc={agg_dict['accuracy']:.3f}  "
                f"BalAcc={agg_bal_acc:.3f}  "
                f"F1={agg_dict['macro avg']['f1-score']:.3f}")
    im = axes[0].imshow(agg_cm, cmap='Blues', aspect='auto')
    axes[0].set_xlabel('Predicted', fontsize=12); axes[0].set_ylabel('True', fontsize=12)
    axes[0].set_title(cm_title, fontsize=13)
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    for ii in range(2):
        for jj in range(2):
            axes[0].text(jj, ii, str(agg_cm[ii, jj]), ha='center', va='center', fontsize=16)
    plt.colorbar(im, ax=axes[0])

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

    with open(OUTPUT_DIR / "cv_aggregate_results.txt", "w") as f:
        f.write(f"NAART  cv1: 5-Fold Participant-Grouped Cross-Validation\n")
        f.write(f"LR={MLP_LR:.1e}  WD={WEIGHT_DECAY:.1e}\n")
        f.write(f"Checkpoint criterion: val balanced accuracy\n")
        f.write(f"Threshold: tuned per-fold on val set from {THRESHOLD_CANDIDATES}\n")
        f.write(f"SKIP_WORDS: {SKIP_WORDS}\n")
        f.write(f"Duration threshold: {DURATION_THRESHOLD_MS} ms\n\n")
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

    print("\n" + "=" * 60)
    print("CV SUMMARY  [cv1]  —  NAART")
    print("=" * 60)
    print(f"\nPer-fold results:")
    print(f"  {'Fold':<6}  {'Acc':>6}  {'BalAcc':>7}  {'MacroF1':>8}  "
          f"{'Cl0Rec':>7}  {'Cl1Rec':>7}  {'Thr':>5}  {'TV-gap':>7}  {'BestEp':>7}")
    print(f"  {'-'*68}")
    for _, row in results_df.iterrows():
        print(f"  {int(row['fold']):<6}  {row['test_acc']:>6.3f}  {row['test_bal_acc']:>7.3f}  "
              f"{row['macro_f1']:>8.3f}  {row['cl0_recall']:>7.3f}  {row['cl1_recall']:>7.3f}  "
              f"{row['threshold']:>5.2f}  {row['tv_gap']:>7.4f}  {int(row['best_epoch']):>7}")
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
