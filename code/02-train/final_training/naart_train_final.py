"""
naart_train_final.py
====================
Trains a single deployment-ready MLP on ALL participants.

This script is the natural follow-on to naart_train.py (cv1). The CV script
establishes that the approach generalises; this script produces the artifact
you actually ship.

Key differences from naart_train.py
-------------------------------------
1. No held-out test fold — all participants are used for training.
2. Early stopping uses a small participant-level holdout (~10%) carved
   from the full set.  This set is NOT evaluation data; it exists only
   to drive early stopping and threshold verification.
3. FIXED_EPOCHS is derived from mean(best_epoch across CV folds) + ~20%
   buffer. Early stopping is still active as a safety net.
   UPDATE THIS after reviewing your cv_fold_results.csv.
4. DEPLOY_THRESHOLD is set from your CV fold results (mean threshold).
   UPDATE THIS after reviewing your cv_fold_results.csv.
5. Saves a single final_model.pt bundling weights + metadata.

Usage
-----
    python naart_train_final.py

Outputs  (../results_naart_final/)
-------
    final_model.pt          deployment checkpoint
    training_history.csv    epoch-level metrics
    learning_curves.png     loss and accuracy curves
    final_model_card.txt    hyperparams, threshold, participant list
"""

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
from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score,
)
import torchaudio.functional as TAF

# ---------- CONFIG ----------
AUDIO_ROOT       = Path("../data/processed/naart")
GROUND_TRUTH_CSV = Path("../scoring/naart_ground_truth.csv")

MODEL_ID    = "facebook/wav2vec2-base-960h"
TARGET_SR   = 16000
RANDOM_SEED = 42

OUTPUT_DIR = Path("../results_naart_final")
OUTPUT_DIR.mkdir(exist_ok=True)

SKIP_WORDS = {"gouge", "placebo"}

FILESTEM_TO_COLUMN: Dict[str, str] = {
    "hors-doeuvre": "hors d'oeuvre",
    "dÃ©tente":     "detente",
    "faÃ§ade":      "facade",
    "détente":      "detente",
    "façade":       "facade",
    "ci-devant":    "ci-devant",
}

WORD_DIFFICULTY_WEIGHT: Dict[str, float] = {
    # hard (2.0)
    "ci-devant": 2.0, "drachm": 2.0, "talipes": 2.0, "gaoled": 2.0,
    "demesne": 2.0, "vivace": 2.0, "sidereal": 2.0, "beatify": 2.0,
    "synecdoche": 2.0, "capon": 2.0, "syncope": 2.0,
    # medium (1.0)
    "abstemious": 1.0, "ennui": 1.0, "detente": 1.0, "assignate": 1.0,
    "sieve": 1.0, "epergne": 1.0, "aeon": 1.0, "gauche": 1.0,
    "reify": 1.0, "radix": 1.0, "indices": 1.0, "leviathan": 1.0,
    "superfluous": 1.0, "gauge": 1.0, "cellist": 1.0, "banal": 1.0,
    # easy (0.5)
    "psalm": 0.5, "depot": 0.5, "equivocal": 0.5, "bouquet": 0.5,
    "indict": 0.5, "caveat": 0.5, "hors d'oeuvre": 0.5, "paradigm": 0.5,
    "corps": 0.5, "impugn": 0.5, "recipe": 0.5, "aisle": 0.5,
    "subtle": 0.5, "quadrupled": 0.5, "simile": 0.5, "heir": 0.5,
    "topiary": 0.5, "zealot": 0.5, "epitome": 0.5, "colonel": 0.5,
    "lingerie": 0.5, "debt": 0.5, "facade": 0.5, "hiatus": 0.5,
    "catacomb": 0.5, "rarefy": 0.5, "prelate": 0.5, "procreate": 0.5,
    "reign": 0.5, "gist": 0.5, "subpoena": 0.5, "debris": 0.5,
}
WORD_DIFFICULTY_WEIGHT["epergne"] = 1.0   # resolve duplicate key

EARLY_STOPPING_PATIENCE  = 15   # slightly more generous — full dataset trains slower
EARLY_STOPPING_MIN_DELTA = 1e-4

DURATION_THRESHOLD_MS = 3000
DURATION_WEIGHT_FLOOR = 0.25

AUG_NOISE_PROB    = 0.5
AUG_NOISE_SNR_DB  = (10, 30)
AUG_PITCH_PROB    = 0.5
AUG_PITCH_STEPS   = (-2, 2)
AUG_STRETCH_PROB  = 0.25
AUG_STRETCH_RANGE = (0.9, 1.1)

MLP_LR       = 1e-4
WEIGHT_DECAY = 2e-4
N_AUGMENT_COPIES = 2

# ---- UPDATE THESE from your cv_fold_results.csv ----
# FIXED_EPOCHS  : mean(best_epoch across folds) + ~20% buffer
# DEPLOY_THRESHOLD : mean(threshold across folds), or the mode if they cluster
FIXED_EPOCHS     = 60    # <-- REVIEW before running
DEPLOY_THRESHOLD = 0.40  # <-- REVIEW before running

# Fraction of participants held out ONLY for early stopping.
# These are NOT included in reported evaluation results.
ES_HOLDOUT_FRACTION = 0.10
# ---------- END CONFIG ----------


# ---------------------------------------------------------------------------
# Utilities (identical to naart_train.py — keep in sync if you update either)
# ---------------------------------------------------------------------------

def duration_weight(duration_ms: float) -> float:
    if duration_ms <= DURATION_THRESHOLD_MS:
        return 1.0
    excess = (duration_ms - DURATION_THRESHOLD_MS) / DURATION_THRESHOLD_MS
    return max(1.0 - (1.0 - DURATION_WEIGHT_FLOOR) * min(excess, 1.0), DURATION_WEIGHT_FLOOR)


def augment_audio(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    audio = audio.copy().astype(np.float32)
    if rng.random() < AUG_NOISE_PROB:
        snr_db    = rng.uniform(*AUG_NOISE_SNR_DB)
        sig_rms   = np.sqrt(np.mean(audio ** 2)) + 1e-9
        noise_rms = sig_rms / (10 ** (snr_db / 20))
        audio     = audio + rng.standard_normal(len(audio)).astype(np.float32) * noise_rms
    if rng.random() < AUG_PITCH_PROB:
        n_steps = rng.uniform(*AUG_PITCH_STEPS)
        wav_t   = torch.from_numpy(audio).unsqueeze(0)
        audio   = TAF.pitch_shift(wav_t, sr, n_steps=float(n_steps)).squeeze(0).numpy()
    if rng.random() < AUG_STRETCH_PROB:
        rate  = rng.uniform(*AUG_STRETCH_RANGE)
        wav_t = torch.from_numpy(audio).unsqueeze(0)
        audio = TAF.resample(wav_t, orig_freq=int(sr * rate), new_freq=sr).squeeze(0).numpy()
    return np.clip(audio, -1.0, 1.0)


def parse_filename(path: Path) -> Tuple[str, str, str]:
    stem  = path.stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {path.name!r}")
    return "_".join(parts[:-2]), parts[-2], parts[-1]


def filestem_to_column(raw_stem: str) -> str:
    return FILESTEM_TO_COLUMN.get(raw_stem.strip(), raw_stem.strip())


def normalize_word(s: str) -> str:
    try:
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    _MAP = str.maketrans({
        "\xe9": "e", "\xe8": "e", "\xea": "e", "\xeb": "e",
        "\xe0": "a", "\xe2": "a", "\xe4": "a", "\xe7": "c",
        "\xee": "i", "\xef": "i", "\xf4": "o", "\xf6": "o",
        "\xfb": "u", "\xfc": "u", "\xf9": "u",
        "\u2019": "'", "\u2018": "'",
    })
    s = s.translate(_MAP).lower().strip()
    s = re.sub(r"[^a-z'\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_label_map(gt_path: Path) -> Dict[Tuple[str, str], int]:
    gt = pd.read_csv(gt_path)
    id_col    = "participant_id"
    skip_cols = ["RA", "score_source"]
    word_cols = [c for c in gt.columns if c not in [id_col] + skip_cols]
    gt_long = gt.melt(id_vars=[id_col], value_vars=word_cols,
                      var_name="column_name", value_name="ground_truth_label")
    gt_long["word_norm"] = gt_long["column_name"].apply(normalize_word)
    print(f"Label map entries before NaN drop: {len(gt_long)}")
    gt_long = gt_long.dropna(subset=["ground_truth_label"])
    print(f"Label map entries after NaN drop:  {len(gt_long)}")
    sys.stdout.flush()
    gt_long["ground_truth_label"] = gt_long["ground_truth_label"].astype(int)
    return {
        (str(row[id_col]), row["word_norm"]): int(row["ground_truth_label"])
        for _, row in gt_long.iterrows()
    }


def expected_items_for_participant(pid: str, label_map: dict) -> dict:
    skip_norms = {normalize_word(w) for w in SKIP_WORDS}
    return {
        wn: lbl for (p, wn), lbl in label_map.items()
        if p == pid and wn not in skip_norms
    }


ZERO_AUDIO = np.zeros(TARGET_SR // 4, dtype=np.float32)


def embed_audio(audio, processor, model, device) -> np.ndarray:
    inputs = processor(audio, sampling_rate=TARGET_SR,
                       return_tensors="pt", padding="longest")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        hidden   = model(**inputs).last_hidden_state
        mean_emb = hidden.mean(dim=1).squeeze(0)
        max_emb  = hidden.max(dim=1).values.squeeze(0)
    return torch.cat([mean_emb, max_emb], dim=0).cpu().numpy()


def extract_embeddings(
    audio_root, processor, model, device, label_map,
    augment=False, n_augment_copies=2, rng=None, train_pids=None,
):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    skip_norms = {normalize_word(w) for w in SKIP_WORDS}
    diff_map   = {0.5: 0.0, 1.0: 0.5, 2.0: 1.0}
    embeddings, labels, participants, words, durations_ms = [], [], [], [], []
    injected_silence = bad_audio = skipped_word = skipped_label = 0

    pid_dirs = sorted([d for d in audio_root.iterdir() if d.is_dir()])
    if not pid_dirs:
        raise RuntimeError(f"No participant dirs under {audio_root}")
    print(f"Found {len(pid_dirs)} participant directories")
    sys.stdout.flush()

    for pid_dir in pid_dirs:
        pid = pid_dir.name
        if train_pids is not None and pid not in train_pids:
            continue
        expected = expected_items_for_participant(pid, label_map)
        if not expected:
            continue

        wav_by_word = {}
        for wav_path in sorted(pid_dir.glob("*.wav")):
            try:
                _, _, raw_stem = parse_filename(wav_path)
            except ValueError:
                continue
            wn = normalize_word(filestem_to_column(raw_stem))
            if wn in skip_norms:
                skipped_word += 1
            else:
                wav_by_word[wn] = wav_path

        for word_norm, label in expected.items():
            if word_norm in skip_norms:
                continue
            if (pid, word_norm) not in label_map:
                skipped_label += 1
                continue

            wav_path  = wav_by_word.get(word_norm)
            audio     = None
            duration  = 0.0
            is_silent = False

            if wav_path is not None:
                try:
                    audio_raw, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
                    if len(audio_raw) == 0 or not np.isfinite(audio_raw).all():
                        raise ValueError("empty or non-finite")
                    duration = len(audio_raw) / TARGET_SR * 1000.0
                    audio    = audio_raw
                except Exception as e:
                    bad_audio += 1
                    print(f"  [BAD AUDIO] {wav_path.name}: {e} — injecting silence")
                    is_silent = True
            else:
                injected_silence += 1
                is_silent = True

            if is_silent:
                audio    = ZERO_AUDIO
                duration = 0.0

            diff_w    = WORD_DIFFICULTY_WEIGHT.get(word_norm, 1.0)
            diff_feat = np.array([diff_map.get(diff_w, 0.5)], dtype=np.float32)

            audio_versions = [audio]
            if augment and not is_silent:
                for _ in range(n_augment_copies):
                    audio_versions.append(
                        augment_audio(audio, sr if wav_path else TARGET_SR, rng)
                    )

            for av in audio_versions:
                emb = np.concatenate([embed_audio(av, processor, model, device), diff_feat])
                embeddings.append(emb)
                labels.append(label)
                participants.append(pid)
                words.append(word_norm)
                durations_ms.append(duration)

    if not embeddings:
        raise RuntimeError("No embeddings extracted.")

    print(f"Skipped {skipped_word} SKIP_WORDS files, {injected_silence} silence injections, "
          f"{bad_audio} bad-audio replacements, {skipped_label} missing labels.")
    return (
        np.stack(embeddings), np.array(labels, dtype=np.int64),
        participants, words, durations_ms,
    )


class AugmentedEmbeddingDataset(data.Dataset):
    def __init__(self, X, y, sample_weights, augment=False, jitter_std=0.01, seed=42):
        self.X              = torch.tensor(X, dtype=torch.float32)
        self.y              = torch.tensor(y, dtype=torch.long)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        self.augment        = augment
        self.jitter_std     = jitter_std
        self.rng            = torch.Generator()
        self.rng.manual_seed(seed)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and self.jitter_std > 0:
            x = x + torch.randn(x.shape, generator=self.rng) * self.jitter_std
        return x, self.y[idx], self.sample_weights[idx]


def participant_es_holdout(participants, holdout_fraction=0.10, seed=42):
    """
    Returns (train_pids, es_pids).
    es_pids drive early stopping only — they are NOT evaluation data.
    """
    unique_pids = np.array(sorted(set(participants)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pids)
    n_es = max(1, min(int(round(len(unique_pids) * holdout_fraction)), len(unique_pids) - 1))
    return sorted(unique_pids[n_es:].tolist()), sorted(unique_pids[:n_es].tolist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("NAART: Final Deployment Model — All Participants")
    print(f"  LR={MLP_LR:.1e}  WD={WEIGHT_DECAY:.1e}")
    print(f"  FIXED_EPOCHS={FIXED_EPOCHS}  (early stopping patience={EARLY_STOPPING_PATIENCE})")
    print(f"  DEPLOY_THRESHOLD={DEPLOY_THRESHOLD}")
    print(f"  ES_HOLDOUT_FRACTION={ES_HOLDOUT_FRACTION}  (NOT evaluation data)")
    print(f"  SKIP_WORDS={SKIP_WORDS}")
    print("=" * 60)
    sys.stdout.flush()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # ---- Frozen encoder ----
    print(f"\nLoading Wav2Vec2: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    wav2vec   = AutoModel.from_pretrained(MODEL_ID)
    wav2vec.to(device).eval()
    for p in wav2vec.parameters():
        p.requires_grad = False
    print("Encoder loaded and frozen.")
    sys.stdout.flush()

    # ---- Label map ----
    print(f"\nBuilding label map from {GROUND_TRUTH_CSV} ...")
    label_map = build_label_map(GROUND_TRUTH_CSV)
    print(f"Label map entries: {len(label_map)}")
    sys.stdout.flush()

    # ---- Clean embeddings (all participants) — needed to get full pid list ----
    print("\nExtracting CLEAN embeddings (all participants) ...")
    X_all, y_all, participants_all, words_all, durations_all = extract_embeddings(
        AUDIO_ROOT, processor, wav2vec, device, label_map, augment=False, rng=rng,
    )
    n_total = len(X_all)
    lc = np.bincount(y_all)
    print(f"Total: {n_total} samples  dim={X_all.shape[1]}")
    print(f"Class dist: 0={lc[0]} ({lc[0]/n_total*100:.1f}%)  1={lc[1]} ({lc[1]/n_total*100:.1f}%)")
    sys.stdout.flush()

    # ---- Participant split ----
    all_pids = sorted(set(participants_all))
    train_pids, es_pids = participant_es_holdout(all_pids, ES_HOLDOUT_FRACTION, RANDOM_SEED)
    print(f"\nParticipant split:")
    print(f"  Training:   {len(train_pids)} participants")
    print(f"  ES holdout: {len(es_pids)} participants  (NOT evaluation data)")
    print(f"  ES PIDs: {es_pids}")
    sys.stdout.flush()

    parr    = np.array(participants_all)
    es_idx  = np.array([i for i, p in enumerate(parr) if p in set(es_pids)], dtype=np.int64)
    X_es    = X_all[es_idx]
    y_es    = y_all[es_idx]

    # ---- Augmented training embeddings (train_pids only) ----
    print(f"\nExtracting augmented training embeddings ({N_AUGMENT_COPIES} copies/file) ...")
    X_tr, y_tr, _, tr_words, tr_durs = extract_embeddings(
        AUDIO_ROOT, processor, wav2vec, device, label_map,
        augment=True, n_augment_copies=N_AUGMENT_COPIES,
        rng=rng, train_pids=set(train_pids),
    )
    print(f"Augmented train set: {len(X_tr)} embeddings")
    sys.stdout.flush()

    # ---- Per-sample weights (class × difficulty × duration) ----
    tr_counts     = np.bincount(y_tr, minlength=2).astype(np.float32)
    class_weights = tr_counts.sum() / (2.0 * tr_counts)
    diff_w_arr    = np.array([WORD_DIFFICULTY_WEIGHT.get(w, 1.0) for w in tr_words], dtype=np.float32)
    dur_w_arr     = np.array([duration_weight(d) for d in tr_durs], dtype=np.float32)
    sw            = np.array([class_weights[y] * diff_w_arr[i] * dur_w_arr[i]
                              for i, y in enumerate(y_tr)], dtype=np.float32)
    sw            = sw / sw.mean()

    train_ds = AugmentedEmbeddingDataset(X_tr, y_tr, sw, augment=True, jitter_std=0.01, seed=RANDOM_SEED)
    es_ds    = AugmentedEmbeddingDataset(X_es, y_es, np.ones(len(y_es), dtype=np.float32), augment=False)
    train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True)
    es_loader    = data.DataLoader(es_ds,    batch_size=64, shuffle=False)

    # ---- MLP ----
    input_dim = X_all.shape[1]  # 1537

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 2),
            )
        def forward(self, x): return self.net(x)

    criterion = nn.CrossEntropyLoss(reduction='none')

    def eval_epoch(model, loader, threshold=0.5):
        model.eval()
        total_loss = correct = total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb, _ in loader:
                xb, yb  = xb.to(device), yb.to(device)
                logits   = model(xb)
                losses   = criterion(logits, yb)
                total_loss += losses.mean().item() * xb.size(0)
                probs    = torch.softmax(logits, dim=-1)
                preds    = (probs[:, 1] >= threshold).long()
                correct += (preds == yb).sum().item()
                total   += xb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        avg_loss = total_loss / total if total else 0.0
        acc      = correct / total if total else 0.0
        bal_acc  = balanced_accuracy_score(np.array(all_labels), np.array(all_preds))
        return avg_loss, acc, bal_acc

    torch.manual_seed(RANDOM_SEED)
    mlp       = MLP().to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=MLP_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6,
    )

    best_bal_acc      = -1.0
    best_val_loss     = float("inf")
    best_epoch        = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_loss_uw": [], "val_loss": [],
               "train_acc": [], "val_acc": [], "val_bal_acc": []}

    print(f"\nTraining up to {FIXED_EPOCHS} epochs (ES patience={EARLY_STOPPING_PATIENCE}) ...")
    sys.stdout.flush()

    for epoch in range(1, FIXED_EPOCHS + 1):
        mlp.train()
        run_w = run_uw = correct = total = 0

        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            logits = mlp(xb)
            psl    = criterion(logits, yb)
            loss   = (psl * wb).mean()
            loss.backward()
            optimizer.step()
            run_w  += loss.item() * xb.size(0)
            run_uw += psl.mean().item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total   += xb.size(0)

        tl_w  = run_w  / total if total else 0.0
        tl_uw = run_uw / total if total else 0.0
        tacc  = correct / total if total else 0.0

        vl, vacc, vbac = eval_epoch(mlp, es_loader)
        scheduler.step(vl)

        history["train_loss"].append(tl_w)
        history["train_loss_uw"].append(tl_uw)
        history["val_loss"].append(vl)
        history["train_acc"].append(tacc)
        history["val_acc"].append(vacc)
        history["val_bal_acc"].append(vbac)

        improved = vbac > best_bal_acc + EARLY_STOPPING_MIN_DELTA
        if improved:
            best_bal_acc      = vbac
            best_val_loss     = vl
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': mlp.state_dict(),
                'val_loss':         vl,
                'val_bal_acc':      vbac,
                'input_dim':        input_dim,
                'num_classes':      2,
                'threshold':        DEPLOY_THRESHOLD,
                'train_pids':       sorted(train_pids),
                'es_pids':          sorted(es_pids),
                'skip_words':       sorted(SKIP_WORDS),
                'model_id':         MODEL_ID,
            }, OUTPUT_DIR / "final_model.pt")
        else:
            epochs_no_improve += 1

        lr_now = optimizer.param_groups[0]['lr']
        tag    = "  [✓]" if improved else f"  [no imp {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}]"
        print(
            f"  Ep {epoch:03d}/{FIXED_EPOCHS}  "
            f"train_loss_uw={tl_uw:.4f}  acc={tacc:.3f}  |  "
            f"es_loss={vl:.4f}  es_acc={vacc:.3f}  es_bal_acc={vbac:.3f}  "
            f"lr={lr_now:.2e}{tag}"
        )
        sys.stdout.flush()

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    actual_epochs = len(history["train_loss"])
    print(f"\nDone.  Best epoch: {best_epoch}  ES bal_acc={best_bal_acc:.3f}")
    sys.stdout.flush()

    # ---- Threshold verification on ES set (informational only) ----
    ckpt = torch.load(OUTPUT_DIR / "final_model.pt", weights_only=True)
    mlp.load_state_dict(ckpt['model_state_dict'])
    print("\nThreshold check on ES holdout (informational — not model selection):")
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50]:
        _, _, bac = eval_epoch(mlp, es_loader, threshold=thr)
        flag = "  <-- DEPLOY_THRESHOLD" if abs(thr - DEPLOY_THRESHOLD) < 1e-6 else ""
        print(f"  threshold={thr:.2f}  es_bal_acc={bac:.3f}{flag}")
    sys.stdout.flush()

    # ---- Save outputs ----
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    epochs_range = range(1, actual_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs_range, history["train_loss_uw"], label="Train loss (unweighted)", linewidth=2)
    axes[0].plot(epochs_range, history["val_loss"],      label="ES holdout loss",         linewidth=2)
    axes[0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.6, label=f"Best (ep {best_epoch})")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="NAART Final — Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history["train_acc"], label="Train acc",      linewidth=2)
    axes[1].plot(epochs_range, history["val_acc"],   label="ES holdout acc", linewidth=2)
    axes[1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.6, label=f"Best (ep {best_epoch})")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="NAART Final — Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    with open(OUTPUT_DIR / "final_model_card.txt", "w") as f:
        f.write("NAART Final Deployment Model\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Encoder:            {MODEL_ID} (frozen)\n")
        f.write(f"MLP:                Linear(1537→128) ReLU Dropout(0.2) "
                f"Linear(128→64) ReLU Dropout(0.3) Linear(64→2)\n")
        f.write(f"Input dim:          1537  (1536 hybrid-pooled + 1 difficulty)\n")
        f.write(f"LR:                 {MLP_LR:.1e}  WD: {WEIGHT_DECAY:.1e}\n")
        f.write(f"Augment copies:     {N_AUGMENT_COPIES}\n")
        f.write(f"Best epoch:         {best_epoch} / {actual_epochs}\n")
        f.write(f"Deploy threshold:   {DEPLOY_THRESHOLD}\n")
        f.write(f"SKIP_WORDS:         {sorted(SKIP_WORDS)}\n\n")
        f.write(f"Training participants ({len(train_pids)}):\n  {sorted(train_pids)}\n\n")
        f.write(f"ES holdout ONLY — NOT evaluation data ({len(es_pids)}):\n  {sorted(es_pids)}\n\n")
        f.write(f"ES holdout bal_acc at best epoch: {best_bal_acc:.4f}\n\n")
        f.write("NOTE: Generalisation estimates come from cv_aggregate_results.txt\n"
                "      produced by naart_train.py (cv1).  This script does not\n"
                "      produce evaluation metrics — it produces the deployment artifact.\n")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("  final_model.pt  |  training_history.csv  |  "
          "learning_curves.png  |  final_model_card.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
