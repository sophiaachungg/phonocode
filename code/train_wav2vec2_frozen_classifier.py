import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import librosa
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as data
from transformers import AutoProcessor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- CONFIG ----------
AUDIO_ROOT = Path("data_processed/phoneme_reversal")
GROUND_TRUTH_CSV = Path("scoring/phoneme_reversal_ground_truth.csv")

# Wav2Vec2 backbone (frozen encoder)
MODEL_ID = "facebook/wav2vec2-base-960h"

TARGET_SR = 16000
RANDOM_SEED = 42

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15  # remaining goes to test
# -----------------------------


def parse_filename(path: Path) -> Tuple[str, str, str]:
    """
    Expect filenames like:
      ReXa_149_01_an.wav

    Format:
      [participant_id]_[audio_num]_[target_word].wav

    participant_id itself may contain underscores.
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {path}")
    target_word = parts[-1]
    audio_num = parts[-2]
    participant_id = "_".join(parts[:-2])
    return participant_id, audio_num, target_word


def normalize_word(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z']+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_label_map(gt_path: Path) -> Dict[Tuple[str, str], int]:
    """
    Returns dict mapping (participant_id, target_word) -> label (0/1).
    Assumes ground truth file is already binary.
    """
    gt = pd.read_csv(gt_path)

    id_col = "participant_id"
    non_word_cols = ["RA_preversal"]
    word_cols = [c for c in gt.columns if c not in [id_col] + non_word_cols]

    gt_long = gt.melt(
        id_vars=[id_col],
        value_vars=word_cols,
        var_name="target_word",
        value_name="ground_truth_label",
    )

    gt_long["target_word_norm"] = gt_long["target_word"].apply(normalize_word)
    gt_long["ground_truth_label"] = gt_long["ground_truth_label"].astype(int)

    label_map: Dict[Tuple[str, str], int] = {}
    for _, row in gt_long.iterrows():
        key = (str(row[id_col]), row["target_word_norm"])
        label_map[key] = int(row["ground_truth_label"])
    return label_map


def extract_embeddings(
    audio_root: Path,
    processor,
    model,
    device,
    label_map: Dict[Tuple[str, str], int],
):
    """
    Iterate over wav files, extract frozen embeddings + labels.
    Returns:
        X: np.ndarray [N, D]
        y: np.ndarray [N]
        participant_ids: List[str] length N
    """
    wav_files = sorted(audio_root.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No .wav files found under {audio_root}")
    print(f"Found {len(wav_files)} wav files")

    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    participants: List[str] = []

    skipped_no_label = 0

    for i, wav_path in enumerate(wav_files, 1):
        try:
            participant_id, audio_num, target_word = parse_filename(wav_path)
        except ValueError as e:
            print(f"[SKIP] {e}")
            continue

        key = (participant_id, normalize_word(target_word))
        if key not in label_map:
            skipped_no_label += 1
            continue
        label = label_map[key]

        # load audio
        audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

        # processor -> tensors
        inputs = processor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding="longest",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # [batch, time, hidden]; take mean over time
            hidden = outputs.last_hidden_state  # shape: [1, T, D]
            emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()

        embeddings.append(emb)
        labels.append(label)
        participants.append(participant_id)

        if i % 20 == 0:
            print(f"Processed {i}/{len(wav_files)} files...")

    if not embeddings:
        raise RuntimeError("No embeddings extracted. Check label_map / filenames.")

    if skipped_no_label > 0:
        print(f"Skipped {skipped_no_label} files with no matching label.")

    X = np.stack(embeddings, axis=0)
    y = np.array(labels, dtype=np.int64)
    return X, y, participants


def group_split(
    participants: List[str],
    train_frac: float,
    val_frac: float,
    seed: int = 42,
):
    """
    Participant-grouped split into train / val / test.
    Returns index arrays for each split.
    """
    rng = np.random.default_rng(seed)
    participants = np.array(participants)
    unique_pids = np.array(sorted(set(participants)))
    rng.shuffle(unique_pids)

    n = len(unique_pids)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_pids = set(unique_pids[:n_train])
    val_pids = set(unique_pids[n_train : n_train + n_val])
    test_pids = set(unique_pids[n_train + n_val :])

    def idx_for(set_pids):
        return np.array(
            [i for i, pid in enumerate(participants) if pid in set_pids],
            dtype=np.int64,
        )

    train_idx = idx_for(train_pids)
    val_idx = idx_for(val_pids)
    test_idx = idx_for(test_pids)

    print(f"Participants total: {n}")
    print(f"Train pids: {len(train_pids)}, Val pids: {len(val_pids)}, Test pids: {len(test_pids)}")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")

    return train_idx, val_idx, test_idx


def main():
    # ---- device ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ---- label map from ground truth ----
    print(f"Building label map from {GROUND_TRUTH_CSV} ...")
    label_map = build_label_map(GROUND_TRUTH_CSV)
    print(f"Label map entries: {len(label_map)}")

    # ---- load frozen Wav2Vec2 encoder ----
    print(f"Loading Wav2Vec2 encoder from: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    # ---- extract embeddings + labels ----
    X, y, participants = extract_embeddings(
        AUDIO_ROOT, processor, model, device, label_map
    )

    # ---- participant-grouped split ----
    train_idx, val_idx, test_idx = group_split(
        participants, TRAIN_FRAC, VAL_FRAC, seed=RANDOM_SEED
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # ---- sanity: binary labels ----
    unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
    assert set(unique_labels) <= {0, 1}, f"Found non-binary labels: {unique_labels}"

    # ---- logistic regression classifier ----
    print("\n=== Training Logistic Regression on frozen Wav2Vec2 embeddings ===")
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    def eval_split(name, X_split, y_split):
        y_pred = clf.predict(X_split)
        acc = accuracy_score(y_split, y_pred)
        print(f"\n[{name}] Accuracy: {acc:.3f}")
        cm = confusion_matrix(y_split, y_pred)
        print(f"[{name}] Confusion matrix:\n{cm}")
        print(
            f"[{name}] Classification report:\n"
            f"{classification_report(y_split, y_pred, digits=3)}"
        )

    eval_split("TRAIN", X_train, y_train)
    if len(X_val) > 0:
        eval_split("VAL", X_val, y_val)
    eval_split("TEST", X_test, y_test)

    # ---- PyTorch MLP classifier with learning curves ----
    print("\n=== Training PyTorch MLP on frozen Wav2Vec2 embeddings ===")

    num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    print(f"Detected {num_classes} classes for MLP.")

    # convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_ds = data.TensorDataset(X_train_t, y_train_t)
    val_ds = data.TensorDataset(X_val_t, y_val_t)
    test_ds = data.TensorDataset(X_test_t, y_test_t)

    train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = data.DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = data.DataLoader(test_ds, batch_size=64, shuffle=False)

    input_dim = X_train.shape[1]

    class MLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    mlp = MLP(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 50
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    def eval_epoch(model, loader):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        avg_loss = total_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    for epoch in range(1, num_epochs + 1):
        mlp.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = mlp(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        val_loss, val_acc = eval_epoch(mlp, val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{num_epochs} "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f}"
        )

    # Final test eval
    test_loss, test_acc = eval_epoch(mlp, test_loader)
    print(f"\n[MLP TEST] loss: {test_loss:.4f}, acc: {test_acc:.3f}")

    # ---- Plot learning curves ----
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["val_acc"], label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig("wav2vec2_mlp_learning_curves.png", dpi=150)
    print("Saved learning curves to wav2vec2_mlp_learning_curves.png")


if __name__ == "__main__":
    main()
