"""
inference_phoneme_reversal.py
------------------------------
Inference for the phoneme reversal task.

Pipeline: frozen Wav2Vec 2.0 encoder → hybrid pooling → difficulty scalar → MLP.

The stimulus word list is hardcoded here. The ground truth CSV is NOT required
at inference time — it contains participant scores (research data) and has no
role in scoring new audio.
"""

import re
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------

MODEL_ID  = "facebook/wav2vec2-base-960h"
TARGET_SR = 16000

# Ordered stimulus list — defines which words this scorer expects and the
# column order in output DataFrames. Must match the ground truth CSV header
# (excluding participant_id and RA columns).
TARGET_WORDS = [
    "an", "do", "pet", "sit", "dime", "boots", "see", "midnight", "pile",
    "seven", "speed", "system", "at", "baseball", "sun", "state", "to",
    "spoon", "cheek", "in", "be", "sometimes",
]

# Difficulty weights — must exactly match phoneme-reversal_train_final.py.
WORD_DIFFICULTY_WEIGHT = {
    "sit": 0.5, "be": 0.5, "pet": 0.5, "sun": 0.5,
    "to":  0.5, "do": 0.5, "speed": 0.5, "in": 0.5,
    "at":  1.0, "see": 1.0, "seven": 1.0, "spoon": 1.0,
    "dime": 1.0, "pile": 1.0, "cheek": 1.0,
    "state": 2.0, "boots": 2.0, "system": 2.0,
    "midnight": 2.0, "baseball": 2.0, "sometimes": 2.0,
}
_DIFF_MAP = {0.5: 0.0, 1.0: 0.5, 2.0: 1.0}

# Words excluded from training and scoring (practice/warm-up items).
SKIP_WORDS = {"an"}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class PhonemeReversalScorer:
    """
    Scores phoneme reversal audio files.

    Parameters
    ----------
    model_path : path to final_model.pt produced by phoneme_reversal_train_final.py
    device     : 'cpu', 'cuda', 'mps', or 'auto'
    threshold  : decision threshold for class 1 (correct). Loaded from checkpoint
                 if not provided.
    """

    def __init__(self, model_path: Path, device: str = "auto", threshold: float = None):
        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)

        # Load encoder
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.encoder   = AutoModel.from_pretrained(MODEL_ID).to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Load checkpoint
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        state_dict  = ckpt.get("model_state_dict", ckpt)
        input_dim   = ckpt.get("input_dim",   state_dict["net.0.weight"].shape[1])
        num_classes = ckpt.get("num_classes", state_dict["net.6.weight"].shape[0])

        self.threshold   = threshold if threshold is not None else ckpt.get("threshold", 0.5)
        self.target_words = [w for w in TARGET_WORDS if w not in SKIP_WORDS]

        self.classifier = MLP(input_dim, num_classes).to(self.device)
        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()
        self._input_dim = input_dim

        print(f"[phoneme_reversal] Loaded. input_dim={input_dim}  threshold={self.threshold}  device={device}")

    # ------------------------------------------------------------------

    def _embed(self, audio_path: Path, word: str) -> np.ndarray:
        audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        inputs   = self.processor(audio, sampling_rate=TARGET_SR,
                                  return_tensors="pt", padding="longest")
        inputs   = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden   = self.encoder(**inputs).last_hidden_state
            mean_emb = hidden.mean(dim=1).squeeze(0)
            max_emb  = hidden.max(dim=1).values.squeeze(0)
            pooled   = torch.cat([mean_emb, max_emb], dim=0).cpu().numpy()

        diff_w    = WORD_DIFFICULTY_WEIGHT.get(normalize_word(word), 1.0)
        diff_feat = np.array([_DIFF_MAP.get(diff_w, 0.5)], dtype=np.float32)
        emb       = np.concatenate([pooled, diff_feat])

        if emb.shape[0] != self._input_dim:
            raise ValueError(
                f"Embedding dim {emb.shape[0]} != checkpoint dim {self._input_dim}. "
                "MODEL_ID does not match the checkpoint."
            )
        return emb

    def predict(self, audio_path: Path, word: str) -> dict:
        """
        Score a single audio file.

        Returns
        -------
        dict with keys: score (0/1), confidence (float), needs_review (bool)
        """
        emb = self._embed(audio_path, word)
        with torch.no_grad():
            t      = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.classifier(t).squeeze(0)
            probs  = torch.softmax(logits, dim=-1)

        p1         = float(probs[1].item())
        score      = int(p1 >= self.threshold)
        confidence = p1 if score == 1 else float(probs[0].item())

        return {"score": score, "confidence": round(confidence, 4), "needs_review": False}


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def normalize_word(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z']+", " ", s)
    return re.sub(r"\s+", " ", s).strip()
