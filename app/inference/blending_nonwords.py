"""
inference_blending_nonwords.py
--------------------------------
Inference for the blending nonwords task.

Pipeline: frozen WavLM-Large encoder → hybrid pooling → [A, B, A-B, A*B]
similarity feature vector → difficulty scalar → Logistic Regression.

The stimulus word list and reference recording paths are hardcoded here.
The ground truth CSV is NOT required at inference time.

Reference recordings
--------------------
One canonical .wav file per stimulus must be present in:
    reference_recordings/blending_nonwords/{stimulus}.wav

These are the same files used during training. They are pre-encoded once at
scorer initialisation and reused for all participant trials.
"""

import pickle
import re
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

# ---------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------

MODEL_ID  = "microsoft/wavlm-large"
TARGET_SR = 16000

# Ordered stimulus list.
TARGET_WORDS = [
    "lander", "jad", "mog", "het", "ko", "nimby", "teb", "shawbo",
    "ghite", "zigopple", "shib", "motabe", "heckobi", "tastains",
    "nysheeboki", "jop", "nass", "vope", "suhnypogh", "nemowk",
    "shyvitch", "basp", "tigu", "koomayg",
]

WORD_DIFFICULTY_WEIGHT: Dict[str, float] = {
    "lander": 0.5, "nimby": 0.5, "mog": 0.5, "ko": 0.5,
    "teb": 0.5, "shawbo": 0.5, "tigu": 0.5, "motabe": 0.5,
    "vope": 1.0, "shib": 1.0, "het": 1.0, "basp": 1.0,
    "nass": 1.0, "jad": 1.0, "heckobi": 1.0, "ghite": 1.0, "jop": 1.0,
    "zigopple": 2.0, "nemowk": 2.0, "koomayg": 2.0, "shyvitch": 2.0,
    "tastains": 2.0, "suhnypogh": 2.0, "nysheeboki": 2.0,
}
_DIFF_MAP = {0.5: 0.0, 1.0: 0.5, 2.0: 1.0}

# Responses longer than this are treated as recording artefacts.
DURATION_THRESHOLD_MS = 8000


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class BlendingNonwordsScorer:
    """
    Scores blending nonwords audio files.

    Parameters
    ----------
    model_path    : path to lr_production_model.pkl produced by
                    blending_nonwords_train_final.py
    ref_root      : directory containing one .wav per stimulus
                    (e.g. reference_recordings/blending_nonwords/)
    device        : 'cpu', 'cuda', 'mps', or 'auto'
    """

    def __init__(self, model_path: Path, ref_root: Path, device: str = "auto"):
        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        self.target_words = list(TARGET_WORDS)

        # Load WavLM encoder (fully frozen at inference)
        self.processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        self.encoder   = AutoModel.from_pretrained(MODEL_ID).to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Load LR classifier
        with open(model_path, "rb") as f:
            self.classifier = pickle.load(f)

        # Pre-encode reference recordings once
        print("[blending_nonwords] Pre-encoding reference recordings...")
        self._ref_embeddings: Dict[str, np.ndarray] = {}
        for word in TARGET_WORDS:
            ref_path = Path(ref_root) / f"{word}.wav"
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"Reference recording not found: {ref_path}\n"
                    f"Ensure all 24 reference .wav files are in {ref_root}/"
                )

            audio, _ = librosa.load(ref_path, sr=TARGET_SR, mono=True)
            self._ref_embeddings[word] = self._encode(audio)

        print(f"[blending_nonwords] Loaded. {len(self._ref_embeddings)} references pre-encoded.  device={device}")

    # ------------------------------------------------------------------

    def _encode(self, audio: np.ndarray) -> np.ndarray:
        """Mean+max pooled WavLM embedding, shape [2*D]."""
        inputs = self.processor(audio, sampling_rate=TARGET_SR,
                                return_tensors="pt", padding="longest")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden   = self.encoder(**inputs).last_hidden_state
            mean_emb = hidden.mean(dim=1).squeeze(0)
            max_emb  = hidden.max(dim=1).values.squeeze(0)
        return torch.cat([mean_emb, max_emb], dim=0).cpu().numpy()

    def _similarity_features(self, emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
        """[A, B, A-B, A*B] — standard sentence similarity feature vector."""
        return np.concatenate([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b])

    def predict(self, audio_path: Path, word: str) -> dict:
        """
        Score a single blending nonwords audio file.

        Parameters
        ----------
        audio_path : path to the participant's .wav response
        word       : normalised target word (must be in TARGET_WORDS)

        Returns
        -------
        dict with keys: score (0/1), confidence (float), needs_review (bool)
        """
        word = normalize_word(word)

        if word not in self._ref_embeddings:
            raise ValueError(f"No reference embedding for word '{word}'.")

        audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        duration_ms = len(audio) / sr * 1000.0

        if duration_ms > DURATION_THRESHOLD_MS:
            # Flag for RA review — likely a recording artefact.
            return {"score": None, "confidence": None, "needs_review": True}

        emb_resp = self._encode(audio)
        emb_ref  = self._ref_embeddings[word]
        sim_feat = self._similarity_features(emb_resp, emb_ref)

        diff_w    = WORD_DIFFICULTY_WEIGHT.get(word, 1.0)
        diff_feat = np.array([_DIFF_MAP.get(diff_w, 0.5)], dtype=np.float32)
        feat      = np.concatenate([sim_feat, diff_feat]).reshape(1, -1)

        score      = int(self.classifier.predict(feat)[0])
        proba      = self.classifier.predict_proba(feat)[0]
        confidence = round(float(proba[score]), 4)

        return {"score": score, "confidence": confidence, "needs_review": False}


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def normalize_word(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z']+", " ", s)
    return re.sub(r"\s+", " ", s).strip()
