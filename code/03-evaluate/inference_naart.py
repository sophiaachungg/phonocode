"""
inference_naart.py
-------------------
Inference for the NAART (North American Adult Reading Test) task.

Pipeline: frozen Wav2Vec 2.0 encoder → hybrid pooling → difficulty scalar → MLP.

The stimulus word list is hardcoded here. The ground truth CSV is NOT required
at inference time — it contains participant scores (research data) and has no
role in scoring new audio.

NAART-specific handling
-----------------------
- gouge and placebo are practice/warm-up items and are excluded from scoring.
- Some wav file stems use filesystem-safe names (e.g. 'hors-doeuvre') that
  differ from the canonical word names used as keys. FILESTEM_TO_COLUMN maps
  these back.
- Missing audio files (participant skipped the item) are treated as incorrect (0).
  The server handles this by returning score=0, confidence=1.0 for missing items.
"""

import re
from pathlib import Path
from typing import Dict

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

# Words excluded from training and scoring (practice/warm-up items).
SKIP_WORDS = {"gouge", "placebo"}

# Ordered stimulus list — defines expected words and DataFrame column order.
TARGET_WORDS = [
    "psalm", "depot", "equivocal", "bouquet", "indict", "caveat",
    "hors d'oeuvre", "paradigm", "corps", "impugn", "recipe", "aisle",
    "subtle", "quadrupled", "simile", "heir", "topiary", "zealot",
    "epitome", "colonel", "lingerie", "debt", "facade", "hiatus",
    "catacomb", "rarefy", "prelate", "procreate", "reign", "gist",
    "subpoena", "debris",
    "abstemious", "ennui", "detente", "assignate", "sieve", "epergne",
    "aeon", "gauche", "reify", "radix", "indices", "leviathan",
    "superfluous", "gauge", "cellist", "banal",
    "ci-devant", "drachm", "talipes", "gaoled", "demesne", "vivace",
    "sidereal", "beatify", "synecdoche", "capon", "syncope",
]

# Maps filesystem-safe wav stem words → canonical column names.
FILESTEM_TO_COLUMN: Dict[str, str] = {
    "hors-doeuvre": "hors d'oeuvre",
    "dÃ©tente":     "detente",
    "faÃ§ade":      "facade",
    "detente":      "detente",
    "facade":       "facade",
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
WORD_DIFFICULTY_WEIGHT["epergne"] = 1.0
_DIFF_MAP = {0.5: 0.0, 1.0: 0.5, 2.0: 1.0}


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

class NAARTScorer:
    """
    Scores NAART audio files.

    Parameters
    ----------
    model_path : path to final_model.pt produced by naart_train_final.py
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

        # Encoder
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.encoder   = AutoModel.from_pretrained(MODEL_ID).to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Checkpoint
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        state_dict  = ckpt.get("model_state_dict", ckpt)
        input_dim   = ckpt.get("input_dim",   state_dict["net.0.weight"].shape[1])
        num_classes = ckpt.get("num_classes", state_dict["net.6.weight"].shape[0])

        self.threshold    = threshold if threshold is not None else ckpt.get("threshold", 0.5)
        skip_norms        = {_normalize(w) for w in SKIP_WORDS}
        self.target_words = [w for w in TARGET_WORDS if _normalize(w) not in skip_norms]

        self.classifier = MLP(input_dim, num_classes).to(self.device)
        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()
        self._input_dim = input_dim

        print(f"[naart] Loaded. input_dim={input_dim}  threshold={self.threshold}  device={device}")

    # ------------------------------------------------------------------

    def filestem_to_word(self, raw_stem: str) -> str:
        """Resolve a wav filestem word to its canonical word name."""
        return FILESTEM_TO_COLUMN.get(raw_stem.strip(), raw_stem.strip())

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

        word_norm = _normalize(word)
        diff_w    = WORD_DIFFICULTY_WEIGHT.get(word_norm, 1.0)
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
        Score a single NAART audio file.

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

    def predict_missing(self) -> dict:
        """
        Return a score of 0 for a missing audio file (participant skipped the item).
        Matches the training behaviour: missing files are silence-injected and
        scored 0 in ground truth.
        """
        return {"score": 0, "confidence": 1.0, "needs_review": False}


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
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


normalize_word = _normalize
