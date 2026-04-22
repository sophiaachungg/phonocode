"""
xai_attention_rollout.py
========================
Post-hoc XAI analysis for the Phonocode Wav2Vec2 + MLP model.

Implements:
  1. Attention rollout from the frozen wav2vec2 encoder
  2. Occlusion sensitivity on the raw waveform
  3. Noise sanity check — compares attributions on clean vs. noisy samples

HOW ATTENTION ROLLOUT WORKS (wav2vec2 specifics)
-------------------------------------------------
Wav2Vec2's transformer encoder has 12 layers. Each layer runs multi-head
self-attention: every time frame (token) attends to every other time frame,
producing a (T x T) attention weight matrix per head. These weights tell you
"how much did frame i look at frame j when building its representation?"

The problem: each layer's attention matrix only reflects *that layer's*
direct connections. Information actually flows through all 12 layers
multiplicatively — layer 2's attention acts on layer 1's already-mixed
representations, not on raw frames. Looking at only the last layer's
attention therefore misses most of the signal.

Attention rollout (Abnar & Zuidema, 2020) propagates attention through all
layers by matrix-multiplying them in sequence, with a residual identity term
added at each step to account for skip connections:

    R_0 = I                          (identity: each frame attends to itself)
    R_l = (A_l + I) / 2  *  R_{l-1} (blend attention + skip, then propagate)

where A_l is the mean-over-heads attention matrix at layer l, averaged to
[T x T]. After 12 layers, R_12[i, j] represents the total attention flow
from output frame i back to input frame j.

For classification, we want: "which input frames drove the final
representation the most?" We take the row of R_12 corresponding to the
[CLS]-equivalent token (index 0 for wav2vec2) and get a T-length importance
vector over time frames. Each frame maps back to a time window in the audio
via the encoder's convolutional feature extractor stride (~20ms per frame
at 16kHz with default wav2vec2 settings).

WHAT TO LOOK FOR
----------------
- High-rollout frames should cluster around the speech onset, not at the
  beginning/end of silent padding or background noise.
- If rollout mass concentrates on the pre-speech or post-speech region
  consistently across participants → the model may be using noise/silence
  as a cue, and your CV accuracy is partially spurious.
- If rollout patterns differ systematically between correct and incorrect
  predictions → that's diagnostic of where the model fails.

USAGE
-----
    python xai_attention_rollout.py \
        --audio_root ../data_processed/phoneme_reversal \
        --gt_csv     ../scoring/phoneme_reversal_ground_truth.csv \
        --results_dir ../results_wav2vec2_cv2 \
        --output_dir  ../xai_results \
        --fold 1              # which fold's checkpoint to use (1-5)
        --n_samples 20        # how many test samples to analyse (default: all)
        --noise_snr_db 15     # SNR for noise sanity check
"""

import re
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel

# ---------------------------------------------------------------------------
# Match constants from training script exactly
# ---------------------------------------------------------------------------
MODEL_ID   = "facebook/wav2vec2-base-960h"
TARGET_SR  = 16000
RANDOM_SEED = 42

SKIP_WORDS = {"an"}

import pandas as pd

WORD_DIFFICULTY_WEIGHT = {
    "sit": 0.5, "be": 0.5, "pet": 0.5, "sun": 0.5,
    "to": 0.5, "do": 0.5, "speed": 0.5, "in": 0.5,
    "at": 1.0, "see": 1.0, "seven": 1.0, "spoon": 1.0,
    "dime": 1.0, "pile": 1.0, "cheek": 1.0,
    "state": 2.0, "boots": 2.0, "system": 2.0,
    "midnight": 2.0, "baseball": 2.0, "sometimes": 2.0,
}

# wav2vec2-base has 12 transformer layers, 768 hidden dim, 8 attention heads
# The CNN feature extractor produces one frame per ~20ms at 16kHz
# (stride = 320 samples = 20ms)
WAV2VEC2_FRAME_STRIDE_SAMPLES = 320   # samples per frame
WAV2VEC2_N_LAYERS = 12


# ---------------------------------------------------------------------------
# Utilities (mirror training script)
# ---------------------------------------------------------------------------

def parse_filename(path: Path) -> Tuple[str, str, str]:
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
    gt = pd.read_csv(gt_path)
    id_col        = "participant_id"
    non_word_cols = ["RA_preversal"]
    word_cols     = [c for c in gt.columns if c not in [id_col] + non_word_cols]
    gt_long = gt.melt(id_vars=[id_col], value_vars=word_cols,
                      var_name="target_word", value_name="ground_truth_label")
    gt_long["target_word_norm"] = gt_long["target_word"].apply(normalize_word)
    gt_long = gt_long.dropna(subset=["ground_truth_label"])
    gt_long["ground_truth_label"] = gt_long["ground_truth_label"].astype(int)
    label_map: Dict[Tuple[str, str], int] = {}
    for _, row in gt_long.iterrows():
        label_map[(str(row[id_col]), row["target_word_norm"])] = int(row["ground_truth_label"])
    return label_map


def frames_to_seconds(frame_idx: np.ndarray, sr: int = TARGET_SR,
                       stride: int = WAV2VEC2_FRAME_STRIDE_SAMPLES) -> np.ndarray:
    """Convert wav2vec2 frame indices to time in seconds."""
    return (frame_idx * stride) / sr


# ---------------------------------------------------------------------------
# MLP — must match training script exactly
# ---------------------------------------------------------------------------

def build_mlp(input_dim: int = 1537, num_classes: int = 2) -> nn.Module:
    """
    Reconstruct the MLP architecture from the training script.
    Must be kept in sync with the MLP class definition there.
    """
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
    return MLP()

# ---------------------------------------------------------------------------
# Embedding extraction (single sample)
# ---------------------------------------------------------------------------

def extract_embedding_single(
    audio: np.ndarray,
    processor,
    wav2vec,
    device: torch.device,
    word_norm: str,
) -> np.ndarray:
    """
    Extract the 1537-dim embedding for a single audio array.
    Mirrors the logic in the training script's extract_embeddings().
    """
    inputs = processor(
        audio, sampling_rate=TARGET_SR,
        return_tensors="pt", padding="longest",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs  = wav2vec(**inputs)
        hidden   = outputs.last_hidden_state          # [1, T, 768]
        mean_emb = hidden.mean(dim=1).squeeze(0)      # [768]
        max_emb  = hidden.max(dim=1).values.squeeze(0)  # [768]
        pooled   = torch.cat([mean_emb, max_emb], dim=0)  # [1536]

    diff_w    = WORD_DIFFICULTY_WEIGHT.get(word_norm, 1.0)
    diff_map  = {0.5: 0.0, 1.0: 0.5, 2.0: 1.0}
    diff_feat = np.array([diff_map.get(diff_w, 0.5)], dtype=np.float32)
    emb = np.concatenate([pooled.cpu().numpy(), diff_feat])  # [1537]
    return emb


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------

def compute_attention_rollout(
    audio: np.ndarray,
    processor,
    wav2vec,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run wav2vec2 with output_attentions=True and apply rollout.

    Returns
    -------
    rollout : np.ndarray  shape [T]
        Per-frame importance score (higher = more attended to overall).
    frame_times : np.ndarray  shape [T]
        Centre time in seconds for each frame.

    How this works step by step
    ---------------------------
    1. We call wav2vec2 with output_attentions=True. This returns a tuple of
       12 attention tensors, one per transformer layer, each of shape
       [batch, n_heads, T, T].

    2. We average over the 8 attention heads to get one [T, T] matrix per
       layer. This is a standard simplification — it treats all heads as
       equally informative, which is not quite true but is a good baseline.

    3. We add a scaled identity matrix to each layer's attention matrix:
           A_l_aug = 0.5 * A_l + 0.5 * I
       This accounts for the residual (skip) connection in each transformer
       block: even if attention weight from frame j to frame i is zero,
       frame i's own representation still flows forward directly.

    4. We chain the augmented matrices by matrix multiplication from layer 1
       through layer 12:
           R = A_1_aug @ A_2_aug @ ... @ A_12_aug
       R[i, j] now captures the total information flow from input frame j
       to output frame i across all layers.

    5. We take row 0 (the first token, analogous to CLS) as the global
       importance vector. This gives us T scores, one per input frame.

    6. We normalise to [0, 1] for interpretability.
    """
    inputs = processor(
        audio, sampling_rate=TARGET_SR,
        return_tensors="pt", padding="longest",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = wav2vec(
            **inputs,
            output_attentions=True,   # <-- this is the key flag
        )

    # outputs.attentions: tuple of 12 tensors, each [1, 8, T, T]
    attentions = outputs.attentions
    T = attentions[0].shape[-1]

    # Start with identity (each frame attends to itself perfectly)
    rollout = torch.eye(T, device=device)

    for attn in attentions:
        # attn: [1, 8, T, T] → average over heads → [T, T]
        attn_mean = attn.squeeze(0).mean(dim=0)  # [T, T]

        # Add residual identity and renormalise rows
        attn_aug  = 0.5 * attn_mean + 0.5 * torch.eye(T, device=device)
        attn_aug  = attn_aug / attn_aug.sum(dim=-1, keepdim=True)

        # Propagate: R = A_aug @ R  (chain information flow forward)
        rollout = attn_aug @ rollout

    # Row 0 = how much each input frame contributed to the first output token
    # (analogous to CLS in BERT; wav2vec2 doesn't have explicit CLS but the
    # first frame aggregates global context after pooling anyway)
    importance = rollout[0].cpu().numpy()                  # [T]
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-9)

    # Map frames to time
    frame_indices = np.arange(T)
    frame_times   = frames_to_seconds(frame_indices)

    return importance, frame_times


# ---------------------------------------------------------------------------
# Occlusion sensitivity
# ---------------------------------------------------------------------------

def compute_occlusion_sensitivity(
    audio: np.ndarray,
    processor,
    wav2vec,
    mlp: nn.Module,
    device: torch.device,
    word_norm: str,
    target_class: int,
    window_ms: float = 100.0,
    stride_ms: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a silence window over the waveform. At each position, measure
    the drop in predicted probability for the target class. Large drops
    indicate the model relied on that time region.

    Parameters
    ----------
    window_ms : float   Width of the silence window in milliseconds.
    stride_ms : float   Step size in milliseconds.

    Returns
    -------
    sensitivity  : np.ndarray  shape [n_windows]  probability drop at each position
    window_times : np.ndarray  shape [n_windows]  centre time (seconds) of each window
    """
    window_samples = int(window_ms * TARGET_SR / 1000)
    stride_samples = int(stride_ms * TARGET_SR / 1000)
    n_samples      = len(audio)

    mlp.eval()

    # Baseline prediction (no occlusion)
    baseline_emb = extract_embedding_single(audio, processor, wav2vec, device, word_norm)
    baseline_inp = torch.tensor(baseline_emb, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        baseline_prob = torch.softmax(mlp(baseline_inp), dim=-1)[0, target_class].item()

    sensitivities = []
    window_centres = []

    start = 0
    while start + window_samples <= n_samples:
        # Silence the window
        occluded         = audio.copy()
        occluded[start : start + window_samples] = 0.0

        emb = extract_embedding_single(occluded, processor, wav2vec, device, word_norm)
        inp = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(mlp(inp), dim=-1)[0, target_class].item()

        # Positive = the model relied on this window (probability dropped)
        sensitivities.append(baseline_prob - prob)
        window_centres.append((start + window_samples / 2) / TARGET_SR)

        start += stride_samples

    return np.array(sensitivities), np.array(window_centres)


# ---------------------------------------------------------------------------
# Noise sanity check
# ---------------------------------------------------------------------------

def noise_sanity_check(
    audio: np.ndarray,
    processor,
    wav2vec,
    device: torch.device,
    snr_db: float = 15.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare attention rollout on clean vs. white-noise-corrupted audio.

    If the rollout pattern shifts substantially toward regions that are
    noise-dominated (typically the silence before/after speech), the model
    is attending to noise rather than phonological content.

    Returns rollout_clean, times_clean, rollout_noisy, times_noisy.
    """
    rng = np.random.default_rng(seed)

    # Add white noise at specified SNR
    sig_rms   = np.sqrt(np.mean(audio ** 2)) + 1e-9
    noise_rms = sig_rms / (10 ** (snr_db / 20))
    noise     = rng.standard_normal(len(audio)).astype(np.float32) * noise_rms
    noisy     = np.clip(audio + noise, -1.0, 1.0)

    rollout_clean, times_clean = compute_attention_rollout(audio, processor, wav2vec, device)
    rollout_noisy, times_noisy = compute_attention_rollout(noisy, processor, wav2vec, device)

    return rollout_clean, times_clean, rollout_noisy, times_noisy


def estimate_speech_onset(audio: np.ndarray, sr: int = TARGET_SR,
                           frame_ms: float = 20.0, threshold_factor: float = 0.15) -> float:
    """
    Rough energy-based speech onset estimate. Returns onset time in seconds.
    Used to divide 'before speech' vs 'during speech' for the sanity check.
    """
    frame_samples = int(frame_ms * sr / 1000)
    energy = np.array([
        np.sqrt(np.mean(audio[i:i+frame_samples]**2))
        for i in range(0, len(audio) - frame_samples, frame_samples)
    ])
    threshold = energy.max() * threshold_factor
    onset_frame = np.argmax(energy > threshold)
    return onset_frame * frame_ms / 1000.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rollout(
    audio: np.ndarray,
    rollout: np.ndarray,
    frame_times: np.ndarray,
    title: str,
    save_path: Path,
    speech_onset: Optional[float] = None,
):
    """
    Three-panel plot:
      top    — raw waveform
      middle — attention rollout over time
      bottom — rollout overlaid as a heatmap on the waveform
    """
    sr = TARGET_SR
    t_audio = np.linspace(0, len(audio) / sr, len(audio))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

    # ---- Panel 1: waveform ----
    axes[0].plot(t_audio, audio, color='steelblue', linewidth=0.6, alpha=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(title, fontsize=13)
    axes[0].set_xlim(0, t_audio[-1])
    if speech_onset is not None:
        axes[0].axvline(speech_onset, color='red', linestyle='--', alpha=0.7,
                        linewidth=1.2, label=f"Speech onset (~{speech_onset:.2f}s)")
        axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # ---- Panel 2: rollout curve ----
    axes[1].fill_between(frame_times, rollout, alpha=0.6, color='darkorange')
    axes[1].plot(frame_times, rollout, color='darkorange', linewidth=1.2)
    axes[1].set_ylabel("Rollout importance")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(0, t_audio[-1])
    axes[1].set_ylim(0, 1.05)
    if speech_onset is not None:
        axes[1].axvline(speech_onset, color='red', linestyle='--', alpha=0.7, linewidth=1.2)
    axes[1].grid(True, alpha=0.2)

    # ---- Panel 3: rollout as heatmap intensity on waveform ----
    # Interpolate frame-level rollout to sample-level for overlay
    rollout_interp = np.interp(t_audio, frame_times, rollout)
    axes[2].plot(t_audio, audio, color='steelblue', linewidth=0.5, alpha=0.4)
    sc = axes[2].scatter(t_audio[::20], audio[::20],
                         c=rollout_interp[::20],
                         cmap='hot', s=1.5, vmin=0, vmax=1, alpha=0.8)
    plt.colorbar(sc, ax=axes[2], label='Rollout importance')
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_xlim(0, t_audio[-1])
    if speech_onset is not None:
        axes[2].axvline(speech_onset, color='cyan', linestyle='--', alpha=0.8, linewidth=1.2)
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_noise_sanity(
    audio: np.ndarray,
    rollout_clean: np.ndarray,
    times_clean: np.ndarray,
    rollout_noisy: np.ndarray,
    times_noisy: np.ndarray,
    speech_onset: float,
    snr_db: float,
    title: str,
    save_path: Path,
):
    """
    Side-by-side rollout comparison: clean vs. noisy.
    Shades the pre-speech region to highlight noise leakage.
    """
    sr = TARGET_SR
    t_audio = np.linspace(0, len(audio) / sr, len(audio))
    duration = t_audio[-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    for col, (rollout, label, color) in enumerate([
        (rollout_clean, "Clean", "darkorange"),
        (rollout_noisy, f"Noisy (SNR={snr_db}dB)", "crimson"),
    ]):
        times = times_clean if col == 0 else times_noisy

        # Top: waveform + rollout overlay
        axes[0, col].plot(t_audio, audio if col == 0 else
                          np.interp(t_audio, times, rollout),
                          color='steelblue', linewidth=0.5, alpha=0.5)
        axes[0, col].fill_between(times, rollout, alpha=0.45, color=color)
        axes[0, col].plot(times, rollout, color=color, linewidth=1.2, label=label)
        axes[0, col].axvline(speech_onset, color='red', linestyle='--',
                             linewidth=1.2, label='Speech onset')
        # Shade pre-speech
        axes[0, col].axvspan(0, speech_onset, alpha=0.08, color='red',
                             label='Pre-speech region')
        axes[0, col].set_title(f"Rollout — {label}", fontsize=12)
        axes[0, col].set_xlim(0, duration)
        axes[0, col].set_ylim(0, 1.05)
        axes[0, col].set_xlabel("Time (s)")
        axes[0, col].set_ylabel("Rollout importance")
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.2)

        # Bottom: bar chart — pre-speech vs. post-onset mass
        pre_mask  = times < speech_onset
        post_mask = times >= speech_onset
        pre_mass  = rollout[pre_mask].sum()  if pre_mask.any()  else 0.0
        post_mass = rollout[post_mask].sum() if post_mask.any() else 0.0
        total     = pre_mass + post_mass + 1e-9

        axes[1, col].bar(["Pre-speech", "Speech region"],
                         [pre_mass / total, post_mass / total],
                         color=['#e05c5c', '#5c9ee0'], alpha=0.85, width=0.5)
        axes[1, col].set_ylabel("Fraction of rollout mass")
        axes[1, col].set_title(f"Rollout distribution — {label}", fontsize=12)
        axes[1, col].set_ylim(0, 1)
        axes[1, col].grid(True, alpha=0.2, axis='y')

        for bar_i, v in enumerate([pre_mass / total, post_mass / total]):
            axes[1, col].text(bar_i, v + 0.02, f"{v:.2f}", ha='center', fontsize=11)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_occlusion(
    sensitivity: np.ndarray,
    window_times: np.ndarray,
    audio: np.ndarray,
    speech_onset: float,
    title: str,
    save_path: Path,
):
    sr = TARGET_SR
    t_audio = np.linspace(0, len(audio) / sr, len(audio))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    axes[0].plot(t_audio, audio, color='steelblue', linewidth=0.6, alpha=0.7)
    axes[0].axvline(speech_onset, color='red', linestyle='--', linewidth=1.2,
                    label='Speech onset')
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(title, fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    axes[1].fill_between(window_times, sensitivity.clip(min=0), alpha=0.6, color='teal')
    axes[1].fill_between(window_times, sensitivity.clip(max=0), alpha=0.4, color='salmon')
    axes[1].plot(window_times, sensitivity, color='teal', linewidth=1.2)
    axes[1].axhline(0, color='k', linewidth=0.8)
    axes[1].axvline(speech_onset, color='red', linestyle='--', linewidth=1.2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Prob. drop (higher = more important)")
    axes[1].set_title("Occlusion sensitivity", fontsize=12)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="XAI: attention rollout + occlusion for Phonocode")
    parser.add_argument("--audio_root",  type=Path, default=Path("../data_processed/phoneme_reversal"))
    parser.add_argument("--gt_csv",      type=Path, default=Path("../scoring/phoneme_reversal_ground_truth.csv"))
    parser.add_argument("--results_dir", type=Path, default=Path("../results_wav2vec2_cv1"),
                        help="Directory containing fold subdirs with mlp_best_model.pt")
    parser.add_argument("--output_dir",  type=Path, default=Path("../xai_results"))
    parser.add_argument("--fold",        type=int,  default=1,
                        help="Which fold checkpoint to load (1-5)")
    parser.add_argument("--n_samples",   type=int,  default=None,
                        help="Max samples to analyse (default: all test samples for the fold)")
    parser.add_argument("--noise_snr_db",type=float, default=15.0,
                        help="SNR in dB for noise sanity check")
    parser.add_argument("--run_occlusion", action="store_true",
                        help="Also run occlusion sensitivity (slower — one forward pass per window)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_output = args.output_dir / f"fold{args.fold}"
    fold_output.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Phonocode XAI — Attention Rollout  [fold {args.fold}]")
    print("=" * 60)

    # ---- Device ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print(x)
    else:
        device = torch.device("cpu")
        print("Device: cpu")

    # ---- Load wav2vec2 (frozen) ----
    print(f"\nLoading {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    wav2vec   = AutoModel.from_pretrained(MODEL_ID)
    wav2vec.to(device).eval()
    for p in wav2vec.parameters():
        p.requires_grad = False
    print("Encoder loaded.")

    # ---- Load MLP checkpoint ----
    ckpt_path = args.results_dir / f"fold{args.fold}" / "mlp_best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run training first.")
    ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
    mlp  = build_mlp(input_dim=1537, num_classes=2).to(device)
    mlp.load_state_dict(ckpt['model_state_dict'])
    mlp.eval()
    print(f"MLP loaded from {ckpt_path}  (best epoch: {ckpt.get('epoch', '?')})")

    # ---- Load fold split to know which participants are in the test set ----
    split_path = args.results_dir / f"fold{args.fold}" / "fold_split_info.json"
    if not split_path.exists():
        raise FileNotFoundError(f"No fold split info at {split_path}.")
    with open(split_path) as f:
        split_info = json.load(f)
    test_pids = set(split_info["test_participants"])
    print(f"Test participants for fold {args.fold}: {sorted(test_pids)}")

    # ---- Build label map ----
    label_map = build_label_map(args.gt_csv)

    # ---- Collect test wav files ----
    wav_files = sorted(args.audio_root.rglob("*.wav"))
    test_files = []
    for wav_path in wav_files:
        try:
            pid, _, word = parse_filename(wav_path)
        except ValueError:
            continue
        word_norm = normalize_word(word)
        if word_norm in SKIP_WORDS:
            continue
        if pid not in test_pids:
            continue
        key = (pid, word_norm)
        if key not in label_map:
            continue
        test_files.append((wav_path, pid, word_norm, label_map[key]))

    if args.n_samples is not None:
        rng = np.random.default_rng(RANDOM_SEED)
        indices = rng.choice(len(test_files), size=min(args.n_samples, len(test_files)), replace=False)
        test_files = [test_files[i] for i in sorted(indices)]

    print(f"\nAnalysing {len(test_files)} test samples...")

    # ---- Per-sample analysis ----
    summary_rows = []

    for sample_idx, (wav_path, pid, word_norm, true_label) in enumerate(test_files, 1):
        print(f"  [{sample_idx}/{len(test_files)}]  {wav_path.name}  "
              f"(pid={pid}, word={word_norm}, label={true_label})")
        sys.stdout.flush()

        audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        duration  = len(audio) / TARGET_SR

        # MLP prediction
        emb  = extract_embedding_single(audio, processor, wav2vec, device, word_norm)
        inp  = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = mlp(inp)
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_label = int(probs.argmax())
        correct    = (pred_label == true_label)

        # Speech onset estimate
        speech_onset = estimate_speech_onset(audio)

        # --- Attention rollout ---
        rollout, frame_times = compute_attention_rollout(audio, processor, wav2vec, device)

        # Rollout mass: fraction falling before vs. after speech onset
        pre_mask   = frame_times < speech_onset
        post_mask  = frame_times >= speech_onset
        total_mass = rollout.sum() + 1e-9
        pre_mass_frac  = rollout[pre_mask].sum()  / total_mass if pre_mask.any()  else 0.0
        post_mass_frac = rollout[post_mask].sum() / total_mass if post_mask.any() else 0.0

        # Peak rollout time
        peak_time = frame_times[np.argmax(rollout)]

        # Plot rollout
        sample_dir = fold_output / f"sample_{sample_idx:03d}_{pid}_{word_norm}"
        sample_dir.mkdir(exist_ok=True)

        plot_rollout(
            audio, rollout, frame_times,
            title=(f"{pid} | '{word_norm}' | true={true_label} pred={pred_label} "
                   f"({'✓' if correct else '✗'})  p={probs[pred_label]:.2f}"),
            save_path=sample_dir / "attention_rollout.png",
            speech_onset=speech_onset,
        )

        # --- Noise sanity check ---
        rollout_clean, tc, rollout_noisy, tn = noise_sanity_check(
            audio, processor, wav2vec, device,
            snr_db=args.noise_snr_db, seed=RANDOM_SEED + sample_idx,
        )
        # Leakage delta: how much more pre-speech mass under noise vs. clean
        pre_noisy = rollout_noisy[tn < speech_onset].sum() / (rollout_noisy.sum() + 1e-9) \
                    if (tn < speech_onset).any() else 0.0
        pre_clean = rollout_clean[tc < speech_onset].sum() / (rollout_clean.sum() + 1e-9) \
                    if (tc < speech_onset).any() else 0.0
        noise_leakage_delta = float(pre_noisy - pre_clean)

        plot_noise_sanity(
            audio, rollout_clean, tc, rollout_noisy, tn,
            speech_onset=speech_onset,
            snr_db=args.noise_snr_db,
            title=(f"Noise sanity — {pid} | '{word_norm}'  "
                   f"leakage Δ={noise_leakage_delta:+.3f}"),
            save_path=sample_dir / "noise_sanity.png",
        )

        # --- Occlusion (optional, slow) ---
        occlusion_peak_time = None
        if args.run_occlusion:
            sensitivity, window_times = compute_occlusion_sensitivity(
                audio, processor, wav2vec, mlp, device, word_norm,
                target_class=true_label,
                window_ms=100.0, stride_ms=50.0,
            )
            plot_occlusion(
                sensitivity, window_times, audio, speech_onset,
                title=f"Occlusion — {pid} | '{word_norm}' | true={true_label}",
                save_path=sample_dir / "occlusion_sensitivity.png",
            )
            if len(sensitivity) > 0:
                occlusion_peak_time = float(window_times[np.argmax(sensitivity)])

        summary_rows.append({
            "sample_idx":          sample_idx,
            "wav_file":            wav_path.name,
            "participant_id":      pid,
            "word":                word_norm,
            "true_label":          true_label,
            "pred_label":          pred_label,
            "correct":             correct,
            "prob_class0":         float(probs[0]),
            "prob_class1":         float(probs[1]),
            "speech_onset_s":      float(speech_onset),
            "duration_s":          float(duration),
            "rollout_peak_time_s": float(peak_time),
            "rollout_pre_speech_frac":  float(pre_mass_frac),
            "rollout_post_speech_frac": float(post_mass_frac),
            "noise_leakage_delta":      noise_leakage_delta,
            "occlusion_peak_time_s":    occlusion_peak_time,
        })

    # ---- Aggregate summary ----
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(fold_output / "xai_summary.csv", index=False)

    # Split correct vs. incorrect predictions
    correct_df   = summary_df[summary_df["correct"]]
    incorrect_df = summary_df[~summary_df["correct"]]

    print("\n" + "=" * 60)
    print("XAI SUMMARY")
    print("=" * 60)
    print(f"  Samples analysed:   {len(summary_df)}")
    print(f"  Correct:            {len(correct_df)}  ({len(correct_df)/len(summary_df)*100:.1f}%)")
    print(f"  Incorrect:          {len(incorrect_df)}")
    print()
    print("  Attention rollout — mean pre-speech fraction:")
    print(f"    Correct preds:    {correct_df['rollout_pre_speech_frac'].mean():.3f}")
    print(f"    Incorrect preds:  {incorrect_df['rollout_pre_speech_frac'].mean():.3f}"
          if len(incorrect_df) > 0 else "    Incorrect preds:  n/a")
    print()
    print(f"  Noise leakage Δ (mean): {summary_df['noise_leakage_delta'].mean():+.4f}")
    print(f"    (positive = rollout shifts toward noise region when noise added)")
    print(f"    (near zero = model attends to phonological content, not noise)")
    print()

    # Flag suspicious samples: high pre-speech rollout AND incorrect
    suspicious = summary_df[
        (summary_df["rollout_pre_speech_frac"] > 0.4) & (~summary_df["correct"])
    ]
    if len(suspicious) > 0:
        print(f"  ⚠ Suspicious samples (pre-speech rollout >40% + wrong prediction):")
        for _, row in suspicious.iterrows():
            print(f"    {row['wav_file']}  pre_frac={row['rollout_pre_speech_frac']:.2f}  "
                  f"noise_Δ={row['noise_leakage_delta']:+.3f}")
    else:
        print("  No suspicious samples flagged (pre-speech rollout >40% + incorrect).")

    print(f"\nResults saved to: {fold_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
