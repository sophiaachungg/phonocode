# PhonoCode

Automated scoring of phoneme reversal responses using frozen Wav2Vec 2.0 embeddings and a lightweight MLP classifier. Built for the **[Communication and Language Lab (CaLL)](https://www.thecommunicationandlanguagelab.com/)** at Vanderbilt University to reduce manual coding burden in speech and language research.

---

## Table of Contents

- [Lab Context](#lab-context)
- [Why This Matters](#why-this-matters)
- [Why Transformer-Based Speech Models](#why-transformer-based-speech-models)
- [Architecture](#architecture)
- [Stimulus Difficulty Weighting](#stimulus-difficulty-weighting)
- [Audio Preprocessing](#audio-preprocessing)
- [Model Performance](#model-performance)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)

---

## Lab Context

The Communication and Language Lab (CaLL) studies individual differences in language processing. One task used is **phoneme reversal**:

- Participants hear a nonword created by reversing the sounds of a real word.
- They must reverse it back to produce the real word.
  - *Example:* hear /næ/ → respond /æn/ ("an")
  - *Example:* hear /tʊp/ → respond /pʊt/ ("put")

**Phoneme reversal is just one of many phonological tasks used during experiments.**

In a previous [pilot study](https://github.com/sophiaachungg/phonocode-pilot), I investigated the following:
1. To what extent are state-of-the-art ASR tools effective "off-the-shelf" for this task?
2. Which speech representation model (Wav2Vec 2.0 or WavLM) works best when adapted for this task?

Currently, ASR tools are not effective "off-the-shelf" for this specific battery of phonological tasks. [Wav2Vec 2.0](https://huggingface.co/facebook/wav2vec2-base-960h) works well for this task.

---

## Why This Matters

Participant responses are currently scored by hand:

- Each participant is live-coded by a research assistant (RA) during the session.
- Audio is re-coded by a second RA for inter-rater reliability.
- RAs require ~5 hours of task-specific training before they can score reliably.
- Data entry and validation are a major time and cost sink across psychology labs.

PhonoCode aims to reduce time spent on repetitive scoring, free RAs for higher-value work (experiment design, analysis), and eventually flag low-confidence trials for targeted human review rather than full manual re-coding.

---

## Why Transformer-Based Speech Models

The audio is noisy and variable in ways that rule out simpler approaches:

- RAs may be audible in the background, giving clarifying instructions mid-trial.
- Microphone quality varies across participants and sessions.
- Some trials cut off early, contain room noise, or include hesitations and repairs before the target word.

What's needed is a system that is robust to noise and channel variation, can generalize across speakers rather than memorizing voice identity, and can work with relatively small amounts of labeled data. Self-supervised models like Wav2Vec 2.0 and WavLM are trained on large unlabeled corpora and learn rich phonetic representations before any task-specific fine-tuning — making them well-suited for this setting.

---

## Architecture

PhonoCode uses a **frozen encoder + lightweight MLP** pipeline:

```
Raw audio (.wav, 16kHz mono)
        │
        ▼
┌────────────────────────────┐
│   Wav2Vec 2.0 (frozen)     │  facebook/wav2vec2-base-960h
│   Transformer encoder      │  No gradient updates during training
└────────────────────────────┘
        │
        ▼  last_hidden_state [T × 768]
┌────────────────────────────┐
│   Hybrid pooling           │  mean pool ⊕ max pool → [1536]
└────────────────────────────┘
        │
        ▼
┌────────────────────────────┐
│   Difficulty feature       │  scalar ∈ {0.0, 0.5, 1.0}  → [1537]
└────────────────────────────┘
        │
        ▼
┌────────────────────────────┐
│   MLP classifier           │  Linear(1537→128) → ReLU → Dropout(0.2)
│                            │  Linear(128→64)   → ReLU → Dropout(0.3)
│                            │  Linear(64→2)
└────────────────────────────┘
        │
        ▼
  Correct (1) / Incorrect (0)
```

**Key design decisions:**

- **Frozen encoder.** Wav2Vec 2.0 weights are not updated during training. This avoids overfitting given the limited labeled dataset size and keeps compute requirements low.
- **Hybrid pooling.** Mean and max pooling of the final hidden states are concatenated. Mean pooling captures average phonetic content; max pooling preserves salient features. Together they outperform either alone.
- **Difficulty feature.** A single normalized scalar encoding phonological difficulty tier is appended to every embedding, giving the classifier an explicit context signal (see [Stimulus Difficulty Weighting](#stimulus-difficulty-weighting)).
- **Acoustic augmentation (training only).** Each training clip is augmented with Gaussian noise injection, pitch shifting (±2 semitones), and time stretching (±10%), forcing the model to rely on phonological structure rather than surface acoustic properties. Clean audio is always used for validation and test.
- **Per-fold threshold tuning.** After selecting the best checkpoint, a threshold sweep (0.30–0.50) is run on the validation fold to maximize balanced accuracy before evaluating on the held-out test fold.

---

## Stimulus Difficulty Weighting

Not all stimuli are equally informative. Stimuli vary in phonological complexity, and harder items are weighted more heavily during training.
Sample weights during training are the product of the class weight and the difficulty weight for each item, then normalized to unit mean.

---

## Audio Preprocessing

All raw audio goes through a standardized preprocessing pipeline before embedding extraction. Raw recordings are collected as `.webm` files and converted to `.wav` using FFmpeg:

- **Format:** WAV
- **Sample rate:** 16,000 Hz (required by Wav2Vec 2.0)
- **Channels:** Mono
- **Tool:** FFmpeg via `preprocess_audio.py`

The script validates each output file (sample rate, channel count, non-zero frame count) and skips files that already pass validation unless `--force` is specified.

**Basic usage:**

```bash
# Single participant
python preprocess_audio.py ../data_raw/phoneme_reversal/ReXa_311

# Multiple participants
python preprocess_audio.py ../data_raw/phoneme_reversal/ReXa_311 ../data_raw/phoneme_reversal/ReXa_221

# Entire task directory (preserves subdirectory structure)
python preprocess_audio.py ../data_raw/phoneme_reversal --recursive

# Force reprocess already-converted files
python preprocess_audio.py ../data_raw/phoneme_reversal/ReXa_311 --force

# Custom output directory
python preprocess_audio.py ../data_raw/phoneme_reversal/ReXa_311 --output ../custom_output

# Verify FFmpeg installation
python preprocess_audio.py --check-ffmpeg
```

**Dependencies:** FFmpeg must be installed and on PATH.

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

---

## Model Performance

Evaluated using 5-fold participant-grouped cross-validation. All samples from a given participant appear in exactly one test fold, preventing data leakage from speaker identity. Performance is reported on held-out test folds only; the validation fold is used solely for checkpoint selection (balanced accuracy) and threshold tuning.

### 50-Participant Results

| Fold | Accuracy | Balanced Acc | Macro F1 | Cl0 Recall | Cl1 Recall | Threshold |
|:----:|:--------:|:------------:|:--------:|:----------:|:----------:|:---------:|
| 1 | — | — | — | — | — | — |
| 2 | — | — | — | — | — | — |
| 3 | — | — | — | — | — | — |
| 4 | — | — | — | — | — | — |
| 5 | — | — | — | — | — | — |
| **Mean** | **—** | **—** | **—** | **—** | **—** | — |
| **Std** | **—** | **—** | **—** | **—** | **—** | — |

> **Class 0** = incorrect response (participant failed to reverse the nonword).  
> **Class 1** = correct response.  
> Balanced accuracy is the primary metric; class-0 recall is the primary clinical signal.

**Logistic regression baseline:** Acc = — ± —, Macro F1 = — ± —

---

### Previous Run (36 Participants, hardcoded class weights, val_loss checkpoint selection)

Included for reference. The updated run above uses inverse-frequency class weights and balanced accuracy for checkpoint selection.

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 0.818 | 0.035 |
| Macro F1 | 0.801 | 0.034 |
| Class-0 Recall | 0.786 | 0.068 |
| Class-1 Recall | 0.829 | 0.076 |

---

## Data

**The dataset is not publicly available.**

Data consists of audio recordings of research participants completing the phoneme reversal task, collected under IRB protocol. Participant recordings contain identifiable voice information and are stored and managed in accordance with the lab's data governance and consent agreements.

If you are interested in collaborating or have questions about the dataset, please contact the CaLL lab directly.

The ground truth labels (correct / incorrect per participant × stimulus) were manually coded by trained research assistants with a two-RA inter-rater reliability procedure.

---

## Installation

```bash
git clone https://github.com/<your-org>/phonocode.git
cd phonocode
pip install -r requirements.txt
```

**Core dependencies:**

```
torch
torchaudio
transformers
librosa
scikit-learn
pandas
numpy
matplotlib
soundfile
tqdm
```

---

## Usage

**Step 1: Preprocess audio**

```bash
python preprocess_audio.py ../data_raw/phoneme_reversal --recursive
```

**Step 2: Run cross-validation training**

```bash
python train.py
```

Results are saved to `../training_results/`, including per-fold learning curves, confusion matrices, test predictions (with class-1 probabilities), and an aggregate summary.

---

## Citation

If you use this code in your research, please cite:

```
@misc{phonocode2026,
  author       = {{Sophia Chung}},
  title        = {PhonoCode: Automated Scoring of Phoneme Reversal Responses},
  year         = {2026},
  howpublished = {\url{https://github.com/<your-org>/phonocode}}
}
```
