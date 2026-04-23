# PhonoCode

Automated scoring of phonological task responses using transformer-based speech embeddings. Built for the **[Communication and Language Lab (CaLL)](https://www.thecommunicationandlanguagelab.com/)** at Vanderbilt University to reduce manual coding burden in speech and language research.

PhonoCode currently supports three tasks: **phoneme reversal**, **NAART**, and **blending nonwords**. Each task uses a pipeline tailored to its scoring structure.

---

## Table of Contents

- [Lab Context](#lab-context)
- [Why This Matters](#why-this-matters)
- [Why Transformer-Based Speech Models](#why-transformer-based-speech-models)
- [Architecture](#architecture)
  - [Phoneme Reversal and NAART](#phoneme-reversal-and-naart-frozen-wav2vec-20--mlp)
  - [Blending Nonwords](#blending-nonwords-wavlm-large--similarity-features--logistic-regression)
- [Stimulus Difficulty Weighting](#stimulus-difficulty-weighting)
- [Audio Preprocessing](#audio-preprocessing)
- [Model Performance](#model-performance)
- [Data](#data)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Usage](#usage)

---

## Lab Context

The Communication and Language Lab (CaLL) studies individual differences in language processing. PhonoCode targets three tasks from the lab's phonological battery:

**Phoneme reversal** — Participants hear a nonword created by reversing the sounds of a real word and must produce the original word.
- *Example:* hear /næ/ → respond /æn/ ("an")
- *Example:* hear /tʊp/ → respond /pʊt/ ("put")

**NAART (North American Adult Reading Test)** — Participants read aloud a list of exception words whose pronunciation cannot be derived from standard phonological rules (e.g., *choir*, *synecdoche*). Correct pronunciation is scored as an estimate of verbal IQ.

**Blending nonwords** — Participants hear two nonword fragments and must blend them into a single nonword response. The task probes phonological assembly and working memory.

For all 3 of the above tasks, correct answers are scored as 1 while incorrect answers are scored as 0.

In a previous [pilot study](https://github.com/sophiaachungg/phonocode-pilot), I investigated the extent to which state-of-the-art ASR tools are effective off-the-shelf for these tasks, and which speech representation model (Wav2Vec 2.0 or WavLM) works best when adapted for each one. ASR tools are not effective off-the-shelf for this battery. Model selection varies by task structure, as described in the Architecture section below.

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
- Some recordings cut off early, contain room noise, or include hesitations and repairs before the target word.

What's needed is a system that is robust to noise and channel variation, can generalize across speakers rather than memorizing voice identity, and can work with relatively small amounts of labeled data. Self-supervised models like [Wav2Vec 2.0](https://huggingface.co/facebook/wav2vec2-base-960h) and [WavLM](https://huggingface.co/microsoft/wavlm-large) are trained on large unlabeled corpora and learn rich phonetic representations before any task-specific fine-tuning, making them well-suited for this setting.

---

## Architecture

The two tasks differ in their scoring structure, which determines the appropriate pipeline.

Phoneme reversal and NAART are **single-response tasks**: a participant produces one response, and correctness is determined by properties of that response alone. A single embedding is sufficient.

Blending nonwords is a **comparative task**: correctness depends on how closely the participant's blended response matches a canonical target form. This requires comparing two embeddings — the participant response and a reference recording — rather than classifying a single one.

---

### Phoneme Reversal and NAART: Frozen Wav2Vec 2.0 + MLP

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

### Blending Nonwords: WavLM-Large + Similarity Features + Logistic Regression

Blending nonwords uses a fundamentally different approach. Because the task requires evaluating how close a participant's response is to a canonical target form, the model compares two embeddings — one from the participant's response and one from a reference recording of the correct blend — rather than classifying a single response embedding.

```
Participant response (.wav)        Reference recording (.wav)
        │                                    │
        ▼                                    ▼
┌───────────────────────┐       ┌───────────────────────┐
│   WavLM-Large         │       │   WavLM-Large         │
│   (frozen at          │       │   (frozen at          │
│    inference)         │       │    inference)         │
└───────────────────────┘       └───────────────────────┘
        │                                    │
        ▼  [T × 1024]                        ▼  [T × 1024]
┌───────────────────────┐       ┌───────────────────────┐
│   Hybrid pooling      │       │   Hybrid pooling      │
│   mean ⊕ max → [2048] │       │   mean ⊕ max → [2048] │
└───────────────────────┘       └───────────────────────┘
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
        ┌──────────────────────────────┐
        │   Similarity feature vector  │  [A, B, A-B, A*B] → [8192]
        └──────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Difficulty feature         │  scalar ∈ {0.0, 0.5, 1.0} → [8193]
        └──────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Logistic Regression        │  class_weight='balanced'
        └──────────────────────────────┘
                       │
                       ▼
              Correct (1) / Incorrect (0)
```

**Key design decisions:**

- **WavLM-Large instead of Wav2Vec 2.0.** WavLM-Large is trained with a masked speech denoising objective and produces richer representations for out-of-distribution phonology, which matters for novel nonword blends. During CV, the top 2 transformer layers were unfrozen to allow partial fine-tuning; the production model freezes all layers.
- **Similarity feature vector.** The four-part vector `[A, B, A-B, A*B]` — borrowed from sentence similarity literature — preserves both absolute representations and their directional relationship. Cosine similarity alone would collapse this to a scalar and discard that information.
- **Logistic Regression as the final classifier.** In cross-validation, LR matched or exceeded the MLP on these features (0.845 vs. 0.842 accuracy) while showing no train/val gap. The MLP showed meaningful overfitting at this N. LR is used in production.
- **Reference recordings.** One canonical recording of each of the 24 target blends is pre-encoded once and reused across all participant trials. Reference embeddings are never augmented.
- **Duration gate.** Participant responses longer than 8 seconds are excluded, as these are almost certainly recording artefacts rather than genuine responses to a pseudoword blend prompt.

---

## Stimulus Difficulty Weighting

Not all stimuli are equally informative. Stimuli vary in phonological complexity, and harder items are weighted more heavily during training across all three tasks. Difficulty tiers are derived from per-stimulus base-rate correct in the ground truth data. Sample weights during training are the product of the class weight and the difficulty weight for each item, then normalized to unit mean. This feature was inspired by my own experience of running hundreds of participants over the past 3 years and keeping myself entertained during manual scoring by guessing whether participants would get upcoming items correct based on their track record so far. This feature improved the model's prediction abilitiles tremendously.

---

## Audio Preprocessing

All raw audio goes through a standardized preprocessing pipeline before embedding extraction. Raw recordings are collected as `.webm` files and converted to `.wav` using FFmpeg:

- **Format:** WAV
- **Sample rate:** 16,000 Hz (required by both Wav2Vec 2.0 and WavLM)
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

All tasks are evaluated using 5-fold participant-grouped cross-validation. All samples from a given participant appear in exactly one test fold, preventing data leakage from speaker identity. Performance is reported on held-out test folds only; the validation fold is used solely for checkpoint selection (balanced accuracy) and threshold tuning.

### Phoneme Reversal

#### Current Run (50 Participants)

| Fold | Accuracy | Balanced Acc | Macro F1 | Cl0 Recall | Cl1 Recall | Threshold |
|:----:|:--------:|:------------:|:--------:|:----------:|:----------:|:---------:|
| 1 | 0.873 | 0.859 | 0.863 | 0.804 | 0.914 | 0.5 |
| 2 | 0.867 | 0.863 | 0.857 | 0.851 | 0.875 | 0.5 |
| 3 | 0.875 | 0.870 | 0.869 | 0.850 | 0.890 | 0.5 |
| 4 | 0.860 | 0.852 | 0.854 | 0.807 | 0.896 | 0.5 |
| 5 | 0.909 | 0.903 | 0.903 | 0.881 | 0.926 | 0.5 |
| **Mean** | **0.877** | **0.870** | **0.869** | **0.839** | **0.900** | — |
| **Std** | **0.019** | **0.020** | **0.020** | **0.033** | **0.020** | — |

> **Class 0** = incorrect response. **Class 1** = correct response.
> Balanced accuracy is the primary metric.

**Logistic regression baseline:** Acc = 0.868 ± 0.016, Macro F1 = 0.862 ± 0.017

#### Previous Run (36 Participants, hardcoded class weights, val_loss checkpoint selection)

Included for reference. The current run uses inverse-frequency class weights and balanced accuracy for checkpoint selection.

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 0.818 | 0.035 |
| Macro F1 | 0.801 | 0.034 |
| Class-0 Recall | 0.786 | 0.068 |
| Class-1 Recall | 0.829 | 0.076 |

---

### NAART

#### Current Run

| Fold | Accuracy | Balanced Acc | Macro F1 | Cl0 Recall | Cl1 Recall | Threshold |
|:----:|:--------:|:------------:|:--------:|:----------:|:----------:|:---------:|
| 1 | 0.844 | 0.850 | 0.829 | 0.867 | 0.834 | 0.5 |
| 2 | 0.843 | 0.834 | 0.831 | 0.804 | 0.865 | 0.5 |
| 3 | 0.842 | 0.831 | 0.832 | 0.781 | 0.880 | 0.5 |
| 4 | 0.867 | 0.857 | 0.854 | 0.827 | 0.888 | 0.5 |
| 5 | 0.853 | 0.839 | 0.831 | 0.804 | 0.874 | 0.5 |
| **Mean** | **0.850** | **0.842** | **0.835** | **0.816** | **0.868** | — |
| **Std** | **0.011** | **0.011** | **0.010** | **0.032** | **0.021** | — |

> **Class 0** = incorrect pronunciation. **Class 1** = correct pronunciation.
> Balanced accuracy is the primary metric; class-0 recall is the primary clinical signal.

**Logistic regression baseline:** Acc = 0.839 ± 0.014, Macro F1 = 0.826 ± 0.013

---

### Blending Nonwords

Evaluated using the WavLM similarity + LR pipeline. The MLP was also evaluated in cross-validation but showed a consistent train/val gap at this N; LR is used in production.

| Fold | Accuracy | Balanced Acc | Macro F1 | Cl0 Recall | Cl1 Recall | Threshold |
|:----:|:--------:|:------------:|:--------:|:----------:|:----------:|:---------:|
| 1 | 0.859 | 0.849 | 0.853 | 0.789 | 0.910 | 0.5 |
| 2 | 0.828 | 0.827 | 0.822 | 0.821 | 0.832 | 0.5 |
| 3 | 0.842 | 0.850 | 0.836 | 0.882 | 0.818 | 0.5 |
| 4 | 0.880 | 0.868 | 0.872 | 0.819 | 0.918 | 0.5 |
| 5 | 0.802 | 0.799 | 0.800 | 0.771 | 0.827 | 0.5 |
| **Mean** | **0.842** | **0.839** | **0.837** | **0.816** | **0.861** | — |
| **Std** | **0.030** | **0.027** | **0.028** | **0.042** | **0.049** | — |

> **Class 0** = incorrect blend. **Class 1** = correct blend.
> Balanced accuracy is the primary metric; class-0 recall is the primary clinical signal.
> Results are from the Stage 2 WavLM similarity CV run (141 participants).

**Logistic regression baseline:** Acc = 0.845 ± 0.026, Macro F1 = 0.839 ± 0.024

---

## Data

**The dataset is not publicly available.**

Data consists of audio recordings of research participants completing phonological tasks, collected under IRB protocol. Participant recordings contain identifiable voice information and are stored and managed in accordance with the lab's data governance and consent agreements.

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

## Repository Structure

```
phonocode/
├── app/                            ← scoring web server
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── blending_nonwords.py
│   │   ├── naart.py
│   │   └── phoneme_reversal.py
│   ├── static/
│   │   └── index.html
│   ├── README.md                   ← README for the web app
│   └── server.py
├── code/
│   ├── 01-preprocess_audio/
│   │   ├── blending-nonwords_from_csv.py
│   │   ├── naart_from_csv.py
│   │   └── phoneme-reversal_from_csv.py
│   ├── 02-train/
│   │   ├── experiment_5foldcv/
│   │   │   ├── blending-nonwords_train.py
│   │   │   ├── naart_train.py
│   │   │   └── phoneme-reversal_train.py
│   │   └── final_training/
│   │       ├── blending_nonwords_train_final.py
│   │       ├── naart_train_final.py
│   │       └── phoneme_reversal_train_final.py
│   ├── 03-evaluate/
│   │       ├── inference_blending-nonwords.py ← same as app/inference/blending-nonwords.py
│   │       ├── inference_naart.py
│   │       └── inference_phoneme-reversal.py
│   └── 04-xai/
│       ├── xai_attention_rollout.py
│       └── xai_results_all_folds/  ← results from XAI analysis on phoneme reversal task
├── data/
│   ├── processed/                  ← .wav files all housed in a participant folder (e.g., ReXa_123)
│   ├── raw/                        ← original CSVs exported from Gorilla with base64 encoding of recordings
│   └── reference_recordings/       ← canonical blending nonwords recordings
├── logs/                           ← RA scoring logs (created on first submit on app)
├── models/
│   ├── blending-nonwords_final_model.pkl
│   ├── naart_final_model.pt
│   └── phoneme-reversal_final_model.pt
└── README.md
```

---

## Usage

**Step 1: Preprocess audio**

```bash
python preprocess_audio.py ../data_raw/<task> --recursive
```

**Step 2: Run cross-validation training**

For phoneme reversal or NAART (frozen Wav2Vec 2.0 + MLP pipeline):
```bash
python phoneme-reversal_train.py
python naart_train.py
```

For blending nonwords (WavLM similarity + LR pipeline):
```bash
python bn_stage2_wavlm_similarity.py
```

Results are saved to the corresponding `../results_<task>/` directory, including per-fold learning curves, confusion matrices, test predictions (with class-1 probabilities), and an aggregate summary.

**Step 3: Train final production models**

After reviewing CV results, update `FIXED_EPOCHS` and `DEPLOY_THRESHOLD` in the final training scripts from your `cv_fold_results.csv`, then run:

```bash
# Phoneme reversal and NAART
python phoneme_reversal_train_final.py
python naart_train_final.py

# Blending nonwords
python blending_nonwords_train_final.py
```

--- 

## Next Steps

1. **The models are not great at labeling 0s.** The models prove to be reliable for classifying true positives but does not excel at true negatives. Currently, I'm not sure the culprit is necessary class imbalance, but some experimentation here could make a real difference in model deployment.
2. **The confidence scores are not perfect and could use some improvement.** In the lab, we actually use 3 labels - 1 for correct, 0 for incorrect, and NA if the participant accidentally skipped the stimulus or the recording is extremely ambiguous. In this case, I sacrificed the NA label and defaulted to labeling it 0 in the ground truth. The goal here was for the model to learn to mark audio files that used to be NA as 0 and/or with low confidence. 
3. **The Explainable AI (XAI) aspects could be expanded upon.** In [Akman and Schuller, 2024](https://spj.science.org/doi/10.34133/icomputing.0074) there's an example of applying classic XAI techniques such as Grad-CAM, LIME, and SHAP on spectrograms generated from .wav files. There are also emerging techniques such as [X-ASR](https://arxiv.org/abs/2302.14062). Alternatively, the waveforms can be converted to specific values and summary statistics corresponding to concepts for the sake of counterfactual and other concept-based XAI approaches.
4. **This project can be expanded to other phonological tasks.** In the lab, we also do a pseudoword repetition task that takes a significant amount of time to manually score since it is not a matter of 0/1 but is a measure of the syllable streak and the total syllables correct. This task would most likely require audio transformers built for extracting and classifying granular phonemes, rather than full words.

---

## Citation

If you use this code in your research, please cite:

```
@misc{phonocode2026,
  author       = {{Sophia Chung}},
  title        = {PhonoCode: Automated Scoring of Phonological Task Responses},
  year         = {2026},
  howpublished = {\url{https://github.com/<your-org>/phonocode}}
}
```
