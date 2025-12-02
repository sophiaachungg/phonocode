# PhonoCode: Transformer-Based Scoring of Phonological Tasks

This repo contains code for experimenting with transformer-based speech models (Wav2Vec 2.0 and WavLM) to *automatically score* phonological task responses.  
Each audio clip is classified as **correct (1)** or **incorrect (0)**.

## 1. Problem

### 1.1 Lab context

The Communication and Language Lab (CaLL) studies **individual differences in language processing**. One task we use is **phoneme reversal**:

- Participants hear a **nonword** created by reversing the sounds of a real word.
- They must **reverse it back** to say the real word.
  - Example: hear /næ/ → respond /æn/ (“an”)
  - Example: hear /tʊp/ → respond /pʊt/ (“put”)

For this pilot:

- Question 1: To what extent are state-of-the-art ASR tools effective "off-the-shelf" for this task?
- Question 2: Which ASR tool (Wav2Vec 2.0 or WavLM) works best fine-tuned for this task?
- **14 participants × 22 audio files each** (`.wav`)
- Each file *should* be a single spoken response (the reversed nonword).
- We have manually coded ground truth for each response (correct vs incorrect).

### 1.2 Why this matters

Right now:

- Each participant is **live-coded** by a research assistant (RA) during the session.
- Then **re-coded from audio** by a second RA for inter-rater reliability.
- Each RA needs ~**5 hours of training** just to learn the scoring scheme.
- In many psychology labs, **data entry + validation is a major time (and money) sink**.

The ROI:

- Reduce time spent on repetitive coding.
- Free RAs for more interesting work (experiment design, analysis).

### 1.3 Why transformer-based speech models?

The audio is messy:

- RAs talking in the background / giving clarifying instructions.
- Variable mic quality; some trials cut off early or have room noise.
- Participants may hesitate, mispronounce, or add extra material before/after the target word.

We need a system that:

- Is **robust to noise and channel variation**.
- Can work **with limited labeled data** (we have ground truth but not a huge corpus).
- Can **generalize across speakers**, not just memorize specific voices.
- Eventually supports **confidence scores** so low-confidence trials can be flagged for human review.

Self-supervised and weakly-supervised transformer models (Wav2Vec 2.0, WavLM) are built exactly for this kind of setting. They learn rich speech representations from huge unlabeled or weakly labeled corpora, and then they can be adapted with relatively small labeled datasets.

---

## 2. Models compared

This project compares two strong ASR backbones:

- **Wav2Vec 2.0** – self-supervised encoder trained with a contrastive objective on quantized latent speech.
- **WavLM** – Wav2Vec 2.0 / HuBERT-style encoder optimized for a broad “full stack” of speech tasks.

### 2.1 Wav2Vec 2.0

![Wav2Vec 2.0 Model Architecture](figures/wav2vec2_architecture.png)

*Figure 1. Wav2Vec 2.0 Model Architecture from Baevski et al., 2020*

- **Self-supervised** transformer encoder trained directly on raw waveforms.
- Uses a CNN feature encoder → masks portions of latent features → Transformer context network → contrastive loss over **quantized speech units**.  
- Breaking down the quantization module: during pre-training, the output of the CNN is fed into the Transformer (branch 1) and to a "Product Quantization" block (branch 2). This module is for generating its own target labels and predicting them within the model.
- Extremely **label-efficient**: with only 10 minutes of labeled Librispeech data, it reaches WER 4.8 / 8.2 (test-clean / test-other), and with full data it achieves state-of-the-art WER while using far less labeled data than previous approaches.
- Uses Convolutional Positional Embedding: runs a convolution over the input features and adds this result to the inputs before they enter the Transformer. Position is encoded implicitly and statically. 

In my pipeline, Wav2Vec 2.0 is used **off-the-shelf** (no additional fine-tuning) as a baseline and as a **frozen encoder**: we extract embeddings for each audio file and train a small classifier head to predict correct (1) vs incorrect (0).

### 2.2 WavLM

![WavLM Model Architecture](figures/wavlm_architecture.png)

*Figure 2. WavLM Model Architecture from Chen et al., 2022*

- Nearly identical to Wav2Vec 2.0 architecture but *without* the quantization module (replaced with a simple classification head for prediction).
- Built on the Wav2Vec 2.0 / HuBERT family but optimized as a **general-purpose speech representation** model for a wide range of tasks (SUPERB benchmark).
- Uses Gated Relative Position Bias: modifies the attention mechanism inside the Transformer layers. It adds a learnable bias to the attention scores based on the relative distance between tokens.
- Adds structured denoising and additional pretraining data (MIX-94k) to better capture speaker, background and other acoustic information.  
- Achieves **State of the Art (SOTA) or near-SOTA results** across tasks like speech separation, speaker verification, diarization, and ASR.

I again use WavLM both **off-the-shelf** and as a **frozen encoder** plus a small classifier head, and compare performance to Wav2Vec 2.0 on the same splits.

## 3. Experimental design

### 3.1 Phase 1 – Off-the-shelf models + regex matching

Main idea: **“What if we just use off-the-shelf ASR and some string matching?”**

Steps:

1. Run WavLM and Wav2Vec 2.0 to **transcribe** each response.
2. Use **regex + fuzzy string matching** to decide if the transcript matches:
   - The expected reversed word.
   - A set of acceptable variants / spellings.
3. Evaluate against the manually coded ground truth (0/1).

![Model Accuracy Per Participant](figures/off_the_shelf_model_acc.png)

*Figure 3. Off-the-shelf model accuracy per participant is unremarkable.*

![Per Item Accuracy](figures/off_the_shelf_per_item_accuracy.png)

*Figure 4. Off-the-shelf model accuracy per task item. Each point indicates a task item. Correct answers are anonymized. Diagonal line indicates equal performance between models.*

What we see:

- Works okay for clean, clear productions.
- **Homophones** are a major problem (e.g., orthographic ambiguity).
- It always outputs *some* word, even when the participant response is incomplete or non-target (“an” vs “Anne”, “put” vs “putt”).
- We’d still need humans to manually inspect borderline / noisy cases, so it wouldn't actually save any time.

Conclusion: this is a reasonable baseline, but **purely transcript-based scoring is not enough** for this task.

### 3.2 Phase 2 – Frozen encoders + binary classifier

Goal: **Stop transcribing; directly classify the waveform** as correct or incorrect.

Pipeline:

1. Freeze a pre-trained encoder (Wav2Vec 2.0 or WavLM).
2. For each audio file:
   - Pass the waveform through the encoder.
   - Pool the hidden states into a fixed-length vector.
3. Train a small classifier head (e.g., logistic regression or a tiny MLP) on top of these embeddings to predict **0/1**.
4. Use **participant-grouped splits** so speakers in the test set are unseen during training.

![WavLM MLP Learning Curves](figures/wavlm_mlp_learning_curves.png)

*Figure 5. WavLM MLP Learning Curves show overfitting.*

![Wav2Vec 2.0 MLP Learning Curves](figures/wav2vec2_mlp_learning_curves.png)

*Figure 6. Wav2Vec 2.0 MLP Learning Curves also show overfitting but promising relatively canonical learning curves for next iteration.*

Empirical takeaway (pilot):

- For this phoneme-reversal classification task:
  - **Wav2Vec 2.0 features separated “correct” vs “incorrect”** responses more cleanly; the classifier generalized better to new participants.
  - **WavLM features tended to overfit speakers** more in this setup, with weaker generalization at this small data scale.
- In other words, **for this specific task and dataset size, “frozen Wav2Vec 2.0 + small head” beat “frozen WavLM + same head”**, despite WavLM’s stronger performance on broad benchmarks.
- Bonus: the homophone/orthographic ambiguity problem from earlier disappears!

Given the current tiny dataset and simple head, **Wav2Vec 2.0 was an easier representation space for the classifier** to carve out a good decision boundary.

---
## 4. Model Biases and Limitations

### Pre-training Data Biases

Both Wav2Vec2 and WavLM models inherit biases from their pre-training data:

**1. Speaker Demographics**
- **Training data**: Primarily English audiobooks (LibriSpeech, LibriLight)
- **Bias**: Better performance on:
  - Standard American English accents
  - Adult speakers (especially those who narrate audiobooks)
  - Clear, studio-quality audio
- **Implication**: May underperform on:
  - Non-native English speakers
  - Regional accents/dialects (Southern, AAVE, etc.)
  - Children's voices
  - Older adults with age-related voice changes

**2. Recording Conditions**
- **Training data**: Clean, professional audiobook recordings
- **Bias**: Models expect:
  - Low background noise
  - Consistent microphone quality
  - Quiet recording environments
- **Implication**: Performance degrades with:
  - Lab room noise (HVAC, equipment)
  - Variable microphone placement
  - Background conversations (RA instructions)

**3. Linguistic Content**
- **Training data**: Read speech from published books
- **Bias**: Optimized for:
  - Standard grammatical English
  - Natural prosody and intonation
  - Complete words and sentences
- **Implication**: Struggles with:
  - Nonwords (phoneme reversals)
  - Hesitations and disfluencies
  - Atypical prosody (speech disorders)
  - Partial utterances

---

## 5. Next Steps

Since this will be my MSDS Capstone project, I have a few plans to use what I've learned to scale up.

Near-term:

- **Scale up Phase 2**:
  - Add more participants data (add ~36 more participants for a total of ~50 participants x 22 audio files).
  - Re-run Wav2Vec 2.0 vs WavLM under the same conditions with better regularization and more robust splitting.
- Add an **explicit confidence score** from the classifier head (or from calibrated probabilities).
  - Use this to set a **threshold for human review**.
  - High-confidence predictions → auto-accepted.
  - Low-confidence predictions → flagged for RA check.

Longer-term:

- Turn this pipeline into a **general tool for psycholinguistics labs**:
  - UI for uploading audio and viewing scores.
  - Visualizations of confidence and error patterns across participants.
  - Hooks for exporting data directly into analysis pipelines (R, Python, etc.).
- Explore **multi-class labels** (e.g., partial credit, specific error types) rather than just binary correct/incorrect.
- Investigate **transfer to other speech tasks** in the lab (e.g., North American Adult Reading Test, blending nonwords, pseudoword repetition).

---

## 6. Intended Use & Licensing

### Intended Use
✅ Research use in psycholinguistics labs
✅ Phoneme reversal task scoring
✅ Assisting RAs with manual coding validation
✅ Flagging uncertain samples for human review

### Not Intended For
❌ Clinical diagnosis or assessment
❌ High-stakes decision making without human oversight
❌ Populations outside of neurotypical adult English speakers (without additional validation)
❌ Real-time assessment without confidence thresholding

### Licenses
- Code: MIT License
- Models: Apache 2.0 (following HuggingFace base models)
- Data: Restricted (human subjects research)

---

### Try It Out for Yourself!
Setup Instructions & Usage Guide
#### 1. Prerequisites

- Python 3.8+
- CUDA-capable GPU recommended (for faster training/inference)
- Google Colab account (if using provided notebooks)
- ~10GB disk space for models and data

#### 2. Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/phonocode.git
cd phonocode
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Required packages:
```
transformers>=4.30.0
datasets>=2.14.0
torch>=2.0.0
librosa>=0.10.0
soundfile>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
evaluate>=0.4.0
jiwer>=3.0.0
accelerate>=0.20.0
tensorboard>=2.13.0
```

#### 3. Project Structure

```
phonocode/
├── data/
│   ├── raw/                    # Original .wav files
│   └── processed/
│       └── phoneme_reversal/   # Organized by participant
├── code/
│   ├── run_wav2vec2_inference.py
│   ├── run_wavlm_inference.py
│   ├── train_wav2vec2_frozen_classifier.py
│   └── train_wavlm_frozen_classifier.py
├── results/                    # Output CSVs and evaluation metrics
├── scoring/                    # Ground truth labels
└── figures/                    # Visualizations for documentation
```

#### 4. Data

Organize audio files following this naming convention:
```
{participant_id}_{item_number}_{target_word}.wav
```

Example:
```
ReXa_008_01_an.wav
ReXa_008_02_put.wav
```

Create ground truth CSV with columns:
```
participant_id, word_1, word_2, ..., word_n
```
Where values are:
- `0` = incorrect
- `1` = correct 

---

## 6. References

- **Wav2Vec 2.0** – Baevski et al., *Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS 2020.
```bibtex
@inproceedings{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  booktitle={NeurIPS},
  year={2020}
}
```

- **WavLM** – Chen et al., *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*, IEEE/ACM TASLP 2022.
```bibtex
@article{chen2022wavlm,
  title={WavLM: Large-scale self-supervised pre-training for full stack speech processing},
  author={Chen, Sanyuan and Wang, Chengyi and Chen, Zhengyang and Wu, Yu and others},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2022}
}
```
