# PhonoCode: Transformer-Based Scoring of Phonological Tasks

This repo contains code for experimenting with transformer-based speech models (e.g., Whisper, Wav2Vec2, WavLM) to *automatically score* phonological task responses.  
Each audio clip is classified as **correct (1)** or **incorrect (0)**.

The main goals:

- Compare **base vs. fine-tuned** ASR transformers (Whisper, Wav2Vec2, etc.) on the same dataset.
- Evaluate how well models can **generalize across participants** (participant-grouped splits).
- Provide clear, reproducible experiments for **binary classification on speech** in a psycholinguistics context.
- (Eventually) Save psycholinguistics labs time and money by significantly reducing manual coding efforts.
