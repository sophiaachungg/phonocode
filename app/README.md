# PhonoCode Scoring App

Local web interface for RAs to review and submit phonological task scores.

## Stack

- **Backend**: FastAPI (Python) — `server.py`
- **Frontend**: Single-file HTML/CSS/JS — `static/index.html`
- **No build step, no npm, no database.**

---

## Setup

### 1. Install dependencies

```bash
pip install fastapi uvicorn[standard] torch torchaudio transformers librosa pandas numpy python-multipart
```

### 2. Place model files

Each task needs its final production model in the `model/` directory:

| Task | File |
|------|------|
| Phoneme reversal | `model/phoneme_reversal_final_model.pt` |
| NAART | `model/naart_final_model.pt` |
| Blending nonwords | `model/blending_nonwords_lr_production_model.pkl` |

These are produced by `phoneme_reversal_train_final.py`, `naart_train_final.py`, and `blending_nonwords_train_final.py` respectively.

### 3. Place blending nonwords reference recordings

The blending nonwords scorer compares each participant response against a canonical reference recording. Place one `.wav` file per stimulus (24 total) in:

```
reference_recordings/blending_nonwords/
    lander.wav
    jad.wav
    mog.wav
    ... (all 24 stimuli)
```

These are the same reference files used during training. Phoneme reversal and NAART do not require reference recordings.

### 4. Start the server

```bash
cd app
uvicorn server:app --reload --port 8000
```

Open **http://localhost:8000** in any browser.

---

## Folder structure

```
app/                                    ← run uvicorn from here
    server.py
    inference_phoneme_reversal.py
    inference_naart.py
    inference_blending_nonwords.py
    static/
        index.html

model/
    phoneme_reversal_final_model.pt
    naart_final_model.pt
    blending_nonwords_lr_production_model.pkl

reference_recordings/
    blending_nonwords/
        lander.wav
        jad.wav
        ...  (24 files)

data_processed/
    phoneme_reversal/
        ReXa_090/
            ReXa_090_01_an.wav
            ...
    naart/
        ReXa_090/
            ReXa_090_01_psalm.wav
            ...
    blending_nonwords/
        ReXa_090/
            ReXa_090_01_lander.wav
            ...

logs/                                   ← created automatically on first submit
    ra_SC_phoneme_reversal.csv
    task_phoneme_reversal.csv
```

> **The `scoring/` directory (ground truth CSVs) is not needed by the app.**
> Word lists are hardcoded in each inference module. Ground truth CSVs contain
> participant data and belong in the research pipeline, not the scoring UI.

---

## Audio filename convention

Files must follow: `{participant_id}_{index}_{word}.wav`

Examples: `ReXa_090_01_an.wav`, `ReXa_090_12_psalm.wav`, `ReXa_090_03_lander.wav`

The last `_`-delimited segment is used as the word key.

**NAART note:** Some NAART stimuli use filesystem-safe stems that differ from the canonical word name (e.g. `hors-doeuvre` → `hors d'oeuvre`). The NAART inference module handles this mapping automatically via `FILESTEM_TO_COLUMN`.

---

## Logs

Two CSVs are written on each submission:

| File | Contents |
|------|----------|
| `logs/ra_{INITIALS}_{task}.csv` | RA initials, timestamp, participant ID, all word scores |
| `logs/task_{task}.csv` | participant ID + word scores only |

---

## Notes

- **Model loading**: The encoder loads once per task selection and stays in memory until the task changes. Switching tasks via "Switch session" reloads the model. WavLM-Large (blending nonwords) is significantly larger than Wav2Vec2-base; allow extra time on first load.
- **Blending nonwords reference pre-encoding**: On session start for blending nonwords, all 24 reference recordings are encoded once and cached in memory. This takes a few seconds.
- **Missing NAART files**: If a participant's audio folder is missing a file for a given stimulus, the NAART scorer automatically returns score=0 (incorrect). This matches the training behaviour, where missing files were silence-injected and scored 0.
- **Confidence threshold**: Currently `0.75` (`CONFIDENCE_THRESHOLD` in `server.py`). Trials below this are flagged for RA review.
- **Embedding dimension mismatch**: The server raises HTTP 500 if the extracted embedding dimension doesn't match the checkpoint. This means `MODEL_ID` in the inference module does not match the model the checkpoint was trained with. Do not change `MODEL_ID` without retraining.
- **VPN / shared server**: When deploying on the lab server, RAs connect over VPN and open `http://<server-ip>:8000`. Audio is streamed via `/api/audio/` so the browser never needs direct filesystem access.
