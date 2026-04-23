"""
PhonoCode Scoring Server
FastAPI backend for the RA scoring interface.

Directory layout expected relative to this file:

    app/
        server.py               ← this file
        inference/
            __init__.py
            phoneme_reversal.py
            naart.py
            blending_nonwords.py
        static/
            index.html

    models/
        phoneme_reversal_final_model.pt
        naart_final_model.pt
        blending_nonwords_lr_production_model.pkl

    reference_recordings/
        blending_nonwords/
            lander.wav
            jad.wav
            ...  (24 files, one per stimulus)

    data_processed/
        phoneme_reversal/
            ReXa_090/
                ReXa_090_01_an.wav
                ...
        naart/
            ReXa_090/
                ...
        blending_nonwords/
            ReXa_090/
                ...

    logs/                       ← created automatically on first submit

Start with:
    cd app
    uvicorn server:app --reload --port 8000
"""

import sys
import csv
import re
import threading
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from inference.phoneme_reversal import PhonemeReversalScorer, normalize_word as norm_pr
from inference.naart import NAARTScorer, normalize_word as norm_naart
from inference.blending_nonwords import BlendingNonwordsScorer, normalize_word as norm_bn

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

# Find project root (one level up from app/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
LOGS_ROOT = PROJECT_ROOT / "logs"
MODEL_ROOT = PROJECT_ROOT / "models"
REF_ROOT = PROJECT_ROOT / "data" / "references"

# Task configuration — no ground truth CSVs. Word lists live in each
# inference module. Model paths point to the final production checkpoints.
TASK_CONFIG = {
    "phoneme_reversal": {
        "data_subdir": "phoneme_reversal",
        "model_path":  Path("../models/phoneme-reversal_final_model.pt"),
        "scorer_class": "phoneme_reversal",
    },
    "naart": {
        "data_subdir": "naart",
        "model_path":  Path("../models/naart_final_model.pt"),
        "scorer_class": "naart",
    },
    "blending_nonwords": {
        "data_subdir": "blending_nonwords",
        "model_path":  Path("../models/blending-nonwords_final_model.pkl"),
        "ref_root":    Path("../data/reference_recordings/blending_nonwords"),
        "scorer_class": "blending_nonwords",
    },
}

CONFIDENCE_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Scorer state — one scorer instance kept alive between requests
# ---------------------------------------------------------------------------

class ScorerState:
    def __init__(self):
        self.lock         = threading.Lock()
        self.current_task: str | None  = None
        self.scorer                    = None

    def is_loaded(self, task: str) -> bool:
        return self.current_task == task and self.scorer is not None

    def load(self, task: str):
        config = TASK_CONFIG[task]

        model_path = config["model_path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Release previous scorer and flush GPU cache if switching tasks
        if self.scorer is not None:
            print(f"[scorer] Unloading '{self.current_task}'...")
            self.scorer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"[scorer] Loading task '{task}'...")

        cls = config["scorer_class"]
        if cls == "phoneme_reversal":
            self.scorer = PhonemeReversalScorer(model_path=model_path)
        elif cls == "naart":
            self.scorer = NAARTScorer(model_path=model_path)
        elif cls == "blending_nonwords":
            ref_root = config["ref_root"]
            if not ref_root.exists():
                raise FileNotFoundError(
                    f"Reference recordings directory not found: {ref_root}\n"
                    "Ensure all 24 blending nonwords reference .wav files are present."
                )
            self.scorer = BlendingNonwordsScorer(model_path=model_path, ref_root=ref_root)
        else:
            raise ValueError(f"Unknown scorer class: {cls}")

        self.current_task = task
        print(f"[scorer] Ready. Words: {self.scorer.target_words}")


scorer_state = ScorerState()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="PhonoCode Scoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SessionRequest(BaseModel):
    ra_initials: str
    task: str

class ScoreRequest(BaseModel):
    participant_id: str

class SubmitRequest(BaseModel):
    ra_initials:    str
    task:           str
    participant_id: str
    scores:         dict   # {word: 0|1}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/session")
def create_session(req: SessionRequest):
    """Validate RA + task, load the correct model if not already loaded."""
    ra   = req.ra_initials.strip().upper()
    task = req.task.strip()

    if not ra or not re.fullmatch(r"[A-Z]{1,5}", ra):
        raise HTTPException(400, "RA initials must be 1-5 letters.")
    if task not in TASK_CONFIG:
        raise HTTPException(400, f"Unknown task '{task}'. Valid: {list(TASK_CONFIG.keys())}")

    with scorer_state.lock:
        if not scorer_state.is_loaded(task):
            try:
                scorer_state.load(task)
            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
            except Exception as e:
                raise HTTPException(500, f"Model load failed: {e}")

    return {
        "status": "ok",
        "ra":     ra,
        "task":   task,
        "words":  scorer_state.scorer.target_words,
    }


@app.post("/api/score")
def score_participant(req: ScoreRequest):
    """Run inference on all audio files for a participant."""
    if scorer_state.scorer is None:
        raise HTTPException(400, "No model loaded. Call /api/session first.")

    task           = scorer_state.current_task
    task_config    = TASK_CONFIG[task]
    participant_folder = DATA_ROOT / task_config["data_subdir"] / req.participant_id

    if not participant_folder.exists():
        raise HTTPException(404, f"Participant folder not found: {participant_folder}")

    wav_files = sorted(participant_folder.glob("*.wav"))
    if not wav_files and task != "naart":
        raise HTTPException(404, f"No .wav files found in {participant_folder}")

    scorer       = scorer_state.scorer
    target_words = scorer.target_words
    results      = []
    errors       = []

    # Build a word → audio_path index from available files
    wav_by_word = {}
    for audio_path in wav_files:
        parts = audio_path.stem.split("_")
        if len(parts) < 3:
            errors.append(f"Skipped (bad filename): {audio_path.name}")
            continue
        raw_stem = parts[-1]

        # NAART needs filestem → canonical word resolution
        if task == "naart":
            word = norm_naart(scorer.filestem_to_word(raw_stem))
        elif task == "blending_nonwords":
            word = norm_bn(raw_stem)
        else:
            word = norm_pr(raw_stem)

        if word in target_words:
            wav_by_word[word] = audio_path

    # Score each expected word; handle missing files per-task
    for word in target_words:
        audio_path = wav_by_word.get(word)

        if audio_path is None:
            if task == "naart":
                # Missing NAART items are treated as incorrect (participant skipped)
                pred = scorer.predict_missing()
                results.append({
                    "word":         word,
                    "score":        pred["score"],
                    "confidence":   pred["confidence"],
                    "needs_review": pred["needs_review"],
                    "audio_file":   None,
                })
            else:
                errors.append(f"No audio file found for word '{word}'")
            continue

        try:
            pred = scorer.predict(audio_path, word=word)
            needs_review = pred["needs_review"] or (
                pred["confidence"] is not None and pred["confidence"] < CONFIDENCE_THRESHOLD
            )
            results.append({
                "word":         word,
                "score":        pred["score"],
                "confidence":   pred["confidence"],
                "needs_review": needs_review,
                "audio_file":   audio_path.name,
            })
        except ValueError as e:
            raise HTTPException(500, str(e))
        except Exception as e:
            errors.append(f"Error on {audio_path.name}: {e}")
            results.append({
                "word":         word,
                "score":        None,
                "confidence":   None,
                "needs_review": True,
                "audio_file":   audio_path.name,
            })

    return {
        "participant_id":       req.participant_id,
        "task":                 task,
        "results":              results,
        "errors":               errors,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.get("/api/audio/{participant_id}/{filename:path}")
def serve_audio(participant_id: str, filename: str):
    """Stream a single audio file to the browser."""
    if scorer_state.current_task is None:
        raise HTTPException(400, "No active session.")

    task_config = TASK_CONFIG[scorer_state.current_task]
    audio_path  = DATA_ROOT / task_config["data_subdir"] / participant_id / filename

    try:
        audio_path.resolve().relative_to(DATA_ROOT.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied.")

    if not audio_path.exists():
        raise HTTPException(404, f"Audio file not found: {filename}")

    return FileResponse(audio_path, media_type="audio/wav")


@app.post("/api/submit")
def submit_scores(req: SubmitRequest):
    """
    Write final RA-reviewed scores to two CSVs:
      logs/ra_{INITIALS}_{task}.csv   — timestamped, includes RA initials
      logs/task_{task}.csv            — word scores only (no confidence)
    """
    ra        = req.ra_initials.strip().upper()
    task      = req.task.strip()
    pid       = req.participant_id.strip()
    timestamp = datetime.now().isoformat(timespec="seconds")

    if task not in TASK_CONFIG:
        raise HTTPException(400, f"Unknown task: {task}")

    clean_scores = {k: v for k, v in req.scores.items() if not k.endswith("_confidence")}

    ra_row   = {"timestamp": timestamp, "ra": ra, "participant_id": pid, **clean_scores}
    task_row = {"participant_id": pid, **clean_scores}

    _append_csv(LOGS_ROOT / f"ra_{ra}_{task}.csv", ra_row)
    _append_csv(LOGS_ROOT / f"task_{task}.csv",    task_row)

    return {
        "status":   "submitted",
        "ra_log":   str(LOGS_ROOT / f"ra_{ra}_{task}.csv"),
        "task_log": str(LOGS_ROOT / f"task_{task}.csv"),
    }


def _append_csv(path: Path, row: dict):
    """Append a row to a CSV, writing the header if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
