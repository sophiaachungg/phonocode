#!/usr/bin/env python3
"""
preprocess_audio_from_csv.py
-----------------------------
Decodes base64-encoded audio responses from a Gorilla CSV export directly
to .wav format (16kHz, mono), bypassing the R base64 decode + .webm
intermediate step entirely.

Output filenames follow the convention:
    {participant_id}/{participant_id}_{item_no:02d}_{stimulus}.wav

The item number comes from exp_trial_index (zero-padded to two digits).
If the stimulus in the CSV does not match the canonical item list, the
CSV stimulus is used for the filename (always trust the data) but the
mismatch is flagged in the validation log.

A validation CSV is written (and updated live) as each participant finishes:
    {output_dir}/validation_log.csv

Flags written to the log
------------------------
  flag_wrong_file_count      : participant has != 22 .wav files
  flag_mismatched_stimulus   : ≥1 trial where CSV stimulus ≠ canonical stimulus
                               for that item number (details in mismatch_detail)
  flag_not_found             : participant ID requested but absent from CSV
  flag_failed_trials         : ≥1 trial that FFmpeg could not convert

Usage
-----
    # Single CSV, default output directory
    python preprocess_audio_from_csv.py data_raw/phoneme_reversal.csv

    # Restrict to specific participants (unit test)
    python preprocess_audio_from_csv.py data_raw/phoneme_reversal.csv \\
        --participants ReXa_311 ReXa_221 --output data_processed/test_pilot

    # Force reprocess already-converted files
    python preprocess_audio_from_csv.py data_raw/phoneme_reversal.csv --force

    # Verify FFmpeg installation
    python preprocess_audio_from_csv.py --check-ffmpeg

Dependencies
------------
    pip install pandas tqdm soundfile

    FFmpeg must be installed and on PATH:
        macOS:          brew install ffmpeg
        Ubuntu/Debian:  sudo apt-get install ffmpeg
"""

import argparse
import base64
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_SR          = 16000
DEFAULT_OUTPUT_DIR = Path("../data_processed/phoneme_reversal")
EXPECTED_N_TRIALS  = 22

# Canonical item-number → stimulus mapping (1-indexed)
CANONICAL = {
     1: "an",
     2: "do",
     3: "pet",
     4: "sit",
     5: "dime",
     6: "boots",
     7: "see",
     8: "midnight",
     9: "pile",
    10: "seven",
    11: "speed",
    12: "system",
    13: "at",
    14: "baseball",
    15: "sun",
    16: "state",
    17: "to",
    18: "spoon",
    19: "cheek",
    20: "in",
    21: "be",
    22: "sometimes",
}

# Expected columns in the Gorilla CSV export
REQUIRED_COLUMNS = {"participant_id", "exp_trial_index", "stimulus", "response"}

# Validation log column order
LOG_COLUMNS = [
    "participant_id",
    "wav_files_found",
    "flag_wrong_file_count",
    "flag_mismatched_stimulus",
    "flag_not_found",
    "flag_failed_trials",
    "failed_trial_count",
    "mismatch_detail",
    "notes",
]
# ---------------------------------------------------------------------------


def check_ffmpeg() -> bool:
    """Return True if FFmpeg is reachable on PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def validate_wav(wav_path: Path) -> bool:
    """Return True if the .wav is valid: 16 kHz, mono, non-empty."""
    try:
        info = sf.info(str(wav_path))
        return (
            info.samplerate == TARGET_SR
            and info.channels == 1
            and info.frames > 0
        )
    except Exception:
        return False


def decode_base64_to_wav(b64_string: str, output_path: Path, force: bool = False) -> str:
    """
    Decode a base64 WebM string and write a 16 kHz mono .wav via FFmpeg.
    Returns 'processed', 'skipped', or 'failed'.
    """
    if output_path.exists() and not force:
        if validate_wav(output_path):
            return "skipped"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        webm_bytes = base64.b64decode(b64_string)
    except Exception as e:
        print(f"\n  Base64 decode error for {output_path.name}: {e}")
        return "failed"

    command = [
        "ffmpeg", "-y",
        "-i", "pipe:0",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-loglevel", "error",
        str(output_path),
    ]

    try:
        subprocess.run(command, input=webm_bytes, check=True, capture_output=True)
        return "processed"
    except subprocess.CalledProcessError as e:
        print(f"\n  FFmpeg failed for {output_path.name}")
        print(f"  Error: {e.stderr.decode(errors='replace').strip()}")
        return "failed"
    except Exception as e:
        print(f"\n  Unexpected error for {output_path.name}: {e}")
        return "failed"


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate a Gorilla CSV export."""
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")
    df = df[df["response"].notna() & (df["response"].str.strip() != "")]
    return df


def resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the last attempt per participant+stimulus (handles Gorilla reloads)."""
    df = df.sort_values("exp_trial_index")
    df = df.drop_duplicates(subset=["participant_id", "stimulus"], keep="last")
    return df


def append_log_row(log_path: Path, row: dict) -> None:
    """Append one participant row to the validation CSV (creates file + header if needed)."""
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in LOG_COLUMNS})


def process_participant(
    pid: str,
    pid_df: pd.DataFrame,
    output_dir: Path,
    force: bool,
) -> dict:
    """
    Process all trials for one participant and return a validation-log row.
    """
    mismatches = []
    failed_count = 0

    for _, row in pid_df.iterrows():
        trial_idx = int(row["exp_trial_index"])
        csv_stimulus = str(row["stimulus"]).strip()
        response = str(row["response"]).strip()

        # Determine canonical stimulus for this item number
        canonical_stimulus = CANONICAL.get(trial_idx)

        # Always use the CSV stimulus for the filename — trust the data.
        # Flag if it differs from the canonical expectation.
        if canonical_stimulus and csv_stimulus != canonical_stimulus:
            mismatches.append(
                f"item {trial_idx:02d}: expected '{canonical_stimulus}', "
                f"got '{csv_stimulus}' (used '{csv_stimulus}')"
            )

        trial_str = f"{trial_idx:02d}"
        output_path = output_dir / pid / f"{pid}_{trial_str}_{csv_stimulus}.wav"

        result = decode_base64_to_wav(response, output_path, force=force)
        if result == "failed":
            failed_count += 1

    # Count .wav files actually on disk for this participant
    pid_dir = output_dir / pid
    wav_count = len(list(pid_dir.glob("*.wav"))) if pid_dir.exists() else 0

    log_row = {
        "participant_id":          pid,
        "wav_files_found":         wav_count,
        "flag_wrong_file_count":   "YES" if wav_count != EXPECTED_N_TRIALS else "",
        "flag_mismatched_stimulus": "YES" if mismatches else "",
        "flag_not_found":          "",
        "flag_failed_trials":      "YES" if failed_count > 0 else "",
        "failed_trial_count":      failed_count if failed_count > 0 else "",
        "mismatch_detail":         " | ".join(mismatches) if mismatches else "",
        "notes":                   "",
    }
    return log_row


def process_csv(
    csv_path: Path,
    output_dir: Path,
    participant_filter: list[str] | None,
    force: bool,
) -> None:
    """Process all audio responses in a single Gorilla CSV export."""
    log_path = output_dir / "validation_log.csv"

    print(f"\nLoading {csv_path.name}...")
    try:
        df = load_csv(csv_path)
    except (ValueError, FileNotFoundError) as e:
        print(f"  Error: {e}")
        return

    # Handle participants requested but not found in the CSV
    if participant_filter:
        found_in_csv = set(df["participant_id"].unique())
        not_found = [p for p in participant_filter if p not in found_in_csv]
        for pid in not_found:
            print(f"  WARNING: participant '{pid}' not found in CSV — flagging.")
            log_row = {
                "participant_id": pid,
                "wav_files_found": 0,
                "flag_wrong_file_count": "YES",
                "flag_mismatched_stimulus": "",
                "flag_not_found": "YES",
                "flag_failed_trials": "",
                "failed_trial_count": "",
                "mismatch_detail": "",
                "notes": "Participant ID not present in CSV",
            }
            append_log_row(log_path, log_row)

        df = df[df["participant_id"].isin(participant_filter)]
        if df.empty:
            print("  No matching participants to process.")
            return

    df = resolve_duplicates(df)
    all_pids = sorted(df["participant_id"].unique())

    print(f"  {len(all_pids)} participant(s) to process")
    print(f"  Output directory:  {output_dir}")
    print(f"  Validation log:    {log_path}")
    print(f"  Target format:     {TARGET_SR // 1000} kHz, mono .wav")
    print("-" * 60)

    total_processed = total_skipped = total_failed = 0

    for pid in all_pids:
        pid_df = df[df["participant_id"] == pid].copy()
        n = len(pid_df)

        print(f"\n  [{pid}]  {n} trial(s)")
        with tqdm(total=n, desc=f"  {pid}", unit="trial", leave=False) as pbar:

            mismatches = []
            failed_count = 0

            for _, row in pid_df.iterrows():
                trial_idx    = int(row["exp_trial_index"])
                csv_stimulus = str(row["stimulus"]).strip()
                response     = str(row["response"]).strip()

                canonical_stimulus = CANONICAL.get(trial_idx)
                if canonical_stimulus and csv_stimulus != canonical_stimulus:
                    mismatches.append(
                        f"item {trial_idx:02d}: expected '{canonical_stimulus}', "
                        f"got '{csv_stimulus}' (used '{csv_stimulus}')"
                    )

                trial_str   = f"{trial_idx:02d}"
                output_path = output_dir / pid / f"{pid}_{trial_str}_{csv_stimulus}.wav"

                result = decode_base64_to_wav(response, output_path, force=force)

                if result == "processed":
                    total_processed += 1
                elif result == "skipped":
                    total_skipped += 1
                else:
                    total_failed += 1
                    failed_count += 1

                pbar.update(1)

        # --- per-participant validation ---
        pid_dir   = output_dir / pid
        wav_count = len(list(pid_dir.glob("*.wav"))) if pid_dir.exists() else 0
        wrong_n   = wav_count != EXPECTED_N_TRIALS

        status_parts = []
        if wrong_n:
            status_parts.append(f"WRONG FILE COUNT ({wav_count}/22)")
        if mismatches:
            status_parts.append(f"STIMULUS MISMATCH ({len(mismatches)} item(s))")
        if failed_count:
            status_parts.append(f"FAILED TRIALS ({failed_count})")

        if status_parts:
            print(f"  ⚠  {pid}: {', '.join(status_parts)}")
        else:
            print(f"  ✓  {pid}: {wav_count}/22 files — OK")

        log_row = {
            "participant_id":           pid,
            "wav_files_found":          wav_count,
            "flag_wrong_file_count":    "YES" if wrong_n else "",
            "flag_mismatched_stimulus": "YES" if mismatches else "",
            "flag_not_found":           "",
            "flag_failed_trials":       "YES" if failed_count > 0 else "",
            "failed_trial_count":       failed_count if failed_count > 0 else "",
            "mismatch_detail":          " | ".join(mismatches) if mismatches else "",
            "notes":                    "",
        }
        append_log_row(log_path, log_row)

    # --- overall summary ---
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Processed:  {total_processed} file(s)")
    print(f"Skipped:    {total_skipped} file(s) (already valid)")
    print(f"Failed:     {total_failed} file(s)")
    print(f"\nValidation log: {log_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode Gorilla base64 audio → 16 kHz mono .wav with validation logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unit test — two participants, custom output dir
  python preprocess_audio_from_csv.py data_raw/phoneme_reversal.csv \\
      --participants ReXa_080 ReXa_235 --output data_processed/test_pilot

  # Full run
  python preprocess_audio_from_csv.py data_raw/phoneme_reversal.csv

  # Force reprocess
  python preprocess_audio_from_csv.py data_raw/phoneme_reversal.csv --force

  # Check FFmpeg
  python preprocess_audio_from_csv.py --check-ffmpeg
        """,
    )

    parser.add_argument("input_csvs", nargs="*", help="Path(s) to Gorilla CSV export file(s)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("-p", "--participants", nargs="+", default=None,
                        metavar="PARTICIPANT_ID",
                        help="Only process these participant IDs")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Reprocess even if a valid .wav already exists")
    parser.add_argument("--check-ffmpeg", action="store_true",
                        help="Verify FFmpeg is installed and exit")

    args = parser.parse_args()

    if args.check_ffmpeg:
        if check_ffmpeg():
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
            print("✓ FFmpeg is installed")
            print(result.stdout.split("\n")[0])
        else:
            print("✗ FFmpeg is not installed or not in PATH")
            sys.exit(1)
        sys.exit(0)

    if not args.input_csvs:
        parser.error("At least one input CSV is required.")

    if not check_ffmpeg():
        print("Error: FFmpeg is not installed or not in PATH")
        print("  macOS:  brew install ffmpeg")
        print("  Linux:  sudo apt-get install ffmpeg")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path_str in args.input_csvs:
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            print(f"Error: CSV not found: {csv_path}")
            continue
        process_csv(
            csv_path=csv_path,
            output_dir=output_dir,
            participant_filter=args.participants,
            force=args.force,
        )


if __name__ == "__main__":
    main()
