#!/usr/bin/env python3
"""
preprocess_audio_naart.py
--------------------------
Decodes base64-encoded audio responses from a Gorilla CSV export (ls_naart.csv)
directly to .wav format (16kHz, mono).

Only participants present in ground_truth_naart.csv are processed.

Output filenames follow the convention:
    {output_dir}/{participant_id}/{participant_id}_{item_no:02d}_{stimulus}.wav

The item number comes from exp_trial_index (zero-padded to two digits).
Item numbers map to stimuli in the canonical NAART order (item 1 = gouge, etc.).
If the stimulus in the CSV does not match the canonical item, the CSV stimulus
is used for the filename (trust the data) but the mismatch is flagged.

A validation CSV is written (and updated live) as each participant finishes:
    {output_dir}/validation_log_naart.csv

Flags written to the log
------------------------
  flag_wrong_file_count      : participant has != 61 .wav files
  flag_mismatched_stimulus   : ≥1 trial where CSV stimulus ≠ canonical stimulus
                               for that item number (details in mismatch_detail)
  flag_not_found             : participant ID in ground truth but absent from CSV
  flag_failed_trials         : ≥1 trial that FFmpeg could not convert
  flag_not_in_ground_truth   : participant in CSV but not in ground truth (skipped)

Usage
-----
    # Unit test on two participants
    python preprocess_audio_naart.py data_raw/ls_naart.csv \\
        --participants ReXa_034 ReXa_046 --output data_processed/naart_test

    # Full run against ground truth
    python preprocess_audio_naart.py data_raw/ls_naart.csv \\
        --ground-truth ground_truth_naart.csv

    # Force reprocess already-converted files
    python preprocess_audio_naart.py data_raw/ls_naart.csv --force

    # Verify FFmpeg installation
    python preprocess_audio_naart.py --check-ffmpeg

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
TARGET_SR             = 16000
DEFAULT_OUTPUT_DIR    = Path("../data/processed/naart")
DEFAULT_GROUND_TRUTH  = Path("../data/raw/naart.csv")
EXPECTED_N_TRIALS     = 61

# Canonical item-number → stimulus mapping (1-indexed, fixed NAART order)
CANONICAL = {
     1: "gouge",
     2: "placebo",
     3: "psalm",
     4: "depot",
     5: "equivocal",
     6: "bouquet",
     7: "demesne",
     8: "ennui",
     9: "indict",
    10: "caveat",
    11: "sidereal",
    12: "gauche",
    13: "hors d'oeuvre",
    14: "paradigm",
    15: "leviathan",
    16: "corps",
    17: "impugn",
    18: "recipe",
    19: "aisle",
    20: "sieve",
    21: "subtle",
    22: "epergne",
    23: "quadrupled",
    24: "gaoled",
    25: "superfluous",
    26: "talipes",
    27: "simile",
    28: "heir",
    29: "banal",
    30: "radix",
    31: "détente",
    32: "topiary",
    33: "assignate",
    34: "gauge",
    35: "reify",
    36: "vivace",
    37: "zealot",
    38: "epitome",
    39: "aeon",
    40: "cellist",
    41: "colonel",
    42: "beatify",
    43: "lingerie",
    44: "debt",
    45: "synecdoche",
    46: "façade",
    47: "hiatus",
    48: "drachm",
    49: "capon",
    50: "ci-devant",
    51: "syncope",
    52: "catacomb",
    53: "rarefy",
    54: "abstemious",
    55: "prelate",
    56: "procreate",
    57: "reign",
    58: "indices",
    59: "gist",
    60: "subpoena",
    61: "debris",
}

# Alternate spellings that appear in Gorilla CSV exports (e.g. encoding
# mojibake, missing hyphens/accents). All map to the canonical form used
# for matching. The filename stem is derived separately via STIMULUS_FILESTEM.
STIMULUS_ALIASES: dict[str, str] = {
    # détente variants
    "détente":   "détente",   # already canonical — kept for completeness
    "detente":   "détente",
    "dÃ©tente":  "détente",   # latin-1 bytes read as UTF-8 mojibake
    # façade variants
    "façade":    "façade",
    "facade":    "façade",
    "faÃ§ade":   "façade",    # mojibake
    # ci-devant variants
    "ci-devant": "ci-devant",
    "cidevant":  "ci-devant",
    "ci devant": "ci-devant",
    # hors d'oeuvre variants
    "hors d'oeuvre":  "hors d'oeuvre",
    "hors doeuvre":   "hors d'oeuvre",
    "hors d\u2019oeuvre": "hors d'oeuvre",  # curly apostrophe
}

# Safe filesystem stem for stimuli that contain characters invalid on some OSes.
# Any stimulus not listed here uses its canonical name as-is.
STIMULUS_FILESTEM: dict[str, str] = {
    "hors d'oeuvre": "hors-doeuvre",
    "détente":       "detente",
    "façade":        "facade",
    "ci-devant":     "ci-devant",   # hyphen is fine on all platforms
}


def normalize_stimulus(raw: str) -> str:
    """Map a raw CSV stimulus string to its canonical form.
    Falls back to the stripped raw value if no alias matches."""
    return STIMULUS_ALIASES.get(raw.strip(), raw.strip())


def filestem(canonical: str) -> str:
    """Return the safe filename stem for a canonical stimulus."""
    return STIMULUS_FILESTEM.get(canonical, canonical)


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
    "flag_not_in_ground_truth",
    "failed_trial_count",
    "mismatch_detail",
    "notes",
]
# ---------------------------------------------------------------------------


def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def validate_wav(wav_path: Path) -> bool:
    try:
        info = sf.info(str(wav_path))
        return info.samplerate == TARGET_SR and info.channels == 1 and info.frames > 0
    except Exception:
        return False


def decode_base64_to_wav(b64_string: str, output_path: Path, force: bool = False) -> str:
    """Decode a base64 WebM string and write a 16 kHz mono .wav via FFmpeg.
    Returns 'processed', 'skipped', or 'failed'."""
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
    df = pd.read_csv(csv_path, encoding="latin-1")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")
    df = df[df["response"].notna() & (df["response"].astype(str).str.strip() != "")]
    return df


def load_ground_truth_ids(gt_path: Path) -> set[str]:
    df = pd.read_csv(gt_path)
    if "participant_id" not in df.columns:
        raise ValueError(f"{gt_path.name} has no 'participant_id' column")
    return set(df["participant_id"].astype(str).str.strip().unique())


def resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("exp_trial_index")
    df = df.drop_duplicates(subset=["participant_id", "stimulus"], keep="last")
    return df


def append_log_row(log_path: Path, row: dict) -> None:
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in LOG_COLUMNS})


def process_csv(
    csv_path: Path,
    output_dir: Path,
    ground_truth_ids: set[str],
    participant_filter: list[str] | None,
    force: bool,
) -> None:
    log_path = output_dir / "validation_log_naart.csv"

    print(f"\nLoading {csv_path.name}...")
    try:
        df = load_csv(csv_path)
    except (ValueError, FileNotFoundError) as e:
        print(f"  Error: {e}")
        return

    all_csv_ids = set(df["participant_id"].astype(str).str.strip().unique())

    # Determine which participants to process
    if participant_filter:
        requested = [str(p).strip() for p in participant_filter]

        # Warn about IDs not in the CSV at all
        not_in_csv = [p for p in requested if p not in all_csv_ids]
        for pid in not_in_csv:
            print(f"  WARNING: '{pid}' not found in CSV — flagging.")
            append_log_row(log_path, {
                "participant_id":         pid,
                "wav_files_found":        0,
                "flag_wrong_file_count":  "YES",
                "flag_not_found":         "YES",
                "notes":                  "Participant ID not present in CSV",
            })

        # Warn about IDs not in the ground truth
        not_in_gt = [p for p in requested if p not in ground_truth_ids]
        for pid in not_in_gt:
            print(f"  WARNING: '{pid}' not in ground truth — will flag but still process.")

        target_ids = [p for p in requested if p in all_csv_ids]
    else:
        # Full run: only process participants present in the ground truth
        not_in_gt_csv = all_csv_ids - ground_truth_ids
        if not_in_gt_csv:
            print(f"  Skipping {len(not_in_gt_csv)} participant(s) not in ground truth.")
        target_ids = sorted(all_csv_ids & ground_truth_ids)

        # Flag ground-truth participants missing from the CSV
        missing_from_csv = ground_truth_ids - all_csv_ids
        for pid in sorted(missing_from_csv):
            print(f"  WARNING: ground truth participant '{pid}' absent from CSV — flagging.")
            append_log_row(log_path, {
                "participant_id":        pid,
                "wav_files_found":       0,
                "flag_wrong_file_count": "YES",
                "flag_not_found":        "YES",
                "notes":                 "Participant in ground truth but not found in CSV",
            })

    if not target_ids:
        print("  No participants to process.")
        return

    df = df[df["participant_id"].astype(str).str.strip().isin(target_ids)]
    df = resolve_duplicates(df)

    print(f"  {len(target_ids)} participant(s) to process")
    print(f"  Output directory: {output_dir}")
    print(f"  Validation log:   {log_path}")
    print(f"  Target format:    {TARGET_SR // 1000} kHz, mono .wav")
    print("-" * 60)

    total_processed = total_skipped = total_failed = 0

    for pid in sorted(target_ids):
        pid_df = df[df["participant_id"].astype(str).str.strip() == pid].copy()
        n = len(pid_df)

        not_in_gt_flag = pid not in ground_truth_ids
        gt_note = " [NOT IN GROUND TRUTH]" if not_in_gt_flag else ""
        print(f"\n  [{pid}]  {n} trial(s){gt_note}")

        mismatches = []
        failed_count = 0

        with tqdm(total=n, desc=f"  {pid}", unit="trial", leave=False) as pbar:
            for _, row in pid_df.iterrows():
                trial_idx    = int(row["exp_trial_index"])
                raw_stimulus = str(row["stimulus"]).strip()
                response     = str(row["response"]).strip()

                # Normalize known alternate spellings / encoding artefacts
                norm_stimulus      = normalize_stimulus(raw_stimulus)
                canonical_stimulus = CANONICAL.get(trial_idx)

                # Flag only if the normalized form still differs from canonical
                if canonical_stimulus and norm_stimulus != canonical_stimulus:
                    mismatches.append(
                        f"item {trial_idx:02d}: expected '{canonical_stimulus}', "
                        f"got '{raw_stimulus}' → normalized '{norm_stimulus}' "
                        f"(used '{norm_stimulus}')"
                    )

                # Use canonical name for the stem if available, else normalized
                stem        = filestem(canonical_stimulus or norm_stimulus)
                trial_str   = f"{trial_idx:02d}"
                output_path = output_dir / pid / f"{pid}_{trial_str}_{stem}.wav"

                result = decode_base64_to_wav(response, output_path, force=force)

                if result == "processed":
                    total_processed += 1
                elif result == "skipped":
                    total_skipped += 1
                else:
                    total_failed += 1
                    failed_count += 1

                pbar.update(1)

        pid_dir   = output_dir / pid
        wav_count = len(list(pid_dir.glob("*.wav"))) if pid_dir.exists() else 0
        wrong_n   = wav_count != EXPECTED_N_TRIALS

        status_parts = []
        if wrong_n:
            status_parts.append(f"WRONG FILE COUNT ({wav_count}/{EXPECTED_N_TRIALS})")
        if mismatches:
            status_parts.append(f"STIMULUS MISMATCH ({len(mismatches)} item(s))")
        if failed_count:
            status_parts.append(f"FAILED TRIALS ({failed_count})")
        if not_in_gt_flag:
            status_parts.append("NOT IN GROUND TRUTH")

        if status_parts:
            print(f"  ⚠  {pid}: {', '.join(status_parts)}")
        else:
            print(f"  ✓  {pid}: {wav_count}/{EXPECTED_N_TRIALS} files — OK")

        append_log_row(log_path, {
            "participant_id":            pid,
            "wav_files_found":           wav_count,
            "flag_wrong_file_count":     "YES" if wrong_n else "",
            "flag_mismatched_stimulus":  "YES" if mismatches else "",
            "flag_not_found":            "",
            "flag_failed_trials":        "YES" if failed_count > 0 else "",
            "flag_not_in_ground_truth":  "YES" if not_in_gt_flag else "",
            "failed_trial_count":        failed_count if failed_count > 0 else "",
            "mismatch_detail":           " | ".join(mismatches) if mismatches else "",
            "notes":                     "",
        })

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
        description="Decode NAART Gorilla base64 audio → 16 kHz mono .wav with validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unit test — two participants, custom output dir
  python preprocess_audio_naart.py data_raw/ls_naart.csv \\
      --participants ReXa_034 ReXa_046 --output data_processed/naart_test

  # Full run (processes only participants in ground truth)
  python preprocess_audio_naart.py data_raw/ls_naart.csv

  # Full run with custom ground truth path
  python preprocess_audio_naart.py data_raw/ls_naart.csv \\
      --ground-truth path/to/ground_truth_naart.csv

  # Force reprocess
  python preprocess_audio_naart.py data_raw/ls_naart.csv --force

  # Check FFmpeg
  python preprocess_audio_naart.py --check-ffmpeg
        """,
    )

    parser.add_argument("input_csv", nargs="?", help="Path to ls_naart.csv Gorilla export")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("-g", "--ground-truth", type=str, default=None,
                        help=f"Path to ground_truth_naart.csv (default: {DEFAULT_GROUND_TRUTH})")
    parser.add_argument("-p", "--participants", nargs="+", default=None,
                        metavar="PARTICIPANT_ID",
                        help="Only process these participant IDs (unit test mode)")
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

    if not args.input_csv:
        parser.error("input_csv is required.")

    if not check_ffmpeg():
        print("Error: FFmpeg is not installed or not in PATH")
        print("  macOS:  brew install ffmpeg")
        print("  Linux:  sudo apt-get install ffmpeg")
        sys.exit(1)

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    gt_path = Path(args.ground_truth) if args.ground_truth else DEFAULT_GROUND_TRUTH
    if not gt_path.exists():
        print(f"Error: ground truth file not found: {gt_path}")
        print("  Pass --ground-truth <path> to specify its location.")
        sys.exit(1)

    print(f"Loading ground truth from {gt_path.name}...")
    try:
        ground_truth_ids = load_ground_truth_ids(gt_path)
    except ValueError as e:
        print(f"  Error: {e}")
        sys.exit(1)
    print(f"  {len(ground_truth_ids)} participant(s) in ground truth")

    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    process_csv(
        csv_path=csv_path,
        output_dir=output_dir,
        ground_truth_ids=ground_truth_ids,
        participant_filter=args.participants,
        force=args.force,
    )


if __name__ == "__main__":
    main()
