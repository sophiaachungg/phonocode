# data/processed/naart

Preprocessed participant audio for the NAART (North American Adult Reading Test) task. Each participant has a subdirectory containing one `.wav` file per scored stimulus word.

## Directory structure

```
naart/
    ReXa_090/
        ReXa_090_01_gouge.wav
        ReXa_090_02_placebo.wav
        ...
    ReXa_091/
        ...
```

## Filename convention

```
{participant_id}_{index}_{word}.wav
```

| Segment | Description |
|---|---|
| `participant_id` | Lab participant ID, e.g. `ReXa_090`. Must match the subdirectory name exactly. |
| `index` | Two-digit stimulus index, e.g. `01`, `13`. |
| `word` | Target word in lowercase, e.g. `gouge`, `placebo`. |

### Special cases

Some stimulus words require filesystem-safe substitutions because the canonical spelling contains characters with accents that are invalid in filenames:

| Canonical word | Filename stem |
|---|---|
| `hors d'oeuvre` | `hors-doeuvre` |
| `detente` | `detente` |
| `facade` | `facade` |
| `ci-devant` | `ci-devant` |

The server resolves these automatically via `FILESTEM_TO_COLUMN` in `app/inference/naart.py`.

### Missing files

NAART participants sometimes skip items. A missing `.wav` file for a given stimulus is treated as an incorrect response (score = 0) by the scorer. Do not create placeholder files as the absence of the file is the signal.

## Preprocessing

Raw `base64` recordings from Gorilla are converted to `.wav` using:

```bash
python code/01-preprocess_audio/naart_from_csv.py
```

Requirements: 16,000 Hz sample rate, mono, non-zero frame count.