# data/processed/phoneme_reversal

Preprocessed participant audio for the phoneme reversal task. Each participant has a subdirectory containing one `.wav` file per scored stimulus.

## Directory structure

```
phoneme_reversal/
    ReXa_090/
        ReXa_090_01_an.wav
        ReXa_090_02_do.wav
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
| `word` | Target word in lowercase, e.g. `an`, `do`. |


## Preprocessing

Raw `base64` recordings from Gorilla are converted to `.wav` using:

```bash
python code/01-preprocess_audio/phoneme-reversal_from_csv.py
```

Requirements: 16,000 Hz sample rate, mono, non-zero frame count.