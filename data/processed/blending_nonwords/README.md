# data/processed/blending_nonwords

Preprocessed participant audio for the blending nonwords task. Each participant has a subdirectory containing one `.wav` file per scored stimulus.

## Directory structure

```
blending_nonwords/
    ReXa_090/
        ReXa_090_01_lander.wav
        ReXa_090_02_jad.wav
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
| `word` | Target nonword in lowercase, e.g. `lander`, `jad`. |


### Valid stimulus names

```
lander  jad     mog      het       ko        nimby
teb     shawbo  ghite    zigopple  shib      motabe
heckobi tastains nysheeboki jop    nass      vope
suhnypogh nemowk shyvitch  basp   tigu      koomayg
```

### Duration gate

Participant responses longer than **8,000 ms** are flagged as recording artefacts and excluded from scoring. The scorer returns `needs_review: true` for these trials rather than a score.

## Reference recordings

The blending nonwords scorer compares each participant response against a canonical reference recording. These are stored separately at:

```
data/reference_recordings/blending_nonwords/{word}.wav
```

One reference file per stimulus (24 total) is required. These are not participant recordings and should not be placed in this directory.

## Preprocessing

Raw `base64` recordings from Gorilla are converted to `.wav` using:

```bash
python code/01-preprocess_audio/blending-nonwords_from_csv.py
```

Requirements: 16,000 Hz sample rate, mono, non-zero frame count.