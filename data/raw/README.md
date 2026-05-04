# data/raw

Raw CSV exports from [Gorilla Experiment Builder](https://gorilla.sc/), one file per task per experiment run. These CSVs are the direct input to the preprocessing scripts in `code/01-preprocess_audio/`.

## Directory structure

```
raw/
    phoneme_reversal_v1.csv
    naart_v1.csv
    blending_nonwords_v1.csv
    ...
```

Naming is flexible — the preprocessing scripts take the CSV path as an argument. What matters is the schema, described below.

---

## CSV schema

Gorilla exports one row per trial. The full set of columns exported is:

| Column | Description |
|---|---|
| `task_name` | Gorilla task name string |
| `exp_version` | Experiment version number |
| `task_version` | Task version number |
| `local_time` | Timestamp of the trial in local time |
| `time_elapsed` | Time elapsed since experiment start (ms) |
| `public_id` | Gorilla-assigned anonymous public ID |
| `participant_id` | **Lab participant ID, e.g. `ReXa_090`.** This becomes the subdirectory name under `data/processed/{task}/`. |
| `status` | Trial status string (e.g. `complete`) |
| `exp_trial_index` | Absolute trial index within the experiment |
| `block` | Block name or number |
| `block_trial_index` | Trial index within the current block |
| `practice` | Whether this was a practice trial (`1` / `0` or `true` / `false`) |
| `stimulus` | **Target word or nonword for this trial**, e.g. `psalm`, `lander`. Used to derive the output filename. |
| `response` | **Participant audio response, base64-encoded `.webm`.** Decoded and converted to `.wav` during preprocessing. |

---

## Key columns for preprocessing

Three columns drive everything in the preprocessing scripts:

### `participant_id`
Becomes the subdirectory name under `data/processed/{task}/`. Must be consistent across all rows for the same participant. The preprocessing scripts group rows by this value.

### `stimulus`
Used to derive the `{word}` segment of the output filename. The scripts apply any necessary normalisation (lowercase, accent stripping, filesystem-safe substitutions) before writing. Practice trials are identified here and excluded — see the task-specific READMEs in `data/processed/` for which stimuli are excluded per task.

### `response`
A base64-encoded `.webm` audio file. The preprocessing pipeline:
1. Decodes the base64 string
2. Writes the raw bytes to a temporary `.webm` file
3. Converts to `.wav` via FFmpeg (16,000 Hz, mono)
4. Validates the output (sample rate, channel count, non-zero frame count)
5. Saves to `data/processed/{task}/{participant_id}/{participant_id}_{index}_{stimulus}.wav`

Rows where `response` is empty, null, or not valid base64 are skipped with a warning.

---

## Preprocessing

Follow the docstring instructions in each of the tasks' .py files in `code/01-preprocess_audio'.

FFmpeg must be installed and on `PATH`. See the [Installation](../../README.md#installation) section of the main README.

---

## Notes

- **Do not commit raw CSVs to the repository.** They contain participant audio data (base64-encoded) and fall under the lab's IRB data governance agreement. This directory is listed in `.gitignore`.
- `public_id` and `participant_id` are different. `public_id` is Gorilla's internal anonymous identifier. `participant_id` is the lab's own ID scheme and is what the rest of the pipeline uses.
- If a participant completed the task more than once (e.g. a session restart), there will be duplicate `participant_id` + `stimulus` combinations in the CSV. The preprocessing scripts take the last valid response per participant × stimulus pair by default.