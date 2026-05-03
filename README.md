# Mandarin Tone Recognition — THCHS-30 Pipeline

End-to-end system for Mandarin tone classification: from raw THCHS-30 audio to a REST API that gives per-syllable pronunciation feedback.

The pipeline trains on isolated syllable clips (MFA-aligned, ~28k samples) and serves predictions via FastAPI + Whisper. Designed as the acoustic backend for a Mandarin study app.

---

## Project structure

```
tone-recognition-dataset/
├── scripts/                          # Offline training pipeline
│   ├── utils.py                      # Shared I/O helpers
│   ├── generate_tone_metadata.py     # Phase 1 — label generation
│   ├── prepare_mfa_corpus.py         # Phase 2a — MFA corpus prep
│   ├── parse_textgrids.py            # Phase 2b — TextGrid → timestamps
│   ├── slice_audio.py                # Phase 3 — syllable slicing
│   ├── extract_f0_features.py        # Phase 4 — F0 feature extraction
│   ├── train_classifier.py           # Phase 5 — SVM/MLP training
│   ├── infer.py                      # Inference core (used by API)
│   ├── record_and_test.py            # Mic recording + API test
│   └── test_api.py                   # API smoke-test client
├── api/
│   └── feedback.py                   # FastAPI /feedback endpoint
├── models/                           # Saved artifacts (committed except svm.joblib)
│   ├── mlp.joblib
│   ├── scaler.joblib
│   └── speaker_stats.json
├── output/                           # Generated data (gitignored)
│   └── <split>/
│       ├── metadata.csv              # Phase 1 output
│       ├── f0_dataset.csv            # Phase 4 output
│       └── clips/tone_{1..5}/        # Phase 3 output
├── mfa_corpus/                       # MFA input corpus (gitignored)
├── Dockerfile
├── requirements.txt                  # Training pipeline deps
└── requirements-api.txt              # API deps
```

---

## Results

| Model                    | Accuracy | Macro F1 |
|--------------------------|----------|----------|
| SVM baseline (dev 80/20) | 63.4%    | 0.489    |
| MLP baseline (dev 80/20) | 63.1%    | 0.489    |
| SVM improved (dev 80/20) | 62.9%    | 0.570    |
| MLP improved (dev 80/20) | 63.7%    | 0.577    |
| SVM train→dev            | 64.8%    | —        |
| **MLP train→dev**        | **67.4%**| —        |
| SVM train→test           | 58.9%    | —        |
| MLP train→test           | 63.2%    | —        |
| API (continuous speech)  | ~48%     | —        |

The ~4pp drop on the test set comes from 10 speakers (D-prefix) absent from training — speaker normalization falls back to the global mean (240 Hz).

The ~19pp gap between isolated clips and continuous speech comes from Whisper word-level alignment + uniform syllable distribution (see [Known limitations](#known-limitations)).

**Feature engineering applied:**
1. Speaker normalisation — Hz → semitones relative to per-speaker mean F0
2. Delta F0 — first differences appended (10 pts → 19 features)
3. StandardScaler — zero mean, unit variance (fit on train only)
4. Class-balanced weights — compensates Tone 5 under-representation (~5× fewer samples)

---

## Installation

### Training pipeline

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

### API and inference

```bash
pip install -r requirements-api.txt
```

### Montreal Forced Aligner (offline alignment only)

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner
conda activate aligner
mfa model download acoustic mandarin_mfa
mfa model download dictionary mandarin_mfa
```

---

## Training pipeline

### Phase 1 — Label generation

Reads `.trn` files (hanzi + pinyin with tone numbers), produces one record per syllable.

```bash
python scripts/generate_tone_metadata.py \
    --data_dir  "path/to/data_thchs30/data" \
    --split_dir "path/to/data_thchs30/dev" \
    --output_dir output/dev/
```

Output: `output/dev/metadata.csv` — columns: `sample_id, hanzi, pinyin, tone, original_audio_path, syllable_index`

### Phase 2 — Forced alignment (MFA)

**Prepare corpus** (uses hanzi characters as `.lab` content — the `mandarin_mfa` dictionary maps hanzi → phonemes, not pinyin):

```bash
python scripts/prepare_mfa_corpus.py \
    --data_dir   "path/to/data_thchs30/data" \
    --split_dir  "path/to/data_thchs30/dev" \
    --corpus_dir mfa_corpus/dev/
```

**Run MFA:**

```bash
conda activate aligner
mfa validate mfa_corpus/dev/ mandarin_mfa mandarin_mfa --clean
mfa align   mfa_corpus/dev/ mandarin_mfa mandarin_mfa output/dev/textgrids/
```

**Parse TextGrids** (joins timestamps back to metadata):

```bash
python scripts/parse_textgrids.py \
    --textgrid_dir output/dev/textgrids/ \
    --metadata_csv output/dev/metadata.csv \
    --output_dir   output/dev/aligned/
```

### Phase 3 — Audio slicing

```bash
python scripts/slice_audio.py \
    --aligned_metadata output/dev/aligned/metadata_aligned.csv \
    --clips_dir        output/dev/clips/
```

Clips shorter than 50ms are discarded. Output is organized into `clips/tone_{1..5}/`.

### Phase 4 — F0 feature extraction

```bash
python scripts/extract_f0_features.py \
    --clips_dir  output/dev/clips/ \
    --output_dir output/dev/ \
    --n_points   10
```

Per-clip pipeline: pYIN → voiced frames → interpolate to N points → CV check. Clips with voiced ratio < 25% or CV > 0.50 are discarded.

Output: `output/dev/f0_dataset.csv` — columns: `sample_id, tone, f0_1 … f0_10`

### Phase 5 — Classifier training

**Production mode** (train on train split, evaluate on dev):

```bash
python scripts/train_classifier.py \
    --train_csv     output/train/f0_dataset.csv \
    --eval_csv      output/dev/f0_dataset.csv \
    --artifacts_dir models/
```

**Quick experiment** (single file, random 80/20 split):

```bash
python scripts/train_classifier.py \
    --train_csv output/dev/f0_dataset.csv \
    --test_size 0.2
```

Saves: `mlp.joblib`, `scaler.joblib`, `speaker_stats.json`, per-model classification reports and confusion matrices.

---

## Inference API

### Run the server

```bash
python -m uvicorn api.feedback:app --reload
```

Models load once at startup from `models/`.

### POST /feedback

| Field            | Type   | Description                                      |
|------------------|--------|--------------------------------------------------|
| `audio`          | file   | `.wav` audio (mono, 16 kHz recommended)          |
| `text`           | string | Expected hanzi string, e.g. `你好世界`            |
| `whisper_model`  | string | `tiny` / `base` / `small` (default: `base`)      |
| `speaker_mean_hz`| float  | Override speaker mean F0 (optional)              |

Pipeline per request:
1. Whisper `word_timestamps=True` → word-level intervals
2. Distribute hanzi characters uniformly within each word's interval
3. Estimate speaker mean F0 from all voiced frames in the full audio
4. Per syllable: slice → pYIN → normalise → delta → scale → MLP
5. pypinyin → expected tone per character
6. Return per-syllable result + overall accuracy

### Test with a recorded sentence

```bash
# From corpus (reads hanzi from .lab automatically)
python scripts/test_api.py --sentence mfa_corpus/dev/A11/A11_101.wav

# Record from microphone
python scripts/record_and_test.py --text 你好世界 --seconds 4
```

### GET /health

Returns `{"status": "ok", "speakers_in_train": 50}`.

---

## Docker

```bash
docker build -t tone-api .
docker run -p 8000:8000 tone-api
```

The image bundles only the inference stack (`api/`, `scripts/infer.py`, `models/`). The training pipeline and MFA corpus are excluded.

---

## Known limitations

**Alignment precision**
Whisper returns word-level timestamps. Characters within a word are distributed uniformly, introducing ~±100ms boundary error per syllable. This is the primary source of the gap between isolated-clip accuracy (~67%) and continuous-speech accuracy (~48%).

**Tone 3 under-coverage**
The dipping-rising contour (╲╱) has the highest discard rate (~35%) because pYIN classifies the low-pitch trough as unvoiced. Tone 3 is also most confused with Tone 4, which shares the same initial falling trajectory.

**Tone 5 (neutral) performance**
~5× fewer training samples than Tone 4. Class-balanced weighting partially compensates but F1 remains low. Neutral tone is highly context-dependent and not well-modelled by isolated F0 shape.

**Tone sandhi**
pypinyin returns lexical tones. In continuous speech, T3+T3 → T2+T3 (e.g. 你好 is phonetically ní hǎo). The classifier may correctly identify the realised tone but the system marks it as an error. This affects any T3 sequence and also the particles 不 and 一.

**Speaker generalisation**
Test set contains 10 speakers (D-prefix) absent from training. Speaker normalisation falls back to the global mean (240 Hz), causing a ~4pp accuracy drop. The training set is majority female (THCHS-30), so male speakers (F0 ~100–150 Hz) are further disadvantaged.

---

## Future work

Based on literature on Mandarin tone recognition in continuous speech (He & Hao 2006, Li/Xu/Zhou, Chen 2007, Zhou et al. 2008).

### 1. Richer per-syllable features

Beyond 10 interpolated F0 points, add:

| Feature | Rationale |
|---------|-----------|
| F0 at onset, midpoint, offset | Captures overall pitch height |
| Initial and final slope | Separates Tone 2 from Tone 4 |
| F0 minimum value | Essential for Tone 3 detection |
| Temporal position of F0 minimum | Distinguishes simple fall from dip-rise |
| Syllable duration | Important cue for Tone 5 |
| RMS energy | Tone 5 is shorter and weaker |
| Voiced ratio as a feature (not just a filter) | Encodes voicing pattern for the model |
| Pinyin initial/final class | Phonetic context of the syllable |

### 2. Contextual model

The current classifier treats each syllable in isolation. A window-based model seeing neighbouring syllables would better handle coarticulation in continuous speech (He & Hao 2006):

```
[syllable_{i-1}, syllable_i, syllable_{i+1}]  →  predicted_tone_i
```

Candidates: BiLSTM/GRU or Temporal CNN over the syllable sequence.

### 3. Tone sandhi handling

Add a `surface_tone` label alongside `lexical_tone`. Apply sandhi rules at metadata generation time and train the model against the realised tone:

| Context | Rule |
|---------|------|
| T3 + T3 | first T3 → T2 |
| T3 before T1/T2/T4 | first T3 → half-third (low) |
| 不 before T4 | 不 → T2 |
| 一 before T4 | 一 → T2; before T1/T2/T3 → T4 |

Compare user pronunciation against `surface_tone` in the API, not `lexical_tone`.

### 4. Better production alignment

Replace Whisper uniform-distribution with syllable-level forced alignment online:
- **Best option**: CTC forced alignment (wav2vec2 or MMS) — fast, no separate conda env
- **Intermediate**: Whisper for phrase validation only; align syllables against the expected text separately
- **Minimal improvement**: onset detection + energy-based boundary refinement within Whisper word intervals

The architecture is described in Chen (2007) for Mandarin CAPT: forced alignment → per-syllable F0 vector → tone classifier.

### 5. Speaker generalisation

- Replace mean-based normalisation with **median + IQR** (more robust to outliers and short clips)
- Compute speaker reference from the full utterance voiced frames (already partially done in the API)
- Add **pitch-shift augmentation** during training to simulate male voices and cover the gap from the female-dominated THCHS-30
- Enforce **speaker-stratified splits** and run Leave-One-Speaker-Out cross-validation to measure true generalisation

### 6. Alternative classifiers

Keep SVM/MLP as baseline; benchmark against:
- **LightGBM / XGBoost** — for tabular feature vectors
- **BiLSTM or Temporal CNN** — over the F0 sequence with contextual window
- Hybrid: per-syllable acoustic model + context model over the sequence
