# Tone Recognition Dataset (THCHS-30)

Pipeline for converting THCHS-30 into a syllable-level Mandarin tone classification dataset.

**Current stage:** label generation (no audio segmentation yet).  
**Next stage:** forced alignment with Montreal Forced Aligner → audio slicing.

---

## Project structure

```
tone-recognition-dataset/
├── data/
│   └── raw/              # place THCHS-30 here (or point --data_dir elsewhere)
├── output/
│   ├── metadata.csv
│   └── metadata.json
├── scripts/
│   ├── inspect_dataset.py       # explore dataset structure
│   ├── generate_tone_metadata.py # build syllable-level labels
│   └── utils.py                 # shared helpers
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended (uses built-in `str | None` union syntax).

---

## Step 1 — Inspect the dataset

Verify paths, file counts, and alignment before generating anything:

```bash
python scripts/inspect_dataset.py \
    --data_dir "C:/path/to/data_thchs30/data_thchs30"
```

Expected output includes:
- Count of `.wav` / `.wav.trn` pairs
- Lexicon sizes (`lm_word`, `lm_phone`)
- Split sizes (train / dev / test)
- A sample `.trn` with alignment check

---

## Step 2 — Generate metadata

### Process all files

```bash
python scripts/generate_tone_metadata.py \
    --data_dir  "C:/path/to/data_thchs30/data_thchs30/data" \
    --output_dir output/
```

### Process one split only (recommended)

```bash
# Training split
python scripts/generate_tone_metadata.py \
    --data_dir  "C:/path/to/data_thchs30/data_thchs30/data" \
    --split_dir "C:/path/to/data_thchs30/data_thchs30/train" \
    --output_dir output/train/

# Dev split
python scripts/generate_tone_metadata.py \
    --data_dir  "C:/path/to/data_thchs30/data_thchs30/data" \
    --split_dir "C:/path/to/data_thchs30/data_thchs30/dev" \
    --output_dir output/dev/
```

Outputs written to `--output_dir`:
- `metadata.csv` — one row per syllable
- `metadata.json` — same data, JSON array

---

## Output format

### CSV (`metadata.csv`)

```
sample_id,original_audio_path,syllable_index,hanzi,pinyin,tone
A11_0_0,/data/A11_0.wav,0,绿,lv4,4
A11_0_1,/data/A11_0.wav,1,是,shi4,4
A11_0_2,/data/A11_0.wav,2,阳,yang2,2
```

### JSON (`metadata.json`)

```json
[
  {
    "sample_id": "A11_0_0",
    "original_audio_path": "/data/A11_0.wav",
    "syllable_index": 0,
    "hanzi": "绿",
    "pinyin": "lv4",
    "tone": 4
  }
]
```

**Tone labels:**

| Value | Meaning              |
|-------|----------------------|
| 1     | First tone (高平)    |
| 2     | Second tone (升)     |
| 3     | Third tone (上)      |
| 4     | Fourth tone (去)     |
| 5     | Neutral tone (轻声)  |

To map neutral tone to `0` instead of `5`, set `NEUTRAL_TONE_AS_ZERO = True`
in [scripts/utils.py](scripts/utils.py).

---

## Pinyin source

THCHS-30 `.trn` files include a pinyin line (one token per syllable), so
`pypinyin` is used only as a fallback for any file that lacks it.

---

## Next steps

### Forced alignment (Montreal Forced Aligner)

This pipeline currently produces **sentence-level** labels.  The next step
is to obtain precise syllable timestamps using MFA:

1. Install MFA: https://montreal-forced-aligner.readthedocs.io
2. Prepare a pronunciation dictionary from `lm_phone/lexicon.txt`
3. Run `mfa align` on the THCHS-30 `data/` directory
4. Parse the resulting `.TextGrid` files to get per-syllable boundaries
5. Slice each `.wav` at those boundaries → one clip per syllable
6. Update `metadata.csv/json` with `clip_path`, `start_ms`, `end_ms` columns

### Model training (future)

After slicing, each clip can be used directly as input to a tone classifier
(e.g. CNN or transformer on mel-spectrograms).
