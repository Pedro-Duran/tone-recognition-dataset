"""
Extract F0 pitch-contour features from per-syllable audio clips.

WHY F0 INSTEAD OF RAW AUDIO
────────────────────────────
Mandarin tone is defined entirely by the F0 contour — the melodic shape of
the fundamental frequency over the duration of a syllable:

  Tone 1 (高平): flat high          ─────
  Tone 2 (升)  : rising             ╱
  Tone 3 (上)  : dipping then rise  ╲╱
  Tone 4 (去)  : sharp falling      ╲
  Tone 5 (轻声): short/neutral      ·

A CNN on a mel-spectrogram sees the full signal and can latch onto recording
artefacts or the small timing errors that forced alignment introduces.  An
MLP/SVM on a normalised 5-10 point F0 vector focuses on the one acoustic cue
that distinguishes tones, making it far more robust to imperfect segmentation.

PIPELINE FOR EACH CLIP
───────────────────────
1. Load audio (16 kHz mono)
2. Run pYIN (probabilistic YIN) → F0 time-series, voiced/unvoiced flags
3. Keep only voiced frames; discard clip if too few voiced frames
4. Interpolate voiced F0 to exactly N points (temporal normalisation)
5. Discard clip if F0 coefficient of variation > threshold (too erratic)
6. Round and store the N values as features

Usage:
    python scripts/extract_f0_features.py \\
        --clips_dir  output/dev/clips/ \\
        --output_dir output/dev/ \\
        --n_points   10

Requires: pip install librosa
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Pitch range for adult Mandarin speakers ─────────────────────────────────
F0_MIN_HZ: float = 50.0   # below any realistic speech fundamental
F0_MAX_HZ: float = 600.0  # above any realistic adult fundamental

# ── Quality thresholds ───────────────────────────────────────────────────────
# Minimum fraction of frames that must be voiced for the clip to be kept.
MIN_VOICED_RATIO: float = 0.25

# Maximum coefficient of variation (std / mean) for the interpolated F0.
# High CV → erratic pitch → likely a bad cut or non-tonal segment.
MAX_CV: float = 0.50

# pYIN analysis window / hop (samples at 16 kHz).
# Smaller than librosa defaults for better resolution on short syllable clips.
FRAME_LENGTH: int = 1024   # ~64 ms
HOP_LENGTH:   int = 256    # ~16 ms


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio(path: Path, sr: int = 16000) -> tuple[np.ndarray, int]:
    try:
        import librosa
        audio, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
        return audio, loaded_sr
    except Exception as exc:
        raise RuntimeError(str(exc))


# ── F0 extraction ─────────────────────────────────────────────────────────────

def extract_voiced_f0(audio: np.ndarray, sr: int) -> tuple[np.ndarray | None, str | None]:
    """
    Run pYIN and return only the voiced F0 frames.

    Returns (voiced_f0_array, None) on success.
    Returns (None, reason_string)   if the clip should be discarded.
    """
    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    total_frames  = len(f0)
    voiced_frames = int(np.sum(voiced_flag))
    voiced_ratio  = voiced_frames / total_frames if total_frames > 0 else 0.0

    if voiced_ratio < MIN_VOICED_RATIO or voiced_frames < 2:
        return None, f"unvoiced (voiced={voiced_ratio:.0%})"

    return f0[voiced_flag], None


# ── Normalisation ─────────────────────────────────────────────────────────────

def interpolate_to_n_points(f0_voiced: np.ndarray, n: int) -> np.ndarray:
    """Resample a voiced F0 sequence to exactly n points."""
    x_src = np.linspace(0.0, 1.0, len(f0_voiced))
    x_dst = np.linspace(0.0, 1.0, n)
    return np.interp(x_dst, x_src, f0_voiced)


# ── Quality check ─────────────────────────────────────────────────────────────

def coefficient_of_variation(arr: np.ndarray) -> float:
    mean = arr.mean()
    return float(arr.std() / mean) if mean > 0 else float("inf")


# ── Per-clip pipeline ─────────────────────────────────────────────────────────

def process_clip(
    path: Path, n_points: int, sr: int = 16000
) -> tuple[np.ndarray | None, str | None]:
    """
    Full pipeline for one clip.

    Returns (f0_array, None)       → keep this clip.
    Returns (None,     reason_str) → discard this clip.
    """
    audio, sr = load_audio(path, sr=sr)

    voiced_f0, reason = extract_voiced_f0(audio, sr)
    if voiced_f0 is None:
        return None, reason

    f0_interp = interpolate_to_n_points(voiced_f0, n_points)

    cv = coefficient_of_variation(f0_interp)
    if cv > MAX_CV:
        return None, f"unstable (CV={cv:.2f})"

    return f0_interp, None


# ── Main ──────────────────────────────────────────────────────────────────────

def build_f0_dataset(clips_dir: str, output_dir: str, n_points: int) -> None:
    clips_path = Path(clips_dir)
    out_path   = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Collect all clips from tone_1/ … tone_5/
    all_clips: list[tuple[Path, int]] = []
    for tone in range(1, 6):
        tone_dir = clips_path / f"tone_{tone}"
        if not tone_dir.exists():
            log.warning(f"Not found, skipping: {tone_dir}")
            continue
        wavs = sorted(tone_dir.glob("*.wav"))
        log.info(f"  tone_{tone}: {len(wavs):,} clips")
        all_clips.extend((w, tone) for w in wavs)

    if not all_clips:
        log.error(f"No .wav clips found under {clips_path}")
        sys.exit(1)

    log.info(f"Total clips to process: {len(all_clips):,}")

    records: list[dict] = []
    discarded: dict[str, int] = {}
    errors = 0

    for clip_path, tone in tqdm(all_clips, desc="Extracting F0", unit="clip"):
        try:
            f0, reason = process_clip(clip_path, n_points)
        except Exception as exc:
            log.debug(f"{clip_path.name}: load error — {exc}")
            errors += 1
            continue

        if f0 is None:
            bucket = reason.split("(")[0].strip()
            discarded[bucket] = discarded.get(bucket, 0) + 1
            continue

        record: dict = {"sample_id": clip_path.stem, "tone": tone}
        for i, val in enumerate(f0, start=1):
            record[f"f0_{i}"] = round(float(val), 2)
        records.append(record)

    # ── Summary logging ───────────────────────────────────────────────────────
    log.info(f"Kept     : {len(records):,} clips")
    for reason, count in sorted(discarded.items(), key=lambda x: -x[1]):
        log.info(f"Discarded ({reason}): {count:,}")
    if errors:
        log.warning(f"Load errors: {errors:,}")

    if not records:
        log.error("No records kept — check your clips_dir and thresholds.")
        sys.exit(1)

    # ── F0 statistics ─────────────────────────────────────────────────────────
    f0_cols  = [f"f0_{i}" for i in range(1, n_points + 1)]
    f0_matrix = np.array([[r[c] for c in f0_cols] for r in records])
    log.info(f"F0 range : {f0_matrix.min():.1f} – {f0_matrix.max():.1f} Hz")
    log.info(f"F0 mean  : {f0_matrix.mean():.1f} Hz  |  std: {f0_matrix.std():.1f} Hz")

    # ── Write outputs ─────────────────────────────────────────────────────────
    fieldnames = ["sample_id", "tone"] + f0_cols
    df = pd.DataFrame(records, columns=fieldnames)

    csv_path  = out_path / "f0_dataset.csv"
    json_path = out_path / "f0_dataset.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)

    log.info(f"Wrote CSV  -> {csv_path}")
    log.info(f"Wrote JSON -> {json_path}")

    print(f"\nTone distribution in F0 dataset (n_points={n_points}):")
    for tone, count in df["tone"].value_counts().sort_index().items():
        print(f"  Tone {tone}: {count:>6,}")
    print(f"  Total : {len(df):>6,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract F0 pitch-contour features from syllable clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--clips_dir",
        required=True,
        help="Directory containing tone_1/ … tone_5/ subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Where to write f0_dataset.csv and f0_dataset.json",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=10,
        help="F0 samples per syllable after interpolation (default: 10)",
    )
    args = parser.parse_args()
    build_f0_dataset(args.clips_dir, args.output_dir, args.n_points)


if __name__ == "__main__":
    main()
