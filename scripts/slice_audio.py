"""
Slice original audio files into individual syllable clips.

Reads metadata_aligned.csv (produced by parse_textgrids.py) and cuts
each .wav at the MFA-provided start/end timestamps.

Output clips are named {sample_id}.wav and organised by tone:
    clips/
      tone_1/  A11_0_0.wav ...
      tone_2/  ...
      tone_3/  ...
      tone_4/  ...
      tone_5/  ...   (neutral)

Requires:
    pip install soundfile numpy

Usage:
    python scripts/slice_audio.py \\
        --aligned_metadata output/dev/aligned/metadata_aligned.csv \\
        --clips_dir        output/dev/clips/
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Clips shorter than this (seconds) are skipped as likely alignment artifacts
MIN_DURATION_S: float = 0.05


def slice_clips(aligned_metadata: str, clips_dir: str) -> None:
    df = pd.read_csv(aligned_metadata)
    out_dir = Path(clips_dir)

    # Drop rows without timestamps (unaligned syllables)
    before = len(df)
    df = df.dropna(subset=["start_time", "end_time"])
    dropped = before - len(df)
    if dropped:
        log.warning(f"Skipping {dropped:,} rows with missing timestamps.")

    log.info(f"Slicing {len(df):,} syllable clips -> {out_dir}")

    # Pre-create tone subdirectories
    for tone in df["tone"].unique():
        (out_dir / f"tone_{tone}").mkdir(parents=True, exist_ok=True)

    written = 0
    too_short = 0
    errors = 0

    # Group by audio file to load each .wav only once
    groups = list(df.groupby("original_audio_path"))
    for audio_path_str, group in tqdm(groups, desc="Audio files", unit="file"):
        audio_path = Path(audio_path_str)

        if not audio_path.exists():
            log.warning(f"Audio not found: {audio_path} — skipping {len(group)} clips")
            errors += len(group)
            continue

        try:
            audio, sr = sf.read(str(audio_path), dtype="float32")
        except Exception as exc:
            log.warning(f"{audio_path.name}: {exc} — skipping")
            errors += len(group)
            continue

        for _, row in group.iterrows():
            start_s = float(row["start_time"])
            end_s   = float(row["end_time"])
            duration = end_s - start_s

            if duration < MIN_DURATION_S:
                log.debug(f"{row['sample_id']}: {duration:.3f}s < minimum — skipped")
                too_short += 1
                continue

            start_i = int(start_s * sr)
            end_i   = int(end_s   * sr)
            clip    = audio[start_i:end_i]

            tone_dir  = out_dir / f"tone_{int(row['tone'])}"
            clip_path = tone_dir / f"{row['sample_id']}.wav"
            sf.write(str(clip_path), clip, sr)
            written += 1

    log.info(f"Written : {written:,} clips")
    if too_short:
        log.info(f"Too short (< {MIN_DURATION_S}s): {too_short:,} clips skipped")
    if errors:
        log.warning(f"Errors  : {errors:,} clips skipped (missing audio)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slice THCHS-30 audio into per-syllable clips using MFA timestamps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--aligned_metadata",
        required=True,
        help="metadata_aligned.csv from parse_textgrids.py",
    )
    parser.add_argument(
        "--clips_dir",
        required=True,
        help="Output directory for syllable clips (organised by tone/)",
    )
    args = parser.parse_args()
    slice_clips(args.aligned_metadata, args.clips_dir)


if __name__ == "__main__":
    main()