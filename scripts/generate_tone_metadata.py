"""
Generate syllable-level tonal metadata (CSV + JSON) from THCHS-30.

For each audio file, reads the matching .wav.trn, aligns Chinese characters
with their pinyin+tone, and writes one flat record per syllable.

NOTE — Audio is NOT segmented here.
    The next step in this pipeline is forced alignment using the
    Montreal Forced Aligner (MFA).  MFA will produce a TextGrid per
    sentence with millisecond-precise boundaries for each syllable.
    Those boundaries will then be used to slice the original .wav into
    individual syllable clips.

Usage:
    # Process all files in data/
    python generate_tone_metadata.py \\
        --data_dir  <path_to_data_thchs30/data> \\
        --output_dir output/

    # Process only the training split
    python generate_tone_metadata.py \\
        --data_dir  <path_to_data_thchs30/data> \\
        --split_dir <path_to_data_thchs30/train> \\
        --output_dir output/train/
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    find_wav_trn_pairs,
    read_trn_file,
    flatten_hanzi,
    extract_tone,
    generate_pinyin_fallback,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# CSV column order
FIELDNAMES = ["sample_id", "original_audio_path", "syllable_index", "hanzi", "pinyin", "tone"]


def build_records(wav_path: Path, trn_path: Path) -> list[dict]:
    """
    Return one record dict per syllable for a single (wav, trn) pair.

    sample_id format: <wav_stem>_<syllable_index>  (e.g. A11_0_3)
    original_audio_path is stored as a POSIX-style string for portability.
    """
    sample_stem = wav_path.stem  # e.g. "A11_0"
    trn_data    = read_trn_file(trn_path)

    hanzi_chars = flatten_hanzi(trn_data["hanzi_words"])
    pinyin_list = trn_data["pinyin_list"]

    if not pinyin_list:
        log.warning(f"{trn_path.name}: no pinyin line — falling back to pypinyin.")
        pinyin_list = generate_pinyin_fallback(hanzi_chars)

    if len(hanzi_chars) != len(pinyin_list):
        log.warning(
            f"{sample_stem}: char/pinyin count mismatch "
            f"({len(hanzi_chars)} chars vs {len(pinyin_list)} pinyin) — skipping."
        )
        return []

    records = []
    for idx, (hanzi, pinyin) in enumerate(zip(hanzi_chars, pinyin_list)):
        records.append(
            {
                "sample_id":           f"{sample_stem}_{idx}",
                "original_audio_path": wav_path.as_posix(),
                "syllable_index":      idx,
                "hanzi":               hanzi,
                "pinyin":              pinyin,
                "tone":                extract_tone(pinyin),
            }
        )
    return records


def collect_pairs(data_dir: str, split_dir: str | None) -> list[tuple[Path, Path]]:
    """
    Return (wav, trn) pairs to process.

    If split_dir is given, its .wav files identify *which* stems to include,
    but the canonical files are always sourced from data_dir.  This is robust
    to Windows extractors that turn Linux symlinks into plain text stubs.
    """
    if not split_dir:
        return find_wav_trn_pairs(data_dir)

    split_path = Path(split_dir)
    data_path  = Path(data_dir)
    pairs: list[tuple[Path, Path]] = []

    for wav_entry in sorted(split_path.glob("*.wav")):
        stem = wav_entry.stem  # e.g. "A11_101"
        wav  = data_path / f"{stem}.wav"
        trn  = data_path / f"{stem}.wav.trn"

        if not wav.exists():
            log.warning(f"wav not found in data_dir: {stem}.wav")
            continue
        if not trn.exists():
            log.warning(f"trn not found in data_dir: {stem}.wav.trn")
            continue

        pairs.append((wav, trn))

    return pairs


def summarise_tones(records: list[dict]) -> None:
    counts: dict[int, int] = {}
    for r in records:
        t = r["tone"]
        counts[t] = counts.get(t, 0) + 1

    labels = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "neutral (5)"}
    print("\nTone distribution:")
    for tone in sorted(counts):
        label = labels.get(tone, str(tone))
        print(f"  Tone {tone} ({label:12s}): {counts[tone]:>8,}")
    print(f"  {'Total':>20s}: {len(records):>8,}")


def generate_metadata(data_dir: str, output_dir: str, split_dir: str | None) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(data_dir, split_dir)
    if not pairs:
        log.error("No (wav, trn) pairs found. Check --data_dir and --split_dir.")
        sys.exit(1)

    log.info(f"Found {len(pairs):,} audio files to process.")

    all_records: list[dict] = []
    skipped = 0

    for wav_path, trn_path in tqdm(pairs, desc="Building records", unit="file"):
        records = build_records(wav_path, trn_path)
        if not records:
            skipped += 1
            continue
        all_records.extend(records)

    log.info(f"Syllable records generated : {len(all_records):,}")
    if skipped:
        log.warning(f"Files skipped (alignment errors): {skipped}")

    # --- Write CSV ---
    csv_path = output_path / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_records)
    log.info(f"Wrote CSV  -> {csv_path}")

    # --- Write JSON ---
    json_path = output_path / "metadata.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(all_records, fh, ensure_ascii=False, indent=2)
    log.info(f"Wrote JSON -> {json_path}")

    summarise_tones(all_records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate syllable-level tonal metadata from THCHS-30.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to data_thchs30/data (contains .wav and .wav.trn files)",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Directory for metadata.csv and metadata.json (default: output/)",
    )
    parser.add_argument(
        "--split_dir",
        default=None,
        help="Optional path to train/, dev/, or test/ to restrict processing to one split",
    )
    args = parser.parse_args()
    generate_metadata(args.data_dir, args.output_dir, args.split_dir)


if __name__ == "__main__":
    main()
