"""
Parse MFA TextGrid output and add syllable timestamps to existing metadata.

MFA produces one TextGrid per utterance.  The 'words' tier contains one
interval per pinyin token from the .lab file, in the same order as
syllable_index in metadata.csv.

This script joins those timestamps back into the metadata, producing
metadata_aligned.csv / metadata_aligned.json with two new columns:
    start_time  (seconds, float)
    end_time    (seconds, float)

Requires:
    pip install praatio

Usage:
    python scripts/parse_textgrids.py \\
        --textgrid_dir  <mfa_output_dir> \\
        --metadata_csv  output/dev/metadata.csv \\
        --output_dir    output/dev/aligned/
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# MFA silence labels to exclude when counting speech intervals
SILENCE_LABELS = {"", "sp", "sil", "spn", "<eps>"}


def load_word_intervals(tg_path: Path) -> list[tuple[float, float, str]]:
    """
    Return (start, end, label) tuples from the 'words' tier of a TextGrid.
    Raises ValueError if the tier is missing.
    """
    try:
        from praatio import textgrid
    except ImportError:
        raise ImportError("praatio is required: pip install praatio")

    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)

    if "words" not in tg.tierNames:
        raise ValueError(f"No 'words' tier. Available: {tg.tierNames}")

    tier = tg.getTier("words")
    return [(e.start, e.end, e.label) for e in tier.entries]


def build_timestamp_index(tg_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """
    Walk tg_dir for all .TextGrid files and return a dict mapping
    file stem -> list of (start, end) for speech-only intervals (in order).
    """
    tg_files = sorted(tg_dir.rglob("*.TextGrid"))
    if not tg_files:
        log.error(f"No .TextGrid files found under {tg_dir}")
        sys.exit(1)

    log.info(f"Found {len(tg_files):,} TextGrid files.")
    index: dict[str, list[tuple[float, float]]] = {}
    skipped = 0

    for tg_path in tqdm(tg_files, desc="Parsing TextGrids", unit="file"):
        try:
            intervals = load_word_intervals(tg_path)
        except Exception as exc:
            log.warning(f"{tg_path.name}: {exc} — skipped")
            skipped += 1
            continue

        speech = [(s, e) for s, e, label in intervals if label not in SILENCE_LABELS]
        index[tg_path.stem] = speech

    if skipped:
        log.warning(f"TextGrids skipped (parse errors): {skipped}")

    return index


def join_timestamps(df: pd.DataFrame, index: dict[str, list[tuple[float, float]]]) -> pd.DataFrame:
    """
    Add start_time / end_time columns by looking up each row's stem + syllable_index.
    Rows with no matching TextGrid or out-of-range index get NaN.
    """
    # Derive stem: "A11_101_3" -> stem = "A11_101", syllable_index = 3
    df = df.copy()
    df["_stem"] = df["sample_id"].str.rsplit("_", n=1).str[0]

    starts, ends = [], []
    mismatches = 0

    for _, row in df.iterrows():
        stem = row["_stem"]
        idx  = int(row["syllable_index"])
        intervals = index.get(stem)

        if intervals is None:
            starts.append(None)
            ends.append(None)
            continue

        if idx >= len(intervals):
            mismatches += 1
            starts.append(None)
            ends.append(None)
        else:
            s, e = intervals[idx]
            starts.append(round(s, 4))
            ends.append(round(e, 4))

    if mismatches:
        log.warning(f"Index out-of-range (syllable count mismatch): {mismatches} rows")

    df["start_time"] = starts
    df["end_time"]   = ends
    return df.drop(columns=["_stem"])


def parse_textgrids(textgrid_dir: str, metadata_csv: str, output_dir: str) -> None:
    tg_dir  = Path(textgrid_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    log.info(f"Loaded {len(df):,} syllable records from {metadata_csv}")

    index = build_timestamp_index(tg_dir)
    df    = join_timestamps(df, index)

    aligned = df["start_time"].notna().sum()
    log.info(f"Aligned {aligned:,} / {len(df):,} syllables ({aligned/len(df)*100:.1f}%).")

    csv_path  = out_dir / "metadata_aligned.csv"
    json_path = out_dir / "metadata_aligned.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    log.info(f"Wrote CSV  -> {csv_path}")
    log.info(f"Wrote JSON -> {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add MFA syllable timestamps to existing metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--textgrid_dir", required=True, help="Directory containing MFA .TextGrid output")
    parser.add_argument("--metadata_csv", required=True, help="metadata.csv from generate_tone_metadata.py")
    parser.add_argument("--output_dir",   required=True, help="Where to write metadata_aligned.csv / .json")
    args = parser.parse_args()
    parse_textgrids(args.textgrid_dir, args.metadata_csv, args.output_dir)


if __name__ == "__main__":
    main()