"""
Prepare an MFA-ready corpus from THCHS-30.

Each audio file gets:
  - a hard-linked (or copied) .wav in corpus/{speaker}/
  - a .lab file with space-separated pinyin tokens from the .trn line 2

Speaker is inferred from the file stem: 'A11_101' -> speaker 'A11'.

MFA expects this layout:
    corpus/
      A11/
        A11_0.wav
        A11_0.lab
        ...
      B12/
        ...

After running this script:
    conda create -n aligner -c conda-forge montreal-forced-aligner
    conda activate aligner
    mfa model download acoustic mandarin_mfa
    mfa model download dictionary mandarin_mfa
    mfa validate <corpus_dir> mandarin_mfa mandarin_mfa
    mfa align   <corpus_dir> mandarin_mfa mandarin_mfa <textgrid_dir>

Usage:
    python scripts/prepare_mfa_corpus.py \\
        --data_dir  <path_to_data_thchs30/data> \\
        --corpus_dir mfa_corpus/

    # Restrict to one split:
    python scripts/prepare_mfa_corpus.py \\
        --data_dir  <path_to_data_thchs30/data> \\
        --split_dir <path_to_data_thchs30/dev> \\
        --corpus_dir mfa_corpus/dev/
"""
import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import find_wav_trn_pairs, read_trn_file, flatten_hanzi

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def speaker_from_stem(stem: str) -> str:
    """'A11_101' -> 'A11'"""
    match = re.match(r"^([A-Za-z]+\d+)", stem)
    return match.group(1) if match else "unknown"


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link src to dst (no extra disk space); fall back to copy if it fails."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def collect_pairs(data_dir: str, split_dir: str | None) -> list[tuple[Path, Path]]:
    if not split_dir:
        return find_wav_trn_pairs(data_dir)

    data_path = Path(data_dir)
    pairs: list[tuple[Path, Path]] = []
    for wav_entry in sorted(Path(split_dir).glob("*.wav")):
        stem = wav_entry.stem
        wav = data_path / f"{stem}.wav"
        trn = data_path / f"{stem}.wav.trn"
        if not wav.exists():
            log.warning(f"wav not found in data_dir: {stem}.wav")
            continue
        if not trn.exists():
            log.warning(f"trn not found in data_dir: {stem}.wav.trn")
            continue
        pairs.append((wav, trn))
    return pairs


def prepare_corpus(data_dir: str, corpus_dir: str, split_dir: str | None) -> None:
    corpus_path = Path(corpus_dir)
    pairs = collect_pairs(data_dir, split_dir)

    if not pairs:
        log.error("No (wav, trn) pairs found. Check --data_dir / --split_dir.")
        sys.exit(1)

    log.info(f"Preparing MFA corpus for {len(pairs):,} files -> {corpus_path}")

    skipped = 0
    for wav_path, trn_path in tqdm(pairs, desc="Building corpus", unit="file"):
        stem = wav_path.stem
        speaker = speaker_from_stem(stem)
        speaker_dir = corpus_path / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)

        trn_data = read_trn_file(trn_path)
        hanzi_chars = flatten_hanzi(trn_data["hanzi_words"])

        if not hanzi_chars:
            log.warning(f"{stem}: no hanzi in .trn — skipped")
            skipped += 1
            continue

        # .lab file: space-separated individual characters (e.g. "绿 是 阳 春 烟 景")
        # MFA's mandarin_mfa dictionary maps hanzi characters to phonemes, not pinyin.
        lab_path = speaker_dir / f"{stem}.lab"
        lab_path.write_text(" ".join(hanzi_chars), encoding="utf-8")

        dst_wav = speaker_dir / wav_path.name
        if not dst_wav.exists():
            link_or_copy(wav_path, dst_wav)

    if skipped:
        log.warning(f"Skipped {skipped} files (no pinyin line).")

    log.info("Corpus ready.")
    print(f"\n--- Next: run MFA ---")
    print(f"conda create -n aligner -c conda-forge montreal-forced-aligner")
    print(f"conda activate aligner")
    print(f"mfa model download acoustic mandarin_mfa")
    print(f"mfa model download dictionary mandarin_mfa")
    print(f"mfa validate {corpus_path} mandarin_mfa mandarin_mfa")
    print(f"mfa align   {corpus_path} mandarin_mfa mandarin_mfa <textgrid_output_dir>")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MFA corpus from THCHS-30.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data_dir",   required=True, help="Path to data_thchs30/data")
    parser.add_argument("--corpus_dir", required=True, help="Output directory for MFA corpus")
    parser.add_argument("--split_dir",  default=None,  help="Optional: restrict to train/, dev/, or test/")
    args = parser.parse_args()
    prepare_corpus(args.data_dir, args.corpus_dir, args.split_dir)


if __name__ == "__main__":
    main()