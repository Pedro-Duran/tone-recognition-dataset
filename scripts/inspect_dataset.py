"""
Inspect the THCHS-30 dataset structure and print a summary report.

Usage:
    python inspect_dataset.py --data_dir <path_to_data_thchs30>

Example:
    python inspect_dataset.py --data_dir C:/Downloads/data_thchs30/data_thchs30
"""
import argparse
import sys
from pathlib import Path

# Ensure Chinese characters print correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow running from scripts/ or from project root
sys.path.insert(0, str(Path(__file__).parent))
from utils import find_wav_trn_pairs, read_trn_file, flatten_hanzi


def inspect(data_dir: str) -> None:
    root = Path(data_dir)
    if not root.exists():
        print(f"[ERROR] Path not found: {root}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  THCHS-30 Dataset Inspection")
    print(f"{'='*50}")
    print(f"Root: {root.resolve()}\n")

    # --- Top-level contents ---
    print("Top-level contents:")
    for item in sorted(root.iterdir()):
        tag = "[DIR] " if item.is_dir() else "[FILE]"
        print(f"  {tag} {item.name}")

    # --- data/ subdirectory ---
    data_subdir = root / "data"
    if not data_subdir.exists():
        print(f"\n[WARNING] 'data/' subdirectory not found at {data_subdir}")
        return

    all_wav = [f for f in data_subdir.glob("**/*.wav") if not f.name.startswith(".")]
    all_trn = [f for f in data_subdir.glob("**/*.trn") if not f.name.startswith(".")]
    pairs   = find_wav_trn_pairs(str(data_subdir))

    print(f"\nFiles in data/:")
    print(f"  .wav files         : {len(all_wav):>6,}")
    print(f"  .wav.trn files     : {len(all_trn):>6,}")
    print(f"  Matched pairs      : {len(pairs):>6,}")
    unpaired = len(all_wav) - len(pairs)
    if unpaired:
        print(f"  [WARNING] {unpaired} .wav files have no matching .trn")

    # --- Lexicons ---
    print(f"\nLexicons:")
    for lex_dir in ("lm_word", "lm_phone"):
        lex_path = root / lex_dir / "lexicon.txt"
        if lex_path.exists():
            n = sum(1 for _ in open(lex_path, encoding="utf-8"))
            print(f"  {lex_dir}/lexicon.txt : {n:,} entries")
        else:
            print(f"  {lex_dir}/lexicon.txt : NOT FOUND")

    # --- Train / dev / test splits ---
    print(f"\nSplit sizes:")
    for split in ("train", "dev", "test"):
        split_dir = root / split
        if split_dir.exists():
            wav_count = len(list(split_dir.glob("*.wav")))
            print(f"  {split:5s} : {wav_count:>5,} .wav files")
        else:
            print(f"  {split:5s} : directory not found")

    # --- Sample .trn inspection ---
    if pairs:
        wav_sample, trn_sample = pairs[0]
        trn_data = read_trn_file(trn_sample)
        chars = flatten_hanzi(trn_data["hanzi_words"])

        print(f"\nSample file: {trn_sample.name}")
        print(f"  Hanzi (joined) : {''.join(chars)}")
        print(f"  Pinyin (first 8): {' '.join(trn_data['pinyin_list'][:8])} ...")
        print(f"  Total chars    : {len(chars)}")
        print(f"  Total pinyin   : {len(trn_data['pinyin_list'])}")

        aligned = len(chars) == len(trn_data["pinyin_list"])
        status  = "OK" if aligned else "MISMATCH — check alignment logic"
        print(f"  Char-Pinyin alignment: {status}")

        # Detect whether pinyin line is present at all
        has_pinyin = len(trn_data["pinyin_list"]) > 0
        print(f"  Pinyin in .trn : {'YES (no pypinyin fallback needed)' if has_pinyin else 'NO (pypinyin will be used)'}")

    print(f"\n{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect THCHS-30 dataset structure.")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the data_thchs30 root directory (the one containing data/, train/, etc.)",
    )
    args = parser.parse_args()
    inspect(args.data_dir)


if __name__ == "__main__":
    main()
