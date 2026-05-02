"""
Shared utilities for THCHS-30 tonal dataset pipeline.
"""
import re
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Set True to map neutral tone (5) to 0 instead.
NEUTRAL_TONE_AS_ZERO: bool = False


def find_wav_trn_pairs(data_dir: str) -> list[tuple[Path, Path]]:
    """
    Scan data_dir for .wav files and return (wav, trn) pairs where both exist.
    Ignores hidden/temporary files (e.g. .swp, .scp).
    """
    data_path = Path(data_dir)
    pairs = []
    for wav_file in sorted(data_path.glob("**/*.wav")):
        if wav_file.name.startswith("."):
            continue
        trn_file = Path(str(wav_file) + ".trn")
        if trn_file.exists():
            pairs.append((wav_file, trn_file))
        else:
            log.debug(f"No .trn for: {wav_file.name}")
    return pairs


def read_trn_file(trn_path: Path) -> dict:
    """
    Parse a THCHS-30 .wav.trn file (UTF-8, 3 lines).

    Line 1 – Chinese text, words separated by spaces (multi-char words possible)
    Line 2 – Pinyin with tone numbers, one token per syllable
    Line 3 – Phone-level transcription

    Returns dict with keys: hanzi_words, pinyin_list, phones.
    Any missing line yields an empty list for that key.
    """
    with open(trn_path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    result: dict = {"hanzi_words": [], "pinyin_list": [], "phones": []}
    if len(lines) >= 1:
        result["hanzi_words"] = lines[0].split()
    if len(lines) >= 2:
        result["pinyin_list"] = lines[1].split()
    if len(lines) >= 3:
        result["phones"] = lines[2].split()
    return result


def flatten_hanzi(hanzi_words: list[str]) -> list[str]:
    """
    Expand word tokens into individual characters.

    ['绿', '阳春', '烟景'] -> ['绿', '阳', '春', '烟', '景']
    """
    return [char for word in hanzi_words for char in word]


def extract_tone(pinyin: str) -> int:
    """
    Return the tone integer embedded at the end of a pinyin token.

    'lv4'   -> 4
    'yang2' -> 2
    'de5'   -> 5  (neutral; returns 0 if NEUTRAL_TONE_AS_ZERO is True)
    'ma'    -> 5  (no digit found, treated as neutral)
    """
    match = re.search(r"(\d)$", pinyin.strip())
    if match:
        tone = int(match.group(1))
        if tone == 5 and NEUTRAL_TONE_AS_ZERO:
            return 0
        return tone
    return 5  # no tone digit → neutral


def generate_pinyin_fallback(hanzi_list: list[str]) -> list[str]:
    """
    Generate pinyin with tone numbers via pypinyin.
    Used only when a .trn file has no pinyin line (rare in THCHS-30).
    """
    try:
        from pypinyin import pinyin, Style
    except ImportError:
        raise ImportError("pypinyin is required: pip install pypinyin")

    result = []
    for char in hanzi_list:
        py = pinyin(char, style=Style.TONE3, heteronym=False)
        result.append(py[0][0] if py and py[0] else char)
    return result
