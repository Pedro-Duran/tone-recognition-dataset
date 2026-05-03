"""
Quick smoke-test for the /feedback endpoint.

Modes:
  Syllable clip (default):
    python scripts/test_api.py --tone 1
    python scripts/test_api.py --clip output/dev/clips/tone_2/A11_101_2.wav --text 国

  Full sentence (reads hanzi from the matching .lab file automatically):
    python scripts/test_api.py --sentence mfa_corpus/dev/A11/A11_101.wav
    python scripts/test_api.py --sentence mfa_corpus/dev/A11/A11_101.wav --model base

Usage (with uvicorn running in another terminal):
    python -m uvicorn api.feedback:app --reload
"""
import argparse
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

API_URL = "http://127.0.0.1:8000/feedback"

TONE_LABELS = {1: "平", 2: "升", 3: "上", 4: "去", 5: "轻"}
EXAMPLE_CHARS = {1: "一", 2: "国", 3: "我", 4: "是", 5: "的"}


def find_clip(tone: int) -> Path:
    base = Path("output/dev/clips") / f"tone_{tone}"
    clips = sorted(base.glob("*.wav"))
    if not clips:
        sys.exit(f"No clips found in {base}.")
    return clips[0]


def load_lab_text(wav_path: Path) -> str:
    """Read hanzi from the matching .lab file (spaces stripped)."""
    lab = wav_path.with_suffix(".lab")
    if not lab.exists():
        sys.exit(f".lab file not found: {lab}")
    return lab.read_text(encoding="utf-8").replace(" ", "").strip()


def call_api(clip_path: Path, text: str, whisper_model: str) -> dict:
    with open(clip_path, "rb") as f:
        resp = requests.post(
            API_URL,
            files={"audio": (clip_path.name, f, "audio/wav")},
            data={"text": text, "whisper_model": whisper_model},
            timeout=300,
        )
    if resp.status_code != 200:
        print(f"ERROR {resp.status_code}: {resp.text}")
        sys.exit(1)
    return resp.json()


def print_results(data: dict, text: str) -> None:
    print(f"\nSpeaker Hz   : {data['speaker_hz']:.1f}")
    print(f"Overall acc  : {data['overall_accuracy']*100:.0f}%  "
          f"({sum(1 for s in data['syllables'] if s['correct'])}/{len(data['syllables'])} correct)")
    print()

    for s in data["syllables"]:
        tick = "✓" if s["correct"] else "✗"
        exp  = TONE_LABELS.get(s["expected_tone"], "?")
        pred = TONE_LABELS.get(s["predicted_tone"], "?")
        timing = f"[{s['start_s']:.2f}–{s['end_s']:.2f}s]"
        print(f"  {tick} {s['char']}  {s['pinyin']:<8}  "
              f"expected T{s['expected_tone']}({exp})  "
              f"predicted T{s['predicted_tone']}({pred})  "
              f"conf {s['confidence']*100:.0f}%  {timing}")

    # Summary of errors
    errors = [s for s in data["syllables"] if not s["correct"]]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for s in errors:
            exp  = TONE_LABELS.get(s["expected_tone"], "?")
            pred = TONE_LABELS.get(s["predicted_tone"], "?")
            print(f"  {s['char']} {s['pinyin']}: expected T{s['expected_tone']}({exp}) "
                  f"→ got T{s['predicted_tone']}({pred})  conf {s['confidence']*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sentence", default=None,
                        help="Full-sentence .wav (reads hanzi from matching .lab)")
    parser.add_argument("--clip",     default=None,
                        help="Single-syllable .wav clip")
    parser.add_argument("--text",     default=None,
                        help="Expected hanzi (required with --clip for multi-char)")
    parser.add_argument("--tone",     type=int, default=1, choices=range(1, 6),
                        help="Auto-pick a clip from tone_N/ folder (default: 1)")
    parser.add_argument("--model",    default="tiny",
                        help="Whisper model: tiny|base|small (default: tiny)")
    args = parser.parse_args()

    if args.sentence:
        wav_path = Path(args.sentence)
        if not wav_path.exists():
            sys.exit(f"File not found: {wav_path}")
        text = load_lab_text(wav_path)
        print(f"Sentence : {wav_path.name}")
        print(f"Hanzi    : {text}  ({len(text)} chars)")
        print(f"Model    : Whisper {args.model}")
        print("Sending to API (Whisper alignment may take 10-30s)...")
    else:
        wav_path = Path(args.clip) if args.clip else find_clip(args.tone)
        text     = args.text or EXAMPLE_CHARS.get(args.tone, "一")
        print(f"Clip  : {wav_path}")
        print(f"Text  : {text}  (expected tone {args.tone})")
        print(f"Model : Whisper {args.model}")

    data = call_api(wav_path, text, args.model)
    print_results(data, text)


if __name__ == "__main__":
    main()
