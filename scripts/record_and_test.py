"""
Record audio from the microphone and send to the /feedback endpoint.

Usage:
    python scripts/record_and_test.py --text 你好 --seconds 3
    python scripts/record_and_test.py --text 七十年代 --seconds 4 --model tiny

The script:
  1. Counts down 3 seconds then records for --seconds
  2. Saves to a temp .wav file at 16 kHz mono
  3. POSTs to the running API and prints the tone feedback

Requires: pip install sounddevice
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

try:
    import sounddevice as sd
except ImportError:
    print("Install sounddevice first:  pip install sounddevice")
    sys.exit(1)

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    print("Install soundfile/numpy:  pip install soundfile numpy")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Install requests:  pip install requests")
    sys.exit(1)

API_URL    = "http://127.0.0.1:8000/feedback"
SAMPLE_RATE = 16000
TONE_LABELS = {1: "平", 2: "升", 3: "上", 4: "去", 5: "轻"}


def record(seconds: int) -> np.ndarray:
    print(f"\nPrepare to speak — recording in:", end="", flush=True)
    for i in (3, 2, 1):
        print(f" {i}", end="", flush=True)
        time.sleep(1)
    print(" GO!", flush=True)

    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print(f"Recording done ({seconds}s).\n")
    return audio.squeeze()


def send_to_api(audio: np.ndarray, text: str, model: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    sf.write(str(tmp_path), audio, SAMPLE_RATE, subtype="PCM_16")

    try:
        with open(tmp_path, "rb") as f:
            resp = requests.post(
                API_URL,
                files={"audio": ("recording.wav", f, "audio/wav")},
                data={"text": text, "whisper_model": model},
                timeout=300,
            )
    finally:
        tmp_path.unlink(missing_ok=True)

    if resp.status_code != 200:
        print(f"API error {resp.status_code}: {resp.text}")
        sys.exit(1)

    return resp.json()


def print_results(data: dict) -> None:
    correct = sum(1 for s in data["syllables"] if s["correct"])
    total   = len(data["syllables"])
    print(f"Speaker Hz   : {data['speaker_hz']:.1f}")
    print(f"Overall acc  : {data['overall_accuracy']*100:.0f}%  ({correct}/{total} correct)\n")

    for s in data["syllables"]:
        tick = "✓" if s["correct"] else "✗"
        exp  = TONE_LABELS.get(s["expected_tone"], "?")
        pred = TONE_LABELS.get(s["predicted_tone"], "?")
        bar_correct = "█" * int(s["confidence"] * 20)
        print(f"  {tick} {s['char']}  {s['pinyin']:<8}  "
              f"expected T{s['expected_tone']}({exp})  "
              f"predicted T{s['predicted_tone']}({pred})  "
              f"conf {s['confidence']*100:.0f}%")

        for t in sorted(s["proba"], key=int):
            prob = float(s["proba"][t])
            bar  = "█" * int(prob * 20)
            mark = " ← expected" if int(t) == s["expected_tone"] else ""
            print(f"       T{t} {prob*100:4.1f}%  {bar}{mark}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Record from mic and test tone feedback API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text",    required=True,
                        help="Expected hanzi, e.g. 你好 or 七十年代")
    parser.add_argument("--seconds", type=int, default=4,
                        help="Recording duration in seconds (default: 4)")
    parser.add_argument("--model",   default="tiny",
                        help="Whisper model: tiny|base|small (default: tiny)")
    args = parser.parse_args()

    print(f"Text    : {args.text}  ({len(args.text)} chars)")
    print(f"Duration: {args.seconds}s")
    print(f"Model   : Whisper {args.model}")
    print(f"\nTip: speak clearly, one char per ~0.4-0.5s")
    print(f"     For {len(args.text)} chars, aim for "
          f"{len(args.text) * 0.4:.1f}–{len(args.text) * 0.6:.1f}s of speech.")

    audio = record(args.seconds)

    print("Sending to API (alignment + inference)...")
    data = send_to_api(audio, args.text, args.model)
    print_results(data)


if __name__ == "__main__":
    main()
