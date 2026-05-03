"""
Single-clip tone inference using saved MLP model artifacts.

Accepts a pre-segmented .wav clip (one syllable) and returns the predicted
Mandarin tone (1-5) along with confidence scores.

Usage:
    python scripts/infer.py path/to/clip.wav
    python scripts/infer.py path/to/clip.wav --speaker_mean_hz 210.5
    python scripts/infer.py path/to/clip.wav --models_dir models/ --n_points 10

Speaker mean:
    If --speaker_mean_hz is not provided, the script estimates the speaker mean
    from the clip itself (poor estimate for short clips, but functional). For
    production use, maintain a running estimate across multiple clips from the
    same speaker session and pass it in.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TONE_LABELS = {1: "平 (flat high)", 2: "升 (rising)", 3: "上 (dip-rise)",
               4: "去 (falling)", 5: "轻 (neutral)"}

F0_MIN_HZ    = 50.0
F0_MAX_HZ    = 600.0
FRAME_LENGTH = 1024
HOP_LENGTH   = 256
MIN_VOICED_RATIO = 0.25
MAX_CV       = 0.50


# ── Audio / F0 ────────────────────────────────────────────────────────────────

def load_audio(path: Path, sr: int = 16000):
    try:
        import librosa
        audio, _ = librosa.load(str(path), sr=sr, mono=True)
        return audio, sr
    except Exception as exc:
        raise RuntimeError(f"Could not load {path}: {exc}")


def extract_voiced_f0(audio: np.ndarray, sr: int):
    import librosa
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=F0_MIN_HZ, fmax=F0_MAX_HZ, sr=sr,
        frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
    )
    voiced_frames = int(np.sum(voiced_flag))
    total_frames  = len(f0)
    voiced_ratio  = voiced_frames / total_frames if total_frames > 0 else 0.0

    if voiced_ratio < MIN_VOICED_RATIO or voiced_frames < 2:
        raise ValueError(
            f"Too few voiced frames ({voiced_ratio:.0%}) — "
            "clip too short, silent, or noisy."
        )
    return f0[voiced_flag]


def interpolate_to_n(f0_voiced: np.ndarray, n: int) -> np.ndarray:
    x_src = np.linspace(0.0, 1.0, len(f0_voiced))
    x_dst = np.linspace(0.0, 1.0, n)
    return np.interp(x_dst, x_src, f0_voiced)


def coefficient_of_variation(arr: np.ndarray) -> float:
    mean = arr.mean()
    return float(arr.std() / mean) if mean > 0 else float("inf")


# ── Feature engineering (must match train_classifier.py exactly) ──────────────

def build_features(
    f0_interp: np.ndarray,
    speaker_mean_hz: float,
) -> np.ndarray:
    """Hz → semitones → add delta → 1-D feature vector (2N-1 dims)."""
    f0_semi = 12.0 * np.log2(np.clip(f0_interp, 1.0, None) / speaker_mean_hz)
    delta   = np.diff(f0_semi)
    return np.hstack([f0_semi, delta])


# ── Model loading ─────────────────────────────────────────────────────────────

def load_artifacts(models_dir: Path):
    try:
        import joblib
    except ImportError:
        raise RuntimeError("joblib not installed — run: pip install joblib")

    mlp    = joblib.load(models_dir / "mlp.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")

    sp_path = models_dir / "speaker_stats.json"
    with open(sp_path, encoding="utf-8") as fh:
        sp_stats = json.load(fh)

    speaker_means   = sp_stats["speaker_means"]
    global_fallback = sp_stats["global_fallback"]

    return mlp, scaler, speaker_means, global_fallback


# ── Public API ────────────────────────────────────────────────────────────────

def predict_clip(
    clip_path: Path,
    mlp,
    scaler,
    speaker_means: dict,
    global_fallback: float,
    n_points: int = 10,
    speaker_id: str | None = None,
    speaker_mean_hz: float | None = None,
) -> dict:
    """
    Predict the tone of a single pre-segmented syllable clip.

    Parameters
    ----------
    clip_path       : path to a .wav file containing one syllable
    mlp / scaler    : loaded joblib artifacts
    speaker_means   : dict mapping speaker-id → mean F0 in Hz
    global_fallback : mean Hz to use when speaker is not in speaker_means
    n_points        : number of F0 interpolation points (must match training)
    speaker_id      : e.g. "A11" — used to look up speaker_means
    speaker_mean_hz : override; if given, bypasses speaker_means lookup

    Returns
    -------
    dict with keys:
        tone        : int 1-5
        label       : human-readable tone name
        confidence  : float, probability of the predicted class
        proba       : dict {tone_int: probability}
        speaker_hz  : the speaker mean F0 that was used
    """
    audio, sr   = load_audio(clip_path)
    voiced_f0   = extract_voiced_f0(audio, sr)
    f0_interp   = interpolate_to_n(voiced_f0, n_points)

    cv = coefficient_of_variation(f0_interp)
    if cv > MAX_CV:
        log.warning(f"High F0 variability (CV={cv:.2f}) — prediction may be unreliable")

    # Resolve speaker mean
    if speaker_mean_hz is None:
        if speaker_id and speaker_id in speaker_means:
            speaker_mean_hz = speaker_means[speaker_id]
            log.debug(f"Using stored mean for {speaker_id}: {speaker_mean_hz:.1f} Hz")
        else:
            speaker_mean_hz = global_fallback
            if speaker_id:
                log.debug(f"Speaker {speaker_id!r} not in training set, using global fallback")

    feat     = build_features(f0_interp, speaker_mean_hz)
    feat_s   = scaler.transform(feat.reshape(1, -1))

    proba_arr = mlp.predict_proba(feat_s)[0]
    classes   = mlp.classes_
    pred_idx  = int(np.argmax(proba_arr))
    pred_tone = int(classes[pred_idx])

    return {
        "tone":       pred_tone,
        "label":      TONE_LABELS.get(pred_tone, str(pred_tone)),
        "confidence": float(proba_arr[pred_idx]),
        "proba":      {int(c): float(p) for c, p in zip(classes, proba_arr)},
        "speaker_hz": speaker_mean_hz,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Mandarin tone from a single syllable clip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("clip", help="Path to .wav clip (one syllable)")
    parser.add_argument("--models_dir",      default="models/",
                        help="Directory with mlp.joblib, scaler.joblib, speaker_stats.json")
    parser.add_argument("--n_points",        type=int, default=10,
                        help="F0 interpolation points — must match training (default: 10)")
    parser.add_argument("--speaker_id",      default=None,
                        help="Speaker label (e.g. A11) for looking up stored mean F0")
    parser.add_argument("--speaker_mean_hz", type=float, default=None,
                        help="Override speaker mean F0 in Hz (skips lookup)")
    args = parser.parse_args()

    clip_path   = Path(args.clip)
    models_dir  = Path(args.models_dir)

    if not clip_path.exists():
        log.error(f"Clip not found: {clip_path}")
        sys.exit(1)

    log.info(f"Loading artifacts from {models_dir} ...")
    mlp, scaler, speaker_means, global_fallback = load_artifacts(models_dir)
    log.info(f"  {len(speaker_means)} speakers in training set  |  "
             f"global fallback {global_fallback:.1f} Hz")

    try:
        result = predict_clip(
            clip_path, mlp, scaler, speaker_means, global_fallback,
            n_points=args.n_points,
            speaker_id=args.speaker_id,
            speaker_mean_hz=args.speaker_mean_hz,
        )
    except ValueError as exc:
        log.error(str(exc))
        sys.exit(1)

    print(f"\nClip      : {clip_path.name}")
    print(f"Predicted : Tone {result['tone']} — {result['label']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Speaker Hz: {result['speaker_hz']:.1f} Hz")
    print(f"\nProbabilities:")
    for tone, prob in sorted(result["proba"].items()):
        bar = "█" * int(prob * 30)
        print(f"  Tone {tone}  {prob*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
