"""
FastAPI feedback endpoint for Mandarin tone assessment.

Pipeline per request
────────────────────
1. Receive audio (multipart upload) + expected hanzi string
2. Whisper word_timestamps=True → list of (word, start_s, end_s)
3. Map each hanzi character to a (start, end) interval
   - Characters within a Whisper word share the word's interval uniformly
   - If Whisper word count diverges from hanzi count, fall back to uniform
     distribution across the total audio duration
4. Estimate speaker mean F0 from all voiced frames in the full audio
5. Slice audio array per syllable → run infer.predict_clip()
6. pypinyin → expected tone per character
7. Return structured JSON: per-syllable result + overall accuracy

Endpoints
─────────
POST /feedback
    Form fields:
        audio           : audio file (.wav recommended, mono 16 kHz)
        text            : expected hanzi string, e.g. "你好世界"
        speaker_mean_hz : (optional) float, overrides automatic estimation
        whisper_model   : (optional) "tiny"|"base"|"small" (default "base")

GET /health
    Returns {"status": "ok"}

Run locally
───────────
    pip install -r requirements-api.txt
    uvicorn api.feedback:app --reload
"""
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# ── FastAPI / Pydantic ────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise RuntimeError("Run: pip install fastapi uvicorn python-multipart")

# ── pypinyin ──────────────────────────────────────────────────────────────────
try:
    from pypinyin import Style, pinyin as to_pinyin
except ImportError:
    raise RuntimeError("Run: pip install pypinyin")

# ── Project inference core ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import infer as _infer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Model artifacts (loaded once at startup) ──────────────────────────────────
_MODELS_DIR = Path(__file__).parent.parent / "models"
_N_POINTS   = 10  # must match training
_mlp = _scaler = _speaker_means = _global_fallback = None


def _load_models() -> None:
    global _mlp, _scaler, _speaker_means, _global_fallback
    log.info(f"Loading model artifacts from {_MODELS_DIR} ...")
    _mlp, _scaler, _speaker_means, _global_fallback = _infer.load_artifacts(_MODELS_DIR)
    log.info(f"  {len(_speaker_means)} speakers in training set  |  "
             f"global fallback {_global_fallback:.1f} Hz")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mandarin Tone Feedback API",
    description="Assesses pronunciation by comparing predicted F0 tones with expected tones.",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    _load_models()


@app.get("/health")
async def health():
    return {"status": "ok", "speakers_in_train": len(_speaker_means or {})}


# ── Pydantic response models ──────────────────────────────────────────────────

class SyllableResult(BaseModel):
    char:           str
    pinyin:         str
    expected_tone:  int
    predicted_tone: int
    correct:        bool
    confidence:     float
    proba:          dict[str, float]
    start_s:        float
    end_s:          float


class FeedbackResponse(BaseModel):
    syllables:        list[SyllableResult]
    overall_accuracy: float
    speaker_hz:       float
    whisper_model:    str


# ── Whisper alignment ─────────────────────────────────────────────────────────

def _align_whisper(
    audio_path: str,
    whisper_model: str = "base",
) -> list[tuple[str, float, float]]:
    """
    Run Whisper with word_timestamps=True.

    Returns a flat list of (word_text, start_s, end_s).
    Filters out empty words and punctuation-only tokens.
    """
    try:
        import whisper
    except ImportError:
        raise RuntimeError("Run: pip install openai-whisper")

    model  = whisper.load_model(whisper_model)
    result = model.transcribe(
        audio_path,
        language="zh",
        word_timestamps=True,
        fp16=False,
    )

    words: list[tuple[str, float, float]] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            text = w.get("word", "").strip()
            if not text or not any("一" <= c <= "鿿" for c in text):
                continue
            words.append((text, float(w["start"]), float(w["end"])))

    return words


def _char_intervals(
    hanzi: str,
    whisper_words: list[tuple[str, float, float]],
    total_duration_s: float,
) -> list[tuple[float, float]]:
    """
    Map each hanzi character to a (start, end) time interval.

    Strategy:
      - Build a flat list of characters from Whisper output, pairing each
        character with its word's (start, end).
      - If the count matches |hanzi|, use those intervals directly.
      - Otherwise fall back: distribute total_duration_s uniformly.
    """
    chars_only = [c for c in hanzi if "一" <= c <= "鿿"]
    n = len(chars_only)

    if n == 0:
        return []

    # Flatten Whisper words → per-char intervals
    whisper_chars: list[tuple[float, float]] = []
    for word_text, start, end in whisper_words:
        word_hans = [c for c in word_text if "一" <= c <= "鿿"]
        if not word_hans:
            continue
        step = (end - start) / len(word_hans)
        for i in range(len(word_hans)):
            whisper_chars.append((start + i * step, start + (i + 1) * step))

    if len(whisper_chars) == n:
        return whisper_chars

    # Fallback: uniform distribution
    log.warning(
        f"Whisper returned {len(whisper_chars)} char intervals for {n} hanzi chars "
        "— falling back to uniform distribution"
    )
    step = total_duration_s / n
    return [(i * step, (i + 1) * step) for i in range(n)]


# ── Speaker F0 estimation ─────────────────────────────────────────────────────

def _estimate_speaker_hz(audio: np.ndarray, sr: int) -> float:
    """Estimate speaker mean F0 from all voiced frames in the full audio."""
    import librosa
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=_infer.F0_MIN_HZ,
        fmax=_infer.F0_MAX_HZ,
        sr=sr,
        frame_length=_infer.FRAME_LENGTH,
        hop_length=_infer.HOP_LENGTH,
    )
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) == 0:
        log.warning("No voiced frames in full audio — using global fallback for speaker Hz")
        return _global_fallback
    return float(np.mean(voiced_f0))


# ── Per-syllable inference ────────────────────────────────────────────────────

def _predict_syllable(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    speaker_mean_hz: float,
) -> dict:
    """Slice audio array and run the F0 inference pipeline in-memory."""
    s = max(0, int(start_s * sr))
    e = min(len(audio), int(end_s * sr))
    clip = audio[s:e]

    if len(clip) == 0:
        raise ValueError("Empty clip after slicing")

    # Write to a temp file so infer.load_audio / librosa.pyin can process it
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        sf.write(str(tmp_path), clip, sr, subtype="PCM_16")
        result = _infer.predict_clip(
            tmp_path, _mlp, _scaler, _speaker_means, _global_fallback,
            n_points=_N_POINTS,
            speaker_mean_hz=speaker_mean_hz,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return result


# ── Expected tones from pypinyin ──────────────────────────────────────────────

def _expected_tones(hanzi: str) -> list[tuple[str, str, int]]:
    """
    Returns list of (char, pinyin_with_tone, tone_number) for each hanzi char.
    Tone 5 (neutral) maps from pypinyin tone number 5.
    """
    import re
    chars = [c for c in hanzi if "一" <= c <= "鿿"]
    pinyin_list = to_pinyin(chars, style=Style.TONE3, heteronym=False)

    results = []
    for char, py_group in zip(chars, pinyin_list):
        py = py_group[0] if py_group else ""
        m  = re.search(r"(\d)$", py)
        tone = int(m.group(1)) if m else 5
        results.append((char, py, tone))
    return results


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    audio: UploadFile = File(..., description="Audio file (.wav, mono 16 kHz)"),
    text:  str        = Form(..., description="Expected hanzi string, e.g. 你好世界"),
    speaker_mean_hz:  Optional[float] = Form(None,    description="Override speaker mean F0 in Hz"),
    whisper_model:    str             = Form("base",  description="Whisper model: tiny|base|small"),
) -> FeedbackResponse:
    if _mlp is None:
        raise HTTPException(503, "Models not loaded yet")

    # Save upload to temp file
    audio_bytes = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        audio_path = Path(tmp.name)

    try:
        import librosa
        full_audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        total_duration  = len(full_audio) / sr

        # 1. Whisper alignment
        try:
            whisper_words = _align_whisper(str(audio_path), whisper_model)
        except Exception as exc:
            log.error(f"Whisper alignment failed: {exc}")
            whisper_words = []

        # 2. Speaker mean F0
        if speaker_mean_hz is None:
            speaker_mean_hz = _estimate_speaker_hz(full_audio, sr)
            log.info(f"Estimated speaker mean: {speaker_mean_hz:.1f} Hz")

        # 3. Character → time intervals
        intervals = _char_intervals(text, whisper_words, total_duration)

        # 4. Expected tones from pypinyin
        expected = _expected_tones(text)

        if len(intervals) != len(expected):
            raise HTTPException(
                422,
                f"Interval count ({len(intervals)}) != char count ({len(expected)}). "
                "Check that `text` contains only standard hanzi characters."
            )

        # 5. Per-syllable inference
        syllables: list[SyllableResult] = []
        correct_count = 0

        for (char, pinyin, exp_tone), (start_s, end_s) in zip(expected, intervals):
            try:
                pred = _predict_syllable(full_audio, sr, start_s, end_s, speaker_mean_hz)
            except ValueError as exc:
                log.warning(f"Char '{char}' [{start_s:.2f}-{end_s:.2f}s]: {exc} — skipping")
                continue

            correct = pred["tone"] == exp_tone
            if correct:
                correct_count += 1

            syllables.append(SyllableResult(
                char=char,
                pinyin=pinyin,
                expected_tone=exp_tone,
                predicted_tone=pred["tone"],
                correct=correct,
                confidence=pred["confidence"],
                proba={str(k): v for k, v in pred["proba"].items()},
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
            ))

        overall_accuracy = correct_count / len(syllables) if syllables else 0.0

    finally:
        audio_path.unlink(missing_ok=True)

    return FeedbackResponse(
        syllables=syllables,
        overall_accuracy=round(overall_accuracy, 4),
        speaker_hz=round(speaker_mean_hz, 1),
        whisper_model=whisper_model,
    )
