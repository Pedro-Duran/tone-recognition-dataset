FROM python:3.11-slim

WORKDIR /app

# System deps for librosa (soundfile → libsndfile) and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# API dependencies only — training pipeline not included
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Inference core + API
COPY scripts/infer.py  scripts/infer.py
COPY api/              api/

# Model artifacts (mlp.joblib, scaler.joblib, speaker_stats.json)
# svm.joblib is intentionally excluded (.gitignore) — not used by the API
COPY models/mlp.joblib          models/mlp.joblib
COPY models/scaler.joblib       models/scaler.joblib
COPY models/speaker_stats.json  models/speaker_stats.json

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.feedback:app", "--host", "0.0.0.0", "--port", "8000"]
