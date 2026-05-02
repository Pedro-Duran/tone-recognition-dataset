"""
Train SVM and MLP tone classifiers on F0 contour features.

FEATURE ENGINEERING (applied to every split)
─────────────────────────────────────────────
1. Speaker normalisation  — Hz → semitones relative to each speaker's mean F0
2. Delta F0               — first differences appended (10 pts → 19 features)
3. Standard scaling       — zero mean, unit variance (fit on train only)
4. Class-balanced weights — correct for tone frequency imbalance

MODES
─────
Production (train on train split, evaluate on dev):
    python scripts/train_classifier.py \\
        --train_csv     output/train/f0_dataset.csv \\
        --eval_csv      output/dev/f0_dataset.csv \\
        --artifacts_dir models/

Quick experiment (single file, random 80/20 split):
    python scripts/train_classifier.py \\
        --train_csv output/dev/f0_dataset.csv \\
        --test_size 0.2
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TONE_LABELS = {
    1: "Tone 1 (平)",
    2: "Tone 2 (升)",
    3: "Tone 3 (上)",
    4: "Tone 4 (去)",
    5: "Tone 5 (轻)",
}

# Reference results from dev 80/20 split (for comparison table)
BASELINE_DEV_8020 = {
    "SVM (no improvements)": {"accuracy": 0.6342, "macro_f1": 0.489},
    "MLP (no improvements)": {"accuracy": 0.6308, "macro_f1": 0.489},
    "SVM (improved)":        {"accuracy": 0.6295, "macro_f1": 0.570},
    "MLP (improved)":        {"accuracy": 0.6371, "macro_f1": 0.577},
}


# ── Speaker extraction ────────────────────────────────────────────────────────

def extract_speaker(sample_id: str) -> str:
    """'A11_101_0' → 'A11'"""
    m = re.match(r"^([A-Za-z]+\d+)", sample_id)
    return m.group(1) if m else "unknown"


# ── Feature engineering ───────────────────────────────────────────────────────

def speaker_normalize(
    df: pd.DataFrame,
    f0_cols: list[str],
    speaker_means: dict[str, float] | None = None,
    global_fallback: float | None = None,
) -> tuple[np.ndarray, dict[str, float], float]:
    """
    Convert F0 from Hz to semitones relative to each speaker's mean.

    When speaker_means is None (training set), compute from df.
    When provided (eval set), apply without refitting to avoid data leakage.
    Speakers absent from speaker_means fall back to global_fallback mean.
    """
    speakers = df["sample_id"].apply(extract_speaker).values
    X_hz = df[f0_cols].values.astype(np.float64)

    if speaker_means is None:
        speaker_means = {}
        for spk in np.unique(speakers):
            mask = speakers == spk
            speaker_means[spk] = float(np.mean(X_hz[mask]))
        global_fallback = float(np.mean(list(speaker_means.values())))

    X_semi = np.empty_like(X_hz)
    for i, spk in enumerate(speakers):
        mean_hz = speaker_means.get(spk, global_fallback)
        X_semi[i] = 12.0 * np.log2(np.clip(X_hz[i], 1.0, None) / mean_hz)

    return X_semi, speaker_means, global_fallback


def add_delta(X: np.ndarray) -> np.ndarray:
    """Append first differences: N F0 points → N + (N-1) features."""
    return np.hstack([X, np.diff(X, axis=1)])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    f0_cols = [c for c in df.columns if c.startswith("f0_")]
    log.info(f"  {path.name}: {len(df):,} samples  |  {len(f0_cols)} F0 pts  |  "
             f"tones {sorted(df['tone'].unique())}")
    return df, f0_cols


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_separator(title: str = "") -> None:
    width = 64
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def evaluate_model(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    mac_f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report  = classification_report(
        y_test, y_pred,
        target_names=[TONE_LABELS[t] for t in sorted(np.unique(y_test))],
        digits=3,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y_test)))

    print_separator(name)
    print(f"Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Macro F1 : {mac_f1:.4f}\n")
    print(report)

    tones = sorted(np.unique(y_test))
    print("Confusion matrix (rows=true, cols=predicted):")
    print("     " + "  ".join(f"T{t}" for t in tones))
    for i, row in enumerate(cm):
        print(f"  T{tones[i]}  " + "  ".join(f"{v:3d}" for v in row))

    return {
        "name":     name,
        "accuracy": acc,
        "macro_f1": mac_f1,
        "report":   report,
        "cm":       cm,
        "tones":    tones,
        "y_pred":   y_pred,
    }


# ── Artifact saving ───────────────────────────────────────────────────────────

def save_artifacts(
    artifacts_dir: Path,
    svm,
    mlp,
    scaler: StandardScaler,
    speaker_means: dict[str, float],
    global_fallback: float,
    svm_results: dict,
    mlp_results: dict,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if _JOBLIB_AVAILABLE:
        joblib.dump(svm,    artifacts_dir / "svm.joblib")
        joblib.dump(mlp,    artifacts_dir / "mlp.joblib")
        joblib.dump(scaler, artifacts_dir / "scaler.joblib")
        log.info(f"Saved models → {artifacts_dir}/svm.joblib, mlp.joblib, scaler.joblib")
    else:
        log.warning("joblib not found — models not saved. Run: pip install joblib")

    speaker_stats = {
        "speaker_means":    speaker_means,
        "global_fallback":  global_fallback,
    }
    sp_path = artifacts_dir / "speaker_stats.json"
    with open(sp_path, "w", encoding="utf-8") as fh:
        json.dump(speaker_stats, fh, indent=2, ensure_ascii=False)
    log.info(f"Saved speaker stats → {sp_path}")

    for results in (svm_results, mlp_results):
        slug = results["name"].split()[0].lower()

        report_path = artifacts_dir / f"report_{slug}.txt"
        report_path.write_text(
            f"Model: {results['name']}\n"
            f"Accuracy : {results['accuracy']:.4f}\n"
            f"Macro F1 : {results['macro_f1']:.4f}\n\n"
            + results["report"],
            encoding="utf-8",
        )

        tones = results["tones"]
        cm_df = pd.DataFrame(
            results["cm"],
            index=[f"true_T{t}" for t in tones],
            columns=[f"pred_T{t}" for t in tones],
        )
        cm_path = artifacts_dir / f"confusion_matrix_{slug}.csv"
        cm_df.to_csv(cm_path)

    log.info(f"Saved reports and confusion matrices → {artifacts_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    train_csv: str,
    eval_csv: str | None,
    test_size: float,
    random_state: int,
    artifacts_dir: str | None,
) -> None:
    log.info("Loading datasets...")
    df_train, f0_cols = load_dataset(Path(train_csv))

    if eval_csv:
        df_eval, _ = load_dataset(Path(eval_csv))
        log.info(f"Mode: train on {Path(train_csv).name}, evaluate on {Path(eval_csv).name}")
    else:
        idx = np.arange(len(df_train))
        y_all = df_train["tone"].values.astype(int)
        idx_tr, idx_ev = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=y_all
        )
        df_eval  = df_train.iloc[idx_ev].reset_index(drop=True)
        df_train = df_train.iloc[idx_tr].reset_index(drop=True)
        log.info(f"Mode: random {1-test_size:.0%}/{test_size:.0%} split")

    log.info(f"Train: {len(df_train):,}  |  Eval: {len(df_eval):,}")

    y_train = df_train["tone"].values.astype(int)
    y_eval  = df_eval["tone"].values.astype(int)

    # Feature engineering — speaker stats from train only
    X_train, spk_means, global_fb = speaker_normalize(df_train, f0_cols)
    X_eval,  _,         _         = speaker_normalize(df_eval, f0_cols, spk_means, global_fb)

    unseen = set(df_eval["sample_id"].apply(extract_speaker)) - set(spk_means)
    if unseen:
        log.warning(f"Eval speakers not in train ({len(unseen)}): {sorted(unseen)} "
                    f"→ using global fallback {global_fb:.1f} Hz")

    log.info(f"Speakers in train: {len(spk_means)}  |  "
             f"F0 range {min(spk_means.values()):.0f}–{max(spk_means.values()):.0f} Hz")

    X_train = add_delta(X_train)
    X_eval  = add_delta(X_eval)
    log.info(f"Feature vector: {X_train.shape[1]} dims  "
             f"({len(f0_cols)} F0 + {len(f0_cols)-1} delta)")

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s  = scaler.transform(X_eval)

    sw_train = compute_sample_weight("balanced", y_train)

    # Train
    log.info("Training SVM (RBF, class_weight=balanced)...")
    svm = SVC(kernel="rbf", C=10, gamma="scale",
              class_weight="balanced", random_state=random_state)
    svm.fit(X_train_s, y_train)

    log.info("Training MLP (64-32, sample_weight=balanced)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(X_train_s, y_train, sample_weight=sw_train)

    # Evaluate
    svm_res = evaluate_model("SVM", svm, X_eval_s, y_eval)
    mlp_res = evaluate_model(f"MLP ({mlp.n_iter_} epochs)", mlp, X_eval_s, y_eval)

    # Comparison table
    print_separator("Comparison with previous dev 80/20 baseline")
    print(f"  {'Model':<36} {'Accuracy':>9}  {'Macro F1':>9}")
    print(f"  {'─'*36}  {'─'*9}  {'─'*9}")
    for model_name, scores in BASELINE_DEV_8020.items():
        print(f"  {model_name:<36} {scores['accuracy']*100:>8.2f}%  {scores['macro_f1']:>9.3f}")
    print(f"  {'─'*36}  {'─'*9}  {'─'*9}")
    label = f"SVM (train→dev)" if eval_csv else "SVM (80/20)"
    print(f"  {label:<36} {svm_res['accuracy']*100:>8.2f}%  {svm_res['macro_f1']:>9.3f}  ← new")
    label = f"MLP (train→dev)" if eval_csv else "MLP (80/20)"
    print(f"  {label:<36} {mlp_res['accuracy']*100:>8.2f}%  {mlp_res['macro_f1']:>9.3f}  ← new")
    print()

    # Save
    if artifacts_dir:
        save_artifacts(
            Path(artifacts_dir), svm, mlp, scaler,
            spk_means, global_fb, svm_res, mlp_res,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate F0-based Mandarin tone classifiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train_csv",     required=True,
                        help="F0 dataset for training")
    parser.add_argument("--eval_csv",      default=None,
                        help="F0 dataset for evaluation (omit for random split)")
    parser.add_argument("--test_size",     type=float, default=0.2,
                        help="Test fraction when --eval_csv is not used (default: 0.2)")
    parser.add_argument("--random_state",  type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--artifacts_dir", default=None,
                        help="Directory to save models, scaler, stats, reports")
    args = parser.parse_args()
    run(args.train_csv, args.eval_csv, args.test_size,
        args.random_state, args.artifacts_dir)


if __name__ == "__main__":
    main()
