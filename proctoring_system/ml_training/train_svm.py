"""
RAIM - train_svm.py

Trains the RBF-SVM fusion classifier on the feature table produced by
extract_features.py and saves the two artifacts fusion.FusionEngine loads:

    ml_models/svm_scaler.joblib  - StandardScaler fitted on the train split only
    ml_models/svm_fusion.joblib  - best GridSearchCV SVC (probability=True)

Labels are the raw strings 'cheating'/'normal'; SVC sorts classes
alphabetically, so predict_proba column 0 = P(cheating).

Usage (from project root):
    venv\\Scripts\\python.exe ml_training\\train_svm.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "ml_models"

CLASS_NAMES = ["cheating", "normal"]

# Contract order - must match extract_features.py and
# proctoring.engine.fusion.FusionEngine.features_from_channels.
FEATURE_COLUMNS = [
    "face_count", "looking_away", "abs_yaw", "abs_pitch", "ear",
    "phone_conf", "book_conf", "laptop_conf", "extra_persons", "cnn_cheat_prob",
]

PARAM_GRID = {"C": [1, 10, 100], "gamma": ["scale", 0.1, 0.01]}


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the RAIM SVM fusion model.")
    parser.add_argument("--features", type=Path,
                        default=DEFAULT_OUT_DIR / "svm_features.csv",
                        help="Feature CSV from extract_features.py "
                             f"(default: {DEFAULT_OUT_DIR / 'svm_features.csv'})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Artifact output directory (default: {DEFAULT_OUT_DIR})")
    return parser.parse_args()


def plot_confusion(matrix, out_path: Path):
    """Save an annotated 2x2 confusion-matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("RAIM SVM fusion - validation confusion matrix")
    threshold = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black",
                    fontsize=13)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    started = time.perf_counter()

    features_path = args.features.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        print(f"ERROR: feature table not found: {features_path}\n"
              "Run extract_features.py first.")
        return 1

    import joblib
    import pandas as pd
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    frame = pd.read_csv(features_path)
    missing = [c for c in FEATURE_COLUMNS + ["label", "split"]
               if c not in frame.columns]
    if missing:
        print(f"ERROR: feature CSV is missing columns: {missing}")
        return 1

    train_rows = frame[frame["split"] == "train"]
    valid_rows = frame[frame["split"] == "valid"]
    if train_rows.empty or valid_rows.empty:
        print("ERROR: need both split=='train' and split=='valid' rows "
              f"(got {len(train_rows)} train / {len(valid_rows)} valid).")
        return 1

    x_train = train_rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_rows["label"].to_numpy()
    x_valid = valid_rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_valid = valid_rows["label"].to_numpy()
    print(f"Loaded {len(frame)} rows - train {len(x_train)}, valid {len(x_valid)}")

    # Scaler is fitted on the TRAIN split only, then applied everywhere -
    # exactly how fusion.py applies it at inference time.
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)

    print(f"Grid search (5-fold): C={PARAM_GRID['C']}, gamma={PARAM_GRID['gamma']}")
    search = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced",
            random_state=42),
        PARAM_GRID, cv=5, n_jobs=-1, verbose=1)
    search.fit(x_train_scaled, y_train)
    best_svc = search.best_estimator_

    print(f"\nBest params  : {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")
    print(f"Class order  : {list(best_svc.classes_)} "
          "(predict_proba column 0 = P(cheating))")

    y_pred = best_svc.predict(x_valid_scaled)
    accuracy = accuracy_score(y_valid, y_pred)
    report_text = classification_report(y_valid, y_pred, labels=CLASS_NAMES,
                                        digits=4, zero_division=0)
    report_dict = classification_report(y_valid, y_pred, labels=CLASS_NAMES,
                                        output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_valid, y_pred, labels=CLASS_NAMES)

    print("\nValidation classification report:")
    print(report_text)
    print("Confusion matrix [rows=true, cols=pred] (cheating, normal):")
    print(matrix)

    scaler_path = out_dir / "svm_scaler.joblib"
    model_path = out_dir / "svm_fusion.joblib"
    joblib.dump(scaler, scaler_path)
    joblib.dump(best_svc, model_path)

    confusion_path = out_dir / "svm_confusion.png"
    plot_confusion(matrix, confusion_path)

    elapsed = time.perf_counter() - started
    metrics = {
        "feature_columns": FEATURE_COLUMNS,
        "class_order": list(best_svc.classes_),
        "n_train": int(len(x_train)),
        "n_valid": int(len(x_valid)),
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "valid_accuracy": float(accuracy),
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "elapsed_seconds": round(elapsed, 1),
    }
    metrics_path = out_dir / "svm_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nSaved scaler          : {scaler_path}")
    print(f"Saved SVC             : {model_path}")
    print(f"Saved confusion plot  : {confusion_path}")
    print(f"Saved metrics         : {metrics_path}")
    print(f"Elapsed: {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user - no artifacts written.")
        sys.exit(130)
