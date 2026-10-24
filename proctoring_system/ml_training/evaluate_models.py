"""
RAIM - evaluate_models.py

Consolidated evaluation for the manuscript (Section 5):
  (a) behaviour CNN on the held-out test split,
  (b) SVM fusion model on the valid rows of svm_features.csv,
  (c) per-module latency benchmark (mean + p95 over 15 timed runs after
      2 warmups) for GazeTracker.analyze, ObjectDetector.detect,
      BehaviorCNN.predict and FaceVerifier.embed,
  (d) ml_models/evaluation_report.json + clean paste-ready tables.

Usage (from project root):
    venv\\Scripts\\python.exe ml_training\\evaluate_models.py
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Django bootstrap: engines read settings.PROCTORING / ML_MODELS_DIR ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "raim.settings")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import django  # noqa: E402

django.setup()

DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "DataSet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ml_models"
BINARY_SET = "online exam cheating detection.v15i.folder"

IMG_SIZE = 160
BATCH_SIZE = 32
CLASS_NAMES = ["cheating", "normal"]  # index 0 = cheating (app contract)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Contract order - must match extract_features.py / train_svm.py / fusion.py.
FEATURE_COLUMNS = [
    "face_count", "looking_away", "abs_yaw", "abs_pitch", "ear",
    "phone_conf", "book_conf", "laptop_conf", "extra_persons", "cnn_cheat_prob",
]

RULE = "-" * 72


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="RAIM consolidated evaluation.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Directory containing the dataset folders "
                             f"(default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Artifact output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--features", type=Path,
                        default=DEFAULT_OUT_DIR / "svm_features.csv",
                        help="Feature CSV from extract_features.py")
    parser.add_argument("--runs", type=int, default=15,
                        help="Timed runs per module in the latency benchmark "
                             "(default: 15)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup runs per module (default: 2)")
    parser.add_argument("--sample-image", type=Path, default=None,
                        help="Image for the latency benchmark (default: first "
                             "test/cheating image of the binary set)")
    return parser.parse_args()


def evaluate_cnn(data_dir: Path, model_path: Path, out_dir: Path):
    """(a) Report the served CNN's held-out test performance.

    train_cnn.py supports two split protocols (``--split-mode pooled``, the
    default: merge every folder and stratify-resplit 70/15/15; or
    ``folder``: use the Roboflow train/valid/test folders as-is) and already
    evaluates the SAVED checkpoint on whichever test set that run actually
    held out, writing the result to ``cnn_metrics.json``. This function
    reuses that report rather than recomputing its own, because recomputing
    against the raw ``test/`` folder would silently leak pooled-mode training
    images into the "test" set - the model would score itself on data it was
    partly trained on and produce inflated, wrong numbers for the paper.
    """
    metrics_path = out_dir / "cnn_metrics.json"
    if not metrics_path.exists():
        print(f"cnn_metrics.json not found ({metrics_path}) - run train_cnn.py "
              "first. Skipping CNN evaluation.")
        return {"available": False, "reason": f"missing {metrics_path.name}"}

    with open(metrics_path, encoding="utf-8") as fh:
        saved = json.load(fh)
    if not saved.get("classification_report"):
        return {"available": False, "reason": "cnn_metrics.json has no report"}

    split_mode = saved.get("split_mode", "folder")
    print(f"Test split: {saved['counts']['test']} images "
          f"(split_mode={split_mode}) - reusing train_cnn.py's own held-out "
          f"evaluation of the served checkpoint (see split_note below).")
    if saved.get("split_note"):
        print(f"  note: {saved['split_note']}")

    return {
        "available": True,
        "split_mode": split_mode,
        "split_note": saved.get("split_note", ""),
        "n_test": saved["counts"]["test"],
        "accuracy": float(saved["test_accuracy"]),
        "classification_report": saved["classification_report"],
        "confusion_matrix": saved["confusion_matrix"],
    }


def evaluate_svm(features_path: Path, out_dir: Path):
    """(b) Score the saved scaler + SVC on the valid rows of the feature CSV."""
    scaler_path = out_dir / "svm_scaler.joblib"
    model_path = out_dir / "svm_fusion.joblib"
    missing = [p.name for p in (features_path, scaler_path, model_path)
               if not p.exists()]
    if missing:
        print(f"SVM artifacts missing ({', '.join(missing)}) - skipping "
              "SVM evaluation.")
        return {"available": False, "reason": f"missing {', '.join(missing)}"}

    import joblib
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    frame = pd.read_csv(features_path)
    valid_rows = frame[frame["split"] == "valid"]
    if valid_rows.empty:
        print("Feature CSV has no valid-split rows - skipping SVM evaluation.")
        return {"available": False, "reason": "no valid rows in feature CSV"}

    scaler = joblib.load(scaler_path)
    svc = joblib.load(model_path)
    x_valid = scaler.transform(valid_rows[FEATURE_COLUMNS].to_numpy(dtype=float))
    y_valid = valid_rows["label"].to_numpy()
    y_pred = svc.predict(x_valid)

    report = classification_report(y_valid, y_pred, labels=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_valid, y_pred, labels=CLASS_NAMES)
    return {
        "available": True,
        "n_valid": int(len(y_valid)),
        "accuracy": float(report["accuracy"]),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def pick_sample_image(data_dir: Path, explicit: Path | None) -> Path | None:
    """Sample frame for the latency benchmark."""
    if explicit is not None:
        return explicit if explicit.exists() else None
    for split in ("test", "valid", "train"):
        for class_name in CLASS_NAMES:
            class_dir = data_dir / split / class_name
            if class_dir.is_dir():
                files = sorted(
                    p for p in class_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
                if files:
                    return files[0]
    return None


def benchmark_latency(sample_path: Path, runs: int, warmup: int):
    """(c) Per-module mean/p95 latency over `runs` timed calls."""
    import cv2
    import numpy as np
    from proctoring.engine.behavior_cnn import BehaviorCNN
    from proctoring.engine.face_verifier import FaceVerifier
    from proctoring.engine.gaze_tracker import GazeTracker
    from proctoring.engine.object_detector import ObjectDetector

    runs = max(1, runs)
    warmup = max(0, warmup)
    bgr = cv2.imread(str(sample_path))
    if bgr is None:
        print(f"Could not read sample image {sample_path} - skipping benchmark.")
        return {"available": False, "reason": "unreadable sample image"}

    gaze = GazeTracker()
    detector = ObjectDetector()
    cnn = BehaviorCNN()
    verifier = FaceVerifier()

    modules = [
        ("GazeTracker.analyze", lambda: gaze.analyze(bgr)),
        ("ObjectDetector.detect", lambda: detector.detect(bgr)),
        ("BehaviorCNN.predict", lambda: cnn.predict(bgr)),
        ("FaceVerifier.embed", lambda: verifier.embed(bgr)),
    ]

    results = {}
    for name, call in modules:
        for _ in range(warmup):
            call()
        timings = []
        for _ in range(runs):
            t0 = time.perf_counter()
            output = call()
            timings.append((time.perf_counter() - t0) * 1000.0)
        available = (output.get("available", True) if isinstance(output, dict)
                     else output is not None)
        results[name] = {
            "mean_ms": float(np.mean(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "runs": runs,
            "available": bool(available),
        }
        print(f"  {name:24s} mean {results[name]['mean_ms']:8.1f} ms   "
              f"p95 {results[name]['p95_ms']:8.1f} ms   "
              f"available={results[name]['available']}")

    per_frame = ["GazeTracker.analyze", "ObjectDetector.detect",
                 "BehaviorCNN.predict"]
    return {
        "available": True,
        "sample_image": str(sample_path),
        "frame_shape": list(bgr.shape),
        "modules": results,
        # Every analysed frame runs gaze + objects + CNN; DeepFace re-verify
        # is added only on the randomized re-verification cadence.
        "pipeline_per_frame_ms": {
            "mean": float(sum(results[m]["mean_ms"] for m in per_frame)),
            "p95": float(sum(results[m]["p95_ms"] for m in per_frame)),
        },
        "pipeline_with_reverify_ms": {
            "mean": float(sum(r["mean_ms"] for r in results.values())),
            "p95": float(sum(r["p95_ms"] for r in results.values())),
        },
    }


def print_classifier_table(title: str, result: dict, class_names):
    """Paste-ready per-class metrics table for the paper."""
    print(f"\n{title}")
    print(RULE)
    if not result.get("available"):
        print(f"  unavailable - {result.get('reason', 'unknown')}")
        return
    report = result["classification_report"]
    print(f"{'Class':<12}{'Precision':>11}{'Recall':>11}{'F1-score':>11}"
          f"{'Support':>10}")
    for name in class_names:
        row = report[name]
        print(f"{name:<12}{row['precision']:>11.4f}{row['recall']:>11.4f}"
              f"{row['f1-score']:>11.4f}{int(row['support']):>10d}")
    macro = report["macro avg"]
    print(f"{'macro avg':<12}{macro['precision']:>11.4f}{macro['recall']:>11.4f}"
          f"{macro['f1-score']:>11.4f}{int(macro['support']):>10d}")
    print(f"{'Accuracy':<12}{result['accuracy']:>11.4f}")
    matrix = result["confusion_matrix"]
    print(f"Confusion [rows=true, cols=pred] ({', '.join(class_names)}): {matrix}")
    if result.get("split_note"):
        print(f"Protocol note: {result['split_note']}")


def print_latency_table(latency: dict):
    """Paste-ready latency table for the paper."""
    print("\nTable - Per-module inference latency (CPU)")
    print(RULE)
    if not latency.get("available"):
        print(f"  unavailable - {latency.get('reason', 'unknown')}")
        return
    print(f"{'Module':<26}{'Mean (ms)':>12}{'P95 (ms)':>12}{'Available':>11}")
    for name, row in latency["modules"].items():
        print(f"{name:<26}{row['mean_ms']:>12.1f}{row['p95_ms']:>12.1f}"
              f"{str(row['available']):>11}")
    per_frame = latency["pipeline_per_frame_ms"]
    full = latency["pipeline_with_reverify_ms"]
    print(RULE)
    print(f"{'Per-frame pipeline sum':<26}{per_frame['mean']:>12.1f}"
          f"{per_frame['p95']:>12.1f}")
    print(f"{'With identity re-verify':<26}{full['mean']:>12.1f}"
          f"{full['p95']:>12.1f}")


def main():
    args = parse_args()
    started = time.perf_counter()

    data_dir = (args.data_root / BINARY_SET).resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("RAIM consolidated evaluation")
    print(f"Dataset : {data_dir}")
    print(f"Output  : {out_dir}")

    print("\n(a) Behaviour CNN - test split")
    cnn_result = evaluate_cnn(data_dir, out_dir / "behavior_cnn.keras", out_dir)

    print(f"\n(b) SVM fusion - valid rows of {args.features.name}")
    svm_result = evaluate_svm(args.features.resolve(), out_dir)

    print(f"\n(c) Latency benchmark - {args.runs} runs after {args.warmup} warmups")
    sample = pick_sample_image(data_dir, args.sample_image)
    if sample is None:
        latency_result = {"available": False, "reason": "no sample image found"}
        print("No sample image found - skipping benchmark.")
    else:
        print(f"Sample image: {sample}")
        latency_result = benchmark_latency(sample, args.runs, args.warmup)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cnn_test": cnn_result,
        "svm_valid": svm_result,
        "latency": latency_result,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    report_path = out_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SECTION 5 TABLES (paste-ready)")
    print("=" * 72)
    print_classifier_table("Table - Behaviour CNN, held-out test split",
                           cnn_result, CLASS_NAMES)
    print_classifier_table("Table - SVM fusion, validation split",
                           svm_result, CLASS_NAMES)
    print_latency_table(latency_result)

    print(f"\nSaved consolidated report: {report_path}")
    print(f"Elapsed: {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user - report not written.")
        sys.exit(130)
