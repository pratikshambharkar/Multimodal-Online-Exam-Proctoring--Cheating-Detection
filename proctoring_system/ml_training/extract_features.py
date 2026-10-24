"""
RAIM - extract_features.py

Builds the SVM fusion training table by running the app's OWN engine modules
(GazeTracker, ObjectDetector, BehaviorCNN) over the labelled binary dataset
(train + valid splits) and writing the 10-dim contract feature vector per
image to ml_models/svm_features.csv.

Feature order (MUST match proctoring.engine.fusion.FusionEngine):
    [face_count, looking_away, abs(yaw), abs(pitch), ear,
     phone_conf, book_conf, laptop_conf, extra_persons, cnn_cheat_prob]

Run train_cnn.py first so behavior_cnn.keras exists - otherwise the
cnn_cheat_prob column is 0 for every row (a loud warning is printed).

Usage (from project root):
    venv\\Scripts\\python.exe ml_training\\extract_features.py [--limit-per-class 400]
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

# --- Django bootstrap: the engines read settings.PROCTORING / ML_MODELS_DIR ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "raim.settings")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import django  # noqa: E402

django.setup()

DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "DataSet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ml_models"
BINARY_SET = "online exam cheating detection.v15i.folder"

SPLITS = ("train", "valid")
CLASS_NAMES = ("cheating", "normal")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42

# Contract order - fusion.FusionEngine.features_from_channels and train_svm.py
# consume these columns in exactly this order.
FEATURE_COLUMNS = [
    "face_count", "looking_away", "abs_yaw", "abs_pitch", "ear",
    "phone_conf", "book_conf", "laptop_conf", "extra_persons", "cnn_cheat_prob",
]


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract SVM fusion features with the RAIM engine modules.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Directory containing the dataset folders "
                             f"(default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Artifact output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--limit-per-class", type=int, default=400,
                        help="Max images sampled per class within each split "
                             "(default: 400; sampled with seed 42)")
    return parser.parse_args()


def collect_images(data_dir: Path, limit_per_class: int):
    """List (path, label, split) tuples: shuffled with seed 42, then limited."""
    rng = random.Random(SEED)
    items = []
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            class_dir = data_dir / split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            files = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            rng.shuffle(files)
            selected = files[:limit_per_class]
            print(f"  {split}/{class_name}: {len(selected)} of {len(files)} images")
            items.extend((path, class_name, split) for path in selected)
    rng.shuffle(items)  # interleave classes/splits for honest progress ETAs
    return items


def check_engine_availability(gaze, detector, cnn):
    """Probe each engine once and warn loudly about degraded channels."""
    import numpy as np
    from django.conf import settings

    probe = np.full((240, 320, 3), 128, dtype=np.uint8)
    warnings = []

    cnn_weights = Path(settings.ML_MODELS_DIR) / "behavior_cnn.keras"
    if not cnn_weights.exists():
        warnings.append(
            f"behavior_cnn.keras not found at {cnn_weights} - "
            "cnn_cheat_prob will be 0 for EVERY row. Run train_cnn.py first "
            "for a meaningful fusion model.")

    if not gaze.analyze(probe).get("available", False):
        warnings.append(
            "GazeTracker unavailable (mediapipe missing?) - face_count, "
            "looking_away, abs_yaw, abs_pitch, ear will be neutral.")
    if not detector.detect(probe).get("available", False):
        warnings.append(
            "ObjectDetector unavailable (ultralytics/weights missing?) - "
            "phone_conf, book_conf, laptop_conf, extra_persons will be neutral.")
    if not cnn.predict(probe).get("available", False):
        warnings.append(
            "BehaviorCNN unavailable - cnn_cheat_prob will be 0.")

    if warnings:
        print("\n" + "!" * 78)
        for warning in warnings:
            print(f"! WARNING: {warning}")
        print("!" * 78 + "\n")
    else:
        print("All three engines report available=True.\n")


def extract_row(bgr, gaze, detector, cnn):
    """Assemble the 10-dim contract feature vector for one frame."""
    g = gaze.analyze(bgr)
    o = detector.detect(bgr)
    c = cnn.predict(bgr)
    return {
        "face_count": float(g.get("face_count", 0) or 0),
        "looking_away": 1.0 if g.get("looking_away") else 0.0,
        "abs_yaw": abs(float(g.get("yaw", 0.0) or 0.0)),
        "abs_pitch": abs(float(g.get("pitch", 0.0) or 0.0)),
        "ear": float(g.get("ear", 0.0) or 0.0),
        "phone_conf": float(o.get("phone_conf", 0.0) or 0.0),
        "book_conf": float(o.get("book_conf", 0.0) or 0.0),
        "laptop_conf": float(o.get("laptop_conf", 0.0) or 0.0),
        "extra_persons": float(max(0, int(o.get("person_count", 0) or 0) - 1)),
        "cnn_cheat_prob": float(c.get("cheat_prob", 0.0) or 0.0),
    }


def main():
    args = parse_args()
    started = time.perf_counter()

    data_dir = (args.data_root / BINARY_SET).resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "svm_features.csv"

    print(f"Dataset : {data_dir}")
    print(f"Output  : {csv_path}")
    print("Collecting image lists (train + valid):")
    items = collect_images(data_dir, args.limit_per_class)
    print(f"Total images queued: {len(items)}")

    import cv2
    import pandas as pd
    from proctoring.engine.behavior_cnn import BehaviorCNN
    from proctoring.engine.gaze_tracker import GazeTracker
    from proctoring.engine.object_detector import ObjectDetector

    gaze = GazeTracker()
    detector = ObjectDetector()
    cnn = BehaviorCNN()
    check_engine_availability(gaze, detector, cnn)

    rows = []
    skipped = 0
    interrupted = False
    try:
        for index, (path, label, split) in enumerate(items, start=1):
            bgr = cv2.imread(str(path))
            if bgr is None:
                skipped += 1
                continue
            row = extract_row(bgr, gaze, detector, cnn)
            row["label"] = label
            row["split"] = split
            rows.append(row)
            if index % 50 == 0:
                rate = index / max(time.perf_counter() - started, 1e-6)
                print(f"  [{index:5d}/{len(items)}] processed "
                      f"(skipped {skipped}, {rate:.1f} img/s)")
    except KeyboardInterrupt:
        interrupted = True
        print(f"\nInterrupted after {len(rows)} rows - writing partial CSV.")

    if not rows:
        print("ERROR: no rows extracted; nothing written.")
        return 1

    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["label", "split"])
    frame.to_csv(csv_path, index=False)

    print(f"\nWrote {len(frame)} rows to {csv_path} "
          f"({skipped} unreadable images skipped)")
    print("\nClass balance (rows per split/label):")
    print(frame.groupby(["split", "label"]).size().to_string())
    print("\nFeature means by label:")
    print(frame.groupby("label")[FEATURE_COLUMNS].mean().round(4).to_string())
    print(f"\nElapsed: {(time.perf_counter() - started) / 60:.1f} min")
    return 130 if interrupted else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
