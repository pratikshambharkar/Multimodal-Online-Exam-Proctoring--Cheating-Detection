# RAIM — ML training pipeline

Offline training scripts for the RAIM proctoring engines. All artifacts are
written to `ml_models/` (the directory `raim.settings.ML_MODELS_DIR` points at),
which is where the runtime engines (`proctoring/engine/`) load them from.

All scripts are absolute-path safe (they resolve their own locations with
`pathlib`, never `os.chdir`), accept `--data-root` / `--out` overrides, and
exit cleanly on Ctrl+C.

## Prerequisites

- The Python 3.11 venv at `venv\` with the project dependencies installed
  (TensorFlow, scikit-learn, pandas, matplotlib, OpenCV, ultralytics,
  mediapipe, deepface, Django).
- The datasets in the folder next to the project root:

```
..\DataSet\online exam cheating detection.v15i.folder\   (train/valid/test x cheating/normal)
..\DataSet\cheating online exam.v3i.yolov8\              (YOLOv8 set + data.yaml)
```

Run everything **from the project root** (`proctoring_system\`) so the venv
interpreter resolves without quoting the space-laden absolute paths:

```bat
venv\Scripts\python.exe ml_training\<script>.py
```

(Or activate first with `venv\Scripts\activate` and use plain `python`.)

## Order of operations

| # | Command | Produces | Approx. CPU runtime |
|---|---------|----------|---------------------|
| 1 | `venv\Scripts\python.exe ml_training\train_cnn.py` | `behavior_cnn.keras`, `cnn_metrics.json`, `cnn_training.png` | 10–25 min (6 epochs, ~1.5k images; add ~8 min with `--fine-tune`) |
| 2 | `venv\Scripts\python.exe ml_training\extract_features.py` | `svm_features.csv` | 20–45 min (~1.1k images through gaze + YOLO + CNN) |
| 3 | `venv\Scripts\python.exe ml_training\train_svm.py` | `svm_scaler.joblib`, `svm_fusion.joblib`, `svm_metrics.json`, `svm_confusion.png` | 1–3 min |
| optional | `venv\Scripts\python.exe ml_training\finetune_yolo.py` | `yolo_cheat.pt`, `yolo_data_abs.yaml`, run dir `yolo_finetune\` | 2–6 h (10 epochs @ 640 px on CPU; smoke-test with `--epochs 2`) |
| 4 | `venv\Scripts\python.exe ml_training\evaluate_models.py` | `evaluation_report.json` + paste-ready Section 5 tables | 3–8 min |

Step order matters: `extract_features.py` runs the app's own `BehaviorCNN`
engine, so the CNN must exist first (it warns loudly and writes
`cnn_cheat_prob = 0` otherwise), and `train_svm.py` consumes the CSV from
step 2. `evaluate_models.py` benchmarks whatever artifacts exist and skips
missing ones gracefully.

## Script notes

- **`train_cnn.py`** — MobileNetV2 (ImageNet, frozen) + GAP + Dropout(0.25) +
  2-way softmax; 160x160 RGB in [0, 1]; Adam 1e-3; augmentation (flip,
  rotation 0.04, brightness/contrast 0.15) on the training pipeline only.
  Datasets are always built from explicit `cheating/` + `normal/` file lists,
  so Roboflow's `Unlabeled/` directory can never leak in (empty ones are
  removed, non-empty ones are ignored with a warning).
  **Class order is a hard contract: softmax index 0 = cheating, 1 = normal.**
  `--fine-tune` unfreezes the last 30 backbone layers for 2 extra epochs at
  1e-4. Best checkpoint (by validation accuracy) is what gets saved and
  evaluated on the test split.
- **`extract_features.py`** — boots Django, then runs `GazeTracker`,
  `ObjectDetector` and `BehaviorCNN` over the train + valid splits
  (`--limit-per-class 400` per split, seeded shuffle) and writes the 10-dim
  contract feature vector per image plus `label` and `split` columns.
  Ctrl+C writes the partial CSV instead of losing progress.
- **`train_svm.py`** — RBF SVC (`probability=True`, `class_weight='balanced'`)
  with 5-fold `GridSearchCV` over `C=[1,10,100]`, `gamma=['scale',0.1,0.01]`
  on the train rows; the `StandardScaler` is fitted on train rows only.
  Saves the scaler and SVC as the two separate joblib artifacts
  `fusion.FusionEngine` loads. `predict_proba` column 0 = P(cheating).
- **`finetune_yolo.py`** — rewrites the set's relative-path `data.yaml` to an
  absolute-path copy, fine-tunes `yolov8n.pt` (2 classes: cheat/normal),
  copies `best.pt` to `yolo_cheat.pt` and prints validation mAP50 / mAP50-95.
  The app keeps using COCO `yolov8n.pt` unless the trunk setting
  `PROCTORING['YOLO_WEIGHTS']` is changed to `yolo_cheat.pt`.
- **`evaluate_models.py`** — consolidated manuscript numbers: CNN test-split
  metrics, SVM valid-split metrics, and per-module latency (mean + p95 over
  15 runs after 2 warmups) for `GazeTracker.analyze`,
  `ObjectDetector.detect`, `BehaviorCNN.predict`, `FaceVerifier.embed`, plus
  per-frame and with-re-verify pipeline sums.

## Artifacts in `ml_models/`

| File | Written by | Consumed by |
|------|------------|-------------|
| `behavior_cnn.keras` | train_cnn.py | `proctoring.engine.behavior_cnn` |
| `cnn_metrics.json`, `cnn_training.png` | train_cnn.py | manuscript |
| `svm_features.csv` | extract_features.py | train_svm.py / evaluate_models.py |
| `svm_scaler.joblib`, `svm_fusion.joblib` | train_svm.py | `proctoring.engine.fusion` |
| `svm_metrics.json`, `svm_confusion.png` | train_svm.py | manuscript |
| `yolo_cheat.pt`, `yolo_data_abs.yaml` | finetune_yolo.py | optional YOLO weights swap |
| `evaluation_report.json` | evaluate_models.py | manuscript (Section 5) |
