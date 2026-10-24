# RAIM — Remote Assessment Integrity Monitoring

**A Multimodal Framework for Real-Time Cheating Detection Using DeepFace, CNN, and SVM**

RAIM is a Django-based online-exam proctoring system implementing the architecture from the
accompanying manuscript: five concurrent detection channels fused into a single real-time
risk score by a support vector machine, running entirely on commodity CPU hardware.

## Detection channels

| Channel | Technology | What it catches |
|---|---|---|
| Identity verification | DeepFace (Facenet embeddings, cosine distance) | Impersonation, candidate swap (continuous re-verification at randomized intervals) |
| Eye-gaze & head pose | MediaPipe FaceMesh + solvePnP + EAR | Sustained off-screen / downward gaze, absent face, multiple faces |
| Object detection | YOLOv8 (COCO) | Mobile phones, books, laptops, extra persons in frame |
| Behavior CNN | MobileNetV2 transfer-learned on exam-cheating dataset | Visual cheating posture/behavior patterns |
| Audio analysis | MFCC / spectral features on mic chunks | Background speech and conversation |
| Browser activity | JS event capture | Tab switching, window blur, fullscreen exit, copy/paste |

Per-channel anomaly scores are fused by an RBF-kernel **SVM** (trained on features extracted
from the labeled dataset) and accumulated into an EWMA session risk score
`R_t = λR_{t-1} + (1-λ) Σ w_i s_i` with a three-tier alert policy
(normal → suspicious → violation) and automatic termination rules.

## Quick start

The Python 3.11 virtual environment lives at **`C:\Users\harsh\raim_venv`** — a short
path outside OneDrive, because the deep library trees (torch, TensorFlow) exceed the
Windows 260-character path limit inside this project folder.

```powershell
cd proctoring_system
$py = "C:\Users\harsh\raim_venv\Scripts\python.exe"

# 1. Dependencies (already installed during setup)
& $py -m pip install -r requirements.txt

# 2. Database + demo data
& $py manage.py migrate
& $py manage.py seed_demo

# 3. Train the ML models (order matters — see ml_training/README.md)
& $py ml_training\train_cnn.py
& $py ml_training\extract_features.py
& $py ml_training\train_svm.py

# 4. Run
& $py manage.py runserver
```

Or simply double-click / run the helper scripts: `run_server.bat`, `train_models.bat`.

Open http://127.0.0.1:8000/ — register a student (webcam face enrollment required),
log in (face + password), start an exam from the dashboard.

Demo accounts created by `seed_demo`:
- Instructor: `instructor` / `instructor123` (no face enrolled → password-only login)
- Student: `student1` / `student123` (must enroll a face before taking exams — or register a fresh account)

> The system works even before the CNN/SVM are trained: the fusion engine falls back to a
> calibrated rule-based score, and the CNN channel reports neutral values. Train the models
> to enable the full learned pipeline.

## Project layout

```
proctoring_system/
├── raim/                 # Django project (settings incl. PROCTORING thresholds)
├── accounts/             # Registration/login with face enrollment + role dashboards
├── exams/                # Exams, questions, sessions, grading, seed_demo command
├── proctoring/           # Monitoring API + AI engine
│   └── engine/           # face_verifier, gaze_tracker, object_detector,
│                         # behavior_cnn, audio_analyzer, fusion, pipeline
├── reports/              # HTML + PDF integrity reports
├── ml_training/          # CNN / SVM / YOLO training + evaluation scripts
├── ml_models/            # Trained artifacts (created by the training scripts)
├── templates/, static/   # UI (vanilla JS proctoring client in static/js/proctor.js)
└── media/                # Enrolled faces, evidence snapshots, audio chunks
```

## Datasets used (from `../DataSet/`)

- `online exam cheating detection.v15i.folder` — binary cheating/normal images:
  CNN training and SVM feature extraction.
- `cheating online exam.v3i.yolov8` — YOLOv8-format detection set (optional fine-tune
  via `ml_training/finetune_yolo.py`).
- `OEP database` (Atoum et al., IEEE TMM) — audio/video for audio-channel testing.

## Database

SQLite by default (zero setup). For PostgreSQL set environment variables:
`RAIM_DB=postgres`, `RAIM_PG_NAME`, `RAIM_PG_USER`, `RAIM_PG_PASSWORD`, `RAIM_PG_HOST`, `RAIM_PG_PORT`.

## Notes

- First registration downloads Facenet weights (~90 MB) to `~/.deepface`; first frame
  analysis downloads `yolov8n.pt` (~6 MB). Internet is required once.
- All processing is local; only the browser ↔ local server exchange frames/audio.
- Proctoring thresholds (gaze windows, risk weights, tab-switch limits, re-verify
  intervals) are configurable in `raim/settings.py` → `PROCTORING`.
