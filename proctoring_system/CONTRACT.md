# RAIM — Module Interface Contract

Single source of truth for every module author. Read this fully before writing code.
The project trunk (settings, all models, root urls, admin) is ALREADY WRITTEN — do not modify
`raim/settings.py`, `raim/urls.py`, or any `models.py`. Import from them exactly as specified.

Project root: `proctoring_system/` (Django project package: `raim`).
Python: 3.11 venv at `venv/`. Windows paths. Django 5.x.

## Data models (already written — import, never redefine)

- `accounts.models.Profile` — user OneToOne `user`, `role` ('student'|'instructor'),
  `face_image` (ImageField), `face_embedding` (JSONField, 128-d list), `face_registered` (bool),
  property `is_instructor`.
- `exams.models.Exam` — `title, description, duration_minutes, is_active, created_by, created_at`,
  property `total_marks`, related `questions`.
- `exams.models.Question` — `exam, text, marks, order`, related `choices`.
- `exams.models.Choice` — `question, text, is_correct`.
- `exams.models.ExamSession` — `exam, student(User), started_at, submitted_at,
  status('in_progress'|'submitted'|'terminated'), score, total_marks, risk_score, peak_risk,
  violation_count, suspicious_count, tab_switches, trust_level('high'|'medium'|'low'),
  termination_reason, last_verify_at, next_verify_gap, gaze_calibrated(bool),
  gaze_calibration(JSON | null — envelope from pipeline.finalize_calibration, see below)`;
  properties `is_active, ends_at, percentage`; related `answers`, `events`, `frames`, `audio_chunks`.
- `exams.models.Answer` — `session, question, selected_choice, is_correct`; unique (session, question).
- `proctoring.models.MonitoringEvent` — `session, created_at, channel, event_type, severity
  ('info'|'suspicious'|'violation'), message, confidence, details(JSON), snapshot(ImageField)`.
- `proctoring.models.FrameAnalysis` — per-frame scores; fields: `face_count, identity_verified,
  identity_distance, yaw, pitch, roll, ear, looking_away, gaze_off_fraction, phone_conf, book_conf,
  extra_persons, cnn_cheat_prob, audio_speech_prob, svm_label, risk_score`.
- `proctoring.models.AudioAnalysis` — `session, rms, speech_prob, label`.

## Settings you must use

`from django.conf import settings` — `settings.PROCTORING` (dict of thresholds; see
`raim/settings.py` for keys), `settings.ML_MODELS_DIR` (Path), `settings.MEDIA_ROOT`.
LOGIN_URL='login', LOGIN_REDIRECT_URL='dashboard'.

## URL names (exact — templates and JS rely on these)

accounts (`accounts/urls.py`, no namespace):
- `''` → name='home'                (landing page, public)
- `'register/'` → name='register'
- `'login/'` → name='login'
- `'logout/'` → name='logout'       (POST)
- `'dashboard/'` → name='dashboard' (role-aware: student vs instructor)

exams (`exams/urls.py`, no namespace, mounted at /exams/):
- `''` → name='exam_list'
- `'<int:exam_id>/start/'` → name='exam_start'   (POST; creates ExamSession, redirects to exam_take)
- `'session/<int:session_id>/calibrate/'` → name='exam_calibrate' (gaze calibration; exam_take
  redirects here first when `not session.gaze_calibrated`; this view redirects to exam_take once
  `gaze_calibrated` is True, so the redirect can never loop)
- `'session/<int:session_id>/'` → name='exam_take'
- `'session/<int:session_id>/submit/'` → name='exam_submit' (POST)
- `'session/<int:session_id>/result/'` → name='exam_result'

proctoring (`proctoring/urls.py`, no namespace, mounted at /proctoring/):
- `'api/frame/'` → name='api_frame'   (POST JSON)
- `'api/audio/'` → name='api_audio'   (POST multipart)
- `'api/event/'` → name='api_event'   (POST JSON)
- `'api/status/<int:session_id>/'` → name='api_status' (GET; current risk + counts)
- `'api/calibration/sample/'` → name='api_calibration_sample' (POST JSON)
- `'api/calibration/finish/'` → name='api_calibration_finish' (POST JSON)

reports (`reports/urls.py`, no namespace, mounted at /reports/):
- `''` → name='report_list'                       (instructor: all sessions; student: own)
- `'session/<int:session_id>/'` → name='session_report'
- `'session/<int:session_id>/pdf/'` → name='session_report_pdf'

## Proctoring API JSON contracts

POST /proctoring/api/frame/   body: `{"session_id": int, "image": "data:image/jpeg;base64,..."}`
response 200:
```json
{"ok": true,
 "channels": {"face_count": 1, "identity_verified": true, "identity_distance": 0.31,
   "identity_checked": false, "yaw": 4.2, "pitch": -3.1, "roll": 1.0, "ear": 0.29,
   "looking_away": false, "gaze_off_fraction": 0.1, "phone_conf": 0.0, "book_conf": 0.0,
   "person_count": 1, "extra_persons": 0, "cnn_cheat_prob": 0.12, "audio_speech_prob": 0.05},
 "svm_label": "normal", "risk_score": 0.08, "risk_level": "normal",
 "violations": [{"channel": "objects", "event_type": "phone_detected",
                 "severity": "violation", "message": "Mobile phone detected (conf 0.87)"}],
 "terminate": false, "terminate_reason": ""}
```
POST /proctoring/api/audio/   multipart: `session_id`, `audio` (webm/ogg/wav blob)
response: `{"ok": true, "rms": 0.02, "speech_prob": 0.1, "label": "silence", "alert": false}`

POST /proctoring/api/event/   body: `{"session_id": int, "event_type": "tab_switch"|"window_blur"|
"fullscreen_exit"|"copy"|"paste"|"right_click"|"reload", "details": {}}`
response: `{"ok": true, "tab_switches": 3, "severity": "suspicious",
            "message": "...", "terminate": false}`

POST /proctoring/api/calibration/sample/   body: `{"session_id": int, "point_index": int,
"image": "data:image/jpeg;base64,..."}`  # one on-screen marker's frame; client sends several
response: `{"ok": true, "face_found": true, "samples_for_point": 3}`

POST /proctoring/api/calibration/finish/   body: `{"session_id": int}`
response: `{"ok": true, "calibrated": true, "sample_count": 41, "point_count": 8}`
# persists ExamSession.gaze_calibration (the envelope, or null if calibrated=false) and sets
# gaze_calibrated=True either way — a failed/skipped calibration never blocks the exam.

All APIs: login required, session must belong to request.user, session must be in_progress
(else `{"ok": false, "error": "...", "terminate": true}` if terminated). CSRF via X-CSRFToken header.

## Engine interfaces (proctoring/engine/) — lazy singletons

Heavy libraries (deepface, mediapipe, ultralytics, tensorflow) MUST be imported inside
methods/functions, never at module top level, so `manage.py migrate` works without them.
Every engine class must degrade gracefully: if its library or model file is missing, it returns
neutral results and sets `"available": False` in its output dict — never raises out of `analyze`.

- `face_verifier.FaceVerifier`
  - `.enroll(bgr) -> (embedding: list[float] | None, error: str)`  # exactly 1 face required
  - `.embed(bgr) -> list[float] | None`
  - `.verify(bgr, stored_embedding: list) -> dict(verified: bool, distance: float, face_found: bool)`
  - cosine distance vs settings.PROCTORING["FACE_DISTANCE_THRESHOLD"]; DeepFace model
    settings.PROCTORING["FACE_MODEL"], detector backend settings.PROCTORING["FACE_DETECTOR"].
- `gaze_tracker.GazeTracker.analyze(bgr, calibration: dict | None = None) -> dict(available,
  face_count, yaw, pitch, roll, ear, gaze_h, gaze_v, looking_away: bool, eyes_closed: bool)`
  # MediaPipe FaceMesh + solvePnP + EAR. `looking_away` uses the global YAW_MAX_DEG/PITCH_MIN_DEG/
  # fixed gaze_h band by default; when `calibration` (an ExamSession.gaze_calibration envelope:
  # yaw_min/max, pitch_min/max, gaze_h_min/max, gaze_v_min/max, any of which may be null) is
  # given, each axis is checked against its calibrated bounds and only falls back to the global
  # default when that axis's bounds are null. Every other caller (e.g. ml_training/extract_features.py)
  # omits `calibration` and gets the unchanged global-threshold behavior.
- `object_detector.ObjectDetector.detect(bgr) -> dict(available, person_count, phone_conf,
  book_conf, laptop_conf, tv_conf, detections: [{"label","conf","box":[x1,y1,x2,y2]}])`
  # ultralytics YOLO, weights settings.ML_MODELS_DIR / settings.PROCTORING["YOLO_WEIGHTS"]
- `behavior_cnn.BehaviorCNN.predict(bgr) -> dict(available, cheat_prob: float)`
  # Keras model at settings.ML_MODELS_DIR / "behavior_cnn.keras"; input 160x160 RGB [0,1]
  # class order: index 0 = cheating, 1 = normal (folder alphabetical); cheat_prob = P(cheating)
- `audio_analyzer.AudioAnalyzer.analyze_file(path) -> dict(available, rms, speech_prob, label)`
  # librosa: RMS energy + spectral features; label in silence|noise|speech|conversation
- `fusion.FusionEngine`
  - `.features_from_channels(ch: dict) -> list[float]` — ORDER (10 dims, must match training):
    `[face_count, looking_away(0/1), abs(yaw), abs(pitch), ear, phone_conf, book_conf,
      laptop_conf, extra_persons, cnn_cheat_prob]`
  - `.classify(ch) -> (label: 'normal'|'suspicious'|'violation', svm_prob: float)`
    # loads ml_models/svm_fusion.joblib + svm_scaler.joblib; SVM is binary (cheating/normal);
    # label mapping: P(cheat) >= 0.75 → violation, >= 0.45 → suspicious, else normal;
    # if model missing → rule-based fallback with same thresholds on weighted score.
  - `.channel_scores(ch, browser_score: float) -> dict` — per-channel s_i in [0,1]
  - `.update_risk(prev_risk, scores: dict) -> float` — EWMA with settings weights + RISK_LAMBDA
- `pipeline.get_pipeline() -> ProctoringPipeline` (module-level singleton, thread-safe)
  - `.analyze_frame(bgr, session: ExamSession, force_verify: bool=False, audio_speech_prob: float=0.0,
    browser_score: float|None=None) -> dict` — runs gaze (passing `session.gaze_calibration` through
    to GazeTracker), objects, CNN every frame; DeepFace re-verify only when due (session.last_verify_at
    + next_verify_gap elapsed, or force_verify); returns the full api_frame response dict (views
    persist FrameAnalysis/MonitoringEvent rows — pipeline itself does NOT write DB).
  - `.record_calibration_sample(session, point_index: int, bgr) -> dict(face_found: bool,
    samples_for_point: int)` — stages one gaze-calibration sample in memory for the given marker;
    used by api_calibration_sample. No DB writes.
  - `.finalize_calibration(session) -> dict(calibrated: bool, envelope: dict|None, sample_count: int,
    point_count: int)` — reduces the session's staged per-marker samples to a gaze envelope (per
    CALIBRATION_MIN_SAMPLES_PER_POINT / CALIBRATION_MIN_POINTS / CALIBRATION_MARGIN_* in
    settings.PROCTORING) and clears the staging data; the caller persists `envelope` onto
    `ExamSession.gaze_calibration`. No DB writes.

## Template conventions

- `templates/base.html` defines blocks: `title`, `content`, `extra_head`, `extra_js`.
  Navbar links by URL name: home, exam_list, report_list, dashboard, login/register or logout.
  Uses `{% static 'css/style.css' %}`. Shows `messages` framework alerts. Footer credit:
  "RAIM — Remote Assessment Integrity Monitoring".
- Auth pages POST multipart with hidden `face_data` input (JPEG dataURL from webcam canvas).
- `exam_take.html` loads `{% static 'js/proctor.js' %}` and boots:
  `Proctor.init({sessionId, frameUrl, audioUrl, eventUrl, statusUrl, frameIntervalMs, audioChunkMs, submitFormId})`.
- `calibration.html` (rendered by `exam_calibrate`, reached before `exam_take` whenever
  `not session.gaze_calibrated`) loads `{% static 'js/calibration.js' %}` and boots:
  `Calibration.init({sessionId, sampleUrl, finishUrl, nextUrl})`. Always resolves to
  `nextUrl` (exam_take), even on camera failure or an explicit skip.
- Colors/branding: dark navy + teal accent, clean academic look. All CSS in
  `static/css/style.css` (no CDN frameworks; hand-rolled, responsive).

## Coding standards

- Python: PEP8-ish, docstrings on public functions, no dead code, no TODOs.
- Never import cv2/numpy at Django import time in views — import inside functions OR at top of
  engine modules only (engine modules are themselves imported lazily by pipeline).
  Exception: `numpy` and `cv2` may be imported at top of engine modules.
- JS: vanilla ES6, no external libraries. Charts hand-rolled on <canvas>.
- All user-facing dates via Django template filters; timezone Asia/Kolkata.
