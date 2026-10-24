"""
Django settings for RAIM — Remote Assessment Integrity Monitoring.

A Multimodal Framework for Real-Time Cheating Detection
Using DeepFace, CNN, and SVM.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    "RAIM_SECRET_KEY",
    "django-insecure-raim-dev-key-change-me-in-production",
)

DEBUG = os.environ.get("RAIM_DEBUG", "1") == "1"

ALLOWED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0", "testserver"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # RAIM apps
    "accounts",
    "exams",
    "proctoring",
    "reports",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "raim.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "raim.context_processors.asset_version",
            ],
        },
    },
]

WSGI_APPLICATION = "raim.wsgi.application"

# Database — SQLite by default (zero-setup for development).
# Set RAIM_DB=postgres plus the RAIM_PG_* variables to use PostgreSQL.
if os.environ.get("RAIM_DB", "sqlite") == "postgres":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.environ.get("RAIM_PG_NAME", "raim"),
            "USER": os.environ.get("RAIM_PG_USER", "postgres"),
            "PASSWORD": os.environ.get("RAIM_PG_PASSWORD", ""),
            "HOST": os.environ.get("RAIM_PG_HOST", "localhost"),
            "PORT": os.environ.get("RAIM_PG_PORT", "5432"),
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "dashboard"
LOGOUT_REDIRECT_URL = "home"

# Where trained model artifacts live (CNN weights, SVM, YOLO weights).
ML_MODELS_DIR = BASE_DIR / "ml_models"

# ---------------------------------------------------------------------------
# Proctoring engine configuration (Section 3 of the manuscript)
# ---------------------------------------------------------------------------
PROCTORING = {
    # -- Identity verification (DeepFace, Sec. 3.2) --
    "FACE_MODEL": "Facenet",            # DeepFace backbone (128-d embeddings)
    "FACE_DETECTOR": "mtcnn",           # detector backend (reliable on webcams; ~1s CPU)
    "FACE_DISTANCE_THRESHOLD": 0.40,    # cosine tau at login/enroll (frontal pose)
    # Continuous re-verification sees candid poses (head turned, motion blur),
    # so it uses a softer tau. Calibrated on OEP subjects: tau=0.40 -> FRR 33%,
    # tau=0.55 -> FAR 13% / FRR 11%; combined with the 2-consecutive-failures
    # termination rule this suppresses false alarms (manuscript Sec. 3.2).
    "FACE_REVERIFY_THRESHOLD": 0.55,
    "REVERIFY_MIN_SECONDS": 60,         # randomized continuous re-verification
    "REVERIFY_MAX_SECONDS": 150,        # (paper: 3-7 min; shortened for demo)

    # -- Gaze / head pose (MediaPipe, Sec. 3.3) --
    "YAW_MAX_DEG": 28.0,                # global default: |yaw| beyond this = looking away
    "PITCH_MIN_DEG": -22.0,             # global default: pitch below this = downward gaze
    "EAR_CLOSED": 0.16,                 # eye-aspect-ratio: eyes closed
    "GAZE_WINDOW_SECONDS": 10,          # sliding window W
    "GAZE_OFF_FRACTION": 0.6,           # fraction of off-screen frames to flag
    # Frames arrive only every FRAME_INTERVAL_MS (2.5s default), so a 10s
    # window holds just ~4 samples - one or two glance-frames alone can
    # already cross GAZE_OFF_FRACTION. The gaze_off_screen alert additionally
    # requires the away-gaze to have persisted continuously for longer than
    # this, so a brief look-away never raises it, only a sustained one does.
    "GAZE_AWAY_MIN_SECONDS": 3.0,

    # -- Per-user gaze calibration (screen-boundary look-around) --
    # Before the exam, the candidate looks at markers placed at the screen's
    # corners/edges/center; the observed yaw/pitch/iris-position range at
    # those markers becomes a personalized "on-screen" envelope, replacing
    # the two global thresholds above for that session. This corrects for
    # webcam position (above/below/beside the screen) and individual eye
    # geometry, which a single global threshold cannot. When calibration is
    # unavailable or a specific axis has too few samples, that axis falls
    # back to the global default above (see GazeTracker.analyze).
    "CALIBRATION_MIN_SAMPLES_PER_POINT": 2,   # samples needed to trust a point
    "CALIBRATION_MIN_POINTS": 5,              # of 9 possible on-screen markers
    "CALIBRATION_MARGIN_YAW_DEG": 4.0,        # padding beyond the observed extremes
    "CALIBRATION_MARGIN_PITCH_DEG": 4.0,
    "CALIBRATION_MARGIN_GAZE": 0.05,

    # -- Object detection (YOLOv8, Sec. 3.4) --
    "YOLO_WEIGHTS": "yolov8n.pt",       # COCO-pretrained nano model
    "YOLO_CONF": 0.35,
    "PHONE_ALERT_CONF": 0.5,
    "PERSON_DEBOUNCE_FRAMES": 3,        # extra person must persist N analyzed frames

    # -- Audio (Sec. 3.5) --
    "AUDIO_RMS_LOUD": 0.06,             # normalized RMS considered "loud"
    "AUDIO_SPEECH_PROB_ALERT": 0.65,

    # -- Browser events (Sec. 3.6) --
    "TAB_SWITCH_WARN": 2,               # warnings before violation
    "TAB_SWITCH_TERMINATE": 5,          # auto-terminate threshold

    # -- Fusion / risk score (Sec. 3.7) --
    "RISK_LAMBDA": 0.82,                # EWMA decay lambda
    "RISK_SUSPICIOUS": 0.45,
    "RISK_VIOLATION": 0.70,
    "CHANNEL_WEIGHTS": {                # w_i for the accumulated risk score
        "identity": 0.24,
        "gaze": 0.16,
        "objects": 0.22,
        "cnn": 0.14,
        "audio": 0.12,
        "browser": 0.12,
    },

    # -- Frame analysis cadence --
    "FRAME_INTERVAL_MS": 2500,          # client sends one frame every N ms
    "AUDIO_CHUNK_MS": 10000,            # client sends one audio chunk every N ms
}
