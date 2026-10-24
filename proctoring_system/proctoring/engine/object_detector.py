"""Prohibited-object and person detection via YOLOv8 (manuscript Sec. 3.4).

Detects persons, cell phones, books, laptops and TVs in webcam frames using
a COCO-pretrained ultralytics model.  Detections at or above the configured
confidence floor drive counts and the evidence list; in addition the maximum
raw confidence per target class (down to a low 0.2 floor) is reported so the
fusion layer can weigh weak evidence.

``ultralytics`` is imported lazily; any failure yields a neutral result with
``available=False``.
"""

import threading
from pathlib import Path

from django.conf import settings

# Class NAMES the proctor cares about; the numeric ids are resolved from the
# loaded model's own ``names`` mapping (COCO defaults shown as fallback), so a
# fine-tuned checkpoint with a different class layout cannot be misread.
_TARGET_NAMES = ("person", "tv", "laptop", "cell phone", "book")
_FALLBACK_IDS = {0: "person", 62: "tv", 63: "laptop", 67: "cell phone", 73: "book"}
_CONF_KEYS_BY_NAME = {
    "cell phone": "phone_conf",
    "book": "book_conf",
    "laptop": "laptop_conf",
    "tv": "tv_conf",
}
# Raw floor passed to predict(): low enough that fusion still sees weak
# evidence below the alerting threshold.
_MIN_RAW_CONF = 0.2

def _neutral():
    """Fresh neutral result (own detections list each call)."""
    return {
        "available": False,
        "person_count": 0,
        "phone_conf": 0.0,
        "book_conf": 0.0,
        "laptop_conf": 0.0,
        "tv_conf": 0.0,
        "detections": [],
    }


class ObjectDetector:
    """YOLOv8 detector restricted to proctoring-relevant COCO classes."""

    def __init__(self):
        self._conf_floor = float(settings.PROCTORING["YOLO_CONF"])
        self._weights_name = settings.PROCTORING["YOLO_WEIGHTS"]
        self._model = None
        self._labels_by_id = dict(_FALLBACK_IDS)
        self._load_failed = False
        self._lock = threading.Lock()

    def _get_model(self):
        """Load YOLO once: local ml_models copy first, else auto-download."""
        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_failed:
                return None
            try:
                from ultralytics import YOLO

                local_path = Path(settings.ML_MODELS_DIR) / self._weights_name
                if local_path.exists():
                    self._model = YOLO(str(local_path))
                else:
                    # Bare name lets ultralytics fetch the official weights.
                    self._model = YOLO(self._weights_name)
                names = getattr(self._model, "names", None)
                if names:
                    resolved = {
                        int(i): str(n)
                        for i, n in dict(names).items()
                        if str(n) in _TARGET_NAMES
                    }
                    if resolved:
                        self._labels_by_id = resolved
            except Exception:
                self._load_failed = True
                return None
            return self._model

    def detect(self, bgr):
        """Detect target objects in one BGR frame; never raises."""
        try:
            model = self._get_model()
            if model is None or bgr is None or getattr(bgr, "size", 0) == 0:
                return _neutral()

            with self._lock:
                results = model.predict(bgr, conf=_MIN_RAW_CONF, verbose=False)

            out = _neutral()
            out["available"] = True
            if not results:
                return out

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return out

            class_ids = [int(c) for c in boxes.cls.tolist()]
            confs = [float(c) for c in boxes.conf.tolist()]
            coords = boxes.xyxy.tolist()

            person_count = 0
            for cls_id, conf, box in zip(class_ids, confs, coords):
                label = self._labels_by_id.get(cls_id)
                if label is None:
                    continue
                # Max raw confidence per non-person target class, even below
                # the alerting floor, so fusion can use weak evidence.
                key = _CONF_KEYS_BY_NAME.get(label)
                if key is not None and conf > out[key]:
                    out[key] = round(conf, 3)
                if conf < self._conf_floor:
                    continue
                if label == "person":
                    person_count += 1
                out["detections"].append(
                    {
                        "label": label,
                        "conf": round(conf, 3),
                        "box": [int(round(v)) for v in box],
                    }
                )
            out["person_count"] = person_count
            return out
        except Exception:
            return _neutral()
