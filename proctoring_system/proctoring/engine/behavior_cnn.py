"""CNN behavior classifier (manuscript Sec. 3.4/3.7 CNN channel).

Loads the trained Keras model from ``ml_models/behavior_cnn.keras`` and
returns P(cheating) for a webcam frame.  The model takes a 160x160 RGB image
scaled to [0, 1] and outputs a 2-unit softmax whose class order follows the
training folders alphabetically: index 0 = cheating, index 1 = normal.

TensorFlow is imported lazily; if the library or the model file is missing
the predictor degrades to ``available=False`` with ``cheat_prob=0.0``.
"""

import threading
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings

_MODEL_FILENAME = "behavior_cnn.keras"
_INPUT_SIZE = (160, 160)  # (width, height) for cv2.resize

_NEUTRAL = {"available": False, "cheat_prob": 0.0}


class BehaviorCNN:
    """Keras CNN that scores each frame with a cheating probability."""

    def __init__(self):
        self._model_path = Path(settings.ML_MODELS_DIR) / _MODEL_FILENAME
        self._model = None
        self._load_failed = False
        self._lock = threading.Lock()

    def _get_model(self):
        """Load the Keras model exactly once; None when unavailable."""
        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_failed:
                return None
            if not self._model_path.exists():
                # Not a hard failure: the file may be trained/added later.
                return None
            try:
                import tensorflow as tf

                self._model = tf.keras.models.load_model(
                    str(self._model_path), compile=False
                )
            except Exception:
                self._load_failed = True
                return None
            return self._model

    def predict(self, bgr):
        """Return ``{"available": bool, "cheat_prob": float}``; never raises."""
        try:
            model = self._get_model()
            if model is None or bgr is None or getattr(bgr, "size", 0) == 0:
                return dict(_NEUTRAL)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, _INPUT_SIZE, interpolation=cv2.INTER_AREA)
            batch = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

            with self._lock:
                probs = model.predict(batch, verbose=0)

            cheat_prob = float(np.asarray(probs).reshape(-1)[0])
            cheat_prob = min(1.0, max(0.0, cheat_prob))
            return {"available": True, "cheat_prob": round(cheat_prob, 4)}
        except Exception:
            return dict(_NEUTRAL)
