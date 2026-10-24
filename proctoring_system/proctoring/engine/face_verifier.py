"""DeepFace identity-verification engine (manuscript Sec. 3.2).

Wraps ``DeepFace.represent`` to produce face embeddings and compare them
against the embedding stored at enrollment using cosine distance.

The ``deepface`` package (and the TensorFlow stack underneath it) is imported
lazily inside the loader so Django management commands work without it.  Every
public method degrades gracefully: on any failure it returns a neutral result
instead of raising.
"""

import threading

import numpy as np
from django.conf import settings


class FaceVerifier:
    """Identity verification via DeepFace embeddings + cosine distance."""

    def __init__(self):
        cfg = settings.PROCTORING
        self._model_name = cfg["FACE_MODEL"]
        self._detector = cfg["FACE_DETECTOR"]
        self._threshold = float(cfg["FACE_DISTANCE_THRESHOLD"])
        self._deepface = None          # cached deepface module (model pre-built)
        self._load_failed = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_deepface(self):
        """Import deepface and build the recognition model exactly once."""
        with self._lock:
            if self._deepface is not None:
                return self._deepface
            if self._load_failed:
                return None
            try:
                from deepface import DeepFace

                # Warm the backbone once so per-frame calls hit DeepFace's
                # internal model cache instead of rebuilding.
                DeepFace.build_model(self._model_name)
                self._deepface = DeepFace
            except Exception:
                self._load_failed = True
                return None
            return self._deepface

    def _represent(self, bgr, enforce_detection):
        """Run DeepFace.represent on a BGR frame; returns list of face dicts."""
        deepface = self._get_deepface()
        if deepface is None:
            return None
        return deepface.represent(
            img_path=bgr,
            model_name=self._model_name,
            detector_backend=self._detector,
            enforce_detection=enforce_detection,
        )

    @staticmethod
    def _facial_area(rep):
        """Pixel area of a represent() result's facial_area box."""
        area = rep.get("facial_area") or {}
        return float(area.get("w", 0)) * float(area.get("h", 0))

    @staticmethod
    def _is_real_face(rep):
        """True unless DeepFace fell back to the whole frame (confidence 0)."""
        try:
            return float(rep.get("face_confidence", 1.0)) > 0.0
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _cosine_distance(a, b):
        """Cosine distance 1 - (a.b / |a||b|), safe against zero norms."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return 1.0
        return float(1.0 - float(np.dot(a, b)) / denom)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def enroll(self, bgr):
        """Extract the enrollment embedding from a registration photo.

        Requires exactly one detectable face.  Returns
        ``(embedding, "")`` on success or ``(None, error_message)``.
        """
        try:
            deepface = self._get_deepface()
            if deepface is None:
                return None, "Face verification engine unavailable"
            try:
                reps = self._represent(bgr, enforce_detection=True)
            except ValueError:
                # DeepFace raises ValueError when no face passes detection.
                return None, "No face detected"
            if not reps:
                return None, "No face detected"
            if len(reps) > 1:
                return None, "Multiple faces detected"
            embedding = [float(v) for v in reps[0]["embedding"]]
            return embedding, ""
        except Exception:
            return None, "Face analysis failed"

    def embed(self, bgr):
        """Embedding of the largest face in the frame, or None."""
        try:
            reps = self._represent(bgr, enforce_detection=False)
            if not reps:
                return None
            faces = [r for r in reps if self._is_real_face(r)]
            if not faces:
                return None
            best = max(faces, key=self._facial_area)
            return [float(v) for v in best["embedding"]]
        except Exception:
            return None

    def verify(self, bgr, stored_embedding, threshold=None):
        """Compare the frame against the enrolled embedding.

        Returns ``{"available": bool, "verified": bool, "distance": float | None,
        "face_found": bool}``.  ``available`` is False when the DeepFace engine
        cannot run (missing dependency, failed weight download); callers must
        then treat the check as "not performed", never as a failed identity.
        With multiple faces present, the best-matching (lowest-distance) face
        is used so the enrolled student is still recognized.
        """
        neutral = {"available": True, "verified": False, "distance": None, "face_found": False}
        try:
            if not stored_embedding:
                return dict(neutral)
            if self._get_deepface() is None:
                return {
                    "available": False,
                    "verified": False,
                    "distance": None,
                    "face_found": False,
                }
            reps = self._represent(bgr, enforce_detection=False)
            if not reps:
                return dict(neutral)
            faces = [r for r in reps if self._is_real_face(r)]
            if not faces:
                return dict(neutral)
            distance = min(
                self._cosine_distance(r["embedding"], stored_embedding)
                for r in faces
            )
            return {
                "available": True,
                "verified": bool(
                    distance <= (self._threshold if threshold is None else float(threshold))
                ),
                "distance": round(float(distance), 4),
                "face_found": True,
            }
        except Exception:
            return dict(neutral)
