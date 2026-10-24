"""Gaze / head-pose tracker built on MediaPipe FaceMesh (manuscript Sec. 3.3).

Per frame it reports face count, head pose (yaw/pitch/roll via solvePnP),
eye-aspect-ratio (EAR) and normalized iris position, plus the derived
``looking_away`` and ``eyes_closed`` flags. ``looking_away`` is decided
against global thresholds by default, or against a per-session calibration
envelope (from the "look at the screen boundaries" step) when one is passed
to ``analyze`` — see ``_looking_away``.

Frames arrive roughly every 2.5 s, so the mesh runs in single-image mode
(no inter-frame tracking assumptions) with iris landmarks enabled.
MediaPipe >= 1.0 exposes only the Tasks API (FaceLandmarker); older
0.10.x builds only the legacy ``mp.solutions.face_mesh``.  Both paths are
supported: Tasks first, legacy as fallback.  MediaPipe is imported
lazily; any failure yields a neutral result with ``available=False``.
"""

import math
import threading
import urllib.request

import cv2
import numpy as np
from django.conf import settings

# Canonical 3D face model points (generic head, arbitrary mm-like units)
# paired index-for-index with the MediaPipe landmarks in _PNP_LANDMARKS.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip          -> landmark 1
        (0.0, -330.0, -65.0),  # chin               -> landmark 152
        (-225.0, 170.0, -135.0),  # left eye outer     -> landmark 33
        (225.0, 170.0, -135.0),  # right eye outer    -> landmark 263
        (-150.0, -150.0, -125.0),  # left mouth corner  -> landmark 61
        (150.0, -150.0, -125.0),  # right mouth corner -> landmark 291
    ],
    dtype=np.float64,
)
_PNP_LANDMARKS = (1, 152, 33, 263, 61, 291)

# EAR landmark sets ordered p1..p6 for (|p2-p6| + |p3-p5|) / (2 |p1-p4|).
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (263, 387, 385, 362, 380, 373)
# Iris rings (available with refine_landmarks=True; mesh has 478 points).
_LEFT_IRIS = (468, 469, 470, 471, 472)
_RIGHT_IRIS = (473, 474, 475, 476, 477)
_REFINED_LANDMARK_COUNT = 478

_NEUTRAL = {
    "available": False,
    "face_count": 0,
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "ear": 0.0,
    "gaze_h": 0.5,
    "gaze_v": 0.5,
    "looking_away": False,
    "eyes_closed": False,
}


class GazeTracker:
    """MediaPipe FaceMesh analysis: head pose, EAR, iris gaze."""

    def __init__(self):
        cfg = settings.PROCTORING
        self._yaw_max = float(cfg["YAW_MAX_DEG"])
        self._pitch_min = float(cfg["PITCH_MIN_DEG"])
        self._ear_closed = float(cfg["EAR_CLOSED"])
        self._mesh = None
        self._api = None  # "tasks" (mediapipe >= 1.0) or "legacy" (0.10.x)
        self._load_failed = False
        self._lock = threading.Lock()  # inference is not thread-safe

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    _TASK_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )

    def _task_model_path(self):
        """Local face_landmarker.task, downloading it once if absent."""
        path = settings.ML_MODELS_DIR / "face_landmarker.task"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self._TASK_MODEL_URL, str(path))
        return path

    def _get_mesh(self):
        with self._lock:
            if self._mesh is not None:
                return self._mesh
            if self._load_failed:
                return None
            try:
                import mediapipe as mp

                if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                    self._mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=3,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                    )
                    self._api = "legacy"
                else:
                    from mediapipe.tasks import python as mp_tasks
                    from mediapipe.tasks.python import vision

                    options = vision.FaceLandmarkerOptions(
                        base_options=mp_tasks.BaseOptions(
                            model_asset_path=str(self._task_model_path())
                        ),
                        running_mode=vision.RunningMode.IMAGE,
                        num_faces=3,
                        min_face_detection_confidence=0.5,
                    )
                    self._mesh = vision.FaceLandmarker.create_from_options(options)
                    self._api = "tasks"
            except Exception:
                self._load_failed = True
                return None
            return self._mesh

    def _detect_faces(self, mesh, rgb):
        """Run inference; return a list of faces, each a list of landmarks."""
        if self._api == "legacy":
            results = mesh.process(rgb)
            return [f.landmark for f in (results.multi_face_landmarks or [])]
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return list(mesh.detect(mp_image).face_landmarks or [])

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalized_bbox_area(face_landmarks):
        xs = [lm.x for lm in face_landmarks]
        ys = [lm.y for lm in face_landmarks]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    @staticmethod
    def _rotation_to_euler_deg(rot):
        """Decompose a rotation matrix (R = Rz Ry Rx) into degrees."""
        sy = math.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
        if sy > 1e-6:
            x = math.atan2(rot[2, 1], rot[2, 2])
            y = math.atan2(-rot[2, 0], sy)
            z = math.atan2(rot[1, 0], rot[0, 0])
        else:  # gimbal lock
            x = math.atan2(-rot[1, 2], rot[1, 1])
            y = math.atan2(-rot[2, 0], sy)
            z = 0.0
        return math.degrees(x), math.degrees(y), math.degrees(z)

    def _head_pose(self, pts, width, height):
        """Yaw/pitch/roll in degrees via solvePnP on six rigid landmarks.

        Convention after normalization: yaw>0 head turned toward image
        right, pitch<0 looking down, pitch>0 looking up.  All values are
        sanity-clamped to [-90, 90].
        """
        image_points = pts[list(_PNP_LANDMARKS)].astype(np.float64)
        focal = float(width)
        camera_matrix = np.array(
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0, 0.0
        rot, _ = cv2.Rodrigues(rvec)
        pitch_raw, yaw, roll = self._rotation_to_euler_deg(rot)
        # The canonical model is Y-up while image coordinates are Y-down, so
        # a frontal face reads pitch ~ +/-180 deg.  Fold back into range and
        # flip the sign so that looking down is negative.
        if pitch_raw > 90.0:
            pitch_raw -= 180.0
        elif pitch_raw < -90.0:
            pitch_raw += 180.0
        pitch = -pitch_raw
        clamp = lambda v: max(-90.0, min(90.0, v))  # noqa: E731
        return clamp(yaw), clamp(pitch), clamp(roll)

    @staticmethod
    def _eye_metrics(pts, eye_ids, iris_ids):
        """Per-eye (EAR, gaze_h, gaze_v); gaze components None if degenerate."""
        p = pts[list(eye_ids)]
        horiz = float(np.linalg.norm(p[0] - p[3]))
        ear = None
        if horiz > 1e-6:
            ear = (
                float(np.linalg.norm(p[1] - p[5]))
                + float(np.linalg.norm(p[2] - p[4]))
            ) / (2.0 * horiz)

        iris = pts[list(iris_ids)].mean(axis=0)
        x_lo = min(p[0, 0], p[3, 0])
        x_hi = max(p[0, 0], p[3, 0])
        gaze_h = None
        if x_hi - x_lo > 1e-6:
            gaze_h = (float(iris[0]) - x_lo) / (x_hi - x_lo)

        top_y = (p[1, 1] + p[2, 1]) / 2.0
        bot_y = (p[4, 1] + p[5, 1]) / 2.0
        gaze_v = None
        if bot_y - top_y > 1e-6:
            gaze_v = (float(iris[1]) - top_y) / (bot_y - top_y)
        return ear, gaze_h, gaze_v

    # ------------------------------------------------------------------ #
    # Per-user calibration
    # ------------------------------------------------------------------ #
    def _looking_away(self, yaw, pitch, gaze_h, calibration):
        """Decide ``looking_away`` from global thresholds or, when present, a
        per-session calibration envelope (see ``pipeline.finalize_calibration``).

        ``calibration`` is the ``ExamSession.gaze_calibration`` dict (or
        ``None``). Each axis falls back to the global default independently,
        so a *partial* calibration (e.g. only the vertical markers produced
        usable samples) still tightens whichever axes it covers.
        """
        cal = calibration or {}

        yaw_min, yaw_max = cal.get("yaw_min"), cal.get("yaw_max")
        if yaw_min is not None and yaw_max is not None:
            if yaw < yaw_min or yaw > yaw_max:
                return True
        elif abs(yaw) > self._yaw_max:
            return True

        pitch_min, pitch_max = cal.get("pitch_min"), cal.get("pitch_max")
        if pitch_min is not None and pitch_max is not None:
            if pitch < pitch_min or pitch > pitch_max:
                return True
        elif pitch < self._pitch_min:
            return True

        gh_min, gh_max = cal.get("gaze_h_min"), cal.get("gaze_h_max")
        if gh_min is not None and gh_max is not None:
            if gaze_h < gh_min or gaze_h > gh_max:
                return True
        elif gaze_h < 0.25 or gaze_h > 0.75:
            return True

        return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(self, bgr, calibration=None):
        """Analyze one BGR frame; never raises.

        ``calibration``, when given, is a per-session gaze-envelope dict
        (see ``_looking_away``) that overrides the global yaw/pitch/gaze_h
        thresholds for the ``looking_away`` decision. Every other caller
        (including offline dataset feature extraction) omits it and gets
        the original global-threshold behavior unchanged.
        """
        try:
            mesh = self._get_mesh()
            if mesh is None or bgr is None or getattr(bgr, "size", 0) == 0:
                return dict(_NEUTRAL)

            height, width = bgr.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            with self._lock:
                faces = self._detect_faces(mesh, rgb)

            out = dict(_NEUTRAL)
            out["available"] = True
            out["face_count"] = len(faces)
            if not faces:
                # Absence is its own signal (handled by callers as
                # face_count == 0), not a "looking away" event.
                return out

            primary = max(faces, key=self._normalized_bbox_area)
            pts = np.array(
                [(lm.x * width, lm.y * height) for lm in primary],
                dtype=np.float64,
            )

            yaw, pitch, roll = self._head_pose(pts, width, height)

            ear = 0.0
            gaze_h, gaze_v = 0.5, 0.5
            if pts.shape[0] >= _REFINED_LANDMARK_COUNT:
                left = self._eye_metrics(pts, _LEFT_EYE, _LEFT_IRIS)
                right = self._eye_metrics(pts, _RIGHT_EYE, _RIGHT_IRIS)
                ears = [v for v in (left[0], right[0]) if v is not None]
                ghs = [v for v in (left[1], right[1]) if v is not None]
                gvs = [v for v in (left[2], right[2]) if v is not None]
                if ears:
                    ear = float(np.mean(ears))
                if ghs:
                    gaze_h = min(1.0, max(0.0, float(np.mean(ghs))))
                if gvs:
                    gaze_v = min(1.0, max(0.0, float(np.mean(gvs))))

            looking_away = self._looking_away(yaw, pitch, gaze_h, calibration)
            out.update(
                {
                    "yaw": round(yaw, 2),
                    "pitch": round(pitch, 2),
                    "roll": round(roll, 2),
                    "ear": round(ear, 3),
                    "gaze_h": round(gaze_h, 3),
                    "gaze_v": round(gaze_v, 3),
                    "looking_away": bool(looking_away),
                    "eyes_closed": bool(ear < self._ear_closed),
                }
            )
            return out
        except Exception:
            return dict(_NEUTRAL)
