"""Real-time proctoring pipeline orchestrating all detection engines.

One frame flows through gaze/head-pose analysis, object detection, the
behavior CNN, periodic DeepFace re-verification and the fusion engine
(manuscript Sec. 3.1-3.8). The pipeline is a process-wide, thread-safe
singleton holding lazily constructed engine instances and lightweight
per-session sliding-window state. It never touches the database — the
views persist FrameAnalysis / MonitoringEvent rows from the returned dict.
"""

import random
import statistics
import threading
import time
from collections import deque

from django.conf import settings
from django.utils import timezone


class ProctoringPipeline:
    """Coordinates the per-frame multimodal analysis for all active sessions."""

    def __init__(self):
        self._engine_lock = threading.Lock()
        self._engines_ready = False
        self._face_verifier = None
        self._gaze_tracker = None
        self._object_detector = None
        self._behavior_cnn = None
        self._fusion = None
        self._state_lock = threading.Lock()
        self._sessions = {}

    # ------------------------------------------------------------------ #
    # Engine and state management
    # ------------------------------------------------------------------ #

    def _ensure_engines(self):
        """Construct every engine exactly once (double-checked locking).

        Sibling engine modules are imported here — not at module top level —
        so `manage.py migrate` never pulls in cv2/numpy/tensorflow.
        """
        if self._engines_ready:
            return
        with self._engine_lock:
            if self._engines_ready:
                return
            from .behavior_cnn import BehaviorCNN
            from .face_verifier import FaceVerifier
            from .fusion import FusionEngine
            from .gaze_tracker import GazeTracker
            from .object_detector import ObjectDetector

            self._face_verifier = FaceVerifier()
            self._gaze_tracker = GazeTracker()
            self._object_detector = ObjectDetector()
            self._behavior_cnn = BehaviorCNN()
            self._fusion = FusionEngine()
            self._engines_ready = True

    def _get_state(self, session_id):
        """Fetch (or create) the in-memory sliding-window state for a session."""
        with self._state_lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = {
                    "gaze_window": deque(),
                    "gaze_away_since": None,  # monotonic ts the current away-streak began
                    "person_streak": 0,
                    "identity_fail_streak": 0,
                    "high_risk_streak": 0,
                    "calibration_samples": {},  # point_index -> [feature dict, ...]
                }
                self._sessions[session_id] = state
            return state

    def reset_session(self, session_id):
        """Drop all in-memory state for a finished or terminated session."""
        with self._state_lock:
            self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------ #
    # Gaze calibration ("look at the screen boundaries" step)
    # ------------------------------------------------------------------ #

    def record_calibration_sample(self, session, point_index, bgr):
        """Capture one calibration sample while the candidate looks at an
        on-screen marker.

        Runs gaze analysis with no calibration envelope (there isn't one
        yet) and, when exactly one face with open eyes is found, appends its
        raw (yaw, pitch, gaze_h, gaze_v) to that marker's in-memory sample
        list. Nothing is written to the database and this never raises.
        Returns ``{"face_found": bool, "samples_for_point": int}``.
        """
        self._ensure_engines()
        cfg = settings.PROCTORING
        try:
            gaze = self._gaze_tracker.analyze(bgr)
        except Exception:
            gaze = {}

        face_found = (
            bool(gaze.get("available"))
            and int(gaze.get("face_count") or 0) == 1
            and float(gaze.get("ear") or 0.0) >= float(cfg["EAR_CLOSED"])
        )

        samples_for_point = 0
        if face_found:
            sample = {
                "yaw": float(gaze.get("yaw") or 0.0),
                "pitch": float(gaze.get("pitch") or 0.0),
                "gaze_h": float(
                    gaze.get("gaze_h") if gaze.get("gaze_h") is not None else 0.5
                ),
                "gaze_v": float(
                    gaze.get("gaze_v") if gaze.get("gaze_v") is not None else 0.5
                ),
            }
            state = self._get_state(session.id)
            with self._state_lock:
                bucket = state["calibration_samples"].setdefault(int(point_index), [])
                bucket.append(sample)
                samples_for_point = len(bucket)

        return {"face_found": face_found, "samples_for_point": samples_for_point}

    def finalize_calibration(self, session):
        """Turn the accumulated per-marker samples into a gaze envelope and
        clear the in-memory staging data for this session.

        A marker only counts once it has ``CALIBRATION_MIN_SAMPLES_PER_POINT``
        good samples (its features are then reduced to a per-marker median,
        which damps single-frame noise); the envelope needs at least
        ``CALIBRATION_MIN_POINTS`` such markers to be considered usable.
        Below that, calibration is reported as not achieved and the caller
        should store ``None`` so ``GazeTracker`` falls back to its global
        thresholds. Never writes to the database.
        """
        cfg = settings.PROCTORING
        state = self._get_state(session.id)
        with self._state_lock:
            samples_by_point = state["calibration_samples"]
            state["calibration_samples"] = {}

        min_per_point = int(cfg["CALIBRATION_MIN_SAMPLES_PER_POINT"])
        point_medians = []
        total_samples = 0
        for bucket in samples_by_point.values():
            total_samples += len(bucket)
            if len(bucket) < min_per_point:
                continue
            point_medians.append({
                axis: statistics.median(s[axis] for s in bucket)
                for axis in ("yaw", "pitch", "gaze_h", "gaze_v")
            })

        point_count = len(point_medians)
        if point_count < int(cfg["CALIBRATION_MIN_POINTS"]):
            return {
                "calibrated": False,
                "envelope": None,
                "sample_count": total_samples,
                "point_count": point_count,
            }

        yaw_margin = float(cfg["CALIBRATION_MARGIN_YAW_DEG"])
        pitch_margin = float(cfg["CALIBRATION_MARGIN_PITCH_DEG"])
        gaze_margin = float(cfg["CALIBRATION_MARGIN_GAZE"])
        yaws = [m["yaw"] for m in point_medians]
        pitches = [m["pitch"] for m in point_medians]
        gh = [m["gaze_h"] for m in point_medians]
        gv = [m["gaze_v"] for m in point_medians]

        envelope = {
            "yaw_min": round(min(yaws) - yaw_margin, 2),
            "yaw_max": round(max(yaws) + yaw_margin, 2),
            "pitch_min": round(min(pitches) - pitch_margin, 2),
            "pitch_max": round(max(pitches) + pitch_margin, 2),
            "gaze_h_min": round(max(0.0, min(gh) - gaze_margin), 3),
            "gaze_h_max": round(min(1.0, max(gh) + gaze_margin), 3),
            "gaze_v_min": round(max(0.0, min(gv) - gaze_margin), 3),
            "gaze_v_max": round(min(1.0, max(gv) + gaze_margin), 3),
            "sample_count": total_samples,
            "point_count": point_count,
            "calibrated_at": timezone.now().isoformat(),
        }
        return {
            "calibrated": True,
            "envelope": envelope,
            "sample_count": total_samples,
            "point_count": point_count,
        }

    # ------------------------------------------------------------------ #
    # Frame analysis
    # ------------------------------------------------------------------ #

    def analyze_frame(
        self, bgr, session, force_verify=False, audio_speech_prob=0.0, browser_score=None
    ):
        """Analyze one BGR frame for the given ExamSession.

        Runs gaze, object and CNN inference every frame; DeepFace identity
        re-verification only when the randomized interval has elapsed (or
        ``force_verify`` is set). ``audio_speech_prob`` is the latest score
        the view read from AudioAnalysis. Returns the full api_frame
        response dict; nothing is written to the database here.
        """
        self._ensure_engines()
        cfg = settings.PROCTORING
        state = self._get_state(session.id)

        gaze = self._gaze_tracker.analyze(
            bgr, calibration=getattr(session, "gaze_calibration", None)
        )
        objects = self._object_detector.detect(bgr)
        cnn = self._behavior_cnn.predict(bgr)

        gaze_available = bool(gaze.get("available"))
        if gaze_available:
            face_count = int(gaze.get("face_count") or 0)
            yaw = float(gaze.get("yaw") or 0.0)
            pitch = float(gaze.get("pitch") or 0.0)
            roll = float(gaze.get("roll") or 0.0)
            ear = float(gaze.get("ear") or 0.0)
            looking_away = bool(gaze.get("looking_away"))
        else:
            # Neutral substitutes so a missing MediaPipe install cannot
            # masquerade as an absent face or closed eyes.
            face_count, yaw, pitch, roll, ear, looking_away = 1, 0.0, 0.0, 0.0, 0.28, False

        person_count = int(objects.get("person_count") or 0)
        phone_conf = float(objects.get("phone_conf") or 0.0)
        book_conf = float(objects.get("book_conf") or 0.0)
        laptop_conf = float(objects.get("laptop_conf") or 0.0)
        cnn_cheat_prob = float(cnn.get("cheat_prob") or 0.0)
        audio_speech_prob = max(0.0, min(1.0, float(audio_speech_prob or 0.0)))

        # -- Periodic identity re-verification (Sec. 3.2) -- #
        identity_checked = False
        identity_verified = None
        identity_distance = None
        next_verify_gap = None
        due = (
            force_verify
            or session.last_verify_at is None
            or (timezone.now() - session.last_verify_at).total_seconds()
            >= (session.next_verify_gap or 0)
        )
        if due:
            profile = getattr(session.student, "profile", None)
            embedding = getattr(profile, "face_embedding", None)
            if embedding:
                result = self._face_verifier.verify(
                    bgr,
                    embedding,
                    threshold=cfg.get("FACE_REVERIFY_THRESHOLD"),
                )
                # Only count the check when the engine ran AND a face was
                # actually visible: an empty frame at a due-verify moment is
                # face_absent's business, not an identity mismatch.
                if result.get("available", True) and result.get("face_found"):
                    identity_checked = True
                    identity_verified = bool(result.get("verified"))
                    raw_distance = result.get("distance")
                    identity_distance = (
                        float(raw_distance) if raw_distance is not None else None
                    )
                    next_verify_gap = random.randint(
                        int(cfg["REVERIFY_MIN_SECONDS"]),
                        int(cfg["REVERIFY_MAX_SECONDS"]),
                    )

        # -- Sliding gaze window and person debounce (Secs. 3.3-3.4) -- #
        window_seconds = float(cfg["GAZE_WINDOW_SECONDS"])
        debounce_frames = int(cfg["PERSON_DEBOUNCE_FRAMES"])
        extra_persons_raw = max(0, person_count - 1)
        now_mono = time.monotonic()
        with self._state_lock:
            window = state["gaze_window"]
            window.append((now_mono, looking_away))
            while window and now_mono - window[0][0] > window_seconds:
                window.popleft()
            gaze_off_fraction = (
                sum(1 for _, away in window if away) / len(window) if window else 0.0
            )
            # Continuous away-streak duration, independent of the coarse
            # windowed fraction above - this is what actually distinguishes
            # a brief glance from a sustained look-away.
            if looking_away:
                if state["gaze_away_since"] is None:
                    state["gaze_away_since"] = now_mono
                gaze_away_seconds = now_mono - state["gaze_away_since"]
            else:
                state["gaze_away_since"] = None
                gaze_away_seconds = 0.0
            if extra_persons_raw > 0:
                state["person_streak"] += 1
            else:
                state["person_streak"] = 0
            person_streak = state["person_streak"]

        extra_person_debounced = person_streak >= debounce_frames
        extra_persons = extra_persons_raw if extra_person_debounced else 0

        channels = {
            "face_count": face_count,
            "identity_verified": identity_verified,
            "identity_distance": (
                round(identity_distance, 4) if identity_distance is not None else None
            ),
            "identity_checked": identity_checked,
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2),
            "ear": round(ear, 3),
            "looking_away": looking_away,
            "gaze_off_fraction": round(gaze_off_fraction, 3),
            "phone_conf": round(phone_conf, 3),
            "book_conf": round(book_conf, 3),
            "laptop_conf": round(laptop_conf, 3),
            "person_count": person_count,
            "extra_persons": extra_persons,
            "cnn_cheat_prob": round(cnn_cheat_prob, 4),
            "audio_speech_prob": round(audio_speech_prob, 4),
        }

        # -- Fusion: frame label + EWMA risk (Sec. 3.7) -- #
        svm_label, svm_prob = self._fusion.classify(channels)
        if browser_score is None:
            terminate_switches = int(cfg["TAB_SWITCH_TERMINATE"])
            browser_score = (
                min(1.0, float(session.tab_switches or 0) / terminate_switches)
                if terminate_switches
                else 0.0
            )
        browser_score = max(0.0, min(1.0, float(browser_score)))
        scores = self._fusion.channel_scores(channels, browser_score=browser_score)
        risk_score = self._fusion.update_risk(session.risk_score, scores)
        if risk_score >= float(cfg["RISK_VIOLATION"]):
            risk_level = "violation"
        elif risk_score >= float(cfg["RISK_SUSPICIOUS"]):
            risk_level = "suspicious"
        else:
            risk_level = "normal"

        # -- Streak counters driving termination (Sec. 3.8) -- #
        with self._state_lock:
            if identity_checked:
                if identity_verified:
                    state["identity_fail_streak"] = 0
                else:
                    state["identity_fail_streak"] += 1
            identity_fail_streak = state["identity_fail_streak"]
            if risk_score >= float(cfg["RISK_VIOLATION"]):
                state["high_risk_streak"] += 1
            else:
                state["high_risk_streak"] = 0
            high_risk_streak = state["high_risk_streak"]

        violations = self._build_violations(
            cfg=cfg,
            gaze_available=gaze_available,
            face_count=face_count,
            phone_conf=phone_conf,
            book_conf=book_conf,
            extra_persons=extra_persons,
            extra_person_debounced=extra_person_debounced,
            identity_checked=identity_checked,
            identity_verified=identity_verified,
            identity_distance=identity_distance,
            gaze_off_fraction=gaze_off_fraction,
            gaze_away_seconds=gaze_away_seconds,
            cnn_cheat_prob=cnn_cheat_prob,
        )

        terminate = False
        terminate_reason = ""
        if identity_fail_streak >= 2:
            terminate = True
            terminate_reason = "Identity verification failed repeatedly"
        elif extra_person_debounced and person_streak >= 2 * debounce_frames:
            terminate = True
            terminate_reason = "Unauthorized person persistently present in frame"
        elif high_risk_streak >= 3:
            terminate = True
            terminate_reason = "Sustained high cheating-risk score"

        return {
            "ok": True,
            "channels": channels,
            "svm_label": svm_label,
            "svm_prob": round(svm_prob, 4),
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "violations": violations,
            "terminate": terminate,
            "terminate_reason": terminate_reason,
            "identity_checked": identity_checked,
            "next_verify_gap": next_verify_gap,
        }

    @staticmethod
    def _build_violations(
        cfg,
        gaze_available,
        face_count,
        phone_conf,
        book_conf,
        extra_persons,
        extra_person_debounced,
        identity_checked,
        identity_verified,
        identity_distance,
        gaze_off_fraction,
        gaze_away_seconds,
        cnn_cheat_prob,
    ):
        """Assemble the frame's violation/suspicion event dicts (Sec. 3.8)."""
        violations = []

        if gaze_available and face_count == 0:
            violations.append({
                "channel": "identity",
                "event_type": "face_absent",
                "severity": "suspicious",
                "message": "No face visible in the camera frame",
                "confidence": 0.6,
            })
        elif gaze_available and face_count > 1:
            violations.append({
                "channel": "identity",
                "event_type": "multiple_faces",
                "severity": "violation",
                "message": f"Multiple faces detected in frame ({face_count})",
                "confidence": 0.8,
            })

        if phone_conf >= float(cfg["PHONE_ALERT_CONF"]):
            violations.append({
                "channel": "objects",
                "event_type": "phone_detected",
                "severity": "violation",
                "message": f"Mobile phone detected (conf {phone_conf:.2f})",
                "confidence": round(phone_conf, 3),
            })
        elif phone_conf >= float(cfg["YOLO_CONF"]):
            violations.append({
                "channel": "objects",
                "event_type": "phone_detected",
                "severity": "suspicious",
                "message": f"Possible mobile phone in frame (conf {phone_conf:.2f})",
                "confidence": round(phone_conf, 3),
            })

        if book_conf >= 0.5:
            violations.append({
                "channel": "objects",
                "event_type": "book_detected",
                "severity": "suspicious",
                "message": f"Book or notes detected (conf {book_conf:.2f})",
                "confidence": round(book_conf, 3),
            })

        if extra_person_debounced and extra_persons > 0:
            violations.append({
                "channel": "objects",
                "event_type": "extra_person",
                "severity": "violation",
                "message": f"{extra_persons} extra person(s) present in the camera frame",
                "confidence": 0.9,
            })

        if identity_checked and not identity_verified:
            if identity_distance is not None:
                message = f"Identity verification failed (distance {identity_distance:.2f})"
            else:
                message = "Identity verification failed (no matching face found)"
            violations.append({
                "channel": "identity",
                "event_type": "identity_mismatch",
                "severity": "violation",
                "message": message,
                "confidence": 1.0,
            })

        # Both conditions must hold: the windowed fraction catches a *pattern*
        # of looking away (manuscript Sec. 3.3), but with frames sampled every
        # ~2.5s a 10s window is only ~4 samples, so one or two glance-frames
        # alone can already cross GAZE_OFF_FRACTION. Requiring the *current*
        # away-streak to have run continuously past GAZE_AWAY_MIN_SECONDS
        # filters out a brief glance while still catching sustained avoidance.
        if (
            gaze_off_fraction >= float(cfg["GAZE_OFF_FRACTION"])
            and gaze_away_seconds > float(cfg["GAZE_AWAY_MIN_SECONDS"])
        ):
            window_s = int(cfg["GAZE_WINDOW_SECONDS"])
            violations.append({
                "channel": "gaze",
                "event_type": "gaze_off_screen",
                "severity": "suspicious",
                "message": (
                    f"Looking away from screen {gaze_off_fraction:.0%} of the "
                    f"last {window_s}s (sustained {gaze_away_seconds:.1f}s)"
                ),
                "confidence": round(gaze_off_fraction, 3),
            })

        if cnn_cheat_prob >= 0.75:
            violations.append({
                "channel": "cnn",
                "event_type": "cnn_suspicious_behavior",
                "severity": "suspicious",
                "message": (
                    f"Behavior model indicates possible cheating (p={cnn_cheat_prob:.2f})"
                ),
                "confidence": round(cnn_cheat_prob, 4),
            })

        return violations


_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    """Return the process-wide ProctoringPipeline singleton (double-checked lock)."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = ProctoringPipeline()
    return _pipeline
