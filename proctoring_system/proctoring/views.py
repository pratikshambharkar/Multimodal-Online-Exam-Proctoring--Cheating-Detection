"""Proctoring API endpoints (RAIM).

Receives webcam frames, microphone chunks, and browser events from the
exam-taking page, runs them through the analysis engine, persists the
results, and drives the three-tier alert / termination policy.

All endpoints return HTTP 200 with a JSON body; failures are reported as
``{"ok": false, "error": "..."}`` so the client-side poller never has to
special-case transport errors. Heavy libraries (cv2, numpy) are imported
inside the view functions per the project contract.
"""
import base64
import json
import threading

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from exams.models import ExamSession
from proctoring.models import AudioAnalysis, FrameAnalysis, MonitoringEvent

# Browser event types accepted by api_event.
BROWSER_EVENT_TYPES = (
    "tab_switch",
    "window_blur",
    "fullscreen_exit",
    "copy",
    "paste",
    "right_click",
    "reload",
)

# Events that count toward the tab-switch escalation policy.
TAB_EVENT_TYPES = ("tab_switch", "window_blur")

# Base messages for browser activity events.
BROWSER_EVENT_MESSAGES = {
    "tab_switch": "Switched away from the exam tab",
    "window_blur": "Exam window lost focus",
    "fullscreen_exit": "Exited fullscreen mode",
    "copy": "Copy attempt detected",
    "paste": "Paste attempt detected",
    "right_click": "Right-click detected",
    "reload": "Page reload detected",
}

# Seconds within which a repeated event_type is considered spam and skipped.
EVENT_DEDUP_SECONDS = 10

# Lazy module-level singleton for the audio analyzer.
_audio_analyzer = None
_audio_analyzer_lock = threading.Lock()


def _get_audio_analyzer():
    """Return the shared AudioAnalyzer instance, creating it on first use."""
    global _audio_analyzer
    if _audio_analyzer is None:
        with _audio_analyzer_lock:
            if _audio_analyzer is None:
                from proctoring.engine.audio_analyzer import AudioAnalyzer

                _audio_analyzer = AudioAnalyzer()
    return _audio_analyzer


def _get_session(request, session_id, require_active=True):
    """Fetch the ExamSession owned by ``request.user``.

    Returns ``(session, None)`` on success or ``(None, error_dict)`` when the
    session is missing, belongs to someone else, or (when ``require_active``)
    is no longer in progress. The error dict is ready to hand to JsonResponse.
    """
    try:
        session = ExamSession.objects.select_related("exam").get(
            pk=int(session_id), student=request.user
        )
    except (ExamSession.DoesNotExist, TypeError, ValueError):
        return None, {"ok": False, "error": "Exam session not found.", "terminate": True}
    if require_active and session.status != "in_progress":
        return None, {
            "ok": False,
            "error": f"Exam session is no longer active (status: {session.status}).",
            "terminate": True,
        }
    return session, None


def _update_trust_level(session):
    """Recompute the continuous trust level from the session's counters."""
    cfg = settings.PROCTORING
    if session.violation_count > 0 or session.risk_score >= cfg["RISK_VIOLATION"]:
        session.trust_level = "low"
    elif session.suspicious_count >= 3 or session.peak_risk >= cfg["RISK_SUSPICIOUS"]:
        session.trust_level = "medium"
    else:
        session.trust_level = "high"


def _recent_duplicate(session, event_type):
    """True if an identical event_type was logged within the dedup window."""
    cutoff = timezone.now() - timezone.timedelta(seconds=EVENT_DEDUP_SECONDS)
    return MonitoringEvent.objects.filter(
        session=session, event_type=event_type, created_at__gte=cutoff
    ).exists()


def _terminate_session(session, reason):
    """Mark the session terminated and log the system event.

    The caller is responsible for ``session.save()``.
    """
    session.status = "terminated"
    session.termination_reason = reason[:200]
    # submitted_at stays null here: the client's auto-submit (or the student's
    # own POST) still gets graded by exam_submit, which then stamps it.
    session.trust_level = "low"
    MonitoringEvent.objects.create(
        session=session,
        channel="system",
        event_type="session_terminated",
        severity="violation",
        message=f"Session terminated: {reason}"[:300],
        confidence=1.0,
        details={"reason": reason},
    )


def _reset_pipeline(session_id):
    """Drop the pipeline's in-memory state (gaze window, debounce) for a session.

    Never lets an engine failure break the termination response.
    """
    try:
        from proctoring.engine.pipeline import get_pipeline

        get_pipeline().reset_session(session_id)
    except Exception:
        pass


@login_required
@require_POST
def api_frame(request):
    """Analyze one webcam frame and update the session's risk state."""
    try:
        import cv2
        import numpy as np

        payload = json.loads(request.body.decode("utf-8"))
        session, error = _get_session(request, payload.get("session_id"))
        if error:
            return JsonResponse(error)

        data_url = payload.get("image", "")
        b64_data = data_url.split(",", 1)[1] if "," in data_url else data_url
        jpeg_bytes = base64.b64decode(b64_data)
        frame_buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
        if bgr is None:
            return JsonResponse(
                {"ok": False, "error": "Could not decode frame image.", "terminate": False}
            )

        cfg = settings.PROCTORING
        latest_audio = session.audio_chunks.order_by("-created_at").first()
        audio_speech_prob = latest_audio.speech_prob if latest_audio else 0.0
        browser_score = min(1.0, session.tab_switches / cfg["TAB_SWITCH_TERMINATE"])

        from proctoring.engine.pipeline import get_pipeline

        pipeline = get_pipeline()
        result = pipeline.analyze_frame(
            bgr,
            session,
            audio_speech_prob=audio_speech_prob,
            browser_score=browser_score,
        )

        channels = result.get("channels", {})
        identity_checked = bool(channels.get("identity_checked", False))

        FrameAnalysis.objects.create(
            session=session,
            face_count=int(channels.get("face_count", 0)),
            identity_verified=(
                bool(channels.get("identity_verified", False)) if identity_checked else None
            ),
            identity_distance=(
                channels.get("identity_distance") if identity_checked else None
            ),
            yaw=float(channels.get("yaw", 0.0)),
            pitch=float(channels.get("pitch", 0.0)),
            roll=float(channels.get("roll", 0.0)),
            ear=float(channels.get("ear", 0.0)),
            looking_away=bool(channels.get("looking_away", False)),
            gaze_off_fraction=float(channels.get("gaze_off_fraction", 0.0)),
            phone_conf=float(channels.get("phone_conf", 0.0)),
            book_conf=float(channels.get("book_conf", 0.0)),
            extra_persons=int(channels.get("extra_persons", 0)),
            cnn_cheat_prob=float(channels.get("cnn_cheat_prob", 0.0)),
            audio_speech_prob=float(channels.get("audio_speech_prob", audio_speech_prob)),
            svm_label=result.get("svm_label", "normal"),
            risk_score=float(result.get("risk_score", 0.0)),
        )

        if identity_checked:
            session.last_verify_at = timezone.now()
            next_gap = result.get("next_verify_gap")
            if next_gap:
                session.next_verify_gap = int(next_gap)

        session.risk_score = float(result.get("risk_score", session.risk_score))
        session.peak_risk = max(session.peak_risk, session.risk_score)

        frame_ts = timezone.now().strftime("%Y%m%d%H%M%S")
        for violation in result.get("violations", []):
            event_type = violation.get("event_type", "unknown")
            if _recent_duplicate(session, event_type):
                continue
            severity = violation.get("severity", "suspicious")
            event = MonitoringEvent(
                session=session,
                channel=violation.get("channel", "system"),
                event_type=event_type,
                severity=severity,
                message=str(violation.get("message", ""))[:300],
                confidence=float(violation.get("confidence", 0.0)),
                details=channels,
            )
            needs_evidence = severity == "violation" or (
                severity == "suspicious"
                and event_type in ("phone_detected", "book_detected", "gaze_off_screen")
            )
            if needs_evidence:
                # upload_to="evidence/" prefixes the final storage path.
                event.snapshot.save(
                    f"s{session.pk}_{frame_ts}.jpg",
                    ContentFile(jpeg_bytes),
                    save=False,
                )
            event.save()
            if severity == "violation":
                session.violation_count += 1
            elif severity == "suspicious":
                session.suspicious_count += 1

        _update_trust_level(session)

        # Termination is the pipeline's call (it requires SUSTAINED high risk,
        # not a single spike — see high_risk_streak in analyze_frame).
        terminate = bool(result.get("terminate", False))
        terminate_reason = str(result.get("terminate_reason", "") or "")
        if terminate:
            _terminate_session(session, terminate_reason or "Proctoring violation")
            _reset_pipeline(session.pk)

        session.save()

        result.pop("next_verify_gap", None)
        result["ok"] = True
        result["terminate"] = terminate
        result["terminate_reason"] = terminate_reason if terminate else ""
        return JsonResponse(result)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc), "terminate": False})


@login_required
@require_POST
def api_audio(request):
    """Analyze one microphone chunk and raise a speech alert if warranted."""
    try:
        session, error = _get_session(request, request.POST.get("session_id"))
        if error:
            return JsonResponse(error)

        upload = request.FILES.get("audio")
        if upload is None:
            return JsonResponse(
                {"ok": False, "error": "No audio file uploaded.", "terminate": False}
            )

        chunk_ts = timezone.now().strftime("%Y%m%d%H%M%S")
        saved_path = default_storage.save(
            f"audio_chunks/s{session.pk}_{chunk_ts}.wav", upload
        )

        analysis = _get_audio_analyzer().analyze_file(default_storage.path(saved_path))
        rms = float(analysis.get("rms", 0.0))
        speech_prob = float(analysis.get("speech_prob", 0.0))
        label = str(analysis.get("label", "silence"))

        AudioAnalysis.objects.create(
            session=session, rms=rms, speech_prob=speech_prob, label=label
        )

        cfg = settings.PROCTORING
        alert = (
            label in ("speech", "conversation")
            and speech_prob >= cfg["AUDIO_SPEECH_PROB_ALERT"]
        )
        event_created = False
        if alert and not _recent_duplicate(session, "speech_detected"):
            MonitoringEvent.objects.create(
                session=session,
                channel="audio",
                event_type="speech_detected",
                severity="suspicious",
                message=f"Background speech detected (probability {speech_prob:.2f})",
                confidence=speech_prob,
                details={
                    "rms": rms,
                    "speech_prob": speech_prob,
                    "label": label,
                    "audio_path": saved_path,
                },
            )
            session.suspicious_count += 1
            _update_trust_level(session)
            session.save()
            event_created = True

        if not event_created:
            try:
                default_storage.delete(saved_path)
            except OSError:
                pass  # leave the orphan chunk rather than fail the response

        return JsonResponse(
            {
                "ok": True,
                "rms": rms,
                "speech_prob": speech_prob,
                "label": label,
                "alert": alert,
            }
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc), "terminate": False})


@login_required
@require_POST
def api_event(request):
    """Record a browser activity event and enforce the tab-switch policy."""
    try:
        payload = json.loads(request.body.decode("utf-8"))
        session, error = _get_session(request, payload.get("session_id"))
        if error:
            return JsonResponse(error)

        event_type = payload.get("event_type", "")
        if event_type not in BROWSER_EVENT_TYPES:
            return JsonResponse(
                {"ok": False, "error": "Unknown event type.", "terminate": False}
            )
        client_details = payload.get("details") or {}
        cfg = settings.PROCTORING

        is_tab_event = event_type in TAB_EVENT_TYPES
        if is_tab_event:
            session.tab_switches += 1

        if event_type in ("right_click", "copy", "paste"):
            severity = "info"
        elif is_tab_event and session.tab_switches > cfg["TAB_SWITCH_WARN"]:
            severity = "violation"
        else:
            severity = "suspicious"

        message = BROWSER_EVENT_MESSAGES[event_type]
        if is_tab_event:
            message = (
                f"{message} ({session.tab_switches}/{cfg['TAB_SWITCH_TERMINATE']} "
                f"tab switches)"
            )

        details = dict(client_details)
        details["tab_switches"] = session.tab_switches
        MonitoringEvent.objects.create(
            session=session,
            channel="browser",
            event_type=event_type,
            severity=severity,
            message=message[:300],
            confidence=1.0,
            details=details,
        )
        if severity == "violation":
            session.violation_count += 1
        elif severity == "suspicious":
            session.suspicious_count += 1

        _update_trust_level(session)

        terminate = is_tab_event and session.tab_switches >= cfg["TAB_SWITCH_TERMINATE"]
        if terminate:
            _terminate_session(session, "Excessive tab switching")
            _reset_pipeline(session.pk)
            message = "Exam terminated: excessive tab switching"

        session.save()

        return JsonResponse(
            {
                "ok": True,
                "tab_switches": session.tab_switches,
                "severity": severity,
                "message": message,
                "terminate": terminate,
            }
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc), "terminate": False})


@login_required
@require_POST
def api_calibration_sample(request):
    """Capture one gaze-calibration sample (candidate looking at a marker).

    See templates/exams/calibration.html and static/js/calibration.js for
    the client-side sequencing: the marker sweeps to ~9 on-screen positions
    and posts a few frames at each. Samples are staged in the pipeline's
    in-memory per-session state; nothing is written to the database here.
    """
    try:
        import cv2
        import numpy as np

        payload = json.loads(request.body.decode("utf-8"))
        session, error = _get_session(request, payload.get("session_id"))
        if error:
            return JsonResponse(error)

        try:
            point_index = int(payload.get("point_index"))
        except (TypeError, ValueError):
            return JsonResponse(
                {"ok": False, "error": "point_index must be an integer.", "terminate": False}
            )

        data_url = payload.get("image", "")
        b64_data = data_url.split(",", 1)[1] if "," in data_url else data_url
        jpeg_bytes = base64.b64decode(b64_data)
        frame_buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
        if bgr is None:
            return JsonResponse(
                {"ok": False, "error": "Could not decode frame image.", "terminate": False}
            )

        from proctoring.engine.pipeline import get_pipeline

        result = get_pipeline().record_calibration_sample(session, point_index, bgr)
        return JsonResponse(
            {
                "ok": True,
                "face_found": bool(result.get("face_found")),
                "samples_for_point": int(result.get("samples_for_point", 0)),
            }
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc), "terminate": False})


@login_required
@require_POST
def api_calibration_finish(request):
    """Compute the gaze-calibration envelope from staged samples and store it.

    Marks the session calibrated (attempted) regardless of outcome: when too
    few markers yielded usable samples, ``gaze_calibration`` stays null and
    ``GazeTracker`` simply falls back to its global thresholds for this
    session — calibration failing never blocks the exam.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
        session, error = _get_session(request, payload.get("session_id"))
        if error:
            return JsonResponse(error)

        from proctoring.engine.pipeline import get_pipeline

        result = get_pipeline().finalize_calibration(session)
        calibrated = bool(result.get("calibrated"))

        session.gaze_calibration = result.get("envelope")
        session.gaze_calibrated = True
        session.save(update_fields=["gaze_calibration", "gaze_calibrated"])

        MonitoringEvent.objects.create(
            session=session,
            channel="gaze",
            event_type="calibration_complete",
            severity="info",
            message=(
                f"Gaze calibrated from {result.get('point_count', 0)} screen markers"
                if calibrated
                else "Gaze calibration skipped or insufficient — using default sensitivity"
            ),
            confidence=1.0,
            details={
                "point_count": result.get("point_count", 0),
                "sample_count": result.get("sample_count", 0),
                "calibrated": calibrated,
            },
        )

        return JsonResponse(
            {
                "ok": True,
                "calibrated": calibrated,
                "sample_count": int(result.get("sample_count", 0)),
                "point_count": int(result.get("point_count", 0)),
            }
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc), "terminate": False})


@login_required
@require_GET
def api_status(request, session_id):
    """Return the session's live risk state and remaining time."""
    try:
        session, error = _get_session(request, session_id, require_active=False)
        if error:
            return JsonResponse(error)

        now = timezone.now()
        seconds_left = int(max(0.0, (session.ends_at - now).total_seconds()))
        cfg = settings.PROCTORING
        if session.risk_score >= cfg["RISK_VIOLATION"]:
            risk_level = "violation"
        elif session.risk_score >= cfg["RISK_SUSPICIOUS"]:
            risk_level = "suspicious"
        else:
            risk_level = "normal"

        return JsonResponse(
            {
                "ok": True,
                "risk_score": round(session.risk_score, 4),
                "risk_level": risk_level,
                "violation_count": session.violation_count,
                "suspicious_count": session.suspicious_count,
                "tab_switches": session.tab_switches,
                "status": session.status,
                "seconds_left": seconds_left,
            }
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc), "terminate": False})
