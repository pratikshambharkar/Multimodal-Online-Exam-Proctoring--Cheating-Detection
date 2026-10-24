"""Exam flow views: listing, session start, the proctored exam room, submit, result."""
import random

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .models import Answer, Exam, ExamSession


def _grade(session, post_data=None):
    """Grade a session and finalize its outcome fields.

    When ``post_data`` (a QueryDict) is given, the submitted ``question_<id>``
    selections are persisted as Answer rows first.  When it is ``None`` (the
    server-side auto-submit on timeout) only the already-saved answers count.
    The session keeps a ``terminated`` status if proctoring ended it; otherwise
    it becomes ``submitted``.  The final trust level is derived from the
    accumulated proctoring counters and risk scores.
    """
    exam = session.exam
    questions = list(exam.questions.prefetch_related("choices"))

    if post_data is not None:
        for question in questions:
            raw = post_data.get(f"question_{question.id}")
            selected = None
            if raw:
                selected = next(
                    (c for c in question.choices.all() if str(c.id) == raw), None
                )
            Answer.objects.update_or_create(
                session=session,
                question=question,
                defaults={
                    "selected_choice": selected,
                    "is_correct": bool(selected and selected.is_correct),
                },
            )

    correct_by_question = {
        a.question_id: a.is_correct for a in session.answers.all()
    }
    session.score = float(
        sum(q.marks for q in questions if correct_by_question.get(q.id))
    )
    session.total_marks = float(sum(q.marks for q in questions))
    session.submitted_at = timezone.now()
    if session.status != "terminated":
        session.status = "submitted"

    cfg = settings.PROCTORING
    if (
        session.status == "terminated"
        or session.violation_count > 0
        or session.risk_score >= cfg["RISK_VIOLATION"]
    ):
        session.trust_level = "low"
    elif session.suspicious_count >= 3 or session.peak_risk >= cfg["RISK_SUSPICIOUS"]:
        session.trust_level = "medium"
    else:
        session.trust_level = "high"
    session.save()


@login_required
def exam_list(request):
    """List active exams with a Start / Resume / Result action per exam."""
    exams = (
        Exam.objects.filter(is_active=True)
        .annotate(question_count=Count("questions"))
        .order_by("-created_at")
    )
    session_map = {}
    user_sessions = ExamSession.objects.filter(
        student=request.user, exam__in=exams
    ).order_by("-started_at")
    for sess in user_sessions:
        current = session_map.get(sess.exam_id)
        if current is None:
            session_map[sess.exam_id] = sess
        elif sess.status == "in_progress" and current.status != "in_progress":
            session_map[sess.exam_id] = sess

    rows = [
        {
            "exam": exam,
            "question_count": exam.question_count,
            "session": session_map.get(exam.id),
        }
        for exam in exams
    ]
    return render(request, "exams/exam_list.html", {"rows": rows})


@login_required
@require_POST
def exam_start(request, exam_id):
    """Create (or resume) a proctored session for the exam and enter the room."""
    exam = get_object_or_404(Exam, pk=exam_id)
    if not exam.is_active:
        messages.error(request, "This exam is not currently active.")
        return redirect("exam_list")

    profile = getattr(request.user, "profile", None)
    if profile is None or not profile.face_registered:
        messages.error(
            request,
            "You must register your face before taking a proctored exam. "
            "Please complete face enrollment from your dashboard.",
        )
        return redirect("dashboard")

    existing = ExamSession.objects.filter(
        exam=exam, student=request.user, status="in_progress"
    ).first()
    if existing:
        messages.info(request, "Resuming your exam already in progress.")
        return redirect("exam_take", session_id=existing.id)

    cfg = settings.PROCTORING
    session = ExamSession.objects.create(
        exam=exam,
        student=request.user,
        total_marks=float(exam.total_marks),
        next_verify_gap=random.randint(
            cfg["REVERIFY_MIN_SECONDS"], cfg["REVERIFY_MAX_SECONDS"]
        ),
    )
    return redirect("exam_take", session_id=session.id)


@login_required
def exam_calibrate(request, session_id):
    """Gaze-calibration step: candidate looks at ~9 on-screen markers so
    GazeTracker can use a personalized on-screen envelope instead of its
    global thresholds for the rest of this attempt (see CONTRACT.md and
    proctoring/engine/pipeline.py ``finalize_calibration``)."""
    session = get_object_or_404(
        ExamSession.objects.select_related("exam"),
        pk=session_id,
        student=request.user,
    )
    if session.status != "in_progress":
        return redirect("exam_result", session_id=session.id)
    if session.gaze_calibrated:
        return redirect("exam_take", session_id=session.id)

    context = {
        "session": session,
        "exam": session.exam,
    }
    return render(request, "exams/calibration.html", context)


@login_required
def exam_take(request, session_id):
    """The proctored exam room: question form plus the live monitoring sidebar."""
    session = get_object_or_404(
        ExamSession.objects.select_related("exam"),
        pk=session_id,
        student=request.user,
    )
    if session.status != "in_progress":
        return redirect("exam_result", session_id=session.id)
    if not session.gaze_calibrated:
        return redirect("exam_calibrate", session_id=session.id)

    if timezone.now() >= session.ends_at:
        _grade(session)
        messages.warning(
            request,
            "Time is up — your exam was submitted automatically with the "
            "answers saved so far.",
        )
        return redirect("exam_result", session_id=session.id)

    exam = session.exam
    questions = list(exam.questions.prefetch_related("choices"))
    saved = {
        a.question_id: a.selected_choice_id
        for a in session.answers.all()
        if a.selected_choice_id
    }
    question_data = [
        {
            "question": q,
            "choices": q.choices.all(),
            "selected_id": saved.get(q.id),
        }
        for q in questions
    ]

    cfg = settings.PROCTORING
    context = {
        "session": session,
        "exam": exam,
        "question_data": question_data,
        "ends_at_ms": int(session.ends_at.timestamp() * 1000),
        "frame_interval_ms": cfg["FRAME_INTERVAL_MS"],
        "audio_chunk_ms": cfg["AUDIO_CHUNK_MS"],
    }
    return render(request, "exams/exam_take.html", context)


@login_required
@require_POST
def exam_submit(request, session_id):
    """Grade the submitted answers and finalize the session."""
    session = get_object_or_404(
        ExamSession.objects.select_related("exam"),
        pk=session_id,
        student=request.user,
    )
    if session.status == "submitted" or (
        session.status == "terminated" and session.answers.exists()
    ):
        messages.info(request, "This exam has already been submitted.")
        return redirect("exam_result", session_id=session.id)

    was_terminated = session.status == "terminated"

    # Reject answers posted long after time expired (small grace period for
    # network/auto-submit latency); grade whatever was already saved instead.
    grace = timezone.timedelta(seconds=30)
    if not was_terminated and timezone.now() > session.ends_at + grace:
        _grade(session, None)
        messages.warning(
            request,
            "Time was already up — your exam was submitted automatically "
            "with the answers saved before the deadline.",
        )
        return redirect("exam_result", session_id=session.id)

    _grade(session, request.POST)
    if was_terminated:
        messages.warning(
            request,
            "Your session was terminated by proctoring; the answers you had "
            "selected were still graded.",
        )
    else:
        messages.success(request, "Your exam was submitted successfully.")
    return redirect("exam_result", session_id=session.id)


@login_required
def exam_result(request, session_id):
    """Show the score and integrity outcome of a finished session."""
    session = get_object_or_404(
        ExamSession.objects.select_related("exam"),
        pk=session_id,
        student=request.user,
    )
    if session.status == "in_progress":
        return redirect("exam_take", session_id=session.id)

    question_count = session.exam.questions.count()
    answered_count = session.answers.filter(selected_choice__isnull=False).count()
    correct_count = session.answers.filter(is_correct=True).count()
    context = {
        "session": session,
        "question_count": question_count,
        "answered_count": answered_count,
        "correct_count": correct_count,
    }
    return render(request, "exams/exam_result.html", context)
