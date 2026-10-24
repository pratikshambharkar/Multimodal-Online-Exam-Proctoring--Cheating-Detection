"""Reports module — integrity report listing, detail view, and PDF export.

Instructors can browse every exam session; students see only their own.
The detail view aggregates monitoring data (events, frames, audio chunks)
into a reviewer-friendly report, and the PDF endpoint renders the same
data with ReportLab for offline archiving.
"""
import io
import os
from xml.sax.saxutils import escape

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
from django.utils import timezone

from exams.models import Exam, ExamSession
from proctoring.models import MonitoringEvent

CHANNEL_LABELS = dict(MonitoringEvent.CHANNEL_CHOICES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_instructor(user) -> bool:
    """True when the user holds the instructor role (staff users included)."""
    profile = getattr(user, "profile", None)
    if profile is not None:
        return profile.is_instructor
    return user.is_staff


def _can_view(user, session: ExamSession) -> bool:
    """Only the session owner or an instructor may open a report."""
    return session.student_id == user.id or _is_instructor(user)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """Human-friendly count phrase, e.g. '1 violation' / '3 violations'."""
    word = singular if count == 1 else (plural or singular + "s")
    return f"{count} {word}"


def _session_end(session: ExamSession, frames: list, events: list):
    """Best estimate of when monitored activity for the session ended."""
    if session.submitted_at:
        return session.submitted_at
    if session.is_active:
        return timezone.now()
    stamps = [session.started_at]
    if frames:
        stamps.append(frames[-1].created_at)
    if events:
        stamps.append(events[-1].created_at)
    return max(stamps)


def _duration_string(delta) -> str:
    """Format a timedelta as '1h 05m 30s' or '12m 05s'."""
    total = max(int(delta.total_seconds()), 0)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def _fmt_dt(dt) -> str:
    """Localized 'day month year, H:M:S' string, or a dash when missing."""
    if not dt:
        return "-"
    return timezone.localtime(dt).strftime("%d %b %Y, %H:%M:%S")


def _fmt_time(dt) -> str:
    """Localized HH:MM:SS string."""
    return timezone.localtime(dt).strftime("%H:%M:%S")


def _mmss(value) -> str:
    """Seconds-count formatted as m:ss (used for chart axis labels)."""
    seconds = max(int(round(float(value))), 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _verdict_text(session: ExamSession, events: list,
                  identity_passed: int, identity_failed: int) -> str:
    """Auto-generate the plain-language verdict paragraph for a session."""
    flagged = sorted({e.channel for e in events
                      if e.severity in ("suspicious", "violation")})
    channel_names = ", ".join(CHANNEL_LABELS.get(c, c) for c in flagged)

    sentences = []
    if session.status == "terminated":
        reason = session.termination_reason or "risk threshold exceeded"
        sentences.append(f"Session terminated by proctoring: {reason}.")

    summary = (
        f"{_plural(session.violation_count, 'violation')} and "
        f"{_plural(session.suspicious_count, 'suspicious flag')} across "
        f"{_plural(len(flagged), 'channel')}"
    )
    if channel_names:
        summary += f" ({channel_names})"

    identity = f"identity verified {_plural(identity_passed, 'time')}"
    if identity_failed:
        identity += f" with {_plural(identity_failed, 'failed check')}"

    risk = (f"final risk {session.risk_score:.2f} "
            f"(peak {session.peak_risk:.2f}) "
            f"→ {session.get_trust_level_display()}")

    sentences.append(f"{summary}; {identity}; {risk}.")
    return " ".join(sentences)


def _report_data(session: ExamSession) -> dict:
    """Aggregate everything both the HTML and PDF reports need."""
    events = list(session.events.all())          # Meta orders by created_at
    frames = list(session.frames.all())          # Meta orders by created_at
    chunks = list(session.audio_chunks.all())

    # Audio summary: counts per label + max speech probability.
    label_counts: dict[str, int] = {}
    for chunk in chunks:
        label_counts[chunk.label] = label_counts.get(chunk.label, 0) + 1
    total_chunks = len(chunks)
    audio_summary = {
        "total": total_chunks,
        "labels": [
            {"label": label, "count": count,
             "pct": round(100.0 * count / total_chunks) if total_chunks else 0}
            for label, count in sorted(label_counts.items(),
                                       key=lambda kv: (-kv[1], kv[0]))
        ],
        "max_speech_prob": max((c.speech_prob for c in chunks), default=0.0),
    }

    # Channel stats: totals per channel/severity, in declared channel order.
    raw_stats: dict[str, dict] = {}
    for event in events:
        entry = raw_stats.setdefault(event.channel, {
            "label": CHANNEL_LABELS.get(event.channel, event.channel),
            "info": 0, "suspicious": 0, "violation": 0, "total": 0,
        })
        if event.severity in entry:
            entry[event.severity] += 1
        entry["total"] += 1
    channel_stats = {channel: raw_stats[channel]
                     for channel, _ in MonitoringEvent.CHANNEL_CHOICES
                     if channel in raw_stats}

    # Risk time-series for the timeline chart.
    start = session.started_at
    risk_series = [
        {"t": round((frame.created_at - start).total_seconds(), 1),
         "r": round(frame.risk_score, 4)}
        for frame in frames
    ]

    identity_passed = sum(1 for f in frames if f.identity_verified is True)
    identity_failed = sum(1 for f in frames if f.identity_verified is False)

    duration = _duration_string(_session_end(session, frames, events) - start)
    verdict = _verdict_text(session, events, identity_passed, identity_failed)

    return {
        "events": events,
        "frames": frames,
        "audio_summary": audio_summary,
        "channel_stats": channel_stats,
        "risk_series": risk_series,
        "identity_passed": identity_passed,
        "identity_failed": identity_failed,
        "duration": duration,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

@login_required
def report_list(request):
    """List exam sessions: instructors see all, students only their own."""
    is_instructor = _is_instructor(request.user)
    sessions = (ExamSession.objects
                .select_related("student", "exam")
                .order_by("-started_at"))

    exam_filter = request.GET.get("exam", "")
    trust_filter = request.GET.get("trust", "")

    if is_instructor:
        if exam_filter.isdecimal():
            sessions = sessions.filter(exam_id=int(exam_filter))
        if trust_filter in dict(ExamSession.TRUST_CHOICES):
            sessions = sessions.filter(trust_level=trust_filter)
        exams = Exam.objects.order_by("title")
    else:
        sessions = sessions.filter(student=request.user)
        exams = Exam.objects.none()

    context = {
        "sessions": sessions,
        "is_instructor": is_instructor,
        "exams": exams,
        "trust_choices": ExamSession.TRUST_CHOICES,
        "selected_exam": exam_filter,
        "selected_trust": trust_filter,
    }
    return render(request, "reports/report_list.html", context)


@login_required
def session_report(request, session_id):
    """Detailed integrity report for one exam session."""
    session = get_object_or_404(
        ExamSession.objects.select_related("exam", "student"), pk=session_id
    )
    if not _can_view(request.user, session):
        raise PermissionDenied("You do not have access to this report.")

    data = _report_data(session)
    context = {
        "session": session,
        "exam": session.exam,
        "student": session.student,
        "events": data["events"],
        "frames": data["frames"],
        "audio_summary": data["audio_summary"],
        "channel_stats": data["channel_stats"],
        "risk_series": data["risk_series"],
        "risk_threshold": float(settings.PROCTORING["RISK_VIOLATION"]),
        "identity_passed": data["identity_passed"],
        "identity_failed": data["identity_failed"],
        "duration": data["duration"],
        "verdict": data["verdict"],
    }
    return render(request, "reports/session_report.html", context)


@login_required
def session_report_pdf(request, session_id):
    """Render the full integrity report as a downloadable A4 PDF."""
    session = get_object_or_404(
        ExamSession.objects.select_related("exam", "student"), pk=session_id
    )
    if not _can_view(request.user, session):
        raise PermissionDenied("You do not have access to this report.")

    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    data = _report_data(session)
    events = data["events"]
    frames = data["frames"]
    audio_summary = data["audio_summary"]

    # ---- palette & styles ---------------------------------------------------
    navy = colors.HexColor("#12233d")
    teal = colors.HexColor("#0d9488")
    grid = colors.HexColor("#d7dee8")
    zebra = colors.HexColor("#f4f7fa")
    muted = colors.HexColor("#5b6b80")
    severity_colors = {
        "info": muted,
        "suspicious": colors.HexColor("#b7791f"),
        "violation": colors.HexColor("#c0392b"),
    }
    trust_colors = {
        "high": colors.HexColor("#1a7f4e"),
        "medium": colors.HexColor("#b7791f"),
        "low": colors.HexColor("#c0392b"),
    }

    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9,
                          leading=12, textColor=colors.HexColor("#1f2b3d"))
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold",
                        fontSize=12, leading=15, textColor=navy,
                        spaceBefore=14, spaceAfter=6)
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10)
    caption = ParagraphStyle("caption", parent=cell, textColor=muted)
    verdict_style = ParagraphStyle("verdict", parent=body,
                                   fontName="Helvetica-Oblique",
                                   spaceBefore=8, spaceAfter=2)
    logo_style = ParagraphStyle("logo", parent=body, fontName="Helvetica-Bold",
                                fontSize=22, leading=26, textColor=teal)
    title_style = ParagraphStyle("title", parent=body,
                                 fontName="Helvetica-Bold", fontSize=15,
                                 leading=26, textColor=navy, alignment=2)
    sub_style = ParagraphStyle("sub", parent=body, fontSize=8.5,
                               textColor=muted)
    sev_styles = {
        key: ParagraphStyle(f"sev_{key}", parent=cell,
                            fontName="Helvetica-Bold", textColor=color)
        for key, color in severity_colors.items()
    }
    msg_styles = {
        key: ParagraphStyle(f"msg_{key}", parent=cell, textColor=color)
        for key, color in severity_colors.items()
    }

    doc_width = A4[0] - 72  # 36pt margins on each side

    def kv_table(rows):
        table = Table(rows, colWidths=[135, doc_width - 135])
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), navy),
            ("GRID", (0, 0), (-1, -1), 0.4, grid),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, zebra]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return table

    story = []

    # ---- header -------------------------------------------------------------
    story.append(Table(
        [[Paragraph("RAIM", logo_style),
          Paragraph("Exam Integrity Report", title_style)]],
        colWidths=[doc_width / 2.0, doc_width / 2.0],
        style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, 0), 1.4, teal),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]),
    ))
    story.append(Paragraph(
        "Remote Assessment Integrity Monitoring - multimodal cheating-detection report",
        sub_style))
    story.append(Spacer(1, 12))

    # ---- session metadata ---------------------------------------------------
    student = session.student
    candidate = student.get_full_name() or student.username
    status_text = session.get_status_display()
    if session.status == "terminated" and session.termination_reason:
        status_text += f" - {session.termination_reason}"

    story.append(Paragraph("Session details", h2))
    story.append(kv_table([
        ["Candidate", f"{candidate} ({student.username})"],
        ["Email", student.email or "-"],
        ["Exam", session.exam.title],
        ["Session", f"#{session.id}"],
        ["Started", _fmt_dt(session.started_at)],
        ["Submitted", _fmt_dt(session.submitted_at)],
        ["Duration", data["duration"]],
        ["Status", status_text],
    ]))

    # ---- outcome summary ----------------------------------------------------
    trust_para = Paragraph(
        f"<b>{escape(session.get_trust_level_display())}</b>",
        ParagraphStyle("trust", parent=body,
                       textColor=trust_colors.get(session.trust_level, navy)))
    if audio_summary["total"]:
        audio_line = (
            f"{audio_summary['total']} chunks - "
            + ", ".join(f"{row['label']} x{row['count']}"
                        for row in audio_summary["labels"])
            + f"; max speech prob {audio_summary['max_speech_prob']:.2f}"
        )
    else:
        audio_line = "No audio chunks recorded"

    story.append(Paragraph("Outcome summary", h2))
    story.append(kv_table([
        ["Score",
         f"{session.score:g} / {session.total_marks:g} ({session.percentage}%)"],
        ["Trust level", trust_para],
        ["Risk score",
         f"final {session.risk_score:.2f} - peak {session.peak_risk:.2f}"],
        ["Alert counts",
         f"{session.violation_count} violations, "
         f"{session.suspicious_count} suspicious flags, "
         f"{session.tab_switches} tab switches"],
        ["Identity checks",
         f"{data['identity_passed']} passed / {data['identity_failed']} failed"],
        ["Audio", audio_line],
    ]))
    story.append(Paragraph(
        escape(data["verdict"].replace("→", "->")), verdict_style))

    # ---- risk timeline chart ------------------------------------------------
    story.append(Paragraph("Risk timeline", h2))
    points = [
        (round((frame.created_at - session.started_at).total_seconds(), 1),
         round(frame.risk_score, 4))
        for frame in frames
    ]
    if len(points) >= 2:
        threshold = float(settings.PROCTORING["RISK_VIOLATION"])
        max_t = max(points[-1][0], 1.0)
        drawing = Drawing(doc_width, 190)
        plot = LinePlot()
        plot.x, plot.y = 42, 28
        plot.width, plot.height = doc_width - 60, 140
        plot.data = [points, [(0.0, threshold), (max_t, threshold)]]
        plot.joinedLines = 1
        plot.lines[0].strokeColor = teal
        plot.lines[0].strokeWidth = 1.6
        plot.lines[1].strokeColor = severity_colors["violation"]
        plot.lines[1].strokeWidth = 1.0
        plot.lines[1].strokeDashArray = [4, 3]
        plot.xValueAxis.valueMin = 0
        plot.xValueAxis.valueMax = max_t
        plot.xValueAxis.labelTextFormat = _mmss
        plot.xValueAxis.labels.fontName = "Helvetica"
        plot.xValueAxis.labels.fontSize = 7
        plot.yValueAxis.valueMin = 0.0
        plot.yValueAxis.valueMax = 1.0
        plot.yValueAxis.valueSteps = [0.0, 0.25, 0.5, 0.75, 1.0]
        plot.yValueAxis.labels.fontName = "Helvetica"
        plot.yValueAxis.labels.fontSize = 7
        drawing.add(plot)
        drawing.add(String(
            42, 176,
            f"Risk score over elapsed exam time - dashed line marks the "
            f"violation threshold ({threshold:.2f})",
            fontName="Helvetica", fontSize=7.5, fillColor=muted))
        story.append(drawing)
    else:
        story.append(Paragraph(
            "Not enough frame analyses were recorded to plot a risk timeline.",
            body))

    # ---- events table -------------------------------------------------------
    story.append(Paragraph("Monitoring events", h2))
    if events:
        rows = [["Time", "Channel", "Type", "Severity", "Message"]]
        for event in events:
            rows.append([
                _fmt_time(event.created_at),
                CHANNEL_LABELS.get(event.channel, event.channel),
                Paragraph(escape(event.event_type), cell),
                Paragraph(escape(event.get_severity_display()),
                          sev_styles.get(event.severity, cell)),
                Paragraph(escape(event.message),
                          msg_styles.get(event.severity, cell)),
            ])
        events_table = Table(
            rows, colWidths=[52, 78, 92, 62, doc_width - 284], repeatRows=1)
        events_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, grid),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, zebra]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(events_table)
    else:
        story.append(Paragraph(
            "No monitoring events were recorded during this session.", body))

    # ---- evidence appendix --------------------------------------------------
    story.append(Paragraph("Evidence appendix", h2))
    # Any flagged event with a captured snapshot belongs here, not just
    # violation-severity ones - a sustained gaze_off_screen flag or a
    # suspicious-tier phone/book detection carries evidence too (see
    # api_frame's needs_evidence in proctoring/views.py), and the HTML
    # report already shows all of them regardless of severity.
    snapshots = [e for e in events if e.snapshot and e.snapshot.name]
    evidence_flows = []
    max_w, max_h = 236.0, 170.0
    for event in snapshots[:8]:
        try:
            path = event.snapshot.path
        except (ValueError, NotImplementedError):
            continue
        if not os.path.exists(path):
            continue
        try:
            img_w, img_h = ImageReader(path).getSize()
        except Exception:
            continue
        if not img_w or not img_h:
            continue
        scale = min(max_w / img_w, max_h / img_h)
        image = Image(path, width=img_w * scale, height=img_h * scale)
        text = f"{_fmt_time(event.created_at)} - {event.message}"
        evidence_flows.append([image, Spacer(1, 3),
                               Paragraph(escape(text), caption)])
    if evidence_flows:
        rows = [evidence_flows[i:i + 2]
                for i in range(0, len(evidence_flows), 2)]
        if len(rows[-1]) == 1:
            rows[-1].append("")
        evidence_table = Table(rows, colWidths=[doc_width / 2.0] * 2)
        evidence_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(evidence_table)
    else:
        story.append(Paragraph(
            "No evidence snapshots were captured for this session.", body))

    # ---- footer & build -----------------------------------------------------
    generated = timezone.localtime().strftime("%d %b %Y, %H:%M:%S")

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#8a93a3"))
        canvas.drawString(
            36, 20,
            f"Generated {generated} | Automatically generated by RAIM")
        canvas.drawRightString(A4[0] - 36, 20, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=36, rightMargin=36, topMargin=42, bottomMargin=42,
        title=f"RAIM Exam Integrity Report - Session #{session.id}",
        author="RAIM")
    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)

    response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
    response["Content-Disposition"] = (
        f'attachment; filename="raim_report_session{session.id}.pdf"')
    return response
