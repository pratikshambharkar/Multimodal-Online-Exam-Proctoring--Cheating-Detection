from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone


class Exam(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    duration_minutes = models.PositiveIntegerField(default=30)
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="exams_created"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    @property
    def total_marks(self) -> int:
        return sum(q.marks for q in self.questions.all())


class Question(models.Model):
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name="questions")
    text = models.TextField()
    marks = models.PositiveIntegerField(default=1)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["order", "id"]

    def __str__(self):
        return f"Q{self.order}: {self.text[:60]}"


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name="choices")
    text = models.CharField(max_length=500)
    is_correct = models.BooleanField(default=False)

    def __str__(self):
        return self.text[:60]


class ExamSession(models.Model):
    """One student's attempt at one exam, with its proctoring outcome."""

    STATUS_CHOICES = [
        ("in_progress", "In progress"),
        ("submitted", "Submitted"),
        ("terminated", "Terminated by proctoring"),
    ]
    TRUST_CHOICES = [
        ("high", "High integrity"),
        ("medium", "Needs review"),
        ("low", "Likely violation"),
    ]

    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name="sessions")
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name="exam_sessions")
    started_at = models.DateTimeField(auto_now_add=True)
    submitted_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default="in_progress")
    score = models.FloatField(default=0)
    total_marks = models.FloatField(default=0)

    # Proctoring outcome (Sec. 3.7-3.8 of the manuscript)
    risk_score = models.FloatField(default=0.0)        # current EWMA risk R_t
    peak_risk = models.FloatField(default=0.0)
    violation_count = models.PositiveIntegerField(default=0)
    suspicious_count = models.PositiveIntegerField(default=0)
    tab_switches = models.PositiveIntegerField(default=0)
    trust_level = models.CharField(max_length=8, choices=TRUST_CHOICES, default="high")
    termination_reason = models.CharField(max_length=200, blank=True)

    # Continuous re-verification bookkeeping
    last_verify_at = models.DateTimeField(null=True, blank=True)
    next_verify_gap = models.PositiveIntegerField(default=90)  # seconds

    # Per-user gaze calibration ("look at the screen boundaries" step done
    # once at the start of this attempt). gaze_calibration holds the derived
    # envelope (yaw/pitch/gaze_h/gaze_v min-max + sample counts) consumed by
    # GazeTracker.analyze(); it stays null when calibration was skipped or
    # produced no usable data, in which case the engine's global thresholds
    # apply instead. gaze_calibrated is set True once the step is *attempted*
    # (successful or not) so the exam room does not re-prompt for it.
    gaze_calibrated = models.BooleanField(default=False)
    gaze_calibration = models.JSONField(blank=True, null=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        return f"{self.student.username} - {self.exam.title} ({self.status})"

    @property
    def is_active(self) -> bool:
        return self.status == "in_progress"

    @property
    def ends_at(self):
        return self.started_at + timezone.timedelta(minutes=self.exam.duration_minutes)

    @property
    def percentage(self) -> float:
        return round(100.0 * self.score / self.total_marks, 1) if self.total_marks else 0.0


class Answer(models.Model):
    session = models.ForeignKey(ExamSession, on_delete=models.CASCADE, related_name="answers")
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    selected_choice = models.ForeignKey(Choice, on_delete=models.SET_NULL, null=True, blank=True)
    is_correct = models.BooleanField(default=False)

    class Meta:
        unique_together = [("session", "question")]
