from django.db import models

from exams.models import ExamSession


class MonitoringEvent(models.Model):
    """A discrete proctoring event: info, suspicious flag, or violation.

    Maps to the three-tier alert policy in Sec. 3.8 of the manuscript.
    """

    CHANNEL_CHOICES = [
        ("identity", "Identity verification"),
        ("gaze", "Gaze / head pose"),
        ("objects", "Object detection"),
        ("cnn", "CNN behavior classifier"),
        ("audio", "Audio analysis"),
        ("browser", "Browser activity"),
        ("system", "System"),
    ]
    SEVERITY_CHOICES = [
        ("info", "Info"),
        ("suspicious", "Suspicious"),
        ("violation", "Violation"),
    ]

    session = models.ForeignKey(ExamSession, on_delete=models.CASCADE, related_name="events")
    created_at = models.DateTimeField(auto_now_add=True)
    channel = models.CharField(max_length=12, choices=CHANNEL_CHOICES)
    event_type = models.CharField(max_length=50)   # e.g. "phone_detected", "tab_switch"
    severity = models.CharField(max_length=12, choices=SEVERITY_CHOICES, default="info")
    message = models.CharField(max_length=300)
    confidence = models.FloatField(default=0.0)
    details = models.JSONField(blank=True, null=True)
    snapshot = models.ImageField(upload_to="evidence/", blank=True, null=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"[{self.severity}] {self.event_type} @ {self.created_at:%H:%M:%S}"


class FrameAnalysis(models.Model):
    """Per-frame fused feature record — feeds the risk-score timeline chart."""

    session = models.ForeignKey(ExamSession, on_delete=models.CASCADE, related_name="frames")
    created_at = models.DateTimeField(auto_now_add=True)

    face_count = models.IntegerField(default=0)
    identity_verified = models.BooleanField(null=True)   # null = not checked this frame
    identity_distance = models.FloatField(null=True)
    yaw = models.FloatField(default=0.0)
    pitch = models.FloatField(default=0.0)
    roll = models.FloatField(default=0.0)
    ear = models.FloatField(default=0.0)
    looking_away = models.BooleanField(default=False)
    gaze_off_fraction = models.FloatField(default=0.0)   # windowed statistic
    phone_conf = models.FloatField(default=0.0)
    book_conf = models.FloatField(default=0.0)
    extra_persons = models.IntegerField(default=0)
    cnn_cheat_prob = models.FloatField(default=0.0)
    audio_speech_prob = models.FloatField(default=0.0)   # last known audio score
    svm_label = models.CharField(max_length=12, default="normal")
    risk_score = models.FloatField(default=0.0)

    class Meta:
        ordering = ["created_at"]
        verbose_name_plural = "frame analyses"


class AudioAnalysis(models.Model):
    """Result of one analyzed microphone chunk (Sec. 3.5)."""

    session = models.ForeignKey(ExamSession, on_delete=models.CASCADE, related_name="audio_chunks")
    created_at = models.DateTimeField(auto_now_add=True)
    rms = models.FloatField(default=0.0)
    speech_prob = models.FloatField(default=0.0)
    label = models.CharField(max_length=20, default="silence")  # silence|noise|speech|conversation

    class Meta:
        ordering = ["created_at"]
        verbose_name_plural = "audio analyses"
