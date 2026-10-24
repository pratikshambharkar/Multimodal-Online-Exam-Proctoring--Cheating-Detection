from django.contrib import admin

from .models import AudioAnalysis, FrameAnalysis, MonitoringEvent


@admin.register(MonitoringEvent)
class MonitoringEventAdmin(admin.ModelAdmin):
    list_display = ("session", "channel", "event_type", "severity", "confidence", "created_at")
    list_filter = ("channel", "severity", "event_type")


@admin.register(FrameAnalysis)
class FrameAnalysisAdmin(admin.ModelAdmin):
    list_display = ("session", "face_count", "svm_label", "risk_score", "created_at")
    list_filter = ("svm_label",)


@admin.register(AudioAnalysis)
class AudioAnalysisAdmin(admin.ModelAdmin):
    list_display = ("session", "label", "rms", "speech_prob", "created_at")
    list_filter = ("label",)
