"""URL routes for the proctoring API (mounted at /proctoring/)."""
from django.urls import path

from . import views

urlpatterns = [
    path("api/frame/", views.api_frame, name="api_frame"),
    path("api/audio/", views.api_audio, name="api_audio"),
    path("api/event/", views.api_event, name="api_event"),
    path("api/status/<int:session_id>/", views.api_status, name="api_status"),
    path(
        "api/calibration/sample/",
        views.api_calibration_sample,
        name="api_calibration_sample",
    ),
    path(
        "api/calibration/finish/",
        views.api_calibration_finish,
        name="api_calibration_finish",
    ),
]
