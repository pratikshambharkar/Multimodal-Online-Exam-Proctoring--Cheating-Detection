"""URL routes for the exams app (mounted at /exams/)."""
from django.urls import path

from . import views

urlpatterns = [
    path("", views.exam_list, name="exam_list"),
    path("<int:exam_id>/start/", views.exam_start, name="exam_start"),
    path("session/<int:session_id>/calibrate/", views.exam_calibrate, name="exam_calibrate"),
    path("session/<int:session_id>/", views.exam_take, name="exam_take"),
    path("session/<int:session_id>/submit/", views.exam_submit, name="exam_submit"),
    path("session/<int:session_id>/result/", views.exam_result, name="exam_result"),
]
