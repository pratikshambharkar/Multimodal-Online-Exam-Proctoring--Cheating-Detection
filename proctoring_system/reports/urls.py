"""URL routes for the reports app (mounted at /reports/)."""
from django.urls import path

from . import views

urlpatterns = [
    path("", views.report_list, name="report_list"),
    path("session/<int:session_id>/", views.session_report, name="session_report"),
    path("session/<int:session_id>/pdf/", views.session_report_pdf, name="session_report_pdf"),
]
