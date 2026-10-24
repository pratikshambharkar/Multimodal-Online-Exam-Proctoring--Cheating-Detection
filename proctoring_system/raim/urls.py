"""Root URL configuration for RAIM."""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("accounts.urls")),
    path("exams/", include("exams.urls")),
    path("proctoring/", include("proctoring.urls")),
    path("reports/", include("reports.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

admin.site.site_header = "RAIM Administration"
admin.site.site_title = "RAIM Admin"
admin.site.index_title = "Remote Assessment Integrity Monitoring"
