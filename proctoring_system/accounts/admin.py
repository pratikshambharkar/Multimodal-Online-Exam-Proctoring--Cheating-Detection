from django.contrib import admin

from .models import Profile


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "role", "face_registered", "created_at")
    list_filter = ("role", "face_registered")
    search_fields = ("user__username", "user__email")
