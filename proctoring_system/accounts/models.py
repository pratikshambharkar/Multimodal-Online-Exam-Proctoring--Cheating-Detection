from django.contrib.auth.models import User
from django.db import models


class Profile(models.Model):
    """Extends the built-in User with a role and enrolled face biometrics."""

    ROLE_CHOICES = [
        ("student", "Student"),
        ("instructor", "Instructor"),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    role = models.CharField(max_length=12, choices=ROLE_CHOICES, default="student")
    face_image = models.ImageField(upload_to="faces/", blank=True, null=True)
    # 128-d Facenet embedding computed by DeepFace at registration time.
    face_embedding = models.JSONField(blank=True, null=True)
    face_registered = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} ({self.role})"

    @property
    def is_instructor(self) -> bool:
        return self.role == "instructor" or self.user.is_staff
