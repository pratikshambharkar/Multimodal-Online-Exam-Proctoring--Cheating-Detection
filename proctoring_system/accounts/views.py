"""Views for the accounts app: landing page, face-enrolled registration,
face-verified login, logout and the role-aware dashboard."""
import base64
import binascii
import logging

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.db import transaction
from django.shortcuts import redirect, render
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from exams.models import Exam, ExamSession

from .forms import LoginForm, RegisterForm
from .models import Profile

logger = logging.getLogger(__name__)

ENGINE_UNAVAILABLE_MSG = (
    "The face-recognition engine is not available right now (its components may "
    "still be installing). Please try again in a few minutes."
)


def _decode_face_data(face_data):
    """Decode a JPEG data-URL captured by the webcam widget.

    Returns a ``(bgr_image, jpeg_bytes, error)`` tuple. ``error`` is an empty
    string on success; on failure the first two items are ``None``.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None, None, ENGINE_UNAVAILABLE_MSG

    if not face_data or "," not in face_data:
        return None, None, (
            "Face capture is required. Use the webcam widget to take your "
            "photo before submitting."
        )
    header, _, payload = face_data.partition(",")
    if not header.startswith("data:image"):
        return None, None, "The captured face image is not in a supported format."
    try:
        jpeg_bytes = base64.b64decode(payload)
    except (binascii.Error, ValueError):
        return None, None, "The captured face image could not be decoded. Please retake it."
    buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, "The captured face image could not be read. Please retake it."
    return bgr, jpeg_bytes, ""


def _safe_next_url(request):
    """Return a validated ``next`` redirect target, or an empty string."""
    candidate = request.POST.get("next") or request.GET.get("next") or ""
    if candidate and url_has_allowed_host_and_scheme(
        candidate,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return candidate
    return ""


def home(request):
    """Public landing page."""
    return render(request, "home.html")


def register(request):
    """Create a student account with mandatory webcam face enrollment."""
    form = RegisterForm(request.POST or None)
    face_data = request.POST.get("face_data", "").strip()

    if request.method == "POST" and form.is_valid():
        bgr, jpeg_bytes, error = _decode_face_data(face_data)
        embedding = None
        if not error:
            try:
                from proctoring.engine.face_verifier import FaceVerifier

                embedding, enroll_error = FaceVerifier().enroll(bgr)
            except Exception:
                logger.exception("Face enrollment engine failure during registration")
                embedding, enroll_error = None, ENGINE_UNAVAILABLE_MSG
            if enroll_error or embedding is None:
                error = enroll_error or (
                    "Your face could not be enrolled from the captured photo. "
                    "Please retake it in good lighting."
                )
        if error:
            messages.error(request, error)
        else:
            with transaction.atomic():
                user = User.objects.create_user(
                    username=form.cleaned_data["username"],
                    email=form.cleaned_data["email"],
                    password=form.cleaned_data["password1"],
                    first_name=form.cleaned_data["first_name"],
                    last_name=form.cleaned_data["last_name"],
                )
                profile = Profile.objects.create(
                    user=user,
                    role="student",
                    face_embedding=embedding,
                    face_registered=True,
                )
                profile.face_image.save(
                    f"{user.username}.jpg", ContentFile(jpeg_bytes), save=True
                )
            login(request, user)
            messages.success(
                request,
                f"Welcome to RAIM, {user.first_name}! Your face has been enrolled "
                "for identity verification.",
            )
            return redirect("dashboard")

    context = {"form": form, "face_data": face_data}
    return render(request, "accounts/register.html", context)


def _verify_login_face(request, profile):
    """Verify the login webcam capture against the enrolled embedding.

    Returns an error message string, or an empty string when verification
    succeeds.
    """
    face_data = request.POST.get("face_data", "").strip()
    if not face_data:
        return (
            "Face verification is required for this account. Use the webcam "
            "widget to capture your face, then sign in again."
        )
    bgr, _, error = _decode_face_data(face_data)
    if error:
        return error
    try:
        from proctoring.engine.face_verifier import FaceVerifier

        result = FaceVerifier().verify(bgr, profile.face_embedding)
    except Exception:
        logger.exception("Face verification engine failure during login")
        return ENGINE_UNAVAILABLE_MSG
    if result.get("available") is False:
        return ENGINE_UNAVAILABLE_MSG
    if not result.get("face_found"):
        return "No face visible to the camera."
    if not result.get("verified"):
        return "Face verification failed — identity does not match our records."
    return ""


def login_view(request):
    """Log in with username or email; enrolled students must also pass face
    verification. Accounts without an enrolled face (e.g. seeded instructors)
    may sign in with password only."""
    if request.user.is_authenticated:
        return redirect("dashboard")

    form = LoginForm(request.POST or None)
    face_data = request.POST.get("face_data", "").strip()

    if request.method == "POST" and form.is_valid():
        identifier = form.cleaned_data["identifier"].strip()
        password = form.cleaned_data["password"]

        username = identifier
        if "@" in identifier:
            email_match = (
                User.objects.filter(email__iexact=identifier).order_by("id").first()
            )
            if email_match:
                username = email_match.username

        user = authenticate(request, username=username, password=password)
        if user is None:
            messages.error(request, "Invalid username/email or password.")
        else:
            profile = Profile.objects.filter(user=user).first()
            error = ""
            if profile and profile.face_registered and profile.face_embedding:
                error = _verify_login_face(request, profile)
            if error:
                messages.error(request, error)
            else:
                login(request, user)
                return redirect(_safe_next_url(request) or "dashboard")

    context = {
        "form": form,
        "face_data": face_data,
        "next": request.POST.get("next") or request.GET.get("next") or "",
    }
    return render(request, "accounts/login.html", context)


@require_POST
def logout_view(request):
    """Log the user out (POST only) and return to the landing page."""
    logout(request)
    messages.info(request, "You have been signed out.")
    return redirect("home")


@login_required
def dashboard(request):
    """Role-aware dashboard: monitoring overview for instructors, available
    exams and attempt history for students."""
    default_role = "instructor" if request.user.is_staff else "student"
    profile, _ = Profile.objects.get_or_create(
        user=request.user, defaults={"role": default_role}
    )

    if profile.is_instructor:
        recent_sessions = (
            ExamSession.objects.select_related("exam", "student")
            .order_by("-started_at")[:25]
        )
        context = {
            "profile": profile,
            "recent_sessions": recent_sessions,
            "exams": Exam.objects.order_by("-created_at"),
            "total_sessions": ExamSession.objects.count(),
            "flagged_sessions": ExamSession.objects.filter(
                trust_level__in=["low", "medium"]
            ).count(),
        }
    else:
        context = {
            "profile": profile,
            "exams": Exam.objects.filter(is_active=True).order_by("-created_at"),
            "my_sessions": (
                ExamSession.objects.filter(student=request.user)
                .select_related("exam")
                .order_by("-started_at")[:10]
            ),
            "face_registered": profile.face_registered,
        }
    return render(request, "accounts/dashboard.html", context)
