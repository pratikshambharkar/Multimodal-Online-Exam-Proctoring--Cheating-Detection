"""Forms for RAIM account registration and login."""
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.core.exceptions import ValidationError


class RegisterForm(forms.Form):
    """Student self-registration form.

    The webcam face capture travels alongside this form as a hidden
    ``face_data`` input (JPEG data-URL) and is processed by the view.
    """

    first_name = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            "class": "form-input",
            "placeholder": "First name",
            "autocomplete": "given-name",
        }),
    )
    last_name = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            "class": "form-input",
            "placeholder": "Last name",
            "autocomplete": "family-name",
        }),
    )
    username = forms.CharField(
        max_length=150,
        validators=[UnicodeUsernameValidator()],
        help_text="Letters, digits and @/./+/-/_ only.",
        widget=forms.TextInput(attrs={
            "class": "form-input",
            "placeholder": "Username",
            "autocomplete": "username",
        }),
    )
    email = forms.EmailField(
        max_length=254,
        widget=forms.EmailInput(attrs={
            "class": "form-input",
            "placeholder": "you@example.com",
            "autocomplete": "email",
        }),
    )
    password1 = forms.CharField(
        label="Password",
        min_length=8,
        help_text="At least 8 characters. Must not be too common, entirely numeric, "
                  "or too similar to your personal details.",
        widget=forms.PasswordInput(attrs={
            "class": "form-input",
            "placeholder": "Password",
            "autocomplete": "new-password",
        }),
    )
    password2 = forms.CharField(
        label="Confirm password",
        widget=forms.PasswordInput(attrs={
            "class": "form-input",
            "placeholder": "Repeat password",
            "autocomplete": "new-password",
        }),
    )

    def clean_username(self):
        """Reject usernames that are already taken (case-insensitive)."""
        username = self.cleaned_data["username"].strip()
        if User.objects.filter(username__iexact=username).exists():
            raise ValidationError("This username is already taken.")
        return username

    def clean_email(self):
        """Require a unique email address (case-insensitive)."""
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError("An account with this email address already exists.")
        return email

    def clean(self):
        """Check the two passwords match, then run Django's password validators."""
        cleaned = super().clean()
        password1 = cleaned.get("password1")
        password2 = cleaned.get("password2")
        if password1 and password2 and password1 != password2:
            self.add_error("password2", "The two password fields do not match.")
        elif password1:
            reference_user = User(
                username=cleaned.get("username", ""),
                email=cleaned.get("email", ""),
                first_name=cleaned.get("first_name", ""),
                last_name=cleaned.get("last_name", ""),
            )
            try:
                validate_password(password1, user=reference_user)
            except ValidationError as exc:
                self.add_error("password1", exc)
        return cleaned


class LoginForm(forms.Form):
    """Login with either username or email plus password.

    Face verification for enrolled students is handled by the view using the
    hidden ``face_data`` input.
    """

    identifier = forms.CharField(
        label="Username or email",
        max_length=254,
        widget=forms.TextInput(attrs={
            "class": "form-input",
            "placeholder": "Username or email",
            "autocomplete": "username",
            "autofocus": "autofocus",
        }),
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            "class": "form-input",
            "placeholder": "Password",
            "autocomplete": "current-password",
        }),
    )
