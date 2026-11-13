from django.contrib.auth.forms import UserCreationForm

from .models import User


class SignupForm(UserCreationForm):
    """UserCreationForm bound to our custom User (the original targets auth.User)."""

    class Meta(UserCreationForm.Meta):
        model = User
