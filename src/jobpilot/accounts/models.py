from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    """Custom user model.

    Empty for now: it behaves exactly like Django's default User, but it can be
    extended later (e.g. profile fields) without a painful migration.
    """
