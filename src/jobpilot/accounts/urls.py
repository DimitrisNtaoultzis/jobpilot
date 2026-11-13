from django.contrib.auth import views as auth_views
from django.urls import path

from .views import SignupView

app_name = "accounts"

urlpatterns = [
    path("signup/", SignupView.as_view(), name="signup"),
    path(
        "login/",
        auth_views.LoginView.as_view(
            template_name="accounts/login.html",
            redirect_authenticated_user=True,
        ),
        name="login",
    ),
    # LogoutView accepts only POST (since Django 5.0), so logout is a form button.
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
]
