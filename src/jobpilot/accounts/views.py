from django.contrib.auth import login
from django.urls import reverse_lazy
from django.views.generic import CreateView

from .forms import SignupForm


class SignupView(CreateView):
    """Create a new account and log the user in right away."""

    form_class = SignupForm
    template_name = "accounts/signup.html"
    # reverse_lazy: class attributes are evaluated at import time, before URLs are loaded.
    success_url = reverse_lazy("home")

    def form_valid(self, form):
        response = super().form_valid(form)  # saves the new user as self.object
        login(self.request, self.object)
        return response
