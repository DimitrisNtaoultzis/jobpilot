from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import CreateView, ListView

from .forms import AnalysisForm, CVForm
from .models import CV, Analysis, JobPosting

# LoginRequiredMixin must come first (leftmost), so its check runs before the view's logic.


class CVListView(LoginRequiredMixin, ListView):
    template_name = "jobs/cv_list.html"
    context_object_name = "cvs"

    def get_queryset(self):
        # Each user sees only their own CVs.
        return CV.objects.filter(user=self.request.user)


class CVCreateView(LoginRequiredMixin, CreateView):
    form_class = CVForm
    template_name = "jobs/cv_form.html"
    success_url = reverse_lazy("jobs:cv_list")

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)


class JobPostingListView(LoginRequiredMixin, ListView):
    template_name = "jobs/job_list.html"
    context_object_name = "jobs"

    def get_queryset(self):
        return JobPosting.objects.filter(user=self.request.user)


class JobPostingCreateView(LoginRequiredMixin, CreateView):
    model = JobPosting
    fields = ["title", "company", "url", "description"]
    template_name = "jobs/job_form.html"
    success_url = reverse_lazy("jobs:job_list")

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)


class AnalysisListView(LoginRequiredMixin, ListView):
    template_name = "jobs/analysis_list.html"
    context_object_name = "analyses"

    def get_queryset(self):
        # Analysis has no user field; ownership goes through its CV.
        # select_related avoids N+1 queries when the template prints cv and job_posting.
        return Analysis.objects.filter(cv__user=self.request.user).select_related(
            "cv", "job_posting"
        )


class AnalysisCreateView(LoginRequiredMixin, CreateView):
    form_class = AnalysisForm
    template_name = "jobs/analysis_form.html"
    success_url = reverse_lazy("jobs:analysis_list")

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs
