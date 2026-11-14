from django import forms

from .models import CV, Analysis, JobPosting


class AnalysisForm(forms.ModelForm):
    """Pick one of your CVs and one of your job postings.

    Limiting the querysets is also the same-user check: the form rejects any id
    outside them ("Select a valid choice"), even in a hand-crafted POST.
    """

    class Meta:
        model = Analysis
        fields = ["cv", "job_posting"]

    def __init__(self, *args, user, **kwargs):
        # `user` is keyword-only and required, so forgetting it fails loudly.
        super().__init__(*args, **kwargs)
        self.fields["cv"].queryset = CV.objects.filter(user=user)
        self.fields["job_posting"].queryset = JobPosting.objects.filter(user=user)
