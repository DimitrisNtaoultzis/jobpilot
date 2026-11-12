from django.conf import settings
from django.db import models


class CV(models.Model):
    """A CV uploaded by a user. For now it stores only the CV text."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="cvs")
    title = models.CharField(max_length=200)  # e.g. "Backend CV 2026"
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "CV"
        verbose_name_plural = "CVs"

    def __str__(self) -> str:
        return self.title


class JobPosting(models.Model):
    """A job posting that the user wants to compare against their CV."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="job_postings"
    )
    title = models.CharField(max_length=200)
    company = models.CharField(max_length=200, blank=True)
    url = models.URLField(blank=True)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.title} @ {self.company}" if self.company else self.title


class Analysis(models.Model):
    """The result of comparing one CV against one job posting (filled in by the AI service)."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    cv = models.ForeignKey(CV, on_delete=models.CASCADE, related_name="analyses")
    job_posting = models.ForeignKey(JobPosting, on_delete=models.CASCADE, related_name="analyses")
    status = models.CharField(max_length=20, choices=Status, default=Status.PENDING)
    # Skills, gaps, learning plan, ... (the exact shape is defined in Phase 2).
    # NULL means "no result yet", which is different from an empty result.
    result = models.JSONField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "analyses"

    def __str__(self) -> str:
        return f"{self.cv} vs {self.job_posting} ({self.status})"
