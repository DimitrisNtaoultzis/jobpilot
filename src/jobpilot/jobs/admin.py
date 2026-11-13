from django.contrib import admin

from .models import CV, Analysis, JobPosting


@admin.register(CV)
class CVAdmin(admin.ModelAdmin):
    list_display = ["title", "user", "created_at"]
    search_fields = ["title", "user__username"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(JobPosting)
class JobPostingAdmin(admin.ModelAdmin):
    list_display = ["title", "company", "user", "created_at"]
    search_fields = ["title", "company", "user__username"]
    readonly_fields = ["created_at"]


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ["cv", "job_posting", "status", "created_at"]
    list_filter = ["status"]
    # __str__ of Analysis touches cv and job_posting; fetch them with a JOIN
    # instead of 2 extra queries per row (the N+1 problem).
    list_select_related = ["cv", "job_posting"]
    readonly_fields = ["created_at", "updated_at"]
