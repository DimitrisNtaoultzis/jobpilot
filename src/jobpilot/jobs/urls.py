from django.urls import path

from . import views

app_name = "jobs"

urlpatterns = [
    path("cvs/", views.CVListView.as_view(), name="cv_list"),
    path("cvs/new/", views.CVCreateView.as_view(), name="cv_create"),
    path("jobs/", views.JobPostingListView.as_view(), name="job_list"),
    path("jobs/new/", views.JobPostingCreateView.as_view(), name="job_create"),
    path("analyses/", views.AnalysisListView.as_view(), name="analysis_list"),
    path("analyses/new/", views.AnalysisCreateView.as_view(), name="analysis_create"),
]
