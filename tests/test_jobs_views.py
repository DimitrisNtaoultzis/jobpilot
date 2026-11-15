import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from pytest_django.asserts import assertContains, assertNotContains, assertRedirects

from jobpilot.jobs.models import CV, Analysis, JobPosting


@pytest.mark.parametrize(
    "url_name",
    [
        "jobs:cv_list",
        "jobs:cv_create",
        "jobs:job_list",
        "jobs:job_create",
        "jobs:analysis_list",
        "jobs:analysis_create",
    ],
)
def test_pages_require_login(client, url_name):
    url = reverse(url_name)

    response = client.get(url)

    assertRedirects(response, f"{reverse('accounts:login')}?next={url}")


@pytest.mark.django_db
def test_lists_show_only_the_users_own_objects(client, alice, bob, make_cv_and_job):
    alice_cv, alice_job = make_cv_and_job(alice)
    bob_cv, bob_job = make_cv_and_job(bob)
    Analysis.objects.create(cv=alice_cv, job_posting=alice_job)
    Analysis.objects.create(cv=bob_cv, job_posting=bob_job)
    client.force_login(alice)

    for url_name in ["jobs:cv_list", "jobs:job_list", "jobs:analysis_list"]:
        response = client.get(reverse(url_name))
        assertContains(response, "alice")
        assertNotContains(response, "bob")


@pytest.mark.django_db
def test_cv_create_saves_the_cv_for_the_logged_in_user(client, alice, bob):
    client.force_login(alice)

    # The extra "user" field is a mass-assignment attempt; it must be ignored.
    response = client.post(
        reverse("jobs:cv_create"), {"title": "New CV", "text": "Django", "user": bob.pk}
    )

    assertRedirects(response, reverse("jobs:cv_list"))
    assert CV.objects.get(title="New CV").user == alice


@pytest.mark.django_db
def test_cv_create_from_an_uploaded_pdf(client, alice, make_pdf):
    client.force_login(alice)
    pdf = SimpleUploadedFile("cv.pdf", make_pdf(text="From PDF"), content_type="application/pdf")

    client.post(reverse("jobs:cv_create"), {"title": "PDF CV", "pdf": pdf})

    assert CV.objects.get(title="PDF CV").text == "From PDF"


@pytest.mark.django_db
def test_job_create_saves_the_job_for_the_logged_in_user(client, alice):
    client.force_login(alice)

    response = client.post(
        reverse("jobs:job_create"), {"title": "Backend dev", "description": "Python"}
    )

    assertRedirects(response, reverse("jobs:job_list"))
    assert JobPosting.objects.get(title="Backend dev").user == alice


@pytest.mark.django_db
def test_analysis_create_with_own_objects_is_pending(client, alice, make_cv_and_job):
    cv, job = make_cv_and_job(alice)
    client.force_login(alice)

    response = client.post(reverse("jobs:analysis_create"), {"cv": cv.pk, "job_posting": job.pk})

    assertRedirects(response, reverse("jobs:analysis_list"))
    assert Analysis.objects.get().status == Analysis.Status.PENDING


@pytest.mark.django_db
def test_analysis_create_rejects_another_users_cv(client, alice, bob, make_cv_and_job):
    _, alice_job = make_cv_and_job(alice)
    bob_cv, _ = make_cv_and_job(bob)
    client.force_login(alice)

    response = client.post(
        reverse("jobs:analysis_create"), {"cv": bob_cv.pk, "job_posting": alice_job.pk}
    )

    assert response.status_code == 200  # form shown again with errors
    assert not Analysis.objects.exists()
