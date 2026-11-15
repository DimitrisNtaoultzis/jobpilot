import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from jobpilot.jobs.forms import AnalysisForm, CVForm


def pdf_upload(content, name="cv.pdf"):
    return SimpleUploadedFile(name, content, content_type="application/pdf")


# --- CVForm ---


def test_cv_form_accepts_pasted_text():
    form = CVForm(data={"title": "My CV", "text": "Python developer"})

    assert form.is_valid(), form.errors
    assert form.cleaned_data["text"] == "Python developer"


def test_cv_form_uses_the_text_of_an_uploaded_pdf(make_pdf):
    form = CVForm(data={"title": "My CV"}, files={"pdf": pdf_upload(make_pdf(text="From PDF"))})

    assert form.is_valid(), form.errors
    assert form.cleaned_data["text"] == "From PDF"


def test_cv_form_pdf_wins_over_pasted_text(make_pdf):
    form = CVForm(
        data={"title": "My CV", "text": "Pasted"},
        files={"pdf": pdf_upload(make_pdf(text="From PDF"))},
    )

    assert form.is_valid(), form.errors
    assert form.cleaned_data["text"] == "From PDF"


def test_cv_form_requires_text_or_pdf():
    form = CVForm(data={"title": "My CV"})

    assert form.errors == {"text": ["Paste your CV text or upload a PDF."]}


def test_cv_form_rejects_a_non_pdf_with_a_single_error():
    form = CVForm(data={"title": "My CV"}, files={"pdf": SimpleUploadedFile("cv.docx", b"x")})

    assert form.errors == {"pdf": ["Please upload a .pdf file."]}


def test_cv_form_rejects_a_too_large_file(make_pdf, monkeypatch):
    # Shrink the limit instead of building a 5 MB file; monkeypatch restores it afterwards.
    monkeypatch.setattr("jobpilot.jobs.forms.MAX_PDF_SIZE", 10)  # bytes
    form = CVForm(data={"title": "My CV"}, files={"pdf": pdf_upload(make_pdf(text="Big"))})

    assert list(form.errors) == ["pdf"]
    assert "larger than" in form.errors["pdf"][0]


def test_cv_form_shows_the_pdf_reading_error(make_pdf):
    form = CVForm(data={"title": "My CV"}, files={"pdf": pdf_upload(make_pdf())})

    assert list(form.errors) == ["pdf"]
    assert "No text found" in form.errors["pdf"][0]


# --- AnalysisForm ---


@pytest.mark.django_db
def test_analysis_form_offers_only_the_users_own_objects(alice, bob, make_cv_and_job):
    alice_cv, alice_job = make_cv_and_job(alice)
    make_cv_and_job(bob)

    form = AnalysisForm(user=alice)

    assert list(form.fields["cv"].queryset) == [alice_cv]
    assert list(form.fields["job_posting"].queryset) == [alice_job]


@pytest.mark.django_db
def test_analysis_form_accepts_own_cv_and_job(alice, make_cv_and_job):
    cv, job = make_cv_and_job(alice)

    form = AnalysisForm(data={"cv": cv.pk, "job_posting": job.pk}, user=alice)

    assert form.is_valid(), form.errors


@pytest.mark.django_db
@pytest.mark.parametrize("field", ["cv", "job_posting"])
def test_analysis_form_rejects_another_users_object(alice, bob, make_cv_and_job, field):
    alice_cv, alice_job = make_cv_and_job(alice)
    bob_cv, bob_job = make_cv_and_job(bob)
    data = {"cv": alice_cv.pk, "job_posting": alice_job.pk}
    data[field] = {"cv": bob_cv, "job_posting": bob_job}[field].pk  # swap in bob's object

    form = AnalysisForm(data=data, user=alice)

    assert list(form.errors) == [field]
    assert "Select a valid choice" in form.errors[field][0]


def test_analysis_form_requires_a_user():
    with pytest.raises(TypeError):
        AnalysisForm()
