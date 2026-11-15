import io

import pytest

from jobpilot.jobs.pdf import PDFTextError, extract_pdf_text


def test_extracts_text(make_pdf):
    pdf = io.BytesIO(make_pdf(text="Python developer"))

    assert extract_pdf_text(pdf) == "Python developer"


def test_pdf_without_text_is_rejected(make_pdf):
    pdf = io.BytesIO(make_pdf())

    with pytest.raises(PDFTextError, match="No text found"):
        extract_pdf_text(pdf)


def test_non_pdf_is_rejected_and_keeps_the_cause():
    not_a_pdf = io.BytesIO(b"just some plain text")

    with pytest.raises(PDFTextError, match="could not be read") as exc_info:
        extract_pdf_text(not_a_pdf)

    assert exc_info.value.__cause__ is not None


def test_password_protected_pdf_is_rejected(make_pdf):
    pdf = io.BytesIO(make_pdf(text="Secret", user_password="pw"))

    with pytest.raises(PDFTextError, match="password-protected"):
        extract_pdf_text(pdf)


def test_pdf_with_only_an_owner_password_is_read(make_pdf):
    pdf = io.BytesIO(make_pdf(text="Printing restricted", owner_password="owner"))

    assert extract_pdf_text(pdf) == "Printing restricted"
