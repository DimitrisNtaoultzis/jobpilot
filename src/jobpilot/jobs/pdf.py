"""Text extraction from uploaded PDF files.

Kept free of Django code so it is easy to test and to reuse (e.g. by the AI service).
"""

from pypdf import PdfReader


class PDFTextError(Exception):
    """No usable text could be read from an uploaded PDF."""


def extract_pdf_text(file) -> str:
    """Return the text of a PDF file object, or raise PDFTextError."""
    try:
        reader = PdfReader(file)
        # Many PDFs are "encrypted" only to restrict printing: they open with an empty password.
        # decrypt() returns PasswordType.NOT_DECRYPTED (== 0, falsy) when that fails.
        if reader.is_encrypted and not reader.decrypt(""):
            raise PDFTextError("This PDF is password-protected.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except PDFTextError:
        raise
    except Exception as exc:
        # The file is untrusted input: a broken or malicious PDF can make the parser raise
        # almost anything. Turn it into a clear error instead of a 500, keeping the cause.
        raise PDFTextError("This file could not be read as a PDF.") from exc

    text = text.strip()
    if not text:
        raise PDFTextError(
            "No text found in this PDF (it may be a scanned image). Please paste the text instead."
        )
    return text
