"""Shared pytest fixtures. Fixtures defined here are available to every test, without imports."""

import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def _build_pdf(text=None, user_password=None, owner_password=None) -> bytes:
    """Build a one-page PDF in memory. text=None gives a page without text (like a scan).

    Keep `text` simple: parentheses and backslashes would need escaping in a PDF string.
    """
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)  # US Letter, in points
    if text is not None:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        content = DecodedStreamObject()
        # PDF drawing commands: begin text, font F1 size 12, move to (72, 720), show text, end
        content.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        page.replace_contents(content)
    if user_password is not None or owner_password is not None:
        writer.encrypt(user_password=user_password or "", owner_password=owner_password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def make_pdf():
    """Factory fixture: make_pdf(text=..., user_password=..., owner_password=...) -> PDF bytes."""
    return _build_pdf
