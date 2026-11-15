from django import forms

from .models import CV, Analysis, JobPosting
from .pdf import PDFTextError, extract_pdf_text

MAX_PDF_SIZE = 5 * 1024 * 1024  # 5 MB


class CVForm(forms.ModelForm):
    """Create a CV from pasted text or from an uploaded PDF (the PDF wins if both are given).

    The PDF itself is not stored: only its extracted text ends up in CV.text.
    """

    # Not a model field: it exists only in the form.
    pdf = forms.FileField(
        required=False,
        label="Or upload a PDF",
        help_text="Its text replaces the text above. Max 5 MB.",
        # `accept` only filters the browser's file picker; it is not a security check.
        widget=forms.FileInput(attrs={"accept": ".pdf,application/pdf"}),
    )

    class Meta:
        model = CV
        fields = ["title", "text", "pdf"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["text"].required = False  # either text or a PDF, checked in clean()

    def clean_pdf(self):
        # The file name is user-controlled, so the extension check is only for a friendly
        # message. The real protection is the size limit plus the parser rejecting non-PDFs.
        pdf = self.cleaned_data["pdf"]
        if pdf:
            if not pdf.name.lower().endswith(".pdf"):
                raise forms.ValidationError("Please upload a .pdf file.")
            if pdf.size > MAX_PDF_SIZE:
                raise forms.ValidationError("The file is larger than 5 MB.")
        return pdf

    def clean(self):
        cleaned_data = super().clean()
        pdf = cleaned_data.get("pdf")
        if pdf:
            try:
                # ModelForm copies cleaned_data to the instance after clean(), so this
                # extracted text is what gets saved in CV.text.
                cleaned_data["text"] = extract_pdf_text(pdf)
            except PDFTextError as exc:
                self.add_error("pdf", str(exc))
        elif "pdf" not in self.errors and not cleaned_data.get("text"):
            # Skip this when the PDF was already rejected, to avoid two confusing errors.
            self.add_error("text", "Paste your CV text or upload a PDF.")
        return cleaned_data


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
