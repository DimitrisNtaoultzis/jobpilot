"""Request/response schemas: the contract between the Django app and the AI service."""

from pydantic import BaseModel, ConfigDict, Field

MAX_TEXT_LENGTH = 50_000  # characters; keeps LLM requests (and their cost) bounded


class AnalyzeRequest(BaseModel):
    """A CV and a job posting to compare."""

    # Strip surrounding whitespace before validation, so "   " fails min_length=1.
    model_config = ConfigDict(str_strip_whitespace=True)

    cv_text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    job_title: str = Field(min_length=1, max_length=200)
    job_description: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class AnalyzeResponse(BaseModel):
    """The result of the analysis; Django stores it in Analysis.result."""

    required_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    match_score: float = Field(ge=0, le=1)
