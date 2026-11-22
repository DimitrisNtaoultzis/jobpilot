"""Request/response schemas: the contract between the Django app and the AI service."""

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# How important a missing skill is for the job (shared by the API and the LLM schema)
Priority = Literal["must_have", "nice_to_have"]

# Length limits in characters, so that one request fits Groq's free tier (8K tokens/minute).
# Measured: ~3.5 chars per input token for English text. Budget: 8,000 tokens minus ~1,000
# output, ~500 prompt/schema and ~500 margin leaves ~6,000 tokens (~21,000 chars) for both
# texts. Greek text needs more tokens per character, so the margin matters.
MAX_CV_LENGTH = 12_000
MAX_JOB_DESCRIPTION_LENGTH = 8_000


class AnalyzeRequest(BaseModel):
    """A CV and a job posting to compare."""

    # Strip surrounding whitespace before validation, so "   " fails min_length=1.
    model_config = ConfigDict(str_strip_whitespace=True)

    cv_text: str = Field(min_length=1, max_length=MAX_CV_LENGTH)
    job_title: str = Field(min_length=1, max_length=200)
    job_description: str = Field(min_length=1, max_length=MAX_JOB_DESCRIPTION_LENGTH)


class ResourceMatch(BaseModel):
    """A learning resource found by a search, with its similarity to the query."""

    title: str
    url: str
    description: str
    # Cosine similarity: useful for ranking, NOT a reliable relevance score on its own
    similarity: float


class SkillResources(BaseModel):
    """Learning resources for one missing skill, most similar first."""

    skill: str
    resources: list[ResourceMatch]


class SkillGap(BaseModel):
    """A skill the candidate lacks, and how important it is for the job."""

    skill: str
    priority: Priority


class AnalyzeResponse(BaseModel):
    """The result of the analysis; Django stores it in Analysis.result."""

    required_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    match_score: float = Field(ge=0, le=1)
    # One entry per missing skill, in the same order (empty when nothing is missing).
    skill_gaps: list[SkillGap] = Field(default_factory=list)
    # One entry per missing skill. Empty when nothing is missing, or when the search
    # failed: the skill analysis is still useful without it (graceful degradation).
    learning_resources: list[SkillResources] = Field(default_factory=list)


class AnalysisRun(BaseModel):
    """The answer of /analyze: the analysis, paused for the user's review.

    The thread id is how the client resumes this run later; it is not part of the
    analysis itself, so it stays outside AnalyzeResponse.
    """

    thread_id: uuid.UUID
    analysis: AnalyzeResponse


class ReviewDecision(BaseModel):
    """The user's answer after reviewing the analysis (the body of the resume request)."""

    # forbid: a typo such as "write_leter" is an error, not silently ignored.
    # strict: only a real JSON boolean; no "yes" or 1 turned into True.
    model_config = ConfigDict(extra="forbid", strict=True)

    # Required: an empty body must never reach the graph, since LangGraph reads
    # Command(resume={}) as "answer no interrupt" and the run silently stays paused.
    write_letter: bool


class ResumeResponse(BaseModel):
    """The answer of a resumed run: the cover letter, or None if the user declined."""

    cover_letter: str | None
