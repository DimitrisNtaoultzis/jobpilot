"""Request/response schemas: the contract between the Django app and the AI service."""

from pydantic import BaseModel, ConfigDict, Field

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


class AnalyzeResponse(BaseModel):
    """The result of the analysis; Django stores it in Analysis.result."""

    required_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    match_score: float = Field(ge=0, le=1)
    # One entry per missing skill. Empty when nothing is missing, or when the search
    # failed: the skill analysis is still useful without it (graceful degradation).
    learning_resources: list[SkillResources] = Field(default_factory=list)
