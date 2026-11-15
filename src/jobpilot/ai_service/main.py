"""JobPilot AI service: the FastAPI app that runs the analysis (LLM, agent).

Run locally with:
    uv run uvicorn jobpilot.ai_service.main:app --reload --port 8001
"""

import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.language_models.chat_models import BaseChatModel

from .extraction import extract_skills
from .llm import get_chat_model
from .matching import compare_skills
from .schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="JobPilot AI service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check: the service is up and answering requests."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    payload: AnalyzeRequest,
    # Dependency injection: tests replace get_chat_model via app.dependency_overrides.
    model: Annotated[BaseChatModel, Depends(get_chat_model)],
) -> AnalyzeResponse:
    """Compare a CV with a job posting: the LLM extracts the skills, plain code compares them."""
    try:
        extraction = await extract_skills(
            model, payload.cv_text, payload.job_title, payload.job_description
        )
    except Exception as exc:
        # Service boundary: log the full traceback, return a safe message to the client.
        # Log the error, never the texts: CVs contain personal data.
        logger.exception("Skill extraction failed")
        raise HTTPException(
            status_code=502, detail="The AI model could not analyze this request."
        ) from exc
    return compare_skills(extraction.job_skills, extraction.cv_skills)
