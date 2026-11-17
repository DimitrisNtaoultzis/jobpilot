"""JobPilot AI service: the FastAPI app that runs the analysis (LLM, agent).

Run locally with:
    uv run uvicorn jobpilot.ai_service.main:app --reload --port 8001
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.language_models.chat_models import BaseChatModel
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import get_settings
from .embeddings import get_embedding_model
from .extraction import extract_skills
from .llm import get_chat_model
from .matching import compare_skills
from .retrieval import ResourceFinder
from .schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup (before yield) and shutdown (after yield): resources shared by all requests."""
    pool = ConnectionPool(
        get_settings().database_url.get_secret_value(),
        min_size=1,
        max_size=4,
        # The pool discards a connection that configure leaves inside a transaction, and
        # register_vector runs a query. With autocommit no transaction stays open
        # (searches only read, so they need none).
        kwargs={"autocommit": True},
        configure=register_vector,  # every new connection learns the vector type
        open=False,
    )
    # wait=False: start even if the database is down; searches then fail gracefully
    pool.open(wait=False)
    # Warm-up: load the embedding model now (in a thread), not during the first request
    model = await asyncio.to_thread(get_embedding_model)
    app.state.resource_finder = ResourceFinder(pool, model)
    try:
        yield
    finally:
        pool.close()


app = FastAPI(title="JobPilot AI service", version="0.1.0", lifespan=lifespan)


def get_resource_finder(request: Request) -> ResourceFinder:
    """The finder created at startup (tests replace it via app.dependency_overrides)."""
    return request.app.state.resource_finder


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check: the service is up and answering requests."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    payload: AnalyzeRequest,
    # Dependency injection: tests replace get_chat_model via app.dependency_overrides.
    model: Annotated[BaseChatModel, Depends(get_chat_model)],
    finder: Annotated[ResourceFinder, Depends(get_resource_finder)],
) -> AnalyzeResponse:
    """Compare a CV with a job posting: the LLM extracts the skills, plain code compares them,
    and a semantic search suggests learning resources for the missing ones."""
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

    result = compare_skills(extraction.job_skills, extraction.cv_skills)

    if result.missing_skills:
        try:
            # Blocking work (embedding, sync query) in a worker thread: the event loop
            # keeps serving other requests meanwhile.
            result.learning_resources = await asyncio.to_thread(finder, result.missing_skills)
        except Exception:
            # Graceful degradation: the skill analysis is useful on its own, so return it
            # without resources (learning_resources stays []) instead of failing.
            logger.exception("Learning resource search failed")

    return result
