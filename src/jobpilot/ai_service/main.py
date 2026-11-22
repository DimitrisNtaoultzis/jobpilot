"""JobPilot AI service: the FastAPI app that runs the analysis (LLM, agent).

Run locally with:
    uv run uvicorn jobpilot.ai_service.main:app --reload --port 8001
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.language_models.chat_models import BaseChatModel
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import get_settings
from .embeddings import get_embedding_model
from .graph import AnalysisContext, analysis_graph
from .llm import get_chat_model
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
    and a semantic search suggests learning resources for the missing ones.

    The workflow itself lives in the analysis graph (graph.py); this endpoint only maps
    HTTP to a graph run and back.
    """
    context = AnalysisContext(model=model, find_resources=finder)
    # Every analysis is its own thread: the checkpointer saves its paused state under
    # this id, and resuming it later (step 2.7γ) needs the same id.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    try:
        paused_state = await analysis_graph.ainvoke(payload.model_dump(), config, context=context)
    except Exception as exc:
        # Service boundary: log the full traceback, return a safe message to the client.
        # Log the error, never the texts: CVs contain personal data.
        # (A failed resource search never gets here: the graph degrades gracefully.)
        logger.exception("Analysis graph failed")
        raise HTTPException(
            status_code=502, detail="The AI model could not analyze this request."
        ) from exc

    # The run stopped at human_review: the analysis is ready for the user to review
    return paused_state["result"]
