"""JobPilot AI service: the FastAPI app that runs the analysis (LLM, agent).

Run locally with:
    uv run uvicorn jobpilot.ai_service.main:app --port 8001 --loop asyncio:SelectorEventLoop

Add --reload to restart on code changes. The selector loop is for Windows: async psycopg
(used by the checkpointer) rejects the default ProactorEventLoop; see the README.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .checkpoints import open_checkpointer
from .config import get_settings
from .embeddings import get_embedding_model
from .graph import AnalysisContext, build_graph
from .llm import get_chat_model
from .retrieval import ResourceFinder
from .schemas import AnalysisRun, AnalyzeRequest, ResumeResponse, ReviewDecision

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup (before yield) and shutdown (after yield): resources shared by all requests."""
    database_url = get_settings().database_url.get_secret_value()
    pool = ConnectionPool(
        database_url,
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
        # Unlike the search, the checkpointer is required: if the database is down,
        # startup stops here (fail fast)
        async with open_checkpointer(database_url) as checkpointer:
            # Compiled here, not at import time: the checkpointer needs an open pool
            app.state.analysis_graph = build_graph(checkpointer)
            yield
    finally:
        pool.close()


app = FastAPI(title="JobPilot AI service", version="0.1.0", lifespan=lifespan)


def get_resource_finder(request: Request) -> ResourceFinder:
    """The finder created at startup (tests replace it via app.dependency_overrides)."""
    return request.app.state.resource_finder


def get_analysis_graph(request: Request) -> CompiledStateGraph:
    """The graph compiled at startup (tests replace it via app.dependency_overrides)."""
    return request.app.state.analysis_graph


def thread_config(thread_id: uuid.UUID) -> RunnableConfig:
    """The LangGraph config of one analysis: its checkpoints are saved under this id."""
    return {"configurable": {"thread_id": str(thread_id)}}


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
    graph: Annotated[CompiledStateGraph, Depends(get_analysis_graph)],
) -> AnalysisRun:
    """Compare a CV with a job posting: the LLM extracts the skills, plain code compares them,
    and a semantic search suggests learning resources for the missing ones.

    The run pauses for the user's review: send the returned thread_id to
    /analyze/{thread_id}/resume to continue it. The workflow itself lives in the analysis
    graph (graph.py); this endpoint only maps HTTP to a graph run and back.
    """
    context = AnalysisContext(model=model, find_resources=finder)
    # Every analysis is its own thread: the checkpointer saves its paused state under
    # this id, and resuming it later needs the same id.
    thread_id = uuid.uuid4()
    try:
        paused_state = await graph.ainvoke(
            payload.model_dump(), thread_config(thread_id), context=context
        )
    except Exception as exc:
        # Service boundary: log the full traceback, return a safe message to the client.
        # Log the error, never the texts: CVs contain personal data.
        # (A failed resource search never gets here: the graph degrades gracefully.)
        logger.exception("Analysis graph failed")
        raise HTTPException(
            status_code=502, detail="The AI model could not analyze this request."
        ) from exc

    # The run stopped at human_review: the analysis is ready for the user to review
    return AnalysisRun(thread_id=thread_id, analysis=paused_state["result"])


@app.post("/analyze/{thread_id}/resume")
async def resume_analysis(
    thread_id: uuid.UUID,  # FastAPI validates it: not a UUID -> 422
    decision: ReviewDecision,
    model: Annotated[BaseChatModel, Depends(get_chat_model)],
    # Not used by the resumed nodes, but part of every run's context
    finder: Annotated[ResourceFinder, Depends(get_resource_finder)],
    graph: Annotated[CompiledStateGraph, Depends(get_analysis_graph)],
) -> ResumeResponse:
    """Continue a paused analysis with the user's decision; write the cover letter if asked.

    If the letter fails, the run pauses again (502): the same request can be sent later.
    """
    config = thread_config(thread_id)
    # Outside the try below: these HTTP errors must not be turned into a 502
    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Analysis not found or expired.")
    if "human_review" not in snapshot.next:
        raise HTTPException(status_code=409, detail="This analysis is not waiting for a review.")

    context = AnalysisContext(model=model, find_resources=finder)
    try:
        state = await graph.ainvoke(Command(resume=decision.model_dump()), config, context=context)
    except Exception as exc:
        logger.exception("Resumed analysis graph failed")
        raise HTTPException(
            status_code=502, detail="The AI model could not finish this request."
        ) from exc

    if "__interrupt__" in state:
        # write_letter failed and the graph paused at human_review again: retry later
        raise HTTPException(
            status_code=502, detail="The cover letter could not be written. Try again later."
        )
    # No cover_letter key when the user declined
    return ResumeResponse(cover_letter=state.get("cover_letter"))
