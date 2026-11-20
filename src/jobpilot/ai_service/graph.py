"""The analysis workflow as a LangGraph graph: state, nodes and edges."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from .extraction import extract_skills
from .gap_analysis import GapReview, apply_gap_review, review_gaps
from .matching import compare_skills
from .schemas import AnalyzeResponse, SkillResources

logger = logging.getLogger(__name__)


class AnalysisState(TypedDict):
    """The data that flows between the nodes of the analysis graph.

    Each node reads the state and returns only the keys it changes; LangGraph merges
    them in. Dependencies (the chat model, the resource finder) are not data, so they
    do not belong here: the checkpointer would try to serialize everything in the state.
    """

    # Inputs: set once when the graph starts (already validated by AnalyzeRequest)
    cv_text: str
    job_title: str
    job_description: str
    # Outputs: missing at the start, filled in by the nodes
    job_skills: NotRequired[list[str]]  # extract node
    cv_skills: NotRequired[list[str]]  # extract node
    # compare node; gap_analysis and find_resources replace it with updated copies
    result: NotRequired[AnalyzeResponse]


@dataclass(frozen=True)
class AnalysisContext:
    """Dependencies for one run, passed with graph.ainvoke(state, context=...).

    The graph is compiled once; every run gets its own context, so tests can pass fakes.
    """

    model: BaseChatModel
    # Any callable "skills -> resources": the real ResourceFinder, or a fake in tests
    find_resources: Callable[[list[str]], list[SkillResources]]


async def extract(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> dict[str, Any]:
    """Node: the LLM extracts the skills of the job posting and of the CV."""
    extraction = await extract_skills(
        runtime.context.model, state["cv_text"], state["job_title"], state["job_description"]
    )
    # Return only the keys this node changes; LangGraph merges them into the state.
    return {"job_skills": extraction.job_skills, "cv_skills": extraction.cv_skills}


def compare(state: AnalysisState) -> dict[str, Any]:
    """Node: plain code compares the two skill lists (no LLM, no dependencies)."""
    return {"result": compare_skills(state["job_skills"], state["cv_skills"])}


async def gap_analysis(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> dict[str, Any]:
    """Node: the LLM reviews the missing skills (covered by the CV? how important?)."""
    result = state["result"]
    try:
        review = await review_gaps(
            runtime.context.model,
            result.missing_skills,
            state["cv_skills"],
            state["job_title"],
            state["job_description"],
        )
    except Exception:
        # Graceful degradation: with an empty review the guardrails keep every missing
        # skill as a must_have gap, so skill_gaps stays complete and consistent.
        logger.exception("Gap review failed")
        review = GapReview(covered=[], gaps=[])
    return {"result": apply_gap_review(result, state["cv_skills"], review)}


async def find_resources(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> dict[str, Any]:
    """Node: a semantic search suggests learning resources for the missing skills."""
    result = state["result"]
    try:
        # Blocking work (embedding, sync query) in a worker thread: the event loop
        # keeps serving other requests meanwhile.
        resources = await asyncio.to_thread(runtime.context.find_resources, result.missing_skills)
    except Exception:
        # Graceful degradation: the skill analysis is useful on its own, so keep it
        # without resources instead of failing the whole run.
        logger.exception("Learning resource search failed")
        return {}  # nothing changes
    # Never mutate the state in place: return a new value and let LangGraph store it.
    return {"result": result.model_copy(update={"learning_resources": resources})}


def route_after_compare(state: AnalysisState) -> str:
    """Conditional edge: review the gaps only when some skills are missing."""
    return "gap_analysis" if state["result"].missing_skills else END


def route_after_gap_analysis(state: AnalysisState) -> str:
    """Conditional edge: search for resources only when real gaps are left."""
    return "find_resources" if state["result"].missing_skills else END


def build_graph() -> CompiledStateGraph:
    """Wire the nodes: extract -> compare -> (gap_analysis) -> (find_resources) -> end."""
    builder = StateGraph(AnalysisState, context_schema=AnalysisContext)

    builder.add_node(extract)  # the node name is the function name
    builder.add_node(compare)
    builder.add_node(gap_analysis)
    builder.add_node(find_resources)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "compare")
    # path_map lists the possible targets, so LangGraph can validate and draw the graph
    builder.add_conditional_edges("compare", route_after_compare, ["gap_analysis", END])
    builder.add_conditional_edges("gap_analysis", route_after_gap_analysis, ["find_resources", END])
    builder.add_edge("find_resources", END)

    # compile() checks the structure (e.g. no node without a way in) and returns a runnable
    return builder.compile()


# Compiled once at import time; every run gets its own state and context.
analysis_graph = build_graph()
