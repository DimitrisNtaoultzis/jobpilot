"""The analysis workflow as a LangGraph graph: state, nodes and edges."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.runtime import Runtime

from .extraction import extract_skills
from .schemas import AnalyzeResponse, SkillResources


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
    result: NotRequired[AnalyzeResponse]  # compare node, resources added later


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
