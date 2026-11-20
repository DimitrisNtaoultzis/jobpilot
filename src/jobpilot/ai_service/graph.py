"""The analysis workflow as a LangGraph graph: state, nodes and edges."""

from typing import NotRequired, TypedDict

from .schemas import AnalyzeResponse


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
    job_skills: NotRequired[list[str]]  # extract_skills node
    cv_skills: NotRequired[list[str]]  # extract_skills node
    result: NotRequired[AnalyzeResponse]  # compare node, resources added later
