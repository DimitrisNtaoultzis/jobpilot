"""The analysis workflow as a LangGraph graph: state, nodes and edges."""

import asyncio
import logging
import operator
from dataclasses import dataclass
from typing import Annotated, Any, NotRequired, Protocol, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from .cover_letter import check_cover_letter, write_cover_letter
from .extraction import extract_skills
from .gap_analysis import GapReview, apply_gap_review, review_gaps
from .matching import compare_skills
from .resource_agent import (
    MAX_AGENT_TURNS,
    SEARCH_LIMIT,
    TOOLS,
    SearchLearningResources,
    SubmitLearningPlan,
    build_messages,
    format_results,
    parse_plan,
    select_resources,
)
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
    # compare node; gap_analysis and apply_resource_plan replace it with updated copies
    result: NotRequired[AnalyzeResponse]

    # Resource agent. These keys have a reducer (the 2nd argument of Annotated): what a
    # node returns for them is ADDED to the current value instead of replacing it.
    # The conversation with the LLM; add_messages appends (and updates by message id).
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]
    # Every search result of this run (prefetch + agent searches); operator.add concatenates
    # lists. Python only accepts resources that appear here (the LLM cannot invent URLs).
    found_resources: NotRequired[Annotated[list[SkillResources], operator.add]]
    # How many times the LLM has answered, to stop the loop (MAX_AGENT_TURNS)
    agent_turns: NotRequired[int]

    # write_letter node (joins the graph in step 2.7, after a human review); None when
    # the letter could not be written or failed its checks
    cover_letter: NotRequired[str | None]


class FindResources(Protocol):
    """Anything callable like ResourceFinder: the real one, or a fake in tests.

    A Protocol is structural typing ("duck typing" a type checker understands): no
    inheritance needed, matching the call signature is enough.
    """

    def __call__(self, skills: list[str], limit: int = 3) -> list[SkillResources]: ...


@dataclass(frozen=True)
class AnalysisContext:
    """Dependencies for one run, passed with graph.ainvoke(state, context=...).

    The graph is compiled once; every run gets its own context, so tests can pass fakes.
    """

    model: BaseChatModel
    find_resources: FindResources


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


async def prefetch_resources(
    state: AnalysisState, runtime: Runtime[AnalysisContext]
) -> dict[str, Any]:
    """Node: search resources for every gap and start the conversation with the agent."""
    gaps = state["result"].missing_skills
    try:
        # Blocking work (embedding, sync query) in a worker thread: the event loop
        # keeps serving other requests meanwhile.
        found = await asyncio.to_thread(runtime.context.find_resources, gaps, SEARCH_LIMIT)
    except Exception:
        # Graceful degradation: the skill analysis is useful on its own, so keep it
        # without resources (the agent could not search either without the database).
        logger.exception("Learning resource search failed")
        return {}  # no found_resources: route_after_prefetch goes to END
    return {"found_resources": found, "messages": build_messages(gaps, found)}


async def resource_agent(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> dict[str, Any]:
    """Node: one turn of the LLM, which answers with a tool call (search again or submit)."""
    turn = state.get("agent_turns", 0) + 1
    # "required": the model must call a tool (no free text). On the last turn it must
    # submit, so the turn limit never throws away the work done so far.
    tool_choice = SubmitLearningPlan.__name__ if turn >= MAX_AGENT_TURNS else "required"
    model = runtime.context.model.bind_tools(TOOLS, tool_choice=tool_choice)
    try:
        answer = await model.ainvoke(state["messages"])
    except Exception:
        # Graceful degradation: no new AI message, so apply_resource_plan finds no plan
        # and falls back to the top prefetch result for every gap.
        logger.exception("Resource agent failed")
        return {"agent_turns": turn}
    return {"messages": [answer], "agent_turns": turn}


async def run_tools(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> dict[str, Any]:
    """Node: run the searches the LLM asked for and answer each tool call.

    Every tool call must get a ToolMessage with its id, or the next LLM call is rejected,
    so even a failed or malformed call gets an answer (an error the LLM can read).
    """
    tool_messages = []
    found = []
    for tool_call in state["messages"][-1].tool_calls:
        query = tool_call["args"].get("query")
        if tool_call["name"] != SearchLearningResources.__name__ or not isinstance(query, str):
            content = "Error: only SearchLearningResources with a text query can run here."
        else:
            try:
                results = await asyncio.to_thread(
                    runtime.context.find_resources, [query], SEARCH_LIMIT
                )
            except Exception:
                logger.exception("Learning resource search failed")
                content = "Search failed. Choose from the results you already have."
            else:
                found.extend(results)
                content = format_results(results)
        tool_messages.append(ToolMessage(content, tool_call_id=tool_call["id"]))
    return {"messages": tool_messages, "found_resources": found}


def apply_resource_plan(state: AnalysisState) -> dict[str, Any]:
    """Node: check the agent's plan (Python guardrails) and put the resources in the result."""
    result = state["result"]
    last = state["messages"][-1]
    # No AI message last (the agent failed): no plan, select_resources falls back
    plan = parse_plan(last) if isinstance(last, AIMessage) else None
    resources = select_resources(result.missing_skills, state["found_resources"], plan)
    # Never mutate the state in place: return a new value and let LangGraph store it.
    return {"result": result.model_copy(update={"learning_resources": resources})}


async def write_letter(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> dict[str, Any]:
    """Node: the LLM writes a cover letter from the CV, the job posting and the analysis.

    Not wired into the graph yet: in step 2.7 it runs after the user approves the analysis.
    """
    result = state["result"]
    try:
        text = await write_cover_letter(
            runtime.context.model,
            state["cv_text"],
            state["job_title"],
            state["job_description"],
            result.matched_skills,
        )
        letter = check_cover_letter(text, result.missing_skills)
    except Exception:
        # The analysis is still useful without a letter. Never log the letter itself:
        # it is built from the CV (personal data).
        logger.exception("Cover letter failed")
        return {"cover_letter": None}
    return {"cover_letter": letter}


def route_after_compare(state: AnalysisState) -> str:
    """Conditional edge: review the gaps only when some skills are missing."""
    return "gap_analysis" if state["result"].missing_skills else END


def route_after_gap_analysis(state: AnalysisState) -> str:
    """Conditional edge: search for resources only when real gaps are left."""
    return "prefetch_resources" if state["result"].missing_skills else END


def route_after_prefetch(state: AnalysisState) -> str:
    """Conditional edge: start the agent only if the search worked."""
    return "resource_agent" if state.get("found_resources") else END


def route_after_agent(state: AnalysisState) -> str:
    """Conditional edge: run the requested searches, or finish.

    Finish when the LLM submitted, failed (no AI message last) or used all its turns.
    """
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or state["agent_turns"] >= MAX_AGENT_TURNS:
        return "apply_resource_plan"
    names = {tool_call["name"] for tool_call in last.tool_calls}
    if SubmitLearningPlan.__name__ in names or not names:
        return "apply_resource_plan"
    return "run_tools"


def build_graph() -> CompiledStateGraph:
    """Wire the nodes: a workflow (extract, compare, gap_analysis, prefetch_resources)
    that ends with an agent loop (resource_agent <-> run_tools, then apply_resource_plan).
    """
    builder = StateGraph(AnalysisState, context_schema=AnalysisContext)

    builder.add_node(extract)  # the node name is the function name
    builder.add_node(compare)
    builder.add_node(gap_analysis)
    builder.add_node(prefetch_resources)
    builder.add_node(resource_agent)
    builder.add_node(run_tools)
    builder.add_node(apply_resource_plan)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "compare")
    # path_map lists the possible targets, so LangGraph can validate and draw the graph
    builder.add_conditional_edges("compare", route_after_compare, ["gap_analysis", END])
    builder.add_conditional_edges(
        "gap_analysis", route_after_gap_analysis, ["prefetch_resources", END]
    )
    builder.add_conditional_edges(
        "prefetch_resources", route_after_prefetch, ["resource_agent", END]
    )
    # The agent loop: the LLM asks for searches, run_tools answers, the LLM decides again
    builder.add_conditional_edges(
        "resource_agent", route_after_agent, ["run_tools", "apply_resource_plan"]
    )
    builder.add_edge("run_tools", "resource_agent")
    builder.add_edge("apply_resource_plan", END)

    # compile() checks the structure (e.g. no node without a way in) and returns a runnable
    return builder.compile()


# Compiled once at import time; every run gets its own state and context.
analysis_graph = build_graph()
