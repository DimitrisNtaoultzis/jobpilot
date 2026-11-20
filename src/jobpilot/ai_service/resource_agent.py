"""The resource agent: an LLM that refines the learning resource search with tool calls.

Flow: the graph prefetches resources for every gap, then the LLM may search again with
better queries and finally submits its choice. Python checks that choice before use.
"""

from pydantic import BaseModel, Field

# The LLM gets at most this many turns; each turn resends the whole history (tokens).
MAX_AGENT_TURNS = 3
# Results per search: more than we show the user, so the LLM has something to choose from.
SEARCH_LIMIT = 5
# Resources per skill in the final answer
MAX_RESOURCES_PER_SKILL = 3


# The tools the LLM can call. We run them ourselves (see the graph's run_tools node);
# the LLM only sees their JSON Schema. As with structured output, the docstrings and the
# field descriptions are part of the prompt, so they are written for the model.
class SearchLearningResources(BaseModel):
    """Search the learning resource catalog with a new query.

    Use it only when the results you already have for a skill are poor or off-topic.
    """

    query: str = Field(
        description="What to search for, in plain English, e.g. 'GitHub Actions CI workflows'."
    )


class SkillSelection(BaseModel):
    """The chosen learning resources for one skill gap."""

    skill: str = Field(description="The skill gap, exactly as given.")
    urls: list[str] = Field(
        description=(
            f"URLs of the most useful resources for this skill, best first, at most "
            f"{MAX_RESOURCES_PER_SKILL}. Only URLs from the search results. Empty if none "
            "of the results is really about this skill."
        )
    )


class SubmitLearningPlan(BaseModel):
    """Submit your final choice of learning resources. Call it once, when you are done."""

    selections: list[SkillSelection]


TOOLS = [SearchLearningResources, SubmitLearningPlan]
