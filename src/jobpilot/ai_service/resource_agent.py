"""The resource agent: an LLM that refines the learning resource search with tool calls.

Flow: the graph prefetches resources for every gap, then the LLM may search again with
better queries and finally submits its choice. Python checks that choice before use.
"""

from pydantic import BaseModel, Field

from jobpilot.skills import normalize_skill

from .schemas import ResourceMatch, SkillResources

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


def select_resources(
    gaps: list[str], found: list[SkillResources], plan: SubmitLearningPlan | None
) -> list[SkillResources]:
    """Turn the agent's choice into the final resources, trusting only what checks out.

    `found` holds every search result of this run: the prefetch (one entry per gap, named
    after it) and the agent's own searches (named after their query). The rules:
    - one entry per gap, in the gaps' order; choices for other skills are ignored,
    - a URL is accepted only if some search in `found` returned it, and its details come
      from that search (never from the LLM); at most MAX_RESOURCES_PER_SKILL, no duplicates,
    - an explicit empty choice is respected (nothing relevant is better than noise),
    - fallback to the top prefetch result when there is no plan, the gap was forgotten,
      or every chosen URL was rejected (invented URLs mean something went wrong).
    """
    # Every resource seen in this run, by URL; the first occurrence wins
    seen: dict[str, ResourceMatch] = {}
    for entry in found:
        for resource in entry.resources:
            seen.setdefault(resource.url, resource)

    # The LLM's choice per gap; for a duplicate skill, the first choice wins
    chosen: dict[str, list[str]] = {}
    for selection in plan.selections if plan else []:
        chosen.setdefault(normalize_skill(selection.skill), selection.urls)

    selected = []
    for gap in gaps:
        urls = chosen.get(normalize_skill(gap))
        if urls is None:  # no plan, or the gap was forgotten
            resources = _top_prefetch_result(gap, found)
        else:
            # dict.fromkeys drops duplicates and keeps the order (dicts are ordered)
            valid = [seen[u] for u in dict.fromkeys(urls) if u in seen]
            if urls and not valid:  # URLs were given, but none was ever found
                resources = _top_prefetch_result(gap, found)
            else:
                resources = valid[:MAX_RESOURCES_PER_SKILL]
        selected.append(SkillResources(skill=gap, resources=resources))
    return selected


def _top_prefetch_result(gap: str, found: list[SkillResources]) -> list[ResourceMatch]:
    """The best result of the prefetch search for this gap (the first one is usually right)."""
    for entry in found:
        if normalize_skill(entry.skill) == normalize_skill(gap):
            return entry.resources[:1]
    return []
