from fakes import search_call, submit_call, tool_call_message
from langchain_core.utils.function_calling import convert_to_openai_tool

from jobpilot.ai_service.resource_agent import (
    SYSTEM_PROMPT,
    TOOLS,
    SkillSelection,
    SubmitLearningPlan,
    build_messages,
    format_results,
    parse_plan,
    select_resources,
)
from jobpilot.ai_service.schemas import ResourceMatch, SkillResources


def test_the_tools_are_described_to_the_model():
    # What bind_tools() sends to the model: a name, a description and a JSON Schema
    search, submit = (convert_to_openai_tool(tool)["function"] for tool in TOOLS)

    assert search["name"] == "SearchLearningResources"
    assert search["parameters"]["required"] == ["query"]
    assert "poor or off-topic" in search["description"]

    assert submit["name"] == "SubmitLearningPlan"
    assert submit["parameters"]["required"] == ["selections"]


# --- select_resources: Python enforces what the tool descriptions only ask for ---


def res(slug):
    """A resource whose url, title and description all come from a short slug."""
    return ResourceMatch(
        title=f"Title {slug}",
        url=f"https://example.com/{slug}",
        description=f"About {slug}.",
        similarity=0.7,
    )


def url(slug):
    return f"https://example.com/{slug}"


GAPS = ["Docker", "GitHub Actions"]

# Two prefetch searches (named after the gaps) and one agent search (named after its query)
FOUND = [
    SkillResources(skill="Docker", resources=[res("docker"), res("k8s"), res("linux")]),
    SkillResources(skill="GitHub Actions", resources=[res("gha"), res("git"), res("prompts")]),
    SkillResources(skill="CI workflows with GitHub", resources=[res("gha"), res("ci-book")]),
]


def plan(**urls_per_skill):
    """SubmitLearningPlan from keyword arguments; underscores in names become spaces."""
    return SubmitLearningPlan(
        selections=[
            SkillSelection(skill=skill.replace("_", " "), urls=[url(name) for name in names])
            for skill, names in urls_per_skill.items()
        ]
    )


def slugs(selected):
    """{skill: [slug, ...]} from select_resources' output, for short assertions."""
    return {
        entry.skill: [r.url.removeprefix("https://example.com/") for r in entry.resources]
        for entry in selected
    }


def test_the_chosen_urls_become_the_resources_in_the_llm_order():
    llm_plan = plan(Docker=["linux", "docker"], GitHub_Actions=["gha"])

    selected = select_resources(GAPS, FOUND, llm_plan)

    assert slugs(selected) == {"Docker": ["linux", "docker"], "GitHub Actions": ["gha"]}
    # The details come from the search results, not from the LLM
    assert selected[0].resources[1] == res("docker")


def test_a_url_found_by_another_search_is_accepted():
    selected = select_resources(GAPS, FOUND, plan(Docker=["docker"], GitHub_Actions=["ci-book"]))

    assert slugs(selected)["GitHub Actions"] == ["ci-book"]


def test_an_invented_url_is_rejected():
    selected = select_resources(
        GAPS, FOUND, plan(Docker=["made-up", "docker"], GitHub_Actions=["gha"])
    )

    assert slugs(selected)["Docker"] == ["docker"]


def test_at_most_three_resources_per_skill_without_duplicates():
    selected = select_resources(
        GAPS, FOUND, plan(Docker=["docker", "docker", "k8s", "linux", "gha"], GitHub_Actions=[])
    )

    assert slugs(selected)["Docker"] == ["docker", "k8s", "linux"]


def test_an_explicit_empty_choice_is_respected():
    # The LLM judged that no result is really about the skill: show nothing, not noise
    selected = select_resources(GAPS, FOUND, plan(Docker=["docker"], GitHub_Actions=[]))

    assert slugs(selected)["GitHub Actions"] == []


def test_when_every_chosen_url_is_rejected_the_top_prefetch_result_is_used():
    selected = select_resources(GAPS, FOUND, plan(Docker=["made-up"], GitHub_Actions=["gha"]))

    assert slugs(selected)["Docker"] == ["docker"]


def test_a_gap_the_llm_forgot_gets_the_top_prefetch_result():
    selected = select_resources(GAPS, FOUND, plan(Docker=["linux"]))

    assert slugs(selected) == {"Docker": ["linux"], "GitHub Actions": ["gha"]}


def test_without_a_plan_every_gap_gets_the_top_prefetch_result():
    selected = select_resources(GAPS, FOUND, None)

    assert slugs(selected) == {"Docker": ["docker"], "GitHub Actions": ["gha"]}


def test_a_gap_without_prefetch_results_falls_back_to_nothing():
    selected = select_resources(["Rust"], FOUND, None)

    assert slugs(selected) == {"Rust": []}


def test_choices_for_skills_that_are_not_gaps_are_ignored_and_the_gap_order_is_kept():
    selected = select_resources(
        GAPS, FOUND, plan(Python=["docker"], GitHub_Actions=["gha"], Docker=["docker"])
    )

    assert list(slugs(selected)) == ["Docker", "GitHub Actions"]


def test_skills_are_compared_by_normalized_name_and_the_first_choice_wins():
    llm_plan = SubmitLearningPlan(
        selections=[
            SkillSelection(skill="docker", urls=[url("k8s")]),
            SkillSelection(skill="Docker", urls=[url("linux")]),
        ]
    )

    selected = select_resources(["Docker"], FOUND, llm_plan)

    assert slugs(selected) == {"Docker": ["k8s"]}


# --- What the LLM sees, and reading its answer ---


def test_format_results_shows_each_search_as_a_tagged_numbered_block():
    text = format_results([FOUND[0]])

    assert text.startswith("<search_results>\nQuery: Docker\n")
    assert "1. Title docker\n   URL: https://example.com/docker\n   About docker." in text
    assert "3. Title linux" in text
    assert text.endswith("</search_results>")
    assert "0.7" not in text  # no similarity score: the LLM judges by content


def test_format_results_says_so_when_a_search_found_nothing():
    text = format_results([SkillResources(skill="Rust", resources=[])])

    assert "No results." in text


def test_build_messages_lists_the_gaps_and_their_results():
    system, user = build_messages(GAPS, FOUND[:2])

    assert system.content == SYSTEM_PROMPT
    assert "<skill_gaps>\nDocker\nGitHub Actions\n</skill_gaps>" in user.content
    assert "Query: Docker" in user.content
    assert "Query: GitHub Actions" in user.content


def test_parse_plan_reads_the_submitted_plan():
    message = submit_call(Docker=[url("docker")], GitHub_Actions=[])

    assert parse_plan(message) == plan(Docker=["docker"], GitHub_Actions=[])


def test_parse_plan_gives_none_without_a_submit_call():
    assert parse_plan(search_call("Docker tutorial")) is None


def test_parse_plan_gives_none_when_the_arguments_do_not_fit_the_schema():
    message = tool_call_message("SubmitLearningPlan", {"selections": [{"skill": "Docker"}]})

    assert parse_plan(message) is None  # "urls" is missing
