import asyncio

from fakes import FakeModel, FakeResourceFinder
from langgraph.runtime import Runtime

from jobpilot.ai_service.extraction import SkillExtraction
from jobpilot.ai_service.graph import AnalysisContext, compare, extract, find_resources
from jobpilot.ai_service.schemas import AnalyzeResponse, ResourceMatch, SkillResources

STATE = {
    "cv_text": "Python developer",
    "job_title": "Backend Engineer",
    "job_description": "We need Python and Docker",
}


def make_runtime(model=None, finder=None):
    """A Runtime like the one LangGraph passes to the nodes, built by hand."""
    context = AnalysisContext(
        model=model or FakeModel(),
        find_resources=finder or FakeResourceFinder(),
    )
    return Runtime(context=context)


def test_extract_returns_only_the_skill_keys():
    model = FakeModel(
        extraction=SkillExtraction(job_skills=["Python", "Docker"], cv_skills=["Python"])
    )

    # Nodes are plain async functions: call one directly, no graph needed.
    update = asyncio.run(extract(STATE, make_runtime(model=model)))

    assert update == {"job_skills": ["Python", "Docker"], "cv_skills": ["Python"]}


def test_extract_sends_the_state_texts_to_the_model():
    model = FakeModel(extraction=SkillExtraction(job_skills=[], cv_skills=[]))

    asyncio.run(extract(STATE, make_runtime(model=model)))

    user_message = model.messages[-1].content
    assert "Python developer" in user_message
    assert "Backend Engineer" in user_message
    assert "We need Python and Docker" in user_message


def test_compare_builds_the_result_from_the_skill_lists():
    state = {**STATE, "job_skills": ["Python", "Docker"], "cv_skills": ["python"]}

    update = compare(state)

    result = update["result"]
    assert result.matched_skills == ["Python"]
    assert result.missing_skills == ["Docker"]
    assert result.match_score == 0.5


DOCKER_RESOURCES = SkillResources(
    skill="Docker",
    resources=[
        ResourceMatch(
            title="Docker Get Started",
            url="https://docs.docker.com/get-started/",
            description="Official Docker tutorial.",
            similarity=0.8,
        )
    ],
)


def make_result():
    return AnalyzeResponse(
        required_skills=["Python", "Docker"],
        matched_skills=["Python"],
        missing_skills=["Docker"],
        match_score=0.5,
    )


def test_find_resources_searches_the_missing_skills():
    finder = FakeResourceFinder(results=[DOCKER_RESOURCES])
    result = make_result()

    update = asyncio.run(find_resources({**STATE, "result": result}, make_runtime(finder=finder)))

    assert finder.skills == ["Docker"]
    assert update["result"].learning_resources == [DOCKER_RESOURCES]
    # The node returns a new result; the one in the state stays untouched
    assert result.learning_resources == []


def test_find_resources_returns_no_update_when_the_search_fails(caplog):
    finder = FakeResourceFinder(error=RuntimeError("database is down"))

    update = asyncio.run(
        find_resources({**STATE, "result": make_result()}, make_runtime(finder=finder))
    )

    assert update == {}  # graceful degradation: the state keeps the result as it was
    assert "Learning resource search failed" in caplog.text
