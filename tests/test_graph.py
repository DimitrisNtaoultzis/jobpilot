import asyncio

import pytest
from fakes import FakeModel, FakeResourceFinder
from langgraph.runtime import Runtime

from jobpilot.ai_service.extraction import SkillExtraction
from jobpilot.ai_service.gap_analysis import CoveredSkill, GapReview, ReviewedGap
from jobpilot.ai_service.graph import (
    AnalysisContext,
    analysis_graph,
    compare,
    extract,
    find_resources,
    gap_analysis,
)
from jobpilot.ai_service.schemas import AnalyzeResponse, ResourceMatch, SkillGap, SkillResources

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
    model = FakeModel(SkillExtraction(job_skills=["Python", "Docker"], cv_skills=["Python"]))

    # Nodes are plain async functions: call one directly, no graph needed.
    update = asyncio.run(extract(STATE, make_runtime(model=model)))

    assert update == {"job_skills": ["Python", "Docker"], "cv_skills": ["Python"]}


def test_extract_sends_the_state_texts_to_the_model():
    model = FakeModel(SkillExtraction(job_skills=[], cv_skills=[]))

    asyncio.run(extract(STATE, make_runtime(model=model)))

    user_message = model.call_for(SkillExtraction).messages[-1].content
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


DOCKER_IS_A_GAP = GapReview(covered=[], gaps=[ReviewedGap(skill="Docker", priority="must_have")])


def gap_state():
    """The state after compare: the job asks for SQL and Docker, the CV has PostgreSQL."""
    result = AnalyzeResponse(
        required_skills=["Python", "SQL", "Docker"],
        matched_skills=["Python"],
        missing_skills=["SQL", "Docker"],
        match_score=0.33,
    )
    return {**STATE, "cv_skills": ["Python", "PostgreSQL"], "result": result}


def test_gap_analysis_applies_the_review():
    model = FakeModel(
        GapReview(
            covered=[CoveredSkill(skill="SQL", cv_skill="PostgreSQL")],
            gaps=[ReviewedGap(skill="Docker", priority="nice_to_have")],
        )
    )

    update = asyncio.run(gap_analysis(gap_state(), make_runtime(model=model)))

    result = update["result"]
    assert result.matched_skills == ["Python", "SQL"]
    assert result.skill_gaps == [SkillGap(skill="Docker", priority="nice_to_have")]


def test_gap_analysis_keeps_every_gap_as_must_have_when_the_llm_fails(caplog):
    model = FakeModel(errors={GapReview: RuntimeError("Groq is down")})

    update = asyncio.run(gap_analysis(gap_state(), make_runtime(model=model)))

    result = update["result"]
    assert result.missing_skills == ["SQL", "Docker"]  # plain matching is kept
    assert result.skill_gaps == [
        SkillGap(skill="SQL", priority="must_have"),
        SkillGap(skill="Docker", priority="must_have"),
    ]
    assert "Gap review failed" in caplog.text


def run_graph(model, finder=None):
    """Run the whole compiled graph, the way /analyze will."""
    context = AnalysisContext(model=model, find_resources=finder or FakeResourceFinder())
    return asyncio.run(analysis_graph.ainvoke(STATE, context=context))


def test_graph_runs_all_nodes_when_skills_are_missing():
    model = FakeModel(
        SkillExtraction(job_skills=["Python", "Docker"], cv_skills=["python"]), DOCKER_IS_A_GAP
    )
    finder = FakeResourceFinder(results=[DOCKER_RESOURCES])

    final_state = run_graph(model, finder)

    result = final_state["result"]
    assert result.missing_skills == ["Docker"]
    assert result.learning_resources == [DOCKER_RESOURCES]
    assert finder.skills == ["Docker"]


def test_graph_skips_the_search_when_nothing_is_missing():
    model = FakeModel(SkillExtraction(job_skills=["Python"], cv_skills=["Python"]))
    finder = FakeResourceFinder()

    final_state = run_graph(model, finder)

    assert final_state["result"].match_score == 1.0
    assert finder.skills is None  # the conditional edge went straight to END


def test_graph_keeps_the_result_when_the_search_fails():
    model = FakeModel(
        SkillExtraction(job_skills=["Python", "Docker"], cv_skills=["python"]), DOCKER_IS_A_GAP
    )
    finder = FakeResourceFinder(error=RuntimeError("database is down"))

    final_state = run_graph(model, finder)

    assert final_state["result"].missing_skills == ["Docker"]
    assert final_state["result"].learning_resources == []


def test_graph_lets_extraction_errors_propagate():
    # The graph does not hide LLM failures: /analyze turns them into a 502
    model = FakeModel(errors={SkillExtraction: RuntimeError("Groq is down")})

    with pytest.raises(RuntimeError, match="Groq is down"):
        run_graph(model)


def test_graph_searches_only_the_gaps_left_after_the_review():
    model = FakeModel(
        SkillExtraction(job_skills=["SQL", "Docker"], cv_skills=["PostgreSQL"]),
        GapReview(
            covered=[CoveredSkill(skill="SQL", cv_skill="PostgreSQL")],
            gaps=[ReviewedGap(skill="Docker", priority="must_have")],
        ),
    )
    finder = FakeResourceFinder()

    final_state = run_graph(model, finder)

    assert final_state["result"].matched_skills == ["SQL"]
    assert finder.skills == ["Docker"]  # no search for the covered SQL


def test_graph_skips_the_search_when_the_review_covers_every_gap():
    model = FakeModel(
        SkillExtraction(job_skills=["SQL"], cv_skills=["PostgreSQL"]),
        GapReview(covered=[CoveredSkill(skill="SQL", cv_skill="PostgreSQL")], gaps=[]),
    )
    finder = FakeResourceFinder()

    final_state = run_graph(model, finder)

    assert final_state["result"].match_score == 1.0
    assert finder.skills is None  # the second conditional edge went straight to END
