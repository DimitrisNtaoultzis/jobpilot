import asyncio

from fakes import FakeModel, FakeResourceFinder
from langgraph.runtime import Runtime

from jobpilot.ai_service.extraction import SkillExtraction
from jobpilot.ai_service.graph import AnalysisContext, extract

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
