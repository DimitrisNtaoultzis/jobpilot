import uuid

import pytest
from fakes import FakeModel, FakeResourceFinder, submit_call
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from jobpilot.ai_service.extraction import SkillExtraction
from jobpilot.ai_service.gap_analysis import GapReview, ReviewedGap
from jobpilot.ai_service.graph import build_graph
from jobpilot.ai_service.llm import get_chat_model
from jobpilot.ai_service.main import app, get_analysis_graph, get_resource_finder
from jobpilot.ai_service.schemas import (
    MAX_CV_LENGTH,
    MAX_JOB_DESCRIPTION_LENGTH,
    AnalyzeResponse,
    ResourceMatch,
    ReviewDecision,
    SkillResources,
)

VALID_PAYLOAD = {
    "cv_text": "Python developer",
    "job_title": "Backend Engineer",
    "job_description": "We need Python",
}

LETTER = "Dear hiring team,\n\nI build Python services."


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


@pytest.fixture
def api_client():
    # Not named "client": that name belongs to pytest-django's Django test client.
    # TestClient without "with" does not run the lifespan: no pool, no model loading.
    return TestClient(app)


@pytest.fixture
def fake_model():
    model = FakeModel(
        SkillExtraction(job_skills=["Python", "Docker"], cv_skills=["python"]),
        GapReview(covered=[], gaps=[ReviewedGap(skill="Docker", priority="nice_to_have")]),
        # The resource agent submits at once (it runs only when the finder finds something)
        turns=[submit_call(Docker=["https://docs.docker.com/get-started/"])],
        text=LETTER,  # the cover letter, for the tests that resume the run
    )
    # Dependency injection pays off: /analyze gets the fake, its code stays unchanged.
    app.dependency_overrides[get_chat_model] = lambda: model
    yield model
    app.dependency_overrides.pop(get_chat_model)


@pytest.fixture(autouse=True)
def fake_finder():
    """Every test gets a finder that returns no resources (the lifespan never runs here)."""
    finder = FakeResourceFinder()
    app.dependency_overrides[get_resource_finder] = lambda: finder
    yield finder
    app.dependency_overrides.pop(get_resource_finder)


@pytest.fixture(autouse=True)
def in_memory_graph():
    """Every test gets the real graph with an in-memory checkpointer (no database)."""
    graph = build_graph(InMemorySaver())
    app.dependency_overrides[get_analysis_graph] = lambda: graph
    yield graph
    app.dependency_overrides.pop(get_analysis_graph)


def test_health(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_the_response_shape(api_client, fake_model):
    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"thread_id", "analysis"}
    uuid.UUID(body["thread_id"])  # raises if it is not a valid UUID
    assert set(body["analysis"]) == {
        "required_skills",
        "matched_skills",
        "missing_skills",
        "match_score",
        "skill_gaps",
        "learning_resources",
    }


def test_analyze_compares_the_extracted_skills(api_client, fake_model):
    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    assert response.json()["analysis"] == {
        "required_skills": ["Python", "Docker"],
        "matched_skills": ["Python"],
        "missing_skills": ["Docker"],
        "match_score": 0.5,
        "skill_gaps": [{"skill": "Docker", "priority": "nice_to_have"}],  # from the review
        "learning_resources": [],  # the default fake finder finds nothing
    }


def test_analyze_searches_resources_for_the_missing_skills(api_client, fake_model, fake_finder):
    fake_finder.results = [DOCKER_RESOURCES]

    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    assert fake_finder.skills == ["Docker"]  # only the missing skills, not "Python"
    assert response.json()["analysis"]["learning_resources"] == [DOCKER_RESOURCES.model_dump()]


def test_analyze_skips_the_search_when_nothing_is_missing(api_client, fake_model, fake_finder):
    fake_model.responses[SkillExtraction] = SkillExtraction(
        job_skills=["Python"], cv_skills=["Python"]
    )

    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    assert fake_finder.skills is None  # never called: no database round trip
    assert response.json()["analysis"]["learning_resources"] == []


def test_analyze_still_succeeds_when_the_search_fails(api_client, fake_model, fake_finder):
    fake_finder.error = RuntimeError("database is down")

    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    # Graceful degradation: the skill analysis is returned without resources
    assert response.status_code == 200
    assert response.json()["analysis"]["missing_skills"] == ["Docker"]
    assert response.json()["analysis"]["learning_resources"] == []
    assert "database is down" not in response.text


def test_analyze_uses_strict_structured_output_and_sends_the_texts(api_client, fake_model):
    api_client.post("/analyze", json=VALID_PAYLOAD)

    call = fake_model.call_for(SkillExtraction)
    assert call.kwargs == {"method": "json_schema", "strict": True}
    user_message = call.messages[-1].content
    assert "Python developer" in user_message  # the CV text
    assert "We need Python" in user_message  # the job description


def test_analyze_returns_502_when_the_model_fails(api_client, fake_model):
    fake_model.errors[SkillExtraction] = RuntimeError("Groq is down")

    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    assert response.status_code == 502
    assert "Groq is down" not in response.text  # internal details stay in the logs


def test_analyze_still_succeeds_when_the_gap_review_fails(api_client, fake_model):
    fake_model.errors[GapReview] = RuntimeError("Groq is down")

    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    # Graceful degradation: plain matching is kept, every gap is a cautious must_have
    assert response.status_code == 200
    assert response.json()["analysis"]["skill_gaps"] == [
        {"skill": "Docker", "priority": "must_have"}
    ]


def start_analysis(api_client):
    """Run /analyze and return the thread id of the paused run."""
    response = api_client.post("/analyze", json=VALID_PAYLOAD)
    assert response.status_code == 200
    return response.json()["thread_id"]


def resume(api_client, thread_id, write_letter):
    return api_client.post(f"/analyze/{thread_id}/resume", json={"write_letter": write_letter})


def test_resume_writes_the_letter_when_asked(api_client, fake_model):
    thread_id = start_analysis(api_client)

    response = resume(api_client, thread_id, write_letter=True)

    assert response.status_code == 200
    assert response.json() == {"cover_letter": LETTER}


def test_resume_without_a_letter_when_the_user_declines(api_client, fake_model):
    thread_id = start_analysis(api_client)

    response = resume(api_client, thread_id, write_letter=False)

    assert response.status_code == 200
    assert response.json() == {"cover_letter": None}
    assert fake_model.text_calls_made() == []  # no tokens spent on a letter


def test_resume_returns_404_for_an_unknown_analysis(api_client, fake_model):
    response = resume(api_client, uuid.uuid4(), write_letter=True)

    assert response.status_code == 404


def test_resume_returns_409_when_the_analysis_is_finished(api_client, fake_model):
    thread_id = start_analysis(api_client)
    resume(api_client, thread_id, write_letter=True)

    response = resume(api_client, thread_id, write_letter=True)

    assert response.status_code == 409
    assert len(fake_model.text_calls_made()) == 1  # the letter was not written twice


def test_resume_rejects_an_empty_decision(api_client, fake_model):
    thread_id = start_analysis(api_client)

    response = api_client.post(f"/analyze/{thread_id}/resume", json={})

    assert response.status_code == 422
    # The run is still paused, so a valid decision works afterwards
    assert resume(api_client, thread_id, write_letter=False).status_code == 200


def test_resume_rejects_a_thread_id_that_is_not_a_uuid(api_client, fake_model):
    response = resume(api_client, "not-a-uuid", write_letter=True)

    assert response.status_code == 422


def test_resume_returns_502_when_the_letter_fails_and_can_be_retried(api_client, fake_model):
    fake_model.text = [RuntimeError("Groq is down"), LETTER]
    thread_id = start_analysis(api_client)

    failed = resume(api_client, thread_id, write_letter=True)

    assert failed.status_code == 502
    assert "Groq is down" not in failed.text
    # The run paused again at the review: the same request works later
    retried = resume(api_client, thread_id, write_letter=True)
    assert retried.status_code == 200
    assert retried.json() == {"cover_letter": LETTER}


def test_analyze_rejects_a_missing_field(api_client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "job_description"}

    response = api_client.post("/analyze", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "job_description"]


# Empty, whitespace only (stripped first), and one character over the limit.
@pytest.mark.parametrize("cv_text", ["", "   ", "x" * (MAX_CV_LENGTH + 1)])
def test_analyze_rejects_invalid_cv_text(api_client, cv_text):
    response = api_client.post("/analyze", json={**VALID_PAYLOAD, "cv_text": cv_text})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "cv_text"]


def test_analyze_rejects_a_too_long_job_description(api_client):
    too_long = "x" * (MAX_JOB_DESCRIPTION_LENGTH + 1)

    response = api_client.post("/analyze", json={**VALID_PAYLOAD, "job_description": too_long})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "job_description"]


def test_response_schema_rejects_a_score_above_one():
    with pytest.raises(ValidationError):
        AnalyzeResponse(required_skills=[], matched_skills=[], missing_skills=[], match_score=1.5)


def test_review_decision_accepts_a_boolean():
    assert ReviewDecision.model_validate({"write_letter": True}).write_letter is True
    assert ReviewDecision.model_validate({"write_letter": False}).write_letter is False


@pytest.mark.parametrize(
    "body",
    [
        {},  # LangGraph would read Command(resume={}) as no answer and stay paused
        {"write_leter": True},  # a typo must not pass as "no letter"
        {"write_letter": True, "extra": 1},
        {"write_letter": "yes"},  # strict: only a JSON boolean
        {"write_letter": 1},
        {"write_letter": None},
    ],
)
def test_review_decision_rejects_anything_but_one_boolean(body):
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate(body)
