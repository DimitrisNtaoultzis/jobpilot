import pytest
from fakes import FakeModel, FakeResourceFinder
from fastapi.testclient import TestClient
from pydantic import ValidationError

from jobpilot.ai_service.extraction import SkillExtraction
from jobpilot.ai_service.llm import get_chat_model
from jobpilot.ai_service.main import app, get_resource_finder
from jobpilot.ai_service.schemas import (
    MAX_CV_LENGTH,
    MAX_JOB_DESCRIPTION_LENGTH,
    AnalyzeResponse,
    ResourceMatch,
    SkillResources,
)

VALID_PAYLOAD = {
    "cv_text": "Python developer",
    "job_title": "Backend Engineer",
    "job_description": "We need Python",
}


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
    model = FakeModel(SkillExtraction(job_skills=["Python", "Docker"], cv_skills=["python"]))
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


def test_health(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_the_response_shape(api_client, fake_model):
    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert set(response.json()) == {
        "required_skills",
        "matched_skills",
        "missing_skills",
        "match_score",
        "skill_gaps",
        "learning_resources",
    }


def test_analyze_compares_the_extracted_skills(api_client, fake_model):
    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    assert response.json() == {
        "required_skills": ["Python", "Docker"],
        "matched_skills": ["Python"],
        "missing_skills": ["Docker"],
        "match_score": 0.5,
        "skill_gaps": [],  # the graph has no gap review node yet
        "learning_resources": [],  # the default fake finder finds nothing
    }


def test_analyze_searches_resources_for_the_missing_skills(api_client, fake_model, fake_finder):
    fake_finder.results = [DOCKER_RESOURCES]

    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    assert fake_finder.skills == ["Docker"]  # only the missing skills, not "Python"
    assert response.json()["learning_resources"] == [DOCKER_RESOURCES.model_dump()]


def test_analyze_skips_the_search_when_nothing_is_missing(api_client, fake_model, fake_finder):
    fake_model.responses[SkillExtraction] = SkillExtraction(
        job_skills=["Python"], cv_skills=["Python"]
    )

    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    assert fake_finder.skills is None  # never called: no database round trip
    assert response.json()["learning_resources"] == []


def test_analyze_still_succeeds_when_the_search_fails(api_client, fake_model, fake_finder):
    fake_finder.error = RuntimeError("database is down")

    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    # Graceful degradation: the skill analysis is returned without resources
    assert response.status_code == 200
    assert response.json()["missing_skills"] == ["Docker"]
    assert response.json()["learning_resources"] == []
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
