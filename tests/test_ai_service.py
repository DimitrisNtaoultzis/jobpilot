import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from jobpilot.ai_service.main import app
from jobpilot.ai_service.schemas import MAX_TEXT_LENGTH, AnalyzeResponse

VALID_PAYLOAD = {
    "cv_text": "Python developer",
    "job_title": "Backend Engineer",
    "job_description": "We need Python",
}


@pytest.fixture
def api_client():
    # Not named "client": that name belongs to pytest-django's Django test client.
    return TestClient(app)


def test_health(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_the_response_shape(api_client):
    response = api_client.post("/analyze", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert set(response.json()) == {
        "required_skills",
        "matched_skills",
        "missing_skills",
        "match_score",
    }


def test_analyze_rejects_a_missing_field(api_client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "job_description"}

    response = api_client.post("/analyze", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "job_description"]


# Empty, whitespace only (stripped first), and one character over the limit.
@pytest.mark.parametrize("cv_text", ["", "   ", "x" * (MAX_TEXT_LENGTH + 1)])
def test_analyze_rejects_invalid_cv_text(api_client, cv_text):
    response = api_client.post("/analyze", json={**VALID_PAYLOAD, "cv_text": cv_text})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "cv_text"]


def test_response_schema_rejects_a_score_above_one():
    with pytest.raises(ValidationError):
        AnalyzeResponse(required_skills=[], matched_skills=[], missing_skills=[], match_score=1.5)
