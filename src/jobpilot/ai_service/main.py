"""JobPilot AI service: the FastAPI app that runs the analysis (LLM, agent).

Run locally with:
    uv run uvicorn jobpilot.ai_service.main:app --reload --port 8001
"""

from fastapi import FastAPI

from .schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI(title="JobPilot AI service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check: the service is up and answering requests."""
    return {"status": "ok"}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    """Compare a CV with a job posting."""
    # Placeholder until the LLM is wired in (step 2.3): right shape, no real analysis yet.
    return AnalyzeResponse(
        required_skills=[], matched_skills=[], missing_skills=[], match_score=0.0
    )
