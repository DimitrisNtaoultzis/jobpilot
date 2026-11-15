"""Comparing skill lists: plain, deterministic Python (no LLM involved)."""

from jobpilot.skills import normalize_skill

from .schemas import AnalyzeResponse


def compare_skills(job_skills: list[str], cv_skills: list[str]) -> AnalyzeResponse:
    """Compare the skills a job asks for with the skills found in a CV.

    Skills are compared by their normalized names (e.g. "K8s" == "Kubernetes"), but the
    result uses the job posting's own names and order.
    """
    cv_keys = {normalize_skill(skill) for skill in cv_skills}

    required: list[str] = []
    seen: set[str] = set()  # the list keeps the order, the set gives fast duplicate checks
    for skill in job_skills:
        key = normalize_skill(skill)
        if not key or key in seen:  # blank or duplicate
            continue
        seen.add(key)
        required.append(skill.strip())

    matched = [skill for skill in required if normalize_skill(skill) in cv_keys]
    missing = [skill for skill in required if normalize_skill(skill) not in cv_keys]
    # Nothing to compare is not a match: 0.0 rather than a misleading 1.0
    score = round(len(matched) / len(required), 2) if required else 0.0

    return AnalyzeResponse(
        required_skills=required,
        matched_skills=matched,
        missing_skills=missing,
        match_score=score,
    )
