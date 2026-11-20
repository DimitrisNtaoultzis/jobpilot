import asyncio

from fakes import FakeModel

from jobpilot.ai_service.gap_analysis import (
    CoveredSkill,
    GapReview,
    ReviewedGap,
    apply_gap_review,
    build_messages,
    review_gaps,
)
from jobpilot.ai_service.schemas import AnalyzeResponse, SkillGap

JOB_TITLE = "ML Platform Engineer"
JOB_DESCRIPTION = "You will use SQL and Docker. Terraform is a plus."


def test_build_messages_wraps_each_input_in_its_own_tags():
    messages = build_messages(
        ["SQL", "Terraform"], ["PostgreSQL", "Python"], JOB_TITLE, JOB_DESCRIPTION
    )

    user_message = messages[-1].content
    assert f"<job_posting>\nTitle: {JOB_TITLE}\n\n{JOB_DESCRIPTION}\n</job_posting>" in user_message
    assert "<cv_skills>\nPostgreSQL\nPython\n</cv_skills>" in user_message
    assert "<missing_skills>\nSQL\nTerraform\n</missing_skills>" in user_message


def test_review_gaps_uses_strict_structured_output():
    review = GapReview(covered=[], gaps=[ReviewedGap(skill="Terraform", priority="nice_to_have")])
    model = FakeModel(extraction=review)  # FakeModel returns whatever it is given

    result = asyncio.run(review_gaps(model, ["Terraform"], ["Python"], JOB_TITLE, JOB_DESCRIPTION))

    assert result == review
    assert model.structured_output_kwargs == {"method": "json_schema", "strict": True}
    assert "<missing_skills>\nTerraform\n</missing_skills>" in model.messages[-1].content


# --- apply_gap_review: Python enforces what the prompt only asks for ---

CV_SKILLS = ["Python", "PostgreSQL"]


def make_result():
    """What compare_skills returns for a job asking for 4 skills, 1 of them in the CV."""
    return AnalyzeResponse(
        required_skills=["Python", "SQL", "Docker", "Terraform"],
        matched_skills=["Python"],
        missing_skills=["SQL", "Docker", "Terraform"],
        match_score=0.25,
    )


def review(covered=(), gaps=()):
    """Build a GapReview from (skill, cv_skill) and (skill, priority) pairs."""
    return GapReview(
        covered=[CoveredSkill(skill=s, cv_skill=c) for s, c in covered],
        gaps=[ReviewedGap(skill=s, priority=p) for s, p in gaps],
    )


def test_a_covered_skill_moves_to_matched_and_the_score_is_recomputed():
    llm_review = review(
        covered=[("SQL", "PostgreSQL")],
        gaps=[("Docker", "must_have"), ("Terraform", "nice_to_have")],
    )

    result = apply_gap_review(make_result(), CV_SKILLS, llm_review)

    assert result.matched_skills == ["Python", "SQL"]  # job posting order
    assert result.missing_skills == ["Docker", "Terraform"]
    assert result.match_score == 0.5
    assert result.skill_gaps == [
        SkillGap(skill="Docker", priority="must_have"),
        SkillGap(skill="Terraform", priority="nice_to_have"),
    ]


def test_a_cover_by_a_skill_that_is_not_in_the_cv_is_rejected():
    llm_review = review(covered=[("SQL", "MySQL")])  # MySQL is not in the CV

    result = apply_gap_review(make_result(), CV_SKILLS, llm_review)

    assert "SQL" in result.missing_skills
    assert SkillGap(skill="SQL", priority="must_have") in result.skill_gaps


def test_a_covered_skill_that_was_not_missing_is_ignored():
    llm_review = review(covered=[("Rust", "Python")])

    result = apply_gap_review(make_result(), CV_SKILLS, llm_review)

    assert "Rust" not in result.matched_skills
    assert result.matched_skills == ["Python"]


def test_a_gap_for_a_skill_that_was_not_missing_is_ignored():
    llm_review = review(gaps=[("Rust", "must_have")])

    result = apply_gap_review(make_result(), CV_SKILLS, llm_review)

    assert [gap.skill for gap in result.skill_gaps] == ["SQL", "Docker", "Terraform"]


def test_a_missing_skill_the_llm_forgot_becomes_a_must_have_gap():
    llm_review = review(gaps=[("Terraform", "nice_to_have")])  # SQL and Docker forgotten

    result = apply_gap_review(make_result(), CV_SKILLS, llm_review)

    assert result.skill_gaps == [
        SkillGap(skill="SQL", priority="must_have"),
        SkillGap(skill="Docker", priority="must_have"),
        SkillGap(skill="Terraform", priority="nice_to_have"),
    ]


def test_a_skill_in_both_lists_stays_a_gap():
    # "When in doubt, it is a gap": a contradictory answer is treated as doubt
    llm_review = review(covered=[("SQL", "PostgreSQL")], gaps=[("SQL", "nice_to_have")])

    result = apply_gap_review(make_result(), CV_SKILLS, llm_review)

    assert "SQL" in result.missing_skills
    assert SkillGap(skill="SQL", priority="nice_to_have") in result.skill_gaps


def test_skills_are_compared_by_normalized_name_but_keep_the_job_posting_names():
    llm_review = review(covered=[("sql", "postgresql")])

    result = apply_gap_review(make_result(), CV_SKILLS, llm_review)

    assert result.matched_skills == ["Python", "SQL"]


def test_the_original_result_is_not_changed():
    original = make_result()

    apply_gap_review(original, CV_SKILLS, review(covered=[("SQL", "PostgreSQL")]))

    assert original == make_result()
