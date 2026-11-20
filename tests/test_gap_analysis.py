import asyncio

from fakes import FakeModel

from jobpilot.ai_service.gap_analysis import GapReview, SkillGap, build_messages, review_gaps

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
    review = GapReview(covered=[], gaps=[SkillGap(skill="Terraform", priority="nice_to_have")])
    model = FakeModel(extraction=review)  # FakeModel returns whatever it is given

    result = asyncio.run(review_gaps(model, ["Terraform"], ["Python"], JOB_TITLE, JOB_DESCRIPTION))

    assert result == review
    assert model.structured_output_kwargs == {"method": "json_schema", "strict": True}
    assert "<missing_skills>\nTerraform\n</missing_skills>" in model.messages[-1].content
