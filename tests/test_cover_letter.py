import asyncio

import pytest
from fakes import TEXT, FakeModel

from jobpilot.ai_service.cover_letter import (
    HAS_REQUIRED_GAPS,
    NO_REQUIRED_GAPS,
    SYSTEM_PROMPT,
    CoverLetterError,
    build_messages,
    check_cover_letter,
    write_cover_letter,
)

CV_TEXT = "Backend developer: Python, Django, PostgreSQL."
JOB_TITLE = "Backend Engineer"
JOB_DESCRIPTION = "We need Python, SQL and Kubernetes."


def test_build_messages_wraps_the_texts_and_adds_the_analysis():
    system, user = build_messages(
        CV_TEXT, JOB_TITLE, JOB_DESCRIPTION, ["Python", "SQL"], has_required_gaps=True
    )

    assert system.content == SYSTEM_PROMPT
    assert f"<job_posting>\nTitle: {JOB_TITLE}\n\n{JOB_DESCRIPTION}\n</job_posting>" in user.content
    assert f"<cv>\n{CV_TEXT}\n</cv>" in user.content
    assert f"<analysis>\nMatched skills: Python, SQL\n{HAS_REQUIRED_GAPS}\n</analysis>" in (
        user.content
    )


def test_build_messages_says_when_no_required_skill_is_missing():
    _, user = build_messages(CV_TEXT, JOB_TITLE, JOB_DESCRIPTION, [], has_required_gaps=False)

    assert "Matched skills: none" in user.content
    assert NO_REQUIRED_GAPS in user.content
    assert HAS_REQUIRED_GAPS not in user.content


def test_write_cover_letter_makes_a_plain_call_and_returns_the_trimmed_text():
    model = FakeModel(text="\n  Dear hiring team,\n\nI am applying...  \n")

    letter = asyncio.run(
        write_cover_letter(model, CV_TEXT, JOB_TITLE, JOB_DESCRIPTION, ["Python"], True)
    )

    assert letter == "Dear hiring team,\n\nI am applying..."
    call = model.call_for(TEXT)  # free text: no structured output, no tools
    assert CV_TEXT in call.messages[-1].content


def test_check_cover_letter_accepts_a_normal_letter():
    letter = "Dear hiring team,\n\nI build Python APIs (3 years).\n\nKind regards"

    assert check_cover_letter(letter) == letter


def test_check_cover_letter_rejects_an_empty_letter():
    with pytest.raises(CoverLetterError, match="empty"):
        check_cover_letter("")


@pytest.mark.parametrize("placeholder", ["[Company Name]", "[Your Name]", "[Hiring Manager]"])
def test_check_cover_letter_rejects_a_placeholder(placeholder):
    with pytest.raises(CoverLetterError, match="placeholder"):
        check_cover_letter(f"Dear {placeholder},\n\nI am applying.")
