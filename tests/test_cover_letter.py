import asyncio

import pytest
from fakes import TEXT, FakeModel

from jobpilot.ai_service.cover_letter import (
    SYSTEM_PROMPT,
    CoverLetterError,
    build_messages,
    check_cover_letter,
    mentions_skill,
    write_cover_letter,
)

CV_TEXT = "Backend developer: Python, Django, PostgreSQL."
JOB_TITLE = "Backend Engineer"
JOB_DESCRIPTION = "We need Python, SQL and Kubernetes."


def test_build_messages_wraps_the_texts_and_adds_the_matched_skills():
    system, user = build_messages(CV_TEXT, JOB_TITLE, JOB_DESCRIPTION, ["Python", "SQL"])

    assert system.content == SYSTEM_PROMPT
    assert f"<job_posting>\nTitle: {JOB_TITLE}\n\n{JOB_DESCRIPTION}\n</job_posting>" in user.content
    assert f"<cv>\n{CV_TEXT}\n</cv>" in user.content
    assert "<analysis>\nMatched skills: Python, SQL\n</analysis>" in user.content


def test_build_messages_says_none_when_no_skill_matched():
    _, user = build_messages(CV_TEXT, JOB_TITLE, JOB_DESCRIPTION, [])

    assert "Matched skills: none" in user.content


def test_write_cover_letter_makes_a_plain_call_and_returns_the_trimmed_text():
    model = FakeModel(text="\n  Dear hiring team,\n\nI am applying...  \n")

    letter = asyncio.run(write_cover_letter(model, CV_TEXT, JOB_TITLE, JOB_DESCRIPTION, ["Python"]))

    assert letter == "Dear hiring team,\n\nI am applying..."
    call = model.call_for(TEXT)  # free text: no structured output, no tools
    assert CV_TEXT in call.messages[-1].content


MISSING = ["Docker", "Kubernetes"]


def test_check_cover_letter_accepts_a_normal_letter():
    letter = "Dear hiring team,\n\nI build Python APIs (3 years).\n\nKind regards"

    assert check_cover_letter(letter, MISSING) == letter


def test_check_cover_letter_rejects_an_empty_letter():
    with pytest.raises(CoverLetterError, match="empty"):
        check_cover_letter("", MISSING)


@pytest.mark.parametrize("placeholder", ["[Company Name]", "[Your Name]", "[Hiring Manager]"])
def test_check_cover_letter_rejects_a_placeholder(placeholder):
    with pytest.raises(CoverLetterError, match="placeholder"):
        check_cover_letter(f"Dear {placeholder},\n\nI am applying.", MISSING)


def test_check_cover_letter_rejects_a_letter_that_names_missing_skills():
    letter = "My resume does not yet list Docker or kubernetes, but I learn fast."

    with pytest.raises(CoverLetterError, match="names missing skills: Docker, Kubernetes"):
        check_cover_letter(letter, MISSING)


@pytest.mark.parametrize(
    ("text", "skill", "named"),
    [
        ("I know Kubernetes.", "Kubernetes", True),
        ("I know kubernetes.", "Kubernetes", True),  # long names: any case
        ("I write JavaScript.", "Java", False),  # whole words only
        ("Pipelines with CI/CD.", "CI/CD", True),  # names with symbols
        ("I wrote C++ code.", "C++", True),
        ("I go to work.", "Go", False),  # short names: exact case
        ("I built services in Go.", "Go", True),
        ("For the rest, I learn fast.", "REST", False),  # acronyms: exact case
        ("I built REST APIs.", "REST", True),
    ],
)
def test_mentions_skill(text, skill, named):
    assert mentions_skill(text, skill) is named
