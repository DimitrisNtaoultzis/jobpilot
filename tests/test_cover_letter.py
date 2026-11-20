import asyncio

import pytest
from fakes import FakeModel

from jobpilot.ai_service.cover_letter import (
    MAX_LETTER_ATTEMPTS,
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


def write(model):
    """write_cover_letter for this module's example: Python matched, Kubernetes missing."""
    return asyncio.run(
        write_cover_letter(model, CV_TEXT, JOB_TITLE, JOB_DESCRIPTION, ["Python"], ["Kubernetes"])
    )


def test_write_cover_letter_makes_a_plain_call_and_returns_the_trimmed_text():
    model = FakeModel(text="\n  Dear hiring team,\n\nI am applying...  \n")

    letter = write(model)

    assert letter == "Dear hiring team,\n\nI am applying..."
    [call] = model.text_calls_made()  # one call: free text, no schema, no tools
    assert CV_TEXT in call.messages[-1].content


def test_a_letter_that_fails_a_check_is_rewritten_with_the_exact_problem():
    bad = "Dear hiring team,\n\nI deploy services with Kubernetes."
    good = "Dear hiring team,\n\nI build Python services."
    model = FakeModel(text=[bad, good])

    letter = write(model)

    assert letter == good
    first, second = model.text_calls_made()
    # The rewrite continues the conversation: the model's own letter, then the problem
    assert second.messages[:-2] == first.messages
    assert second.messages[-2].content == bad
    feedback = second.messages[-1].content
    assert "names skills that the CV does not show: Kubernetes" in feedback
    assert "Rewrite the whole letter" in feedback


def test_write_cover_letter_gives_up_after_the_rewrite_also_fails():
    bad = "Dear hiring team,\n\nI know Kubernetes."
    model = FakeModel(text=[bad, bad, "never asked for"])

    with pytest.raises(CoverLetterError, match="Kubernetes"):
        write(model)

    assert len(model.text_calls_made()) == MAX_LETTER_ATTEMPTS


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

    with pytest.raises(CoverLetterError, match="does not show: Docker, Kubernetes"):
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
