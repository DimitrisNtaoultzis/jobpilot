import pytest

from jobpilot.skills import normalize_skill


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Python", "python"),  # lowercase
        ("  Docker  ", "docker"),  # leading/trailing whitespace
        ("Machine   Learning", "machine learning"),  # repeated inner whitespace
        ("K8s", "kubernetes"),  # known alias
        ("postgres", "postgresql"),  # known alias
        ("Postgres", "postgresql"),  # known alias with uppercase letter
        ("RAG", "retrieval-augmented generation"),  # AI/ML abbreviations
        ("LLMs", "large language models"),
        ("ML", "machine learning"),
        ("NLP", "natural language processing"),
    ],
)
def test_normalize_skill(raw: str, expected: str) -> None:
    assert normalize_skill(raw) == expected


def test_normalize_skill_keeps_unknown_skills() -> None:
    assert normalize_skill("LangGraph") == "langgraph"
