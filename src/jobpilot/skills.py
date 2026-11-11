"""Utilities for working with skill names."""

# Synonyms/abbreviations -> the canonical form of a skill.
# Keys are already lowercase, with no extra whitespace.
SKILL_ALIASES: dict[str, str] = {
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
}


def normalize_skill(name: str) -> str:
    """Return the canonical form of a skill name, so that skills can be compared.

    Example: "  K8s " -> "kubernetes", "Machine   Learning" -> "machine learning".
    """
    cleaned = " ".join(name.split()).lower()
    return SKILL_ALIASES.get(cleaned, cleaned)
