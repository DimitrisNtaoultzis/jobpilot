"""Curated learning resources for RAG: loading and validating the data file."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

# Inside the package, so the data ships with the code (e.g. in a Docker image)
RESOURCES_FILE = Path(__file__).parent / "data" / "learning_resources.toml"


class LearningResource(BaseModel):
    """One learning resource, as written in the data file."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",  # an unknown key is most likely a typo, e.g. "descripton"
        frozen=True,  # loaded data is read-only
    )

    title: str = Field(min_length=1)
    url: HttpUrl
    description: str = Field(min_length=1)

    # mode="after": runs once the value is a valid, stripped, non-empty str
    @field_validator("title", "description", mode="after")
    @classmethod
    def collapse_whitespace(cls, value: str) -> str:
        """Turn line breaks and repeated spaces into single spaces.

        TOML multi-line strings keep the file's line breaks, which should not reach
        the API (or the stored text we embed).
        """
        return " ".join(value.split())

    @property
    def passage(self) -> str:
        """The text we embed: the title often holds the key term (e.g. "Kubernetes Basics")."""
        return f"{self.title}: {self.description}"


def load_resources(path: Path = RESOURCES_FILE) -> list[LearningResource]:
    """Load and validate the resources; raise on invalid entries or duplicate URLs."""
    with path.open("rb") as file:  # tomllib reads bytes: TOML files are always UTF-8
        data = tomllib.load(file)

    resources = [LearningResource.model_validate(item) for item in data["resources"]]

    # The URL is the upsert key: a duplicate would silently overwrite the first resource.
    seen: set[str] = set()
    for resource in resources:
        url = str(resource.url)  # compared after HttpUrl normalization
        if url in seen:
            raise ValueError(f"Duplicate URL in {path.name}: {url}")
        seen.add(url)

    return resources
