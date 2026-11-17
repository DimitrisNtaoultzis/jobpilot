import pytest
from pydantic import ValidationError

from jobpilot.ai_service.resources import load_resources

DOCKER = """
[[resources]]
title = "Docker Get Started"
url = "https://docs.docker.com/get-started/"
description = "Official Docker tutorial: images, containers and Compose."
"""

SQL = """
[[resources]]
title = "SQL Tutorial"
url = "https://example.com/sql/"
description = "SELECT, JOIN and GROUP BY on relational databases."
"""


@pytest.fixture
def write_toml(tmp_path):
    """Factory fixture: write_toml(text) -> path of a TOML file holding that text."""

    def _write(text):
        path = tmp_path / "resources.toml"
        path.write_text(text, encoding="utf-8")
        return path

    return _write


def test_load_resources_reads_every_resource_in_order(write_toml):
    resources = load_resources(write_toml(DOCKER + SQL))

    assert [r.title for r in resources] == ["Docker Get Started", "SQL Tutorial"]
    assert str(resources[0].url) == "https://docs.docker.com/get-started/"
    assert resources[0].description == "Official Docker tutorial: images, containers and Compose."


def test_passage_combines_title_and_description(write_toml):
    [resource] = load_resources(write_toml(DOCKER))

    assert resource.passage == (
        "Docker Get Started: Official Docker tutorial: images, containers and Compose."
    )


def test_text_fields_are_stripped(write_toml):
    text = DOCKER.replace('"Docker Get Started"', '"  Docker Get Started \\n"')

    [resource] = load_resources(write_toml(text))

    assert resource.title == "Docker Get Started"


def test_line_breaks_and_repeated_spaces_inside_text_are_collapsed(write_toml):
    # TOML multi-line strings keep the file's line breaks; the API should not show them
    text = DOCKER.replace(
        '"Official Docker tutorial: images, containers and Compose."',
        '"""\nOfficial Docker tutorial:\nimages,   containers\nand Compose."""',
    )

    [resource] = load_resources(write_toml(text))

    assert resource.description == "Official Docker tutorial: images, containers and Compose."


def test_an_invalid_url_is_rejected(write_toml):
    text = DOCKER.replace("https://docs.docker.com/get-started/", "docs.docker.com")

    with pytest.raises(ValidationError):
        load_resources(write_toml(text))


def test_an_empty_description_is_rejected(write_toml):
    text = DOCKER.replace('"Official Docker tutorial: images, containers and Compose."', '"   "')

    with pytest.raises(ValidationError):
        load_resources(write_toml(text))


def test_an_unknown_field_is_rejected(write_toml):
    # Catches typos such as "descripton", which would otherwise be ignored silently
    text = DOCKER + 'kind = "docs"\n'

    with pytest.raises(ValidationError):
        load_resources(write_toml(text))


def test_duplicate_urls_are_rejected(write_toml):
    # The URL is the upsert key: a duplicate would silently overwrite the first resource
    duplicate = SQL.replace("https://example.com/sql/", "https://docs.docker.com/get-started/")

    with pytest.raises(ValueError, match="Duplicate URL"):
        load_resources(write_toml(DOCKER + duplicate))


def test_the_bundled_resources_file_is_valid():
    # The real data file, loaded with the default path: a broken edit fails CI
    assert len(load_resources()) > 0
