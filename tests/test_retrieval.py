"""Search tests against the real (test) Postgres, with hand-made vectors whose
similarities are known in advance. pg_conn comes from conftest.py."""

import numpy as np
import pytest

from jobpilot.ai_service.embeddings import EMBEDDING_DIM, QUERY_PREFIX
from jobpilot.ai_service.ingest import ensure_schema
from jobpilot.ai_service.retrieval import find_learning_resources, search_resources


def basis(i):
    """Unit vector along axis i: e0 = [1, 0, 0, ...], e1 = [0, 1, 0, ...]."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[i] = 1.0
    return vector


E0, E1 = basis(0), basis(1)
QUERY = E0


def insert_resource(conn, name, vector):
    conn.execute(
        "INSERT INTO ai.learning_resources (url, title, description, embedding, embedding_model)"
        " VALUES (%s, %s, %s, %s, %s)",
        (f"https://example.com/{name}", name, f"About {name}", vector, "test-model"),
    )


@pytest.fixture
def resources_table(pg_conn):
    """Three resources at known angles from QUERY (= e0)."""
    ensure_schema(pg_conn)
    insert_resource(pg_conn, "C", E1)  # 90 degrees: similarity 0
    insert_resource(pg_conn, "A", E0)  # same direction: similarity 1
    insert_resource(pg_conn, "B", (E0 + E1) / np.sqrt(2))  # 45 degrees: similarity ~0.707
    return pg_conn


def test_results_are_ordered_from_most_to_least_similar(resources_table):
    results = search_resources(resources_table, QUERY)

    assert [r.title for r in results] == ["A", "B", "C"]


def test_results_carry_the_cosine_similarity(resources_table):
    results = search_resources(resources_table, QUERY)

    # float32 vectors: compare approximately
    assert [r.similarity for r in results] == pytest.approx([1.0, 1 / np.sqrt(2), 0.0], abs=1e-6)


def test_results_carry_the_resource_fields(resources_table):
    [first, *_] = search_resources(resources_table, QUERY)

    assert first.url == "https://example.com/A"
    assert first.description == "About A"


def test_limit_caps_the_number_of_results(resources_table):
    results = search_resources(resources_table, QUERY, limit=2)

    assert [r.title for r in results] == ["A", "B"]


def test_min_similarity_drops_weak_matches(resources_table):
    results = search_resources(resources_table, QUERY, min_similarity=0.5)

    assert [r.title for r in results] == ["A", "B"]


def test_an_empty_table_gives_no_results(pg_conn):
    ensure_schema(pg_conn)

    assert search_resources(pg_conn, QUERY) == []


# find_learning_resources: one search per missing skill


class AxisEmbeddingModel:
    """Fake embedding model: a text containing a key of `axes` gets that axis' unit vector.

    Unlike FakeEmbeddingModel (parallel vectors), this gives each skill a known direction,
    so we know in advance which resource is closest to it.
    """

    def __init__(self, axes):
        self.axes = axes
        self.texts = []

    def embed(self, texts):
        for text in texts:
            self.texts.append(text)
            [axis] = [axis for key, axis in self.axes.items() if key in text]
            yield basis(axis)


@pytest.fixture
def axis_model():
    # Kubernetes points like A (e0); SQL points like C (e1)
    return AxisEmbeddingModel({"Kubernetes": 0, "SQL": 1})


def test_find_learning_resources_gives_one_entry_per_skill_in_order(resources_table, axis_model):
    results = find_learning_resources(resources_table, axis_model, ["Kubernetes", "SQL"])

    assert [entry.skill for entry in results] == ["Kubernetes", "SQL"]
    assert [r.title for r in results[0].resources] == ["A", "B", "C"]
    assert [r.title for r in results[1].resources] == ["C", "B", "A"]


def test_find_learning_resources_embeds_each_skill_as_a_query(resources_table, axis_model):
    find_learning_resources(resources_table, axis_model, ["Kubernetes", "SQL"])

    assert axis_model.texts == [QUERY_PREFIX + "Kubernetes", QUERY_PREFIX + "SQL"]


def test_find_learning_resources_applies_the_limit_per_skill(resources_table, axis_model):
    results = find_learning_resources(resources_table, axis_model, ["Kubernetes", "SQL"], limit=1)

    assert [[r.title for r in entry.resources] for entry in results] == [["A"], ["C"]]


def test_no_missing_skills_gives_no_learning_resources(resources_table, axis_model):
    assert find_learning_resources(resources_table, axis_model, []) == []
