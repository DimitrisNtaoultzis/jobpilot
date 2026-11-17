"""Ingestion tests against the real (test) Postgres: ON CONFLICT, the vector type and the
schema cannot be faked. pg_conn and fake_embedding_model come from conftest.py."""

import numpy as np

from jobpilot.ai_service.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL
from jobpilot.ai_service.ingest import sync_resources
from jobpilot.ai_service.resources import LearningResource


def make_resource(n, description="A resource about something."):
    return LearningResource(
        title=f"Resource {n}", url=f"https://example.com/{n}", description=description
    )


def fetch_rows(conn):
    """All stored resources, as dicts keyed by column name, ordered by URL."""
    cursor = conn.execute(
        "SELECT url, title, description, embedding, embedding_model"
        " FROM ai.learning_resources ORDER BY url"
    )
    columns = [column.name for column in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def test_first_sync_creates_the_table_and_inserts_every_resource(pg_conn, fake_embedding_model):
    count = sync_resources(pg_conn, fake_embedding_model, [make_resource(1), make_resource(2)])

    assert count == 2
    assert [row["url"] for row in fetch_rows(pg_conn)] == [
        "https://example.com/1",
        "https://example.com/2",
    ]


def test_sync_embeds_each_passage(pg_conn, fake_embedding_model):
    resources = [make_resource(1), make_resource(2)]

    sync_resources(pg_conn, fake_embedding_model, resources)

    assert fake_embedding_model.texts == [r.passage for r in resources]


def test_sync_stores_the_vectors_and_the_model_name(pg_conn, fake_embedding_model):
    sync_resources(pg_conn, fake_embedding_model, [make_resource(1), make_resource(2)])

    first, second = fetch_rows(pg_conn)
    # pgvector returns its own Vector type (numpy in, Vector out); the fake gives
    # the i-th text a vector filled with i
    assert np.array_equal(first["embedding"].to_numpy(), np.full(EMBEDDING_DIM, 0.0))
    assert np.array_equal(second["embedding"].to_numpy(), np.full(EMBEDDING_DIM, 1.0))
    assert first["embedding_model"] == EMBEDDING_MODEL


def test_sync_twice_creates_no_duplicates(pg_conn, fake_embedding_model):
    resources = [make_resource(1), make_resource(2)]

    sync_resources(pg_conn, fake_embedding_model, resources)
    sync_resources(pg_conn, fake_embedding_model, resources)

    assert len(fetch_rows(pg_conn)) == 2


def test_sync_updates_a_changed_resource(pg_conn, fake_embedding_model):
    sync_resources(pg_conn, fake_embedding_model, [make_resource(1)])

    sync_resources(pg_conn, fake_embedding_model, [make_resource(1, description="Updated.")])

    [row] = fetch_rows(pg_conn)
    assert row["description"] == "Updated."


def test_sync_deletes_resources_removed_from_the_file(pg_conn, fake_embedding_model):
    sync_resources(pg_conn, fake_embedding_model, [make_resource(1), make_resource(2)])

    count = sync_resources(pg_conn, fake_embedding_model, [make_resource(2)])

    assert count == 1
    assert [row["url"] for row in fetch_rows(pg_conn)] == ["https://example.com/2"]
