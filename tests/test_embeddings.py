from pathlib import Path

from jobpilot.ai_service.embeddings import (
    EMBEDDING_DIM,
    QUERY_PREFIX,
    embed_passages,
    embed_query,
)

SCHEMA_SQL = Path(__file__).resolve().parents[1] / "src/jobpilot/ai_service/schema.sql"

# fake_embedding_model comes from conftest.py


def test_embed_query_adds_the_bge_query_prefix(fake_embedding_model):
    model = fake_embedding_model

    vector = embed_query(model, "Docker")

    assert model.texts == [QUERY_PREFIX + "Docker"]
    assert vector.shape == (EMBEDDING_DIM,)


def test_embed_passages_sends_the_texts_unchanged(fake_embedding_model):
    model = fake_embedding_model

    embed_passages(model, ["SQL Tutorial", "Docker for Beginners"])

    assert model.texts == ["SQL Tutorial", "Docker for Beginners"]


def test_embed_passages_returns_one_vector_per_text_in_order(fake_embedding_model):
    model = fake_embedding_model

    vectors = embed_passages(model, ["first", "second", "third"])

    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]


def test_embedding_dim_matches_the_database_schema():
    # The vector column size is fixed in SQL; it must follow the model's output size.
    assert f"vector({EMBEDDING_DIM})" in SCHEMA_SQL.read_text(encoding="utf-8")
