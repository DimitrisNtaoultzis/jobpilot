"""Ingestion: sync the curated learning resources file into Postgres (schema "ai").

The file is the source of truth: resources are upserted by URL, and rows whose URL is
no longer in the file are deleted. Every run re-embeds everything (a few dozen short
texts), so all stored vectors always come from the current model.

Run with: uv run python -m jobpilot.ai_service.ingest
"""

from pathlib import Path

import psycopg
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector

from .config import get_settings
from .embeddings import EMBEDDING_MODEL, embed_passages, get_embedding_model
from .resources import LearningResource, load_resources

SCHEMA_SQL = Path(__file__).parent / "schema.sql"

UPSERT_SQL = """
INSERT INTO ai.learning_resources (url, title, description, embedding, embedding_model)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (url) DO UPDATE SET
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    embedding = EXCLUDED.embedding,
    embedding_model = EXCLUDED.embedding_model,
    updated_at = now()
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create the "ai" schema objects if missing, and teach the connection the vector type."""
    # Multiple statements in one execute() are allowed because no parameters are passed.
    conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    # Only after schema.sql: it looks up the "vector" type, created by CREATE EXTENSION.
    register_vector(conn)


def sync_resources(
    conn: psycopg.Connection, model: TextEmbedding, resources: list[LearningResource]
) -> int:
    """Make the database match `resources`; return how many resources are stored.

    Never commits: the caller owns the transaction (main() commits; tests roll back).
    """
    ensure_schema(conn)

    vectors = embed_passages(model, [resource.passage for resource in resources])
    urls = [str(resource.url) for resource in resources]

    with conn.cursor() as cursor:
        cursor.executemany(
            UPSERT_SQL,
            [
                (url, resource.title, resource.description, vector, EMBEDDING_MODEL)
                for url, resource, vector in zip(urls, resources, vectors, strict=True)
            ],
        )
        # A Python list is sent as a Postgres array: delete rows not in the file
        cursor.execute("DELETE FROM ai.learning_resources WHERE url <> ALL(%s)", (urls,))

    return len(resources)


def main() -> None:
    settings = get_settings()
    resources = load_resources()
    # Load the model (maybe a download) before connecting, so no transaction is held open
    model = get_embedding_model()

    # On exit, the connection commits if the block succeeded and rolls back on an
    # exception: the database is never left half-synced.
    with psycopg.connect(settings.database_url.get_secret_value()) as conn:
        count = sync_resources(conn, model, resources)

    print(f"Synced {count} learning resources (embedding model: {EMBEDDING_MODEL}).")


if __name__ == "__main__":
    main()
