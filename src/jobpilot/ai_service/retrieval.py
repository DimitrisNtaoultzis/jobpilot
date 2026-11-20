"""Retrieval (the R in RAG): semantic search over the learning resources."""

import numpy as np
import psycopg
from fastembed import TextEmbedding
from psycopg.rows import class_row
from psycopg_pool import ConnectionPool

from .embeddings import embed_query
from .schemas import ResourceMatch, SkillResources

# Seconds to wait for a pooled connection (the pool's default is 30): if the database is
# down, give up quickly so the request can still return its skill analysis.
POOL_TIMEOUT = 5.0

# <=> is pgvector's cosine distance (1 - cosine similarity).
# ORDER BY uses the distance expression itself (not the alias) so that a vector index,
# if we ever add one, can serve the query.
# ::float8 tells Postgres the type of min_similarity even when it is NULL.
SEARCH_SQL = """
SELECT title, url, description, 1 - (embedding <=> %(query)s) AS similarity
FROM ai.learning_resources
WHERE %(min_similarity)s::float8 IS NULL
   OR 1 - (embedding <=> %(query)s) >= %(min_similarity)s::float8
ORDER BY embedding <=> %(query)s
LIMIT %(limit)s
"""


def search_resources(
    conn: psycopg.Connection,
    query_vector: np.ndarray,
    limit: int = 3,
    min_similarity: float | None = None,
) -> list[ResourceMatch]:
    """The `limit` resources most similar to `query_vector`, most similar first.

    `query_vector` should come from embed_query(). With `min_similarity`, weaker matches
    are dropped, so fewer than `limit` (even zero) results may be returned.
    The connection must know the vector type (see ingest.ensure_schema / register_vector).
    """
    # class_row builds a ResourceMatch from each row, matching columns to fields by name
    with conn.cursor(row_factory=class_row(ResourceMatch)) as cursor:
        cursor.execute(
            SEARCH_SQL,
            {"query": query_vector, "min_similarity": min_similarity, "limit": limit},
        )
        return cursor.fetchall()


def find_learning_resources(
    conn: psycopg.Connection, model: TextEmbedding, skills: list[str], limit: int = 3
) -> list[SkillResources]:
    """Search learning resources for each skill (e.g. the missing ones), keeping their order.

    Blocking work (CPU-bound embedding, a sync database query): async code should run it
    in a worker thread, e.g. with asyncio.to_thread().
    """
    return [
        SkillResources(
            skill=skill,
            resources=search_resources(conn, embed_query(model, skill), limit=limit),
        )
        for skill in skills
    ]


class ResourceFinder:
    """Finds learning resources for skills, with a connection pool and an embedding model.

    One object, created at startup, so the endpoint has a single dependency to call
    (and tests a single dependency to replace). Calling it is blocking.
    """

    def __init__(self, pool: ConnectionPool, model: TextEmbedding) -> None:
        self.pool = pool
        self.model = model

    def __call__(self, skills: list[str], limit: int = 3) -> list[SkillResources]:
        # Borrow a connection; it goes back to the pool at the end of the block
        with self.pool.connection(timeout=POOL_TIMEOUT) as conn:
            return find_learning_resources(conn, self.model, skills, limit=limit)
