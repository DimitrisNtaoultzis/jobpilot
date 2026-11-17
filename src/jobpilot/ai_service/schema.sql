-- Database objects owned by the AI service (RAG over learning resources).
-- Django owns the "public" schema; the AI service keeps its tables in "ai".
-- Idempotent (IF NOT EXISTS everywhere): safe to run before every ingestion.
-- No migration tool yet: when a schema change is needed, introduce Alembic.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS ai;

CREATE TABLE IF NOT EXISTS ai.learning_resources (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- Natural key: ingestion upserts on it (INSERT ... ON CONFLICT (url) DO UPDATE)
    url             text        NOT NULL UNIQUE,
    title           text        NOT NULL,
    description     text        NOT NULL,
    -- 384 = output size of BAAI/bge-small-en-v1.5; a model with another size needs a schema change
    embedding       vector(384) NOT NULL,
    -- Vectors from different models are not comparable: changing the model means re-embedding all rows
    embedding_model text        NOT NULL,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- No vector index (HNSW/IVFFlat) on purpose: with a few dozen rows an exact scan is
-- sub-millisecond, while those indexes trade accuracy for speed at much larger sizes.
