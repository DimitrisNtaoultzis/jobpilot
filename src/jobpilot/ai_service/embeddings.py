"""Text embeddings: the one place that knows which embedding model we use.

The model is a constant, not a setting: stored vectors and the vector(384) column in
schema.sql depend on it. Changing it means a schema change and re-embedding every row.
"""

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # output size of EMBEDDING_MODEL; must match vector(...) in schema.sql

# BGE v1.5 instruction for short-query -> passage retrieval. Passages get no prefix.
# Added by us on purpose: fastembed's query_embed() adds nothing for this model, and
# another model's query_embed() might add its own prefix on top of ours.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def get_embedding_model() -> TextEmbedding:
    """Load the embedding model once per process.

    Unlike the chat model (a light HTTP client), this loads the model weights and an
    ONNX Runtime session; the first call also downloads the weights (~67 MB) to a cache.
    """
    return TextEmbedding(EMBEDDING_MODEL)


def embed_query(model: TextEmbedding, text: str) -> np.ndarray:
    """Embed a search query (e.g. a missing skill), with the BGE query prefix."""
    [vector] = model.embed([QUERY_PREFIX + text])
    return vector


def embed_passages(model: TextEmbedding, texts: list[str]) -> list[np.ndarray]:
    """Embed documents to search in (e.g. learning resource descriptions), unchanged."""
    # embed() returns a lazy generator; list() runs it and keeps the vectors.
    return list(model.embed(texts))
