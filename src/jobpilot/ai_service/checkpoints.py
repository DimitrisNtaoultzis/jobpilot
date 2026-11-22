"""The Postgres checkpointer: where paused analyses wait for the user's answer."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

# How long startup waits for the database before it gives up (fail fast)
CONNECT_TIMEOUT = 10  # seconds

CONNECTION_KWARGS = {
    # Required by AsyncPostgresSaver (the same settings as its from_conn_string)
    "autocommit": True,
    "row_factory": dict_row,
    "prepare_threshold": 0,
    # Its SQL names the tables without a schema: keep them in "ai", away from
    # Django's tables in "public"
    "options": "-c search_path=ai",
}


@asynccontextmanager
async def open_checkpointer(database_url: str) -> AsyncIterator[AsyncPostgresSaver]:
    """Open a connection pool, create the checkpoint tables if missing, and yield the
    checkpointer; the pool is closed on exit.

    Raises if the database cannot be reached: without checkpoints no analysis can run
    (human_review needs them), so the service should not start at all.
    """
    pool = AsyncConnectionPool(
        database_url, min_size=1, max_size=4, kwargs=CONNECTION_KWARGS, open=False
    )
    try:
        # wait=True: fail here, with a clear error, instead of on the first request
        await pool.open(wait=True, timeout=CONNECT_TIMEOUT)
        async with pool.connection() as conn:
            # setup() creates its tables in the search_path schema, so it must exist
            # (a new database may not have run the ingestion yet)
            await conn.execute("CREATE SCHEMA IF NOT EXISTS ai")
        checkpointer = AsyncPostgresSaver(pool)
        # Idempotent: creates the tables once, then runs only new migrations
        await checkpointer.setup()
        yield checkpointer
    finally:
        await pool.close()
