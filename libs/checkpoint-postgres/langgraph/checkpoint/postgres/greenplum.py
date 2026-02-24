"""Greenplum-optimized checkpoint savers.

Greenplum (and other MPP databases with similar planners) handle
correlated subqueries by broadcasting the entire inner table to all
segments (Broadcast Motion), which is extremely slow on large datasets.
These savers replace the default correlated subquery with a CTE + JOIN
approach that uses targeted Redistribute Motion instead.

Best suited for: Greenplum and other MPP databases that cannot
decorrelate subqueries and thus choose Broadcast Motion. Not all
PostgreSQL-based distributed databases have this behavior; use the
regular `PostgresSaver` / `AsyncPostgresSaver` when in doubt.

Requirements:
    Greenplum 6+ (based on PostgreSQL 9.4) or PostgreSQL 9.3+, since
    the CTE query uses `CROSS JOIN LATERAL`.

Usage::

    from langgraph.checkpoint.postgres.greenplum import AsyncGreenplumSaver

    async with AsyncGreenplumSaver.from_conn_string(DB_URI) as saver:
        await saver.setup()
        # use as a drop-in replacement for AsyncPostgresSaver
"""

from __future__ import annotations

from typing import Any

CTE_LIST_SELECT_SQL = """
WITH filtered_checkpoints AS (
    SELECT
        thread_id,
        checkpoint,
        checkpoint_ns,
        checkpoint_id,
        parent_checkpoint_id,
        metadata
    FROM checkpoints
    {where}
    ORDER BY checkpoint_id DESC
    {limit}
),
checkpoint_data AS (
    SELECT
        fc.thread_id,
        fc.checkpoint,
        fc.checkpoint_ns,
        fc.checkpoint_id,
        fc.parent_checkpoint_id,
        fc.metadata,
        cv.key AS channel_key,
        cv.value AS channel_version
    FROM filtered_checkpoints fc
    CROSS JOIN LATERAL jsonb_each_text(fc.checkpoint -> 'channel_versions') AS cv(key, value)
),
channel_values AS (
    SELECT
        cd.thread_id,
        cd.checkpoint_id,
        cd.checkpoint_ns,
        array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob]) AS channel_values
    FROM checkpoint_data cd
    INNER JOIN checkpoint_blobs bl
        ON bl.thread_id = cd.thread_id
        AND bl.checkpoint_ns = cd.checkpoint_ns
        AND bl.channel = cd.channel_key
        AND bl.version = cd.channel_version
    GROUP BY cd.thread_id, cd.checkpoint_id, cd.checkpoint_ns
),
pending_writes AS (
    SELECT
        cw.thread_id,
        cw.checkpoint_ns,
        cw.checkpoint_id,
        array_agg(
            array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob]
            ORDER BY cw.task_id, cw.idx
        ) AS pending_writes
    FROM checkpoint_writes cw
    INNER JOIN filtered_checkpoints fc
        ON cw.thread_id = fc.thread_id
        AND cw.checkpoint_ns = fc.checkpoint_ns
        AND cw.checkpoint_id = fc.checkpoint_id
    GROUP BY cw.thread_id, cw.checkpoint_ns, cw.checkpoint_id
)
SELECT
    fc.thread_id,
    fc.checkpoint,
    fc.checkpoint_ns,
    fc.checkpoint_id,
    fc.parent_checkpoint_id,
    fc.metadata,
    cv.channel_values,
    pw.pending_writes
FROM filtered_checkpoints fc
LEFT JOIN channel_values cv
    ON fc.thread_id = cv.thread_id
    AND fc.checkpoint_id = cv.checkpoint_id
    AND fc.checkpoint_ns = cv.checkpoint_ns
LEFT JOIN pending_writes pw
    ON fc.thread_id = pw.thread_id
    AND fc.checkpoint_id = pw.checkpoint_id
    AND fc.checkpoint_ns = pw.checkpoint_ns
ORDER BY fc.checkpoint_id DESC
"""


class _GreenplumListMixin:
    """Mixin that switches list queries to the CTE-based approach."""

    def _build_list_query(
        self, where: str, args: list[Any], limit: int | None
    ) -> tuple[str, list[Any]]:
        params = list(args)
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT %s"
            params.append(int(limit))
        # Use a single space when where is empty so the CTE template produces
        # consistent SQL (no bare blank line between FROM and ORDER BY).
        normalized_where = where if where else " "
        query = CTE_LIST_SELECT_SQL.format(
            where=normalized_where, limit=limit_clause
        )
        return query, params


def _make_greenplum_classes() -> tuple[type, type]:
    """Lazily import and construct Greenplum saver classes."""
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    class AsyncGreenplumSaver(_GreenplumListMixin, AsyncPostgresSaver):
        """Async checkpoint saver optimized for Greenplum/MPP databases.

        Drop-in replacement for `AsyncPostgresSaver`. The only difference is
        that `alist()` uses a CTE-based query instead of correlated subqueries,
        avoiding Broadcast Motion in Greenplum's MPP execution engine.
        """

    class GreenplumSaver(_GreenplumListMixin, PostgresSaver):
        """Sync checkpoint saver optimized for Greenplum/MPP databases.

        Drop-in replacement for `PostgresSaver`. The only difference is
        that `list()` uses a CTE-based query instead of correlated subqueries,
        avoiding Broadcast Motion in Greenplum's MPP execution engine.
        """

    return AsyncGreenplumSaver, GreenplumSaver


def __getattr__(name: str) -> type:  # noqa: N807
    if name in ("AsyncGreenplumSaver", "GreenplumSaver"):
        async_cls, sync_cls = _make_greenplum_classes()
        globals()["AsyncGreenplumSaver"] = async_cls
        globals()["GreenplumSaver"] = sync_cls
        return async_cls if name == "AsyncGreenplumSaver" else sync_cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
