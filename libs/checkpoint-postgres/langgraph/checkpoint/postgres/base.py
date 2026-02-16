from __future__ import annotations

import random
import re
import warnings
from collections.abc import Sequence
from importlib.metadata import version as get_version
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS
from psycopg.types.json import Jsonb

MetadataInput = dict[str, Any] | None

try:
    major, minor = get_version("langgraph").split(".")[:2]
    if int(major) == 0 and int(minor) < 5:
        warnings.warn(
            "You're using incompatible versions of langgraph and checkpoint-postgres. Please upgrade langgraph to avoid unexpected behavior.",
            DeprecationWarning,
            stacklevel=2,
        )
except Exception:
    # skip version check if running from source
    pass

"""
To add a new migration, add a new string to the MIGRATIONS list.
The position of the migration in the list is the version number.
"""
MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
);""",
    """CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);""",
    """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);""",
    """CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);""",
    "ALTER TABLE checkpoint_blobs ALTER COLUMN blob DROP not null;",
    # NOTE: this is a no-op migration to ensure that the versions in the migrations table are correct.
    # This is necessary due to an empty migration previously added to the list.
    "SELECT 1;",
    """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id);
    """,
    """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);
    """,
    """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);
    """,
    """ALTER TABLE checkpoint_writes ADD COLUMN IF NOT EXISTS task_path TEXT NOT NULL DEFAULT '';""",
]

SELECT_SQL = """
select
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata,
    (
        select array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob])
        from jsonb_each_text(checkpoint -> 'channel_versions')
        inner join checkpoint_blobs bl
            on bl.thread_id = checkpoints.thread_id
            and bl.checkpoint_ns = checkpoints.checkpoint_ns
            and bl.channel = jsonb_each_text.key
            and bl.version = jsonb_each_text.value
    ) as channel_values,
    (
        select
        array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob] order by cw.task_id, cw.idx)
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns = checkpoints.checkpoint_ns
            and cw.checkpoint_id = checkpoints.checkpoint_id
    ) as pending_writes
from checkpoints """

# Optimized list query using two-phase CTEs.
# Phase 1 narrows/limits checkpoints first, phase 2 computes channel_values and
# pending_writes only for that reduced checkpoint set.
SELECT_LIST_CTE_SQL = """
WITH base AS (
  SELECT
    c.thread_id,
    c.checkpoint,
    c.checkpoint_ns,
    c.checkpoint_id,
    c.parent_checkpoint_id,
    c.metadata
  FROM checkpoints c
  {where_base}
  ORDER BY c.checkpoint_id DESC
  {limit_clause}
)
SELECT
  b.thread_id,
  b.checkpoint,
  b.checkpoint_ns,
  b.checkpoint_id,
  b.parent_checkpoint_id,
  b.metadata,
  cv.channel_values,
  pw.pending_writes
FROM base b
LEFT JOIN LATERAL (
  SELECT
    array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob]) AS channel_values
  FROM jsonb_each_text(
    COALESCE(b.checkpoint -> 'channel_versions', '{{}}'::jsonb)
  ) AS kv
  INNER JOIN checkpoint_blobs bl
  ON bl.thread_id = b.thread_id
    AND bl.checkpoint_ns = b.checkpoint_ns
    AND bl.channel = kv.key
    AND bl.version = kv.value
) cv ON true
LEFT JOIN LATERAL (
  SELECT
    array_agg(
      array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob]
      ORDER BY cw.task_id, cw.idx
    ) AS pending_writes
  FROM checkpoint_writes cw
  WHERE cw.thread_id = b.thread_id
    AND cw.checkpoint_ns = b.checkpoint_ns
    AND cw.checkpoint_id = b.checkpoint_id
) pw ON true
ORDER BY b.checkpoint_id DESC
"""

SELECT_PENDING_SENDS_SQL = f"""
select
    checkpoint_id,
    array_agg(array[type::bytea, blob] order by task_path, task_id, idx) as sends
from checkpoint_writes
where thread_id = %s
    and checkpoint_id = any(%s)
    and channel = '{TASKS}'
group by checkpoint_id
"""

UPSERT_CHECKPOINT_BLOBS_SQL = """
    INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING
"""

UPSERT_CHECKPOINTS_SQL = """
    INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
    DO UPDATE SET
        checkpoint = EXCLUDED.checkpoint,
        metadata = EXCLUDED.metadata;
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
        channel = EXCLUDED.channel,
        type = EXCLUDED.type,
        blob = EXCLUDED.blob;
"""

INSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING
"""


def _qualify_where(where: str, alias: str = "c") -> str:
    """Prefix column names in a WHERE clause with table alias for use in CTEs.

    Returns 'WHERE TRUE' when where is empty.
    """
    if not where or not where.strip():
        return "WHERE TRUE"
    for col in ("thread_id", "checkpoint_ns", "checkpoint_id", "metadata"):
        where = re.sub(rf"\b{re.escape(col)}\b", f"{alias}.{col}", where)
    return where


class BasePostgresSaver(BaseCheckpointSaver[str]):
    SELECT_SQL = SELECT_SQL
    SELECT_LIST_CTE_SQL = SELECT_LIST_CTE_SQL
    SELECT_PENDING_SENDS_SQL = SELECT_PENDING_SENDS_SQL
    LIST_CTE_LIMIT_THRESHOLD = 5
    # Thread-scoped + metadata filters are typically cheap with legacy SQL;
    # keep CTE for very large pages only.
    THREAD_FILTERED_LIST_CTE_LIMIT_THRESHOLD = 5000
    # Broad metadata-only scans can benefit from CTE at larger page sizes.
    METADATA_ONLY_LIST_CTE_LIMIT_THRESHOLD = 5000
    NARROW_LIST_CTE_LIMIT_THRESHOLD = 500
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL

    supports_pipeline: bool

    def _migrate_pending_sends(
        self,
        pending_sends: list[tuple[bytes, bytes]],
        checkpoint: dict[str, Any],
        channel_values: list[tuple[bytes, bytes, bytes]],
    ) -> None:
        if not pending_sends:
            return
        # add to values
        enc, blob = self.serde.dumps_typed(
            [self.serde.loads_typed((c.decode(), b)) for c, b in pending_sends],
        )
        channel_values.append((TASKS.encode(), enc.encode(), blob))
        # add to versions
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None, None)
        )

    def _load_blobs(
        self, blob_values: list[tuple[bytes, bytes, bytes]]
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k.decode(): self.serde.loads_typed((t.decode(), v))
            for k, t, v in blob_values
            if t.decode() != "empty"
        }

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, bytes | None]]:
        if not versions:
            return []

        return [
            (
                thread_id,
                checkpoint_ns,
                k,
                cast(str, ver),
                *(
                    self.serde.dumps_typed(values[k])
                    if k in values
                    else ("empty", None)
                ),
            )
            for k, ver in versions.items()
        ]

    def _load_writes(
        self, writes: list[tuple[bytes, bytes, bytes, bytes]]
    ) -> list[tuple[str, str, Any]]:
        return (
            [
                (
                    tid.decode(),
                    channel.decode(),
                    self.serde.loads_typed((t.decode(), v)),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, str, int, str, str, bytes]]:
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _search_where(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None = None,
    ) -> tuple[str, list[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, before.

        This method returns a tuple of a string and a tuple of values. The string
        is the parametered WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = $1 AND column2 IS $2". The list of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = []

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = %s ")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = %s")
                param_values.append(checkpoint_ns)

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = %s ")
                param_values.append(checkpoint_id)

        # construct predicate for metadata filter
        if filter:
            wheres.append("metadata @> %s ")
            param_values.append(Jsonb(filter))

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < %s ")
            param_values.append(get_checkpoint_id(before))

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )

    def _build_list_query(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None,
        limit: int | None,
    ) -> tuple[str, list[Any]]:
        """Build a list query and params using legacy or CTE strategy."""
        if not self._use_cte_for_list(config, filter, before, limit):
            where, args = self._search_where(config, filter, before)
            query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
            params = list(args)
            if limit is not None:
                query += " LIMIT %s"
                params.append(int(limit))
            return query, params

        where, args = self._search_where(config, filter, before)
        where_base = _qualify_where(where, alias="c")
        limit_clause = "LIMIT %s" if limit is not None else ""
        query = self.SELECT_LIST_CTE_SQL.format(
            where_base=where_base,
            limit_clause=limit_clause,
        )
        params = list(args)
        if limit is not None:
            params.append(int(limit))
        return query, params

    def _use_cte_for_list(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None,
        limit: int | None,
    ) -> bool:
        """Choose CTE strategy for broader list queries.

        Heuristic:
        - Use legacy query for very small pages (limit <= LIST_CTE_LIMIT_THRESHOLD).
        - For thread+namespace scoped filters, use a higher limit threshold.
        - For metadata-only wide scans, use CTE only above a larger threshold.
        - Use CTE query for larger pages (limit > LIST_CTE_LIMIT_THRESHOLD).
        - For unbounded scans (limit is None), use CTE only when filters are broad.
        """
        if self._is_narrow_list_filter(config, filter, before):
            if limit is None:
                return False
            return limit > self.NARROW_LIST_CTE_LIMIT_THRESHOLD

        if self._is_thread_ns_list_filter(config):
            if before is not None:
                return False
            if limit is None:
                return False
            return limit > self.THREAD_FILTERED_LIST_CTE_LIMIT_THRESHOLD

        if self._is_metadata_only_wide_filter(config, filter, before):
            if limit is None:
                return True
            return limit > self.METADATA_ONLY_LIST_CTE_LIMIT_THRESHOLD

        if limit is not None:
            return limit > self.LIST_CTE_LIMIT_THRESHOLD
        return self._is_wide_list_filter(config, filter, before)

    def _is_narrow_list_filter(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None,
    ) -> bool:
        """Return whether list filter is constrained to a single thread+namespace."""
        if config is None or filter or before is not None:
            return False
        configurable = config.get("configurable", {})
        if configurable.get("thread_id") is None:
            return False
        if configurable.get("checkpoint_ns") is None:
            return False
        if get_checkpoint_id(config):
            return False
        return True

    def _is_thread_ns_list_filter(self, config: RunnableConfig | None) -> bool:
        """Return whether list filter is constrained to a thread+namespace."""
        if config is None:
            return False
        configurable = config.get("configurable", {})
        if configurable.get("thread_id") is None:
            return False
        if configurable.get("checkpoint_ns") is None:
            return False
        if get_checkpoint_id(config):
            return False
        return True

    def _is_metadata_only_wide_filter(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None,
    ) -> bool:
        """Return whether list filter is metadata-only across all threads."""
        return config is None and bool(filter) and before is None

    def _is_wide_list_filter(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None,
    ) -> bool:
        """Return whether list filters are broad enough to favor CTE reads."""
        if config is None:
            return True

        configurable = config.get("configurable", {})
        if configurable.get("thread_id") is None:
            return True
        if configurable.get("checkpoint_ns") is None:
            return True
        if get_checkpoint_id(config):
            return False
        if filter:
            return False
        if before is not None:
            return False
        return False

    def _build_get_tuple_query(
        self, config: RunnableConfig
    ) -> tuple[str, tuple[Any, ...]]:
        """Build the lightweight get_tuple query and params."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"

        return self.SELECT_SQL + where, args
