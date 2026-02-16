from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb

from langgraph.checkpoint.postgres import PostgresSaver

DEFAULT_ADMIN_URI = "postgres://postgres:postgres@localhost:5441/"
BENCH_ANALYZE_ENV_VAR = "BENCH_ANALYZE"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass
class BenchmarkResult:
    scenario: str
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def _read_bool_env(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid value for {name}: {raw!r}. Use one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}."
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    rank = (len(sorted_values) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def _benchmark_query(
    conn: Connection[DictRow],
    scenario: str,
    query: str,
    params: Sequence[Any],
    *,
    repeats: int,
    warmup: int,
    fetch_one: bool,
) -> BenchmarkResult:
    with conn.cursor() as cur:
        for _ in range(warmup):
            cur.execute(query, params)
            if fetch_one:
                cur.fetchone()
            else:
                cur.fetchall()

        timings_ms: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            cur.execute(query, params)
            if fetch_one:
                cur.fetchone()
            else:
                cur.fetchall()
            timings_ms.append((time.perf_counter() - start) * 1000)

    sorted_timings = sorted(timings_ms)
    return BenchmarkResult(
        scenario=scenario,
        mean_ms=statistics.fmean(timings_ms),
        median_ms=statistics.median(timings_ms),
        p95_ms=_percentile(sorted_timings, 0.95),
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
    )


def _build_main_list_query(
    saver: PostgresSaver,
    config: RunnableConfig | None,
    metadata_filter: dict[str, Any] | None,
    limit: int,
) -> tuple[str, list[Any]]:
    where, args = saver._search_where(config, metadata_filter, None)
    query = saver.SELECT_SQL + where + " ORDER BY checkpoint_id DESC LIMIT %s"
    return query, [*args, limit]


def _build_main_get_latest_query(config: RunnableConfig) -> tuple[str, tuple[str, str]]:
    thread_id = config["configurable"]["thread_id"]
    checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
    query = (
        "WHERE thread_id = %s AND checkpoint_ns = %s "
        "ORDER BY checkpoint_id DESC LIMIT 1"
    )
    return PostgresSaver.SELECT_SQL + query, (thread_id, checkpoint_ns)


def _insert_in_chunks(
    conn: Connection[DictRow],
    query: str,
    rows: Sequence[Sequence[Any]],
    chunk_size: int = 2000,
) -> None:
    with conn.cursor() as cur:
        for start in range(0, len(rows), chunk_size):
            cur.executemany(query, rows[start : start + chunk_size])


def _seed_dataset(
    conn: Connection[DictRow],
    *,
    threads: int,
    checkpoints_per_thread: int,
    channels_per_checkpoint: int,
    writes_per_checkpoint: int,
) -> tuple[str, str]:
    checkpoint_rows: list[tuple[Any, ...]] = []
    blob_rows: list[tuple[Any, ...]] = []
    write_rows: list[tuple[Any, ...]] = []

    target_thread = "thread-0000"
    target_ns = ""

    for t in range(threads):
        thread_id = f"thread-{t:04d}"
        checkpoint_ns = ""
        for i in range(checkpoints_per_thread):
            checkpoint_id = f"{i:08d}"
            parent_checkpoint_id = f"{i - 1:08d}" if i > 0 else None
            channel_versions = {
                f"ch_{c:02d}": f"{i:032d}.{c:016d}"
                for c in range(channels_per_checkpoint)
            }
            checkpoint_rows.append(
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    Jsonb(
                        {
                            "v": 4,
                            "id": checkpoint_id,
                            "ts": "2025-01-01T00:00:00+00:00",
                            "channel_values": {},
                            "channel_versions": channel_versions,
                        }
                    ),
                    Jsonb({"source": "input" if i % 2 == 0 else "loop", "step": i}),
                )
            )

            for channel, version in channel_versions.items():
                blob_rows.append(
                    (
                        thread_id,
                        checkpoint_ns,
                        channel,
                        version,
                        "json",
                        f"value-{t}-{i}-{channel}".encode(),
                    )
                )

            for w in range(writes_per_checkpoint):
                write_rows.append(
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        f"task-{t:04d}-{i:08d}-{w:02d}",
                        "",
                        w,
                        f"w_{w % 4}",
                        "json",
                        f"blob-{t}-{i}-{w}".encode(),
                    )
                )

    _insert_in_chunks(
        conn,
        """
        INSERT INTO checkpoints
            (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        checkpoint_rows,
    )
    _insert_in_chunks(
        conn,
        """
        INSERT INTO checkpoint_blobs
            (thread_id, checkpoint_ns, channel, version, type, blob)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        blob_rows,
    )
    _insert_in_chunks(
        conn,
        """
        INSERT INTO checkpoint_writes
            (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        write_rows,
    )
    conn.commit()
    return target_thread, target_ns


def _print_result(
    main_result: BenchmarkResult, optimized_result: BenchmarkResult
) -> None:
    speedup = main_result.median_ms / optimized_result.median_ms
    improvement = (speedup - 1.0) * 100
    print(f"[{main_result.scenario}]")
    print(
        "  main median: "
        f"{main_result.median_ms:.3f} ms (mean {main_result.mean_ms:.3f}, p95 {main_result.p95_ms:.3f})"
    )
    print(
        "  new  median: "
        f"{optimized_result.median_ms:.3f} ms (mean {optimized_result.mean_ms:.3f}, p95 {optimized_result.p95_ms:.3f})"
    )
    print(f"  speedup: {speedup:.2f}x ({improvement:+.1f}%)")
    print()


def run_benchmark(args: argparse.Namespace) -> int:
    database = f"benchmark_{uuid4().hex[:16]}"
    with Connection.connect(args.admin_uri, autocommit=True) as admin_conn:
        admin_conn.execute(f"CREATE DATABASE {database}")

    benchmark_db_uri = f"{args.admin_uri}{database}"
    try:
        with Connection.connect(
            benchmark_db_uri,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as setup_conn:
            PostgresSaver(setup_conn).setup()

        with Connection.connect(
            benchmark_db_uri,
            autocommit=False,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as seed_conn:
            target_thread, target_ns = _seed_dataset(
                seed_conn,
                threads=args.threads,
                checkpoints_per_thread=args.checkpoints_per_thread,
                channels_per_checkpoint=args.channels_per_checkpoint,
                writes_per_checkpoint=args.writes_per_checkpoint,
            )

        with Connection.connect(
            benchmark_db_uri,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as bench_conn:
            saver = PostgresSaver(bench_conn)
            bench_conn.execute("SET jit = off")
            if args.analyze:
                bench_conn.execute("ANALYZE")

            thread_config: RunnableConfig = {
                "configurable": {
                    "thread_id": target_thread,
                    "checkpoint_ns": target_ns,
                }
            }
            metadata_filter = {"source": "input"}

            main_list_query, main_list_params = _build_main_list_query(
                saver, thread_config, metadata_filter, args.limit
            )
            new_list_query, new_list_params = saver._build_list_query(
                thread_config, metadata_filter, None, args.limit
            )

            main_global_list_query, main_global_list_params = _build_main_list_query(
                saver, None, metadata_filter, args.limit
            )
            new_global_list_query, new_global_list_params = saver._build_list_query(
                None, metadata_filter, None, args.limit
            )

            main_get_query, main_get_params = _build_main_get_latest_query(
                thread_config
            )
            new_get_query, new_get_params = saver._build_get_tuple_query(thread_config)

            results = [
                (
                    _benchmark_query(
                        bench_conn,
                        "list_thread_filtered_main",
                        main_list_query,
                        main_list_params,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        fetch_one=False,
                    ),
                    _benchmark_query(
                        bench_conn,
                        "list_thread_filtered_new",
                        new_list_query,
                        new_list_params,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        fetch_one=False,
                    ),
                ),
                (
                    _benchmark_query(
                        bench_conn,
                        "list_metadata_only_main",
                        main_global_list_query,
                        main_global_list_params,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        fetch_one=False,
                    ),
                    _benchmark_query(
                        bench_conn,
                        "list_metadata_only_new",
                        new_global_list_query,
                        new_global_list_params,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        fetch_one=False,
                    ),
                ),
                (
                    _benchmark_query(
                        bench_conn,
                        "get_latest_main",
                        main_get_query,
                        main_get_params,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        fetch_one=True,
                    ),
                    _benchmark_query(
                        bench_conn,
                        "get_latest_new",
                        new_get_query,
                        new_get_params,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        fetch_one=True,
                    ),
                ),
            ]

            print("Benchmark dataset:")
            print(
                f"  threads={args.threads}, checkpoints/thread={args.checkpoints_per_thread}, "
                f"channels/checkpoint={args.channels_per_checkpoint}, "
                f"writes/checkpoint={args.writes_per_checkpoint}"
            )
            print(
                f"  list limit={args.limit}, warmup={args.warmup}, repeats={args.repeats}"
            )
            print(
                f"  analyze={'on' if args.analyze else 'off'} "
                f"(from {BENCH_ANALYZE_ENV_VAR} or CLI)"
            )
            print()

            has_regression = False
            for main_result, optimized_result in results:
                _print_result(main_result, optimized_result)
                if optimized_result.median_ms >= main_result.median_ms:
                    has_regression = True

            if args.require_speedup and has_regression:
                print("At least one benchmark scenario did not improve.")
                return 1
            return 0
    finally:
        with Connection.connect(args.admin_uri, autocommit=True) as admin_conn:
            admin_conn.execute(f"DROP DATABASE {database}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark checkpoint query performance. "
            "Compares main-branch SQL shape with current branch SQL."
        )
    )
    parser.add_argument("--admin-uri", default=DEFAULT_ADMIN_URI)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--checkpoints-per-thread", type=int, default=1200)
    parser.add_argument("--channels-per-checkpoint", type=int, default=4)
    parser.add_argument("--writes-per-checkpoint", type=int, default=8)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--require-speedup", action="store_true")
    try:
        analyze_default = _read_bool_env(BENCH_ANALYZE_ENV_VAR, default=True)
    except ValueError as exc:
        parser.error(str(exc))
    parser.add_argument(
        "--analyze",
        action=argparse.BooleanOptionalAction,
        default=analyze_default,
        help=(
            "Run SQL ANALYZE before benchmark queries. "
            f"Default is controlled by {BENCH_ANALYZE_ENV_VAR}."
        ),
    )
    args = parser.parse_args()
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
