"""Measurements of different benchmark versions are never analysed together.

Every attempt carries the version of the benchmark that measured it (``quality.measurement.bench_version``, exposed by
the read model as ``bench_version``). When the bench changes how L is taken (v4 replays a CUDA Graph), the latencies of
the old and the new version are not comparable, so an analysis over several runs refuses a mix instead of averaging it.

The rule is per run: a run measured by two versions is refused outright (a campaign run is on frozen code, so this
means the database was written by two codes), and the runs of an analysis must all share one version, or the caller
names the version to keep (``bench_version=``): the other runs are then left out and listed. Attempts without a version
(synthetic ones, failures before the benchmark) do not count; a run with no versioned attempt has the version ``None``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cnnlab.tcc.storage.errors import StorageError


class BenchVersionError(StorageError):
    """The runs mix benchmark versions and the caller did not choose one."""


def version_of_rows(rows: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """The distinct benchmark versions of the valid original attempts among ``rows`` (read-model shape)."""
    return frozenset(
        str(r["bench_version"])
        for r in rows
        if r.get("bench_version") and r.get("status", "valid") == "valid" and not r.get("cache_hit", False)
    )


def label(version: str | None) -> str:
    return "(none)" if version is None else version


def select_runs(
    versions_by_run: Mapping[str, frozenset[str]], bench_version: str | None = None
) -> tuple[list[str], dict[str, str | None]]:
    """(run ids to analyse in the given order, ``{run_id: version}`` of the runs left out).

    Raises :class:`BenchVersionError` for a run with two versions, and for several versions without a choice.
    """
    single: dict[str, str | None] = {}
    for run_id, versions in versions_by_run.items():
        if len(versions) > 1:
            raise BenchVersionError(
                f"run {run_id!r} was measured by more than one benchmark version ({sorted(versions)}); "
                "its latencies are not comparable with each other"
            )
        single[run_id] = next(iter(versions), None)
    distinct = set(single.values())
    if bench_version is None:
        if len(distinct) > 1:
            by_version: dict[str | None, list[str]] = {}
            for run_id, version in single.items():
                by_version.setdefault(version, []).append(run_id)
            table = "; ".join(
                f"{label(v)}: {len(ids)} run(s), e.g. {ids[0]}"
                for v, ids in sorted(by_version.items(), key=lambda kv: label(kv[0]))
            )
            raise BenchVersionError(
                f"the runs were measured by different benchmark versions ({table}); latencies of different versions "
                "are not mixed. Choose one with --bench-version"
            )
        return list(single), {}
    kept = [r for r, v in single.items() if v == bench_version]
    if not kept:
        raise BenchVersionError(
            f"no run was measured by benchmark version {bench_version!r} (found: {sorted(label(v) for v in distinct)})"
        )
    return kept, {r: v for r, v in single.items() if v != bench_version}


__all__ = ["BenchVersionError", "version_of_rows", "select_runs", "label"]
