"""Benchmark versions are never analysed together (task: bench v4 separates old and new latency data). CPU only."""

from __future__ import annotations

import pytest

from cnnlab.tcc.operation import bench_versions as bv

V3, V4 = "tcc-bench/preliminary-v3", "tcc-bench/preliminary-v4"


def rows(*versions: str | None, status: str = "valid", cache_hit: bool = False) -> list[dict[str, object]]:
    return [{"bench_version": v, "status": status, "cache_hit": cache_hit} for v in versions]


def test_only_valid_original_attempts_with_a_version_count() -> None:
    assert bv.version_of_rows(rows(V3, V3, None)) == {V3}
    assert bv.version_of_rows(rows(V4, status="invalid")) == frozenset()
    assert bv.version_of_rows(rows(V4, cache_hit=True)) == frozenset()  # a cache hit repeats the source's measurement
    assert bv.version_of_rows(rows(V3) + rows(V4)) == {V3, V4}


def test_one_version_or_none_is_accepted_as_it_is() -> None:
    assert bv.select_runs({"a": frozenset({V4}), "b": frozenset({V4})}) == (["a", "b"], {})
    assert bv.select_runs({"a": frozenset(), "b": frozenset()}) == (["a", "b"], {})  # synthetic runs carry none


def test_a_mix_is_refused_with_the_runs_named() -> None:
    with pytest.raises(bv.BenchVersionError, match="different benchmark versions.*--bench-version"):
        bv.select_runs({"old": frozenset({V3}), "new": frozenset({V4})})
    with pytest.raises(bv.BenchVersionError, match="(none)"):
        bv.select_runs({"real": frozenset({V4}), "fake": frozenset()})


def test_a_run_with_two_versions_is_refused_even_when_one_is_chosen() -> None:
    with pytest.raises(bv.BenchVersionError, match="more than one benchmark version"):
        bv.select_runs({"x": frozenset({V3, V4}), "y": frozenset({V4})}, V4)


def test_choosing_a_version_keeps_its_runs_and_lists_the_others() -> None:
    kept, left_out = bv.select_runs({"old": frozenset({V3}), "new": frozenset({V4}), "new2": frozenset({V4})}, V4)
    assert kept == ["new", "new2"] and left_out == {"old": V3}
    with pytest.raises(bv.BenchVersionError, match="no run was measured"):
        bv.select_runs({"old": frozenset({V3})}, V4)
