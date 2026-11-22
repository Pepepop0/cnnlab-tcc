from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path

import pytest

from cnnlab.tcc.storage import (
    ArtifactIntegrityError,
    SQLiteRunStore,
    normalize_rel_path,
    publish_bytes,
    publish_file,
    scan_artifacts,
)
from cnnlab.tcc.storage.artifacts import attempt_artifact_dir, sha256_file
from cnnlab.tcc.storage.canonical import sha256_hex
from cnnlab.tcc.storage.testing import FakeEvaluator, make_genome, make_job, make_request, run_candidate


@pytest.mark.parametrize("path", ["engine.plan", "runs/r-1/attempts/a_1/engine.plan", "a/b-c/D.onnx"])
def test_relative_paths_accepted(path: str) -> None:
    assert normalize_rel_path(path) == path


@pytest.mark.parametrize(
    "path",
    ["", "/abs/x", "C:/x", "a\\b", "..", "a/../b", "./a", "a//b", "a/", ".hidden", "a b", "con.plan", "a/nul"],
)
def test_unsafe_paths_rejected(path: str) -> None:
    with pytest.raises(ArtifactIntegrityError):
        normalize_rel_path(path)


def test_attempt_dir_rejects_traversal_ids() -> None:
    assert attempt_artifact_dir("r1", "a1") == "runs/r1/attempts/a1"
    with pytest.raises(ArtifactIntegrityError):
        attempt_artifact_dir("..", "a1")


def test_publish_bytes_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    published = publish_bytes(tmp_path, "runs/r/engine.plan", b"engine-bytes")
    target = tmp_path / "runs" / "r" / "engine.plan"
    assert target.read_bytes() == b"engine-bytes"
    assert published.sha256 == sha256_hex(b"engine-bytes") and published.size_bytes == 12
    assert [p.name for p in target.parent.iterdir()] == ["engine.plan"]  # no temp left behind
    with pytest.raises(ArtifactIntegrityError, match="exists"):
        publish_bytes(tmp_path, "runs/r/engine.plan", b"other")
    assert target.read_bytes() == b"engine-bytes"


def test_publish_file_copies_and_hashes(tmp_path: Path) -> None:
    source = tmp_path / "scratch.onnx"
    source.write_bytes(b"x" * 3_000_000)
    published = publish_file(tmp_path / "root", "runs/r/model.onnx", source)
    assert source.exists()
    assert sha256_file(tmp_path / "root" / "runs" / "r" / "model.onnx") == (published.sha256, 3_000_000)


def test_scan_reports_missing_mismatch_orphans_and_temps(tmp_path: Path) -> None:
    ok = publish_bytes(tmp_path, "a/ok.plan", b"ok")
    bad = publish_bytes(tmp_path, "a/bad.plan", b"bad")
    publish_bytes(tmp_path, "a/orphan.plan", b"orphan")
    (tmp_path / "a" / ".x.plan.abc.tmp").write_bytes(b"partial")
    (tmp_path / "a" / "bad.plan").write_bytes(b"tampered")
    catalog = {ok.rel_path: ok.sha256, bad.rel_path: bad.sha256, "a/gone.plan": "0" * 64}
    scan = scan_artifacts(tmp_path, catalog)
    assert scan.missing == ("a/gone.plan",)
    assert scan.mismatched == ("a/bad.plan",)
    assert scan.orphans == ("a/orphan.plan",)
    assert scan.temporaries == ("a/.x.plan.abc.tmp",)
    assert not scan.ok


def test_integrity_report_detects_lost_and_corrupted_artifacts(seeded: SQLiteRunStore, artifact_root: Path) -> None:
    job = make_job("r-art")
    seeded.create_run(job)
    first = run_candidate(seeded, job, "c1", "c1.a1", make_genome(1), FakeEvaluator())
    second = run_candidate(seeded, job, "c2", "c2.a1", make_genome(2), FakeEvaluator())
    assert seeded.check_integrity().ok
    (artifact_root / first.artifact_paths["engine"]).unlink()
    (artifact_root / second.artifact_paths["engine"]).write_bytes(b"corrupted")
    report = seeded.check_integrity()
    assert report.artifacts.missing == (first.artifact_paths["engine"],)
    assert report.artifacts.mismatched == (second.artifact_paths["engine"],)
    assert not report.ok


def test_catalog_path_is_unique_case_insensitively(seeded: SQLiteRunStore, artifact_root: Path) -> None:
    job = make_job("r-case")
    seeded.create_run(job)
    first = run_candidate(seeded, job, "c1", "c1.a1", make_genome(1), FakeEvaluator())
    seeded.register_candidate(job.run_id, "c2", make_genome(2), {})
    second = FakeEvaluator().evaluate(make_request(job, "c2", "c2.a1", make_genome(2), artifact_root))
    reuse = dataclasses.replace(second, artifact_paths=first.artifact_paths, artifact_sha256=first.artifact_sha256)
    with pytest.raises(ArtifactIntegrityError, match="already cataloged"):
        seeded.commit_evaluation(reuse)
    assert seeded.get_attempt("c2.a1") is None

    seeded.commit_evaluation(second)
    # Windows file names are case-insensitive, so the catalog is too.
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        seeded._conn.execute(
            "INSERT INTO artifacts VALUES ('x', ?, 'c2.a1', 'onnx', ?, ?, 1, '2026-09-27T00:00:00Z')",
            (job.run_id, first.artifact_paths["engine"].upper(), "0" * 64),
        )
