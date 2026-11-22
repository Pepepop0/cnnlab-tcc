"""Read model on a schema v1 database and on both shapes of environment manifest.

A schema v1 database is what a run written without the operation queue looks
like (no jobs, no lease). The fixture is a *copy* reshaped to v1; its results
come from the fake evaluator and are labeled synthetic.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sqlite3
from pathlib import Path

import pytest

from cnnlab.tcc.contracts import SearchCheckpoint
from cnnlab.tcc.operation.demo import demo_status, format_evaluations, format_status
from cnnlab.tcc.storage import SQLiteRunStore, open_read_model, scan_artifacts
from cnnlab.tcc.storage.readmodel import generation_summary, system_of, version_text
from cnnlab.tcc.storage.testing import (
    SCIENCE_CODE_SHA,
    FakeEvaluator,
    make_genome,
    make_job,
    make_v1_copy,
    run_candidate,
    science_style_manifest,
)


@pytest.fixture
def v1_db(seeded: SQLiteRunStore, tmp_path: Path) -> Path:
    s = seeded
    s.register_environment("env-science-style", science_style_manifest())
    job = dataclasses.replace(make_job("run-v1", method="joint", seed_block=6), environment_id="env-science-style")
    s.create_run(job)
    evaluator = FakeEvaluator(gpu_seconds=10.0)
    ids = tuple(f"run-v1.g{i // 2}.c{i}" for i in range(4))
    for i, cid in enumerate(ids):
        run_candidate(s, job, cid, f"{cid}.a1", make_genome(i), evaluator, generation=i // 2)
    hashes = s.run_hashes(job.run_id)
    for generation, unique, archive in ((0, 2, ids[:1]), (1, 4, ids[1:3])):
        s.save_checkpoint(
            SearchCheckpoint(
                run_id=job.run_id,
                generation=generation,
                population=ids[:2],
                pareto_archive=archive,
                pending_candidates=(),
                rng_states={"synthetic": generation},
                unique_evaluations=unique,
                active_gpu_seconds=10.0 * unique,
                protocol_hash=hashes.protocol_hash,
                plan_hash=hashes.plan_hash,
                space_hash=hashes.space_hash,
            )
        )
        s.record_front(job.run_id, archive, 10.0 * unique, None)
    return make_v1_copy(s.db_path, tmp_path / "v1" / "demo.sqlite")


def test_schema_v1_database_is_read_as_it_is(v1_db: Path) -> None:
    before = hashlib.sha256(v1_db.read_bytes()).hexdigest()
    with open_read_model(v1_db) as rm:
        assert rm.schema_version() == 1 and rm.has_job_queue is False
        assert rm.jobs() == [] and rm.job_events("any-job") == []  # no queue: nothing to list, nothing fails
        [run] = rm.runs()
        assert run["run_id"] == "run-v1" and run["version_source"] == "science-manifest"
        assert run["git_sha"] == SCIENCE_CODE_SHA and run["code_dirty"] is None and run["system_version"] is None
        assert (run["wiring_label"], run["evaluator_label"], run["runner_label"]) == (
            "synthetic-wiring/v0",
            "synthetic-evaluator/v0",
            "synthetic-runner/v0",
        )
        assert version_text(run) == (
            f"git {SCIENCE_CODE_SHA[:12]} | synthetic-wiring/v0 | synthetic-evaluator/v0 | synthetic-runner/v0"
        )
        progress = rm.run_progress("run-v1")
        assert (progress["generations"], progress["checkpoints"], progress["front_snapshots"]) == (2, 2, 2)
        assert progress["attempts"] == {"valid": 4} and progress["active_gpu_s"] == pytest.approx(40.0)
        rows = rm.evaluations("run-v1")
        assert [r["generation"] for r in rows] == [0, 0, 1, 1] and all(r["synthetic"] for r in rows)
        assert not any(r["preliminary"] or r["not_confirmatory"] for r in rows)  # only what the evaluator declared
        checkpoints = rm.checkpoints("run-v1")
        assert [
            (c["generation"], c["unique_evaluations"], c["population"], c["pareto_archive"], c["pending"])
            for c in checkpoints
        ] == [(0, 2, 2, 1, 0), (1, 4, 2, 2, 0)]
        fronts = rm.fronts("run-v1")
        assert [len(f["candidate_ids"]) for f in fronts] == [1, 2] and fronts[0]["hypervolume"] is None
        with pytest.raises(sqlite3.OperationalError):
            rm._conn.execute("CREATE TABLE jobs (job_id TEXT)")  # reading never "repairs" the schema
    assert hashlib.sha256(v1_db.read_bytes()).hexdigest() == before
    with open_read_model(v1_db) as rm:
        assert rm.schema_version() == 1  # still v1: nothing migrated it


def test_artifact_catalog_is_checked_against_disk_without_a_writer(v1_db: Path, artifact_root: Path) -> None:
    with open_read_model(v1_db) as rm:
        catalog = rm.artifacts("run-v1")
        assert rm.artifacts() == catalog and rm.artifacts("no-such-run") == []
    assert [a["kind"] for a in catalog] == ["engine"] * 4
    assert all(a["rel_path"].startswith("runs/run-v1/attempts/") and a["size_bytes"] > 0 for a in catalog)
    by_path = {a["rel_path"]: a["sha256"] for a in catalog}
    assert scan_artifacts(artifact_root, by_path).ok

    victim = artifact_root / catalog[0]["rel_path"]
    victim.write_bytes(b"changed after the commit")
    (artifact_root / catalog[1]["rel_path"]).unlink()
    scan = scan_artifacts(artifact_root, by_path)
    assert scan.mismatched == (catalog[0]["rel_path"],) and scan.missing == (catalog[1]["rel_path"],)
    with open_read_model(v1_db) as rm:
        assert rm.schema_version() == 1 and rm.artifacts("run-v1") == catalog  # detected, nothing repaired


def test_cli_status_lists_runs_that_have_no_job(v1_db: Path) -> None:
    status = demo_status(v1_db)
    assert status["schema_version"] == 1 and status["has_job_queue"] is False and status["jobs"] == []
    [loose] = status["runs_without_job"]
    assert loose["run"]["run_id"] == "run-v1" and loose["progress"]["attempts"] == {"valid": 4}
    text = format_status(status)
    assert "no job queue in this database (schema v1)" in text
    assert "- (no job) run=run-v1 confirmatory/joint/b6" in text
    assert f"system git {SCIENCE_CODE_SHA[:12]} | synthetic-wiring/v0" in text
    assert "generations=2  checkpoints=2  fronts=2" in text and "SYNTHETIC: 4 results are fake" in text
    with open_read_model(v1_db) as rm:
        table = format_evaluations(rm.evaluations("run-v1"))
    lines = table.splitlines()
    assert lines[0].split()[:3] == ["attempt", "gen", "status"]
    assert lines[1].split()[:3] == ["run-v1.g0.c0.a1", "0", "valid"] and lines[-1].split()[1] == "1"
    assert all(line.rstrip().endswith("synthetic") for line in lines[1:])


def test_generation_summary_keeps_minima_independent_and_unmeasured_empty() -> None:
    def row(generation: int | None, cid: str, status: str, e: float | None, latency: float | None) -> dict[str, object]:
        measured = e is not None
        return {
            "generation": generation,
            "candidate_id": cid,
            "status": status,
            "cache_hit": False,
            "gpu_seconds": 2.5,
            "e": e,
            "l_ms": latency,
            "v_mb": 3.0 if measured else None,
            "s_mb": 4.0 if measured else None,
        }

    summary = generation_summary(
        [
            row(0, "a", "valid", 0.9, 0.2),
            row(0, "b", "valid", 0.8, 0.5),
            row(0, "c", "invalid", None, None),
            row(1, "d", "interrupted", None, None),
            row(None, "e", "valid", 0.5, 0.1),
        ]
    )
    assert [g["generation"] for g in summary] == [0, 1, None]
    first = summary[0]
    assert (first["candidates"], first["valid"], first["invalid"], first["gpu_seconds"]) == (3, 2, 1, 7.5)
    assert (first["min_e"], first["min_l_ms"]) == (0.8, 0.2)  # from two different candidates
    assert summary[1]["valid"] == 0 and summary[1]["min_e"] is None and summary[1]["interrupted"] == 1
    assert generation_summary([]) == []


def test_system_of_reads_both_manifest_shapes_and_invents_nothing() -> None:
    operation = system_of(
        {
            "system": {
                "readable": "cnnlab-tcc 0.1.0 | git abc",
                "code_version": {"label": "demo", "git_sha": "a" * 40, "dirty": True},
            },
            "runner": "r/1",
            "evaluator": "e/1",
        }
    )
    assert operation["version_source"] == "operation-manifest"
    assert operation["system_version"] == "cnnlab-tcc 0.1.0 | git abc"
    assert (operation["git_sha"], operation["code_dirty"], operation["code_label"]) == ("a" * 40, True, "demo")
    assert (operation["runner_label"], operation["evaluator_label"], operation["wiring_label"]) == ("r/1", "e/1", None)

    science = system_of(science_style_manifest())
    assert science["git_sha"] == SCIENCE_CODE_SHA
    assert science["system_version"] is None and science["code_dirty"] is None  # not recorded, so not shown

    dirty = system_of({"science": {"code_version": "b" * 40 + "+dirty", "wiring": "w/1"}})
    assert (dirty["git_sha"], dirty["code_dirty"], dirty["code_label"]) == ("b" * 40, True, None)
    assert version_text(dirty) == f"git {'b' * 12}+dirty | w/1"

    odd = system_of({"science": {"code_version": "unknown (not a git checkout)", "evaluator": "plain text"}})
    assert odd["git_sha"] is None and odd["code_label"] == "unknown (not a git checkout)"
    assert odd["evaluator_label"] is None
    assert version_text(odd) == "git (not recorded) | unknown (not a git checkout)"

    empty = system_of({"gpu": "whatever"})
    assert empty["version_source"] is None and set(empty.values()) == {None}
    assert version_text(empty, missing="?") == "git ?"
    assert version_text({**operation, "system_version": None}) == f"git {'a' * 12}+dirty | demo | e/1 | r/1"
