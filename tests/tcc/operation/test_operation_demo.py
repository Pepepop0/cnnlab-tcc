"""Pilot demo bootstrap + CLI with the synthetic wiring (CPU only, fake evaluator).

The fitness values here are synthetic stand-ins that exist only in tests
(`FakeEvaluator` derives E/L/V/S from a hash and labels them
`quality.synthetic = true`). The real run injects the science runner and
evaluator through the same factory interface.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import cnnlab
from cnnlab.tcc.operation.demo import (
    EXIT_REFUSED,
    CodeVersionMismatchError,
    DemoRunSpec,
    SyntheticWiringError,
    demo_status,
    format_status,
    main,
    prepare_demo,
    resume_job,
    start_demo,
)
from cnnlab.tcc.operation.provenance import CodeVersion, default_repo_root
from cnnlab.tcc.operation.testing import synthetic_demo_wiring
from cnnlab.tcc.storage import JobQueue, ReadModel, SQLiteRunStore

FACTORY = "cnnlab.tcc.operation.testing:synthetic_demo_wiring"
CODE_A = CodeVersion("demo-test-a", "a" * 40, False, "1" * 64, (), (), 0)
CODE_B = CodeVersion("demo-test-b", "b" * 40, True, "2" * 64, ("src/x.py",), (), 0)


def _spec(**kw: Any) -> DemoRunSpec:
    base: dict[str, Any] = {
        "method": "joint",
        "seed_block": 0,
        "search_seed": 11,
        "train_seed": 22,
        "calibration_seed": 33,
        "budget_gpu_s": 3600.0,
        "code_label": "demo-test",
        "author": "test-suite",
        "factory": FACTORY,
        "factory_args": {"n_candidates": 4, "sleep_s": 0.0},
    }
    base.update(kw)
    return DemoRunSpec.from_mapping(base)


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "db" / "tcc.sqlite3", tmp_path / "artifacts"


def test_prepare_bootstraps_a_versioned_pilot_run_idempotently(paths: tuple[Path, Path]) -> None:
    db, art = paths
    spec = _spec()
    with SQLiteRunStore(db, art) as store:
        first = prepare_demo(store, spec, synthetic_demo_wiring(spec), CODE_A, allow_synthetic=True)
        again = prepare_demo(store, spec, synthetic_demo_wiring(spec), CODE_A, allow_synthetic=True)
        assert again.job_spec == first.job_spec and again.job.job_id == first.job.job_id
        assert first.job_spec.stage == "pilot" and first.job_spec.budget_gpu_s == 3600.0
        plan = store.get_plan(first.plan_id)
        assert plan.status == "frozen" and plan.content["confirmatory"] is False
        assert plan.content["purpose"] == "exploratory-demo"
        protocol_doc = (default_repo_root() / "docs" / "TCC-01-PROTOCOLO.md").read_bytes()
        assert first.protocol_id.startswith("tcc01-")
        assert (
            store._conn.execute(
                "SELECT document_sha256 FROM protocol_versions WHERE protocol_id = ?", (first.protocol_id,)
            ).fetchone()[0]
            == hashlib.sha256(protocol_doc).hexdigest()
        )
        assert "test" not in store.dataset_splits_for_run(first.job_spec.run_id)  # pilot never sees test

        other = prepare_demo(
            store, spec, synthetic_demo_wiring(spec), CODE_B, allow_synthetic=True
        )  # other code -> other run
        assert other.job_spec.run_id != first.job_spec.run_id
    with ReadModel(db) as rm:
        runs = {r["run_id"]: r for r in rm.runs()}
    a = runs[first.job_spec.run_id]
    assert a["git_sha"] == "a" * 40 and a["code_dirty"] is False and a["code_label"] == "demo-test-a"
    assert "cnnlab-tcc 0.1.0" in a["system_version"] and "pre-release" in a["system_version"]
    assert "1.0" not in a["system_version"].replace("0.1.0", "")
    assert runs[other.job_spec.run_id]["system_version"].count("+dirty") == 1


def test_start_runs_the_injected_pipeline_and_records_four_objectives(paths: tuple[Path, Path]) -> None:
    db, art = paths
    events: list[Any] = []
    result = start_demo(
        db,
        art,
        _spec(),
        synthetic_demo_wiring,
        worker_id="w1",
        code_version=CODE_A,
        on_event=events.append,
        allow_synthetic=True,
    )
    assert result.outcome is not None and result.outcome.end_state == "done"
    assert [e.kind for e in events] == ["job_started", "job_ended"]
    with ReadModel(db) as rm:
        rows = rm.evaluations(result.run_id)
        claimed = [e for e in rm.job_events(result.job_id) if e["kind"] == "claimed"]
    assert [r["attempt_id"] for r in rows] == ["c0.a1", "c1.a1", "c2.a1", "c3.a1"]
    for r in rows:
        assert r["status"] == "valid" and r["synthetic"] is True  # fake values, labeled as such
        values = (r["e"], r["l_ms"], r["v_mb"], r["s_mb"])
        assert all(isinstance(v, float) and math.isfinite(v) for v in values)
        assert 0 <= r["e"] <= 1 and r["l_ms"] > 0 and r["v_mb"] >= 0 and r["s_mb"] > 0
    assert claimed[0]["detail"]["code_version"]["git_sha"] == "a" * 40

    again = start_demo(
        db, art, _spec(), synthetic_demo_wiring, worker_id="w1", code_version=CODE_A, allow_synthetic=True
    )
    assert again.outcome is None and "already done" in again.message
    text = format_status(demo_status(db))
    assert "SYNTHETIC evaluator" in text and "[done]" in text and "pre-release" in text


def test_resume_with_other_code_is_refused_unless_allowed_and_then_recorded(paths: tuple[Path, Path]) -> None:
    db, art = paths
    spec = _spec()
    with SQLiteRunStore(db, art) as store:
        prepared = prepare_demo(store, spec, synthetic_demo_wiring(spec), CODE_A, allow_synthetic=True)
    with pytest.raises(CodeVersionMismatchError):
        resume_job(db, art, prepared.job.job_id, worker_id="w2", code_version=CODE_B, allow_synthetic=True)
    with SQLiteRunStore(db, art) as store:
        assert JobQueue(store).get(prepared.job.job_id).claim_count == 0
    result = resume_job(
        db, art, prepared.job.job_id, worker_id="w2", code_version=CODE_B, allow_code_change=True, allow_synthetic=True
    )
    assert result.outcome.end_state == "done" and result.run_id == prepared.job_spec.run_id
    with ReadModel(db) as rm:
        [claim] = [e for e in rm.job_events(prepared.job.job_id) if e["kind"] == "claimed"]
    assert claim["detail"]["code_changed_from"]["git_sha"] == "a" * 40
    assert claim["detail"]["code_version"]["git_sha"] == "b" * 40


def _cli(args: list[str], timeout: float = 120) -> subprocess.CompletedProcess[str]:
    src = str(Path(cnnlab.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", ""), "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(
        [sys.executable, "-m", "cnnlab.tcc.operation.demo", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_crash_then_resume_by_job_id_without_duplicates(paths: tuple[Path, Path], tmp_path: Path) -> None:
    db, art = paths
    spec_file = tmp_path / "spec.json"
    spec = _spec(factory_args={"n_candidates": 4, "sleep_s": 0.05, "crash_on_attempt": "c2.a1", "crash_after_s": 0.3})
    spec_file.write_text(json.dumps(spec.to_payload()), encoding="utf-8")
    base = ["--db", str(db), "--artifacts", str(art)]

    crashed = _cli(
        [
            "start",
            *base,
            "--spec",
            str(spec_file),
            "--worker-id",
            "cli-1",
            "--lease-s",
            "1",
            "--heartbeat-s",
            "0.1",
            "--allow-synthetic",
        ]
    )
    assert crashed.returncode == 9, crashed.stderr  # killed inside c2.a1

    status = json.loads(_cli(["status", "--db", str(db), "--json"]).stdout)
    [job] = status["jobs"]
    assert job["state"] == "running" and job["progress"]["attempts"] == {"valid": 2, "running": 1}

    time.sleep(1.3)  # let the dead worker's 1 s lease expire
    # The stored spec still says "crash on c2.a1"; the retry is c2.a2, so it completes.
    refused = _cli(["start", *base, "--job-id", job["job_id"], "--worker-id", "cli-2"])  # no opt-in: no claim
    assert refused.returncode == 5 and "SyntheticWiringError" in refused.stderr
    resumed = _cli(
        [
            "start",
            *base,
            "--job-id",
            job["job_id"],
            "--worker-id",
            "cli-2",
            "--lease-s",
            "5",
            "--heartbeat-s",
            "0.5",
            "--allow-synthetic",
        ]
    )
    assert resumed.returncode == 0, resumed.stderr
    assert "ended done" in resumed.stdout

    with ReadModel(db) as rm:
        rows = rm.evaluations(job["run_id"])
        kinds = [e["kind"] for e in rm.job_events(job["job_id"])]
        progress = rm.run_progress(job["run_id"])
    valid = sorted(r["candidate_id"] for r in rows if r["status"] == "valid")
    assert valid == ["c0", "c1", "c2", "c3"]  # each individual confirmed once
    crashed_row = next(r for r in rows if r["attempt_id"] == "c2.a1")
    assert crashed_row["status"] == "interrupted" and crashed_row["failure_code"] == "lease_expired"
    assert crashed_row["gpu_seconds"] > 0  # time spent before the crash still counts
    assert kinds == ["enqueued", "claimed", "lease_expired", "claimed", "done"]
    assert progress["unique_valid_evaluations"] == 4

    table = _cli(["status", "--db", str(db), "--run", job["run_id"], "--evaluations"])
    assert table.returncode == 0 and "c2.a1" in table.stdout and "lease_expired" in table.stdout

    backup = _cli(["backup", *base, "--dest", str(tmp_path / "bk"), "--label", "demo-test"])
    assert backup.returncode == 0 and "complete=True" in backup.stdout


def test_cli_cancel_and_argument_errors(paths: tuple[Path, Path], tmp_path: Path) -> None:
    db, art = paths
    spec = _spec()
    with SQLiteRunStore(db, art) as store:
        prepared = prepare_demo(store, spec, synthetic_demo_wiring(spec), CODE_A, allow_synthetic=True)
    assert (
        main(["cancel", "--db", str(db), "--artifacts", str(art), "--job-id", prepared.job.job_id, "--reason", "demo"])
        == 0
    )
    with ReadModel(db) as rm:
        assert rm.jobs()[0]["state"] == "cancelled"
    with pytest.raises(SystemExit):
        main(["status", "--db", str(db), "--evaluations"])  # needs --run
    with pytest.raises(ValueError):
        _spec(code_label=" ")  # no version is invented


# -- synthetic wiring is never the default ----------------------------------------


def test_synthetic_wiring_is_refused_by_default_and_writes_nothing(paths: tuple[Path, Path], tmp_path: Path) -> None:
    db, art = paths
    spec = _spec()
    with pytest.raises(SyntheticWiringError):
        start_demo(db, art, spec, synthetic_demo_wiring, worker_id="w", code_version=CODE_A)
    assert not db.exists() and not art.exists()  # refused before the database is created

    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec.to_payload()), encoding="utf-8")
    base = ["--db", str(db), "--artifacts", str(art), "--spec", str(spec_file)]
    for command in ("prepare", "start"):
        refused = _cli([command, *base])
        assert refused.returncode == EXIT_REFUSED, refused.stderr
        assert refused.stderr.strip().startswith("error: SyntheticWiringError")
        assert "Traceback" not in refused.stderr
        assert not db.exists()

    with SQLiteRunStore(db, art) as store:
        with pytest.raises(SyntheticWiringError):
            prepare_demo(store, spec, synthetic_demo_wiring(spec), CODE_A)
        for table in ("runs", "jobs", "environments", "experiment_plans", "datasets"):
            assert store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_a_job_never_mixes_real_and_synthetic_wiring(paths: tuple[Path, Path]) -> None:
    db, art = paths
    spec = _spec()
    with SQLiteRunStore(db, art) as store:
        prepared = prepare_demo(store, spec, synthetic_demo_wiring(spec), CODE_A, allow_synthetic=True)

    def claims_real(s: DemoRunSpec) -> object:  # a wiring that says "real" for a job prepared as synthetic
        return dataclasses.replace(synthetic_demo_wiring(s), synthetic=False)

    def other_labels(s: DemoRunSpec) -> object:
        return dataclasses.replace(synthetic_demo_wiring(s), evaluator_label="another-evaluator")

    job_id = prepared.job.job_id
    with pytest.raises(SyntheticWiringError, match="never mixed"):
        resume_job(db, art, job_id, worker_id="w", code_version=CODE_A, factory=claims_real, allow_code_change=True)
    with pytest.raises(CodeVersionMismatchError, match="wiring"):
        resume_job(db, art, job_id, worker_id="w", code_version=CODE_A, factory=other_labels, allow_synthetic=True)
    with SQLiteRunStore(db, art) as store:
        assert JobQueue(store).get(job_id).claim_count == 0  # nothing was claimed by the refused sessions

    result = resume_job(
        db,
        art,
        job_id,
        worker_id="w",
        code_version=CODE_A,
        factory=other_labels,
        allow_synthetic=True,
        allow_code_change=True,
    )
    assert result.outcome.end_state == "done"
    with ReadModel(db) as rm:
        [claim] = [e for e in rm.job_events(job_id) if e["kind"] == "claimed"]
    assert claim["detail"]["wiring_changed_from"]["evaluator"] == "sleeping-fake-evaluator"
    assert claim["detail"]["wiring"]["evaluator"] == "another-evaluator"


def test_cli_reports_errors_in_one_clean_line(paths: tuple[Path, Path], tmp_path: Path) -> None:
    db, _art = paths
    missing = _cli(["status", "--db", str(db)])
    assert missing.returncode == EXIT_REFUSED and missing.stderr.strip().startswith("error: ReadModelError")
    assert "Traceback" not in missing.stderr and not db.exists()

    bad_spec = tmp_path / "bad.json"
    bad_spec.write_text(json.dumps({**_spec().to_payload(), "factory": "no_such_module_xyz:factory"}), encoding="utf-8")
    bad = _cli(["prepare", "--db", str(db), "--artifacts", str(tmp_path / "a"), "--spec", str(bad_spec)])
    assert bad.returncode == EXIT_REFUSED and "ModuleNotFoundError" in bad.stderr and not db.exists()


def test_spec_file_with_utf8_bom_loads_like_one_without(tmp_path: Path) -> None:
    """Windows PowerShell 5.1 writes a BOM with Set-Content -Encoding utf8; the spec must still load."""
    payload = json.dumps(_spec().to_payload())
    plain, with_bom = tmp_path / "plain.json", tmp_path / "bom.json"
    plain.write_text(payload, encoding="utf-8")
    with_bom.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))
    assert DemoRunSpec.load(with_bom) == DemoRunSpec.load(plain)
