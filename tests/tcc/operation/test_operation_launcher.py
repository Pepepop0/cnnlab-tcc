"""Launcher used by the TUI: child processes for the CLI, detached worker (CPU, synthetic wiring).

The synthetic wiring is always opted into explicitly (`allow_synthetic=True`);
the default path is checked to refuse it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.operation import launcher
from cnnlab.tcc.operation.demo import EXIT_REFUSED, DemoRunSpec
from cnnlab.tcc.storage import ReadModel, ReadModelError

FACTORY = "cnnlab.tcc.operation.testing:synthetic_demo_wiring"
NO_GPU = {"CUDA_VISIBLE_DEVICES": ""}


def _write_spec(path: Path, **factory_args: Any) -> Path:
    spec = DemoRunSpec.from_mapping(
        {
            "method": "joint",
            "seed_block": 0,
            "search_seed": 11,
            "train_seed": 22,
            "calibration_seed": 33,
            "budget_gpu_s": 3600.0,
            "code_label": "launcher-test",
            "author": "test-suite",
            "factory": FACTORY,
            "factory_args": {"n_candidates": 3, "sleep_s": 0.05, **factory_args},
        }
    )
    path.write_text(json.dumps(spec.to_payload()), encoding="utf-8")
    return path


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "db" / "tcc.sqlite3", tmp_path / "artifacts", _write_spec(tmp_path / "spec.json")


def _wait(handle: launcher.WorkerHandle, timeout_s: float = 60.0) -> int:
    return handle.process.wait(timeout=timeout_s)


def _wait_for_state(db: Path, job_id: str, state: str, timeout_s: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if db.exists():
            with ReadModel(db) as rm:
                job = next((j for j in rm.jobs() if j["job_id"] == job_id), {})
            if job.get("state") == state:
                return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not reach {state}; last seen {job.get('state')}")


def test_prepare_refuses_synthetic_wiring_by_default(paths: tuple[Path, Path, Path]) -> None:
    db, art, spec = paths
    refused = launcher.run_prepare(db, art, spec, extra_env=NO_GPU)
    assert refused.returncode == EXIT_REFUSED and not refused.ok
    assert refused.message.startswith("error: SyntheticWiringError")
    assert not db.exists()

    prepared = launcher.run_prepare(db, art, spec, allow_synthetic=True, extra_env=NO_GPU)
    assert prepared.ok, prepared.stderr
    assert prepared.message.startswith("prepared run demo-joint-b0-")
    status = launcher.read_status(db)
    [job] = status["jobs"]
    assert job["state"] == "queued" and job["options"]["synthetic_evaluator"] is True
    assert status["evaluations"] == []


def test_detached_worker_runs_the_job_and_logs_to_a_file(paths: tuple[Path, Path, Path]) -> None:
    db, art, spec = paths
    assert launcher.run_prepare(db, art, spec, allow_synthetic=True, extra_env=NO_GPU).ok
    job_id = launcher.read_status(db)["jobs"][0]["job_id"]

    refused = launcher.spawn_worker(db, art, job_id=job_id, extra_env=NO_GPU)  # no opt-in
    assert _wait(refused) == EXIT_REFUSED
    assert any("SyntheticWiringError" in line for line in launcher.tail(refused.log_path))
    assert launcher.read_status(db)["jobs"][0]["claim_count"] == 0

    handle = launcher.spawn_worker(
        db,
        art,
        job_id=job_id,
        worker_id="launcher-test",
        lease_s=30,
        heartbeat_s=0.5,
        allow_synthetic=True,
        extra_env=NO_GPU,
    )
    assert handle.alive or handle.poll() == 0
    assert _wait(handle) == 0 and not handle.alive
    assert handle.log_path.parent == db.parent / "logs" and handle.log_path != refused.log_path
    lines = launcher.tail(handle.log_path)
    assert lines[0].startswith("# ") and "launched:" in lines[0]
    assert any("job_started" in line for line in lines) and "ended done" in lines[-1]
    assert launcher.latest_log(db) == handle.log_path

    run_id = launcher.read_status(db)["jobs"][0]["run_id"]
    status = launcher.read_status(db, run_id=run_id)
    assert status["jobs"][0]["state"] == "done" and status["jobs"][0]["lease_owner"] is None
    assert [e["status"] for e in status["evaluations"]] == ["valid"] * 3
    assert all(e["synthetic"] for e in status["evaluations"])  # labeled, never passed as measurements


def test_worker_survives_the_process_that_launched_it(paths: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Closing the interface must not kill the worker: the launcher dies, the job still finishes."""
    db, art, _ = paths
    spec = _write_spec(tmp_path / "slow.json", n_candidates=4, sleep_s=0.4)
    code = textwrap.dedent(
        """
        import os, sys
        from pathlib import Path
        from cnnlab.tcc.operation import launcher
        handle = launcher.spawn_worker(
            Path(sys.argv[1]), Path(sys.argv[2]), spec_path=Path(sys.argv[3]), worker_id="orphan-worker",
            lease_s=30, heartbeat_s=0.2, allow_synthetic=True, extra_env={"CUDA_VISIBLE_DEVICES": ""},
        )
        print(handle.pid, flush=True)
        os._exit(0)  # the "interface" dies at once, without any cleanup
        """
    )
    parent = subprocess.run(
        [sys.executable, "-c", code, str(db), str(art), str(spec)],
        env=launcher.cli_environment(NO_GPU),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert parent.returncode == 0, parent.stderr
    assert int(parent.stdout.strip()) > 0
    with_parent_dead = _wait_for_state(db, _job_id(db), "running", timeout_s=30)
    assert with_parent_dead["lease_owner"] == "orphan-worker" and with_parent_dead["lease_live"] is True
    done = _wait_for_state(db, with_parent_dead["job_id"], "done", timeout_s=60)
    assert done["heartbeat_count"] >= 2
    with ReadModel(db) as rm:
        assert [e["kind"] for e in rm.job_events(done["job_id"])] == ["enqueued", "claimed", "done"]
        assert len(rm.evaluations(done["run_id"])) == 4


def _job_id(db: Path, timeout_s: float = 30.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if db.exists():
            try:
                jobs = launcher.read_status(db)["jobs"]
            except ReadModelError:
                jobs = []
            if jobs:
                return jobs[0]["job_id"]
        time.sleep(0.1)
    raise AssertionError("the worker never created its job")


def test_cancel_is_a_short_write_and_never_creates_a_database(paths: tuple[Path, Path, Path]) -> None:
    db, art, spec = paths
    with pytest.raises(ReadModelError) as info:
        launcher.request_cancel(db, art, "job-x", reason="operator")
    assert info.value.code == "not_found" and not db.exists() and not art.exists()

    assert launcher.run_prepare(db, art, spec, allow_synthetic=True, extra_env=NO_GPU).ok
    job_id = launcher.read_status(db)["jobs"][0]["job_id"]
    job = launcher.request_cancel(db, art, job_id, reason="operator changed the plan")
    assert job.state == "cancelled" and job.cancel_reason == "operator changed the plan"
    after = launcher.spawn_worker(db, art, job_id=job_id, allow_synthetic=True, extra_env=NO_GPU)
    assert _wait(after) == 0 and "already cancelled" in launcher.tail(after.log_path)[-1]


def test_spec_validation_and_small_helpers(tmp_path: Path, paths: tuple[Path, Path, Path]) -> None:
    db, art, spec = paths
    assert launcher.load_spec(spec).factory == FACTORY
    for bad, text in (("", "no spec file"), (tmp_path / "none.json", "does not exist")):
        with pytest.raises(ValueError, match=text):
            launcher.load_spec(bad)
    broken = tmp_path / "broken.json"
    broken.write_text('{"method": "joint"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid demo spec"):
        launcher.load_spec(broken)

    assert launcher.is_testing_factory(FACTORY) and launcher.is_testing_factory("a.testing.b:f")
    assert not launcher.is_testing_factory("cnnlab.tcc.integration:demo_wiring")
    assert not launcher.is_testing_factory(None)

    with pytest.raises(ValueError, match="exactly one"):
        launcher.spawn_worker(db, art)
    with pytest.raises(ValueError, match="outside the artifact root"):
        launcher.spawn_worker(db, art, job_id="j", log_dir=art / "logs")
    assert launcher.tail(tmp_path / "no.log") == [] and launcher.latest_log(db) is None
    env = launcher.cli_environment({"X_TEST": "1"})
    assert env["X_TEST"] == "1" and env["PYTHONPATH"].split(os.pathsep)[0].endswith("src")
