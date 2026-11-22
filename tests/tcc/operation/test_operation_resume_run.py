"""Continuing a run by its id, and the warning when a spec creates another run (CPU only).

Everything here uses the synthetic test wiring, opted into explicitly: the
values are stand-ins labeled `quality.synthetic`, and only the operation
(queue, run identity, refusals) is under test.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import cnnlab.tcc.operation.demo as demo
from cnnlab.tcc.operation import cli, launcher
from cnnlab.tcc.operation.demo import (
    EXIT_REFUSED,
    CodeVersionMismatchError,
    DemoRunSpec,
    SyntheticWiringError,
    prepare_demo,
    resume_run,
    start_demo,
)
from cnnlab.tcc.operation.provenance import CodeVersion
from cnnlab.tcc.operation.testing import synthetic_demo_wiring
from cnnlab.tcc.storage import JobQueue, ReadModel, SQLiteRunStore
from cnnlab.tcc.storage.errors import NotFoundError

FACTORY = "cnnlab.tcc.operation.testing:synthetic_demo_wiring"
NO_GPU = {"CUDA_VISIBLE_DEVICES": "-1"}
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


def _prepare(db: Path, art: Path, spec: DemoRunSpec, code: CodeVersion) -> demo.PreparedDemo:
    with SQLiteRunStore(db, art) as store:
        return prepare_demo(store, spec, synthetic_demo_wiring(spec), code, allow_synthetic=True)


def _cancel(db: Path, art: Path, job_id: str, reason: str = "test: stop") -> None:
    with SQLiteRunStore(db, art) as store:
        JobQueue(store).request_cancel(job_id, reason=reason)


def _cancel_after(db: Path, art: Path, job: Any, n_valid: int) -> threading.Thread:
    """Ask for a cooperative cancel once `n_valid` attempts are committed."""

    def watch() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with ReadModel(db) as rm:
                if rm.run_progress(job.run_id)["attempts"].get("valid", 0) >= n_valid:
                    break
            time.sleep(0.02)
        _cancel(db, art, job.job_id, "test: stop midway")

    thread = threading.Thread(target=watch)
    thread.start()
    return thread


def test_resume_run_continues_the_same_run_with_a_new_job(paths: tuple[Path, Path]) -> None:
    db, art = paths
    spec = _spec(factory_args={"n_candidates": 6, "sleep_s": 0.2})
    prepared = _prepare(db, art, spec, CODE_A)
    run_id = prepared.job_spec.run_id
    watcher = _cancel_after(db, art, prepared.job, 2)
    stopped = start_demo(
        db, art, spec, synthetic_demo_wiring, worker_id="w1", code_version=CODE_A, allow_synthetic=True
    )
    watcher.join(timeout=30)
    assert stopped.outcome is not None and stopped.outcome.end_state == "cancelled"
    with ReadModel(db) as rm:
        before = sorted(r["candidate_id"] for r in rm.evaluations(run_id) if r["status"] == "valid")
    assert 2 <= len(before) < 6

    resumed = resume_run(db, art, run_id, worker_id="w2", code_version=CODE_A, allow_synthetic=True)
    assert resumed.outcome is not None and resumed.outcome.end_state == "done"
    assert resumed.run_id == run_id and resumed.job_id == f"job-{run_id}-r2"
    with ReadModel(db) as rm:
        assert [r["run_id"] for r in rm.runs()] == [run_id]  # the run was looked up, not derived again
        old, new = rm.jobs()
        rows = rm.evaluations(run_id)
        [claim] = [e for e in rm.job_events(new["job_id"]) if e["kind"] == "claimed"]
    assert (old["state"], new["state"]) == ("cancelled", "done") and new["run_id"] == run_id
    assert new["options"]["resumed_from_job"] == old["job_id"]
    assert new["options"]["demo_spec"]["job_id"] == new["job_id"]
    # the spec is the stored one, only the job id is new (the wiring factory sees what it saw the first time)
    assert {**new["options"]["demo_spec"], "job_id": None} == {**old["options"]["demo_spec"], "job_id": None}
    assert new["options"]["code_version"] == old["options"]["code_version"]  # the code recorded for the run
    assert "code_changed_from" not in claim["detail"]
    valid = sorted(r["candidate_id"] for r in rows if r["status"] == "valid")
    assert valid == [f"c{i}" for i in range(6)] and len(rows) == 6  # only what was missing ran; none twice

    again = resume_run(db, art, run_id, worker_id="w3", code_version=CODE_A, allow_synthetic=True)
    assert again.outcome is None and "already done" in again.message
    with ReadModel(db) as rm:
        assert len(rm.jobs()) == 2


def test_resume_run_with_other_code_is_refused_before_queueing_unless_allowed(paths: tuple[Path, Path]) -> None:
    db, art = paths
    prepared = _prepare(db, art, _spec(), CODE_A)
    run_id = prepared.job_spec.run_id
    _cancel(db, art, prepared.job.job_id)  # queued: cancelled at once

    with pytest.raises(NotFoundError, match="does not exist"):
        resume_run(db, art, "demo-joint-b0-00000000", worker_id="w", code_version=CODE_A, allow_synthetic=True)
    with pytest.raises(SyntheticWiringError):
        resume_run(db, art, run_id, worker_id="w", code_version=CODE_A)  # no opt-in
    with pytest.raises(CodeVersionMismatchError, match="started with code"):
        resume_run(db, art, run_id, worker_id="w", code_version=CODE_B, allow_synthetic=True)
    with ReadModel(db) as rm:
        assert [j["state"] for j in rm.jobs()] == ["cancelled"] and len(rm.runs()) == 1  # refusals left nothing

    allowed = resume_run(
        db, art, run_id, worker_id="w", code_version=CODE_B, allow_code_change=True, allow_synthetic=True
    )
    assert allowed.outcome is not None and allowed.outcome.end_state == "done" and allowed.run_id == run_id
    with ReadModel(db) as rm:
        assert [r["run_id"] for r in rm.runs()] == [run_id]  # other code, still the same run
        [claim] = [e for e in rm.job_events(allowed.job_id) if e["kind"] == "claimed"]
        [run] = rm.runs()
    assert claim["detail"]["code_changed_from"]["git_sha"] == "a" * 40
    assert claim["detail"]["code_version"]["git_sha"] == "b" * 40
    assert run["git_sha"] == "a" * 40  # the run keeps what it was created with; the event tells the change


def test_resume_run_works_the_active_job_of_the_run(paths: tuple[Path, Path]) -> None:
    db, art = paths
    prepared = _prepare(db, art, _spec(), CODE_A)
    result = resume_run(db, art, prepared.job_spec.run_id, worker_id="w", code_version=CODE_A, allow_synthetic=True)
    assert result.outcome is not None and result.outcome.end_state == "done"
    assert result.job_id == prepared.job.job_id
    with ReadModel(db) as rm:
        [job] = rm.jobs()
        assert [e["kind"] for e in rm.job_events(job["job_id"])] == ["enqueued", "claimed", "done"]


def test_same_spec_with_other_code_warns_that_it_is_another_run(
    paths: tuple[Path, Path], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db, art = paths
    spec = _spec()
    first = _prepare(db, art, spec, CODE_A)
    assert first.warnings == () and _prepare(db, art, spec, CODE_A).warnings == ()
    assert _prepare(db, art, _spec(search_seed=12), CODE_B).warnings == ()  # another spec: nothing to confuse
    _cancel(db, art, first.job.job_id)

    # The way the operator meets it: the same spec file, after a commit.
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec.to_payload()), encoding="utf-8")
    monkeypatch.setattr(demo, "capture_code_version", lambda _label, _root=None: CODE_B)
    assert (
        demo.main(["prepare", "--db", str(db), "--artifacts", str(art), "--spec", str(spec_file), "--allow-synthetic"])
        == 0
    )
    warning, prepared_line = capsys.readouterr().out.strip().splitlines()
    with ReadModel(db) as rm:
        new_run = next(
            j["run_id"]
            for j in rm.jobs()
            if j["options"]["code_version"]["git_sha"] == "b" * 40 and j["options"]["demo_spec"]["search_seed"] == 11
        )
    old_run = first.job_spec.run_id
    assert new_run != old_run and prepared_line.startswith(f"prepared run {new_run} ")
    assert warning.startswith(f"warning: {new_run} is a NEW run, not the continuation of {old_run} ")
    assert f"(job {first.job.job_id} is cancelled)" in warning
    assert "the code changed (git aaaaaaaaaaaa dirty=False -> git bbbbbbbbbbbb dirty=True)" in warning
    assert warning.endswith(f"To continue that run: start --run-id {old_run}")
    result = launcher.CommandResult(("x",), 0, f"{warning}\n{prepared_line}\n", "")
    assert result.message == prepared_line and result.warnings == (warning[len("warning:") :].strip(),)

    assert _prepare(db, art, spec, CODE_B).warnings == ()  # the run exists now: said once, when it was created

    # A finished run is not something to continue: no warning about it.
    done = resume_run(db, art, old_run, worker_id="w", code_version=CODE_A, allow_synthetic=True)
    assert done.outcome is not None and done.outcome.end_state == "done"
    code_c = CodeVersion("demo-test-c", "c" * 40, False, "3" * 64, (), (), 0)
    [only] = _prepare(db, art, spec, code_c).warnings
    assert f"not the continuation of {new_run} " in only and "is queued" in only and old_run not in only


def test_start_warns_before_the_work_begins(paths: tuple[Path, Path]) -> None:
    db, art = paths
    spec = _spec()
    first = _prepare(db, art, spec, CODE_A)
    _cancel(db, art, first.job.job_id)
    seen: list[str] = []
    events: list[str] = []
    result = start_demo(
        db,
        art,
        spec,
        synthetic_demo_wiring,
        worker_id="w",
        code_version=CODE_B,
        allow_synthetic=True,
        on_warning=lambda text: seen.append(f"{len(events)} events: {text}"),
        on_event=lambda e: events.append(e.kind),
    )
    assert result.outcome is not None and result.outcome.end_state == "done"
    assert result.run_id != first.job_spec.run_id and events
    [warning] = seen
    assert warning.startswith(f"0 events: warning: {result.run_id} is a NEW run")
    assert f"start --run-id {first.job_spec.run_id}" in warning


def test_cli_run_by_run_id_and_the_old_paths_still_work(paths: tuple[Path, Path], tmp_path: Path) -> None:
    db, art = paths
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(_spec().to_payload()), encoding="utf-8")
    base = ["--db", str(db), "--artifacts", str(art)]
    assert launcher.run_prepare(db, art, spec_file, allow_synthetic=True, extra_env=NO_GPU).ok
    with ReadModel(db) as rm:
        [job] = rm.jobs()
    run_id = job["run_id"]
    launcher.request_cancel(db, art, job["job_id"], reason="ensaio")

    def run(*args: str) -> Any:
        return launcher.subprocess.run(
            [launcher.sys.executable, "-m", "cnnlab.tcc.operation.cli", "run", *base, *args],
            env=launcher.cli_environment(NO_GPU),
            capture_output=True,
            text=True,
            timeout=120,
        )

    both = run("--run-id", run_id, "--spec", str(spec_file))
    assert both.returncode == 2 and "not allowed with argument" in both.stderr
    unknown = run("--run-id", "no-such-run", "--allow-synthetic")
    assert unknown.returncode == EXIT_REFUSED and unknown.stderr.strip().startswith("error: NotFoundError")
    refused = run("--run-id", run_id)  # synthetic wiring is never the default
    assert refused.returncode == EXIT_REFUSED and "SyntheticWiringError" in refused.stderr

    # Old behavior: the same spec does not resume a cancelled job.
    same = run("--spec", str(spec_file), "--allow-synthetic")
    assert same.returncode == 0 and "already cancelled; nothing to do" in same.stdout

    resumed = run("--run-id", run_id, "--allow-synthetic", "--worker-id", "cli-2")
    assert resumed.returncode == 0, resumed.stderr
    assert f"job job-{run_id}-r2 ended done" in resumed.stdout
    with ReadModel(db) as rm:
        assert [(j["state"], j["run_id"]) for j in rm.jobs()] == [("cancelled", run_id), ("done", run_id)]
        assert len(rm.runs()) == 1

    # The launcher of the interfaces takes the same path (nothing left to do here: the run is done).
    handle = launcher.spawn_worker(db, art, run_id=run_id, allow_synthetic=True, extra_env=NO_GPU)
    assert handle.process.wait(timeout=60) == 0 and handle.target == run_id
    assert "--run-id" in handle.command and "already done" in handle.log_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        launcher.spawn_worker(db, art, job_id="j", run_id=run_id)
    assert "--run-id RUN" in (cli.__doc__ or "")
