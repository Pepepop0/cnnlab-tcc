"""Queue executor: ordering, sleep draws, stop, reconciliation, refusal of a second executor."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.operation import schedule as sc
from cnnlab.tcc.operation import scheduler as sch
from cnnlab.tcc.storage.errors import NotFoundError


class FakeProcess:
    def __init__(self, running_polls: int, on_poll: Any = None) -> None:
        self.left, self.on_poll = running_polls, on_poll

    def poll(self) -> int | None:
        if self.on_poll:
            self.on_poll()
        if self.left > 0:
            self.left -= 1
            return None
        return 0


class FakeHandle:
    def __init__(self, process: FakeProcess, log_path: Path) -> None:
        self.process, self.log_path = process, log_path


class Env:
    """Fakes for spawn / job_state / cancel around a tmp queue directory."""

    def __init__(self, queue_dir: Path, *, polls: int = 2, create_jobs: bool = True) -> None:
        self.dir, self.polls, self.create_jobs = queue_dir, polls, create_jobs
        self.spawned: list[tuple[Path, sc.QueueItem, Path]] = []
        self.jobs: dict[str, dict[str, Any]] = {}
        self.cancels: list[tuple[str, str]] = []
        self.cancel_fail = 0
        self.outcome = "done"
        self.on_poll: Any = None

    def spawn(self, spec_path: Path, item: sc.QueueItem, work_dir: Path) -> FakeHandle:
        self.spawned.append((spec_path, item, work_dir))
        log = self.dir / f"w-{item.item_id}.log"
        log.write_text("linha 1\nboom: traceback\n", encoding="utf-8")
        if self.create_jobs:
            self.jobs[item.job_id] = {"job_id": item.job_id, "state": self.outcome, "final_detail": "ok"}
        return FakeHandle(FakeProcess(self.polls, self.on_poll), log)

    def job_state(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str, reason: str) -> None:
        if self.cancel_fail > 0:
            self.cancel_fail -= 1
            raise NotFoundError(f"no job {job_id}")
        self.cancels.append((job_id, reason))
        if job_id in self.jobs:
            self.jobs[job_id]["state"] = "cancelled"

    def scheduler(self, **kw: Any) -> sch.Scheduler:
        kw.setdefault("sleep", lambda _s: None)
        return sch.Scheduler(
            self.dir, spawn=self.spawn, job_state=self.job_state, cancel=self.cancel, heartbeat_s=0.05, **kw
        )


@pytest.fixture
def qdir(tmp_path: Path) -> Path:
    return tmp_path / "fila"


def _add(qdir: Path, name: str, tag: str = sc.MANUAL) -> None:
    sc.mutate(qdir, lambda s: sc.add_item(s, name, sc.RunSettings(), tag=tag))


def _items(qdir: Path) -> dict[str, sc.QueueItem]:
    return {i.name: i for i in sc.load_state(qdir).items}


def test_manual_runs_before_sleep_and_state_is_mirrored(qdir: Path) -> None:
    _add(qdir, "sono-a", sc.SLEEP)
    _add(qdir, "manual-b")
    env = Env(qdir)
    assert env.scheduler().run() == 0
    assert [i.name for _, i, _ in env.spawned] == ["manual-b", "sono-a"]
    items = _items(qdir)
    for name, item in items.items():
        assert item.state == sc.DONE
        assert item.run_id == f"fila-{item.item_id}-{name}"
        assert item.job_id == f"job-{item.run_id}"
        assert item.started_at and item.ended_at
    spec, _, work = env.spawned[0]
    assert spec.is_file() and work == qdir / "work" / "0002-manual-b"
    assert sc.read_executor(qdir) is None  # heartbeat file removed on clean exit


def test_sleep_draws_until_deadline_then_exits(qdir: Path) -> None:
    clock = [datetime(2026, 9, 29, 22, 0, 0)]
    until = (clock[0] + timedelta(minutes=10)).isoformat()
    sc.mutate(qdir, lambda s: sc.QueueState(sleep=sc.SleepConfig(enabled=True, until=until, seed=7)))
    env = Env(qdir)

    def now() -> datetime:
        return clock[0]

    def advance(_s: float) -> None:
        clock[0] += timedelta(minutes=4)  # each poll moves the clock

    assert env.scheduler(now=now, sleep=advance).run() == 0
    state = sc.load_state(qdir)
    assert len(env.spawned) == state.sleep.draws >= 2
    assert all(i.tag == sc.SLEEP and i.state == sc.DONE for i in state.items)
    assert sc.draw_settings(state.sleep, 0) == state.items[0].settings


def test_empty_queue_exits(qdir: Path) -> None:
    env = Env(qdir)
    assert env.scheduler().run() == 0
    assert env.spawned == []


def test_stop_cancels_current_job_once_and_exits(qdir: Path) -> None:
    _add(qdir, "um")
    _add(qdir, "dois")
    env = Env(qdir, polls=10_000)
    env.cancel_fail = 2  # job not there yet: retried on later polls
    ticks = {"n": 0}

    def on_poll() -> None:
        ticks["n"] += 1
        if ticks["n"] == 2:
            sc.request_stop(qdir)
        if env.cancels:
            proc.left = 0

    env.on_poll = on_poll
    proc: FakeProcess = None  # type: ignore[assignment]
    orig = env.spawn

    def spawn(spec: Path, item: sc.QueueItem, work: Path) -> FakeHandle:
        nonlocal proc
        handle = orig(spec, item, work)
        proc = handle.process
        return handle

    env.spawn = spawn  # type: ignore[method-assign]
    assert env.scheduler().run() == 0
    assert env.cancels == [("job-fila-0001-um", "parar tudo")]
    items = _items(qdir)
    assert items["um"].state == sc.CANCELLED
    assert items["dois"].state == sc.PLANNED
    assert len(env.spawned) == 1


def test_missing_job_fails_with_log_tail(qdir: Path) -> None:
    _add(qdir, "sem-job")
    env = Env(qdir, create_jobs=False)
    env.scheduler().run()
    item = _items(qdir)["sem-job"]
    assert item.state == sc.FAILED
    assert "boom: traceback" in item.detail and "w-0001.log" in item.detail


def test_failed_job_is_mirrored_with_log_path(qdir: Path) -> None:
    _add(qdir, "quebra")
    env = Env(qdir)
    env.outcome = "failed"
    env.scheduler().run()
    item = _items(qdir)["quebra"]
    assert item.state == sc.FAILED and "w-0001.log" in item.detail


def test_crash_reconciliation(qdir: Path) -> None:
    for name in ("a", "b", "c", "d"):
        _add(qdir, name)
    for iid, job in (("0001", "done"), ("0002", "cancelled"), ("0003", "running"), ("0004", None)):
        sc.mutate(qdir, lambda s, iid=iid: sc.set_item_state(s, iid, sc.RUNNING, run_id=f"r{iid}", job_id=f"j{iid}"))
    env = Env(qdir)
    env.jobs = {
        "j0001": {"job_id": "j0001", "state": "done", "final_detail": "fim"},
        "j0002": {"job_id": "j0002", "state": "cancelled", "final_detail": ""},
        "j0003": {"job_id": "j0003", "state": "running", "final_detail": ""},
    }
    assert env.scheduler().run() == 0
    items = _items(qdir)
    assert items["a"].state == sc.DONE and items["a"].detail == "fim"
    assert items["b"].state == sc.CANCELLED
    assert items["c"].state == sc.FAILED and "aba Monitor (tecla s)" in items["c"].detail
    assert items["d"].state == sc.FAILED
    assert env.spawned == []


def test_second_live_executor_is_refused(qdir: Path) -> None:
    _add(qdir, "x")
    sc.write_executor(qdir, pid=os.getpid() + 12345, item_id=None, message="vivo")
    env = Env(qdir)
    assert env.scheduler().run() == sch.EXIT_ALREADY_RUNNING
    assert env.spawned == [] and _items(qdir)["x"].state == sc.PLANNED
    assert sc.read_executor(qdir)["pid"] == os.getpid() + 12345  # the live one is left alone


def test_stale_executor_does_not_block(qdir: Path) -> None:
    _add(qdir, "x")
    sc.write_executor(qdir, pid=1, item_id=None, message="morto")
    data = sc.read_executor(qdir)
    assert data is not None
    stale = time.time() - 3600
    os.utime(qdir / sc.EXECUTOR_FILE, (stale, stale))
    import json

    (qdir / sc.EXECUTOR_FILE).write_text(json.dumps({"pid": 1, "beat": stale}), encoding="utf-8")
    env = Env(qdir)
    assert env.scheduler().run() == 0
    assert _items(qdir)["x"].state == sc.DONE


def test_existing_stop_flag_exits_without_running(qdir: Path) -> None:
    _add(qdir, "x")
    sc.request_stop(qdir)
    env = Env(qdir)
    assert env.scheduler().run() == 0
    assert env.spawned == []


def test_spawn_executor_builds_detached_command(qdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakePopen:
        pid = 4242

        def __init__(self, command: list[str], **kw: Any) -> None:
            captured["command"], captured["kw"] = command, kw

    monkeypatch.setattr(sch.subprocess, "Popen", FakePopen)
    handle = sch.spawn_executor(qdir, repo_root=qdir / "repo", extra_env={"X": "1"})
    cmd = captured["command"]
    assert cmd[1:] == ["-m", "cnnlab.tcc.operation.cli", "fila", "--dir", str(qdir), "--repo-root", str(qdir / "repo")]
    assert captured["kw"]["env"]["X"] == "1"
    assert captured["kw"]["stdin"] is not None
    assert handle.pid == 4242
    assert handle.log_path.parent == qdir / "logs" and handle.log_path.name.startswith("executor-")
    assert handle.log_path.is_file()


def test_consecutive_failures_turn_sleep_off_and_stop(qdir: Path) -> None:
    sc.mutate(qdir, lambda s: sc.QueueState(sleep=sc.SleepConfig(enabled=True, seed=1)))
    env = Env(qdir, create_jobs=False)  # every worker dies before creating its job
    assert env.scheduler().run() == 0
    state = sc.load_state(qdir)
    assert len(env.spawned) == sch.MAX_CONSECUTIVE_FAILURES
    assert all(i.state == sc.FAILED for i in state.items) and not state.sleep.enabled


def test_a_success_resets_the_failure_count(qdir: Path) -> None:
    for name in ("f1", "f2", "ok", "f3", "f4"):
        _add(qdir, name)
    env = Env(qdir)
    orig = env.spawn

    def spawn(spec: Path, item: sc.QueueItem, work: Path) -> FakeHandle:
        env.outcome = "done" if item.name == "ok" else "failed"
        return orig(spec, item, work)

    env.spawn = spawn  # type: ignore[method-assign]
    env.scheduler().run()
    assert len(env.spawned) == 5


def test_reconcile_waits_for_a_worker_that_outlived_the_executor(qdir: Path) -> None:
    _add(qdir, "vivo")
    _add(qdir, "proximo")
    sc.mutate(qdir, lambda s: sc.set_item_state(s, "0001", sc.RUNNING, run_id="r1", job_id="j1"))
    env = Env(qdir)
    env.jobs["j1"] = {"job_id": "j1", "state": "running", "lease_live": True, "final_detail": ""}
    waits = {"n": 0}

    def sleep(_s: float) -> None:
        waits["n"] += 1
        if waits["n"] == 3:
            env.jobs["j1"].update(state="done", lease_live=None, final_detail="terminou sozinho")

    assert env.scheduler(sleep=sleep).run() == 0
    items = _items(qdir)
    assert items["vivo"].state == sc.DONE and items["vivo"].detail == "terminou sozinho"
    assert [i.name for _, i, _ in env.spawned] == ["proximo"]  # only after the live worker ended
    assert waits["n"] >= 3
