"""Queue executor of the demo: runs the items of ``fila.json`` one at a time.

One detached process (``cnnlab-tcc fila --dir DIR``) drives a queue directory
(see ``operation.schedule`` for the files). For each planned run it writes the
spec, spawns the usual ``start --spec`` worker and mirrors the outcome back into
the queue file. It never touches the GPU itself and never imports the science
code: the worker does. The database stays the source of truth about a job;
this file only mirrors it so the operator can edit what has not started.

A live heartbeat (``executor.json``) is how the TUI knows an executor exists,
and it is also what refuses a second executor on the same queue.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from cnnlab.tcc.operation import launcher, schedule
from cnnlab.tcc.storage.errors import JobStateError, NotFoundError, StorageError
from cnnlab.tcc.storage.readmodel import open_read_model

CLI_MODULE = "cnnlab.tcc.operation.cli"
EXIT_ALREADY_RUNNING = 3
HEARTBEAT_S = 5.0
_JOB_TO_ITEM = {"done": schedule.DONE, "cancelled": schedule.CANCELLED, "failed": schedule.FAILED}
_LOG_LINES = 8
MAX_CONSECUTIVE_FAILURES = 3  # then sleep mode is turned off: a broken setup must not draw all night


class _Handle(Protocol):
    process: Any
    log_path: Path


Spawn = Callable[[Path, schedule.QueueItem, Path], _Handle]
JobState = Callable[[str], "dict[str, Any] | None"]
Cancel = Callable[[str, str], object]


def default_job_state(db_path: Path) -> JobState:
    """Read-only lookup of a job in the database; None while the worker has not created it."""

    def lookup(job_id: str) -> dict[str, Any] | None:
        if not Path(db_path).exists():
            return None
        try:
            with open_read_model(db_path) as rm:
                return next((j for j in rm.jobs() if j["job_id"] == job_id), None)
        except (StorageError, OSError):
            return None

    return lookup


class Scheduler:
    """Runs the planned items of one queue directory, one worker at a time."""

    def __init__(
        self,
        queue_dir: Path,
        *,
        spawn: Spawn | None = None,
        job_state: JobState | None = None,
        cancel: Cancel | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = datetime.now,
        poll_s: float = 2.0,
        heartbeat_s: float = HEARTBEAT_S,
        repo_root: Path | None = None,
        lease_s: float = 120.0,
        allow_synthetic: bool = False,
        pid: int | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.queue_dir = Path(queue_dir)
        self.paths = schedule.queue_dir_paths(self.queue_dir)
        self.repo_root, self.lease_s, self.allow_synthetic = repo_root, lease_s, allow_synthetic
        self._spawn = spawn or self._default_spawn
        self._job_state = job_state or default_job_state(self.paths["db"])
        self._cancel = cancel or self._default_cancel
        self._sleep, self._now, self.poll_s, self.heartbeat_s = sleep, now, poll_s, heartbeat_s
        self.pid = os.getpid() if pid is None else pid
        self._log = log or (lambda _msg: None)
        self._item_id: str | None = None
        self._message = "iniciando"
        self.consecutive_failures = 0

    # -- defaults -------------------------------------------------------------

    def _default_spawn(self, spec_path: Path, item: schedule.QueueItem, work_dir: Path) -> launcher.WorkerHandle:
        return launcher.spawn_worker(
            self.paths["db"],
            self.paths["artifacts"],
            spec_path=spec_path,
            lease_s=self.lease_s,
            repo_root=self.repo_root,
            extra_env={"CNNLAB_TCC_WORK_DIR": str(work_dir)},
            allow_synthetic=self.allow_synthetic,
        )

    def _default_cancel(self, job_id: str, reason: str) -> object:
        return launcher.request_cancel(self.paths["db"], self.paths["artifacts"], job_id, reason=reason)

    # -- heartbeat ------------------------------------------------------------

    def beat(self) -> None:
        """Write the heartbeat; a failed write is skipped (the next beat retries), never fatal."""
        try:
            schedule.write_executor(self.queue_dir, pid=self.pid, item_id=self._item_id, message=self._message)
        except OSError as exc:
            self._log(f"heartbeat nao gravado agora: {exc}")

    def _set(self, item_id: str | None, message: str) -> None:
        self._item_id, self._message = item_id, message
        self.beat()

    def _heartbeat_loop(self, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_s):
            try:
                self.beat()
            except OSError:
                pass  # a busy disk must not kill the executor; the next beat retries

    # -- queue helpers --------------------------------------------------------

    def _mark(self, item_id: str, new_state: str, **kw: Any) -> None:
        schedule.mutate(self.queue_dir, lambda s: schedule.set_item_state(s, item_id, new_state, **kw))

    def _pick(self) -> schedule.QueueItem | None:
        """Choose the next item and mark it running in the same locked write."""
        picked: list[schedule.QueueItem] = []

        def change(state: schedule.QueueState) -> schedule.QueueState:
            item = schedule.next_planned(state)
            if item is None and schedule.sleep_active(state.sleep, self._now()):
                state = schedule.add_sleep_draw(state, self._now())
                item = schedule.next_planned(state)
            if item is None:
                return state
            run_id = schedule.run_id_of(item)
            state = schedule.set_item_state(
                state, item.item_id, schedule.RUNNING, run_id=run_id, job_id=f"job-{run_id}"
            )
            picked.append(next(i for i in state.items if i.item_id == item.item_id))
            return state

        schedule.mutate(self.queue_dir, change)
        return picked[0] if picked else None

    def _reconcile(self) -> None:
        """Items left running by a crashed executor: mirror their job, or fail them with a hint.

        A worker is detached, so it may outlive the executor that started it. While
        its lease is live we wait for it: starting the next run now would put two
        jobs on the GPU.
        """
        for item in schedule.load_state(self.queue_dir).items:
            if item.state != schedule.RUNNING:
                continue
            job = self._job_state(item.job_id) if item.job_id else None
            while job and job["state"] == "running" and job.get("lease_live"):
                self._set(item.item_id, f"aguardando o worker ainda vivo de {item.name}")
                if schedule.stop_requested(self.queue_dir):
                    self._try_cancel(str(item.job_id))
                self._sleep(self.poll_s)
                job = self._job_state(str(item.job_id))
            mapped = _JOB_TO_ITEM.get(str(job["state"])) if job else None
            if mapped:
                self._mark(item.item_id, mapped, detail=str(job.get("final_detail") or ""))
            else:
                self._mark(
                    item.item_id, schedule.FAILED, detail="executor interrompido; retome pela aba Monitor (tecla s)"
                )

    # -- one item ---------------------------------------------------------------

    def _run_item(self, item: schedule.QueueItem) -> None:
        job_id = item.job_id or f"job-{schedule.run_id_of(item)}"
        self._set(item.item_id, f"preparando {item.name}")
        try:
            spec_path = schedule.write_spec(self.queue_dir, item)
            handle = self._spawn(spec_path, item, self.paths["work"] / f"{item.item_id}-{item.name}")
        except (OSError, ValueError, StorageError) as exc:
            self._mark(item.item_id, schedule.FAILED, detail=f"nao foi possivel iniciar o worker: {exc}")
            return
        self._set(item.item_id, f"rodando {item.name}")
        cancelled = False
        while handle.process.poll() is None:
            if not cancelled and schedule.stop_requested(self.queue_dir):
                cancelled = self._try_cancel(job_id)
                self._message = f"parando {item.name}"
            self.beat()
            self._sleep(self.poll_s)
        self._finish(item, job_id, handle, handle.process.poll())

    def _try_cancel(self, job_id: str) -> bool:
        """True once the cancel is recorded (or moot); False while the job does not exist yet."""
        try:
            self._cancel(job_id, "parar tudo")
        except NotFoundError:
            return False  # worker still preparing; retry on the next poll
        except JobStateError:
            return True  # already ended by itself
        except (StorageError, OSError, ValueError) as exc:
            self._log(f"cancelamento de {job_id} ainda nao aceito: {exc}")
            return False
        return True

    def _finish(self, item: schedule.QueueItem, job_id: str, handle: _Handle, code: int | None) -> None:
        job = self._job_state(job_id)
        log_note = f"log: {handle.log_path}"
        if job is None:
            tail = " | ".join(launcher.tail(handle.log_path, max_lines=_LOG_LINES))
            self._mark(
                item.item_id,
                schedule.FAILED,
                detail=f"worker saiu (codigo {code}) sem criar o job. {tail} [{log_note}]".strip(),
            )
            return
        mapped = _JOB_TO_ITEM.get(str(job["state"]))
        detail = str(job.get("final_detail") or "")
        if mapped is None:
            mapped = schedule.FAILED
            detail = f"worker saiu (codigo {code}) com o job em {job['state']}; {detail}".strip()
        if mapped != schedule.DONE:
            detail = f"{detail} [{log_note}]".strip()
        self._mark(item.item_id, mapped, detail=detail)

    # -- main loop --------------------------------------------------------------

    def run(self) -> int:
        """Run until the queue is empty or "stop all"; 3 when another executor is alive."""
        other = schedule.read_executor(self.queue_dir)
        if other and other.get("alive") and other.get("pid") != self.pid:
            self._log(f"ja existe um executor vivo (pid {other.get('pid')}); nada a fazer")
            return EXIT_ALREADY_RUNNING
        self.beat()
        stop = threading.Event()
        thread = threading.Thread(target=self._heartbeat_loop, args=(stop,), daemon=True)
        thread.start()
        try:
            self._reconcile()
            while True:
                if schedule.stop_requested(self.queue_dir):
                    self._log("parada solicitada; encerrando")
                    break
                item = self._pick()
                if item is None:
                    self._log("fila vazia")
                    break
                self._log(f"iniciando {item.name}")
                self._run_item(item)
                self._set(None, f"{item.name} terminou")
                ended = next((i for i in schedule.load_state(self.queue_dir).items if i.item_id == item.item_id), None)
                self.consecutive_failures = (
                    self.consecutive_failures + 1 if ended and ended.state == schedule.FAILED else 0
                )
                if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    schedule.mutate(self.queue_dir, lambda st: replace(st, sleep=replace(st.sleep, enabled=False)))
                    self._log(
                        f"{self.consecutive_failures} runs seguidas falharam; sono desligado e fila parada. "
                        "Veja o detalhe das runs e o log do worker"
                    )
                    break
        finally:
            stop.set()
            thread.join(timeout=2)
            self.paths["executor"].unlink(missing_ok=True)  # a clean exit must not look alive for 30 s
        return 0


def spawn_executor(
    queue_dir: Path, *, repo_root: Path | None = None, extra_env: Mapping[str, str] | None = None
) -> launcher.WorkerHandle:
    """Start ``cnnlab-tcc fila --dir DIR`` detached, logging to ``logs/executor-<timestamp>.log``.

    Built here and not with ``launcher.cli_command`` because that one targets
    the ``operation.demo`` module, which has no ``fila`` command.
    """
    queue_dir = Path(queue_dir)
    command = [sys.executable, "-m", CLI_MODULE, "fila", "--dir", str(queue_dir)]
    if repo_root is not None:
        command += ["--repo-root", str(repo_root)]
    started = datetime.now()
    logs = queue_dir / launcher.LOG_DIRNAME
    logs.mkdir(parents=True, exist_ok=True)
    log_path, attempt = logs / f"executor-{started.strftime('%Y%m%dT%H%M%S%f')}.log", 1
    while True:
        try:
            log = open(log_path, "xb")
            break
        except FileExistsError:
            attempt += 1
            log_path = logs / f"executor-{started.strftime('%Y%m%dT%H%M%S%f')}.{attempt}.log"
    with log:
        log.write(f"# {started.isoformat()} launched: {' '.join(command)}\n".encode())
        log.flush()
        process = subprocess.Popen(
            command,
            env=launcher.cli_environment(extra_env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            **launcher._detached_kwargs(),
        )
    return launcher.WorkerHandle(
        pid=process.pid,
        target="fila",
        log_path=log_path,
        command=tuple(command),
        started_at=started.isoformat(),
        process=process,
    )
