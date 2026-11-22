"""Continue the campaign runs that a pause left unfinished, before the queue executor starts.

The frozen executor (``cnnlab-tcc fila``) cancels the running job when ``parar.flag`` appears and then treats the
cancelled item as final: on the next start it moves on to the next planned run, so the paused run would stay
unfinished and the plan's order would change. A crash leaves the item failed with the job still active, same effect.

This continues each such run, in queue order, with ``start --run-id`` (same run, its checkpoint and the GPU budget it
has left; the interrupted attempt keeps its time, as the protocol's interruption policy says) and mirrors the end in
the queue file. A run whose last job failed by itself is left alone: that needs a look, not a blind retry.

Runs with the code of the frozen worktree (PYTHONPATH=<worktree>/src), like the executor. Exit 0: nothing left to
continue (the executor may start); 2: paused again (parar.flag); 1: a continued run did not finish; 3: another
executor is alive.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from cnnlab.tcc.operation import launcher, schedule
from cnnlab.tcc.storage.errors import JobStateError, NotFoundError, StorageError
from cnnlab.tcc.storage.readmodel import open_read_model

JOB_TO_ITEM = {"done": schedule.DONE, "cancelled": schedule.CANCELLED, "failed": schedule.FAILED}
ACTIVE = ("queued", "running", "interrupted")
POLL_S = 2.0


def run_jobs(db: Path, run_id: str) -> list[dict[str, Any]]:
    """The run's jobs in enqueue order (read-only)."""
    with open_read_model(db) as rm:
        return [j for j in rm.jobs() if j["run_id"] == run_id]


def unfinished(queue_dir: Path, db: Path) -> list[schedule.QueueItem]:
    """Campaign items a pause or a crash left unfinished, in queue order."""
    out = []
    for item in schedule.load_state(queue_dir).items:
        if not item.from_plan or item.state not in (schedule.CANCELLED, schedule.FAILED, schedule.RUNNING):
            continue
        jobs = run_jobs(db, schedule.run_id_of(item))
        if not jobs:
            continue  # refused before its job existed: nothing to continue
        last = jobs[-1]
        if last["state"] == "running" and last.get("lease_live"):
            raise SystemExit(f"{item.name}: o worker do job {last['job_id']} ainda esta vivo; espere ele terminar")
        if last["state"] == "cancelled" or last["state"] in ACTIVE:
            if last["state"] in ACTIVE and last["cancel_requested"] and not last.get("lease_live"):
                raise SystemExit(
                    f"{item.name}: job {last['job_id']} ficou {last['state']} com cancelamento pendente "
                    "e sem worker; o codigo congelado nao sai desse estado sozinho. Veja antes de seguir."
                )
            out.append(item)
    return out


def log(text: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} retomar: {text}", flush=True)


def continue_item(queue_dir: Path, paths: dict[str, Path], item: schedule.QueueItem, repo_root: Path) -> str:
    """Continue one run with ``start --run-id``; returns the item's final state."""
    run_id = schedule.run_id_of(item)
    schedule.mutate(
        queue_dir,
        lambda s: schedule.set_item_state(
            s, item.item_id, schedule.RUNNING, run_id=run_id, job_id=item.job_id or f"job-{run_id}"
        ),
    )
    handle = launcher.spawn_worker(
        paths["db"],
        paths["artifacts"],
        run_id=run_id,
        lease_s=120.0,
        repo_root=repo_root,
        extra_env={"CNNLAB_TCC_WORK_DIR": str(paths["work"] / f"{item.item_id}-{item.name}")},
    )
    log(f"{item.name}: continuando ({run_id}); log {handle.log_path}")
    cancelled = False
    while handle.process.poll() is None:
        schedule.write_executor(queue_dir, pid=os.getpid(), item_id=item.item_id, message=f"retomando {item.name}")
        if not cancelled and schedule.stop_requested(queue_dir):
            active = [j for j in run_jobs(paths["db"], run_id) if j["state"] in ACTIVE]
            if active:
                try:
                    launcher.request_cancel(paths["db"], paths["artifacts"], active[-1]["job_id"], reason="parar tudo")
                    cancelled = True
                    log(f"{item.name}: pausa pedida; a tentativa em curso termina e e gravada")
                except (NotFoundError, JobStateError, StorageError, OSError, ValueError) as exc:
                    log(f"{item.name}: cancelamento ainda nao aceito ({exc}); tento de novo")
        time.sleep(POLL_S)
    last = run_jobs(paths["db"], run_id)[-1]
    state = JOB_TO_ITEM.get(str(last["state"]))
    detail = str(last.get("final_detail") or "")
    if state is None:
        state, detail = schedule.FAILED, f"worker saiu (codigo {handle.process.poll()}) com o job em {last['state']}"
    schedule.mutate(
        queue_dir,
        lambda s: schedule.set_item_state(
            s, item.item_id, state, detail=f"retomada: {detail} [log: {handle.log_path}]".strip()
        ),
    )
    log(f"{item.name}: {state} ({detail})")
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, required=True, help="campaign queue folder")
    parser.add_argument("--repo-root", type=Path, required=True, help="the frozen worktree")
    args = parser.parse_args(argv)
    queue_dir = args.dir
    paths = schedule.queue_dir_paths(queue_dir)
    other = schedule.read_executor(queue_dir)
    if other and other.get("alive") and other.get("pid") != os.getpid():
        log(f"ja existe um executor vivo (pid {other.get('pid')}); nada a fazer")
        return 3
    try:
        for item in unfinished(queue_dir, paths["db"]):
            if schedule.stop_requested(queue_dir):
                log("parar.flag presente; nada retomado")
                return 2
            state = continue_item(queue_dir, paths, item, args.repo_root)
            if state == schedule.CANCELLED:
                return 2
            if state != schedule.DONE:
                return 1
        log("nenhuma run pendente de retomada")
        return 0
    finally:
        paths["executor"].unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
