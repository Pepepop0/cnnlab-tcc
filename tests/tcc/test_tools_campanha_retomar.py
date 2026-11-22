"""The campaign resume step: which queue items it continues, and what it leaves alone."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.operation import schedule

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("retomar", ROOT / "tools" / "campanha" / "retomar.py")
assert spec is not None and spec.loader is not None
retomar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retomar)


def item(item_id: str, state: str, *, plan: bool = True) -> schedule.QueueItem:
    settings = schedule.RunSettings(
        slice="admitted",
        method="joint",
        plan_id="campaign-plan-x" if plan else None,
        seed_block=0,
        population=30,
        generations=None,
        epochs=5,
        train_per_class=None,
    )
    return schedule.QueueItem(
        item_id,
        f"joint-b{item_id}",
        "campanha",
        settings,
        state=state,
        run_id=f"run-{item_id}",
        job_id=f"job-run-{item_id}",
    )


def job(state: str, *, cancel: bool = False, live: bool | None = None) -> dict[str, Any]:
    return {"job_id": "j", "state": state, "cancel_requested": cancel, "lease_live": live, "final_detail": ""}


@pytest.fixture
def queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    def setup(items: list[schedule.QueueItem], jobs: dict[str, list[dict[str, Any]]]) -> list[str]:
        monkeypatch.setattr(retomar.schedule, "load_state", lambda _d: schedule.QueueState(items=tuple(items)))
        monkeypatch.setattr(retomar.schedule, "run_id_of", lambda it: it.run_id)
        monkeypatch.setattr(retomar, "run_jobs", lambda _db, run_id: jobs.get(run_id, []))
        return [it.item_id for it in retomar.unfinished(tmp_path, tmp_path / "db")]

    return setup


def test_a_paused_or_crashed_run_is_continued_and_a_finished_or_failed_one_is_not(queue: Any) -> None:
    items = [
        item("1", schedule.DONE),
        item("2", schedule.CANCELLED),
        item("3", schedule.FAILED),
        item("4", schedule.FAILED),
        item("5", schedule.FAILED),
        item("6", schedule.PLANNED),
        item("7", schedule.CANCELLED, plan=False),
    ]
    jobs = {
        "run-1": [job("done")],
        "run-2": [job("cancelled", cancel=True)],
        "run-3": [job("interrupted", live=False)],
        "run-4": [job("failed")],
        "run-5": [],
        "run-7": [job("cancelled")],
    }
    assert queue(items, jobs) == ["2", "3"]  # paused, crashed; not done, not a real failure, not refused, not manual


def test_a_job_stuck_with_a_pending_cancel_and_no_worker_stops_the_resume(queue: Any) -> None:
    with pytest.raises(SystemExit, match="cancelamento pendente"):
        queue([item("1", schedule.RUNNING)], {"run-1": [job("interrupted", cancel=True, live=False)]})


def test_a_live_worker_is_never_doubled(queue: Any) -> None:
    with pytest.raises(SystemExit, match="ainda esta vivo"):
        queue([item("1", schedule.RUNNING)], {"run-1": [job("running", cancel=True, live=True)]})
