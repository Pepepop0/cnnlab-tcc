"""Opt-in cross-check of the read side against a REAL run database (read-only).

Skipped unless `TCC_REAL_DB` names a database file. Nothing here is faked and
nothing is written: the database is opened `mode=ro` + `query_only` through
the product read model, the page data layer and the TUI (in `--read-only`
mode), and its files are compared byte by byte before and after.

    $env:TCC_REAL_DB = "C:\\...\\demo.sqlite"
    $env:TCC_REAL_DB_EXPECT = '{"valid": 12, "generations": 3, "checkpoints": 6, "fronts": 3, "git_sha": "6912bf0..."}'
    pytest tests/tcc/test_demo_real_db.py -q

`TCC_REAL_DB_EXPECT` is optional; without it only the invariants are checked.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import DataTable, Static

import cnnlab.tcc.demo_streamlit as page
import cnnlab.tcc.demo_tui as tui
from cnnlab.tcc.operation.demo import demo_status, format_status
from cnnlab.tcc.storage import open_read_model

REAL_DB = os.environ.get("TCC_REAL_DB", "").strip()
EXPECT: dict[str, Any] = json.loads(os.environ.get("TCC_REAL_DB_EXPECT", "") or "{}")
pytestmark = pytest.mark.skipif(not REAL_DB, reason="set TCC_REAL_DB to a real run database to enable this cross-check")


def _files(db: Path) -> dict[str, tuple[int, str]]:
    """Size and hash of the database file and of its WAL (the -shm index is not evidence)."""
    out = {}
    for path in (db, Path(str(db) + "-wal")):
        if path.exists():
            out[path.name] = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    return out


def _queue(db: Path) -> list[tuple[Any, ...]]:
    """Lease-relevant columns of every job, or an empty list when the database has no queue."""
    with open_read_model(db) as rm:
        return [
            (
                j["job_id"],
                j["state"],
                j["lease_owner"],
                j["lease_expires_at"],
                j["heartbeat_count"],
                j["claim_count"],
                j["cancel_requested"],
            )
            for j in rm.jobs()
        ]


@pytest.fixture
def db() -> Path:
    path = Path(REAL_DB)
    assert path.is_file(), f"TCC_REAL_DB does not name a file: {path}"
    return path


def test_read_model_shows_real_measurements_and_changes_nothing(db: Path) -> None:
    before, queue_before = _files(db), _queue(db)
    with open_read_model(db) as rm:
        assert rm._conn.execute("PRAGMA query_only").fetchone()[0] == 1
        runs = rm.runs()
        assert runs, "the database has no run"
        totals = {"valid": 0, "synthetic": 0, "generations": 0, "checkpoints": 0, "fronts": 0}
        for run in runs:
            rows = rm.evaluations(run["run_id"])
            progress = rm.run_progress(run["run_id"])
            assert len(rows) == sum(progress["attempts"].values())
            for row in rows:
                values = [row[k] for k in ("e", "l_ms", "v_mb", "s_mb")]
                if row["status"] == "valid":
                    assert all(isinstance(v, float) and math.isfinite(v) for v in values), row["attempt_id"]
                    assert 0 <= row["e"] <= 1 and row["l_ms"] > 0 and row["v_mb"] >= 0 and row["s_mb"] > 0
                else:
                    assert values == [None] * 4, row["attempt_id"]  # never a zero in place of a measurement
            totals["valid"] += sum(1 for r in rows if r["status"] == "valid")
            totals["synthetic"] += sum(1 for r in rows if r["synthetic"])
            totals["generations"] += progress["generations"]
            totals["checkpoints"] += len(rm.checkpoints(run["run_id"]))
            totals["fronts"] += len(rm.fronts(run["run_id"]))
        assert totals["synthetic"] == EXPECT.get("synthetic", 0), "a real run database must not hold synthetic results"
        for key in ("valid", "generations", "checkpoints", "fronts"):
            if key in EXPECT:
                assert totals[key] == EXPECT[key], key
        if "git_sha" in EXPECT:
            assert [r["git_sha"] for r in runs] == [EXPECT["git_sha"]] * len(runs)
    assert _files(db) == before and _queue(db) == queue_before


def test_page_data_layer_lists_every_real_evaluation_as_a_measurement(db: Path) -> None:
    before = _files(db)
    snapshot = page.load_snapshot(db)
    assert [tuple(r) for r in page.run_rows(snapshot)] == [page.RUN_COLUMNS] * len(snapshot.runs)
    for run in snapshot.runs:
        assert not snapshot.is_synthetic_run(run["run_id"])
        view = page.load_run(db, run["run_id"])
        assert not view.synthetic_run and view.synthetic_evaluations == ()
        table = page.evaluation_rows(view.real_evaluations, front_ids=view.latest_front_ids)
        assert len(table) == sum(view.progress["attempts"].values())
        assert all("SINTETICO" not in " ".join(map(str, row.values())) for row in table)
        measured = [row for row in table if row["status"] == "valid"]
        assert all(row[c] is not None for row in measured for c in ("E_erro_top1", "L_ms", "V_MB", "S_MB"))
        assert len(page.generation_rows(view.real_evaluations)) == view.progress["generations"]
        assert len(page.checkpoint_rows(view.checkpoints)) == view.progress["checkpoints"]
        assert len(page.front_rows(view.fronts)) == view.progress["front_snapshots"]
    status = demo_status(db)
    text = format_status(status)
    assert "SYNTHETIC" not in text and all(r["run_id"] in text for r in snapshot.runs)
    assert _files(db) == before


@pytest.mark.asyncio
async def test_tui_read_only_status_lists_the_real_run(db: Path, tmp_path: Path) -> None:
    before, queue_before = _files(db), _queue(db)
    app = tui.DemoTui(db, tmp_path / "unused-artifact-root", read_only=True, refresh_s=0.3)
    async with app.run_test(size=(150, 46)) as pilot:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not (app.status and app.query_one("#evaluations", DataTable).row_count):
            await pilot.pause(0.1)
        assert app.status is not None and app.read_error is None, app.last_message
        table = app.query_one("#evaluations", DataTable)
        rows = [[str(c) for c in table.get_row_at(i)] for i in range(table.row_count)]
        assert rows and len(rows) == sum(app.selected["progress"]["attempts"].values())
        assert all("SINTETICO" not in row[-1] and tui.HIDDEN not in row for row in rows)
        assert all(row[3] != "-" for row in rows if row[2] == "valid")  # E is shown for every valid attempt
        title = str(app.query_one("#evaluations-title", Static).content)
        assert "SINTETIC" not in title.upper()
        assert "sistema: " in str(app.query_one("#detail", Static).content)
        for key in ("p", "s", "c"):  # the writing keys are disabled in this mode
            await pilot.press(key)
            await pilot.pause(0.1)
            assert app.last_message == f"[info] {tui.READ_ONLY_REFUSED}"
        assert app.launched == []
    assert _files(db) == before and _queue(db) == queue_before
    assert not (tmp_path / "unused-artifact-root").exists()
