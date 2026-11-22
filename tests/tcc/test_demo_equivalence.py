"""Gate 3 of TCC-05: the TUI and the CLI produce equivalent records (CPU only).

The same spec is run twice, in separate databases: once driven through the
headless TUI (prepare, confirm, detached worker) and once through
`cnnlab-tcc run`. Everything the read model returns is then compared, field
by field. Only the paths listed in `MAY_DIFFER` are allowed to differ, and
each has its reason next to it; anything else is a real difference.

Both runs use the synthetic test wiring, opted into explicitly. Its values
are stand-ins labeled `quality.synthetic`: equality here proves that the two
interfaces record the same thing, and says nothing about measurements.
The synthetic runner records no front, so fronts are compared but empty.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import cnnlab.tcc.demo_tui as tui
from cnnlab.tcc.operation import launcher
from cnnlab.tcc.operation.demo import DemoRunSpec
from cnnlab.tcc.storage import ReadModel

FACTORY = "cnnlab.tcc.operation.testing:synthetic_demo_wiring"
NO_GPU = {"CUDA_VISIBLE_DEVICES": "-1"}
N_CANDIDATES = 3

# Normalized path (list indexes and ids removed) -> why the two runs may differ there.
MAY_DIFFER = {
    "runs[].created_at": "clock",
    "jobs[].created_at": "clock",
    "jobs[].finished_at": "clock",
    "jobs[].heartbeat_at": "clock",
    "jobs[].heartbeat_count": "how many heartbeats fit in the run depends on timing",
    "events[].recorded_at": "clock",
    "events[].worker_id": "each interface names its own worker",
    "events[].detail.pid": "another process",
    "events[].detail.expires_at": "clock (lease deadline)",
    "evaluations[].started_at": "clock",
    "evaluations[].committed_at": "clock",
    "checkpoints[].created_at": "clock",
    "fronts[].created_at": "clock",
    "artifacts[].created_at": "clock",
    "progress.latest_checkpoint.created_at": "clock",
}
# Ids carry a hash of the plan, which includes the code version. Both runs happen on the
# same checkout, so here even the ids must be equal; across checkouts they would differ.


def snapshot(db: Path) -> dict[str, Any]:
    """Everything the read model returns about the database, as plain JSON."""
    with ReadModel(db) as rm:
        runs, jobs = rm.runs(), rm.jobs()
        out: dict[str, Any] = {
            "schema_version": rm.schema_version(),
            "runs": runs,
            "jobs": jobs,
            "artifacts": rm.artifacts(),
            "events": [event for job in jobs for event in rm.job_events(job["job_id"])],
        }
        [run] = runs
        run_id = run["run_id"]
        out["progress"] = rm.run_progress(run_id)
        for name, rows in (
            ("evaluations", rm.evaluations(run_id)),
            ("genomes", rm.genomes(run_id)),
            ("checkpoints", rm.checkpoints(run_id)),
            ("fronts", rm.fronts(run_id)),
        ):
            out[name] = rows
    return json.loads(json.dumps(out, default=str))


def differences(a: Any, b: Any, path: str = "") -> list[tuple[str, str]]:
    """(normalized path, exact path with both values) of every leaf that differs."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            where = f"{path}.{key}" if path else key
            if key not in a or key not in b:
                out.append(
                    (re.sub(r"\[\d+\]", "[]", where), f"{where}: only in the {'CLI' if key not in a else 'TUI'} run")
                )
            else:
                out += differences(a[key], b[key], where)
        return out
    if isinstance(a, list) and isinstance(b, list):
        out = [] if len(a) == len(b) else [(re.sub(r"\[\d+\]", "[]", path), f"{path}: {len(a)} vs {len(b)} rows")]
        for i, (x, y) in enumerate(zip(a, b)):
            out += differences(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [(re.sub(r"\[\d+\]", "[]", path), f"{path}: {a!r} (TUI) vs {b!r} (CLI)")]


async def _until(pilot: Any, condition: Callable[[], bool], what: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await pilot.pause(0.1)
        if condition():
            return
    raise AssertionError(f"timed out waiting for: {what} (last message: {pilot.app.last_message!r})")


async def _run_by_tui(db: Path, art: Path, spec: Path) -> None:
    app = tui.DemoTui(
        db,
        art,
        spec_path=spec,
        allow_synthetic=True,
        worker_id="by-tui",
        lease_s=30.0,
        heartbeat_s=0.2,
        refresh_s=0.2,
        extra_env=NO_GPU,
    )
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.press("p")
        await _until(
            pilot, lambda: not app.busy and bool(app.selected) and app.selected["state"] == "queued", "prepared"
        )
        await pilot.press("s")
        await pilot.pause(0.2)
        assert isinstance(app.screen, tui.ConfirmScreen)
        await pilot.click("#confirm")
        await _until(pilot, lambda: len(app.launched) == 1, "worker spawned")
        assert app.launched[0].process.wait(timeout=60) == 0
        await _until(pilot, lambda: bool(app.selected) and app.selected["state"] == "done", "job done")


def _run_by_cli(db: Path, art: Path, spec: Path) -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "cnnlab.tcc.operation.cli",
            "run",
            "--db",
            str(db),
            "--artifacts",
            str(art),
            "--spec",
            str(spec),
            "--allow-synthetic",
            "--worker-id",
            "by-cli",
            "--lease-s",
            "30",
            "--heartbeat-s",
            "0.2",
        ],
        env=launcher.cli_environment(NO_GPU),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert "ended done" in done.stdout


@pytest.mark.asyncio
async def test_tui_and_cli_produce_equivalent_records(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    payload: dict[str, Any] = {
        "method": "joint",
        "seed_block": 0,
        "search_seed": 11,
        "train_seed": 22,
        "calibration_seed": 33,
        "budget_gpu_s": 3600.0,
        "code_label": "equivalence",
        "author": "test-suite",
        "factory": FACTORY,
        "factory_args": {"n_candidates": N_CANDIDATES, "sleep_s": 0.05},
    }
    spec.write_text(json.dumps(DemoRunSpec.from_mapping(payload).to_payload()), encoding="utf-8")
    tui_db, tui_art = tmp_path / "tui" / "db" / "tcc.sqlite3", tmp_path / "tui" / "artifacts"
    cli_db, cli_art = tmp_path / "cli" / "db" / "tcc.sqlite3", tmp_path / "cli" / "artifacts"
    await _run_by_tui(tui_db, tui_art, spec)
    _run_by_cli(cli_db, cli_art, spec)

    by_tui, by_cli = snapshot(tui_db), snapshot(cli_db)
    found = differences(by_tui, by_cli)
    unexpected = [exact for normalized, exact in found if normalized not in MAY_DIFFER]
    assert unexpected == [], "the TUI and the CLI recorded different things:\n" + "\n".join(unexpected)
    # The comparison is not vacuous: what must differ by construction does differ.
    assert {"events[].worker_id", "events[].detail.pid", "jobs[].created_at"} <= {n for n, _ in found}

    # What was compared, stated: the run is complete and every table that should have rows has them.
    for record in (by_tui, by_cli):
        [job] = record["jobs"]
        assert job["state"] == "done" and job["claim_count"] == 1 and job["options"]["synthetic_evaluator"] is True
        assert [e["kind"] for e in record["events"]] == ["enqueued", "claimed", "done"]
        assert record["progress"]["attempts"] == {"valid": N_CANDIDATES}
        assert [r["attempt_id"] for r in record["evaluations"]] == [f"c{i}.a1" for i in range(N_CANDIDATES)]
        assert all(r["synthetic"] and r["status"] == "valid" and r["e"] is not None for r in record["evaluations"])
        assert len(record["genomes"]) == N_CANDIDATES and len(record["artifacts"]) == N_CANDIDATES
        assert len(record["checkpoints"]) >= 1
        assert record["fronts"] == []  # the synthetic runner records none: fronts are not exercised here
    assert by_tui["runs"][0]["run_id"] == by_cli["runs"][0]["run_id"]  # same checkout: same plan, same ids
    # Files on disk, not only the catalog: same relative paths and same bytes.
    files = [
        {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        for root in (tui_art, cli_art)
    ]
    assert files[0] == files[1] and len(files[0]) == N_CANDIDATES
