"""Textual TUI of the pilot demo, driven headless (CPU only).

Jobs here use the synthetic test wiring, always opted into explicitly; the
default (no opt-in) is checked to refuse it. Every stored result is labeled
synthetic. Rows used to check the formatting of measured values are in-memory
dicts in the read-model shape, never stored as results.
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import DataTable, Input, Static

import cnnlab.tcc.demo_tui as tui
from cnnlab.tcc.operation import launcher
from cnnlab.tcc.operation.demo import DemoRunSpec
from cnnlab.tcc.storage import JobQueue, ReadModel, SQLiteRunStore, open_read_model
from cnnlab.tcc.storage.testing import make_v1_copy

FACTORY = "cnnlab.tcc.operation.testing:synthetic_demo_wiring"
NO_GPU = {"CUDA_VISIBLE_DEVICES": ""}
SIZE = (140, 44)


def _write_spec(path: Path, **kw: Any) -> Path:
    base: dict[str, Any] = {
        "method": "joint",
        "seed_block": 0,
        "search_seed": 11,
        "train_seed": 22,
        "calibration_seed": 33,
        "budget_gpu_s": 3600.0,
        "code_label": "tui-test",
        "author": "test-suite",
        "factory": FACTORY,
        "factory_args": {"n_candidates": 3, "sleep_s": 0.05},
    }
    base.update(kw)
    path.write_text(json.dumps(DemoRunSpec.from_mapping(base).to_payload()), encoding="utf-8")
    return path


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "db" / "tcc.sqlite3", tmp_path / "artifacts", _write_spec(tmp_path / "spec.json")


def _app(paths: tuple[Path, Path, Path], *, spec: bool = True, allow_synthetic: bool = False) -> tui.DemoTui:
    db, art, spec_path = paths
    return tui.DemoTui(
        db,
        art,
        spec_path=spec_path if spec else None,
        allow_synthetic=allow_synthetic,
        worker_id="tui-test",
        lease_s=30.0,
        heartbeat_s=0.2,
        refresh_s=0.2,
        extra_env=NO_GPU,
    )


async def _until(pilot: Any, condition: Callable[[], bool], what: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await pilot.pause(0.1)
        if condition():
            return
    raise AssertionError(f"timed out waiting for: {what} (last message: {pilot.app.last_message!r})")


def _rows(app: tui.DemoTui, table_id: str) -> list[list[str]]:
    table = app.query_one(table_id, DataTable)
    return [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]


def _job_state(app: tui.DemoTui) -> str | None:
    return None if app.selected is None else app.selected["state"]


# -- pure formatting ---------------------------------------------------------------------


def _row(**kw: Any) -> dict[str, Any]:
    """In-memory evaluation row (read-model shape); formatting input only."""
    row = {
        "attempt_id": "c0.a1",
        "candidate_id": "c0",
        "generation": 0,
        "status": "valid",
        "e": 0.4125,
        "l_ms": 1.25,
        "v_mb": 12.5,
        "s_mb": 3.75,
        "gpu_seconds": 61.0,
        "cache_hit": False,
        "source_attempt_id": None,
        "recovered": False,
        "failure_stage": None,
        "failure_code": None,
        "synthetic": False,
        "preliminary": False,
    }
    row.update(kw)
    return row


def test_evaluation_cells_show_measurements_and_hide_synthetic_numbers() -> None:
    assert len(tui.EVALUATION_HEADERS) == 10
    assert tui.evaluation_cells(_row(), show_synthetic=False) == (
        "c0.a1",
        "0",
        "valid",
        "0.4125",
        "1.2500",
        "12.500",
        "3.750",
        "61.0",
        tui.NOT_RECORDED,
        "",
    )
    failed = _row(status="invalid", e=None, l_ms=None, v_mb=None, s_mb=None, failure_stage="build", failure_code="oom")
    assert tui.evaluation_cells(failed, show_synthetic=True)[3:7] == ("-", "-", "-", "-")  # never zero
    assert tui.evaluation_cells(failed, show_synthetic=True)[-1] == "build/oom"

    fake = _row(synthetic=True)
    hidden = tui.evaluation_cells(fake, show_synthetic=False)
    assert hidden[3:8] == (tui.HIDDEN,) * 5 and hidden[-1] == "SINTETICO"
    shown = tui.evaluation_cells(fake, show_synthetic=True)
    assert shown[3] == "0.4125 (sint.)" and all(v.endswith("(sint.)") for v in shown[3:8])

    measured = _row(attempt_id="run-x.g002.c0010.a1", candidate_id="run-x.g002.c0010", generation=2, preliminary=True)
    genome = {"blocks": [{"precision": "fp32"}, {"precision": "fp16"}, {"precision": "fp16"}]}
    cells = tui.evaluation_cells(
        measured, show_synthetic=False, run_id="run-x", front_ids=frozenset({"run-x.g002.c0010"}), genome=genome
    )
    assert cells == (
        "g002.c0010.a1",
        "2",
        "valid",
        "0.4125",
        "1.2500",
        "12.500",
        "3.750",
        "61.0",
        "fp32 fp16 fp16",
        "preliminar, frente",
    )
    assert "SINTETICO" not in cells[-1] and "(sint.)" not in " ".join(cells)  # a real row is never labeled synthetic


def test_job_detail_shows_only_what_the_database_records() -> None:
    run = {
        "run_id": "run-x",
        "stage": "pilot",
        "plan_id": "plan-x",
        "system_version": None,
        "git_sha": "6" * 40,
        "code_dirty": None,
        "code_label": None,
        "wiring_label": "wiring/v1",
        "evaluator_label": "evaluator/v1",
        "runner_label": "runner/v2",
    }
    progress = {
        "synthetic_results": 0,
        "generations": 3,
        "checkpoints": 6,
        "front_snapshots": 3,
        "latest_checkpoint": {"generation": 2, "unique_evaluations": 12, "active_gpu_seconds": 383.14},
    }
    entry = tui.entries_of({"jobs": [], "runs_without_job": [{"run": run, "progress": progress}]})[0]
    assert entry["key"] == "run:run-x" and entry["job_id"] is None and entry["state"] == tui.NO_JOB
    fronts = [{"hypervolume": None, "candidate_ids": ["run-x.g000.c0001", "run-x.g002.c0010"]}]
    text = tui.job_detail(entry, fronts=fronts, has_job_queue=False)
    assert text.splitlines() == [
        "sistema: git 666666666666 | wiring/v1 | evaluator/v1 | runner/v2",
        "run run-x | estagio pilot | plano plan-x | piloto exploratorio (nao confirmatorio)",
        "busca: 3 geracoes | 6 checkpoints (ultimo: ger. 2, 12 unicas, 383.1 s GPU) | 3 frentes",
        "frente mais recente (2 membros, HV nao calculado): g000.c0001, g002.c0010",
        "fila: banco sem fila de jobs (esquema v1); sem job, lease nem heartbeat",
    ]
    assert tui.job_detail(None) == "Nenhuma run selecionada."


def _imports(module_file: str) -> set[str]:
    tree = ast.parse(Path(module_file).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    return imported


def test_log_excerpt_and_transition_messages() -> None:
    lines = [
        "# 2026 launched: python -m ... very long command",
        "",
        "a" * 200,
        "[t] job_started {}",
        "job x ended done: ok",
    ]
    assert tui.log_excerpt(lines, max_lines=2, width=40) == ["[t] job_started {}", "job x ended done: ok"]
    assert tui.log_excerpt(lines, max_lines=3, width=40)[0] == "a" * 37 + "..."
    job = {"job_id": "j1", "state": "failed", "lease_owner": None, "final_detail": "RuntimeError: boom"}
    assert tui.transition_message(job, "running") == ("erro", "Job j1: running -> failed (RuntimeError: boom)")
    job = {"job_id": "j1", "state": "interrupted", "lease_owner": None, "final_detail": None}
    level, text = tui.transition_message(job, "running")
    assert level == "erro" and "lease vencido" in text and "tecle s para retomar" in text
    job = {"job_id": "j1", "state": "running", "lease_owner": "gauss", "final_detail": None}
    assert tui.transition_message(job, "queued") == ("ok", "Job j1: queued -> running (worker gauss)")


def test_module_never_imports_science_nor_a_writer() -> None:
    imported = _imports(tui.__file__)
    assert {m for m in imported if m.startswith("cnnlab")} == {
        "cnnlab.tcc.operation",
        "cnnlab.tcc.storage.errors",
        "cnnlab.tcc.storage.readmodel",
        "cnnlab.tcc.demo_queue",  # the Fila tab, imported lazily when --queue-dir is given
    }
    assert not {m for m in imported if m.split(".")[0] in ("torch", "sqlite3", "subprocess")}
    for module in _imports(launcher.__file__):  # what the TUI drives stays free of science too
        assert not any(part in ("science", "model_pipeline", "torch") for part in module.split(".")), module


def test_build_app_needs_explicit_paths_and_has_no_default_wiring(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        tui.build_app([])
    with pytest.raises(SystemExit):
        tui.build_app(["--db", str(tmp_path / "art" / "db.sqlite3"), "--artifacts", str(tmp_path / "art")])
    app = tui.build_app(["--db", str(tmp_path / "db" / "t.sqlite3"), "--artifacts", str(tmp_path / "art")])
    assert app.allow_synthetic is False and app.initial_spec == "" and app.show_synthetic is False
    assert (
        tui.build_app(
            ["--db", str(tmp_path / "d" / "t.sqlite3"), "--artifacts", str(tmp_path / "a"), "--allow-synthetic"]
        ).allow_synthetic
        is True
    )


# -- headless app ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_state_explains_the_next_step(paths: tuple[Path, Path, Path]) -> None:
    db, _art, _spec = paths
    app = _app(paths, spec=False)
    async with app.run_test(size=SIZE) as pilot:
        await _until(pilot, lambda: "Banco ainda nao existe" in app.last_message, "empty state message")
        assert app.last_message.startswith("[info]") and "tecle p" in app.last_message
        assert _rows(app, "#jobs") == [] and _rows(app, "#evaluations") == []

        await pilot.press("s")
        assert app.last_message.startswith("[erro] Informe o caminho do spec")
        assert isinstance(app.focused, Input)  # the field to fix gets the focus
        await pilot.press("escape")
        assert isinstance(app.focused, DataTable)
        await pilot.press("c")
        assert app.last_message == "[erro] Nenhum job selecionado para cancelar."
        await pilot.press("p")
        assert app.last_message.startswith("[erro] Informe o caminho do spec")
    assert not db.exists() and app.launched == []  # watching and failing never create a database


@pytest.mark.asyncio
async def test_invalid_spec_and_synthetic_wiring_are_refused_by_default(
    paths: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    db, _art, _spec = paths
    app = _app(paths)  # spec points to the synthetic test wiring; no --allow-synthetic
    async with app.run_test(size=SIZE) as pilot:
        await _until(pilot, lambda: "Banco ainda nao existe" in app.last_message, "first read")
        for key in ("p", "s"):
            await pilot.press(key)
            await pilot.pause(0.2)
            assert app.last_message.startswith("[erro] Fiacao SINTETICA recusada"), app.last_message
            assert not isinstance(app.screen, tui.ConfirmScreen)

        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        app.query_one("#spec", Input).value = str(broken)
        await pilot.press("p")
        assert app.last_message.startswith("[erro] Spec invalido")
        app.query_one("#spec", Input).value = str(tmp_path / "missing.json")
        await pilot.press("p")
        assert "does not exist" in app.last_message
    assert not db.exists() and app.launched == []


@pytest.mark.asyncio
async def test_prepare_start_watch_with_explicit_synthetic_opt_in(paths: tuple[Path, Path, Path]) -> None:
    db, _art, _spec = paths
    app = _app(paths, allow_synthetic=True)
    async with app.run_test(size=SIZE) as pilot:
        assert "--allow-synthetic ATIVO" in str(app.query_one("#banner", Static).content)
        await pilot.press("p")
        assert app.busy and app.last_message.startswith("[info] Preparando run")
        await pilot.press("p")
        assert app.last_message == "[info] Ja existe um preparo em andamento; aguarde."
        await _until(pilot, lambda: not app.busy and _job_state(app) == "queued", "prepared job in the table")
        assert app.last_message.startswith("[ok] prepared run demo-joint-b0-")
        [job_row] = _rows(app, "#jobs")
        assert len(job_row) == len(tui.JOB_HEADERS)
        assert job_row[1] == "queued" and job_row[3] == "SINTETICO (teste)"
        assert job_row[4].startswith(tui.HIDDEN) and job_row[5] == tui.HIDDEN
        detail = str(app.query_one("#detail", Static).content)
        assert "cnnlab-tcc 0.1.0" in detail and "pre-release" in detail and "piloto exploratorio" in detail

        await pilot.press("s")
        await pilot.pause(0.2)
        assert isinstance(app.screen, tui.ConfirmScreen)
        await pilot.press("escape")  # backing out starts nothing
        await pilot.pause(0.2)
        assert app.launched == [] and app.last_message == "[info] Inicio desistido."

        await pilot.press("s")
        await pilot.pause(0.2)
        await pilot.click("#confirm")
        await _until(pilot, lambda: len(app.launched) == 1, "worker spawned")
        assert any(m.startswith("[ok] Worker iniciado (pid ") for m in app.messages)
        await _until(pilot, lambda: _job_state(app) == "done", "job done")
        assert app.last_message.startswith("[ok] Job job-demo-joint-b0-")
        assert app.last_message.endswith("-> done (runner completed)")
        await _until(pilot, lambda: len(_rows(app, "#evaluations")) == 3, "evaluations listed")
        assert app.launched[0].process.wait(timeout=30) == 0

        hidden = _rows(app, "#evaluations")
        assert all(r[2] == "valid" and r[3:8] == [tui.HIDDEN] * 5 and r[9] == "SINTETICO" for r in hidden)
        assert {r[8] for r in hidden} == {tui.NOT_RECORDED}  # the test genomes record no block
        assert [r[0] for r in hidden] == ["c0.a1", "c1.a1", "c2.a1"] and {r[1] for r in hidden} == {"0"}
        assert "numeros ocultos" in str(app.query_one("#evaluations-title", Static).content)

        await pilot.press("y")
        await _until(pilot, lambda: _rows(app, "#evaluations")[0][3].endswith("(sint.)"), "labeled synthetic values")
        assert "NAO SAO MEDICOES" in str(app.query_one("#evaluations-title", Static).content)
        assert _rows(app, "#jobs")[0][4].startswith("30.0 (sint.)")
        await _until(
            pilot, lambda: "ended done" in str(app.query_one("#worker-log", Static).content), "worker log tail"
        )

        await pilot.press("c")
        assert "ja terminou (done)" in app.last_message
        await pilot.press("s")
        await pilot.pause(0.2)
        assert isinstance(app.screen, tui.ConfirmScreen)  # by spec: the CLI will answer "already done"
        await pilot.press("escape")

    with ReadModel(db) as rm:
        [job] = rm.jobs()
        assert job["state"] == "done" and job["claim_count"] == 1 and job["options"]["synthetic_evaluator"] is True
        assert [e["kind"] for e in rm.job_events(job["job_id"])] == ["enqueued", "claimed", "done"]
        assert all(e["synthetic"] for e in rm.evaluations(job["run_id"]))


@pytest.mark.asyncio
async def test_cancel_needs_a_reason_and_is_recorded(paths: tuple[Path, Path, Path]) -> None:
    db, art, spec = paths
    assert launcher.run_prepare(db, art, spec, allow_synthetic=True, extra_env=NO_GPU).ok
    app = _app(paths, allow_synthetic=True)
    async with app.run_test(size=SIZE) as pilot:
        await _until(pilot, lambda: _job_state(app) == "queued", "queued job shown")
        assert app.last_message.startswith("[ok] 1 job(s) no banco")
        await pilot.press("c")
        await pilot.pause(0.2)
        assert isinstance(app.screen, tui.CancelScreen)
        await pilot.click("#confirm")
        await pilot.pause(0.2)
        assert isinstance(app.screen, tui.CancelScreen)  # still open: empty reason refused
        assert "Informe o motivo" in str(app.screen.query_one("#reason-error", Static).content)
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert app.last_message == "[info] Cancelamento desistido." and _job_state(app) == "queued"

        await pilot.press("c")
        await pilot.pause(0.2)
        app.screen.query_one("#reason", Input).value = "ensaio da demo"
        await pilot.click("#confirm")
        await _until(pilot, lambda: _job_state(app) == "cancelled", "job cancelled")
        await _until(
            pilot, lambda: any("queued -> cancelled (ensaio da demo)" in m for m in app.messages), "transition"
        )
        assert any(m.startswith("[ok] Job ") and m.endswith(" cancelado.") for m in app.messages)
        assert "cancelamento pedido: ensaio da demo" in str(app.query_one("#detail", Static).content)
    with ReadModel(db) as rm:
        [job] = rm.jobs()
        assert job["state"] == "cancelled" and job["cancel_reason"] == "ensaio da demo"


@pytest.mark.asyncio
async def test_start_on_a_cancelled_job_continues_the_same_run(paths: tuple[Path, Path, Path]) -> None:
    db, art, spec = paths
    assert launcher.run_prepare(db, art, spec, allow_synthetic=True, extra_env=NO_GPU).ok
    with ReadModel(db) as rm:
        [job] = rm.jobs()
    launcher.request_cancel(db, art, job["job_id"], reason="ensaio")
    app = _app(paths, allow_synthetic=True)
    async with app.run_test(size=SIZE) as pilot:
        await _until(pilot, lambda: _job_state(app) == "cancelled", "cancelled job shown")
        await pilot.press("s")
        await pilot.pause(0.2)
        assert isinstance(app.screen, tui.ConfirmScreen)
        shown = " ".join(str(w.content) for w in app.screen.query(Static))
        assert f"Continuar a run {job['run_id']} com um job novo" in shown and "terminou cancelled" in shown
        await pilot.click("#confirm")
        await _until(pilot, lambda: len(app.launched) == 1, "worker spawned")
        assert app.launched[0].target == job["run_id"] and "--run-id" in app.launched[0].command
        assert app.launched[0].process.wait(timeout=30) == 0
        await _until(
            pilot, lambda: [j["state"] for j in app.status["jobs"]] == ["cancelled", "done"], "second job done"
        )
    with ReadModel(db) as rm:
        assert [(j["state"], j["run_id"]) for j in rm.jobs()] == [("cancelled", job["run_id"]), ("done", job["run_id"])]
        assert len(rm.runs()) == 1 and len(rm.evaluations(job["run_id"])) == 3


@pytest.mark.asyncio
async def test_start_is_refused_while_another_worker_holds_the_gpu(paths: tuple[Path, Path, Path]) -> None:
    db, art, spec = paths
    assert launcher.run_prepare(db, art, spec, allow_synthetic=True, extra_env=NO_GPU).ok
    with SQLiteRunStore(db, art) as store:
        queue = JobQueue(store)
        lease = queue.claim_next("other-worker", lease_s=120)
        assert lease is not None
        app = _app(paths, allow_synthetic=True)
        async with app.run_test(size=SIZE) as pilot:
            await _until(pilot, lambda: _job_state(app) == "running", "running job shown")
            assert _rows(app, "#jobs")[0][6] == "other-worker vivo"
            await pilot.press("s")
            await pilot.pause(0.2)
            assert "Um job de GPU por vez" in app.last_message and not isinstance(app.screen, tui.ConfirmScreen)
            assert app.launched == []

            await pilot.press("c")  # cooperative: only a request while the worker is alive
            await pilot.pause(0.2)
            app.screen.query_one("#reason", Input).value = "parar apos a tentativa"
            await pilot.click("#confirm")
            await _until(pilot, lambda: bool(app.selected and app.selected["cancel_requested"]), "cancel requested")
            assert any("o worker termina a tentativa em curso" in m for m in app.messages)
            assert _job_state(app) == "running" and _rows(app, "#jobs")[0][1] == "running (cancel. pedido)"
        assert queue.is_cancel_requested(lease)
        assert queue.heartbeat(lease).lease_owner == "other-worker"  # the TUI never touched the lease


# -- schema v1 database and read-only mode -------------------------------------------------


@pytest.fixture
def v1_paths(paths: tuple[Path, Path, Path], tmp_path: Path) -> tuple[Path, Path, Path]:
    """A finished (synthetic, opted-in) run copied into the shape of schema v1: no job queue."""
    db, art, spec = paths
    handle = launcher.spawn_worker(db, art, spec_path=spec, allow_synthetic=True, extra_env=NO_GPU)
    assert handle.process.wait(timeout=60) == 0
    return make_v1_copy(db, tmp_path / "v1" / "demo.sqlite"), art, spec


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.asyncio
async def test_read_only_mode_watches_a_v1_database_without_touching_it(v1_paths: tuple[Path, Path, Path]) -> None:
    db, art, spec = v1_paths
    before = _digest(db)
    app = tui.DemoTui(db, art, spec_path=spec, read_only=True, refresh_s=0.2, extra_env=NO_GPU)
    async with app.run_test(size=SIZE) as pilot:
        assert "SOMENTE LEITURA" in str(app.query_one("#banner", Static).content)
        await _until(pilot, lambda: app.selected is not None, "run listed")
        await _until(pilot, lambda: len(_rows(app, "#evaluations")) == 3, "evaluations listed")
        assert "1 run(s) no banco, banco sem fila de jobs (tcc-read/1, esquema v1)" in app.last_message
        [row] = _rows(app, "#jobs")
        assert row[0].startswith("(sem job) run demo-joint-b0-") and row[1] == tui.NO_JOB and row[7] == "-"
        assert app.selected["job_id"] is None and app.selected["key"].startswith("run:")
        detail = str(app.query_one("#detail", Static).content)
        assert "fila: banco sem fila de jobs (esquema v1)" in detail and "busca: 1 geracoes | " in detail

        for key in ("p", "s", "c"):
            await pilot.press(key)
            await pilot.pause(0.2)
            assert app.last_message == f"[info] {tui.READ_ONLY_REFUSED}", key
            assert not isinstance(app.screen, (tui.ConfirmScreen, tui.CancelScreen))
        assert app.launched == [] and not app.busy
    assert _digest(db) == before
    with open_read_model(db) as rm:
        assert rm.schema_version() == 1 and not rm.has_job_queue  # watched, not migrated
    assert not (db.parent / "logs").exists()


@pytest.mark.asyncio
async def test_writing_keys_never_migrate_an_older_database(v1_paths: tuple[Path, Path, Path]) -> None:
    db, art, spec = v1_paths
    before = _digest(db)
    app = tui.DemoTui(db, art, spec_path=spec, allow_synthetic=True, refresh_s=0.2, extra_env=NO_GPU)
    async with app.run_test(size=SIZE) as pilot:
        await _until(pilot, lambda: app.selected is not None, "run listed")
        for key in ("p", "s", "c"):
            await pilot.press(key)
            await pilot.pause(0.2)
            assert app.last_message.startswith("[erro] Este banco tem esquema anterior"), (key, app.last_message)
            assert "Nada foi alterado" in app.last_message
            assert not isinstance(app.screen, (tui.ConfirmScreen, tui.CancelScreen))
        assert app.launched == []
    assert _digest(db) == before
    with open_read_model(db) as rm:
        assert rm.schema_version() == 1
    for refused in (
        lambda: launcher.run_prepare(db, art, spec, allow_synthetic=True, extra_env=NO_GPU),
        lambda: launcher.spawn_worker(db, art, spec_path=spec, allow_synthetic=True, extra_env=NO_GPU),
        lambda: launcher.request_cancel(db, art, "any-job", reason="x"),
    ):
        with pytest.raises(launcher.MigrationNeededError, match="would migrate"):
            refused()
    assert _digest(db) == before
