"""Read-only Streamlit page: data layer, read-only guarantees and headless page tests.

Every result stored in a database here comes from the synthetic wiring and
is labeled `quality.synthetic = true`. The few "measured" rows used to check
how the page formats real results are in-memory dicts in the read-model shape;
they are never stored as results anywhere.

The headless page tests need `streamlit` (not a project dependency yet) and
are skipped without it.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import pytest

import cnnlab.tcc.demo_streamlit as page
from cnnlab.tcc.operation.demo import DemoRunSpec, prepare_demo, start_demo
from cnnlab.tcc.operation.provenance import CodeVersion
from cnnlab.tcc.operation.testing import synthetic_demo_wiring
from cnnlab.tcc.storage import JobQueue, ReadModelError, SQLiteRunStore, open_read_model
from cnnlab.tcc.storage.schema import LATEST_VERSION
from cnnlab.tcc.storage.testing import make_v1_copy

CODE = CodeVersion("ui-test", "c" * 40, False, "3" * 64, (), (), 0)
PAGE_FILE = Path(page.__file__)
HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None
needs_streamlit = pytest.mark.skipif(not HAS_STREAMLIT, reason="streamlit is not installed in this environment")


def _spec(**kw: Any) -> DemoRunSpec:
    base: dict[str, Any] = {
        "method": "joint",
        "seed_block": 0,
        "search_seed": 11,
        "train_seed": 22,
        "calibration_seed": 33,
        "budget_gpu_s": 3600.0,
        "code_label": "ui-test",
        "author": "test-suite",
        "factory": "cnnlab.tcc.operation.testing:synthetic_demo_wiring",
        "factory_args": {"n_candidates": 3, "sleep_s": 0.0},
    }
    base.update(kw)
    return DemoRunSpec.from_mapping(base)


@pytest.fixture
def empty_db(tmp_path: Path) -> Path:
    db = tmp_path / "db" / "tcc.sqlite3"
    with SQLiteRunStore(db, tmp_path / "artifacts"):
        pass
    return db


@pytest.fixture
def synthetic_db(tmp_path: Path) -> Path:
    """A finished run of the synthetic wiring plus a second job still queued."""
    db, art = tmp_path / "db" / "tcc.sqlite3", tmp_path / "artifacts"
    result = start_demo(db, art, _spec(), synthetic_demo_wiring, worker_id="w", code_version=CODE, allow_synthetic=True)
    assert result.outcome is not None and result.outcome.end_state == "done"
    queued = _spec(method="random", seed_block=1)
    with SQLiteRunStore(db, art) as store:
        prepare_demo(store, queued, synthetic_demo_wiring(queued), CODE, allow_synthetic=True)
    return db


def _measured_row(i: int, **kw: Any) -> dict[str, Any]:
    """In-memory row in the read-model shape (formatting input only; never stored)."""
    row = {
        "attempt_id": f"c{i}.a1",
        "candidate_id": f"c{i}",
        "generation": 0,
        "attempt_number": 1,
        "status": "valid",
        "cache_hit": False,
        "source_attempt_id": None,
        "e": 0.41,
        "l_ms": 1.25,
        "v_mb": 12.5,
        "s_mb": 3.75,
        "gpu_seconds": 61.0,
        "failure_stage": None,
        "failure_code": None,
        "recovered": False,
        "synthetic": False,
        "preliminary": False,
        "not_confirmatory": False,
        "phenotype_hash": "p",
        "cache_context_hash": "c",
        "started_at": "2026-09-29T10:00:00.000000Z",
        "committed_at": "2026-09-29T10:01:01.000000Z",
        "commit_seq": i + 1,
    }
    row.update(kw)
    return row


# -- data layer ------------------------------------------------------------------------


def test_snapshot_lists_jobs_runs_and_versions(synthetic_db: Path) -> None:
    snapshot = page.load_snapshot(str(synthetic_db))
    assert snapshot.read_model == "tcc-read/1" and snapshot.schema_version == LATEST_VERSION
    assert [j["state"] for j in snapshot.jobs] == ["done", "queued"]
    runs = page.run_rows(snapshot)
    assert [tuple(r) for r in runs] == [page.RUN_COLUMNS] * 2
    assert all(r["git_sha"] == "c" * 40 and r["codigo_sujo"] == "nao" and r["estagio"] == "pilot" for r in runs)
    assert all("cnnlab-tcc 0.1.0" in r["versao_do_sistema"] and "pre-release" in r["versao_do_sistema"] for r in runs)


def test_synthetic_numbers_are_hidden_unless_asked(synthetic_db: Path) -> None:
    snapshot = page.load_snapshot(synthetic_db)
    hidden = page.job_rows(snapshot, show_synthetic=False)
    assert [tuple(r) for r in hidden] == [page.JOB_COLUMNS] * 2
    assert all(r["avaliador"] == page.SYNTHETIC_LABEL for r in hidden)
    assert all(r["gpu_ativa_s"] == page.HIDDEN and r["validas"] == page.HIDDEN for r in hidden)
    shown = page.job_rows(snapshot, show_synthetic=True)
    assert shown[0]["gpu_ativa_s"] == "30.0" and shown[0]["validas"] == "3" and shown[1]["validas"] == "0"

    view = page.load_run(synthetic_db, snapshot.runs[0]["run_id"])
    assert view.synthetic_run and view.real_evaluations == () and len(view.synthetic_evaluations) == 3
    assert view.environment["manifest"]["system"]["code_version"]["git_sha"] == "c" * 40
    assert [e["kind"] for e in view.job_events] == ["enqueued", "claimed", "done"]
    rows = page.evaluation_rows(view.synthetic_evaluations, synthetic=True)
    assert [tuple(r) for r in rows] == [page.SYNTHETIC_EVALUATION_COLUMNS] * 3
    assert all(isinstance(r["E_SINTETICO"], float) and "E_erro_top1" not in r for r in rows)
    with pytest.raises(ValueError, match="never listed in the same table"):
        page.evaluation_rows(view.synthetic_evaluations)  # a fake row cannot enter the measurement table


def test_measured_rows_keep_units_and_leave_unmeasured_values_empty() -> None:
    rows = page.evaluation_rows(
        [
            _measured_row(0),
            _measured_row(
                1,
                status="invalid",
                e=None,
                l_ms=None,
                v_mb=None,
                s_mb=None,
                failure_stage="build",
                failure_code="engine_build_failed",
            ),
            _measured_row(2, cache_hit=True, source_attempt_id="c0.a1", gpu_seconds=0.0),
        ]
    )
    assert [tuple(r) for r in rows] == [page.EVALUATION_COLUMNS] * 3
    assert (rows[0]["E_erro_top1"], rows[0]["L_ms"], rows[0]["V_MB"], rows[0]["S_MB"]) == (0.41, 1.25, 12.5, 3.75)
    long = page.evaluation_rows(
        [_measured_row(3, e=0.123456789, l_ms=2.000049, v_mb=7.12345, s_mb=1.0006, gpu_seconds=9.96)]
    )
    assert list(long[0].values())[4:9] == [0.1235, 2.0, 7.123, 1.001, 10.0]  # rounded on screen only
    assert (rows[1]["E_erro_top1"], rows[1]["L_ms"], rows[1]["V_MB"], rows[1]["S_MB"]) == (None,) * 4  # never zero
    assert rows[1]["falha"] == "build/engine_build_failed" and rows[2]["cache"] == "sim"
    with pytest.raises(ValueError):
        page.evaluation_rows([_measured_row(0)], synthetic=True)


def test_errors_are_explained_with_a_next_step(tmp_path: Path, empty_db: Path) -> None:
    for bad, code in (
        (None, "empty_path"),
        ("  ", "empty_path"),
        (tmp_path / "x.sqlite3", "not_found"),
        (tmp_path, "not_a_file"),
    ):
        with pytest.raises(ReadModelError) as info:
            page.load_snapshot(bad)  # type: ignore[arg-type]
        assert info.value.code == code
        title, hint = page.explain_error(info.value)
        assert title == page.ERROR_TEXT[code][0] and "Detalhe:" in hint
    assert not (tmp_path / "x.sqlite3").exists()
    assert set(page.ERROR_TEXT) >= {
        "empty_path",
        "not_found",
        "not_a_file",
        "not_sqlite",
        "not_tcc",
        "schema_too_old",
        "schema_too_new",
    }

    snapshot = page.load_snapshot(empty_db)  # valid database, nothing in it yet
    assert snapshot.jobs == () and snapshot.runs == () and page.job_rows(snapshot, show_synthetic=False) == []
    with pytest.raises(ReadModelError) as info:
        page.load_run(empty_db, "no-such-run")
    assert info.value.code == "run_not_found"
    assert page.explain_error(info.value)[0] == "Não foi possível ler o banco"


def test_real_rows_carry_what_the_evaluator_declared_and_front_membership() -> None:
    rows = page.evaluation_rows(
        [_measured_row(0, preliminary=True, not_confirmatory=True), _measured_row(1)],
        front_ids=frozenset({"c0"}),
    )
    assert rows[0]["observacao"] == "preliminar; nao confirmatoria; na frente mais recente"
    assert rows[1]["observacao"] == "" and "SINTETICO" not in " ".join(rows[0])
    assert {r["precisao_por_bloco"] for r in rows} == {page.NOT_RECORDED}  # no genome given: nothing invented
    blocks = [{"precision": "fp32"}, {"precision": "fp16"}]
    with_genome = page.evaluation_rows(
        [_measured_row(0), _measured_row(1)], genomes=[{"candidate_id": "c0", "blocks": blocks}]
    )
    assert [r["precisao_por_bloco"] for r in with_genome] == ["fp32 fp16", page.NOT_RECORDED]
    assert list(page.EVALUATION_COLUMNS).index("precisao_por_bloco") == 9  # right after the measured values

    generations = page.generation_rows([_measured_row(0), _measured_row(1, e=0.2, generation=1)])
    assert [tuple(g) for g in generations] == [page.GENERATION_COLUMNS] * 2
    assert (generations[0]["geracao"], generations[0]["validas"], generations[0]["menor_E"]) == (0, 1, 0.41)
    assert generations[1]["menor_E"] == 0.2 and generations[1]["gpu_s"] == 61.0

    fronts = page.front_rows(
        [
            {
                "snapshot_id": 3,
                "active_gpu_seconds": 383.14,
                "unique_valid_evaluations": 12,
                "hypervolume": None,
                "created_at": "2026-09-29T01:18:49.541157Z",
                "candidate_ids": ["a", "b"],
            },
            {
                "snapshot_id": 4,
                "active_gpu_seconds": 400.0,
                "unique_valid_evaluations": 13,
                "hypervolume": 0.125,
                "created_at": "2026-09-29T01:20:00.000000Z",
                "candidate_ids": [],
            },
        ]
    )
    assert [tuple(f) for f in fronts] == [page.FRONT_COLUMNS] * 2
    assert fronts[0]["hipervolume"] == page.NOT_COMPUTED and fronts[0]["membros"] == 2  # never shown as zero
    assert fronts[1]["hipervolume"] == "0.125" and fronts[0]["candidatas"] == "a, b"


def test_genome_rows_and_scatter_points_show_only_what_is_recorded() -> None:
    genomes = [
        {
            "candidate_id": "run-x.g000.c0001",
            "generation": 0,
            "origin": "initial",
            "n_blocks": 3,
            "stem_channels": 16,
            "stem_precision": "fp32",
            "head_precision": "fp32",
            "activation": "hardswish",
            "head": "gap_linear",
            "repairs": 0,
            "blocks": [
                {"precision": "fp32", "channels": 32},
                {"precision": "fp16", "channels": 64},
                {"precision": "fp16", "channels": 64},
            ],
        },
        {
            "candidate_id": "run-x.g000.c0002",
            "generation": None,
            "origin": None,
            "n_blocks": None,
            "stem_channels": None,
            "stem_precision": None,
            "head_precision": None,
            "activation": None,
            "head": None,
            "repairs": None,
            "blocks": [],
        },
    ]
    rows = page.genome_rows(genomes)
    assert [tuple(r) for r in rows] == [page.GENOME_COLUMNS] * 2
    assert rows[0]["precisao_por_bloco"] == "fp32 fp16 fp16" and rows[0]["canais_por_bloco"] == "32 64 64"
    assert rows[0]["op_por_bloco"] == " ".join([page.NOT_RECORDED] * 3)  # a field the genome does not record
    assert set(rows[1].values()) == {"run-x.g000.c0002", page.NOT_RECORDED}
    assert all(isinstance(v, str) for r in rows for v in r.values())

    evaluations = [
        _measured_row(1, candidate_id="run-x.g000.c0001", e=0.8122, l_ms=0.2321199, v_mb=0.972288, s_mb=0.648724),
        _measured_row(2, candidate_id="run-x.g000.c0002"),
        _measured_row(3, status="invalid", e=None, l_ms=None, v_mb=None, s_mb=None),
        _measured_row(4, status="interrupted", e=None, l_ms=None, v_mb=None, s_mb=None),
    ]
    points = page.scatter_points(
        evaluations, front_ids=frozenset({"run-x.g000.c0001"}), genomes=genomes, run_id="run-x"
    )
    assert [p["candidata"] for p in points] == ["g000.c0001", "g000.c0002"]  # only measured attempts are points
    assert points[0] == {
        "candidata": "g000.c0001",
        "geracao": 0,
        "E": 0.8122,
        "L_ms": 0.2321199,
        "V_MB": 0.972288,
        "S_MB": 0.648724,
        "frente": page.IN_FRONT,
        "precisao_por_bloco": "fp32 fp16 fp16",
    }
    assert points[1]["frente"] == page.OFF_FRONT and points[1]["precisao_por_bloco"] == page.NOT_RECORDED
    assert page.scatter_points([]) == [] and set(page.SCATTER_AXES.values()) <= set(points[0])
    assert len(page.FRONT_COLORS) == len(page.FRONT_SHAPES) == len(page.FRONT_DOMAIN) == 2  # color and shape


@pytest.fixture
def v1_db(synthetic_db: Path, tmp_path: Path) -> Path:
    return make_v1_copy(synthetic_db, tmp_path / "v1" / "demo.sqlite")


def test_schema_v1_database_loads_without_jobs_and_is_not_migrated(v1_db: Path) -> None:
    digest = hashlib.sha256(v1_db.read_bytes()).hexdigest()
    snapshot = page.load_snapshot(v1_db)
    assert snapshot.schema_version == 1 and snapshot.has_job_queue is False and snapshot.jobs == ()
    assert len(snapshot.runs) == 2 and page.job_rows(snapshot, show_synthetic=True) == []
    runs = page.run_rows(snapshot)
    assert [tuple(r) for r in runs] == [page.RUN_COLUMNS] * 2
    assert runs[0]["git_sha"] == "c" * 40 and runs[0]["plano_confirmatorio"] == "nao"
    view = page.load_run(v1_db, snapshot.runs[0]["run_id"])
    assert view.has_job_queue is False and view.job_events == () and len(view.synthetic_evaluations) == 3
    assert len(view.checkpoints) >= 1 and page.checkpoint_rows(view.checkpoints)[0].keys() == set(
        page.CHECKPOINT_COLUMNS
    )
    assert hashlib.sha256(v1_db.read_bytes()).hexdigest() == digest
    with open_read_model(v1_db) as rm:
        assert rm.schema_version() == 1


# -- read-only guarantees --------------------------------------------------------------


def test_page_module_has_no_way_to_write() -> None:
    tree = ast.parse(PAGE_FILE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    project = {m for m in imported if m.startswith("cnnlab")}
    assert project == {
        "cnnlab",
        "cnnlab.tcc.demo_analysis",
        "cnnlab.tcc.demo_analysis.render_analysis",
        "cnnlab.tcc.demo_glossary",
        "cnnlab.tcc.demo_glossary.render_glossary",
        "cnnlab.tcc.demo_methods",
        "cnnlab.tcc.demo_methods.render_methods",
        "cnnlab.tcc.storage.readmodel",
        "cnnlab.tcc.storage.readmodel.READ_MODEL_VERSION",
        "cnnlab.tcc.storage.readmodel.ReadModelError",
        "cnnlab.tcc.storage.readmodel.generation_summary",
        "cnnlab.tcc.storage.readmodel.open_read_model",
        "cnnlab.tcc.storage.readmodel.per_block",
        "cnnlab.tcc.storage.readmodel.version_text",
    }
    assert "sqlite3" not in imported  # no connection of its own, hence no SQL of its own
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    forbidden = {"SQLiteRunStore", "JobQueue", "open_writer", "execute", "executescript", "executemany", "text_area"}
    assert not names & forbidden


def test_reading_does_not_change_the_database_nor_the_lease(synthetic_db: Path, tmp_path: Path) -> None:
    with SQLiteRunStore(synthetic_db, tmp_path / "artifacts") as store:
        lease = JobQueue(store).claim_next("worker", lease_s=60)
        assert lease is not None
        before_job = JobQueue(store).get(lease.job_id)
        store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        digest = hashlib.sha256(synthetic_db.read_bytes()).hexdigest()
        for _ in range(3):  # the page reruns on every interaction
            snapshot = page.load_snapshot(synthetic_db)
            for run in snapshot.runs:
                page.load_run(synthetic_db, run["run_id"])
        assert snapshot.jobs[1]["state"] == "running" and snapshot.jobs[1]["lease_live"] is True
        assert hashlib.sha256(synthetic_db.read_bytes()).hexdigest() == digest
        assert JobQueue(store).get(lease.job_id) == before_job  # no heartbeat, no claim, no cancel
        assert JobQueue(store).heartbeat(lease).heartbeat_count == 1  # the worker goes on normally


def test_module_imports_without_streamlit_and_reports_how_to_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import builtins

    real_import = builtins.__import__

    def no_streamlit(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "streamlit" or name.startswith("streamlit."):
            raise ModuleNotFoundError("No module named 'streamlit'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_streamlit)
    assert page.main(["--db", "x.sqlite3"]) == 2
    assert page.STREAMLIT_INSTALL_HINT in capsys.readouterr().err


def test_launch_command_keeps_the_page_arguments_after_the_separator() -> None:
    command = page.launch_command("D:/demo/tcc.sqlite3", ["--server.port", "8502"])
    assert command[1:4] == ["-m", "streamlit", "run"] and Path(command[4]) == PAGE_FILE.resolve()
    assert command[5:] == [
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8502",
        "--",
        "--db",
        "D:/demo/tcc.sqlite3",
    ]  # local only unless the operator decides otherwise
    assert page.launch_command("x", ["--server.address", "0.0.0.0"])[5:7] == ["--server.address", "0.0.0.0"]
    assert page.launch_command("x", ["--server.address=0.0.0.0"]).count("127.0.0.1") == 0
    assert page.launch_command("")[-1] == "--"  # no hidden default database


# -- headless page (needs streamlit) ------------------------------------------------------


def _app() -> Any:
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(PAGE_FILE), default_timeout=60)


def _charts(at: Any) -> list[Any]:
    return list(at.get("vega_lite_chart"))


def _texts(elements: Any) -> str:
    return "\n".join(str(e.value) for e in elements)


@needs_streamlit
def test_page_empty_state_and_bad_path(tmp_path: Path) -> None:
    at = _app().run()
    assert not at.exception
    assert "Informe o caminho do banco" in _texts(at.info)
    assert len(at.text_input) == 1 and len(at.dataframe) == 0  # one input: the path; no SQL box

    missing = tmp_path / "missing.sqlite3"
    at.text_input(key="db_path").set_value(str(missing)).run()
    assert not at.exception
    assert "Arquivo não encontrado" in _texts(at.error) and not missing.exists()


@needs_streamlit
def test_page_with_a_database_without_jobs(empty_db: Path) -> None:
    at = _app()
    at.run()
    at.text_input(key="db_path").set_value(str(empty_db)).run()
    assert not at.exception and not at.error
    info = _texts(at.info)
    assert "Nenhum job na fila" in info and "Nenhuma run registrada" in info


@needs_streamlit
def test_page_hides_synthetic_numbers_until_the_operator_asks(synthetic_db: Path) -> None:
    digest = hashlib.sha256(synthetic_db.read_bytes()).hexdigest()
    at = _app()
    at.run()
    at.text_input(key="db_path").set_value(str(synthetic_db)).run()
    assert not at.exception and not at.error
    jobs, runs = at.dataframe[0].value, at.dataframe[1].value
    assert list(jobs["estado"]) == ["done", "queued"]
    assert set(jobs["avaliador"]) == {page.SYNTHETIC_LABEL}
    assert set(jobs["gpu_ativa_s"]) == {page.HIDDEN} and set(jobs["validas"]) == {page.HIDDEN}
    assert set(runs["git_sha"]) == {"c" * 40} and runs["versao_do_sistema"].str.contains("pre-release").all()
    assert "SINTÉTICO" in _texts(at.warning)
    assert "ocultos por serem sintéticos" in _texts(at.info)
    assert len(at.metric) == 0
    shown = [list(d.value.columns) for d in at.dataframe]
    assert list(page.EVALUATION_COLUMNS) not in shown and list(page.SYNTHETIC_EVALUATION_COLUMNS) not in shown
    assert _charts(at) == [] and len(at.radio) == 0  # no plot of hidden synthetic numbers
    assert "Genoma das candidatas" in _texts(at.markdown)  # newest run: queued, no candidate yet

    at.selectbox(key="run_id").set_value(jobs["run"][0])
    at.checkbox(key="show_synthetic").check().run()
    assert not at.exception and not at.error
    tables = {tuple(d.value.columns): d.value for d in at.dataframe}
    synthetic = tables[page.SYNTHETIC_EVALUATION_COLUMNS]
    assert len(synthetic) == 3 and synthetic["E_SINTETICO"].between(0, 1).all()
    assert page.EVALUATION_COLUMNS not in tables  # still no "measurement" table
    assert len(_charts(at)) == 1 and "Dispersão das medições válidas (SINTÉTICO)" in _texts(at.markdown)
    assert len(tables[page.GENOME_COLUMNS]) == 3 and "Genoma das candidatas (SINTÉTICO)" in _texts(at.markdown)
    assert "NÃO são medições" in _texts(at.markdown)
    assert all("SINTÉTICO" in m.label for m in at.metric if m.label != "Orçamento GPU (s)")
    assert hashlib.sha256(synthetic_db.read_bytes()).hexdigest() == digest


@needs_streamlit
def test_page_lists_measurements_when_rows_are_not_synthetic(synthetic_db: Path) -> None:
    """Formatting of real results, driven by in-memory rows (nothing unlabeled is stored)."""
    from streamlit.testing.v1 import AppTest

    def script(db: str) -> None:
        import dataclasses

        import streamlit as st

        import cnnlab.tcc.demo_streamlit as page

        load_run = page.load_run

        def measured(db_path: str, run_id: str) -> object:
            view = load_run(db_path, run_id)
            rows = tuple(
                {**r, "synthetic": False, "e": 0.25 + i / 10, "l_ms": 1.5, "v_mb": 20.0, "s_mb": 4.0}
                for i, r in enumerate(view.synthetic_evaluations)
            )
            progress = {**view.progress, "synthetic_results": 0}
            return dataclasses.replace(
                view, synthetic_run=False, real_evaluations=rows, synthetic_evaluations=(), progress=progress
            )

        page.load_run = measured
        try:
            page.render(st, db)
        finally:
            page.load_run = load_run

    at = AppTest.from_function(script, args=(str(synthetic_db),), default_timeout=60).run()
    assert not at.exception and not at.error
    assert "Nenhuma tentativa registrada ainda" in _texts(at.info)  # newest run (still queued) is the default
    finished = page.load_snapshot(synthetic_db).runs[0]["run_id"]
    at.selectbox(key="run_id").set_value(finished).run()
    assert not at.exception and not at.error
    tables = {tuple(d.value.columns): d.value for d in at.dataframe}
    measured = tables[page.EVALUATION_COLUMNS]
    assert list(measured["E_erro_top1"]) == [0.25, 0.35, 0.45] and set(measured["L_ms"]) == {1.5}
    assert page.SYNTHETIC_EVALUATION_COLUMNS not in tables
    assert list(tables[page.GENERATION_COLUMNS]["menor_E"]) == [0.25]  # summary of the measured rows
    assert len(tables[page.GENOME_COLUMNS]) == 3
    assert len(_charts(at)) == 1 and [r.label for r in at.radio] == ["Eixo X"]
    assert list(at.radio[0].options) == ["L (ms)", "V (MB)", "S (MB)"]
    at.radio[0].set_value("S (MB)").run()
    assert not at.exception and "S_MB" in str(_charts(at)[0].proto)
    headings = _texts(at.markdown)
    assert "**Gerações**" in headings and "SINTÉTICO" not in headings and not at.warning
    assert [m.label for m in at.metric][0] == "GPU ativa (s)"
    assert "Run piloto exploratória" in _texts(at.caption)


@needs_streamlit
def test_page_with_a_real_job_next_to_a_synthetic_one(synthetic_db: Path, tmp_path: Path) -> None:
    """A job prepared with a wiring that declares itself real, never started: no result is stored for it."""
    import dataclasses

    spec = _spec(method="mono", seed_block=2)
    with SQLiteRunStore(synthetic_db, tmp_path / "artifacts") as store:
        prepare_demo(store, spec, dataclasses.replace(synthetic_demo_wiring(spec), synthetic=False), CODE)
    at = _app()
    at.run()
    at.text_input(key="db_path").set_value(str(synthetic_db)).run()
    assert not at.exception and not at.error
    jobs = at.dataframe[0].value
    assert list(jobs["avaliador"]) == [page.SYNTHETIC_LABEL, page.SYNTHETIC_LABEL, page.REAL_LABEL]
    assert list(jobs["gpu_ativa_s"]) == [page.HIDDEN, page.HIDDEN, "0.0"]  # real numbers shown, fake ones hidden
    assert list(jobs["validas"]) == [page.HIDDEN, page.HIDDEN, "0"]
    assert "Nenhuma tentativa registrada ainda" in _texts(at.info)  # newest run is the real one, still queued
    assert not at.warning and [m.label for m in at.metric][0] == "GPU ativa (s)"


@needs_streamlit
def test_page_shows_a_v1_database_with_generations_checkpoints_and_fronts(v1_db: Path) -> None:
    digest = hashlib.sha256(v1_db.read_bytes()).hexdigest()
    at = _app()
    at.run()
    at.text_input(key="db_path").set_value(str(v1_db)).run()
    assert not at.exception and not at.error
    assert "não tem fila de jobs (esquema v1)" in _texts(at.info)
    assert "esquema v1" in _texts(at.caption)
    runs = at.dataframe[0].value  # no job table: the first table is the run table
    assert list(runs.columns) == list(page.RUN_COLUMNS) and len(runs) == 2

    at.selectbox(key="run_id").set_value(runs["run"][0])
    at.checkbox(key="show_synthetic").check().run()
    assert not at.exception and not at.error
    tables = {tuple(d.value.columns): d.value for d in at.dataframe}
    assert len(tables[page.SYNTHETIC_EVALUATION_COLUMNS]) == 3
    assert list(tables[page.GENERATION_COLUMNS]["geracao"]) == [0]
    assert len(tables[page.CHECKPOINT_COLUMNS]) >= 1
    headings = _texts(at.markdown)
    for heading in ("Gerações (SINTÉTICO)", "Checkpoints (SINTÉTICO)", "Frentes registradas (SINTÉTICO)"):
        assert heading in headings
    assert "Banco sem fila de jobs (esquema v1)" in _texts(at.caption)
    assert hashlib.sha256(v1_db.read_bytes()).hexdigest() == digest


@needs_streamlit
def test_page_auto_refresh_mode_renders_the_same_content(synthetic_db: Path) -> None:
    at = _app()
    at.run()
    at.text_input(key="db_path").set_value(str(synthetic_db))
    at.number_input(key="refresh_s").set_value(5).run()
    assert not at.exception and not at.error
    assert len(at.dataframe) >= 2 and "SINTÉTICO" in _texts(at.warning)
