"""Methods tab of the page (task A7): data layer and headless page. CPU only.

The runs come from the real search runner with the synthetic evaluator, so every number is labeled synthetic and the
page shows them only when asked to. The hand-worked hypervolume cases are in ``operation/test_operation_hv_analysis.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

import cnnlab.tcc.demo_methods as methods
import cnnlab.tcc.demo_streamlit as page
from cnnlab.tcc.contracts import TCCJobSpec
from cnnlab.tcc.operation import hv_analysis as hv
from cnnlab.tcc.science.operators import FP32_SLICE
from cnnlab.tcc.science.record import CacheContext
from cnnlab.tcc.science.search import DirectContext, DirectHashes, DirectResume, SearchConfig, SearchRunner
from cnnlab.tcc.storage import SQLiteRunStore
from cnnlab.tcc.storage.testing import FakeEvaluator, make_job, seed_reference_rows

HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None
needs_streamlit = pytest.mark.skipif(not HAS_STREAMLIT, reason="streamlit is not installed in this environment")
REF = (0.99, 5.2, 52.0, 20.2)  # beyond everything FakeEvaluator can return


def context_for(job: TCCJobSpec) -> CacheContext:
    return CacheContext(
        split_fingerprint="split-45k-5k-cal1024-v1",
        train_seed=job.train_seed,
        calibration_seed=job.calibration_seed,
        train_policy={"epochs": 1},
        pruning_policy={},
        quantization_policy={},
        code_version="test-fixture",
        backend={"torch": "cpu-test"},
        builder_config={},
        space_hash=FP32_SLICE.space_hash(),
    )


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """Joint, mono and random on two seed blocks, six runs, 3 s of GPU per evaluation."""
    store = SQLiteRunStore(tmp_path / "db" / "tcc.sqlite3", tmp_path / "artifacts")
    seed_reference_rows(store)
    for method in ("joint", "mono", "random"):
        for block in (0, 1):
            job = make_job(
                f"{method}-b{block}", method=method, stage="pilot", seed_block=block, space_hash=FP32_SLICE.space_hash()
            )
            store.create_run(job)
            hashes = store.run_hashes(job.run_id)
            ctx = DirectContext(
                job=job,
                store=store,
                evaluator=FakeEvaluator(gpu_seconds=3.0),
                resume=DirectResume(),
                hashes=DirectHashes(hashes.protocol_hash, hashes.plan_hash, hashes.space_hash),
                artifact_root=store.artifact_root,
                emit=lambda e: None,
                should_stop=lambda: False,
                active_gpu_seconds=lambda j=job: store.active_gpu_seconds(j.run_id),
                unique_valid_evaluations=lambda j=job: store.unique_valid_evaluations(j.run_id),
            )
            SearchRunner(SearchConfig(population=4, generations=2, reference_point=REF), FP32_SLICE, context_for)(ctx)
    store.close()
    return tmp_path / "db" / "tcc.sqlite3"


def test_the_reference_point_typed_on_the_page_is_parsed_or_explained() -> None:
    assert methods.parse_reference("1, 2.5; 12.5 ,5") == (1.0, 2.5, 12.5, 5.0)
    for bad in ("1,2,3", "1,2,3,4,5", "a,b,c,d", "1,2,3,0", "1,2,3,-4", "1,2,3,nan"):
        with pytest.raises(ValueError):
            methods.parse_reference(bad)


def test_the_frames_follow_the_analysis_and_carry_the_method_and_the_seed(db: Path) -> None:
    curves, ref, _ = methods.load(str(db), REF)
    assert ref == REF and len(curves) == 6
    grid = hv.time_grid(curves, 600.0)
    rows = methods.gpu_frame(curves, grid)
    assert len(rows) == 6 * len(grid) and {r["metodo"] for r in rows} == {"joint", "mono", "random"}
    assert {r["semente"] for r in rows} == {0, 1} and all(0.0 <= r["hv"] <= 1.0 for r in rows)
    assert rows[0]["x"] == pytest.approx(600.0 / 3600.0) and all(r["encerrada"] for r in rows)  # 36 s of 19,800 s
    unique = methods.unique_frame(curves)
    assert (
        len(unique) == 6 * 12
        and {r["x"] for r in unique} == set(range(1, 13))
        and not any(r["encerrada"] for r in unique)
    )

    means = methods.mean_frame(unique)
    assert {(m["metodo"], m["sementes"]) for m in means} == {("joint", 2), ("mono", 2), ("random", 2)}
    first = next(m for m in means if m["metodo"] == "joint" and m["x"] == 1)
    mine = [r["hv"] for r in unique if r["metodo"] == "joint" and r["x"] == 1]
    assert first["hv"] == pytest.approx(sum(mine) / 2)  # the mean of the two seeds
    assert methods.mean_frame([r for r in unique if r["serie"] == "joint-b0"]) == []  # one seed: no mean to draw

    table = methods.final_table(curves)
    assert [t["hv"] for t in table] == [hv.final_values(c)["hv_final"] for c in curves]
    assert {t["avaliacoes_unicas"] for t in table} == {12} and {t["gpu_s"] for t in table} == {36.0}
    assert all(t["nao_usado_s"] == t["orcamento_s"] - 36.0 and t["excesso_s"] == 0.0 for t in table)
    paired = methods.paired_table(curves)
    assert [p["seed_block"] for p in paired] == [0, 1] and {"joint_menos_mono", "joint_menos_random"} <= set(paired[0])


def test_the_origin_of_the_reference_point_is_said_in_portuguese() -> None:
    assert methods.origin_text("given on the command line") == "digitado nesta página"
    assert (
        methods.origin_text("the frozen plan of the runs: campaign-plan-x")
        == "do plano congelado das runs: campaign-plan-x"
    )


def test_every_method_has_its_own_color_and_its_own_dash_in_both_themes() -> None:
    for theme in ("light", "dark"):
        colors = methods.METHOD_COLORS[theme]
        assert set(colors) == set(hv.METHOD_ORDER) and len(set(colors.values())) == 4
    assert methods.METHOD_COLORS["light"].keys() == methods.METHOD_COLORS["dark"].keys()
    assert len({tuple(d) for d in methods.METHOD_DASH.values()}) == 4  # a second encoding: color is not alone


def _script(db_path: str) -> None:
    import streamlit as st

    import cnnlab.tcc.demo_streamlit as demo_page

    demo_page.render(st, db_path)


def _texts(elements: Any) -> str:
    return "\n".join(str(e.value) for e in elements)


@needs_streamlit
def test_the_page_hides_synthetic_runs_and_shows_them_labeled_when_asked(db: Path) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_script, args=(str(db),), default_timeout=90).run()
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == ["Operação", "Análise", "Métodos", "Glossário"]
    # the plans of these runs carry no reference point and none is typed: the tab asks for it, and guesses nothing
    assert "Informe o ponto de referência" in _texts(at.info)

    at.text_input(key="methods_reference").set_value("0.99, 5.2, 52.0, 20.2").run()
    assert not at.exception and not at.error
    assert "ficam escondidas" in _texts(at.info)  # six runs, all synthetic, hidden by default
    assert not any(h.value == "Valor final por método e semente" for h in at.subheader)

    at.checkbox(key="show_synthetic").check().run()
    assert not at.exception and not at.error
    assert "Resultados SINTÉTICOS" in _texts(at.warning)
    assert [h.value for h in at.subheader if h.value.startswith(("Valor final", "Diferença", "Candidatas"))] == [
        "Valor final por método e semente",
        "Diferença pareada por semente",
        "Candidatas além do ponto de referência",
    ]
    frames = {tuple(d.value.columns): d.value for d in at.dataframe}
    finals = next(v for k, v in frames.items() if "hv_no_orcamento" in k)
    assert len(finals) == 6 and set(finals["metodo"]) == {"joint", "mono", "random"}
    assert "Ponto de referência E=0.99" in _texts(at.caption) and "digitado nesta página" in _texts(at.caption)
    methods_charts = [c for c in at.get("vega_lite_chart") if "hv" in str(c.proto.spec)]
    assert len(methods_charts) == 1

    at.radio(key="methods_axis").set_value("unique").run()
    assert not at.exception and not at.error
    assert "unique" in str(at.radio(key="methods_axis").value)
    assert not at.exception

    at.text_input(key="methods_reference").set_value("1,2,3").run()
    assert "quatro números" in _texts(at.error) or "quatro" in _texts(at.error)


def test_neither_the_tab_nor_the_analysis_behind_it_can_write() -> None:
    import ast

    for module in (methods, hv):  # the tab, and the module that does the reading for it and for the CSV export
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        assert not any("sqlite_store" in m or "science.search" in m for m in imported), module.__name__
        assert not names & {"SQLiteRunStore", "JobQueue", "execute", "executescript", "executemany"}, module.__name__
        assert "sqlite3" not in imported
    assert "cnnlab.tcc.storage.readmodel" in {
        n.module
        for n in ast.walk(ast.parse(Path(hv.__file__).read_text(encoding="utf-8")))
        if isinstance(n, ast.ImportFrom)
    }
