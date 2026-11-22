"""Hypervolume by GPU time and by unique evaluations, per method and seed (task A7). CPU only.

The hand-worked cases use in-memory rows in the read-model shape (nothing stored, nothing measured). The runs
stored in a database come from the real search runner and the operation service with the synthetic evaluator,
opted into explicitly; their numbers are labeled synthetic wherever they leave.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.contracts import TCCJobSpec
from cnnlab.tcc.operation import cli
from cnnlab.tcc.operation import hv_analysis as hv
from cnnlab.tcc.operation.demo import EXIT_REFUSED
from cnnlab.tcc.science.hypervolume import hypervolume_of
from cnnlab.tcc.science.operators import FP32_SLICE
from cnnlab.tcc.science.record import CacheContext
from cnnlab.tcc.science.search import (
    DirectContext,
    DirectHashes,
    DirectResume,
    SearchConfig,
    SearchRunner,
)
from cnnlab.tcc.storage import ReadModel, SQLiteRunStore
from cnnlab.tcc.storage.errors import NotFoundError
from cnnlab.tcc.storage.testing import FakeEvaluator, make_job, seed_reference_rows

REF = (1.0, 2.0, 4.0, 4.0)
PROD = 32.0  # 1 * 2 * 4 * 4
RUN = {"run_id": "r1", "method": "joint", "seed_block": 0, "stage": "pilot", "budget_gpu_s": 30.0}


def row(
    attempt: str,
    status: str = "valid",
    *,
    gpu: float = 10.0,
    vector: tuple[float, ...] | None = None,
    cache_hit: bool = False,
    synthetic: bool = False,
    candidate: str | None = None,
) -> dict[str, Any]:
    """An evaluation row as the read model returns it (formatting input only; never stored)."""
    e = l = v = s = None
    if vector is not None and status == "valid":
        e, l, v, s = vector
    return {
        "attempt_id": attempt,
        "candidate_id": candidate or attempt.split(".")[0],
        "generation": 0,
        "status": status,
        "cache_hit": cache_hit,
        "e": e,
        "l_ms": l,
        "v_mb": v,
        "s_mb": s,
        "gpu_seconds": gpu,
        "synthetic": synthetic,
    }


A, B, C, D = (0.5, 1.0, 2.0, 2.0), (0.25, 1.5, 3.0, 1.0), (0.1, 3.0, 1.0, 1.0), (0.2, 0.5, 1.0, 1.0)
ROWS = [
    row("a.a1", vector=A, gpu=10.0),  # 10 s: the first point, 2 / 32
    row("x.a1", "invalid", gpu=5.0),  # 15 s spent on an attempt that gave nothing
    row("b.a1", vector=B, gpu=10.0),  # 25 s: adds to the union
    row("b2.a1", vector=B, gpu=0.0, cache_hit=True, candidate="b2"),  # a cache hit: not a unique evaluation
    row("c.a1", vector=C, gpu=7.0),  # 32 s: beyond the reference point in L: adds nothing, is listed
    row("d.a1", vector=D, gpu=8.0),  # 40 s: crosses the budget of 30 s, improves the front
]


def test_the_curve_matches_the_volumes_worked_out_by_hand() -> None:
    curve = hv.run_curve(RUN, ROWS, REF)
    assert [p["unique"] for p in curve.points] == [1, 2, 3, 4]  # the invalid attempt and the cache hit are no points
    assert [p["gpu_s"] for p in curve.points] == [10.0, 25.0, 32.0, 40.0]  # the invalid attempt spent 5 s
    # A alone: 0.5 * 1 * 2 * 2 = 2. A with B: 2 + (0.75 * 0.5 * 1 * 3 = 1.125) - (0.5 * 0.5 * 1 * 2 = 0.5) = 2.625
    assert curve.points[0]["hv_raw"] == pytest.approx(2.0) and curve.points[0]["hv"] == pytest.approx(2.0 / PROD)
    assert curve.points[1]["hv_raw"] == pytest.approx(2.625) and curve.points[1]["hv"] == pytest.approx(2.625 / PROD)
    assert curve.points[2]["hv"] == curve.points[1]["hv"]  # C is beyond L = 2: no volume, the value is carried
    assert curve.outside == ("c",) and curve.points[3]["hv"] > curve.points[1]["hv"]
    assert [p["front"] for p in curve.points] == [1, 2, 2, 1]  # D dominates A and B: the front is D alone
    assert curve.points[3]["hv_raw"] == pytest.approx(hypervolume_of({"a": A, "b": B, "d": D}, REF).raw)
    assert curve.points[3]["hv_raw"] == pytest.approx(0.8 * 1.5 * 3.0 * 3.0)  # D alone: 10.8, more than A with B
    assert (curve.attempts, curve.valid, curve.unique, curve.gpu_total_s) == (6, 5, 4, 40.0)
    # the sensitivity point: 1.5 x the reference in L, V, S; a bigger box, a different normalization
    sens = (1.0, 3.0, 6.0, 6.0)
    assert curve.points[0]["hv_sens"] == pytest.approx(0.5 * 2.0 * 4.0 * 4.0 / math.prod(sens))


def test_the_value_at_a_time_is_the_last_confirmed_one_and_zero_before_the_first() -> None:
    curve = hv.run_curve(RUN, ROWS, REF)
    assert hv.value_at_gpu(curve, 9.9) == 0.0 and hv.value_at_gpu(curve, 10.0) == pytest.approx(2.0 / PROD)
    assert hv.value_at_gpu(curve, 24.9) == pytest.approx(2.0 / PROD)  # between two evaluations: the last confirmed
    assert hv.value_at_gpu(curve, 25.0) == pytest.approx(2.625 / PROD)
    assert hv.value_at_gpu(curve, 1e9) == curve.points[-1]["hv"]  # after the end: kept
    assert hv.value_at_unique(curve, 0) == 0.0 and hv.value_at_unique(curve, 2) == pytest.approx(2.625 / PROD)
    assert hv.value_at_unique(curve, 5) is None  # a run has no value beyond its own count


def test_the_attempt_that_crosses_the_budget_counts_in_the_final_value_and_not_in_the_one_at_the_budget() -> None:
    curve = hv.run_curve(RUN, ROWS, REF)  # budget 30 s; D arrives at 40 s
    f = hv.final_values(curve)
    assert f["hv_at_budget"] == pytest.approx(2.625 / PROD)  # what had been confirmed within the budget
    assert f["hv_final"] == curve.points[-1]["hv"] > f["hv_at_budget"]  # the cut rule: the attempt in progress counts
    empty = hv.run_curve(RUN, [row("x.a1", "invalid")], REF)
    assert hv.final_values(empty)["hv_final"] == 0.0 and empty.points == () and empty.unique == 0


def test_a_run_that_ends_before_the_budget_keeps_its_value_and_declares_the_time_it_did_not_use() -> None:
    short = hv.run_curve({**RUN, "run_id": "r2", "budget_gpu_s": 3600.0}, ROWS[:3], REF)  # 25 s of a 3600 s budget
    rows = hv.time_grid_rows([short], [600.0, 3000.0])
    assert [(r["gpu_s"], r["run_ended"], r["unused_gpu_s"]) for r in rows] == [
        (600.0, True, 575.0),
        (3000.0, True, 2975.0),
    ]
    assert all(r["hv"] == pytest.approx(2.625 / PROD) for r in rows)  # the last value, kept to the end
    final = hv.final_rows([short])[0]
    assert final["unused_gpu_s"] == 3575.0 and final["excess_gpu_s"] == 0.0 and final["unique_valid"] == 2
    over = hv.final_rows([hv.run_curve(RUN, ROWS, REF)])[0]
    assert over["excess_gpu_s"] == 10.0 and over["unused_gpu_s"] == 0.0  # the excess stays visible, never hidden
    assert hv.time_grid([short], 600.0) == [600.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0]


def test_the_evaluation_grid_has_no_value_where_a_run_never_got() -> None:
    long = hv.run_curve(RUN, ROWS, REF)
    short = hv.run_curve({**RUN, "run_id": "r2", "method": "mono"}, ROWS[:3], REF)
    rows = hv.evaluation_grid_rows([long, short])
    assert [r["unique"] for r in rows if r["run_id"] == "r1"] == [1, 2, 3, 4]
    assert [r["unique"] for r in rows if r["run_id"] == "r2"] == [1, 2]  # nothing carried to 3 and 4
    assert next(r for r in rows if r["run_id"] == "r2" and r["unique"] == 2)["gpu_s"] == 25.0


def test_paired_rows_give_the_values_and_the_differences_before_any_test() -> None:
    def curve(method: str, block: int, vector: tuple[float, ...]) -> hv.RunCurve:
        return hv.run_curve(
            {**RUN, "run_id": f"{method}{block}", "method": method, "seed_block": block},
            [row("p.a1", vector=vector)],
            REF,
        )

    curves = [
        curve("joint", 0, A),
        curve("mono", 0, C),
        curve("random", 0, B),
        curve("joint", 1, B),
        curve("random", 1, A),
    ]  # block 1 has no mono: its cells stay empty
    rows = hv.paired_rows(curves)
    assert [r["seed_block"] for r in rows] == [0, 1] and set(rows[0]) >= {"joint", "mono", "random", "joint_menos_mono"}
    j0, m0, r0 = (hv.final_values(c)["hv_final"] for c in curves[:3])
    assert rows[0]["joint_menos_mono"] == pytest.approx(j0 - m0) and rows[0]["joint_menos_random"] == pytest.approx(
        j0 - r0
    )
    assert m0 == 0.0  # C is beyond the reference point: no volume, and it is a number, not a gap
    assert rows[1]["mono"] is None and rows[1]["joint_menos_mono"] is None  # missing is empty, never zero
    assert rows[1]["joint_menos_random"] == pytest.approx(
        hv.final_values(curves[3])["hv_final"] - hv.final_values(curves[4])["hv_final"]
    )


# -- runs in a database ------------------------------------------------------------------------------------------


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


# FakeEvaluator's ranges: E in [0.05, 0.95], L in [0.1, 5.1], V in [1, 51], S in [0.1, 20.1]
DB_REF = (0.99, 5.2, 52.0, 20.2)


@pytest.fixture
def two_methods(tmp_path: Path) -> Path:
    """Four runs of one database: joint and mono on two seed blocks, with a hypervolume recorded in every snapshot."""
    store = SQLiteRunStore(tmp_path / "db" / "tcc.sqlite3", tmp_path / "artifacts")
    seed_reference_rows(store)
    for method in ("joint", "mono"):
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
            runner = SearchRunner(
                SearchConfig(population=4, generations=3, reference_point=DB_REF), FP32_SLICE, context_for
            )
            assert runner(ctx) == "completed"
    store.close()
    return tmp_path / "db" / "tcc.sqlite3"


def test_the_curve_from_the_evaluations_equals_the_hypervolume_the_runner_recorded_in_every_snapshot(
    two_methods: Path,
) -> None:
    curves, ref, origin = hv.read_curves(two_methods, reference=DB_REF, recorded_reference=DB_REF)
    assert ref == DB_REF and origin == "given on the command line"
    assert {c.method for c in curves} == {"joint", "mono"} and len(curves) == 4
    with ReadModel(two_methods) as rm:
        for c in curves:
            snapshots = rm.fronts(c.run_id)
            assert len(snapshots) == 4 and all(s["hypervolume"] is not None for s in snapshots)
            for s in snapshots:  # the independent computation agrees with the recorded one, to the last digit
                assert hv.value_at_unique(c, s["unique_valid_evaluations"]) == pytest.approx(
                    s["hypervolume"], rel=1e-12
                )
            assert c.snapshot_mismatches == () and c.unique == 16 and c.gpu_total_s == 48.0
            # monotone: the archive only ever gets better
            values = [p["hv"] for p in c.points]
            assert values == sorted(values) and 0.0 < values[-1] <= 1.0


def test_a_hypervolume_that_disagrees_with_the_evaluations_is_reported(two_methods: Path) -> None:
    with ReadModel(two_methods) as rm:
        run = next(r for r in rm.runs() if r["run_id"] == "joint-b0")
        evaluations = rm.evaluations("joint-b0")
        snapshots = [dict(s) for s in rm.fronts("joint-b0")]
    snapshots[1]["hypervolume"] = snapshots[1]["hypervolume"] * 1.01  # as if the runner had recorded a wrong value
    curve = hv.run_curve(run, evaluations, DB_REF, snapshots, DB_REF)
    assert len(curve.snapshot_mismatches) == 1 and "snapshot" in curve.snapshot_mismatches[0]
    other = hv.run_curve(run, evaluations, (1.0, 6.0, 60.0, 25.0), snapshots, DB_REF)  # another point: not comparable
    assert other.snapshot_mismatches == ()


def test_the_export_writes_every_table_with_hashes_and_labels_synthetic_numbers(
    two_methods: Path, tmp_path: Path
) -> None:
    before = hashlib.sha256(two_methods.read_bytes()).hexdigest()
    dest = tmp_path / "hv"
    with pytest.raises(hv.AnalysisError, match="synthetic results"):
        hv.export_hv(two_methods, dest, reference=DB_REF)
    assert not dest.exists() and list(tmp_path.glob(".hv.partial-*")) == []

    done = hv.export_hv(two_methods, dest, reference=DB_REF, recorded_reference=DB_REF, allow_synthetic=True)
    assert hashlib.sha256(two_methods.read_bytes()).hexdigest() == before  # read-only
    names = sorted(p.name for p in dest.iterdir())
    assert names == sorted(
        [
            "hv_curvas.csv",
            "hv_grade_tempo.csv",
            "hv_grade_avaliacoes.csv",
            "hv_final.csv",
            "hv_pareado.csv",
            "hv_pareado_sensibilidade.csv",
            "hv_fora_do_ponto.csv",
            hv.MANIFEST_NAME,
        ]
    )
    manifest = json.loads((dest / hv.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["format"] == "tcc-hv-analysis/1" and manifest["synthetic"] is True
    assert manifest["reference_point"]["E"] == DB_REF[0] and manifest["snapshot_mismatches"] == []
    assert manifest["sensitivity_point"]["L_ms"] == pytest.approx(1.5 * DB_REF[1])
    for entry in manifest["files"]:
        data = (dest / entry["name"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"] and b"\r" not in data
        assert data.decode("utf-8").splitlines()[0].split(",") == entry["columns"]
    assert done.manifest["files"] == manifest["files"] and len(done.curves) == 4

    def table(name: str) -> list[dict[str, str]]:
        with open(dest / name, encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    finals = table("hv_final.csv")
    assert {"hv_final_SINTETICO", "hv_raw_final_SINTETICO", "hv_at_budget_SINTETICO"} <= set(finals[0])
    assert "hv_final" not in finals[0] and "hv_raw_final" not in finals[0]  # no unlabeled synthetic number leaves
    assert len(finals) == 4 and {f["method"] for f in finals} == {"joint", "mono"} and finals[0]["unique_valid"] == "16"
    assert float(finals[0]["unused_gpu_s"]) == float(finals[0]["budget_gpu_s"]) - 48.0
    pared = table("hv_pareado.csv")
    assert [p["seed_block"] for p in pared] == ["0", "1"] and "joint_menos_mono_SINTETICO" in pared[0]
    assert float(pared[0]["joint_SINTETICO"]) - float(pared[0]["mono_SINTETICO"]) == pytest.approx(
        float(pared[0]["joint_menos_mono_SINTETICO"])
    )
    assert len(table("hv_curvas.csv")) == 64 and "hv_SINTETICO" in table("hv_curvas.csv")[0]
    grid = table("hv_grade_tempo.csv")
    assert {float(g["gpu_s"]) for g in grid} == {600.0 * i for i in range(1, 34)}  # up to the 19,800 s budget
    assert all(g["run_ended"] == "true" for g in grid)  # 48 s of GPU: every grid point is after the end
    assert len(table("hv_grade_avaliacoes.csv")) == 64
    assert table("hv_fora_do_ponto.csv") == []  # DB_REF lies beyond everything FakeEvaluator can return

    with pytest.raises(hv.AnalysisError, match="already exists"):
        hv.export_hv(two_methods, dest, reference=DB_REF, allow_synthetic=True)
    with pytest.raises(hv.AnalysisError, match="positive number of seconds"):
        hv.export_hv(two_methods, tmp_path / "x", reference=DB_REF, allow_synthetic=True, grid_s=0)
    with pytest.raises(hv.AnalysisError, match="four finite positive"):
        hv.export_hv(two_methods, tmp_path / "y", reference=(1.0, 2.0, 3.0), allow_synthetic=True)
    with pytest.raises(NotFoundError):
        hv.export_hv(two_methods, tmp_path / "z", reference=DB_REF, run_ids=["no-such-run"], allow_synthetic=True)


def test_the_candidates_beyond_the_reference_point_are_listed(two_methods: Path, tmp_path: Path) -> None:
    tight = (0.5, 1.0, 10.0, 5.0)  # most of FakeEvaluator's range lies beyond it
    done = hv.export_hv(two_methods, tmp_path / "tight", reference=tight, allow_synthetic=True)
    listed = (tmp_path / "tight" / "hv_fora_do_ponto.csv").read_text(encoding="utf-8").splitlines()
    per_run = {c.run_id: c.outside for c in done.curves}
    assert len(listed) - 1 == sum(len(v) for v in per_run.values()) > 0
    with ReadModel(two_methods) as rm:
        for c in done.curves:
            vectors = {
                e["candidate_id"]: (e["e"], e["l_ms"], e["v_mb"], e["s_mb"])
                for e in rm.evaluations(c.run_id)
                if e["status"] == "valid"
            }
            assert set(c.outside) == {k for k, v in vectors.items() if not all(x < r for x, r in zip(v, tight))}
    assert sum(1 for c in done.curves if c.snapshot_mismatches) == 0  # nothing recorded against this point: no check


def test_the_reference_point_comes_from_the_frozen_plan_or_must_be_given(two_methods: Path, tmp_path: Path) -> None:
    with pytest.raises(hv.AnalysisError, match="no reference point"):
        hv.read_curves(two_methods)  # these runs have a plan without one

    from cnnlab.tcc.operation import campaign
    from cnnlab.tcc.operation.campaign import CampaignSettings
    from cnnlab.tcc.operation.demo import DemoRunSpec, default_repo_root, start_demo
    from cnnlab.tcc.operation.provenance import CodeVersion
    from cnnlab.tcc.operation.testing import synthetic_demo_wiring

    code = CodeVersion("campanha", "c" * 40, False, "7" * 64, (), (), 0)
    s = CampaignSettings.from_mapping(
        {
            "master_seed": 1,
            "factory": "cnnlab.tcc.operation.testing:synthetic_demo_wiring",
            "methods": ["joint", "mono"],
            "factory_args": {"slice": "fp16_regions", "epochs": 5, "n_candidates": 3, "sleep_s": 0.0},
            "reference_point": [1.0, 2.5, 12.5, 5.0],
            "viable_minima": [0.2, 0.5, 2.0, 1.0],
            "max_attempt_train_s": 1800,
            "latency_cv_max": 0.02,
            "v_audit_rel": 0.05,
        }
    )
    db, art = tmp_path / "camp" / "tcc.sqlite3", tmp_path / "camp" / "artifacts"
    with SQLiteRunStore(db, art) as store:
        plan = campaign.create_plan(
            store, s, synthetic_demo_wiring(campaign.representative_spec(s)), code, default_repo_root(), freeze=True
        )
    for method in ("joint", "mono"):
        block = plan.content["blocks"][0]
        spec = DemoRunSpec.from_mapping(
            {
                "method": method,
                "seed_block": 0,
                "search_seed": block["search_seed"],
                "train_seed": block["train_seed"],
                "calibration_seed": block["calibration_seed"],
                "budget_gpu_s": 19_800.0,
                "code_label": "campanha-cccccccc",
                "author": "t",
                "run_id": f"camp-{method}",
                "factory": s.factory,
                "factory_args": dict(plan.content["factory_args"]),
                "stage": "confirmatory",
                "plan_id": plan.plan_id,
            }
        )
        assert (
            start_demo(
                db,
                art,
                spec,
                synthetic_demo_wiring,
                worker_id="w",
                code_version=code,
                repo_root=default_repo_root(),
                allow_synthetic=True,
            ).outcome.end_state
            == "done"
        )
    curves, ref, origin = hv.read_curves(db)
    assert ref == (1.0, 2.5, 12.5, 5.0) and plan.plan_id in origin and len(curves) == 2
    assert {c.stage for c in curves} == {"confirmatory"}
    explicit, _, how = hv.read_curves(db, reference=(1.0, 3.0, 13.0, 6.0))[0:3]
    assert how == "given on the command line"  # what is given wins, and says so


def test_the_cli_writes_the_analysis_and_refuses_what_it_should(
    two_methods: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "curvas"
    base = [
        "campaign",
        "curves",
        "--db",
        str(two_methods),
        "--dest",
        str(dest),
        "--reference",
        ",".join(map(str, DB_REF)),
    ]
    assert cli.main(base) == EXIT_REFUSED and "synthetic results" in capsys.readouterr().err and not dest.exists()
    assert cli.main([*base, "--allow-synthetic", "--grid-s", "1200"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC results analysed in labeled columns" in out and "hv_final.csv: 4 row(s) sha256=" in out
    assert f"export {dest}: 4 run(s)" in out and "given on the command line" in out
    assert cli.main([*base, "--allow-synthetic"]) == EXIT_REFUSED and "already exists" in capsys.readouterr().err
    one = tmp_path / "um"
    assert (
        cli.main(
            [
                "campaign",
                "curves",
                "--db",
                str(two_methods),
                "--dest",
                str(one),
                "--reference",
                ",".join(map(str, DB_REF)),
                "--run",
                "joint-b0",
                "--allow-synthetic",
            ]
        )
        == 0
    )
    assert "1 run(s)" in capsys.readouterr().out
