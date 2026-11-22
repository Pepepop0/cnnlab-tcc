"""H4: latency against MACs and parameters, and the predictive model (TCC-01 section 6; task A10c). CPU only.

The statistics are worked out by hand on small series. The database holds genomes sampled from the real search space and
**scripted** latencies (a function of a scripted cost), so what the analysis should find is known beforehand; the
evaluator is the synthetic one and every number leaves labeled synthetic. One test rebuilds a real network on the CPU
(torch, no GPU) to check the recomputed MACs.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.contracts import EvaluationResult, Fitness
from cnnlab.tcc.operation import cli, h4
from cnnlab.tcc.operation.demo import EXIT_REFUSED
from cnnlab.tcc.science.genome import BlockGenes, Genome
from cnnlab.tcc.science.record import CacheContext, prepare_genome
from cnnlab.tcc.science.sampling import GeneRng, sample_genome
from cnnlab.tcc.science.space import SPACE_V1
from cnnlab.tcc.storage import SQLiteRunStore
from cnnlab.tcc.storage.testing import FakeEvaluator, make_job, make_request, seed_reference_rows

CONTEXT = CacheContext(
    split_fingerprint="f" * 64,
    train_seed=2000,
    calibration_seed=3000,
    train_policy={},
    pruning_policy={},
    quantization_policy={},
    code_version="test",
    backend={},
    builder_config={},
    space_hash=SPACE_V1.space_hash(),
)


def genome(seed: int, **changes: Any) -> Genome:
    g = sample_genome(SPACE_V1, GeneRng(seed))
    return dataclasses.replace(g, **changes) if changes else g


def expressed(g: Genome) -> dict[str, Any]:
    return prepare_genome(g, SPACE_V1, CONTEXT).record.expressed_genome  # type: ignore[return-value]


def fake_cost(expr: Any) -> tuple[int, int]:
    """A scripted cost that grows with the kept widths: what the test knows, not what a network has."""
    blocks = expr["blocks"]
    kept = [h4.pruned_width(int(b["channels"]), int(b.get("prune_pct") or 0)) for b in blocks]
    return 1000 * sum(k * k for k in kept) + 7 * expr["stem_channels"], 10 * sum(kept) + expr["stem_channels"]


# -- statistics by hand -------------------------------------------------------------------------------------------


def test_spearman_worked_out_by_hand() -> None:
    assert h4.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)  # d = 0,1,1,0: 1 - 6*2/(4*15)
    assert h4.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert h4.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # ties take the mean rank: ranks of x are 1.5, 1.5, 3; of y 1, 2, 3
    assert h4.spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.8660254037844387)
    assert h4.spearman([5, 5, 5], [1, 2, 3]) is None and h4.spearman([1], [1]) is None
    with pytest.raises(h4.H4Error):
        h4.spearman([1, 2], [1])


def test_the_bootstrap_resamples_whole_architectures_and_is_deterministic() -> None:
    clusters = [([float(i), float(i) + 0.5], [float(i) * 2, float(i) * 2 + 1.0]) for i in range(10)]
    first = h4.cluster_bootstrap_ci(clusters)
    assert first == h4.cluster_bootstrap_ci(clusters) and first is not None
    assert first[0] <= 1.0 <= first[1] + 1e-12 and first[0] > 0.9  # a monotone relation: a tight interval near 1
    assert h4.cluster_bootstrap_ci(clusters[:2]) is None  # two architectures say nothing about the interval


def test_ridge_recovers_a_linear_relation_and_standardizes_with_the_training_set_only() -> None:
    import numpy as np

    x = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 8.0], [4.0, 1.0], [5.0, 9.0]])
    y = 2.0 * x[:, 0] - 0.5 * x[:, 1] + 1.0
    prediction = h4.ridge_fit_predict(x, y, x, alpha=1e-9)
    assert prediction == pytest.approx(y, abs=1e-6)
    constant = np.column_stack([x[:, 0], np.full(5, 3.0)])  # a constant column must not divide by zero
    assert np.all(np.isfinite(h4.ridge_fit_predict(constant, y, constant)))


def test_folds_are_deterministic_and_balanced_by_architecture() -> None:
    names = [f"{i:02d}" for i in range(12)]
    folds = h4.assign_folds(names * 3, 5)
    assert folds == h4.assign_folds(reversed(list(names * 3)), 5)  # the order of the points does not matter
    counts = sorted(sum(1 for f in folds.values() if f == k) for k in range(5))
    assert counts == [2, 2, 2, 3, 3]


# -- architectures ---------------------------------------------------------------------------------------------------


def test_the_effective_architecture_ignores_precision_and_training_genes_and_keeps_the_pruned_width() -> None:
    base = genome(3)
    other_lr = dataclasses.replace(base, log10_lr=-2.5 if base.log10_lr < -3.0 else -3.5)
    flipped = dataclasses.replace(
        base, blocks=tuple(dataclasses.replace(b, precision="fp32") for b in base.blocks), stem_precision="fp32"
    )
    a, b, c = expressed(base), expressed(other_lr), expressed(flipped)
    assert h4.architecture_hash(a) == h4.architecture_hash(b) == h4.architecture_hash(c)
    if any(p != "fp32" for p in h4.precision_key(a)):
        assert h4.representation_hash(a) != h4.representation_hash(c)  # precision is part of the representation
    wider = dataclasses.replace(base, stem_channels=64 if base.stem_channels != 64 else 32)
    assert h4.architecture_hash(expressed(wider)) != h4.architecture_hash(a)
    # two genes asking different pruning for the same kept width are one architecture
    block = BlockGenes(**{**dataclasses.asdict(base.blocks[0]), "channels": 64, "prune_pct": 0, "sparsity": None})
    pruned = dataclasses.replace(block, channels=128, prune_pct=50)
    wide = {**a, "blocks": [block.to_dict(), *a["blocks"][1:]]}
    narrow = {**a, "blocks": [pruned.to_dict(), *a["blocks"][1:]]}
    assert h4.architecture_key(wide)[4][0][0] == h4.architecture_key(narrow)[4][0][0] == 64


def test_precision_shares_count_stem_blocks_and_head() -> None:
    expr = {
        "stem_precision": "fp16",
        "head_precision": "fp32",
        "blocks": [{"precision": "fp16"}, {"precision": "int8"}],
    }
    assert h4.precision_shares(expr) == (0.5, 0.25)


# -- a database -----------------------------------------------------------------------------------------------------


def build(
    tmp_path: Path, *, n: int = 24, recorded_every: int = 0, name: str = "h4", bench: dict[str, str] | None = None
) -> Path:
    """Two runs of candidates sampled from the space; L is scripted from the scripted cost, fp16 layers a bit faster.

    Each genome appears twice in a run (two learning rates) so every architecture has several points; with
    ``recorded_every`` k > 0 the evaluator "recorded" the cost of every k-th evaluation; ``bench`` maps a method to
    the benchmark version its attempts carry.
    """
    db = tmp_path / name / "tcc.sqlite3"
    with SQLiteRunStore(db, tmp_path / name / "art") as store:
        seed_reference_rows(store)
        for method, offset in (("joint", 0), ("random", 1000)):
            job = make_job(f"{method}-b0", method=method, stage="confirmatory", space_hash=SPACE_V1.space_hash())  # type: ignore[arg-type]
            store.create_run(job)
            evaluator = FakeEvaluator(gpu_seconds=1.0)
            for i in range(n):
                g = genome(offset + i % (n // 2))
                g = dataclasses.replace(g, log10_lr=-3.0 - 0.1 * (i // (n // 2)))
                record = prepare_genome(g, SPACE_V1, CONTEXT).record
                macs, params = fake_cost(record.expressed_genome)
                fp16, _ = h4.precision_shares(record.expressed_genome)
                cid, aid = f"{job.run_id}.c{i}", f"{job.run_id}.c{i}.a1"
                store.register_candidate(job.run_id, cid, record, {"generation": 0})
                store.begin_attempt(job.run_id, cid, aid)
                result = evaluator.evaluate(make_request(job, cid, aid, record, store.artifact_root))
                model: dict[str, Any] = {"parameters": 10 * params}  # the unpruned build: never used for H4
                if recorded_every and i % recorded_every == 0:
                    model.update(macs=macs, parameters_pruned=params)
                l_ms = macs / 1e6 * (1.0 - 0.3 * fp16) + 0.01
                store.commit_evaluation(
                    dataclasses.replace(
                        result,
                        fitness=Fitness(e=0.3, l_ms=l_ms, v_mb=1.0, s_mb=1.0),
                        quality={
                            "synthetic": True,
                            "model": model,
                            **({"measurement": {"bench_version": bench[method]}} if bench else {}),
                        },
                    )
                )
    return db


def test_the_points_say_where_each_number_came_from(tmp_path: Path) -> None:
    db = build(tmp_path, recorded_every=4)
    got = h4.collect(db, allow_synthetic=True, cost=fake_cost)
    assert got.synthetic and len(got.points) == 48 and got.skipped == {}
    sources = {(p.macs_source, p.parameters_source) for p in got.points}
    assert sources == {(h4.RECORDED, h4.RECORDED), (h4.RECOMPUTED, h4.RECOMPUTED)}
    assert sum(1 for p in got.points if p.macs_source == h4.RECORDED) == 12  # every 4th of 24, in two runs
    assert all(p.parameters != 10 * p.parameters for p in got.points)  # the unpruned `parameters` is not what is used
    off = h4.collect(db, allow_synthetic=True, recompute=False)
    assert (
        len(off.points) == 12
        and off.skipped == {"cost_not_recorded": 36}
        and all(p.macs_source == h4.RECORDED for p in off.points)
    )


def test_synthetic_results_need_opt_in_and_a_missing_run_is_named(tmp_path: Path) -> None:
    db = build(tmp_path)
    with pytest.raises(h4.H4Error, match="synthetic"):
        h4.collect(db, cost=fake_cost)
    with pytest.raises(Exception, match="nothing-b9"):
        h4.collect(db, ["nothing-b9"], allow_synthetic=True, cost=fake_cost)
    one = h4.collect(db, ["joint-b0"], allow_synthetic=True, cost=fake_cost)
    assert one.runs == ("joint-b0",) and len(one.points) == 24


def test_the_correlation_is_high_where_latency_follows_the_cost(tmp_path: Path) -> None:
    points = h4.collect(build(tmp_path), allow_synthetic=True, cost=fake_cost).points
    rows = {(r["scope"], r["x"]): r for r in h4.correlations(points)}
    assert set(rows) == {(s, x) for s in h4.SCOPES for x in ("macs", "parameters")}
    macs = rows[("todas_avaliacoes", "macs")]
    assert macs["n"] == 48 and macs["spearman"] > 0.85 and macs["ci95_low"] <= macs["spearman"] <= macs["ci95_high"]
    one = rows[("uma_por_arquitetura", "macs")]
    assert one["n"] <= 48 and one["architectures"] == macs["architectures"]
    assert one["n"] == len({p.representation for p in points})  # one point per architecture with its representation


def test_validation_by_group_never_tests_on_an_architecture_it_trained_on(tmp_path: Path) -> None:
    points = h4.collect(build(tmp_path), allow_synthetic=True, cost=fake_cost).points
    result = h4.grouped_validation(points)
    folds: dict[str, set[int]] = {}
    for row in result["predictions"]:
        folds.setdefault(row["architecture"], set()).add(row["fold"])
    assert all(len(f) == 1 for f in folds.values()) and len(folds) > 5
    models = {m["model"]: m for m in result["models"]}
    assert set(models) == {"macs", "macs_parameters", "full", "mediana"}
    assert models["macs"]["r2_log_l"] > 0.5  # latency follows MACs: new architectures are predicted
    assert models["mediana"]["r2_log_l"] <= 0.0 + 1e-9  # the floor explains nothing out of group
    assert (
        models["full"]["r2_log_l"] >= models["macs"]["r2_log_l"] - 0.05
    )  # the precision share adds, or at least does not hurt
    assert all(m["mae_pct"] > 0 for m in models.values())
    assert len(result["folds"]) == 4 * models["macs"]["folds"]


def test_too_few_points_or_one_architecture_refuse_the_model() -> None:
    def point(i: int, arch: str) -> h4.Point:
        return h4.Point(
            "r",
            "joint",
            0,
            f"c{i}",
            f"a{i}",
            arch,
            arch,
            1.0 + i,
            1000 + i,
            100 + i,
            h4.RECORDED,
            h4.RECORDED,
            0.0,
            0.0,
            False,
        )

    with pytest.raises(h4.H4Error, match="at least 8 points"):
        h4.grouped_validation([point(i, "x") for i in range(5)])
    with pytest.raises(h4.H4Error, match="two architectures"):
        h4.grouped_validation([point(i, "x") for i in range(10)])


# -- export ------------------------------------------------------------------------------------------------------------


def test_export_writes_the_tables_with_labels_and_a_manifest(tmp_path: Path) -> None:
    db = build(tmp_path, recorded_every=4)
    dest = tmp_path / "out"
    manifest = h4.export_h4(db, dest, allow_synthetic=True, cost=fake_cost)
    names = sorted(p.name for p in dest.iterdir())
    assert names == [
        "h4_correlacao.csv",
        "h4_modelo.csv",
        "h4_modelo_dobras.csv",
        "h4_pontos.csv",
        "h4_predicoes.csv",
        "manifesto.json",
    ]
    assert manifest["format"] == "tcc-h4/1" and manifest["synthetic"] and manifest["points"] == 48
    assert manifest["cost_sources"] == {
        "macs_recorded": 12,
        "macs_recomputed": 36,
        "parameters_recorded": 12,
        "parameters_recomputed": 36,
    }
    assert json.loads((dest / "manifesto.json").read_text(encoding="utf-8")) == manifest
    with open(dest / "h4_pontos.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert (
        "l_ms_SINTETICO" in rows[0]
        and "l_ms" not in rows[0]
        and {r["macs_source"] for r in rows} == {"recorded", "recomputed"}
    )
    with open(dest / "h4_correlacao.csv", encoding="utf-8") as fh:
        corr = list(csv.DictReader(fh))
    assert "spearman_SINTETICO" in corr[0] and len(corr) == 4
    with pytest.raises(h4.H4Error, match="already exists"):
        h4.export_h4(db, dest, allow_synthetic=True, cost=fake_cost)
    with pytest.raises(h4.H4Error, match="synthetic"):
        h4.export_h4(db, tmp_path / "other", cost=fake_cost)
    assert not (tmp_path / "other").exists() and not [p for p in tmp_path.iterdir() if ".partial" in p.name]


def test_the_database_is_not_touched(tmp_path: Path) -> None:
    import hashlib

    db = build(tmp_path)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    h4.export_h4(db, tmp_path / "out", allow_synthetic=True, cost=fake_cost)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_cli_h4_refuses_synthetic_and_runs_with_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    db = build(tmp_path)
    assert cli.main(["campaign", "h4", "--db", str(db), "--dest", str(tmp_path / "x")]) == EXIT_REFUSED
    assert "synthetic" in capsys.readouterr().err
    monkeypatch.setattr(h4, "recompute_cost", fake_cost)
    assert cli.main(["campaign", "h4", "--db", str(db), "--dest", str(tmp_path / "x"), "--allow-synthetic"]) == 0
    out = capsys.readouterr().out
    assert "spearman" in out and "macs_recomputed" in out and "SYNTHETIC" in out


# -- the real recount (torch on the CPU) --------------------------------------------------------------------------------


def test_recomputed_cost_is_that_of_the_pruned_network() -> None:
    pytest.importorskip("torch")
    from cnnlab.tcc.model_pipeline.builder import build_model, count_macs
    from cnnlab.tcc.science.genome import ExpressedGenome

    g = genome(5)
    expr = expressed(g)
    macs, params = h4.recompute_cost(expr)
    unpruned = [dataclasses.replace(b, prune_pct=0) for b in ExpressedGenome.from_dict(expr).blocks]
    full = dataclasses.replace(ExpressedGenome.from_dict(expr), blocks=tuple(unpruned))
    model, info = build_model(full, 0)
    full_macs = count_macs(model)
    assert 0 < macs <= full_macs and 0 < params <= info.parameters
    if any(b["prune_pct"] for b in expr["blocks"]):
        assert macs < full_macs and params < info.parameters
    assert h4.recompute_cost(expr) == (macs, params)  # deterministic: the widths, not the weights, decide
    assert math.isfinite(macs)


# -- benchmark versions are never mixed ---------------------------------------------------------------------------------


def test_runs_of_two_bench_versions_are_refused_unless_one_is_chosen(tmp_path: Path) -> None:
    db = build(tmp_path, bench={"joint": "tcc-bench/preliminary-v3", "random": "tcc-bench/preliminary-v4"})
    with pytest.raises(Exception, match="different benchmark versions"):
        h4.collect(db, allow_synthetic=True, cost=fake_cost)
    kept = h4.collect(db, allow_synthetic=True, cost=fake_cost, bench_version="tcc-bench/preliminary-v4")
    assert kept.runs == ("random-b0",) and kept.bench_version == "tcc-bench/preliminary-v4"
    assert kept.left_out == {"joint-b0": "tcc-bench/preliminary-v3"} and len(kept.points) == 24
    with pytest.raises(Exception, match="no run was measured by benchmark version"):
        h4.collect(db, allow_synthetic=True, cost=fake_cost, bench_version="tcc-bench/other")
    manifest = h4.export_h4(
        db, tmp_path / "out", allow_synthetic=True, cost=fake_cost, bench_version="tcc-bench/preliminary-v3"
    )
    assert manifest["bench_version"] == "tcc-bench/preliminary-v3" and manifest["left_out_by_bench_version"] == {
        "random-b0": "tcc-bench/preliminary-v4"
    }


def test_the_hypervolume_analysis_and_the_cli_apply_the_same_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cnnlab.tcc.operation import hv_analysis

    db = build(tmp_path, bench={"joint": "tcc-bench/preliminary-v3", "random": "tcc-bench/preliminary-v4"})
    args = ["campaign", "curves", "--db", str(db), "--allow-synthetic", "--reference", "1,10000,100000,100000"]
    assert cli.main([*args, "--dest", str(tmp_path / "a")]) == EXIT_REFUSED
    assert "different benchmark versions" in capsys.readouterr().err and not (tmp_path / "a").exists()
    assert cli.main([*args, "--dest", str(tmp_path / "b"), "--bench-version", "tcc-bench/preliminary-v4"]) == 0
    manifest = json.loads((tmp_path / "b" / hv_analysis.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert [r["run_id"] for r in manifest["runs"]] == ["random-b0"] and manifest[
        "bench_version"
    ] == "tcc-bench/preliminary-v4"
    assert manifest["left_out_by_bench_version"] == {"joint-b0": "tcc-bench/preliminary-v3"}
