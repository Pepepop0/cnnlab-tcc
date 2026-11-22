"""The run auditor on a CPU run whose engine and benchmark are stand-ins (labelled, not measurements).

Training and ONNX export are real (CPU, a few synthetic images); the TensorRT
build and the isolated benchmark are replaced by deterministic stand-ins that
write files with the same structure, so the auditor's cross-checks and its
tamper detection can be exercised without a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_pipeline_training import cpu_data, inspector_from_onnx  # same directory

from cnnlab.tcc.model_pipeline import evaluator as evaluator_mod
from cnnlab.tcc.model_pipeline.engine import BuiltEngine
from cnnlab.tcc.model_pipeline.evaluator import EngineEvaluator, EvaluatorConfig, pilot_cache_context
from cnnlab.tcc.model_pipeline.pilot_demo import AttemptOpener, bootstrap
from cnnlab.tcc.model_pipeline.pilot_wiring import PilotScience, split_descriptors
from cnnlab.tcc.model_pipeline.trainer import TrainConfig
from cnnlab.tcc.model_pipeline.verify_run import verify
from cnnlab.tcc.science.operators import FP32_SLICE
from cnnlab.tcc.science.search import DirectContext, DirectHashes, DirectResume, SearchConfig, SearchRunner
from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore

RUN_ID = "run-audit-cpu"


def standin_build(onnx_path: Path, plan_path: Path, workspace_bytes: int = 0) -> BuiltEngine:
    """Stand-in plan: bytes derived from the ONNX graph, so its size differs between architectures."""
    graph = onnx_path.read_bytes()
    plan = hashlib.sha256(graph).digest() * (len(graph) // 64)
    plan_path.write_bytes(plan)
    return BuiltEngine(
        plan_path, len(plan), hashlib.sha256(plan).hexdigest(), inspector_from_onnx(onnx_path), {"standin": True}, 0.0
    )


def standin_bench(
    plan_path: Path, out_path: Path, *, val_path: Path, ref_logits_path: Path, **_: Any
) -> dict[str, Any]:
    """Stand-in report with the real structure; E comes from the PyTorch reference logits."""
    size = plan_path.stat().st_size
    labels = np.load(val_path)["y"]
    predicted = np.load(ref_logits_path).argmax(1)
    assert len(predicted) == len(labels), "the evaluator must hand over logits of every validation image"
    errors = int((predicted != labels).sum())
    base = 0.05 + size / 4e6
    samples = [base * (1 + 0.02 * ((i * 7919) % 11)) for i in range(101)]
    ordered = sorted(samples)
    report = {
        "bench_version": "cpu-standin (not a measurement)",
        "warmup": 0,
        "iters": len(samples),
        "latency": {
            "median_ms": ordered[50],
            "p5_ms": ordered[5],
            "p95_ms": ordered[95],
            "iqr_ms": ordered[75] - ordered[25],
            "mean_ms": sum(samples) / len(samples),
            "n": 101.0,
        },
        "latency_samples_ms": samples,
        "memory": {
            "v_bytes_allocator_peak": 3 * size,
            "trt_allocator_peak_live_bytes": 3 * size - 64,
            "io_buffer_bytes": 64,
            "free_mem_drop_bytes_audit": 0,
        },
        "e": {"e_fraction": errors / len(labels), "errors": errors, "n": int(len(labels)), "all_finite": True},
        "quality": {"parity_vs_torch_fp32": {"n": int(len(labels)), "max_abs_diff": 0.0, "top1_agreement": 1.0}},
    }
    out_path.write_text(json.dumps(report), encoding="utf-8")
    return report


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One CPU run (population 4, one offspring generation) shared by the tests, which work on copies."""
    base = tmp_path_factory.mktemp("audit")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(evaluator_mod, "build_engine", standin_build)
    patcher.setattr(evaluator_mod, "run_isolated", standin_bench)
    try:
        data = cpu_data()
        manifest = {
            "fingerprints": {"train": "a" * 64, "val": "b" * 64},
            "n_train_subset": 32,
            "n_val": 16,
            "official_test_read": False,
            "spec": {"standin": True},
        }
        data = type(data)(data.x_train, data.y_train, data.x_val, data.y_val, manifest)
        evaluator_config = EvaluatorConfig(train=TrainConfig(epochs=1, batch_size=16))
        search_config = SearchConfig(population=4, generations=1)
        space_hash = FP32_SLICE.space_hash()

        def context_for(job: Any) -> Any:
            return pilot_cache_context(
                data, evaluator_config, job.train_seed, "test-code", space_hash, {"standin": True}
            )

        science = PilotScience(
            runner=SearchRunner(search_config, FP32_SLICE, context_for),
            evaluator=EngineEvaluator(
                data, base / "work" / "val.npz", evaluator_config, device="cpu", work_root=base / "work" / "attempts"
            ),
            data=data,
            dataset_id="cifar100-pilot-standin",
            dataset_manifest=dict(manifest),
            splits=split_descriptors(data),
            space_hash=space_hash,
            runner_label="r",
            evaluator_label="e",
            environment={"backend": {"standin": True}, "code_version": "test-code"},
            search_config=search_config,
            evaluator_config=evaluator_config,
            code_version="test-code",
            arguments={"cifar_root": "x", "work_dir": "y"},
        )
        store = SQLiteRunStore(base / "db" / "run.sqlite", base / "artifacts")
        args = argparse.Namespace(run_id=RUN_ID, search_seed=1001, train_seed=2001, budget_gpu_s=600.0)
        job = bootstrap(store, science, args, {"python": "test"})
        hashes = store.run_hashes(RUN_ID)
        ctx = DirectContext(
            job=job,
            store=store,
            evaluator=AttemptOpener(science.evaluator, store),
            resume=DirectResume(),
            hashes=DirectHashes(hashes.protocol_hash, hashes.plan_hash, hashes.space_hash),
            artifact_root=store.artifact_root,
            emit=lambda event: None,
            should_stop=lambda: False,
            active_gpu_seconds=lambda: store.active_gpu_seconds(RUN_ID),
            unique_valid_evaluations=lambda: store.unique_valid_evaluations(RUN_ID),
        )
        assert science.runner(ctx) == "completed"
        assert store.check_integrity(hash_files=True).ok
        store.close()
        yield base
    finally:
        patcher.undo()


def copy_of(run: Path, tmp_path: Path) -> tuple[Path, Path]:
    target = tmp_path / "copy"
    shutil.copytree(run / "db", target / "db")
    shutil.copytree(run / "artifacts", target / "artifacts")
    return target / "db" / "run.sqlite", target / "artifacts"


def first_artifact(root: Path, name: str) -> Path:
    return sorted(root.rglob(name))[0]


def test_consistent_run_passes_every_check(finished_run: Path, tmp_path: Path) -> None:
    db, root = copy_of(finished_run, tmp_path)
    report = verify(db, root, RUN_ID)
    assert report["ok"], report["problems"]
    assert report["counts"]["candidates"] == report["counts"]["valid"] == 8
    assert report["counts"]["unresolved_candidates"] == [] and report["counts"]["running"] == 0
    assert [(g["generation"], g["status"], g["attempts"]) for g in report["by_generation"]] == [
        (0, "valid", 4),
        (1, "valid", 4),
    ]
    assert report["plan"]["confirmatory"] is False and report["dataset"]["official_test_read"] is False
    assert report["dataset"]["n_train_subset"] == 32 and report["system"]["backend"] == {"standin": True}
    assert report["test_evaluations_rows"] == 0
    assert report["cross_check"] == {"archives": 2, "survivor_sets": 2}
    offspring = [a for a in report["attempts"] if a["generation"] == 1]
    assert all(a["origin"] == "offspring" and len(a["parents"]) == 2 for a in offspring)
    assert all(a["latency_samples"] == 101 and a["train"]["epochs"] == 1 for a in report["attempts"])
    assert len({a["S_mb"] for a in report["attempts"]}) > 1  # the stand-in varies with the architecture


def test_the_audit_is_read_only(finished_run: Path, tmp_path: Path) -> None:
    db, root = copy_of(finished_run, tmp_path)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    verify(db, root, RUN_ID)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("name", ["bench.json", "engine.plan", "train.json", "model.onnx"])
def test_changed_artifact_is_reported(finished_run: Path, tmp_path: Path, name: str) -> None:
    db, root = copy_of(finished_run, tmp_path)
    victim = first_artifact(root, name)
    victim.write_bytes(victim.read_bytes() + b" ")
    report = verify(db, root, RUN_ID)
    assert not report["ok"]
    assert any("differs from its cataloged hash or size" in p for p in report["problems"])


def test_missing_artifact_is_reported(finished_run: Path, tmp_path: Path) -> None:
    db, root = copy_of(finished_run, tmp_path)
    first_artifact(root, "engine.plan").unlink()
    report = verify(db, root, RUN_ID)
    assert not report["ok"] and any("missing on disk" in p for p in report["problems"])


def test_unknown_run_is_reported(finished_run: Path, tmp_path: Path) -> None:
    db, root = copy_of(finished_run, tmp_path)
    report = verify(db, root, "no-such-run")
    assert not report["ok"] and "does not exist" in report["problems"][0]


def test_sql_value_that_disagrees_with_the_raw_report_is_reported(finished_run: Path, tmp_path: Path) -> None:
    """A measurement edited in the database (triggers dropped by hand) no longer matches the raw files."""
    import sqlite3

    db, root = copy_of(finished_run, tmp_path)
    conn = sqlite3.connect(db)
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'measurements'"
    ).fetchall():
        conn.execute(f'DROP TRIGGER "{name}"')
    conn.execute(
        "UPDATE measurements SET l_ms = l_ms * 0.5 WHERE attempt_id = (SELECT MIN(attempt_id) FROM measurements)"
    )
    conn.commit()
    conn.close()
    report = verify(db, root, RUN_ID)
    assert not report["ok"]
    assert any("L_ms in SQL" in p and "raw artifacts give" in p for p in report["problems"])


def set_quality(db: Path, **changes: Any) -> None:
    """Edit the stored quality of every valid attempt (triggers dropped by hand, as above)."""
    import sqlite3

    conn = sqlite3.connect(db)
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'evaluation_attempts'"
    ).fetchall():
        conn.execute(f'DROP TRIGGER "{name}"')
    for attempt_id, text, e in conn.execute(
        "SELECT a.attempt_id, a.quality_json, m.e_fraction FROM evaluation_attempts a JOIN measurements m "
        "ON m.attempt_id = a.attempt_id"
    ).fetchall():
        quality = json.loads(text)
        quality.update({k: (v(e) if callable(v) else v) for k, v in changes.items()})
        conn.execute(
            "UPDATE evaluation_attempts SET quality_json = ? WHERE attempt_id = ?", (json.dumps(quality), attempt_id)
        )
    conn.commit()
    conn.close()


def test_an_int8_attempt_is_compared_with_the_simulated_quantized_network(finished_run: Path, tmp_path: Path) -> None:
    """After QAT the engine runs another network than the FP32 one before quantization: the diagnostic that
    describes it is the simulated Q/DQ accuracy, and the plain one must not make the audit fail."""
    db, root = copy_of(finished_run, tmp_path)
    set_quality(db, torch_val_accuracy_diagnostic=0.9, torch_val_accuracy_quantized_diagnostic=lambda e: 1 - e)
    report = verify(db, root, RUN_ID)
    assert report["ok"], report["problems"]
    assert all(a["torch_val_accuracy_quantized_diagnostic"] is not None for a in report["attempts"])


def test_an_int8_attempt_whose_simulation_disagrees_with_the_engine_is_reported(
    finished_run: Path, tmp_path: Path
) -> None:
    db, root = copy_of(finished_run, tmp_path)
    set_quality(db, torch_val_accuracy_quantized_diagnostic=lambda e: 1 - e - 0.2)
    report = verify(db, root, RUN_ID)
    assert not report["ok"] and any("simulated Q/DQ network" in p for p in report["problems"])
