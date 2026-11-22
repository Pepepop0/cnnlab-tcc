"""CPU tests of the pilot pipeline pieces: splits, trainer, benchmark statistics and evaluator orchestration.

Data here are synthetic tensors; no CIFAR-100 file is read. The evaluator
tests replace the TensorRT build and the isolated benchmark with fakes that
are labelled synthetic, so they check orchestration and failure mapping, not
measurements.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from cnnlab.tcc.contracts import EvaluationRequest
from cnnlab.tcc.model_pipeline import evaluator as evaluator_mod
from cnnlab.tcc.model_pipeline.bench import latency_stats
from cnnlab.tcc.model_pipeline.builder import build_model
from cnnlab.tcc.model_pipeline.demo_data import DemoData, prepare_demo_data
from cnnlab.tcc.model_pipeline.engine import BuiltEngine, EngineBuildError
from cnnlab.tcc.model_pipeline.evaluator import (
    EngineEvaluator,
    EvaluatorConfig,
    pilot_cache_context,
    representation_problems,
)
from cnnlab.tcc.model_pipeline.splits import DemoSplitSpec, poc_stratified_split, stratified_subset
from cnnlab.tcc.model_pipeline.trainer import (
    TrainConfig,
    TrainingDiverged,
    augment,
    lr_factor,
    normalize_images,
    train_model,
)
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.operators import FP32_SLICE, prepare_in_slice
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.space import SPACE_V1

# --- splits --------------------------------------------------------------------------


def synthetic_labels(per_class: int = 30, classes: int = 100, seed: int = 0) -> torch.Tensor:
    labels = torch.arange(classes).repeat_interleave(per_class)
    return labels[torch.randperm(labels.numel(), generator=torch.Generator().manual_seed(seed))]


def test_poc_split_matches_the_poc_implementation() -> None:
    from cnnlab.core.data import _stratified_split

    labels = synthetic_labels(50)
    train, val = poc_stratified_split(labels)
    poc_train, poc_val = _stratified_split(labels, 0.1, 100)
    assert train == poc_train.tolist() and val == poc_val.tolist()
    assert not set(train) & set(val) and len(train) + len(val) == labels.numel()


def test_stratified_subset_is_balanced_deterministic_and_inside_pool() -> None:
    labels = synthetic_labels(40)
    train, _ = poc_stratified_split(labels)
    a = stratified_subset(train, labels, 10, seed=5)
    assert a == stratified_subset(train, labels, 10, seed=5)
    assert a != stratified_subset(train, labels, 10, seed=6)
    assert set(a) <= set(train) and len(a) == len(set(a)) == 1000
    assert torch.bincount(labels[a], minlength=100).tolist() == [10] * 100
    with pytest.raises(ValueError):
        stratified_subset(train, labels, 1000, seed=5)


def test_split_spec_definition_is_frozen_json() -> None:
    d = DemoSplitSpec(train_per_class=100, subset_seed=1).definition()
    assert d["split_seed"] == 20260901 and d["val_fraction"] == 0.1 and d["train_per_class"] == 100


class FakeTrainFile:
    """50k-image stand-in for the official train file (tiny images, labels 0..99)."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    def read(self, official_file: str) -> tuple[torch.Tensor, torch.Tensor]:
        self.reads.append(official_file)
        assert official_file == "train", "the demo must never read the official test file"
        labels = torch.arange(100).repeat_interleave(500)
        images = (labels.view(-1, 1, 1, 1) % 256).to(torch.uint8).expand(-1, 32, 32, 3).contiguous()
        return images, labels


def test_demo_data_reads_only_the_train_file_and_freezes_the_manifest() -> None:
    source = FakeTrainFile()
    data = prepare_demo_data(source, DemoSplitSpec(train_per_class=3, subset_seed=9), "run-x", torch.device("cpu"))
    assert set(source.reads) == {"train"}  # labels, then images; the real reader caches
    assert data.x_train.shape == (300, 3, 32, 32) and data.x_val.shape == (5000, 3, 32, 32)
    assert data.manifest["official_test_read"] is False and data.manifest["n_train_full_split"] == 45000
    assert set(data.manifest["fingerprints"]) == {"train", "val"}
    again = prepare_demo_data(
        FakeTrainFile(), DemoSplitSpec(train_per_class=3, subset_seed=9), "run-y", torch.device("cpu")
    )
    assert again.fingerprint == data.fingerprint
    other = prepare_demo_data(
        FakeTrainFile(), DemoSplitSpec(train_per_class=3, subset_seed=10), "run-y", torch.device("cpu")
    )
    assert other.fingerprint != data.fingerprint


def test_full_split_spec_trains_on_every_index_in_the_order_of_the_poc_split() -> None:
    spec = DemoSplitSpec(train_per_class=None, subset_seed=1)
    d = spec.definition()
    assert d["train_per_class"] is None and d["subset_policy"] == "tcc-subset/none-full-45k-v1"
    assert d["subset_seed"] is None  # the seed does nothing here, so it is not part of the identity
    assert DemoSplitSpec(train_per_class=None, subset_seed=2).definition() == d
    assert DemoSplitSpec(train_per_class=5, subset_seed=1).definition()["subset_policy"] == "tcc-subset/stratified-v1"

    source = FakeTrainFile()  # 50k images: 45000 train, 5000 validation, as the real file
    full = prepare_demo_data(source, spec, "run-x", torch.device("cpu"))
    assert set(source.reads) == {"train"} and full.manifest["official_test_read"] is False
    assert full.manifest["train_is_full_split"] is True
    assert full.manifest["n_train_subset"] == full.manifest["n_train_full_split"] == 45000
    assert full.x_train.shape == (45000, 3, 32, 32) and full.x_val.shape == (5000, 3, 32, 32)
    _, labels = FakeTrainFile().read("train")
    train_idx, val_idx = poc_stratified_split(labels)
    assert full.y_train.tolist() == labels[train_idx].tolist()  # no resampling, no reordering
    assert not set(train_idx) & set(val_idx)

    # 450 per class is also 45000 images, but drawn and shuffled by the subset seed: another dataset.
    subset = prepare_demo_data(
        FakeTrainFile(), DemoSplitSpec(train_per_class=450, subset_seed=1), "run-x", torch.device("cpu")
    )
    assert subset.manifest["n_train_subset"] == 45000 and "train_is_full_split" not in subset.manifest
    assert subset.fingerprint != full.fingerprint
    again = prepare_demo_data(
        FakeTrainFile(), DemoSplitSpec(train_per_class=None, subset_seed=77), "run-y", torch.device("cpu")
    )
    assert again.fingerprint == full.fingerprint


# --- trainer ---------------------------------------------------------------------------


def test_normalize_images_layout_and_stats() -> None:
    x = normalize_images(torch.full((2, 32, 32, 3), 255, dtype=torch.uint8))
    assert x.shape == (2, 3, 32, 32) and x.dtype == torch.float32
    assert x[0, 0, 0, 0] == pytest.approx((1 - 0.5071) / 0.2673, rel=1e-5)


def test_augment_keeps_shape_and_is_seeded() -> None:
    x = torch.randn(8, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    a = augment(x, 4, torch.Generator().manual_seed(1))
    b = augment(x, 4, torch.Generator().manual_seed(1))
    assert a.shape == x.shape and torch.equal(a, b)
    assert not torch.equal(a, augment(x, 4, torch.Generator().manual_seed(2)))
    zero_pad = augment(x, 0, torch.Generator().manual_seed(3))  # only flips remain
    assert all(torch.equal(zero_pad[i], x[i]) or torch.equal(zero_pad[i], x[i].flip(2)) for i in range(8))


def test_lr_schedule_warms_up_then_decays_to_zero() -> None:
    assert lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100) == pytest.approx(0.0, abs=1e-12)
    assert lr_factor(55, 10, 100) == pytest.approx(0.5)


def tiny_genome() -> Any:
    genome = FP32_SLICE.sample(GeneRng(3))
    return express(genome, SPACE_V1).genome


def fixed_data(n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    y = torch.arange(n) % 4
    x = torch.randn(n, 3, 32, 32, generator=g) + y.view(n, 1, 1, 1).float()
    return x, y


def test_training_is_deterministic_and_learns_on_cpu() -> None:
    genome = tiny_genome()
    x, y = fixed_data()
    cfg = TrainConfig(epochs=3, batch_size=16, warmup_epochs=1)
    reports, weights = [], []
    for _ in range(2):
        model, _ = build_model(genome, 1)
        reports.append(train_model(model, x, y, log10_lr=-3.0, log10_wd=-4.0, config=cfg, seed=5))
        weights.append(model.stem[0].weight.detach().clone())
    assert reports[0].epoch_loss == reports[1].epoch_loss
    assert torch.equal(weights[0], weights[1])
    assert reports[0].steps == 3 * 4 and reports[0].epoch_loss[-1] < reports[0].epoch_loss[0]
    assert all(math.isfinite(v) for v in reports[0].epoch_loss)


def test_training_leaves_the_global_rng_untouched() -> None:
    model, _ = build_model(tiny_genome(), 1)
    x, y = fixed_data(16)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    train_model(model, x, y, log10_lr=-3.0, log10_wd=-4.0, config=TrainConfig(epochs=1, batch_size=8), seed=0)
    assert torch.equal(torch.get_rng_state(), before)


def test_nonfinite_loss_raises_training_diverged() -> None:
    model, _ = build_model(tiny_genome(), 1)
    x, y = fixed_data(16)
    x[0] = float("nan")  # a whole image: a random crop cannot remove it
    with pytest.raises(TrainingDiverged):
        train_model(model, x, y, log10_lr=-3.0, log10_wd=-4.0, config=TrainConfig(epochs=1, batch_size=16), seed=0)


# --- benchmark statistics ----------------------------------------------------------------


def test_latency_stats_median_and_quantiles() -> None:
    stats = latency_stats([float(v) for v in range(1, 101)])
    assert stats["median_ms"] == 50.5 and stats["n"] == 100.0
    assert stats["p5_ms"] == pytest.approx(5.95) and stats["p95_ms"] == pytest.approx(95.05)
    assert stats["iqr_ms"] == pytest.approx(49.5)


# --- evaluator orchestration with synthetic engine/benchmark fakes ------------------------------------


def cpu_data() -> DemoData:
    x, y = fixed_data(32)
    return DemoData(x, y, x[:16].clone(), y[:16].clone(), {"schema": "synthetic", "official_test_read": False})


def request_for(
    genome_seed: int, tmp_path: Path, config: EvaluatorConfig, data: DemoData, slice_=FP32_SLICE
) -> EvaluationRequest:
    ctx = pilot_cache_context(data, config, 11, "test", slice_.space_hash(), backend={"synthetic": True})
    record = prepare_in_slice(slice_.sample(GeneRng(genome_seed)), slice_, ctx).record
    return EvaluationRequest(
        "run1", "c1", "c1.a1", "pilot", record, 11, 0, "ds", "env", "proto", "plan", tmp_path / "artifacts"
    )


def inspector_from_onnx(onnx_path: Path, executed: dict[str, str] | None = None) -> dict[str, Any]:
    """Stand-in inspector: one engine layer per weighted ONNX node, typed like its weights.

    ``executed`` overrides the type of chosen nodes, to imitate an engine
    that ran something else than the graph asked for.
    """
    from cnnlab.tcc.model_pipeline.precision import weight_layers

    engine_type = {"FLOAT": "Float", "FLOAT16": "Half", "BFLOAT16": "BFloat16", "INT8": "Int8"}
    layers, _ = weight_layers(onnx_path)
    out = []
    for index, layer in enumerate(layers):
        dtype = (executed or {}).get(layer["node"], engine_type[layer["onnx_weight_dtypes"][0]])
        entry = {
            "Name": f"{layer['node']}_myl0_{index}",
            "LayerType": "correlation",
            "Inputs": [{"Datatype": dtype}],
            "Outputs": [{"Datatype": dtype}],
            "TacticName": "stand-in",
            "Metadata": f"[ONNX Layer: {layer['node']}]",
        }
        if layer["onnx_weight_dtypes"] == ["INT8"]:  # Q/DQ layers keep INT8 weights inside the engine layer
            entry["Constants"] = [{"Name": "w", "Dimensions": [4, 4], "Datatype": engine_type["INT8"]}]
        out.append(entry)
    return {"Layers": out}


def fake_build(onnx_path: Path, plan_path: Path, workspace_bytes: int = 0) -> BuiltEngine:
    plan_path.write_bytes(b"synthetic-plan-bytes" * 50)
    return BuiltEngine(plan_path, 1000, "0" * 64, inspector_from_onnx(onnx_path), {"synthetic": True}, 0.0)


def fake_bench(plan_path: Path, out_path: Path, **_: Any) -> dict[str, Any]:
    report = {
        "bench_version": "synthetic",
        "latency": {"median_ms": 0.25},
        "memory": {"v_bytes_allocator_peak": 2_500_000},
        "e": {"e_fraction": 0.75, "errors": 12, "n": 16, "all_finite": True},
        "quality": {},
    }
    out_path.write_text(json.dumps(report), encoding="utf-8")
    return report


@pytest.fixture
def cpu_evaluator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[EngineEvaluator, DemoData, EvaluatorConfig]:
    monkeypatch.setattr(evaluator_mod, "build_engine", fake_build)
    monkeypatch.setattr(evaluator_mod, "run_isolated", fake_bench)
    config = EvaluatorConfig(train=TrainConfig(epochs=1, batch_size=16))
    data = cpu_data()
    return EngineEvaluator(data, tmp_path / "val.npz", config, device="cpu", work_root=tmp_path / "work"), data, config


def test_valid_result_units_and_published_artifacts(cpu_evaluator: Any, tmp_path: Path) -> None:
    evaluator, data, config = cpu_evaluator
    request = request_for(3, tmp_path, config, data)
    result = evaluator.evaluate(request)
    assert result.status == "valid" and result.failure_stage is None
    assert result.fitness is not None
    assert (result.fitness.e, result.fitness.l_ms) == (0.75, 0.25)
    assert result.fitness.v_mb == pytest.approx(2.5) and result.fitness.s_mb == pytest.approx(1000 / 1e6)
    assert result.quality["preliminary"] and result.quality["not_confirmatory"]
    assert {"engine", "bench", "train_log", "onnx", "inspector"} <= set(result.artifact_paths)
    for kind, rel in result.artifact_paths.items():
        assert rel.startswith("runs/run1/attempts/c1.a1/")
        import hashlib

        assert hashlib.sha256((request.artifact_root / rel).read_bytes()).hexdigest() == result.artifact_sha256[kind]
    assert not (tmp_path / "work" / "run1" / "c1.a1").exists()  # scratch removed


def test_build_failure_is_invalid_without_fitness(
    cpu_evaluator: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator, data, config = cpu_evaluator

    def failing(*_: Any, **__: Any) -> BuiltEngine:
        raise EngineBuildError("trt_parse", "Only FLOAT, HALF or INT8 input is supported for HardSwish")

    monkeypatch.setattr(evaluator_mod, "build_engine", failing)
    result = evaluator.evaluate(request_for(3, tmp_path, config, data))
    assert result.status == "invalid" and result.fitness is None
    assert (result.failure_stage, result.failure_code) == ("trt_parse", "trt_parse_failed")
    assert result.gpu_seconds > 0 and "train" in result.durations_s
    assert not result.artifact_paths


def test_nonfinite_engine_output_is_invalid(
    cpu_evaluator: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator, data, config = cpu_evaluator

    def nan_bench(plan_path: Path, out_path: Path, **_: Any) -> dict[str, Any]:
        return {
            "bench_version": "synthetic",
            "latency": {"median_ms": 0.2},
            "memory": {"v_bytes_allocator_peak": 1},
            "e": {"e_fraction": 0.5, "all_finite": False},
            "quality": {},
        }

    monkeypatch.setattr(evaluator_mod, "run_isolated", nan_bench)
    result = evaluator.evaluate(request_for(3, tmp_path, config, data))
    assert (result.status, result.failure_code) == ("invalid", "nonfinite_engine_output")


def test_unsupported_representation_is_invalid_before_training(cpu_evaluator: Any, tmp_path: Path) -> None:
    evaluator, data, config = cpu_evaluator
    from dataclasses import replace

    from cnnlab.tcc.science.operators import SearchSlice

    evaluator.config = config = replace(config, calibration_examples=16)  # the 32 synthetic images, if INT8 is drawn
    free_slice = SearchSlice(SPACE_V1, (), (), "test: no pins")
    request = request_for(1, tmp_path, config, data, slice_=free_slice)
    from cnnlab.tcc.science.genome import ExpressedGenome

    problems = representation_problems(ExpressedGenome.from_dict(request.genome.expressed_genome))
    result = evaluator.evaluate(request)
    if problems:
        assert (result.status, result.failure_stage) == ("invalid", "representation")
        assert "train" not in result.durations_s
    else:  # the draw happened to be fp32/unpruned
        assert result.status == "valid"


def test_representation_problems_lists_every_unsupported_gene() -> None:
    genome = express(FP32_SLICE.sample(GeneRng(3)), SPACE_V1).genome
    assert representation_problems(genome) == []
    import dataclasses

    changed = dataclasses.replace(
        genome, stem_precision="fp8", blocks=(dataclasses.replace(genome.blocks[0], prune_pct=25), *genome.blocks[1:])
    )
    assert representation_problems(changed) == [
        "stem_precision=fp8"
    ]  # pruning is applied by the evaluator, not refused


def test_deterministic_option_is_part_of_the_policy_and_restores_settings() -> None:
    assert TrainConfig().definition()["deterministic"] is False
    assert TrainConfig(deterministic=True).definition() != TrainConfig().definition()
    before = (torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled())
    x, y = fixed_data(16)
    weights = []
    for _ in range(2):
        model, _ = build_model(tiny_genome(), 1)
        train_model(
            model,
            x,
            y,
            log10_lr=-3.0,
            log10_wd=-4.0,
            config=TrainConfig(epochs=1, batch_size=8, deterministic=True),
            seed=4,
        )
        weights.append(model.stem[0].weight.detach().clone())
    assert torch.equal(weights[0], weights[1])
    assert (torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled()) == before
