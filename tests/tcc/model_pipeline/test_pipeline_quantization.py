"""INT8 post-training quantization with explicit Q/DQ (task G3): calibration, Q/DQ layers, graph and engine check (CPU).

The ONNX export is real; the engine inspector is the stand-in written from the
graph. What TensorRT does with the Q/DQ graph was measured on the GPU with
``precision_probe --matrix int8`` during development.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_pipeline_precision import evaluator_for  # same directory
from test_pipeline_training import inspector_from_onnx  # same directory
from torch import nn

from cnnlab.tcc.model_pipeline import quantization as Q
from cnnlab.tcc.model_pipeline.builder import build_model, fold_batchnorm
from cnnlab.tcc.model_pipeline.engine import export_network
from cnnlab.tcc.model_pipeline.precision import (
    check_engine_precision,
    region_precisions,
    representation_problems,
    typed_network,
    weight_layers,
)
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.operators import INT8_PTQ_SLICE, INT8_QAT_SLICE, SLICES
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.space import SPACE_V1


def int8_genome(
    stem: str = "int8",
    head: str = "int8",
    blocks: tuple[str, ...] = ("fp32", "int8", "fp32"),
    granularity: str = "per_channel",
    calibration: str = "max",
    mode: str = "ptq",
    activation: str = "relu",
    op: str = "standard",
    seed: int = 5,
) -> Any:
    raw = INT8_PTQ_SLICE.sample(GeneRng(seed))
    chosen = tuple(
        dataclasses.replace(
            b,
            precision=blocks[i] if i < len(blocks) else "fp32",
            op=op,
            kernel=3,
            n_convs=1,
            residual=False,
            downsample="none",
            channels=16,
        )
        for i, b in enumerate(raw.blocks)
    )
    raw = dataclasses.replace(
        raw,
        n_blocks=len(blocks),
        stem_precision=stem,
        head_precision=head,
        activation=activation,
        head="gap_linear",
        blocks=chosen,
        int8_mode=mode,
        weight_granularity=granularity,
        calibration=calibration,
    )
    assert INT8_PTQ_SLICE.contains(raw) or mode != "ptq"
    return express(raw, SPACE_V1).genome


def calibrated(genome: Any, n: int = 64, seed: int = 0) -> tuple[Any, Q.QuantizationPlan, torch.Tensor]:
    model, _ = build_model(genome, 1)
    model.eval()
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, 32, 32, generator=g)
    labels = torch.arange(n) % 10
    chosen = Q.calibration_indices(labels, n, seed)
    r = region_precisions(genome)
    plan = Q.plan_quantization(
        fold_batchnorm(model).float().eval(),
        Q.int8_regions(r["stem"], r["blocks"], r["head"]),
        images[chosen],
        genome.int8_mode or "ptq",
        genome.weight_granularity,
        genome.calibration or Q.QAT_INIT_CALIBRATION,
        chosen,
    )
    return model, plan, images


# --- the calibration set -----------------------------------------------------------------------------------


def test_calibration_indices_are_fixed_sorted_balanced_and_inside_the_training_labels() -> None:
    labels = np.repeat(np.arange(10), 200)  # 2,000 training examples
    idx = Q.calibration_indices(labels, 100, seed=3)
    assert idx == Q.calibration_indices(labels, 100, seed=3) and idx == sorted(idx)
    assert len(set(idx)) == 100 and 0 <= min(idx) and max(idx) < len(labels)
    assert set(np.bincount(labels[idx])) == {10}
    assert idx != Q.calibration_indices(labels, 100, seed=4)
    assert Q.indices_sha256(idx) == Q.indices_sha256(list(idx)) != Q.indices_sha256(idx[:-1])


def test_the_default_calibration_set_is_1024_examples() -> None:
    assert Q.N_CALIBRATION == 1024
    idx = Q.calibration_indices(np.arange(5000) % 100, seed=0)
    assert len(idx) == 1024 and len(set(idx)) == 1024


def test_too_few_training_examples_for_the_calibration_set_is_an_error() -> None:
    with pytest.raises(Q.QuantizationError) as info:
        Q.calibration_indices(np.arange(32), 1024)
    assert info.value.code == "calibration_too_small"


# --- the three calibration methods ---------------------------------------------------------------------------


def gaussian_histogram(outlier: float = 0.0) -> tuple[np.ndarray, float]:
    """Histogram of |x| for 200,000 N(0,1) values plus a few outliers, over [0, amax]."""
    rng = np.random.default_rng(0)
    values = np.abs(rng.standard_normal(200_000))
    values = np.concatenate([values, np.full(5, outlier)]) if outlier else values
    amax = float(values.max())
    hist, _ = np.histogram(values, bins=Q.HISTOGRAM_BINS, range=(0.0, amax))
    return hist.astype(np.float64), amax


def test_max_percentile_and_entropy_order_by_how_much_they_clip() -> None:
    hist, amax = gaussian_histogram(outlier=40.0)
    thresholds = {m: Q.threshold_from_histogram(hist, amax, m) for m in Q.CALIBRATIONS}
    assert thresholds["max"] == pytest.approx(40.0)
    assert 3.0 < thresholds["percentile_99_99"] < 5.5  # 99.99% of a standard normal sits near 3.9
    assert 2.0 < thresholds["entropy"] < thresholds["max"] / 2
    assert all(0 < t <= amax for t in thresholds.values())


def test_without_outliers_no_method_goes_far_below_the_data_range() -> None:
    hist, amax = gaussian_histogram()
    for method in ("percentile_99_99", "entropy"):
        assert Q.threshold_from_histogram(hist, amax, method) > 0.5 * amax


def test_thresholds_are_deterministic_and_unknown_methods_are_refused() -> None:
    hist, amax = gaussian_histogram(outlier=20.0)
    assert Q.threshold_from_histogram(hist, amax, "entropy") == Q.threshold_from_histogram(hist, amax, "entropy")
    with pytest.raises(Q.QuantizationError, match="unknown_calibration"):
        Q.threshold_from_histogram(hist, amax, "minmax")


def test_a_constant_zero_activation_does_not_break_the_scale() -> None:
    assert Q.threshold_from_histogram(np.zeros(Q.HISTOGRAM_BINS), 0.0, "entropy") >= Q.MIN_RANGE
    assert Q.threshold_from_histogram(np.zeros(Q.HISTOGRAM_BINS), 0.0, "max") >= Q.MIN_RANGE


@pytest.mark.parametrize("method", Q.CALIBRATIONS)
def test_calibrate_sees_the_input_of_the_layers_of_the_int8_regions_only(method: str) -> None:
    genome = int8_genome(stem="int8", head="fp32", blocks=("fp32", "int8", "fp32"), calibration=method)
    _, plan, _ = calibrated(genome)
    assert set(plan.regions) == {"stem", "blocks.1"}
    assert plan.input_ranges and all(k.startswith(("stem.", "blocks.1.")) for k in plan.input_ranges)
    assert all(v > 0 for v in plan.input_ranges.values())
    _, again, _ = calibrated(genome)
    assert dict(again.input_ranges) == dict(plan.input_ranges) and again.calibration_sha256 == plan.calibration_sha256


def test_methods_give_the_ranges_in_the_expected_order() -> None:
    ranges = {m: calibrated(int8_genome(calibration=m))[1].input_ranges for m in Q.CALIBRATIONS}
    for name, top in ranges["max"].items():
        assert ranges["percentile_99_99"][name] <= top + 1e-6 and ranges["entropy"][name] <= top + 1e-6


# --- Q/DQ layers ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("granularity", Q.GRANULARITIES)
def test_qdq_layer_quantizes_inputs_and_weights_symmetrically(granularity: str) -> None:
    conv = nn.Conv2d(8, 16, 3, padding=1)
    layer = Q.QDQLayer(conv, input_range=4.0, granularity=granularity)
    assert layer.input_scale.item() == pytest.approx(4.0 / 127)
    expected = 16 if granularity == "per_channel" else 1
    assert layer.weight_scale.numel() == expected
    assert layer.weight_scale.max().item() == pytest.approx(conv.weight.abs().max().item() / 127)
    x = torch.randn(2, 8, 8, 8).clamp(-4, 4)
    y, ref = layer(x), conv(x)
    assert y.shape == ref.shape and 0 < float((y - ref).abs().max()) < 0.2 * float(ref.abs().max())
    levels = torch.unique(
        torch.round(
            layer.quantized_weight() / (layer.weight_scale.view(-1, 1, 1, 1) if expected > 1 else layer.weight_scale)
        )
    )
    assert levels.abs().max() <= 127


def test_qdq_layer_works_for_linear_and_refuses_other_layers_and_unknown_granularity() -> None:
    linear = nn.Linear(8, 4)
    out = Q.QDQLayer(linear, 2.0, "per_channel")(torch.randn(3, 8).clamp(-2, 2))
    assert out.shape == (3, 4)
    with pytest.raises(Q.QuantizationError, match="unsupported_layer"):
        Q.QDQLayer(nn.ReLU(), 1.0, "per_tensor")  # type: ignore[arg-type]
    with pytest.raises(Q.QuantizationError, match="unknown_granularity"):
        Q.QDQLayer(linear, 1.0, "per_group")


# --- the typed network and the exported graph ------------------------------------------------------------------------


@pytest.fixture(scope="module", params=Q.GRANULARITIES)
def exported(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any, Any]:
    genome = int8_genome(granularity=request.param)
    model, plan, _ = calibrated(genome)
    r = region_precisions(genome)
    typed = typed_network(model, r["blocks"], r["stem"], r["head"], plan)
    assert typed.casts == 0  # INT8 regions take and give FP32: Q/DQ, not casts
    return export_network(typed, tmp_path_factory.mktemp(f"int8-{request.param}")).onnx_path, genome, plan


def test_int8_regions_need_a_plan_calibrated_for_them() -> None:
    genome = int8_genome()
    model, plan, _ = calibrated(genome)
    r = region_precisions(genome)
    with pytest.raises(ValueError, match="quantization plan"):
        typed_network(model, r["blocks"], r["stem"], r["head"])
    other = dataclasses.replace(plan, regions=("stem",))
    with pytest.raises(ValueError, match="quantization plan"):
        typed_network(model, r["blocks"], r["stem"], r["head"], other)


def test_typed_network_does_not_touch_the_original_model() -> None:
    genome = int8_genome()
    model, plan, images = calibrated(genome)
    before = model(images[:2]).detach()
    r = region_precisions(genome)
    typed_network(model, r["blocks"], r["stem"], r["head"], plan)
    assert not any(isinstance(m, Q.QDQLayer) for m in model.modules())
    assert torch.equal(model(images[:2]).detach(), before)


def test_int8_output_stays_close_to_fp32_on_a_calibrated_network() -> None:
    genome = int8_genome(calibration="entropy")
    model, plan, images = calibrated(genome)
    r = region_precisions(genome)
    typed = typed_network(model, r["blocks"], r["stem"], r["head"], plan)
    with torch.no_grad():
        ref, out = model(images[:16]), typed(images[:16])
    assert torch.isfinite(out).all()
    assert float((ref - out).abs().max()) < 0.5 * float(ref.abs().max()) + 1e-3


def test_graph_has_qdq_on_inputs_and_weights_of_the_int8_regions_only(exported: tuple[Path, Any, Any]) -> None:
    path, genome, plan = exported
    layers, casts = weight_layers(path)
    by_region = {}
    for layer in layers:
        by_region.setdefault(layer["region"], []).append(layer)
    for region in ("stem", "blocks.1", "head"):
        assert all(l["onnx_weight_dtypes"] == ["INT8"] and l["input_qdq"] for l in by_region[region]), region
    for region in ("blocks.0", "blocks.2"):
        assert all(l["onnx_weight_dtypes"] == ["FLOAT"] and not l["input_qdq"] for l in by_region[region]), region
    assert not casts and len(plan.input_ranges) == sum(len(by_region[r]) for r in ("stem", "blocks.1", "head"))


def test_matching_engine_passes_and_confirms_int8(exported: tuple[Path, Any, Any]) -> None:
    path, genome, _ = exported
    r = region_precisions(genome)
    check = check_engine_precision(path, inspector_from_onnx(path), r["blocks"], r["stem"], r["head"])
    assert check["ok"], check["problems"]
    assert check["regions"]["stem"]["executed"] == ["Int8"] and check["regions"]["blocks.0"]["executed"] == ["Float"]
    assert {row["evidence"] for row in check["layers"] if row["requested"] == "int8"} == {"int8 input and weights"}


def test_engine_that_ran_an_int8_layer_in_float_is_reported(exported: tuple[Path, Any, Any]) -> None:
    path, genome, _ = exported
    r = region_precisions(genome)
    node = next(l["node"] for l in weight_layers(path)[0] if l["region"] == "blocks.1")
    check = check_engine_precision(path, inspector_from_onnx(path, {node: "Float"}), r["blocks"], r["stem"], r["head"])
    assert not check["ok"] and any("int8 requested, engine runs Float" in p for p in check["problems"])


def real_shaped(path: Path, region: str, compute_input: str = "Int8", weights: str = "Int8") -> dict[str, Any]:
    """Inspector shaped like TensorRT 11.3's for a Q/DQ layer (observed with ``precision_probe --matrix int8``).

    The activation Q is a helper kernel (Float -> Int8) that also lists the ONNX node, the convolution itself is a
    ``correlation`` layer (Int8 -> Float) whose constants are the Int8 weights plus Float 4-D Q/DQ scales.
    """
    out = inspector_from_onnx(path)
    node = next(l["node"] for l in weight_layers(path)[0] if l["region"] == region)
    index = next(i for i, l in enumerate(out["Layers"]) if node in l["Metadata"])
    layer = out["Layers"][index]
    helper = {
        "Name": "__myl_MulMinMaxRounCast",
        "LayerType": "kgen",
        "Inputs": [{"Datatype": "Float"}],
        "Outputs": [{"Datatype": "Int8"}],
        "Constants": [{"Name": "pad", "Dimensions": [1, 13, 32, 32], "Datatype": "Int8"}],
        "Metadata": layer["Metadata"],
    }
    layer.update(
        LayerType="correlation",
        Inputs=[{"Datatype": compute_input}],
        Outputs=[{"Datatype": "Float"}],
        Constants=[
            {"Name": "w", "Dimensions": [32, 32, 3, 3], "Datatype": weights},
            {"Name": "s", "Dimensions": [1, 32, 1, 1], "Datatype": "Float"},
            {"Name": "z", "Dimensions": [1], "Datatype": "Float"},
        ],
    )
    out["Layers"].insert(index, helper)
    return out


def test_real_shaped_inspector_is_judged_on_the_compute_layer_and_its_main_weights(
    exported: tuple[Path, Any, Any],
) -> None:
    path, genome, _ = exported
    r = region_precisions(genome)

    def check(**kwargs: str) -> dict[str, Any]:
        return check_engine_precision(path, real_shaped(path, "stem", **kwargs), r["blocks"], r["stem"], r["head"])

    ok = check()
    assert ok["ok"], ok["problems"]  # the Float scales do not count; the helper kernel does not hide the layer
    assert not check(compute_input="Float")["ok"]  # the helper's Int8 output must not vouch for the convolution
    assert not check(weights="Float")["ok"]


def test_int8_requested_against_a_graph_without_qdq_is_reported(tmp_path: Path) -> None:
    genome = int8_genome(stem="fp32", head="fp32", blocks=("fp32", "fp32", "fp32"))
    model, _ = build_model(genome, 1)
    exported = export_network(typed_network(model, ["fp32"] * 3), tmp_path).onnx_path
    check = check_engine_precision(exported, inspector_from_onnx(exported), ["fp32", "int8", "fp32"], "fp32", "fp32")
    assert not check["ok"] and any("blocks.1" in p for p in check["problems"])


# --- representation and the slice ---------------------------------------------------------------------------------


def test_int8_ptq_and_qat_are_materialized() -> None:
    assert representation_problems(int8_genome()) == []
    assert representation_problems(int8_genome(mode="qat")) == []
    plain = int8_genome(stem="fp32", head="fp32", blocks=("fp32", "fp16", "bf16"), mode="qat")
    assert representation_problems(plain) == []  # the INT8 genes are not expressed without an INT8 region
    broken = dataclasses.replace(int8_genome(), int8_mode=None)
    assert representation_problems(broken) == ["int8_mode=None"]


def test_the_slice_is_registered_and_pins_the_ptq_mode() -> None:
    assert SLICES["int8_ptq"] is INT8_PTQ_SLICE
    for seed in range(60):
        raw = INT8_PTQ_SLICE.sample(GeneRng(seed))
        assert raw.int8_mode == "ptq" and INT8_PTQ_SLICE.contains(raw)
        assert all(b.prune_pct == 0 for b in raw.blocks)
    assert not INT8_PTQ_SLICE.contains(dataclasses.replace(INT8_PTQ_SLICE.sample(GeneRng(1)), int8_mode="qat"))


def test_the_qat_slice_is_registered_and_pins_the_qat_mode() -> None:
    assert SLICES["int8_qat"] is INT8_QAT_SLICE and INT8_QAT_SLICE.space_hash() != INT8_PTQ_SLICE.space_hash()
    for seed in range(60):
        raw = INT8_QAT_SLICE.sample(GeneRng(seed))
        assert raw.int8_mode == "qat" and INT8_QAT_SLICE.contains(raw) and all(b.prune_pct == 0 for b in raw.blocks)
    assert not INT8_QAT_SLICE.contains(dataclasses.replace(INT8_QAT_SLICE.sample(GeneRng(1)), int8_mode="ptq"))


# --- the evaluator -------------------------------------------------------------------------------------------------


def evaluate_int8(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    executed: dict[str, str] | None = None,
    examples: int = 16,
    space: Any = INT8_PTQ_SLICE,
    **genes: Any,
) -> Any:
    from cnnlab.tcc.contracts import EvaluationRequest
    from cnnlab.tcc.model_pipeline.evaluator import pilot_cache_context
    from cnnlab.tcc.science.operators import prepare_in_slice

    evaluator, config, data = evaluator_for(monkeypatch, tmp_path, executed)
    config = dataclasses.replace(config, calibration_examples=examples)
    evaluator.config = config  # type: ignore[misc]
    ctx = pilot_cache_context(data, config, 11, "test", space.space_hash(), backend={"synthetic": True})
    raw = space.sample(GeneRng(5))
    blocks = tuple(dataclasses.replace(b, op="standard", precision="fp32", prune_pct=0) for b in raw.blocks)
    raw = dataclasses.replace(
        raw,
        stem_precision="int8",
        head_precision="fp32",
        activation="relu",
        blocks=(dataclasses.replace(blocks[0], precision="int8"), *blocks[1:]),
        **genes,
    )
    record = prepare_in_slice(raw, space, ctx).record
    request = EvaluationRequest(
        "run1", "c1", "c1.a1", "pilot", record, 11, 0, "ds", "env", "proto", "plan", tmp_path / "artifacts"
    )
    return evaluator.evaluate(request)


@pytest.mark.parametrize(
    ("granularity", "calibration"),
    [("per_channel", "max"), ("per_tensor", "entropy"), ("per_channel", "percentile_99_99")],
)
def test_evaluator_quantizes_calibrates_and_confirms_int8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, granularity: str, calibration: str
) -> None:
    result = evaluate_int8(monkeypatch, tmp_path, weight_granularity=granularity, calibration=calibration)
    assert result.status == "valid", (result.failure_code, result.failure_detail)
    quantization = result.quality["quantization"]
    assert (quantization["granularity"], quantization["calibration"]) == (granularity, calibration)
    assert quantization["calibration_examples"] == 16 and quantization["calibration_indices_sha256"]
    assert set(quantization["regions"]) == {"stem", "blocks.0"} and "quantize" in result.durations_s
    precision = result.quality["precision"]
    assert precision["ok"] and precision["regions"]["stem"]["executed"] == ["Int8"]


def test_calibration_set_comes_from_the_training_data_and_is_the_same_for_every_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    a = evaluate_int8(monkeypatch, tmp_path / "a", calibration="max").quality["quantization"]
    b = evaluate_int8(monkeypatch, tmp_path / "b", calibration="entropy").quality["quantization"]
    assert a["calibration_indices_sha256"] == b["calibration_indices_sha256"]


def test_engine_that_did_not_run_int8_makes_the_candidate_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = evaluate_int8(monkeypatch, tmp_path, executed={"stem": "Float"})
    assert (result.status, result.failure_stage, result.failure_code) == (
        "invalid",
        "precision_check",
        "precision_not_executed",
    )
    assert "int8 requested" in (result.failure_detail or "") and result.fitness is None


def test_calibration_set_larger_than_the_training_data_is_an_invalid_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = evaluate_int8(monkeypatch, tmp_path, examples=1024)
    assert (result.status, result.failure_stage, result.failure_code) == (
        "invalid",
        "quantize",
        "calibration_too_small",
    )
    assert result.fitness is None


# --- QAT (task G4) -----------------------------------------------------------------------------------------------------


def test_qat_layer_follows_the_batch_ranges_and_the_weights_only_while_training() -> None:
    conv = nn.Conv2d(4, 8, 3, padding=1)
    layer = Q.QDQLayer(conv, input_range=1.0, granularity="per_channel", qat=True)
    x = torch.randn(4, 4, 8, 8) * 5
    layer.eval()
    layer(x)
    assert layer.input_amax.item() == pytest.approx(1.0)  # evaluation leaves the ranges alone
    layer.train()
    layer(x)
    moved = Q.QAT_RANGE_MOMENTUM * 1.0 + (1 - Q.QAT_RANGE_MOMENTUM) * float(x.abs().max())
    assert layer.input_amax.item() == pytest.approx(moved) and layer.input_scale.item() == pytest.approx(moved / 127)
    with torch.no_grad():
        conv.weight.mul_(3.0)
    layer(x)
    assert layer.weight_scale.max().item() == pytest.approx(conv.weight.abs().max().item() / 127)


def test_ptq_layer_never_moves_its_scales() -> None:
    layer = Q.QDQLayer(nn.Conv2d(4, 8, 3, padding=1), input_range=1.0, granularity="per_tensor")
    before = (layer.input_scale.clone(), layer.weight_scale.clone())
    layer.train()
    layer(torch.randn(4, 4, 8, 8) * 5)
    assert torch.equal(layer.input_scale, before[0]) and torch.equal(layer.weight_scale, before[1])


def test_gradients_reach_weights_and_inputs_through_the_quantizers() -> None:
    layer = Q.QDQLayer(nn.Conv2d(4, 8, 3, padding=1), input_range=3.0, granularity="per_channel", qat=True).train()
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    layer(x).square().sum().backward()
    assert layer.layer.weight.grad is not None and float(layer.layer.weight.grad.abs().sum()) > 0
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


def test_qat_plan_makes_qat_layers_and_an_unknown_mode_is_refused() -> None:
    genome = int8_genome(mode="qat")
    model, plan, _ = calibrated(genome)
    assert plan.mode == "qat"
    net = fold_batchnorm(model)
    Q.insert_qdq(net, plan)
    assert all(m.qat for m in net.modules() if isinstance(m, Q.QDQLayer))
    with pytest.raises(Q.QuantizationError, match="unknown_mode"):
        Q.plan_quantization(
            fold_batchnorm(model), plan.regions, torch.randn(8, 3, 32, 32), "ste", "per_tensor", "max", list(range(8))
        )


def test_evaluator_fine_tunes_with_simulated_quantization_and_confirms_int8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = evaluate_int8(monkeypatch, tmp_path, space=INT8_QAT_SLICE, weight_granularity="per_channel")
    assert result.status == "valid", (result.failure_code, result.failure_detail)
    assert "qat" in result.durations_s and "quantize" in result.durations_s
    quantization = result.quality["quantization"]
    assert quantization["mode"] == "qat" and quantization["calibration"] == Q.QAT_INIT_CALIBRATION
    assert set(quantization["final_input_ranges"]) == set(quantization["input_ranges"])
    assert quantization["final_input_ranges"] != quantization["input_ranges"]  # the ranges moved while training
    qat = result.quality["qat"]
    assert qat["epochs"] == 1 and qat["steps"] >= 1 and all(math.isfinite(v) for v in qat["epoch_loss"])
    assert result.quality["precision"]["ok"] and result.quality["precision"]["regions"]["stem"]["executed"] == ["Int8"]


def test_ptq_candidate_has_no_fine_tune_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = evaluate_int8(monkeypatch, tmp_path)
    assert result.quality["quantization"]["mode"] == "ptq" and "qat" not in result.durations_s
    assert "qat" not in result.quality


def test_a_nonfinite_loss_during_qat_is_an_invalid_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from cnnlab.tcc.model_pipeline import evaluator as evaluator_mod
    from cnnlab.tcc.model_pipeline.trainer import TrainingDiverged

    real = evaluator_mod.train_model
    calls: list[int] = []

    def train(model: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:  # the fine-tune epoch, not the main training
            raise TrainingDiverged("loss is nan")
        return real(model, *args, **kwargs)

    monkeypatch.setattr(evaluator_mod, "train_model", train)
    result = evaluate_int8(monkeypatch, tmp_path, space=INT8_QAT_SLICE)
    assert (result.status, result.failure_stage, result.failure_code) == ("invalid", "qat", "nonfinite_loss")
    assert result.fitness is None


# --- an FP16 region before an INT8 region (observed in the admitted-slice smoke) ----------------------------------


def fp16_then_int8(tmp_path: Path) -> tuple[Path, Any]:
    genome = int8_genome(stem="fp32", head="fp32", blocks=("fp16", "int8", "fp32"))
    model, plan, _ = calibrated(genome)
    r = region_precisions(genome)
    typed = typed_network(model, r["blocks"], r["stem"], r["head"], plan)
    return export_network(typed, tmp_path).onnx_path, genome


def activation_quantizer(path: Path) -> str:
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    initializers = {t.name for t in graph.initializer}
    return next(
        n.name
        for n in graph.node
        if n.op_type == "QuantizeLinear"
        and n.input[0] not in initializers
        and n.output[0]
        and "block" not in n.input[0]
    )


def requantizing(path: Path, with_q: bool) -> dict[str, Any]:
    """Inspector where the last FP16 layer runs Half -> Int8, as TensorRT 11.3 fuses the next region's Q into it."""
    out = inspector_from_onnx(path)
    node = [l["node"] for l in weight_layers(path)[0] if l["region"] == "blocks.0"][-1]
    layer = next(l for l in out["Layers"] if node in l["Metadata"])
    layer["Outputs"] = [{"Datatype": "Int8"}]
    if with_q:
        layer["Metadata"] += f"\x1f[ONNX Layer: {activation_quantizer(path)}]"
    return out


def test_fp16_layer_whose_result_is_requantized_by_a_fused_q_is_accepted(tmp_path: Path) -> None:
    path, genome = fp16_then_int8(tmp_path)
    r = region_precisions(genome)
    check = check_engine_precision(path, requantizing(path, with_q=True), r["blocks"], r["stem"], r["head"])
    assert check["ok"], check["problems"]
    assert check["regions"]["blocks.0"]["executed"] == ["Half"]
    assert "input; output requantized by a fused Q" in {row["evidence"] for row in check["layers"]}


def test_int8_output_without_a_q_in_the_layer_is_still_a_wrong_type(tmp_path: Path) -> None:
    path, genome = fp16_then_int8(tmp_path)
    r = region_precisions(genome)
    check = check_engine_precision(path, requantizing(path, with_q=False), r["blocks"], r["stem"], r["head"])
    assert not check["ok"] and any("fp16 requested, engine runs Half -> Int8" in p for p in check["problems"])


def test_a_layer_that_took_int8_input_is_not_excused_by_a_fused_q(tmp_path: Path) -> None:
    path, genome = fp16_then_int8(tmp_path)
    r = region_precisions(genome)
    inspector = requantizing(path, with_q=True)
    node = [l["node"] for l in weight_layers(path)[0] if l["region"] == "blocks.0"][-1]
    next(l for l in inspector["Layers"] if node in l["Metadata"])["Inputs"] = [{"Datatype": "Int8"}]
    assert not check_engine_precision(path, inspector, r["blocks"], r["stem"], r["head"])["ok"]
