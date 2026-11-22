"""Per-block FP16: typed network, exported graph and the engine precision check (CPU).

The ONNX export is real; the engine inspector is a stand-in written from the
graph (``inspector_from_onnx``), so these tests cover the comparison logic.
That a real TensorRT engine runs the requested types was observed on the GPU during development.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
from test_pipeline_training import cpu_data, fake_bench, inspector_from_onnx  # same directory
from torch import nn

from cnnlab.tcc.contracts import EvaluationRequest
from cnnlab.tcc.model_pipeline import evaluator as evaluator_mod
from cnnlab.tcc.model_pipeline.builder import build_model, fold_batchnorm
from cnnlab.tcc.model_pipeline.engine import BuiltEngine, export_network
from cnnlab.tcc.model_pipeline.evaluator import (
    EngineEvaluator,
    EvaluatorConfig,
    pilot_cache_context,
    representation_problems,
)
from cnnlab.tcc.model_pipeline.precision import (
    check_engine_precision,
    region_of,
    requested_precisions,
    typed_network,
    weight_layers,
)
from cnnlab.tcc.model_pipeline.trainer import TrainConfig
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.genome import ExpressedGenome
from cnnlab.tcc.science.operators import FP16_BLOCK_SLICE, prepare_in_slice
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.space import SPACE_V1


def mixed_genome(seed: int = 3, n_blocks: int = 3, precisions: tuple[str, ...] = ("fp16", "fp32", "fp16")) -> Any:
    raw = FP16_BLOCK_SLICE.sample(GeneRng(seed))
    blocks = tuple(
        dataclasses.replace(b, precision=precisions[i] if i < len(precisions) else "fp32")
        for i, b in enumerate(raw.blocks)
    )
    return dataclasses.replace(raw, n_blocks=n_blocks, blocks=blocks)


def expressed(seed: int = 3, **kw: Any) -> ExpressedGenome:
    return express(mixed_genome(seed, **kw), SPACE_V1).genome


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[str], Any]:
    genome = expressed()
    precisions = [b.precision for b in genome.blocks]
    model, _ = build_model(genome, 1)
    typed = typed_network(model, precisions)
    out = tmp_path_factory.mktemp("typed")
    return export_network(typed, out).onnx_path, precisions, typed


# --- typed network ----------------------------------------------------------------------


def test_blocks_get_their_precision_and_the_original_stays_fp32() -> None:
    genome = expressed()
    model, _ = build_model(genome, 1)
    typed = typed_network(model, ["fp16", "fp32", "fp16"])
    dtypes = [{p.dtype for p in block.parameters()} for block in typed.blocks]
    assert dtypes == [{torch.float16}, {torch.float32}, {torch.float16}]
    assert {p.dtype for p in typed.stem.parameters()} == {torch.float32}
    assert {p.dtype for p in typed.head.parameters()} == {torch.float32}
    assert not any(isinstance(m, nn.BatchNorm2d) for m in typed.modules())
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert any(isinstance(m, nn.BatchNorm2d) for m in model.modules())


@pytest.mark.parametrize(
    ("precisions", "casts"),
    [
        (["fp32", "fp32", "fp32"], 0),
        (["fp16", "fp16", "fp16"], 2),
        (["fp16", "fp32", "fp16"], 4),
        (["fp32", "fp16", "fp16"], 2),
        (["fp32", "fp16", "fp32"], 2),
    ],
)
def test_casts_only_at_precision_boundaries(precisions: list[str], casts: int) -> None:
    model, _ = build_model(expressed(), 1)
    typed = typed_network(model, precisions)
    assert typed.casts == casts
    x = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits = typed(x)
        reference = fold_batchnorm(model)(x)
    assert logits.dtype == torch.float32 and logits.shape == (2, 100) and torch.isfinite(logits).all()
    tolerance = 0.0 if casts == 0 else 0.05
    assert float((logits - reference).abs().max()) <= tolerance
    if casts:
        assert not torch.equal(logits, reference)


def test_unsupported_precisions_are_refused_not_approximated() -> None:
    model, _ = build_model(expressed(), 1)
    for bad in ("fp8", "int4w"):
        with pytest.raises(ValueError, match="not materialized"):
            typed_network(model, ["fp32", bad, "fp32"])
    with pytest.raises(ValueError, match="quantization plan"):  # INT8 is materialized, but needs its calibration
        typed_network(model, ["fp32", "int8", "fp32"])
    with pytest.raises(ValueError, match="one precision per block"):
        typed_network(model, ["fp32"])


def test_regions_and_requested_precisions() -> None:
    assert region_of("stem.0.weight") == "stem"
    assert region_of("blocks.12.body.3.weight") == "blocks.12"
    assert region_of("head.layers.3.bias") == "head"
    assert region_of("val_7") is None
    assert requested_precisions(["fp16", "fp32"]) == {
        "stem": "fp32",
        "head": "fp32",
        "blocks.0": "fp16",
        "blocks.1": "fp32",
    }


# --- exported graph -----------------------------------------------------------------------------


def test_exported_graph_carries_the_types(exported: tuple[Path, list[str], Any]) -> None:
    onnx_path, precisions, typed = exported
    layers, casts = weight_layers(onnx_path)
    assert len(casts) == typed.casts == 4
    wanted = {"fp32": ["FLOAT"], "fp16": ["FLOAT16"]}
    requested = requested_precisions(precisions)
    assert {layer["region"] for layer in layers} == set(requested)
    for layer in layers:
        assert layer["onnx_weight_dtypes"] == wanted[requested[layer["region"]]], layer
    convs = sum(isinstance(m, nn.Conv2d) for m in typed.modules())
    linears = sum(isinstance(m, nn.Linear) for m in typed.modules())
    assert len(layers) == convs + linears


# --- engine check ---------------------------------------------------------------------------------


def test_matching_engine_passes(exported: tuple[Path, list[str], Any]) -> None:
    onnx_path, precisions, _ = exported
    check = check_engine_precision(onnx_path, inspector_from_onnx(onnx_path), precisions)
    assert check["ok"] and not check["problems"]
    assert {name: region["executed"] for name, region in check["regions"].items()} == {
        "stem": ["Float"],
        "head": ["Float"],
        "blocks.0": ["Half"],
        "blocks.1": ["Float"],
        "blocks.2": ["Half"],
    }


def first_node(onnx_path: Path, region: str) -> str:
    return next(layer["node"] for layer in weight_layers(onnx_path)[0] if layer["region"] == region)


def test_substituted_precision_is_reported(exported: tuple[Path, list[str], Any]) -> None:
    """The engine ran an FP16 block in FP32 (or the reverse): the candidate cannot be accepted."""
    onnx_path, precisions, _ = exported
    for region, executed in (("blocks.0", "Float"), ("blocks.1", "Half"), ("blocks.2", "BFloat16")):
        node = first_node(onnx_path, region)
        check = check_engine_precision(onnx_path, inspector_from_onnx(onnx_path, {node: executed}), precisions)
        assert not check["ok"]
        assert any(node in p and f"engine runs {executed}" in p for p in check["problems"]), check["problems"]


def test_layer_missing_from_the_engine_is_reported(exported: tuple[Path, list[str], Any]) -> None:
    onnx_path, precisions, _ = exported
    inspector = inspector_from_onnx(onnx_path)
    inspector["Layers"].pop(2)
    check = check_engine_precision(onnx_path, inspector, precisions)
    assert not check["ok"] and any("no engine layer was built" in p for p in check["problems"])
    empty = check_engine_precision(onnx_path, {"Layers": []}, precisions)
    assert not empty["ok"] and any("names no ONNX layer" in p for p in empty["problems"])


def test_cast_fused_into_a_layer_is_accepted_on_one_side_only(exported: tuple[Path, list[str], Any]) -> None:
    """Observed on the GPU: the last FP16 convolution is fused with the cast back to FP32."""
    onnx_path, precisions, _ = exported
    _, casts = weight_layers(onnx_path)
    cast = sorted(casts)[0]
    node = first_node(onnx_path, "blocks.2")
    inspector = inspector_from_onnx(onnx_path)
    layer = next(l for l in inspector["Layers"] if node in l["Metadata"])
    layer["Outputs"] = [{"Datatype": "Float"}]
    assert not check_engine_precision(onnx_path, inspector, precisions)["ok"]  # no cast fused: a substitution
    layer["Metadata"] += f"\\x1f[ONNX Layer: {cast}]"
    fused = check_engine_precision(onnx_path, inspector, precisions)
    assert fused["ok"] and fused["regions"]["blocks.2"]["fused_casts"] == 1
    assert fused["regions"]["blocks.2"]["executed"] == ["Half"]
    layer["Inputs"] = [{"Datatype": "Float"}]  # neither side is FP16 any more
    assert not check_engine_precision(onnx_path, inspector, precisions)["ok"]


def test_requested_precision_that_the_graph_does_not_carry_is_reported(exported: tuple[Path, list[str], Any]) -> None:
    onnx_path, _, _ = exported
    check = check_engine_precision(onnx_path, inspector_from_onnx(onnx_path), ["fp32", "fp32", "fp16"])
    assert not check["ok"] and any("ONNX weights are ['FLOAT16']" in p for p in check["problems"])


# --- evaluator -------------------------------------------------------------------------------------


def request_with(genome: Any, tmp_path: Path, config: EvaluatorConfig, data: Any) -> EvaluationRequest:
    ctx = pilot_cache_context(data, config, 11, "test", FP16_BLOCK_SLICE.space_hash(), backend={"synthetic": True})
    record = prepare_in_slice(genome, FP16_BLOCK_SLICE, ctx).record
    return EvaluationRequest(
        "run1", "c1", "c1.a1", "pilot", record, 11, 0, "ds", "env", "proto", "plan", tmp_path / "artifacts"
    )


def evaluator_for(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    executed: dict[str, str] | None = None,
) -> tuple[EngineEvaluator, EvaluatorConfig, Any]:
    def build(onnx_path: Path, plan_path: Path, workspace_bytes: int = 0) -> BuiltEngine:
        plan_path.write_bytes(b"stand-in" * 100)
        overrides = {first_node(onnx_path, region): dtype for region, dtype in (executed or {}).items()}
        return BuiltEngine(plan_path, 800, "0" * 64, inspector_from_onnx(onnx_path, overrides), {"standin": True}, 0.0)

    monkeypatch.setattr(evaluator_mod, "build_engine", build)
    monkeypatch.setattr(evaluator_mod, "run_isolated", fake_bench)
    config = EvaluatorConfig(train=TrainConfig(epochs=1, batch_size=16))
    data = cpu_data()
    return EngineEvaluator(data, tmp_path / "val.npz", config, device="cpu", work_root=tmp_path / "work"), config, data


def test_fp16_blocks_are_evaluated_and_the_check_is_published(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evaluator, config, data = evaluator_for(monkeypatch, tmp_path)
    result = evaluator.evaluate(request_with(mixed_genome(), tmp_path, config, data))
    assert result.status == "valid", (result.failure_code, result.failure_detail)
    precision = result.quality["precision"]
    assert precision["ok"] and precision["requested_blocks"] == ["fp16", "fp32", "fp16"]
    assert precision["casts_in_graph"] == 4 and precision["regions"]["blocks.0"]["executed"] == ["Half"]
    assert "precision" in result.artifact_paths
    assert (tmp_path / "artifacts" / result.artifact_paths["precision"]).is_file()


def test_engine_that_ran_another_precision_makes_the_candidate_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluator, config, data = evaluator_for(monkeypatch, tmp_path, executed={"blocks.0": "Float"})
    result = evaluator.evaluate(request_with(mixed_genome(), tmp_path, config, data))
    assert (result.status, result.failure_stage, result.failure_code) == (
        "invalid",
        "precision_check",
        "precision_not_executed",
    )
    assert result.fitness is None and "engine runs Float" in (result.failure_detail or "")
    assert "train" in result.durations_s and "benchmark" not in result.durations_s
    assert not result.artifact_paths


def test_fp8_stays_unsupported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    precision = "fp8"
    genome = expressed()
    changed = dataclasses.replace(
        genome, blocks=(dataclasses.replace(genome.blocks[0], precision=precision), *genome.blocks[1:])
    )
    assert representation_problems(genome) == []
    assert representation_problems(changed) == [f"blocks[0].precision={precision}"]
    assert representation_problems(dataclasses.replace(genome, stem_precision="fp8")) == ["stem_precision=fp8"]
    assert representation_problems(dataclasses.replace(genome, head_precision="fp16")) == []  # G1: fp16 head


# --- casts fused on both sides (cross-review finding C5) ------------------------------------------------


def both_sides_fused(onnx_path: Path, weights: str | None) -> dict[str, Any]:
    """A one-layer FP16 region whose two boundary casts were fused into the layer: Float on both sides."""
    _, casts = weight_layers(onnx_path)
    node = first_node(onnx_path, "blocks.0")
    inspector = inspector_from_onnx(onnx_path)
    layer = next(l for l in inspector["Layers"] if node in l["Metadata"])
    layer["Inputs"] = [{"Datatype": "Float"}]
    layer["Outputs"] = [{"Datatype": "Float"}]
    layer["Metadata"] += "".join(f"\x1f[ONNX Layer: {cast}]" for cast in sorted(casts)[:2])
    if weights is not None:
        layer["Constants"] = [
            {"Name": "w", "Dimensions": [32, 16, 3, 3], "Datatype": weights},
            {"Name": "scalar", "Dimensions": [1], "Datatype": "Float"},
        ]
    return inspector


def test_layer_hidden_by_two_fused_casts_is_accepted_by_its_engine_weights(
    exported: tuple[Path, list[str], Any],
) -> None:
    onnx_path, precisions, _ = exported
    check = check_engine_precision(onnx_path, both_sides_fused(onnx_path, "Half"), precisions)
    assert check["ok"], check["problems"]
    row = next(r for r in check["layers"] if r["node"] == first_node(onnx_path, "blocks.0"))
    assert row["evidence"] == "engine weights" and row["fused_casts"] == 2 and row["engine_weights"] == ["Half"]
    assert check["regions"]["blocks.0"]["by_weights_only"] == 1
    assert "Half" in check["regions"]["blocks.0"]["executed"]


@pytest.mark.parametrize("weights", ["Float", None])
def test_layer_hidden_by_two_fused_casts_without_matching_weights_is_refused(
    exported: tuple[Path, list[str], Any], weights: str | None
) -> None:
    onnx_path, precisions, _ = exported
    check = check_engine_precision(onnx_path, both_sides_fused(onnx_path, weights), precisions)
    assert not check["ok"]
    assert any("2 cast(s) fused" in p for p in check["problems"]), check["problems"]


def test_evidence_is_recorded_for_every_layer(exported: tuple[Path, list[str], Any]) -> None:
    onnx_path, precisions, _ = exported
    check = check_engine_precision(onnx_path, inspector_from_onnx(onnx_path), precisions)
    assert {row["evidence"] for row in check["layers"]} == {"input and output"}
