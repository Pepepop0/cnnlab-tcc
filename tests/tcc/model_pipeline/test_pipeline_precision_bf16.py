"""BF16 per region (task G2): typed network, exported graph, engine limits and the engine check (CPU).

The ONNX export is real; the engine inspector is the stand-in written from the
graph. What TensorRT 11.3 really does with BF16 was measured on the GPU with
``precision_probe --matrix bf16`` during development:
standard and grouped blocks and the stem run in BF16 with ReLU or SiLU, HardSwish
does not build, depthwise convolutions run as FP16 and the head's Gemm takes a
FP32 input with BF16 weights and output.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
from test_pipeline_precision import evaluator_for  # same directory
from test_pipeline_training import inspector_from_onnx  # same directory

from cnnlab.tcc.model_pipeline.builder import build_model
from cnnlab.tcc.model_pipeline.engine import export_network
from cnnlab.tcc.model_pipeline.precision import (
    BF16_DEPTHWISE_OPS,
    BF16_UNSUPPORTED_ACTIVATIONS,
    bf16_engine_limits,
    check_engine_precision,
    region_precisions,
    representation_problems,
    typed_network,
    weight_layers,
)
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.operators import BF16_REGIONS_SLICE, FP16_REGIONS_SLICE, SLICES
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.space import SPACE_V1


def bf16_genome(
    stem: str = "fp32",
    head: str = "fp32",
    blocks: tuple[str, ...] = ("fp32", "bf16", "fp32"),
    op: str = "standard",
    activation: str = "relu",
    seed: int = 5,
    head_type: str = "gap_linear",
) -> Any:
    raw = BF16_REGIONS_SLICE.sample(GeneRng(seed))
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
        head=head_type,
        blocks=chosen,
    )
    assert BF16_REGIONS_SLICE.contains(raw)
    return express(raw, SPACE_V1).genome


# --- the slice ------------------------------------------------------------------------------------------


def test_slice_is_registered_and_contains_the_fp16_slice() -> None:
    assert SLICES["bf16_regions"] is BF16_REGIONS_SLICE
    rng = GeneRng(4)
    assert all(BF16_REGIONS_SLICE.contains(FP16_REGIONS_SLICE.sample(rng)) for _ in range(40))
    bf16 = [BF16_REGIONS_SLICE.sample(GeneRng(s)) for s in range(60)]
    assert not all(FP16_REGIONS_SLICE.contains(g) for g in bf16)
    assert not BF16_REGIONS_SLICE.contains(dataclasses.replace(bf16[0], stem_precision="int8"))


# --- representation and engine limits ---------------------------------------------------------------------


def test_bf16_is_materialized_where_the_engine_runs_it() -> None:
    for activation in ("relu", "silu"):
        for op in ("standard", "grouped_g4"):
            genome = bf16_genome("bf16", "bf16", ("bf16", "bf16", "fp32"), op=op, activation=activation)
            assert representation_problems(genome) == [] and bf16_engine_limits(genome) == []


def test_hardswish_is_refused_only_where_bf16_is_requested() -> None:
    assert BF16_UNSUPPORTED_ACTIVATIONS == ("hardswish",)
    assert representation_problems(bf16_genome("fp32", "bf16", ("fp16", "fp32"), activation="hardswish")) == []
    stem = representation_problems(bf16_genome("bf16", "fp32", ("fp32", "fp32"), activation="hardswish"))
    assert len(stem) == 1 and stem[0].startswith("stem.precision=bf16 with activation hardswish")
    blocks = representation_problems(bf16_genome("fp32", "fp32", ("bf16", "fp32", "bf16"), activation="hardswish"))
    assert [p.split(".")[0] for p in blocks] == ["blocks[0]", "blocks[2]"]


def test_hardswish_is_also_refused_in_a_bf16_head_that_has_a_hidden_layer() -> None:
    """Measured with precision_probe --matrix bf16: the fc256 head with HardSwish does not build in BF16."""
    problems = representation_problems(
        bf16_genome("fp32", "bf16", ("fp32", "fp32"), activation="hardswish", head_type="gap_fc256_linear")
    )
    assert len(problems) == 1 and problems[0].startswith("head.precision=bf16 with activation hardswish")
    for activation in ("relu", "silu"):
        assert (
            representation_problems(
                bf16_genome("fp32", "bf16", ("fp32", "fp32"), activation=activation, head_type="gap_fc256_linear")
            )
            == []
        )


@pytest.mark.parametrize("op", BF16_DEPTHWISE_OPS)
def test_depthwise_blocks_are_refused_in_bf16_and_accepted_in_fp16(op: str) -> None:
    problems = representation_problems(bf16_genome(op=op))
    assert len(problems) == 1 and problems[0].startswith(f"blocks[1].precision=bf16 with op {op}")
    assert representation_problems(bf16_genome(blocks=("fp32", "fp16", "fp32"), op=op)) == []
    assert representation_problems(bf16_genome(blocks=("bf16", "fp32", "fp32"), op="standard")) == []


def test_fp8_pruning_and_sparsity_stay_listed() -> None:
    genome = bf16_genome()
    bad = dataclasses.replace(
        genome,
        head_precision="fp8",
        blocks=(dataclasses.replace(genome.blocks[0], prune_pct=25, sparsity="2:4"), *genome.blocks[1:]),
    )
    assert representation_problems(bad) == ["head_precision=fp8", "blocks[0].sparsity=2:4"]


# --- typed network and exported graph -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any]:
    genome = bf16_genome("bf16", "bf16", ("fp32", "bf16", "fp16"))
    model, _ = build_model(genome, 1)
    regions = region_precisions(genome)
    typed = typed_network(model, regions["blocks"], regions["stem"], regions["head"])
    assert {p.dtype for p in typed.parameters()} >= {torch.bfloat16, torch.float16, torch.float32}
    return export_network(typed, tmp_path_factory.mktemp("bf16")).onnx_path, genome


def test_graph_carries_bf16_weights_in_the_requested_regions(exported: tuple[Path, Any]) -> None:
    path, genome = exported
    layers, _ = weight_layers(path)
    types = {layer["region"]: layer["onnx_weight_dtypes"][0] for layer in layers}
    assert types["stem"] == "BFLOAT16" and types["head"] == "BFLOAT16"
    assert types["blocks.0"] == "FLOAT" and types["blocks.1"] == "BFLOAT16" and types["blocks.2"] == "FLOAT16"


def test_matching_engine_passes(exported: tuple[Path, Any]) -> None:
    path, genome = exported
    r = region_precisions(genome)
    check = check_engine_precision(path, inspector_from_onnx(path), r["blocks"], r["stem"], r["head"])
    assert check["ok"], check["problems"]
    assert check["regions"]["blocks.1"]["executed"] == ["BFloat16"]


def test_engine_that_ran_a_bf16_region_as_fp16_is_reported(exported: tuple[Path, Any]) -> None:
    path, genome = exported
    r = region_precisions(genome)
    node = next(l["node"] for l in weight_layers(path)[0] if l["region"] == "blocks.1")
    check = check_engine_precision(path, inspector_from_onnx(path, {node: "Half"}), r["blocks"], r["stem"], r["head"])
    assert not check["ok"] and any("bf16 requested, engine runs Half -> Half" in p for p in check["problems"])


def head_inspector(path: Path, node_input: str, node_output: str, weights: str | None) -> dict[str, Any]:
    out = inspector_from_onnx(path)
    node = next(l["node"] for l in weight_layers(path)[0] if l["region"] == "head")
    layer = next(l for l in out["Layers"] if node in l["Metadata"])
    layer["Inputs"], layer["Outputs"] = [{"Datatype": node_input}], [{"Datatype": node_output}]
    if weights:
        layer["Constants"] = [{"Name": "w", "Dimensions": [32, 100], "Datatype": weights}]
    return out


def test_head_gemm_with_float_input_is_accepted_by_its_bf16_weights_and_output(exported: tuple[Path, Any]) -> None:
    """Observed on TensorRT 11.3: pooling stays FP32 and the Gemm fuses the cast to BF16 into its kernel."""
    path, genome = exported
    r = region_precisions(genome)

    def check(*args: Any) -> dict[str, Any]:
        return check_engine_precision(path, head_inspector(path, *args), r["blocks"], r["stem"], r["head"])

    ok = check("Float", "BFloat16", "BFloat16")
    assert ok["ok"], ok["problems"]
    head = next(row for row in ok["layers"] if row["region"] == "head")
    assert head["evidence"] == "engine weights and output" and ok["regions"]["head"]["executed"] == ["BFloat16"]
    assert not check("Float", "BFloat16", "Float")["ok"]  # weights say it is not BF16
    assert not check("Float", "Float", "BFloat16")["ok"]  # the result is not BF16
    assert not check("Float", "BFloat16", None)["ok"]  # nothing to confirm the weights


# --- the evaluator --------------------------------------------------------------------------------------------


def evaluate_bf16(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    genome_kwargs: dict[str, Any],
    executed: dict[str, str] | None = None,
) -> Any:
    from cnnlab.tcc.contracts import EvaluationRequest
    from cnnlab.tcc.model_pipeline.evaluator import pilot_cache_context
    from cnnlab.tcc.science.operators import prepare_in_slice

    evaluator, config, data = evaluator_for(monkeypatch, tmp_path, executed)
    ctx = pilot_cache_context(data, config, 11, "test", BF16_REGIONS_SLICE.space_hash(), backend={"synthetic": True})
    raw = BF16_REGIONS_SLICE.sample(GeneRng(5))
    blocks = tuple(
        dataclasses.replace(b, op=genome_kwargs.get("op", "standard"), precision="fp32", prune_pct=0)
        for b in raw.blocks
    )
    raw = dataclasses.replace(
        raw,
        stem_precision=genome_kwargs.get("stem", "bf16"),
        head_precision=genome_kwargs.get("head", "bf16"),
        activation=genome_kwargs.get("activation", "relu"),
        blocks=(dataclasses.replace(blocks[0], precision="bf16"), *blocks[1:]),
    )
    record = prepare_in_slice(raw, BF16_REGIONS_SLICE, ctx).record
    request = EvaluationRequest(
        "run1", "c1", "c1.a1", "pilot", record, 11, 0, "ds", "env", "proto", "plan", tmp_path / "artifacts"
    )
    return evaluator.evaluate(request)


def test_evaluator_accepts_bf16_where_the_engine_runs_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = evaluate_bf16(monkeypatch, tmp_path, {})
    assert result.status == "valid", (result.failure_code, result.failure_detail)
    precision = result.quality["precision"]
    assert (precision["requested_stem"], precision["requested_head"]) == ("bf16", "bf16") and precision["ok"]
    assert precision["regions"]["stem"]["executed"] == ["BFloat16"]


def test_evaluator_marks_the_known_engine_limits_invalid_before_any_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for kwargs in ({"activation": "hardswish"}, {"op": "mbconv_e3", "stem": "fp32", "head": "fp32"}):
        result = evaluate_bf16(monkeypatch, tmp_path / kwargs.get("activation", "dw"), kwargs)
        assert (result.status, result.failure_stage, result.failure_code) == (
            "invalid",
            "representation",
            "unsupported_representation",
        )
        assert "train" not in result.durations_s and result.fitness is None


def test_engine_that_ran_bf16_as_another_type_makes_the_candidate_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = evaluate_bf16(monkeypatch, tmp_path, {}, executed={"stem": "Half"})
    assert (result.status, result.failure_stage, result.failure_code) == (
        "invalid",
        "precision_check",
        "precision_not_executed",
    )
    assert "(stem)" in (result.failure_detail or "") and result.fitness is None
