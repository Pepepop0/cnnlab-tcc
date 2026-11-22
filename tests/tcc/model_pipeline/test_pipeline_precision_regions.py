"""FP16 in the stem and in the head (task G1): typed network, exported graph and engine check (CPU).

The ONNX export is real; the engine inspector is the stand-in written from the
graph. That a real TensorRT engine runs stem and head in FP16 was observed on
the GPU with ``precision_probe`` (16 of 16 genomes, all four stem x head
combinations) during development.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
from test_pipeline_training import inspector_from_onnx  # same directory

from cnnlab.tcc.model_pipeline import precision_probe
from cnnlab.tcc.model_pipeline.builder import build_model, fold_batchnorm
from cnnlab.tcc.model_pipeline.engine import export_network
from cnnlab.tcc.model_pipeline.precision import (
    check_engine_precision,
    region_precisions,
    representation_problems,
    requested_precisions,
    typed_network,
    weight_layers,
)
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.operators import FP16_REGIONS_SLICE, PROBE_SLICES, SLICES
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.space import SPACE_V1

COMBOS = [("fp32", "fp32"), ("fp16", "fp32"), ("fp32", "fp16"), ("fp16", "fp16")]


def genome_with(stem: str, head: str, blocks: tuple[str, ...] = ("fp32", "fp16", "fp32"), seed: int = 5) -> Any:
    raw = FP16_REGIONS_SLICE.sample(GeneRng(seed))
    chosen = tuple(
        dataclasses.replace(b, precision=blocks[i] if i < len(blocks) else "fp32") for i, b in enumerate(raw.blocks)
    )
    raw = dataclasses.replace(raw, n_blocks=len(blocks), stem_precision=stem, head_precision=head, blocks=chosen)
    assert FP16_REGIONS_SLICE.contains(raw)
    return express(raw, SPACE_V1).genome


@pytest.fixture(scope="module")
def exports(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str], tuple[Path, Any, Any]]:
    out = {}
    for stem, head in COMBOS:
        genome = genome_with(stem, head)
        model, _ = build_model(genome, 1)
        regions = region_precisions(genome)
        typed = typed_network(model, regions["blocks"], regions["stem"], regions["head"])
        path = export_network(typed, tmp_path_factory.mktemp(f"typed-{stem}-{head}")).onnx_path
        out[(stem, head)] = (path, genome, typed)
    return out


# --- typed network ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("stem", "head"), COMBOS)
def test_stem_and_head_get_their_precision(stem: str, head: str) -> None:
    genome = genome_with(stem, head)
    model, _ = build_model(genome, 1)
    regions = region_precisions(genome)
    typed = typed_network(model, regions["blocks"], regions["stem"], regions["head"])
    dtype = {"fp32": torch.float32, "fp16": torch.float16}
    assert {p.dtype for p in typed.stem.parameters()} == {dtype[stem]}
    assert {p.dtype for p in typed.head.parameters()} == {dtype[head]}
    assert [{p.dtype for p in b.parameters()} for b in typed.blocks] == [{dtype[x]} for x in regions["blocks"]]
    assert {p.dtype for p in model.parameters()} == {torch.float32}  # the trained model is untouched
    x = torch.randn(3, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits = typed(x)
    assert logits.dtype == torch.float32 and logits.shape == (3, 100) and torch.isfinite(logits).all()


@pytest.mark.parametrize(
    ("stem", "blocks", "head", "casts"),
    [
        ("fp32", ["fp32", "fp32"], "fp32", 0),
        ("fp16", ["fp16", "fp16"], "fp16", 2),
        ("fp16", ["fp32", "fp32"], "fp32", 2),
        ("fp32", ["fp32", "fp32"], "fp16", 2),
        ("fp16", ["fp32", "fp16"], "fp16", 4),
        ("fp32", ["fp16", "fp16"], "fp32", 2),
    ],
)
def test_casts_at_the_input_the_logits_and_between_regions(stem: str, blocks: list[str], head: str, casts: int) -> None:
    genome = genome_with("fp32", "fp32", ("fp32", "fp32"))
    model, _ = build_model(genome, 1)
    typed = typed_network(model, blocks, stem, head)
    assert typed.casts == casts
    x = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        got, reference = typed(x), fold_batchnorm(model)(x)
    assert float((got - reference).abs().max()) <= (0.0 if casts == 0 else 0.05)


def test_precision_that_is_not_materialized_in_stem_or_head_is_refused() -> None:
    model, _ = build_model(genome_with("fp32", "fp32"), 1)
    for bad in ("fp8", "int4w"):
        with pytest.raises(ValueError, match="not materialized"):
            typed_network(model, ["fp32", "fp32", "fp32"], stem_precision=bad)
        with pytest.raises(ValueError, match="not materialized"):
            typed_network(model, ["fp32", "fp32", "fp32"], head_precision=bad)


def test_region_precisions_and_requested_precisions_agree() -> None:
    genome = genome_with("fp16", "fp32")
    regions = region_precisions(genome)
    assert regions == {"stem": "fp16", "blocks": ["fp32", "fp16", "fp32"], "head": "fp32"}
    assert requested_precisions(regions["blocks"], regions["stem"], regions["head"]) == {
        "stem": "fp16",
        "head": "fp32",
        "blocks.0": "fp32",
        "blocks.1": "fp16",
        "blocks.2": "fp32",
    }
    assert requested_precisions(["fp16"]) == {"stem": "fp32", "head": "fp32", "blocks.0": "fp16"}  # old call


# --- exported graph and engine check -----------------------------------------------------------------


@pytest.mark.parametrize(("stem", "head"), COMBOS)
def test_graph_carries_the_types_of_every_region(
    exports: dict[tuple[str, str], tuple[Path, Any, Any]], stem: str, head: str
) -> None:
    path, genome, typed = exports[(stem, head)]
    wanted = {"fp32": ["FLOAT"], "fp16": ["FLOAT16"]}
    requested = requested_precisions(
        **{"block_precisions": region_precisions(genome)["blocks"], "stem_precision": stem, "head_precision": head}
    )
    layers, casts = weight_layers(path)
    assert {layer["region"] for layer in layers} == set(requested)
    assert all(layer["onnx_weight_dtypes"] == wanted[requested[layer["region"]]] for layer in layers)
    assert len(casts) == typed.casts


@pytest.mark.parametrize(("stem", "head"), COMBOS)
def test_matching_engine_passes_for_every_combination(
    exports: dict[tuple[str, str], tuple[Path, Any, Any]], stem: str, head: str
) -> None:
    path, genome, _ = exports[(stem, head)]
    r = region_precisions(genome)
    check = check_engine_precision(path, inspector_from_onnx(path), r["blocks"], stem, head)
    assert check["ok"], check["problems"]
    names = {"fp32": "Float", "fp16": "Half"}
    assert check["regions"]["stem"]["executed"] == [names[stem]]
    assert check["regions"]["head"]["executed"] == [names[head]]


def first_node(path: Path, region: str) -> str:
    return next(layer["node"] for layer in weight_layers(path)[0] if layer["region"] == region)


@pytest.mark.parametrize("region", ["stem", "head"])
def test_engine_that_ran_stem_or_head_in_another_type_is_reported(
    exports: dict[tuple[str, str], tuple[Path, Any, Any]], region: str
) -> None:
    path, genome, _ = exports[("fp16", "fp16")]
    r = region_precisions(genome)
    node = first_node(path, region)
    check = check_engine_precision(path, inspector_from_onnx(path, {node: "Float"}), r["blocks"], "fp16", "fp16")
    assert not check["ok"]
    assert any(node in p and f"({region})" in p and "engine runs Float" in p for p in check["problems"])


def test_asking_for_the_wrong_stem_or_head_against_a_graph_is_reported(
    exports: dict[tuple[str, str], tuple[Path, Any, Any]],
) -> None:
    path, genome, _ = exports[("fp32", "fp32")]
    r = region_precisions(genome)
    wrong = check_engine_precision(path, inspector_from_onnx(path), r["blocks"], "fp16", "fp32")
    assert not wrong["ok"] and any("(stem)" in p and "ONNX weights are ['FLOAT']" in p for p in wrong["problems"])


def test_one_layer_stem_with_both_casts_fused_is_accepted_only_by_its_engine_weights(
    exports: dict[tuple[str, str], tuple[Path, Any, Any]],
) -> None:
    """The stem is one convolution: an FP16 stem between the FP32 input and an FP32 block may lose both casts."""
    path, genome, _ = exports[("fp16", "fp32")]
    r = region_precisions(genome)
    _, casts = weight_layers(path)
    node = first_node(path, "stem")

    def inspector(weights: str | None) -> dict[str, Any]:
        out = inspector_from_onnx(path)
        layer = next(l for l in out["Layers"] if node in l["Metadata"])
        layer["Inputs"], layer["Outputs"] = [{"Datatype": "Float"}], [{"Datatype": "Float"}]
        layer["Metadata"] += "".join(f"\x1f[ONNX Layer: {c}]" for c in sorted(casts)[:2])
        if weights:
            layer["Constants"] = [{"Name": "w", "Dimensions": [16, 3, 3, 3], "Datatype": weights}]
        return out

    ok = check_engine_precision(path, inspector("Half"), r["blocks"], "fp16", "fp32")
    assert ok["ok"] and ok["regions"]["stem"]["by_weights_only"] == 1
    assert not check_engine_precision(path, inspector("Float"), r["blocks"], "fp16", "fp32")["ok"]
    assert not check_engine_precision(path, inspector(None), r["blocks"], "fp16", "fp32")["ok"]


# --- representation rule ---------------------------------------------------------------------------------


def test_representation_problems_accept_fp16_everywhere_and_refuse_the_rest() -> None:
    assert representation_problems(genome_with("fp16", "fp16", ("fp16", "fp32"))) == []
    genome = genome_with("fp32", "fp32")
    bad = dataclasses.replace(
        genome,
        stem_precision="fp8",
        head_precision="int4w",
        blocks=(dataclasses.replace(genome.blocks[0], precision="fp8", prune_pct=25), *genome.blocks[1:]),
    )
    assert representation_problems(bad) == ["stem_precision=fp8", "head_precision=int4w", "blocks[0].precision=fp8"]


# --- the probe's helpers -----------------------------------------------------------------------------------


# --- the evaluator with FP16 in stem and head ----------------------------------------------------------------


def evaluate_with(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stem: str, head: str, executed: dict[str, str] | None = None
) -> Any:
    from test_pipeline_precision import evaluator_for  # same directory
    from test_pipeline_training import cpu_data

    from cnnlab.tcc.contracts import EvaluationRequest
    from cnnlab.tcc.model_pipeline.evaluator import pilot_cache_context
    from cnnlab.tcc.science.operators import prepare_in_slice

    evaluator, config, data = evaluator_for(monkeypatch, tmp_path, executed)
    ctx = pilot_cache_context(data, config, 11, "test", FP16_REGIONS_SLICE.space_hash(), backend={"synthetic": True})
    raw = FP16_REGIONS_SLICE.sample(GeneRng(5))
    raw = dataclasses.replace(raw, stem_precision=stem, head_precision=head)
    record = prepare_in_slice(raw, FP16_REGIONS_SLICE, ctx).record
    request = EvaluationRequest(
        "run1", "c1", "c1.a1", "pilot", record, 11, 0, "ds", "env", "proto", "plan", tmp_path / "artifacts"
    )
    return evaluator.evaluate(request)


@pytest.mark.parametrize(("stem", "head"), COMBOS)
def test_evaluator_accepts_fp16_in_stem_and_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stem: str, head: str
) -> None:
    result = evaluate_with(monkeypatch, tmp_path, stem, head)
    assert result.status == "valid", (result.failure_code, result.failure_detail)
    precision = result.quality["precision"]
    assert (precision["requested_stem"], precision["requested_head"]) == (stem, head) and precision["ok"]
    names = {"fp32": ["Float"], "fp16": ["Half"]}
    assert precision["regions"]["stem"]["executed"] == names[stem]
    assert precision["regions"]["head"]["executed"] == names[head]
    assert result.quality["evaluator"]["representation"].startswith("stem, blocks and head fp32, fp16, bf16")


@pytest.mark.parametrize("region", ["stem", "head"])
def test_engine_that_did_not_run_fp16_in_stem_or_head_makes_the_candidate_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, region: str
) -> None:
    result = evaluate_with(monkeypatch, tmp_path, "fp16", "fp16", executed={region: "Float"})
    assert (result.status, result.failure_stage, result.failure_code) == (
        "invalid",
        "precision_check",
        "precision_not_executed",
    )
    assert f"({region})" in (result.failure_detail or "") and result.fitness is None


def test_probe_cycles_through_every_combination_the_slice_allows() -> None:
    assert precision_probe.allowed_values(FP16_REGIONS_SLICE, "stem_precision") == ("fp32", "fp16")
    assert precision_probe.allowed_values(SLICES["fp32"], "head_precision") == ("fp32",)
    genomes = precision_probe.probe_genomes(PROBE_SLICES["fp16_regions"], 8, 11)
    combos = [(g.stem_precision, g.head_precision) for g in genomes]
    product = [(a, b) for a in ("fp32", "fp16") for b in ("fp32", "fp16")]  # stem varies slowest
    assert combos == product + product
    assert precision_probe.probe_genomes(PROBE_SLICES["fp16_regions"], 8, 11) == genomes
    fp32_only = precision_probe.probe_genomes(SLICES["fp32"], 3, 11)
    assert {(g.stem_precision, g.head_precision) for g in fp32_only} == {("fp32", "fp32")}


def test_probe_reference_and_engine_input_come_from_the_same_images() -> None:
    """A probe once compared the engine on images of seed+1 with a reference made from seed+7 images."""
    assert torch.equal(precision_probe.probe_images(3), precision_probe.probe_images(3))
    assert not torch.equal(precision_probe.probe_images(3), precision_probe.probe_images(4))
    source = Path(precision_probe.__file__).read_text(encoding="utf-8")
    assert source.count("structured_inputs(") == 1  # only inside probe_images; no other source of images
