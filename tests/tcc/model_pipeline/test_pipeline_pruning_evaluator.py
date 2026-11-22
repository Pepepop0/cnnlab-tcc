"""Pruning inside the evaluator (task G5): train -> prune -> fine-tune -> fold -> (quantize) -> export (CPU)."""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import Any

import pytest
import torch
from test_pipeline_precision import evaluator_for  # same directory

from cnnlab.tcc.contracts import EvaluationRequest
from cnnlab.tcc.model_pipeline import evaluator as evaluator_mod
from cnnlab.tcc.model_pipeline.builder import build_model, count_macs, count_parameters, effective_widths
from cnnlab.tcc.model_pipeline.evaluator import PRUNING_DEFINITION, pilot_cache_context
from cnnlab.tcc.model_pipeline.precision import representation_problems
from cnnlab.tcc.model_pipeline.pruning import PRUNING_VERSION, PruningError
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.operators import COMPRESSED_SLICE, PRUNE_FP32_SLICE, SLICES, prepare_in_slice
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.space import SPACE_V1

PRUNE_FP32_SLICE_HASH = "2f88da46306d53fbf76aaf90b100c4e6c8b9878123c87e9ea0614ba4723ff0f8"
COMPRESSED_SLICE_HASH = "41f428af8f1a3c32c5658f38a9859a1bbad2102b8d49c8c01cb3424f5db84446"


def pruned_raw(space: Any = PRUNE_FP32_SLICE, seed: int = 5, pct: int = 50, **genes: Any) -> Any:
    """A genome of the slice with 3 wide standard blocks (so 50% is a real cut), every block pruning ``pct``."""
    raw = space.sample(GeneRng(seed))
    blocks = tuple(
        dataclasses.replace(
            b, op="standard", kernel=3, n_convs=2, residual=False, downsample="none", channels=64, prune_pct=pct
        )
        for b in raw.blocks
    )
    raw = dataclasses.replace(
        raw, n_blocks=3, stem_channels=32, activation="relu", head="gap_linear", blocks=blocks, **genes
    )
    assert space.contains(raw)
    return raw


def evaluate_pruned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw: Any | None = None,
    space: Any = PRUNE_FP32_SLICE,
    executed: dict[str, str] | None = None,
    examples: int = 16,
) -> Any:
    from cnnlab.tcc.model_pipeline.evaluator import EngineEvaluator  # noqa: F401

    evaluator, config, data = evaluator_for(monkeypatch, tmp_path, executed)
    config = dataclasses.replace(config, calibration_examples=examples)
    evaluator.config = config  # type: ignore[misc]
    ctx = pilot_cache_context(data, config, 11, "test", space.space_hash(), backend={"synthetic": True})
    record = prepare_in_slice(raw if raw is not None else pruned_raw(space), space, ctx).record
    request = EvaluationRequest(
        "run1", "c1", "c1.a1", "pilot", record, 11, 0, "ds", "env", "proto", "plan", tmp_path / "artifacts"
    )
    return evaluator.evaluate(request)


# --- the slices and the representation rule --------------------------------------------------------------------


def test_pruning_slices_are_registered_and_have_fixed_hashes() -> None:
    assert SLICES["prune_fp32"] is PRUNE_FP32_SLICE and SLICES["compressed"] is COMPRESSED_SLICE
    assert PRUNE_FP32_SLICE.space_hash() == PRUNE_FP32_SLICE_HASH
    assert COMPRESSED_SLICE.space_hash() == COMPRESSED_SLICE_HASH != PRUNE_FP32_SLICE_HASH


def test_the_pruning_slices_draw_pruned_blocks_and_keep_their_pins() -> None:
    seen: set[int] = set()
    for seed in range(80):
        raw = PRUNE_FP32_SLICE.sample(GeneRng(seed))
        assert PRUNE_FP32_SLICE.contains(raw)
        assert raw.stem_precision == raw.head_precision == "fp32" and all(b.precision == "fp32" for b in raw.blocks)
        seen |= {b.prune_pct for b in raw.blocks}
        wide = COMPRESSED_SLICE.sample(GeneRng(seed))
        assert COMPRESSED_SLICE.contains(wide) and "bf16" not in {
            wide.stem_precision,
            wide.head_precision,
            *(b.precision for b in wide.blocks),
        }
    assert seen == {0, 25, 50}


def test_pruning_is_no_longer_a_representation_problem() -> None:
    genome = express(pruned_raw(), SPACE_V1).genome
    assert any(b.prune_pct for b in genome.blocks) and representation_problems(genome) == []


# --- what the evaluator does -------------------------------------------------------------------------------------------


def test_a_pruned_candidate_is_trained_pruned_fine_tuned_and_measured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = evaluate_pruned(monkeypatch, tmp_path)
    assert result.status == "valid", (result.failure_stage, result.failure_code, result.failure_detail)
    assert {"train", "prune", "export", "benchmark"} <= set(result.durations_s)
    pruning = result.quality["pruning"]
    assert pruning["version"] == PRUNING_VERSION and pruning["parameters_after"] < pruning["parameters_before"]
    assert pruning["finetune"]["epochs"] == 1 and pruning["finetune"]["steps"] >= 1
    assert set(pruning["val_accuracy"]) == {"before", "after_prune"} and pruning["groups"] and pruning["blocks"]
    assert all(b["requested_pct"] == 50 for b in pruning["blocks"])


def test_model_quality_records_parameters_macs_and_widths_after_the_pruning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = evaluate_pruned(monkeypatch, tmp_path)
    model = result.quality["model"]
    pruning = result.quality["pruning"]
    assert model["parameters_pruned"] == pruning["parameters_after"] < model["parameters"]  # the old key stays
    assert 0 < model["macs"] < model["macs_unpruned"]
    assert all(w < 64 for w in model["widths"]["blocks"])  # the block outputs shrank (identity ones follow their input)


def test_an_unpruned_candidate_records_equal_parameters_and_macs_and_has_no_pruning_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = evaluate_pruned(monkeypatch, tmp_path, raw=pruned_raw(pct=0))
    assert result.status == "valid", (result.failure_stage, result.failure_code, result.failure_detail)
    model = result.quality["model"]
    assert model["parameters_pruned"] == model["parameters"] and model["macs"] == model["macs_unpruned"]
    assert "prune" not in result.durations_s and "pruning" not in result.quality


def test_the_pruned_network_is_what_gets_exported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The ONNX weights have the pruned shapes: an engine built from it is smaller, as a measurement must be."""
    import onnx

    pruned = evaluate_pruned(monkeypatch, tmp_path / "p")
    full = evaluate_pruned(monkeypatch, tmp_path / "f", raw=pruned_raw(pct=0))

    def weight_elements(result: Any, base: Path) -> int:
        path = base / "artifacts" / result.artifact_paths["onnx"]
        model = onnx.load(str(path), load_external_data=False)
        return sum(int(torch.tensor(list(t.dims)).prod()) for t in model.graph.initializer if len(t.dims) >= 2)

    assert weight_elements(pruned, tmp_path / "p") < weight_elements(full, tmp_path / "f")


def test_pruning_composes_with_ptq_and_qat_and_the_layers_are_quantized_after_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for mode, name in (("ptq", "a"), ("qat", "b")):
        raw = pruned_raw(
            COMPRESSED_SLICE,
            stem_precision="int8",
            head_precision="fp32",
            int8_mode=mode,
            weight_granularity="per_channel",
            calibration="max",
        )
        raw = dataclasses.replace(raw, blocks=(dataclasses.replace(raw.blocks[0], precision="int8"), *raw.blocks[1:]))
        result = evaluate_pruned(monkeypatch, tmp_path / name, raw=raw, space=COMPRESSED_SLICE)
        assert result.status == "valid", (mode, result.failure_stage, result.failure_code, result.failure_detail)
        assert {"prune", "quantize"} <= set(result.durations_s) and ("qat" in result.durations_s) == (mode == "qat")
        assert result.quality["precision"]["ok"] and result.quality["precision"]["regions"]["stem"]["executed"] == [
            "Int8"
        ]
        assert result.quality["quantization"]["layers"] >= 2 and result.quality["pruning"]["parameters_after"] > 0


def test_a_pruning_error_is_an_invalid_candidate_at_the_prune_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def refuse(model: Any, pcts: Any) -> Any:
        raise PruningError("group_too_small", "block0.out would keep fewer than 16 channels")

    monkeypatch.setattr(evaluator_mod, "prune_network", refuse)
    result = evaluate_pruned(monkeypatch, tmp_path)
    assert (result.status, result.failure_stage, result.failure_code) == ("invalid", "prune", "group_too_small")
    assert "fewer than 16" in (result.failure_detail or "") and result.fitness is None
    assert "export" not in result.durations_s and "benchmark" not in result.durations_s


def test_a_nonfinite_loss_in_the_fine_tune_is_an_invalid_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cnnlab.tcc.model_pipeline.trainer import TrainingDiverged

    real, calls = evaluator_mod.train_model, itertools.count()

    def train(model: Any, *args: Any, **kwargs: Any) -> Any:
        if next(calls) == 1:  # the fine-tune after the pruning
            raise TrainingDiverged("loss is nan")
        return real(model, *args, **kwargs)

    monkeypatch.setattr(evaluator_mod, "train_model", train)
    result = evaluate_pruned(monkeypatch, tmp_path)
    assert (result.status, result.failure_stage, result.failure_code) == ("invalid", "prune", "nonfinite_loss")


def test_the_training_time_limit_covers_the_fine_tune_too(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The first training stays under the limit; the clock then passes it during the fine-tune."""
    reading = itertools.count()
    steps_in_training = 2  # 32 images, batches of 16, one epoch: two checks

    def now() -> float:
        i = next(reading)
        return 0.0 if i <= steps_in_training else 100.0  # the start and the two checks of the training, then a jump

    monkeypatch.setattr(evaluator_mod, "_now", now)
    evaluator, config, data = evaluator_for(monkeypatch, tmp_path)
    evaluator.config = dataclasses.replace(config, train_time_limit_s=10.0)
    ctx = pilot_cache_context(
        data, evaluator.config, 11, "test", PRUNE_FP32_SLICE.space_hash(), backend={"synthetic": True}
    )
    record = prepare_in_slice(pruned_raw(), PRUNE_FP32_SLICE, ctx).record
    request = EvaluationRequest(
        "run1", "c1", "c1.a1", "pilot", record, 11, 0, "ds", "env", "proto", "plan", tmp_path / "artifacts"
    )
    result = evaluator.evaluate(request)
    assert (result.status, result.failure_stage, result.failure_code) == ("invalid", "train", "train_time_limit")
    assert "prune" not in result.durations_s  # stopped inside the fine-tune, before the stage finished


def test_the_pruning_policy_is_part_of_the_cache_identity() -> None:
    assert PRUNING_DEFINITION["version"] == PRUNING_VERSION and PRUNING_DEFINITION["seed"] == "train seed + 2"
    from cnnlab.tcc.model_pipeline.demo_data import DemoData
    from cnnlab.tcc.model_pipeline.evaluator import EvaluatorConfig

    x = torch.zeros(2, 3, 32, 32)
    data = DemoData(x, torch.zeros(2, dtype=torch.long), x, torch.zeros(2, dtype=torch.long), {"schema": "t"})
    ctx = pilot_cache_context(data, EvaluatorConfig(), 1, "t", "0" * 64, backend={})
    assert dict(ctx.pruning_policy) == PRUNING_DEFINITION


# --- builder helpers ------------------------------------------------------------------------------------------------------


def test_effective_widths_read_the_real_channels_of_a_pruned_network() -> None:
    from cnnlab.tcc.model_pipeline.pruning import prune_network

    genome = express(pruned_raw(pct=0), SPACE_V1).genome
    model, _ = build_model(genome, 1)
    before = effective_widths(model)
    assert before["blocks"] == [64, 64, 64] and before["stem"] == 32 and before["head_hidden"] is None
    macs, params = count_macs(model), count_parameters(model)
    prune_network(model, [50, 50, 50])
    after = effective_widths(model)
    assert after["stem"] == 32 and all(w < 64 for w in after["blocks"])
    assert count_macs(model) < macs and count_parameters(model) < params
