"""Final evaluation of a finalist in two passes (task G9): final_build never sees the test; final_test runs it once (CPU).

The official test is a fake here: a source with the right protocol (own sizes, a spy on every read). The real file is
never opened before the finalist list and its artifacts are frozen, and these tests do not need it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
from test_pipeline_training import inspector_from_onnx  # same directory

from cnnlab.tcc.model_pipeline import data_access, final_eval
from cnnlab.tcc.model_pipeline import evaluator as evaluator_mod
from cnnlab.tcc.model_pipeline.bench import BenchProcessError, classification_summary
from cnnlab.tcc.model_pipeline.data_access import SealedSplitError, grant_search_access
from cnnlab.tcc.model_pipeline.engine import BuiltEngine
from cnnlab.tcc.model_pipeline.final_eval import (
    FinalConfig,
    MemoryLedger,
    TestAccess,
    TestAlreadyEvaluated,
    final_build,
    final_grant,
    final_test,
    per_class_metrics,
    prepare_final_data,
)
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.operators import COMPRESSED_SLICE, FP32_SLICE
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.space import SPACE_V1

N_TRAIN, N_TEST = 200, 100
CONFIG = FinalConfig(epochs=1, builds=3, batch_size=16, calibration_examples=16, gpu_wait_s=1.0)


@dataclass
class FakeCifar:
    """The official files with the protocol's shape but tiny sizes; ``events`` is the spy."""

    events: list[str] = field(default_factory=list)

    def read(self, official_file: str) -> tuple[torch.Tensor, torch.Tensor]:
        self.events.append(f"read:{official_file}")
        n = N_TRAIN if official_file == "train" else N_TEST
        generator = torch.Generator().manual_seed(1 if official_file == "train" else 2)
        images = torch.randint(0, 256, (n, 32, 32, 3), dtype=torch.uint8, generator=generator)
        return images, torch.arange(n, dtype=torch.int64) % 10  # 10 classes: a 10% validation split stays small


class SpyLedger(MemoryLedger):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def claim(self, engine_sha256: str) -> None:
        self.events.append("claim")
        super().claim(engine_sha256)

    def release(self, engine_sha256: str) -> None:
        self.events.append("release")
        super().release(engine_sha256)


@pytest.fixture(autouse=True)
def small_official_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(data_access.OFFICIAL_SIZES, "train", N_TRAIN)
    monkeypatch.setitem(data_access.OFFICIAL_SIZES, "test", N_TEST)


class Rig:
    """Stand-ins for the engine build and the benchmark, with the real report structure and an event log."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        source: FakeCifar,
        l_by_build: tuple[float, ...] = (0.30, 0.10, 0.20),
        executed: dict[str, str] | None = None,
    ) -> None:
        self.source, self.builds = source, 0
        self.l_by_build, self.executed, self.fail_test_bench = l_by_build, executed or {}, False
        monkeypatch.setattr(final_eval, "build_engine", self.build)
        monkeypatch.setattr(evaluator_mod, "run_isolated", self.bench)

    def build(self, onnx_path: Path, plan_path: Path, workspace_bytes: int = 0, **_: Any) -> BuiltEngine:
        index = self.builds
        self.builds += 1
        self.source.events.append(f"build:{index}")
        plan = hashlib.sha256(onnx_path.read_bytes()).digest() * (20 + index)  # differs between the builds
        plan_path.write_bytes(plan)
        inspector = inspector_from_onnx(onnx_path, {self._node(onnx_path, r): d for r, d in self.executed.items()})
        return BuiltEngine(plan_path, len(plan), hashlib.sha256(plan).hexdigest(), inspector, {"standin": True}, 0.0)

    @staticmethod
    def _node(onnx_path: Path, region: str) -> str:
        from cnnlab.tcc.model_pipeline.precision import weight_layers

        return next(layer["node"] for layer in weight_layers(onnx_path)[0] if layer["region"] == region)

    def bench(
        self, plan_path: Path, out_path: Path, *, val_path: Path, ref_logits_path: Path, **_: Any
    ) -> dict[str, Any]:
        import numpy as np

        labels = torch.from_numpy(np.load(val_path)["y"])
        kind = "test" if len(labels) == N_TEST else "val"
        self.source.events.append(f"bench:{kind}")
        if kind == "test" and self.fail_test_bench:
            raise BenchProcessError("spawn", None, "the child never started")
        logits = torch.from_numpy(np.load(ref_logits_path))
        preds = logits.argmax(1)
        in_top5 = (logits.topk(5, dim=1).indices == labels.view(-1, 1)).any(1)
        name = plan_path.stem
        index = int(name.removeprefix("engine")) if name.startswith("engine") and name[6:].isdigit() else 0
        report = {
            "bench_version": "tcc-bench/preliminary-v4",
            "cuda_graph": True,
            "latency": {"median_ms": self.l_by_build[index % len(self.l_by_build)], "p5_ms": 0.0, "p95_ms": 1.0},
            "memory": {"v_bytes_allocator_peak": 1_000_000 * (index + 1)},
            "e": {
                "e_fraction": float((preds != labels).float().mean()),
                "errors": int((preds != labels).sum()),
                "n": int(len(labels)),
                "all_finite": True,
                **classification_summary(preds, labels, in_top5, 100),
            },
            "quality": {"parity_vs_torch_fp32": {"n": int(len(labels)), "max_abs_diff": 0.0, "top1_agreement": 1.0}},
        }
        out_path.write_text(json.dumps(report), encoding="utf-8")
        return report


def expressed(slice_: Any = FP32_SLICE, seed: int = 3) -> dict[str, Any]:
    return express(slice_.sample(GeneRng(seed)), SPACE_V1).genome.to_dict()


def build_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    genome: dict[str, Any] | None = None,
    seed: int = 77,
    config: FinalConfig = CONFIG,
    **rig_kwargs: Any,
) -> tuple[Any, FakeCifar, Rig]:
    source = FakeCifar()
    rig = Rig(monkeypatch, source, **rig_kwargs)
    grant = final_grant(source, "final-run", "list-1")
    data = prepare_final_data(grant, source, torch.device("cpu"))
    result = final_build(
        genome or expressed(), seed, config, data, tmp_path / "w", tmp_path / "art", label="finalist-a", device="cpu"
    )
    return result, source, rig


def run_test_pass(
    source: FakeCifar, tmp_path: Path, manifest: Any, ledger: Any = None, config: FinalConfig = CONFIG
) -> Any:
    access = TestAccess(final_grant(source, "final-run", "list-1"), source, ledger or MemoryLedger())
    return final_test(manifest, tmp_path / "art", config, access, tmp_path / "wt", device="cpu")


# --- the barrier ------------------------------------------------------------------------------------------------------


def test_test_access_needs_a_finalist_grant_that_holds_the_test_split() -> None:
    source = FakeCifar()
    search = grant_search_access(
        "run",
        "pilot",
        [
            data_access.SplitIndices("train", "train", tuple(range(0, 180))),
            data_access.SplitIndices("val", "train", tuple(range(180, 200))),
        ],
    )
    with pytest.raises(SealedSplitError):
        TestAccess(search, source, MemoryLedger())
    assert source.events == []


def test_the_grant_and_the_final_data_never_touch_the_test_file() -> None:
    source = FakeCifar()
    grant = final_grant(source, "run", "list")
    data = prepare_final_data(grant, source, torch.device("cpu"))
    assert set(source.events) == {"read:train"}  # labels for the split, then the train/val images
    assert data.x_train.shape[0] + data.x_val.shape[0] == N_TRAIN and data.manifest["official_test_read"] is False
    assert "test" in data.manifest["fingerprints"]  # the index set of the test is known, its images are not loaded


def test_the_split_is_the_poc_45k_5k_one_scaled_down() -> None:
    grant = final_grant(FakeCifar(), "run", "list")
    assert len(grant.splits["train"].indices) == 180 and len(grant.splits["val"].indices) == 20
    assert not set(grant.splits["train"].indices) & set(grant.splits["val"].indices)
    assert grant.splits["test"].indices == tuple(range(N_TEST)) and grant.finalist_list_id == "list"


def test_config_needs_an_odd_number_of_builds() -> None:
    for builds in (0, 2, 4):
        with pytest.raises(ValueError, match="odd"):
            FinalConfig(builds=builds)
    assert FinalConfig().epochs == 30 and FinalConfig().builds == 3


# --- pass 1: the build ----------------------------------------------------------------------------------------------------


def test_the_build_makes_three_engines_picks_the_median_one_and_never_reads_the_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, source, rig = build_pass(monkeypatch, tmp_path, l_by_build=(0.30, 0.10, 0.20))
    assert result.status == "valid", (result.failure_stage, result.failure_code, result.failure_detail)
    assert [b.index for b in result.builds] == [0, 1, 2] and len({b.plan_sha256 for b in result.builds}) == 3
    assert result.chosen_build == 2 and result.l_ms == pytest.approx(0.20)  # L 0.30, 0.10, 0.20: the median is build 2
    assert result.s_mb == pytest.approx(sorted(b.s_mb for b in result.builds)[1])
    assert result.v_mb == pytest.approx(2.0)  # V 1, 2, 3 MB
    assert result.engine_sha256 == result.builds[2].plan_sha256 and rig.builds == 3
    assert "read:test" not in source.events and "bench:test" not in source.events
    assert result.quality["engine_builds"][2]["l_ms"] == pytest.approx(0.20)


def test_build_artifacts_are_written_and_hashed_with_the_references_for_the_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, *_ = build_pass(monkeypatch, tmp_path)
    assert result.artifact_paths["engine"] == "finalist-a/engine.plan"
    assert {
        "engine",
        "inspector",
        "precision",
        "bench",
        "train_log",
        "builds",
        "onnx",
        "reference_fp32",
        "reference_simulated",
        "engine_build0",
        "engine_build1",
    } <= set(result.artifact_paths)
    assert "engine_build2" not in result.artifact_paths  # build 2 is the final engine itself
    for kind, relative in result.artifact_paths.items():
        assert hashlib.sha256((tmp_path / "art" / relative).read_bytes()).hexdigest() == result.artifact_sha256[kind]
    assert result.artifact_sha256["engine"] == result.engine_sha256
    assert json.loads(json.dumps(result.to_dict()))["engine_sha256"] == result.engine_sha256  # the manifest is JSON


def test_training_runs_the_configured_epochs_on_the_whole_train_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, *_ = build_pass(monkeypatch, tmp_path, config=dataclasses.replace(CONFIG, epochs=2))
    train = result.quality["train"]
    assert train["epochs"] == 2 and train["steps"] == 2 * 12  # 180 images in batches of 16
    assert result.quality["protocol_stage"] == "final" and result.quality["final"]["epochs"] == 2


def test_pruning_and_quantization_are_part_of_the_final_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = COMPRESSED_SLICE.sample(GeneRng(11))
    raw = dataclasses.replace(
        raw,
        stem_precision="int8",
        int8_mode="qat",
        weight_granularity="per_channel",
        blocks=tuple(dataclasses.replace(b, prune_pct=25, precision="fp32") for b in raw.blocks),
    )
    genome = express(raw, SPACE_V1).genome.to_dict()
    result, *_ = build_pass(monkeypatch, tmp_path, genome=genome)
    assert result.status == "valid", (result.failure_stage, result.failure_code, result.failure_detail)
    assert {"prune", "quantize", "qat"} <= set(result.durations_s)
    assert result.quality["pruning"]["parameters_after"] < result.quality["pruning"]["parameters_before"]
    assert result.quality["quantization"]["mode"] == "qat"


def test_an_engine_that_did_not_run_the_requested_type_is_an_invalid_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, source, _ = build_pass(monkeypatch, tmp_path, executed={"stem": "Half"})
    assert (result.status, result.failure_stage, result.failure_code) == (
        "invalid",
        "precision_check",
        "precision_not_executed",
    )
    assert result.engine_sha256 is None and not result.artifact_paths and "read:test" not in source.events


def test_a_representation_the_pipeline_cannot_run_is_an_invalid_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    genome = expressed()
    genome["head_precision"] = "fp8"
    result, source, rig = build_pass(monkeypatch, tmp_path, genome=genome)
    assert (result.status, result.failure_code) == ("invalid", "unsupported_representation")
    assert rig.builds == 0 and "read:test" not in source.events


# --- pass 2: the official test --------------------------------------------------------------------------------------------


def test_the_test_runs_on_the_final_engine_after_the_ledger_claims_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built, source, _ = build_pass(monkeypatch, tmp_path)
    ledger = SpyLedger(source.events)
    result = run_test_pass(source, tmp_path, built.to_dict(), ledger)
    assert result.status == "valid", (result.failure_stage, result.failure_code, result.failure_detail)
    assert result.engine_sha256 == built.engine_sha256 and ledger.claimed == [built.engine_sha256]
    tail = source.events[source.events.index("claim") :]
    assert [e for e in tail if e != "read:train"] == ["claim", "read:test", "bench:test"]
    assert source.events.count("read:test") == 1


def test_test_metrics_come_from_the_engine_with_top5_and_per_class(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built, source, _ = build_pass(monkeypatch, tmp_path)
    test = run_test_pass(source, tmp_path, built).test
    assert test is not None and test["n"] == N_TEST and 0 <= test["e_fraction"] <= 1
    assert test["top5_accuracy"] >= test["top1_accuracy"] == pytest.approx(1 - test["e_fraction"])
    per_class = test["per_class"]
    assert len(per_class) == 100 and sum(c["support"] for c in per_class) == N_TEST
    assert sum(c["true_positive"] for c in per_class) == N_TEST - test["errors"]
    assert test["parity_vs_torch_fp32"]["n"] == N_TEST and test["split_fingerprint"]
    assert 0 <= test["torch_top1_error_fp32"] <= 1 and 0 <= test["torch_top1_error_simulated"] <= 1


def test_the_quantized_final_engine_is_tested_against_its_own_simulated_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = COMPRESSED_SLICE.sample(GeneRng(11))
    raw = dataclasses.replace(
        raw,
        stem_precision="int8",
        int8_mode="ptq",
        weight_granularity="per_channel",
        calibration="max",
        blocks=tuple(dataclasses.replace(b, precision="fp32") for b in raw.blocks),
    )
    built, source, _ = build_pass(monkeypatch, tmp_path, genome=express(raw, SPACE_V1).genome.to_dict())
    result = run_test_pass(source, tmp_path, built)
    assert result.status == "valid" and result.test is not None
    assert result.test["parity_vs_torch_fp32"]["n"] == N_TEST  # the engine is compared with the FP32 reference


def test_a_changed_artifact_is_refused_before_the_test_is_opened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for kind in ("engine", "reference_fp32", "reference_simulated"):
        built, source, _ = build_pass(monkeypatch, tmp_path / kind)
        victim = tmp_path / kind / "art" / built.artifact_paths[kind]
        victim.write_bytes(victim.read_bytes() + b"x")
        ledger = SpyLedger(source.events)
        result = final_test(
            built,
            tmp_path / kind / "art",
            CONFIG,
            TestAccess(final_grant(source, "run", "list"), source, ledger),
            tmp_path / kind / "wt",
            device="cpu",
        )
        assert (result.status, result.failure_stage, result.failure_code) == ("invalid", "verify", "artifact_changed")
        assert "read:test" not in source.events and ledger.claimed == [] and result.test is None


def test_the_manifest_of_a_failed_build_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    built, source, _ = build_pass(monkeypatch, tmp_path, executed={"stem": "Half"})
    result = run_test_pass(source, tmp_path, built)
    assert (result.status, result.failure_code) == ("invalid", "no_final_engine") and "read:test" not in source.events


def test_an_artifact_already_evaluated_raises_and_the_test_stays_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built, source, _ = build_pass(monkeypatch, tmp_path)
    ledger = MemoryLedger()
    ledger.claim(built.engine_sha256)  # type: ignore[arg-type]
    with pytest.raises(TestAlreadyEvaluated):
        run_test_pass(source, tmp_path, built, ledger)
    assert "read:test" not in source.events


def test_a_measurement_that_died_without_a_result_gives_the_claim_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built, source, rig = build_pass(monkeypatch, tmp_path)
    ledger = SpyLedger(source.events)
    rig.fail_test_bench = True
    first = run_test_pass(source, tmp_path, built, ledger)
    assert first.status == "interrupted" and ledger.claimed == [] and first.test is None
    rig.fail_test_bench = False
    second = run_test_pass(source, tmp_path, built, ledger)
    assert second.status == "valid" and ledger.claimed == [built.engine_sha256]
    with pytest.raises(TestAlreadyEvaluated):  # once there is a result, the engine is spent
        run_test_pass(source, tmp_path, built, ledger)


def test_per_class_metrics_on_a_hand_computed_case() -> None:
    rows = per_class_metrics({"support": [2, 2, 2], "predicted": [2, 3, 1], "true_positive": [1, 2, 1]})
    assert [r["precision"] for r in rows] == [0.5, 2 / 3, 1.0] and [r["recall"] for r in rows] == [0.5, 1.0, 0.5]
    assert rows[0]["f1"] == pytest.approx(0.5) and rows[2]["f1"] == pytest.approx(2 / 3)
    assert per_class_metrics({"support": [0], "predicted": [0], "true_positive": [0]})[0]["f1"] == 0.0


def test_the_memory_ledger_claims_each_engine_once_and_can_release() -> None:
    ledger = MemoryLedger()
    ledger.claim("a" * 64)
    ledger.claim("b" * 64)
    with pytest.raises(TestAlreadyEvaluated):
        ledger.claim("a" * 64)
    ledger.release("a" * 64)
    ledger.claim("a" * 64)
