"""Spike plumbing on CPU: variants, stage classification and report hygiene.

GPU/TensorRT paths run only through the spike command (or with
CNNLAB_TCC_GPU_TESTS=1), never by default: another job may own the GPU.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from cnnlab.tcc.model_pipeline import spike_trt
from cnnlab.tcc.model_pipeline.environment import PACKAGES, probe_environment
from cnnlab.tcc.model_pipeline.spike_trt import (
    IMPLEMENTED_STAGES,
    TRT_STAGES,
    VARIANTS,
    StageRecord,
    build_variant,
    campaign_gpu_processes,
    compare_logits,
    export_onnx,
    run_spike,
    summarize_inspector,
    synthetic_inputs,
    trt_stages,
)


def logits(variant: str, seed: int = 3) -> torch.Tensor:
    model = build_variant(variant, seed)
    with torch.no_grad():
        return torch.cat([model(x.unsqueeze(0)) for x in synthetic_inputs(seed + 2, 4)])


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_every_variant_runs_batch_one_on_cpu(variant: str) -> None:
    out = logits(variant)
    assert out.shape == (4, 100) and out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_variants_share_float_weights() -> None:
    reference = logits("fp32")
    for variant in ("fp16", "bf16", "int8", "mixed"):
        diff = compare_logits(logits(variant), reference)
        assert 0 < diff["max_abs_diff"] < 0.05, variant


def test_build_is_deterministic_and_leaves_global_rng_alone() -> None:
    state = torch.get_rng_state()
    a, b = logits("int8", 9), logits("int8", 9)
    assert torch.equal(a, b)
    assert torch.equal(torch.get_rng_state(), state)


def test_int8_variant_folds_bn_and_uses_fake_quant() -> None:
    model = build_variant("int8", 1)
    kinds = {type(m).__name__ for m in model.modules()}
    assert "_FakeQuantConv2d" in kinds and "_FakeQuantLinear" in kinds
    assert "BatchNorm2d" not in kinds
    assert "BatchNorm2d" in {type(m).__name__ for m in build_variant("fp32", 1).modules()}


def test_gpu_guard_ignores_desktop_contexts_and_itself() -> None:
    apps = [  # base names, as stored by nvidia_smi_snapshot
        {"pid": "13356", "process_name": "explorer.exe"},
        {"pid": "1060", "process_name": "[Insufficient Permissions]"},
        {"pid": "42", "process_name": "python.exe"},
        {"pid": "77", "process_name": "python.exe"},
    ]
    assert [a["pid"] for a in campaign_gpu_processes(apps, own_pid=42)] == ["77"]
    assert campaign_gpu_processes(apps[:2], own_pid=1) == []


def test_trace_tail_drops_directories() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        tail = spike_trt._trace_tail()
    assert 'File "test_pipeline_spike.py"' in tail
    assert str(Path(__file__).parent) not in tail


def test_non_ok_stage_needs_a_detail() -> None:
    with pytest.raises(ValueError):
        StageRecord("trt_build", "fp32", "blocked", "import tensorrt")
    StageRecord("trt_build", "fp32", "ok", "build")


def _raise_missing(*_: Any, **__: Any) -> None:
    raise ModuleNotFoundError("No module named 'onnx'")


@pytest.mark.parametrize(("deps_present", "expected"), [(False, "blocked"), (True, "failed")])
def test_export_failure_is_classified_by_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, deps_present: bool, expected: str
) -> None:
    monkeypatch.setattr(torch.onnx, "export", _raise_missing)
    record = export_onnx(
        build_variant("fp32", 1),
        synthetic_inputs(0, 1),
        tmp_path / "m.onnx",
        "torchscript",
        "fp32",
        available=lambda _: deps_present,
    )
    assert record.status == expected
    assert "No module named 'onnx'" in record.detail


def test_export_success_records_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_export(_model: Any, _args: Any, path: str, **_: Any) -> None:
        Path(path).write_bytes(b"onnx-bytes")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    record = export_onnx(
        build_variant("fp32", 1),
        synthetic_inputs(0, 1),
        tmp_path / "m.onnx",
        "dynamo",
        "fp32",
        available=lambda _: True,
    )
    assert record.status == "ok"
    assert record.observations["bytes"] == len(b"onnx-bytes")
    assert len(record.observations["sha256"]) == 64
    assert record.observations["external_data"] == {}


def test_export_hashes_dynamo_external_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_export(_model: Any, _args: Any, path: str, **_: Any) -> None:
        Path(path).write_bytes(b"graph")
        Path(path + ".data").write_bytes(b"weights!")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    record = export_onnx(
        build_variant("fp32", 1),
        synthetic_inputs(0, 1),
        tmp_path / "m.onnx",
        "dynamo",
        "fp32",
        available=lambda _: True,
    )
    assert record.observations["external_data"]["m.onnx.data"]["bytes"] == len(b"weights!")
    assert record.observations["total_bytes"] == len(b"graph") + len(b"weights!")


def test_trt_stages_block_cleanly_without_tensorrt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, *args: Any) -> Any:
        if name == "tensorrt":
            raise ModuleNotFoundError("No module named 'tensorrt'")
        return real_import(name, *args)

    monkeypatch.setattr(spike_trt.importlib, "import_module", fake_import)
    x = synthetic_inputs(0, 2)
    records = trt_stages(tmp_path / "m.onnx", tmp_path, "fp32", x, x, x, "dynamo")
    assert [r.status for r in records] == ["blocked", "skipped", "skipped", "skipped"]
    assert [r.stage for r in records] == [f"{s}[dynamo]" for s in TRT_STAGES]
    assert records[0].detail == "ModuleNotFoundError: No module named 'tensorrt'"


def test_inspector_summary_reads_trt11_datatype_keys() -> None:
    # Shape observed in the TRT 11.3 inspector JSON of wave 2A (trimmed).
    info = {
        "Layers": [
            {"Name": "__mye_0", "LayerType": "custom_layer", "Outputs": []},
            {
                "Name": "/stem/conv/Conv_myl0_2",
                "LayerType": "correlation",
                "Outputs": [{"Datatype": "Int8", "Format": "(NHWC) Channel major format"}],
                "TacticName": "sm80_xmma_fprop_first_layer_i8i8_i8i32_f32",
            },
            {"Name": "legacy", "LayerType": "Scale", "Outputs": [{"Format/Datatype": "Half"}]},
        ],
        "I/O Tensors": [],
    }
    summary = summarize_inspector(info)
    assert summary["output_dtypes"] == {"Int8": 1, "Half": 1}
    assert summary["unfused_scale_layers"] == 1
    assert summary["per_layer"][1]["tactic"].endswith("i8i8_i8i32_f32")
    assert summarize_inspector({})["layers"] == 0


def test_run_spike_on_cpu_reports_every_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.onnx, "export", _raise_missing)
    report = run_spike(tmp_path / "out", variants=("fp32", "int8"), device="cpu", available=lambda _: False)
    stored = json.loads((tmp_path / "out" / "spike_report.json").read_text(encoding="utf-8"))
    assert stored["implemented_stages"] == list(IMPLEMENTED_STAGES)
    by_variant = {v: [(r.stage, r.status) for r in report.stages if r.variant == v] for v in ("fp32", "int8")}
    for v, stages in by_variant.items():
        assert stages == [
            ("torch_build", "ok"),
            ("torch_forward", "ok"),
            ("onnx_export[torchscript]", "blocked"),
            ("onnx_export[dynamo]", "blocked"),
            *[
                (stage, "skipped")
                for exporter in ("torchscript", "dynamo")
                for stage in (f"onnx_inspect[{exporter}]", *(f"{s}[{exporter}]" for s in TRT_STAGES))
            ],
        ], v
    text = (tmp_path / "out" / "spike_report.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text and str(tmp_path).replace("\\", "\\\\") not in text


def test_environment_probe_lists_pipeline_packages_without_touching_cuda() -> None:
    env = probe_environment(touch_cuda=False)
    assert set(PACKAGES) <= set(env["packages"])
    assert env["packages"]["torch"] is not None
    assert "cuda_available" not in env["torch"]
    assert env["captured_utc"].endswith("Z")


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("CNNLAB_TCC_GPU_TESTS") != "1", reason="GPU tests are opt-in")
def test_variants_run_on_cuda() -> None:
    assert torch.cuda.is_available()
    x = synthetic_inputs(1, 2).cuda()
    for variant in VARIANTS:
        model = build_variant(variant, 1).cuda()
        with torch.no_grad():
            assert torch.isfinite(model(x[:1])).all()
