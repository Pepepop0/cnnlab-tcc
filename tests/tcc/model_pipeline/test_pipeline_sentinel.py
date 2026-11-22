"""Sentinel network (CPU): buildable genome, engine reuse by hash, drift and V audit rules."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from cnnlab.tcc.model_pipeline import sentinel as sentinel_mod
from cnnlab.tcc.model_pipeline.builder import build_model
from cnnlab.tcc.model_pipeline.sentinel import (
    SENTINEL_GENOME,
    SENTINEL_SEED,
    drift,
    sentinel_engine,
    summarize,
    v_audit,
)
from cnnlab.tcc.science.genome import ExpressedGenome

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "piloto"))
import ruido  # noqa: E402


def measurement(
    l_ms: float = 0.3, v_mb: float = 10.0, context_mb: float = 2.0, s_mb: float = 8.1, sha: str = "a" * 64
) -> dict[str, Any]:
    return {
        "plan_sha256": sha,
        "l_ms": l_ms,
        "v_mb": v_mb,
        "v_free_drop_mb": 48.2,
        "engine_context_mb": context_mb,
        "s_mb": s_mb,
    }


def test_sentinel_genome_builds_and_runs() -> None:
    model, info = build_model(ExpressedGenome.from_dict(SENTINEL_GENOME), SENTINEL_SEED)
    with torch.no_grad():
        assert model.eval()(torch.zeros(1, 3, 32, 32)).shape == (1, 100)
    ops = {b["op"] for b in SENTINEL_GENOME["blocks"]}
    assert {"standard", "mbconv_e3", "dw_separable", "grouped_g4"} <= ops
    assert info.parameters > 100_000


def test_drift_flags_above_five_percent() -> None:
    ref = measurement()
    assert drift(ref, measurement(l_ms=0.314))["drifted"] is False  # 4.7%
    out = drift(ref, measurement(l_ms=0.318))
    assert out["drifted"] is True and out["l_rel"] == pytest.approx(0.06)
    assert drift(ref, measurement(v_mb=10.6))["drifted"] is True
    with pytest.raises(ValueError, match="different engines"):
        drift(ref, measurement(sha="b" * 64))


def test_v_audit_compares_context_plus_weights() -> None:
    # values seen in the audit of 06/10/2026: 6.82 MB context + 14.78 MB plan, V 21.51 MB
    assert v_audit(measurement(v_mb=21.51, context_mb=6.82, s_mb=14.78))["agrees"] is True
    assert v_audit(measurement(v_mb=0.30, context_mb=0.21, s_mb=0.26))["agrees"] is True  # within 1 MB
    assert v_audit(measurement(v_mb=3.0, context_mb=1.0, s_mb=0.5))["agrees"] is False
    assert v_audit(measurement(v_mb=200.0, context_mb=50.0, s_mb=159.0))["agrees"] is True  # 4.5%
    assert v_audit(measurement(v_mb=200.0, context_mb=50.0, s_mb=162.0))["agrees"] is False


def test_engine_is_reused_while_its_hash_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[Path] = []

    class Exported:
        def __init__(self, path: Path) -> None:
            self.onnx_path = path

    class Built:
        plan_bytes = 1234
        builder = {"fake": True}

    def fake_export(_model: Any, folder: Path, name: str = "model") -> Exported:
        path = folder / f"{name}.onnx"
        path.write_bytes(b"onnx")
        return Exported(path)

    def fake_build(_onnx: Path, plan: Path) -> Built:
        builds.append(plan)
        plan.write_bytes(b"plan-%d" % len(builds))
        return Built()

    monkeypatch.setattr(sentinel_mod, "export_network", fake_export)
    monkeypatch.setattr(sentinel_mod, "build_engine", fake_build)
    plan, manifest = sentinel_engine(tmp_path)
    again, _ = sentinel_engine(tmp_path)
    assert again == plan and len(builds) == 1
    plan.write_bytes(b"tampered")  # a plan that no longer matches its manifest is rebuilt
    _, rebuilt = sentinel_engine(tmp_path)
    assert len(builds) == 2 and rebuilt["plan_sha256"] != manifest["plan_sha256"]


def test_summary_reads_the_bench_report() -> None:
    report = {
        "latency": {"median_ms": 0.25, "p5_ms": 0.24, "p95_ms": 0.3, "iqr_ms": 0.01, "n": 1000.0},
        "memory": {
            "v_bytes_allocator_peak": 12_000_000,
            "free_mem_drop_bytes_audit": 14_680_064,
            "engine_device_memory_size_v2": 3_000_000,
        },
    }
    out = summarize(report, {"plan_sha256": "c" * 64, "plan_bytes": 2_000_000})
    assert (out["l_ms"], out["v_mb"], out["s_mb"]) == (0.25, 12.0, 2.0)
    assert out["v_free_drop_mb"] == pytest.approx(14.680064)


def test_noise_summary_cv_of_session_medians() -> None:
    sessions = [measurement(l_ms=x) for x in (0.30, 0.31, 0.29, 0.30)]
    out = ruido.summarize(sessions)
    assert out["cv_of_medians"] == pytest.approx(0.0272, abs=1e-3)
    assert out["v_audit_agrees"] is True and out["n_sessions"] == 4
    assert ruido.cv([1.0]) is None
