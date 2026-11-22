"""Sentinel network: a fixed engine re-measured to watch the measurement itself (TCC-01 section 3).

The protocol re-measures one fixed network every 50 evaluations; a drift of L or V above
5% marks the stretch of attempts measured meanwhile. This module provides the network, its
engine (built once and reused while its hash matches) and the measurement through the same
isolated benchmark the evaluator uses. It also provides the two checks the pilot needs:

* ``drift(reference, current)``: relative change of L and V against the run's first
  sentinel measurement;
* ``v_audit(measurement)``: the bench counts V through an instrumented allocator. The audit
  of 06/10/2026 (``tools/piloto/auditoria_v.py``, 25 engines) showed that the drop of free
  device memory is not an independent reading of V: TensorRT reserves the same 33.55 MB on
  every deserialization and 12.58 MB on the first execution, whatever the engine, and the
  rest moves in pools of 2 MiB. The engine-dependent memory is the execution context
  (``device_memory_size_v2``) plus the weights (the plan), and the allocator's V matched it
  within 0.5 MB on all 24 engines. That sum is what the audit compares V with.

The sentinel's weights are untrained and fixed by a seed: latency and memory do not depend
on their values, and no accuracy is measured.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from cnnlab.tcc.model_pipeline.bench import DEFAULT_ITERS, DEFAULT_WARMUP, run_isolated
from cnnlab.tcc.model_pipeline.builder import build_model
from cnnlab.tcc.model_pipeline.engine import build_engine, export_network
from cnnlab.tcc.model_pipeline.precision import typed_network
from cnnlab.tcc.science.genome import ExpressedGenome

SENTINEL_VERSION = "tcc-sentinel/v1"
SENTINEL_SEED = 20261005
DRIFT_LIMIT = 0.05  # TCC-01 section 3: a drift above 5% opens a new measurement
V_AUDIT_ABS_MB = 1.0  # plan metadata makes the sum differ from V by up to ~0.5 MB on tiny engines

# Mid-sized FP32 network with every op family the space has, residuals and all reductions kinds.
SENTINEL_GENOME: dict[str, Any] = {
    "schema": "tcc-genome-expressed/v1",
    "n_blocks": 4,
    "stem_channels": 32,
    "activation": "relu",
    "head": "gap_linear",
    "stem_precision": "fp32",
    "head_precision": "fp32",
    "log10_lr": -3.0,
    "log10_wd": -4.0,
    "head_dropout": 0.1,
    "blocks": [
        {
            "channels": 64,
            "op": "standard",
            "kernel": 3,
            "n_convs": 2,
            "residual": True,
            "downsample": "none",
            "precision": "fp32",
            "prune_pct": 0,
        },
        {
            "channels": 96,
            "op": "mbconv_e3",
            "kernel": 3,
            "n_convs": 2,
            "residual": True,
            "downsample": "conv_s2",
            "precision": "fp32",
            "prune_pct": 0,
        },
        {
            "channels": 128,
            "op": "dw_separable",
            "kernel": 5,
            "n_convs": 1,
            "residual": True,
            "downsample": "avgpool",
            "precision": "fp32",
            "prune_pct": 0,
        },
        {
            "channels": 256,
            "op": "grouped_g4",
            "kernel": 3,
            "n_convs": 1,
            "residual": False,
            "downsample": "maxpool",
            "precision": "fp32",
            "prune_pct": 0,
        },
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sentinel_engine(work_dir: Path) -> tuple[Path, dict[str, Any]]:
    """The sentinel's ``.plan`` under ``work_dir/sentinel``, built on first use and reused after.

    The manifest records the plan's SHA-256; a plan that no longer matches it is rebuilt,
    so a measurement never runs on a file other than the one recorded.
    """
    folder = Path(work_dir) / "sentinel"
    plan, manifest_path = folder / "sentinel.plan", folder / "sentinel.json"
    if plan.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("plan_sha256") == _sha256(plan) and manifest.get("version") == SENTINEL_VERSION:
            return plan, manifest
    folder.mkdir(parents=True, exist_ok=True)
    genome = ExpressedGenome.from_dict(SENTINEL_GENOME)
    model, info = build_model(genome, SENTINEL_SEED)
    typed = typed_network(model, [b["precision"] for b in SENTINEL_GENOME["blocks"]])
    exported = export_network(typed, folder, name="sentinel")
    built = build_engine(exported.onnx_path, plan)
    manifest = {
        "version": SENTINEL_VERSION,
        "seed": SENTINEL_SEED,
        "genome": SENTINEL_GENOME,
        "parameters": info.parameters,
        "plan_bytes": built.plan_bytes,
        "plan_sha256": _sha256(plan),
        "builder": built.builder,
        "built_at": datetime.now(UTC).isoformat(),
    }
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, manifest_path)
    return plan, manifest


def summarize(report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """The fields of one sentinel measurement that are recorded and compared."""
    latency, memory = report["latency"], report["memory"]
    return {
        "sentinel_version": SENTINEL_VERSION,
        "plan_sha256": manifest["plan_sha256"],
        "l_ms": latency["median_ms"],
        "l_p5_ms": latency["p5_ms"],
        "l_p95_ms": latency["p95_ms"],
        "l_iqr_ms": latency["iqr_ms"],
        "l_n": latency["n"],
        "v_mb": memory["v_bytes_allocator_peak"] / 1e6,
        "v_free_drop_mb": memory["free_mem_drop_bytes_audit"] / 1e6,
        "engine_context_mb": memory["engine_device_memory_size_v2"] / 1e6,
        "s_mb": manifest["plan_bytes"] / 1e6,
        "measured_at": datetime.now(UTC).isoformat(),
    }


def measure_sentinel(
    work_dir: Path, *, warmup: int = DEFAULT_WARMUP, iters: int = DEFAULT_ITERS, timeout_s: float = 900.0
) -> dict[str, Any]:
    """Measure the sentinel in a fresh process, exactly as the evaluator measures a candidate."""
    plan, manifest = sentinel_engine(work_dir)
    out = Path(work_dir) / "sentinel" / "bench.json"
    report = run_isolated(plan, out, warmup=warmup, iters=iters, timeout_s=timeout_s)
    return summarize(report, manifest)


def drift(reference: dict[str, Any], current: dict[str, Any], *, limit: float = DRIFT_LIMIT) -> dict[str, Any]:
    """Relative change of L and V from ``reference``; ``drifted`` when either passes ``limit``."""
    if reference["plan_sha256"] != current["plan_sha256"]:
        raise ValueError("sentinel measurements of different engines cannot be compared")
    rel = {key: abs(current[key] - reference[key]) / reference[key] for key in ("l_ms", "v_mb")}
    return {
        "l_rel": rel["l_ms"],
        "v_rel": rel["v_mb"],
        "limit": limit,
        "drifted": rel["l_ms"] > limit or rel["v_mb"] > limit,
    }


def v_audit(measurement: dict[str, Any], *, tolerance: float = 0.05) -> dict[str, Any]:
    """Allocator V against context memory plus weights; agreement within ``tolerance`` or 1 MB.

    The drop of free memory is reported for reference only: it carries TensorRT's fixed
    reservations (identical for every engine, so they do not change any comparison) and its
    pool granularity.
    """
    v = measurement["v_mb"]
    expected = measurement["engine_context_mb"] + measurement["s_mb"]
    gap = abs(v - expected)
    slack = max(tolerance * v, V_AUDIT_ABS_MB)
    return {
        "v_mb": v,
        "context_plus_weights_mb": expected,
        "gap_mb": gap,
        "allowed_gap_mb": slack,
        "agrees": gap <= slack,
        "v_free_drop_mb": measurement.get("v_free_drop_mb"),
    }
