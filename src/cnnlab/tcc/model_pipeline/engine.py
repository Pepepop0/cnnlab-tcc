"""Export and TensorRT build of a trained builder network.

Export policy (wave-2A report, P5): fold BN in PyTorch, then the dynamo ONNX
exporter; the ONNX graph and its external-data sidecar are hashed together.
The engine is built in strongly typed mode with FP32 I/O and batch 1, so the
types written in the graph (FP32 or FP16 per block, see ``precision.py``) are
the types TensorRT has to run. TF32 stays at the TensorRT default and is
recorded, so the build configuration is identical for every candidate.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cnnlab.tcc.model_pipeline.builder import fold_batchnorm

INPUT_SHAPE = (1, 3, 32, 32)
WORKSPACE_BYTES = 1 << 30
ENGINE_POLICY = "tcc-engine/typed-blocks-strongly-typed-dynamo-v2"


class ExportError(RuntimeError):
    """ONNX export failed (failure_stage ``export``)."""


class EngineBuildError(RuntimeError):
    """TensorRT parse or build failed (failure_stage ``trt_parse`` or ``trt_build``)."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ExportedModel:
    onnx_path: Path
    data_path: Path | None
    sha256: str  # graph plus external data
    total_bytes: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class BuiltEngine:
    plan_path: Path
    plan_bytes: int
    plan_sha256: str
    inspector: dict[str, Any]
    builder: dict[str, Any]
    duration_s: float


def write_atomic(path: Path, data: bytes) -> None:
    """Temp file, fsync, ``os.replace``: a reader never sees a partial artifact."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_fp32(model: nn.Module, out_dir: Path, name: str = "model") -> ExportedModel:
    """Fold BN and export the network entirely in FP32."""
    return export_network(fold_batchnorm(model).float().eval(), out_dir, name)


def export_network(folded: nn.Module, out_dir: Path, name: str = "model") -> ExportedModel:
    """Export an eval-mode network whose BN is already folded (FP32 input, batch 1); hash graph + data."""
    started = time.perf_counter()
    device = next(folded.parameters()).device
    example = torch.zeros(INPUT_SHAPE, device=device)
    path = out_dir / f"{name}.onnx"
    sink = io.StringIO()
    try:
        # The exporter prints progress with non-ASCII marks; capture it so a
        # cp1252 console cannot turn an export into UnicodeEncodeError.
        with contextlib.redirect_stdout(sink), torch.no_grad():
            torch.onnx.export(
                folded, (example,), str(path), input_names=["input"], output_names=["logits"], dynamo=True
            )
    except Exception as exc:
        raise ExportError(f"{type(exc).__name__}: {exc}"[:2000]) from exc
    data = path.with_name(path.name + ".data")
    h = hashlib.sha256(path.read_bytes())
    total = path.stat().st_size
    if data.exists():
        h.update(data.read_bytes())
        total += data.stat().st_size
    return ExportedModel(path, data if data.exists() else None, h.hexdigest(), total, time.perf_counter() - started)


def summarize_layers(info: dict[str, Any]) -> dict[str, Any]:
    layers = [layer for layer in info.get("Layers", []) if isinstance(layer, dict)]
    dtypes = Counter(
        str(t.get("Datatype") or t.get("Format/Datatype") or "?") for layer in layers for t in layer.get("Outputs", [])
    )
    return {
        "layers": len(layers),
        "layer_types": dict(Counter(str(layer.get("LayerType", "?")) for layer in layers)),
        "output_dtypes": dict(dtypes),
    }


def build_engine(
    onnx_path: Path,
    plan_path: Path,
    workspace_bytes: int = WORKSPACE_BYTES,
    timing_cache: bytes | None = None,
    timing_cache_out: Path | None = None,
) -> BuiltEngine:
    """Parse and build a strongly typed engine; write the plan atomically.

    By default every build times its tactics from scratch (``timing_cache`` None, ``timing_cache_out`` None), so
    two builds of one network may pick different tactics and differ in L by a few percent. ``timing_cache`` is a
    serialized TensorRT timing cache that the build reads (a *frozen* cache: layers it covers reuse the recorded
    tactic, others are timed as usual). ``timing_cache_out`` also writes the cache after the build, with what it
    learned (a *live* cache). The SHA-256 of the cache that was read, and of the one written, go to the builder
    record.
    """
    import tensorrt as trt

    started = time.perf_counter()
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise EngineBuildError("trt_parse", "; ".join(errors)[:2000] or "parse returned False")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cache = None
    cache_info: Any = "none (fresh per build)"
    if timing_cache is not None or timing_cache_out is not None:
        cache = config.create_timing_cache(timing_cache or b"")
        if not config.set_timing_cache(cache, False):
            raise EngineBuildError("trt_build", "the timing cache does not match this TensorRT version or device")
        cache_info = {
            "read_sha256": None if timing_cache is None else hashlib.sha256(timing_cache).hexdigest(),
            "read_bytes": 0 if timing_cache is None else len(timing_cache),
        }
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise EngineBuildError("trt_build", "build_serialized_network returned None")
    if cache is not None and timing_cache_out is not None:
        learned = bytes(memoryview(cache.serialize()))
        write_atomic(timing_cache_out, learned)
        cache_info["written_sha256"] = hashlib.sha256(learned).hexdigest()
        cache_info["written_bytes"] = len(learned)
    plan = bytes(memoryview(serialized))
    write_atomic(plan_path, plan)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    inspector = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    builder_info = {
        "tensorrt": trt.__version__,
        "policy": ENGINE_POLICY,
        "strongly_typed": True,
        "workspace_bytes": workspace_bytes,
        "tf32_flag": bool(config.get_flag(trt.BuilderFlag.TF32)),
        "timing_cache": cache_info,
    }
    del engine, runtime
    return BuiltEngine(
        plan_path, len(plan), hashlib.sha256(plan).hexdigest(), inspector, builder_info, time.perf_counter() - started
    )
