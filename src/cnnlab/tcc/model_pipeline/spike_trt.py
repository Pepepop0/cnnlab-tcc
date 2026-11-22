"""Viability spike (waves 1 and 2A): PyTorch -> ONNX -> TensorRT, batch 1, on the local GPU.

This is neither the evaluator nor a benchmark. It builds a tiny synthetic CNN
(stem, standard conv + maxpool, depthwise-separable residual block, GAP +
linear) in five precision layouts, and records for each stage whether it was
executed and what was observed. Inputs are seeded random tensors: no CIFAR-100
split, and never the official test set, is read. Timings and memory deltas are
smoke diagnostics of this toy network, never E/L/V/S of a candidate.

Run: ``python -m cnnlab.tcc.model_pipeline.spike_trt --out <dir outside repo>``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from cnnlab.tcc.model_pipeline.environment import probe_environment

StageStatus = Literal["ok", "failed", "blocked", "skipped"]
Exporter = Literal["torchscript", "dynamo"]

REGIONS: tuple[str, ...] = ("stem", "block0", "block1", "head")
VARIANTS: dict[str, dict[str, str]] = {
    "fp32": dict.fromkeys(REGIONS, "fp32"),
    "fp16": dict.fromkeys(REGIONS, "fp16"),
    "bf16": dict.fromkeys(REGIONS, "bf16"),
    "int8": dict.fromkeys(REGIONS, "int8"),
    "mixed": {"stem": "fp32", "block0": "fp16", "block1": "int8", "head": "bf16"},
}
EXPORTERS: tuple[Exporter, ...] = ("torchscript", "dynamo")
EXPORTER_DEPENDENCIES: dict[Exporter, tuple[str, ...]] = {
    "torchscript": ("onnx",),
    "dynamo": ("onnx", "onnxscript"),
}
PIPELINE_MODULES: tuple[str, ...] = ("onnx", "onnxscript", "tensorrt")
IMPLEMENTED_STAGES: tuple[str, ...] = (
    "gpu_guard",
    "environment",
    "dependency_probe",
    "torch_build",
    "torch_forward",
    "onnx_export[torchscript]",
    "onnx_export[dynamo]",
    "onnx_inspect[torchscript]",
    "onnx_inspect[dynamo]",
    *(
        f"{stage}[{exporter}]"
        for exporter in ("torchscript", "dynamo")
        for stage in ("trt_build", "trt_inspect", "trt_infer", "trt_timing_smoke")
    ),
)
TRT_STAGES: tuple[str, ...] = ("trt_build", "trt_inspect", "trt_infer", "trt_timing_smoke")
_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
NUM_CLASSES = 100
INPUT_SHAPE = (1, 3, 32, 32)
OPSET = 17
WARMUP, TIMED = 20, 100  # smoke only; the protocol's 200/1000 belongs to the evaluator


@dataclass(frozen=True)
class StageRecord:
    """What one stage did. ``blocked``/``skipped`` stages never ran their payload."""

    stage: str
    variant: str
    status: StageStatus
    command: str
    detail: str = ""
    observations: Mapping[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        if self.status in ("failed", "blocked", "skipped") and not self.detail:
            raise ValueError(f"{self.stage}: status {self.status} needs a detail")


@dataclass(frozen=True)
class SpikeReport:
    started_utc: str
    finished_utc: str
    device: str
    seed: int
    implemented_stages: tuple[str, ...]
    environment: Mapping[str, Any]
    stages: tuple[StageRecord, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)


# --- tiny synthetic network --------------------------------------------------


class _ConvBnAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, groups: int = 1, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, groups=groups, bias=False)
        self.bn: nn.Module = nn.BatchNorm2d(cout)
        self.act: nn.Module = nn.ReLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def fold_bn(self) -> None:
        if isinstance(self.bn, nn.BatchNorm2d):
            self.conv = fuse_conv_bn_eval(self.conv, self.bn)
            self.bn = nn.Identity()


class _Block0(nn.Module):
    """Standard 3x3 conv followed by a 2x2 maxpool reduction."""

    def __init__(self) -> None:
        super().__init__()
        self.body = _ConvBnAct(16, 32, 3)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.body(x))


class _Block1(nn.Module):
    """Depthwise-separable 3x3 with an identity residual."""

    def __init__(self) -> None:
        super().__init__()
        self.dw = _ConvBnAct(32, 32, 3, groups=32)
        self.pw = _ConvBnAct(32, 32, 1, act=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.pw(self.dw(x)))


class _Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(32, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class _FloatNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = _ConvBnAct(3, 16, 3)
        self.block0 = _Block0()
        self.block1 = _Block1()
        self.head = _Head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name in REGIONS:
            x = getattr(self, name)(x)
        return x


class _FakeQuantConv2d(nn.Module):
    """Symmetric INT8 Q/DQ: per-tensor activations, per-output-channel weights."""

    def __init__(self, conv: nn.Conv2d, input_amax: float) -> None:
        super().__init__()
        self.conv = conv
        w_amax = conv.weight.detach().abs().amax(dim=(1, 2, 3)).clamp_min(1e-8)
        self.register_buffer("w_scale", (w_amax / 127.0).float())
        self.register_buffer("w_zero", torch.zeros(w_amax.numel(), dtype=torch.int32))
        self.x_scale = max(float(input_amax), 1e-8) / 127.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fake_quantize_per_tensor_affine(x, self.x_scale, 0, -128, 127)
        w = torch.fake_quantize_per_channel_affine(self.conv.weight, self.w_scale, self.w_zero, 0, -128, 127)
        c = self.conv
        return F.conv2d(x, w, c.bias, c.stride, c.padding, c.dilation, c.groups)


class _FakeQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, input_amax: float) -> None:
        super().__init__()
        self.linear = linear
        w_amax = linear.weight.detach().abs().amax(dim=1).clamp_min(1e-8)
        self.register_buffer("w_scale", (w_amax / 127.0).float())
        self.register_buffer("w_zero", torch.zeros(w_amax.numel(), dtype=torch.int32))
        self.x_scale = max(float(input_amax), 1e-8) / 127.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fake_quantize_per_tensor_affine(x, self.x_scale, 0, -128, 127)
        w = torch.fake_quantize_per_channel_affine(self.linear.weight, self.w_scale, self.w_zero, 0, -128, 127)
        return F.linear(x, w, self.linear.bias)


class _Region(nn.Module):
    """Runs ``inner`` in one precision with FP32 at both boundaries (explicit casts in ONNX)."""

    def __init__(self, inner: nn.Module, precision: str) -> None:
        super().__init__()
        self.precision = precision
        self.dtype = _DTYPES.get(precision)
        self.inner = inner.to(self.dtype) if self.dtype is not None else inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dtype is None:
            return self.inner(x)
        return self.inner(x.to(self.dtype)).to(torch.float32)


class SpikeNet(nn.Module):
    def __init__(self, regions: Mapping[str, _Region]) -> None:
        super().__init__()
        self.regions = nn.ModuleDict(dict(regions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name in REGIONS:
            x = self.regions[name](x)
        return x


def synthetic_inputs(seed: int, n: int) -> torch.Tensor:
    """Seeded N(0,1) images; stands in for data without touching any split."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((n, *INPUT_SHAPE[1:]), generator=generator)


def _float_net(seed: int) -> _FloatNet:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = _FloatNet()
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):  # non-trivial BN so folding is observable
                n = module.num_features
                module.running_mean.copy_(0.1 * torch.randn(n))
                module.running_var.copy_(0.5 + torch.rand(n))
                module.weight.data.copy_(1 + 0.1 * torch.randn(n))
                module.bias.data.copy_(0.1 * torch.randn(n))
    return net.eval()


def _input_amax(net: nn.Module, calibration: torch.Tensor) -> dict[str, float]:
    amax: dict[str, float] = {}
    hooks = []
    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):

            def hook(_m: nn.Module, args: tuple[torch.Tensor, ...], key: str = name) -> None:
                amax[key] = max(amax.get(key, 0.0), float(args[0].detach().abs().max()))

            hooks.append(module.register_forward_pre_hook(hook))
    with torch.no_grad():
        net(calibration)
    for h in hooks:
        h.remove()
    return amax


def _quantize(region: nn.Module, amax: Mapping[str, float], prefix: str) -> nn.Module:
    for module in region.modules():
        if isinstance(module, _ConvBnAct):
            module.fold_bn()
    for name, child in list(region.named_children()):
        path = f"{prefix}.{name}"
        if isinstance(child, nn.Conv2d):
            setattr(region, name, _FakeQuantConv2d(child, amax[path]))
        elif isinstance(child, nn.Linear):
            setattr(region, name, _FakeQuantLinear(child, amax[path]))
        else:
            _quantize(child, amax, path)
    return region


def build_variant(variant: str, seed: int) -> SpikeNet:
    """Same float weights for every variant; only the representation changes."""
    layout = VARIANTS[variant]
    float_net = _float_net(seed)
    amax = _input_amax(float_net, synthetic_inputs(seed + 1, 32)) if "int8" in layout.values() else {}
    regions: dict[str, _Region] = {}
    for name in REGIONS:
        inner = copy.deepcopy(getattr(float_net, name))
        if layout[name] == "int8":
            inner = _quantize(inner, amax, name)
        regions[name] = _Region(inner, layout[name])
    return SpikeNet(regions).eval()


# --- stage helpers -------------------------------------------------------------


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def campaign_gpu_processes(apps: Sequence[Mapping[str, str]], own_pid: int) -> list[Mapping[str, str]]:
    """Python processes holding the GPU, other than this one.

    Under Windows WDDM ``--query-compute-apps`` also lists desktop programs
    (explorer, browsers) with graphics contexts; those do not count as a job.
    """
    return [app for app in apps if "python" in app.get("process_name", "").lower() and app.get("pid") != str(own_pid)]


def _missing(deps: Sequence[str], available: Callable[[str], bool]) -> list[str]:
    return [d for d in deps if not available(d)]


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:2000]


_TRACE_FILE = re.compile(r'File "([^"]*[\\/])?([^"\\/]+)"')


def _trace_tail(limit: int = 1500) -> str:
    """Current traceback with file paths reduced to base names (reports carry no user paths)."""
    return _TRACE_FILE.sub(r'File "\2"', traceback.format_exc())[-limit:]


def _utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def compare_logits(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    c, r = candidate.float().cpu(), reference.float().cpu()
    return {
        "max_abs_diff": float((c - r).abs().max()),
        "top1_agreement": float((c.argmax(1) == r.argmax(1)).float().mean()),
        "all_finite": bool(torch.isfinite(c).all()),
    }


def probe_imports(modules: Sequence[str] = PIPELINE_MODULES) -> StageRecord:
    """Import each pipeline module and keep the exact error of those that fail."""
    started = time.perf_counter()
    results: dict[str, Any] = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
            results[name] = {"importable": True, "version": getattr(module, "__version__", None)}
        except Exception as exc:  # broken native libs raise OSError, not ImportError
            results[name] = {"importable": False, "error": _error(exc)}
    missing = [name for name, r in results.items() if not r["importable"]]
    return StageRecord(
        "dependency_probe",
        "-",
        "blocked" if missing else "ok",
        "; ".join(f"import {name}" for name in modules),
        f"not importable: {missing}" if missing else "",
        results,
        time.perf_counter() - started,
    )


def export_onnx(
    model: nn.Module,
    example: torch.Tensor,
    path: Path,
    exporter: Exporter,
    variant: str,
    available: Callable[[str], bool] = module_available,
) -> StageRecord:
    """Try one exporter; record the exact error if it cannot run."""
    stage = f"onnx_export[{exporter}]"
    kwargs: dict[str, Any] = {"input_names": ["input"], "output_names": ["logits"], "dynamo": exporter == "dynamo"}
    if exporter == "torchscript":
        kwargs["opset_version"] = OPSET
    command = f"torch.onnx.export(model, (x,), '{path.name}', {', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"
    missing = _missing(EXPORTER_DEPENDENCIES[exporter], available)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with torch.no_grad():
                torch.onnx.export(model, (example,), str(path), **kwargs)
        except Exception as exc:  # the exporter raises several unrelated types
            status: StageStatus = "blocked" if missing else "failed"
            return StageRecord(
                stage,
                variant,
                status,
                command,
                _error(exc),
                {
                    "missing_dependencies": missing,
                    "stdout_encoding": getattr(sys.stdout, "encoding", None),
                    "warnings": [str(w.message)[:300] for w in caught],
                },
                time.perf_counter() - started,
            )
    # The dynamo exporter writes weights to a sidecar ``<name>.onnx.data``;
    # the graph file alone does not identify the model.
    sidecars = sorted(p for p in path.parent.glob(path.name + ".*") if p.suffix != ".tmp")
    return StageRecord(
        stage,
        variant,
        "ok",
        command,
        "",
        {
            "onnx_file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "external_data": {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in sidecars},
            "total_bytes": path.stat().st_size + sum(p.stat().st_size for p in sidecars),
            "stdout_encoding": getattr(sys.stdout, "encoding", None),
            "warnings": [str(w.message)[:300] for w in caught],
        },
        time.perf_counter() - started,
    )


def inspect_onnx(path: Path, variant: str, exporter: Exporter = "torchscript") -> StageRecord:
    stage = f"onnx_inspect[{exporter}]"
    command = f"onnx.checker.check_model(onnx.load('{path.name}'))"
    started = time.perf_counter()
    try:
        onnx = importlib.import_module("onnx")
    except ImportError as exc:
        return StageRecord(stage, variant, "blocked", command, _error(exc))
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        ops = Counter(node.op_type for node in model.graph.node)
        cast_to = Counter(
            onnx.TensorProto.DataType.Name(attr.i)
            for node in model.graph.node
            if node.op_type == "Cast"
            for attr in node.attribute
            if attr.name == "to"
        )
        initializers = Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in model.graph.initializer)
        observations = {
            "opset": [f"{o.domain or 'ai.onnx'}:{o.version}" for o in model.opset_import],
            "op_types": dict(ops),
            "cast_targets": dict(cast_to),
            "initializer_dtypes": dict(initializers),
            "batchnorm_nodes": ops.get("BatchNormalization", 0),
        }
    except Exception as exc:
        return StageRecord(stage, variant, "failed", command, _error(exc), {}, time.perf_counter() - started)
    return StageRecord(stage, variant, "ok", command, "", observations, time.perf_counter() - started)


def summarize_inspector(info: Mapping[str, Any]) -> dict[str, Any]:
    """Per-layer output datatype and tactic from the TensorRT JSON inspector.

    TRT 11.3 reports ``Datatype`` and ``Format`` as separate keys on each
    tensor; the wave-1 code assumed a combined ``Format/Datatype`` key and read
    only ``"?"`` (observed in run 1 of wave 2A). Both spellings are accepted.
    """
    layers = [layer for layer in info.get("Layers", []) if isinstance(layer, dict)]
    per_layer = [
        {
            "name": str(layer.get("Name", "?")),
            "type": str(layer.get("LayerType", "?")),
            "out_dtypes": [str(t.get("Datatype") or t.get("Format/Datatype") or "?") for t in layer.get("Outputs", [])],
            "tactic": str(layer.get("TacticName", "")),
        }
        for layer in layers
    ]
    return {
        "layers": len(per_layer),
        "layer_types": dict(Counter(p["type"] for p in per_layer)),
        "output_dtypes": dict(Counter(d for p in per_layer for d in p["out_dtypes"])),
        "unfused_scale_layers": sum(1 for p in per_layer if p["type"].lower() == "scale"),
        "per_layer": per_layer,
    }


# --- TensorRT stages (implemented; executed only when tensorrt is importable) ---


def trt_stages(
    onnx_path: Path,
    out_dir: Path,
    variant: str,
    inputs: torch.Tensor,
    torch_variant: torch.Tensor,
    torch_fp32: torch.Tensor,
    exporter: Exporter = "torchscript",
) -> list[StageRecord]:
    """Build, inspect, run and time one engine. Requires CUDA and ``tensorrt``."""
    names = tuple(f"{n}[{exporter}]" for n in TRT_STAGES)
    try:
        trt = importlib.import_module("tensorrt")
    except ImportError as exc:
        detail = _error(exc)
        return [StageRecord(names[0], variant, "blocked", "import tensorrt", detail)] + [
            StageRecord(n, variant, "skipped", "-", "trt_build blocked") for n in names[1:]
        ]
    records: list[StageRecord] = []
    logger = trt.Logger(trt.Logger.WARNING)

    # build
    started = time.perf_counter()
    flag = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    command = (
        f"trt.Builder.create_network({'STRONGLY_TYPED' if flag is not None else 'default'}); "
        f"OnnxParser.parse_from_file('{onnx_path.name}'); build_serialized_network(DETAILED)"
    )
    plan_path = out_dir / f"{variant}.{exporter}.plan"
    try:
        builder = trt.Builder(logger)
        network = builder.create_network((1 << int(flag)) if flag is not None else 0)
        parser = trt.OnnxParser(network, logger)
        parsed = parser.parse_from_file(str(onnx_path))
        parse_errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        if not parsed:
            raise RuntimeError(f"ONNX parse failed: {parse_errors}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("build_serialized_network returned None")
        plan = bytes(memoryview(serialized))
        _write_atomic(plan_path, plan)
    except Exception as exc:
        records.append(
            StageRecord(
                names[0],
                variant,
                "failed",
                command,
                _error(exc),
                {"trace": _trace_tail()},
                time.perf_counter() - started,
            )
        )
        return records + [StageRecord(n, variant, "skipped", "-", "trt_build failed") for n in names[1:]]
    records.append(
        StageRecord(
            names[0],
            variant,
            "ok",
            command,
            "",
            {
                "tensorrt": trt.__version__,
                "strongly_typed_flag": flag is not None,
                "plan_file": plan_path.name,
                "plan_bytes": len(plan),
                "plan_sha256": hashlib.sha256(plan).hexdigest(),
                "parse_warnings": parse_errors,
            },
            time.perf_counter() - started,
        )
    )

    # deserialize + inspect
    started = time.perf_counter()
    device = torch.device("cuda")
    try:
        torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0]
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        torch.cuda.synchronize()
        free_after = torch.cuda.mem_get_info()[0]
        inspector = engine.create_engine_inspector()
        info_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        (out_dir / f"{variant}.{exporter}.inspector.json").write_text(info_text, encoding="utf-8")
        summary = summarize_inspector(json.loads(info_text))
        device_memory = getattr(engine, "device_memory_size_v2", None) or getattr(engine, "device_memory_size", None)
    except Exception as exc:
        records.append(
            StageRecord(
                names[1],
                variant,
                "failed",
                "Runtime.deserialize_cuda_engine + EngineInspector",
                _error(exc),
                {},
                time.perf_counter() - started,
            )
        )
        return records + [StageRecord(n, variant, "skipped", "-", "trt_inspect failed") for n in names[2:]]
    records.append(
        StageRecord(
            names[1],
            variant,
            "ok",
            "Runtime.deserialize_cuda_engine + EngineInspector(JSON)",
            "",
            {
                **summary,
                "engine_device_memory_bytes": device_memory,
                "free_mem_delta_bytes_diagnostic": free_before - free_after,
            },
            time.perf_counter() - started,
        )
    )

    # batch-1 inference and parity
    started = time.perf_counter()
    try:
        io = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        in_name = next(n for n in io if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        out_name = next(n for n in io if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        x_buf = torch.empty(INPUT_SHAPE, device=device, dtype=torch.float32)
        y_buf = torch.empty(tuple(context.get_tensor_shape(out_name)), device=device, dtype=torch.float32)
        context.set_tensor_address(in_name, x_buf.data_ptr())
        context.set_tensor_address(out_name, y_buf.data_ptr())
        # A dedicated stream: TRT warned in run 1 that the default stream forces
        # extra cudaStreamSynchronize calls inside enqueueV3.
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        outputs = []
        for x in inputs:
            with torch.cuda.stream(stream):
                x_buf.copy_(x.unsqueeze(0))
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("execute_async_v3 returned False")
            stream.synchronize()
            outputs.append(y_buf.clone())
        trt_out = torch.cat(outputs)
    except Exception as exc:
        records.append(
            StageRecord(
                names[2], variant, "failed", "execute_async_v3 batch 1", _error(exc), {}, time.perf_counter() - started
            )
        )
        return records + [StageRecord(names[3], variant, "skipped", "-", "trt_infer failed")]
    records.append(
        StageRecord(
            names[2],
            variant,
            "ok",
            f"execute_async_v3 x{len(inputs)} (batch 1, FP32 I/O, non-default stream)",
            "",
            {
                "vs_torch_same_variant": compare_logits(trt_out, torch_variant),
                "vs_torch_fp32": compare_logits(trt_out, torch_fp32),
            },
            time.perf_counter() - started,
        )
    )

    # timing smoke (not L: same process, no isolation, no sentinel)
    started = time.perf_counter()
    try:
        for _ in range(WARMUP):
            context.execute_async_v3(stream.cuda_stream)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(TIMED):
            context.execute_async_v3(stream.cuda_stream)
        end.record(stream)
        end.synchronize()
        mean_ms = start.elapsed_time(end) / TIMED
    except Exception as exc:
        records.append(
            StageRecord(names[3], variant, "failed", "cuda events", _error(exc), {}, time.perf_counter() - started)
        )
        return records
    records.append(
        StageRecord(
            names[3],
            variant,
            "ok",
            f"{WARMUP} warmup + {TIMED} timed execute_async_v3, CUDA events",
            "diagnostic only; not the protocol's L",
            {"mean_ms_not_fitness": mean_ms},
            time.perf_counter() - started,
        )
    )
    return records


# --- orchestration -------------------------------------------------------------


def run_spike(
    out_dir: Path,
    variants: Sequence[str] = tuple(VARIANTS),
    device: str = "cuda",
    seed: int = 20260927,
    allow_busy_gpu: bool = False,
    available: Callable[[str], bool] = module_available,
) -> SpikeReport:
    """Run every stage that can run here and report the rest as blocked/skipped."""
    started_utc = _utc()
    out_dir.mkdir(parents=True, exist_ok=True)
    environment = probe_environment(touch_cuda=device == "cuda")
    stages: list[StageRecord] = []

    def finish() -> SpikeReport:
        report = SpikeReport(started_utc, _utc(), device, seed, IMPLEMENTED_STAGES, environment, tuple(stages))
        _write_atomic(out_dir / "spike_report.json", report.to_json().encode("utf-8"))
        return report

    apps = environment["nvidia_smi"].get("compute_apps", [])
    busy = campaign_gpu_processes(apps, os.getpid())
    guard_obs = {"python_gpu_processes": busy, "other_gpu_processes": len(apps) - len(busy)}
    if device == "cuda" and busy and not allow_busy_gpu:
        stages.append(
            StageRecord("gpu_guard", "-", "failed", "nvidia-smi --query-compute-apps", f"GPU busy: {busy}", guard_obs)
        )
        return finish()
    stages.append(StageRecord("gpu_guard", "-", "ok", "nvidia-smi --query-compute-apps", "", guard_obs))
    if device == "cuda" and not torch.cuda.is_available():
        stages.append(StageRecord("environment", "-", "failed", "torch.cuda.is_available()", "CUDA unavailable"))
        return finish()

    deps = probe_imports()
    stages.append(deps)
    trt_note = "" if deps.observations["tensorrt"]["importable"] else "; tensorrt not importable (dependency_probe)"
    dev = torch.device(device)
    inputs = synthetic_inputs(seed + 2, 8).to(dev)
    reference: torch.Tensor | None = None
    for variant in variants:
        t0 = time.perf_counter()
        try:
            model = build_variant(variant, seed).to(dev)
        except Exception as exc:
            stages.append(
                StageRecord(
                    "torch_build",
                    variant,
                    "failed",
                    f"build_variant({variant!r})",
                    _error(exc),
                    {},
                    time.perf_counter() - t0,
                )
            )
            continue
        stages.append(
            StageRecord(
                "torch_build",
                variant,
                "ok",
                f"build_variant({variant!r})",
                "",
                {"layout": VARIANTS[variant]},
                time.perf_counter() - t0,
            )
        )

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                logits = torch.cat([model(x.unsqueeze(0)) for x in inputs])
            if device == "cuda":
                torch.cuda.synchronize()
        except Exception as exc:
            stages.append(
                StageRecord(
                    "torch_forward", variant, "failed", "model(x) batch 1", _error(exc), {}, time.perf_counter() - t0
                )
            )
            continue
        if reference is None and variant == "fp32":
            reference = logits
        obs: dict[str, Any] = {"shape": list(logits.shape), "all_finite": bool(torch.isfinite(logits).all())}
        if reference is not None:
            obs["vs_torch_fp32"] = compare_logits(logits, reference)
        stages.append(
            StageRecord(
                "torch_forward",
                variant,
                "ok",
                f"model(x) batch 1 x{len(inputs)} on {device}",
                "",
                obs,
                time.perf_counter() - t0,
            )
        )

        exported: dict[Exporter, Path] = {}
        for exporter in EXPORTERS:
            path = out_dir / f"{variant}.{exporter}.onnx"
            record = export_onnx(model, inputs[:1], path, exporter, variant, available)
            stages.append(record)
            if record.status == "ok":
                exported[exporter] = path
        # Each exporter gets its own engine, so the TorchScript (being removed
        # from torch) and dynamo paths are compared on the same weights.
        for exporter in EXPORTERS:
            trt_names = [f"{n}[{exporter}]" for n in TRT_STAGES]
            if exporter not in exported:
                reason = f"onnx_export[{exporter}] produced no file"
                stages.append(StageRecord(f"onnx_inspect[{exporter}]", variant, "skipped", "-", reason))
                stages += [StageRecord(n, variant, "skipped", "-", reason + trt_note) for n in trt_names]
                continue
            stages.append(inspect_onnx(exported[exporter], variant, exporter))
            if device != "cuda":
                stages += [StageRecord(n, variant, "skipped", "-", "TensorRT needs --device cuda") for n in trt_names]
                continue
            stages += trt_stages(
                exported[exporter],
                out_dir,
                variant,
                inputs,
                logits,
                reference if reference is not None else logits,
                exporter,
            )
    return finish()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="artifact directory, outside the repository")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args(argv)
    # The dynamo exporter prints non-ASCII status marks; on a cp1252 console it
    # raised UnicodeEncodeError in run 1 of wave 2A. Reconfigure only the CLI.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    report = run_spike(args.out, args.variants, args.device, args.seed, args.allow_busy_gpu)
    for record in report.stages:
        print(f"{record.variant:>6} {record.stage:<32} {record.status:<8} {record.detail[:110]}")
    print(f"report: {args.out / 'spike_report.json'}")
    return 2 if report.stages and report.stages[0].stage == "gpu_guard" and report.stages[0].status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
