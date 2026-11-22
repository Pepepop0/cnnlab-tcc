"""Isolated benchmark of one TensorRT plan: L, V and E in a fresh process (TCC-01 section 3).

Run as ``python -m cnnlab.tcc.model_pipeline.bench --plan P --val V.npz --out R.json``.
The parent (training) process calls it through :func:`run_isolated` after
releasing the model; this process owns its own CUDA context, so training
state cannot inflate V.

* **V**: every device allocation TensorRT makes (engine weights, execution
  context, activations/workspace) is routed through an ``IGpuAllocator`` that
  uses the PyTorch caching allocator; the I/O buffers are PyTorch tensors.
  ``V = (max_memory_allocated - baseline) / 1e6`` after load, warm-up and the
  timed loop, with the CUDA context already initialized in the baseline.
  The drop in ``cudaMemGetInfo`` free memory is recorded next to it as an
  audit value (it also sees driver-side allocations such as kernel images,
  but moves in 2 MiB pages).
* **L**: median of per-inference GPU times from CUDA events, batch 1, FP32
  input already on the GPU, no synchronization between inferences, after a
  warm-up. p5/p95/IQR and the raw samples are kept.
* **E**: top-1 error of the engine itself, batch 1, over the validation
  tensors, measured after L and V so the validation data never counts in V.
  The engine's logits are compared, image by image, with the PyTorch FP32
  logits of the same trained weights (parity diagnostic).

These are **preliminary pilot measurements**: no sentinel, no drift check,
no clock/temperature control yet. They are not confirmatory values.

Process protocol. The child writes ``<out>.started`` before importing CUDA
libraries, then either the report (exit 0) or ``{"error": ...}`` (exit 1).
:func:`run_isolated` uses that to tell apart a child that never started
(infrastructure: the candidate was not measured at all) from one that failed
or crashed while measuring this engine. The child is created with its own
hidden console: a parent whose console host died (observed on 28/09/2026,
when the launching session ended mid-run) can no longer start console
children attached to it, and every one exits ``0xC0000142``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Literal

BENCH_VERSION = "tcc-bench/preliminary-v4"  # v4: the engine runs as a CUDA Graph (v3: top-5, macro P/R/F1)
# The proposal measures L on the engine captured in a CUDA Graph. Without it, on this Windows
# driver about two thirds of the sentinel's 0.2 ms were host-side launch overhead and the
# session medians varied by 4-12%; replaying a graph gives the GPU time with no such noise
# (TCC-01 section 3; docs/piloto/cuda-graph-2026-10-06.md).
BENCH_CUDA_GRAPH = True
GRAPH_WARMUP = 50
DEFAULT_WARMUP = 200
DEFAULT_ITERS = 1000

BenchFailure = Literal["spawn", "crash", "timeout", "measurement"]


class BenchProcessError(RuntimeError):
    """The isolated benchmark produced no report.

    ``kind``: ``spawn`` (the child never reached Python code; nothing about
    the candidate was observed), ``crash`` (it started and died without a
    report), ``timeout``, or ``measurement`` (it raised while measuring).
    """

    def __init__(self, kind: BenchFailure, exit_code: int | None, detail: str) -> None:
        code = "none" if exit_code is None else f"{exit_code} (0x{exit_code & 0xFFFFFFFF:08X})"
        super().__init__(f"bench {kind}: exit {code}: {detail}"[:4000])
        self.kind: BenchFailure = kind
        self.exit_code = exit_code
        self.detail = detail[:4000]


def child_process_kwargs() -> dict[str, Any]:
    """Spawn options that do not depend on the parent's console being alive."""
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def started_marker(out_path: Path) -> Path:
    return out_path.with_name(out_path.name + ".started")


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def latency_stats(samples_ms: list[float]) -> dict[str, float]:
    s = sorted(samples_ms)
    return {
        "median_ms": statistics.median(s),
        "p5_ms": _quantile(s, 0.05),
        "p95_ms": _quantile(s, 0.95),
        "iqr_ms": _quantile(s, 0.75) - _quantile(s, 0.25),
        "mean_ms": statistics.fmean(s),
        "n": float(len(s)),
    }


def classification_summary(preds: Any, labels: Any, in_top5: Any, num_classes: int) -> dict[str, Any]:
    """Auxiliary metrics of the engine on the validation split; none of them enters the fitness.

    Macro averages weigh every class equally. A class the engine never predicts
    has precision 0 (no true positive), which is the conservative reading.
    """
    import torch

    counts = torch.bincount(labels * num_classes + preds, minlength=num_classes * num_classes)
    confusion = counts.reshape(num_classes, num_classes).double()
    tp = confusion.diag()
    predicted, actual = confusion.sum(0), confusion.sum(1)
    precision = torch.where(predicted > 0, tp / predicted.clamp(min=1), torch.zeros_like(tp))
    recall = torch.where(actual > 0, tp / actual.clamp(min=1), torch.zeros_like(tp))
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2 * precision * recall / denominator.clamp(min=1e-12), torch.zeros_like(tp))
    present = actual > 0
    n = int(labels.numel())
    return {
        "top1_accuracy": float((preds == labels).sum()) / n,
        "top5_accuracy": float(in_top5.sum()) / n,
        "macro_precision": float(precision[present].mean()),
        "macro_recall": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        # per class (index = class id), for the per-class metrics of the final evaluation
        "per_class": {
            "support": [int(v) for v in actual],
            "predicted": [int(v) for v in predicted],
            "true_positive": [int(v) for v in tp],
        },
    }


def _measure(plan_path: Path, val_path: Path | None, ref_path: Path | None, warmup: int, iters: int) -> dict[str, Any]:
    import numpy as np
    import tensorrt as trt
    import torch

    device = torch.device("cuda")
    torch.cuda.init()
    torch.zeros(1, device=device)  # create the context before the baseline
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    free_base, total = torch.cuda.mem_get_info()

    class TorchAllocator(trt.IGpuAllocator):
        """Routes TensorRT device memory through PyTorch so it is counted in its peak."""

        def __init__(self) -> None:
            trt.IGpuAllocator.__init__(self)
            self.live: dict[int, int] = {}
            self.peak_live = 0
            self.calls = 0

        def allocate(self, size: int, alignment: int, flags: int) -> int:  # noqa: ARG002
            if size == 0:
                return 0
            ptr = torch.cuda.caching_allocator_alloc(int(size))
            if alignment and ptr % alignment:
                torch.cuda.caching_allocator_delete(ptr)
                raise MemoryError(f"unaligned pointer for alignment {alignment}")
            self.live[ptr] = int(size)
            self.calls += 1
            self.peak_live = max(self.peak_live, sum(self.live.values()))
            return ptr

        def deallocate(self, memory: int) -> bool:
            ptr = int(memory)
            if ptr in self.live:
                del self.live[ptr]
                torch.cuda.caching_allocator_delete(ptr)
            return True

        def allocate_async(self, size: int, alignment: int, flags: int, stream: int) -> int:  # noqa: ARG002
            return self.allocate(size, alignment, flags)

        def deallocate_async(self, memory: int, stream: int) -> bool:  # noqa: ARG002
            return self.deallocate(memory)

    allocator = TorchAllocator()
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    runtime.gpu_allocator = allocator
    plan = plan_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("deserialize_cuda_engine returned None")
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    x_buf = torch.zeros(tuple(engine.get_tensor_shape(in_name)), device=device, dtype=torch.float32)
    y_buf = torch.zeros(tuple(context.get_tensor_shape(out_name)), device=device, dtype=torch.float32)
    context.set_tensor_address(in_name, x_buf.data_ptr())
    context.set_tensor_address(out_name, y_buf.data_ptr())
    stream = torch.cuda.Stream()
    handle = stream.cuda_stream

    val = np.load(val_path) if val_path is not None else None
    if val is not None:
        x_buf.copy_(torch.from_numpy(val["x"][:1]))  # a real image for timing; values barely matter
    torch.cuda.synchronize()

    for _ in range(warmup):
        context.execute_async_v3(handle)
    stream.synchronize()
    # One inference captured once (input and output addresses fixed), then replayed.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        context.execute_async_v3(handle)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with torch.cuda.stream(stream):
        for _ in range(GRAPH_WARMUP):
            graph.replay()
        stream.synchronize()
        wall = time.perf_counter()
        for i in range(iters):
            starts[i].record(stream)
            graph.replay()
            ends[i].record(stream)
    stream.synchronize()
    wall = time.perf_counter() - wall
    samples = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    free_after_run = torch.cuda.mem_get_info()[0]
    peak_alloc = torch.cuda.max_memory_allocated()
    v_bytes = peak_alloc - base_alloc
    memory = {
        "v_bytes_allocator_peak": int(v_bytes),
        "trt_allocator_peak_live_bytes": int(allocator.peak_live),
        "trt_allocator_calls": allocator.calls,
        "io_buffer_bytes": int(x_buf.numel() * 4 + y_buf.numel() * 4),
        "engine_device_memory_size_v2": int(engine.device_memory_size_v2),
        "free_mem_drop_bytes_audit": int(free_base - free_after_run),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "total_device_bytes": int(total),
    }

    quality: dict[str, Any] = {}
    e_result: dict[str, Any] | None = None
    if val is not None:
        xv = torch.from_numpy(val["x"]).to(device)
        yv = torch.from_numpy(val["y"]).to(device)
        n = xv.shape[0]
        preds = torch.empty(n, dtype=torch.int64, device=device)
        in_top5 = torch.zeros(n, dtype=torch.bool, device=device)
        k = min(5, y_buf.shape[-1])
        finite = torch.ones((), dtype=torch.bool, device=device)
        ref = torch.from_numpy(np.load(ref_path)).to(device) if ref_path is not None else None
        keep = n if ref is None else min(n, ref.shape[0])
        kept = torch.empty((keep, y_buf.shape[-1]), device=device)
        with torch.cuda.stream(stream):
            for i in range(n):
                x_buf.copy_(xv[i : i + 1])
                graph.replay()  # E comes from the same captured inference that was timed
                preds[i] = y_buf.argmax(1)[0]
                in_top5[i] = (y_buf[0].topk(k).indices == yv[i]).any()
                finite &= torch.isfinite(y_buf).all()
                if i < keep:
                    kept[i] = y_buf[0]
        stream.synchronize()
        errors = int((preds != yv).sum())
        e_result = {
            "e_fraction": errors / n,
            "errors": errors,
            "n": int(n),
            "all_finite": bool(finite),
            **classification_summary(preds, yv, in_top5, y_buf.shape[-1]),
        }
        if ref is not None:
            ref = ref[:keep]
            agree = kept.argmax(1) == ref.argmax(1)
            quality["parity_vs_torch_fp32"] = {
                "n": int(keep),
                "max_abs_diff": float((kept - ref).abs().max()),
                "mean_abs_diff": float((kept - ref).abs().mean()),
                "top1_agreement": float(agree.float().mean()),
                "top1_disagreements": int((~agree).sum()),
                "torch_errors": int((ref.argmax(1) != yv[:keep]).sum()),
            }
    return {
        "bench_version": BENCH_VERSION,
        "tensorrt": trt.__version__,
        "device": torch.cuda.get_device_name(0),
        "warmup": warmup,
        "iters": iters,
        "timed_loop_wall_s": wall,
        "latency": latency_stats(samples),
        "latency_samples_ms": samples,
        "memory": memory,
        "e": e_result,
        "quality": quality,
        "cuda_graph": BENCH_CUDA_GRAPH,
        "input": "fp32 1x3x32x32 resident on GPU",
        "preliminary": True,
    }


def run_isolated(
    plan_path: Path,
    out_path: Path,
    *,
    val_path: Path | None = None,
    ref_logits_path: Path | None = None,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    """Run this module in a fresh interpreter and return its report.

    Raises :class:`BenchProcessError` with the failure kind; never returns a
    partial or invented report.
    """
    marker = started_marker(out_path)
    for stale in (marker, out_path):
        stale.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "cnnlab.tcc.model_pipeline.bench",
        "--plan",
        str(plan_path),
        "--out",
        str(out_path),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    if val_path is not None:
        cmd += ["--val", str(val_path)]
    if ref_logits_path is not None:
        cmd += ["--ref", str(ref_logits_path)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            timeout=timeout_s,
            env=env,
            **child_process_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        kind: BenchFailure = "timeout" if marker.exists() else "spawn"
        raise BenchProcessError(kind, None, f"no result after {timeout_s:.0f} s") from exc
    except OSError as exc:
        raise BenchProcessError("spawn", None, f"{type(exc).__name__}: {exc}") from exc
    report: dict[str, Any] | None = None
    if out_path.exists():
        try:
            report = json.loads(out_path.read_text(encoding="utf-8"))
        except ValueError:
            report = None
    if proc.returncode == 0 and report is not None and "error" not in report:
        return report
    stderr = (proc.stderr or "")[-2000:]
    if report is not None and "error" in report:
        error = report["error"]
        detail = " | ".join(str(error.get(key, "")) for key in ("type", "message", "trace"))
        raise BenchProcessError("measurement", proc.returncode, detail)
    if not marker.exists():
        raise BenchProcessError("spawn", proc.returncode, stderr or "child never reached Python code")
    raise BenchProcessError("crash", proc.returncode, stderr or "child died without a report")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="isolated TensorRT benchmark (preliminary L/V/E)")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--ref", type=Path)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    args = parser.parse_args(argv)
    started_marker(args.out).write_text("started", encoding="ascii")  # before any CUDA library loads
    try:
        report = _measure(args.plan, args.val, args.ref, args.warmup, args.iters)
        code = 0
    except Exception as exc:  # reported to the parent, which decides what it means for the candidate
        report = {
            "error": {"type": type(exc).__name__, "message": str(exc)[:2000], "trace": traceback.format_exc()[-2000:]}
        }
        code = 1
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    os.replace(tmp, args.out)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
