"""Scientific evaluator of the pilot demo: train -> ONNX -> TensorRT -> isolated E, L, V, S.

Implements the ``Evaluator`` port of ``contracts.py`` for unpruned networks
with stem, blocks and head each in FP32, FP16, BF16 or INT8 (PTQ or QAT) (the representations the
pipeline can materialize today). Training is always FP32; the precision of
each block is applied to the trained weights before export, and the engine
inspector has to confirm it. Per request:

``parse expressed genome -> check representation -> build_model(train_seed)
-> train -> fold BN + dynamo ONNX -> strongly typed FP32 engine -> free the
training state -> isolated benchmark process (L, V, then E over the 5k
validation split) -> S from the plan bytes``.

No exception leaves :meth:`EngineEvaluator.evaluate` (operator interrupts
aside). Any failure of the candidate becomes ``status="invalid"`` with
``failure_stage``/``code`` and the time already spent; no objective value is
ever invented. An exception nobody anticipated is classified after a health
check of the device: out of memory and failures on a healthy device belong
to the candidate (``invalid``), an unhealthy device or an I/O error is
infrastructure (``interrupted``). A failure of the *infrastructure* (the benchmark process could
not even start) says nothing about the candidate: after bounded retries the
attempt is returned as ``status="interrupted"`` so the search can retry it
with a new attempt instead of ranking the candidate as invalid. Valid results
carry the plan, ONNX, benchmark report and training log as artifacts,
published atomically under ``runs/<run_id>/attempts/<attempt_id>/`` of the
store's artifact root, and label every number as preliminary.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from cnnlab.tcc.contracts import EvaluationRequest, EvaluationResult, Fitness
from cnnlab.tcc.model_pipeline.bench import (
    BENCH_CUDA_GRAPH,
    BENCH_VERSION,
    DEFAULT_ITERS,
    DEFAULT_WARMUP,
    BenchProcessError,
    run_isolated,
)
from cnnlab.tcc.model_pipeline.builder import (
    BUILDER_VERSION,
    ModelBuildError,
    build_model,
    count_macs,
    count_parameters,
    effective_widths,
    fold_batchnorm,
)
from cnnlab.tcc.model_pipeline.demo_data import DemoData, write_val_file
from cnnlab.tcc.model_pipeline.engine import (
    ENGINE_POLICY,
    EngineBuildError,
    ExportError,
    build_engine,
    export_network,
    sha256_file,
    summarize_layers,
    write_atomic,
)
from cnnlab.tcc.model_pipeline.precision import (
    PRECISION_POLICY,
    check_engine_precision,
    region_precisions,
    representation_problems,
    typed_network,
)
from cnnlab.tcc.model_pipeline.pruning import (
    IMPORTANCE,
    PRUNING_VERSION,
    PruningError,
    finetune_config,
    prune_network,
)
from cnnlab.tcc.model_pipeline.quantization import (
    N_CALIBRATION,
    QAT_INIT_CALIBRATION,
    QAT_RANGE_MOMENTUM,
    QUANTIZATION_POLICY,
    QDQLayer,
    QuantizationError,
    calibration_indices,
    insert_qdq,
    int8_regions,
    plan_quantization,
)
from cnnlab.tcc.model_pipeline.trainer import (
    TrainConfig,
    TrainingDiverged,
    evaluate_accuracy,
    run_in_batches,
    train_model,
)
from cnnlab.tcc.science.genome import ExpressedGenome, GenomeError
from cnnlab.tcc.science.record import CacheContext

EVALUATOR_VERSION = "tcc-evaluator/pilot-pruned-quantized-v8"


def _now() -> float:
    """Clock of the training time limit (a module function so a test can replace it)."""
    return time.monotonic()


BYTES_PER_MB = 1_000_000  # decimal MB, TCC-01 section 3
PRUNING_DEFINITION: dict[str, Any] = {
    "version": PRUNING_VERSION,
    "importance": IMPORTANCE,
    "fine_tune": "1 epoch after pruning, no warm-up, the genome's lr and weight decay",
    "seed": "train seed + 2",
    "order": "train, prune, fine-tune, then BN folding, quantization (and QAT), export",
}


INFRASTRUCTURE_PREFIX = "infrastructure_"


class Invalid(Exception):
    """Candidate-level failure mapped to an invalid attempt."""

    def __init__(self, stage: str, code: str, detail: str) -> None:
        super().__init__(f"{stage}/{code}: {detail}")
        self.stage, self.code, self.detail = stage, code, detail[:4000]


class Interrupted(Exception):
    """Infrastructure failure: the candidate was not observed; mapped to an interrupted attempt."""

    def __init__(self, stage: str, code: str, detail: str) -> None:
        super().__init__(f"{stage}/{code}: {detail}")
        self.stage, self.code, self.detail = stage, INFRASTRUCTURE_PREFIX + code, detail[:4000]


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    bench_warmup: int = DEFAULT_WARMUP
    bench_iters: int = DEFAULT_ITERS
    bench_timeout_s: float = 900.0
    bench_spawn_retries: int = 2  # extra tries when the child never started
    bench_crash_retries: int = 1  # extra tries when the child died without a report
    bench_retry_wait_s: float = 2.0
    bench_shared_retries: int = 2  # extra tries when another GPU job appeared during the measurement
    bench_measurement_retries: int = 1  # extra tries when the child raised while measuring
    gpu_wait_s: float = 180.0  # how long to wait for an exclusive GPU before giving the attempt up
    gpu_poll_s: float = 5.0
    keep_onnx: bool = True
    # Free VRAM left to the desktop when the session starts. PyTorch may not allocate beyond
    # the rest, so a network that does not fit fails with out-of-memory (invalid, as the
    # protocol says) instead of spilling into system memory, which the Windows driver
    # allows and which makes training tens of times slower.
    vram_margin_mb: int = 1024
    # INT8 PTQ: how many training examples calibrate the activation ranges (TCC-02: 1,024, fixed).
    calibration_examples: int = N_CALIBRATION
    # Longest training an attempt may take, in seconds; None: no limit. Another program taking VRAM mid-run makes
    # the driver spill into system memory and training tens of times slower, which the VRAM cap does not catch.
    # The value comes from the pilot (p99 of the training time x 3) and is recorded in the campaign plan.
    train_time_limit_s: float | None = None

    def definition(self) -> dict[str, Any]:
        return {
            "evaluator": EVALUATOR_VERSION,
            "train": self.train.definition(),
            "bench": {
                "version": BENCH_VERSION,
                "warmup": self.bench_warmup,
                "iters": self.bench_iters,
                "cuda_graph": BENCH_CUDA_GRAPH,
                "batch": 1,
                "input": "fp32",
                "exclusive_gpu": "checked before and after every measurement",
            },
            "engine_policy": ENGINE_POLICY,
            "builder": BUILDER_VERSION,
            "precision_policy": PRECISION_POLICY,
            "representation": "stem, blocks and head fp32, fp16, bf16 or int8 (ptq or qat; bf16 except where the "
            "engine cannot run it); physical structured pruning per block",
            "pruning": PRUNING_DEFINITION,
            "quantization": {
                "policy": QUANTIZATION_POLICY,
                "calibration_examples": self.calibration_examples,
                "calibration_set": "class-balanced training examples, seed 0",
                "qat": {
                    "initial_ranges": QAT_INIT_CALIBRATION,
                    "range_momentum": QAT_RANGE_MOMENTUM,
                    "fine_tune": "1 epoch, no warm-up, the genome's lr and weight decay, after BN folding",
                    "seed": "train seed + 1",
                },
            },
            "train_time_limit_s": self.train_time_limit_s,
            "memory_cap": {"policy": "free VRAM at session start minus margin", "margin_mb": self.vram_margin_mb},
            "gpu_seconds": "wall time of the attempt minus GPU waits, session setup and training pauses",
        }


def backend_manifest() -> dict[str, Any]:
    """Versions that change what the engine is; part of the cache context.

    GPU name and driver come from ``nvidia-smi``, so building the manifest
    creates no CUDA context.
    """
    import tensorrt as trt

    from cnnlab.tcc.model_pipeline.environment import nvidia_smi_snapshot

    gpus = nvidia_smi_snapshot().get("gpus") or [{}]
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "tensorrt": trt.__version__,
        "device": gpus[0].get("name"),
        "driver": gpus[0].get("driver_version"),
        "exporter": "torch.onnx dynamo",
    }


def pilot_cache_context(
    data: DemoData,
    config: EvaluatorConfig,
    train_seed: int,
    code_version: str,
    space_hash: str,
    backend: dict[str, Any] | None = None,
) -> CacheContext:
    """Cache context of the FP32 pilot slice: data slice, seed, policies, code and backend."""
    return CacheContext(
        split_fingerprint=data.fingerprint,
        train_seed=train_seed,
        calibration_seed=0,
        train_policy=config.train.definition(),
        pruning_policy=PRUNING_DEFINITION,
        quantization_policy={
            "precision": "per region fp32/fp16/bf16/int8",
            "q_dq": QUANTIZATION_POLICY,
            "calibration_examples": config.calibration_examples,
            "qat": [QAT_INIT_CALIBRATION, QAT_RANGE_MOMENTUM, "1 epoch"],
        },
        code_version=code_version,
        backend=backend if backend is not None else backend_manifest(),
        builder_config=config.definition(),
        space_hash=space_hash,
    )


def cap_process_memory(device: torch.device, margin_mb: int) -> dict[str, Any]:
    """Limit this process's PyTorch allocations to the VRAM free now minus ``margin_mb``.

    Beyond the cap the caching allocator raises out-of-memory. Without it, the Windows
    driver may back an oversized network with system memory and train it at a fraction of
    the speed, charging hours to the GPU budget for a network that does not fit.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()  # "cuda" has no index
    free, total = torch.cuda.mem_get_info(index)
    allowed = max(free - margin_mb * 1_000_000, 0)
    fraction = min(max(allowed / total, 0.05), 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return {
        "device": index,
        "free_mb": free / 1e6,
        "total_mb": total / 1e6,
        "margin_mb": margin_mb,
        "fraction": fraction,
        "cap_mb": fraction * total / 1e6,
    }


def process_is_alive(pid: int) -> bool:
    """Whether ``pid`` names a running process; unknown answers count as alive (the cautious side)."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no such process
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def other_gpu_python_jobs() -> list[dict[str, str]]:
    """Python processes holding a GPU context besides this one (one GPU job at a time).

    ``nvidia-smi`` may still list a process for a moment after it ended (the
    benchmark child of this very evaluator, for one); entries whose process
    no longer runs are dropped.
    """
    from cnnlab.tcc.model_pipeline.environment import nvidia_smi_snapshot
    from cnnlab.tcc.model_pipeline.spike_trt import campaign_gpu_processes

    apps = nvidia_smi_snapshot().get("compute_apps", [])
    jobs = [dict(app) for app in campaign_gpu_processes(apps, os.getpid())]
    return [job for job in jobs if not job.get("pid", "").isdigit() or process_is_alive(int(job["pid"]))]


@dataclass(frozen=True)
class _Onnx:
    """What the first half of an evaluation hands to the second: the exported graph and its references."""

    exported: Any
    ref_path: Path
    block_precisions: list[str]
    regions: dict[str, Any]
    genome: ExpressedGenome
    networks: tuple[Any, Any] | None = None  # (FP32 folded reference, simulated typed network), on the CPU


class EngineEvaluator:
    """``Evaluator`` for the pilot demo; owns the GPU while ``evaluate`` runs.

    Building the evaluator touches neither the GPU nor the validation file:
    the data may live on the CPU and move to ``device`` at the first
    evaluation.

    One GPU job at a time: ``gpu_guard`` lists other Python processes that
    hold a GPU context. Every evaluation waits for an exclusive GPU before
    training and again before measuring, and a measurement during which
    another job appeared is discarded and repeated. (A process that merely
    *trains on the CPU* with PyTorch 2.13 creates a CUDA context, so test runs
    of other agents show up here.) A GPU that stays busy is an infrastructure
    condition: the attempt is ``interrupted``, never ``invalid``.
    """

    def __init__(
        self,
        data: DemoData,
        val_file: Path,
        config: EvaluatorConfig | None = None,
        device: str = "cuda",
        work_root: Path | None = None,
        gpu_guard: Callable[[], list[dict[str, str]]] | None = None,
    ) -> None:
        self.data = data
        self.val_file = Path(val_file)
        self.config = config or EvaluatorConfig()
        self.device = torch.device(device)
        self.work_root = Path(work_root) if work_root is not None else Path(tempfile.gettempdir()) / "tcc-eval-work"
        self.gpu_guard = gpu_guard
        self.val_file_sha256: str | None = None
        self.memory_cap: dict[str, Any] | None = None
        self._ready = False

    def _gpu_healthy(self) -> tuple[bool, str]:
        """A tiny computation on the device; tells a broken device from a failing candidate."""
        if self.device.type != "cuda":
            return True, "cpu device"
        try:
            total = float(torch.ones(8, device=self.device).sum())
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # any failure here means the device cannot be trusted
            return False, f"{type(exc).__name__}: {exc}"[:300]
        return (total == 8.0), ("ok" if total == 8.0 else f"wrong result {total}")

    def _classify(self, exc: Exception, stage: str, quality: dict[str, Any]) -> tuple[str, str, str]:
        """``(status, failure_code, detail)`` for an exception no stage anticipated."""
        detail = f"{type(exc).__name__}: {exc}"[:2000]
        quality["unexpected_exception"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:500]}
        if isinstance(exc, torch.cuda.OutOfMemoryError) or (
            isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        ):
            return "invalid", "out_of_memory", detail
        try:
            self._release()
        except Exception:  # the health check below reports the state of the device
            pass
        healthy, note = self._gpu_healthy()
        quality["unexpected_exception"]["device_health"] = note
        if not healthy:
            return "interrupted", INFRASTRUCTURE_PREFIX + "device_unhealthy", f"{detail}; device: {note}"
        if isinstance(exc, (OSError, MemoryError)):
            return "interrupted", INFRASTRUCTURE_PREFIX + "host_error", detail
        return "invalid", "unexpected_error", detail

    def _wait_exclusive(self, stage: str, quality: dict[str, Any]) -> None:
        """Block until no other Python job holds the GPU; give up after ``gpu_wait_s``."""
        if self.gpu_guard is None:
            return
        started = time.monotonic()
        try:
            self._wait_for_gpu(stage, quality, started)
        finally:
            excluded = quality.setdefault("excluded_from_gpu_seconds", {"gpu_wait": 0.0, "session_prepare": 0.0})
            excluded["gpu_wait"] += time.monotonic() - started

    def _wait_for_gpu(self, stage: str, quality: dict[str, Any], started: float) -> None:
        while True:
            busy = self.gpu_guard() if self.gpu_guard is not None else []
            if not busy:
                waited = time.monotonic() - started
                if waited >= self.config.gpu_poll_s:
                    quality.setdefault("gpu_waits", []).append({"before": stage, "waited_s": waited})
                return
            if time.monotonic() - started >= self.config.gpu_wait_s:
                quality.setdefault("gpu_waits", []).append(
                    {"before": stage, "waited_s": time.monotonic() - started, "gave_up": True}
                )
                raise Interrupted(
                    stage, "gpu_busy", f"other Python GPU jobs stayed alive for {self.config.gpu_wait_s:.0f} s: {busy}"
                )
            time.sleep(self.config.gpu_poll_s)

    def _prepare(self) -> None:
        """First evaluation of a session: data on the device and the validation file."""
        if self._ready:
            return
        if self.device.type == "cuda" and self.memory_cap is None:
            self.memory_cap = cap_process_memory(self.device, self.config.vram_margin_mb)
        if self.data.x_train.device != self.device:
            self.data = self.data.to(self.device)
        self.val_file.parent.mkdir(parents=True, exist_ok=True)
        self.val_file_sha256 = write_val_file(self.data, self.val_file)
        self._ready = True

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        started = time.monotonic()
        durations: dict[str, float] = {}
        excluded = {"gpu_wait": 0.0, "session_prepare": 0.0, "paused": 0.0}
        quality: dict[str, Any] = {
            "preliminary": True,
            "not_confirmatory": True,
            "evaluator": self.config.definition(),
            "data": self.data.manifest,
            "excluded_from_gpu_seconds": excluded,
        }
        progress = {"stage": "guard"}
        work = self.work_root / request.run_id / request.attempt_id

        def spent() -> float:
            """Time of this attempt: GPU waits, the one-off session setup and training pauses do not count."""
            return max(
                0.0,
                time.monotonic() - started - excluded["gpu_wait"] - excluded["session_prepare"] - excluded["paused"],
            )

        def failed(status: str, stage: str, code: str, detail: str) -> EvaluationResult:
            return EvaluationResult(
                run_id=request.run_id,
                candidate_id=request.candidate_id,
                attempt_id=request.attempt_id,
                status=status,
                fitness=None,
                gpu_seconds=spent(),
                durations_s=durations,  # type: ignore[arg-type]
                quality=quality,
                failure_stage=stage,
                failure_code=code,
                failure_detail=detail[:4000],
            )

        try:
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            self._wait_exclusive("guard", quality)
            progress["stage"] = "prepare"
            t = time.monotonic()
            self._prepare()
            excluded["session_prepare"] += time.monotonic() - t
            quality["val_file_sha256"] = self.val_file_sha256
            fitness, artifacts = self._run(request, work, durations, quality, progress)
        except Invalid as bad:
            return failed("invalid", bad.stage, bad.code, bad.detail)
        except Interrupted as stop:
            return failed("interrupted", stop.stage, stop.code, stop.detail)
        except Exception as exc:  # nothing may escape: a raw exception would end the whole job
            status, code, detail = self._classify(exc, progress["stage"], quality)
            return failed(status, progress["stage"], code, detail)
        finally:
            try:
                self._release()
            except Exception:  # a broken device was already reported by the branch above
                pass
            shutil.rmtree(work, ignore_errors=True)
        paths = {kind: rel for kind, (rel, _) in artifacts.items()}
        hashes = {kind: sha for kind, (_, sha) in artifacts.items()}
        return EvaluationResult(
            run_id=request.run_id,
            candidate_id=request.candidate_id,
            attempt_id=request.attempt_id,
            status="valid",
            fitness=fitness,
            gpu_seconds=spent(),
            durations_s=durations,
            artifact_paths=paths,
            artifact_sha256=hashes,
            quality=quality,
        )

    # -- stages -----------------------------------------------------------------------

    def _make_onnx(
        self,
        expressed: Mapping[str, Any],
        train_seed: int,
        work: Path,
        durations: dict[str, float],
        quality: dict[str, Any],
        progress: dict[str, str],
        keep_networks: bool = False,
    ) -> _Onnx:
        """Train the network, prune it, quantize it and export it: everything before the engine is built.

        Uses ``self.data`` and ``self.config.train``; the search and the final evaluation differ in those only.
        """
        progress["stage"] = "parse"
        try:
            genome = ExpressedGenome.from_dict(expressed)
        except (GenomeError, KeyError, TypeError) as exc:
            raise Invalid("parse", "bad_expressed_genome", str(exc)) from exc
        problems = representation_problems(genome)
        if problems:
            raise Invalid(
                "representation",
                "unsupported_representation",
                "pipeline materializes FP32, FP16, BF16 or INT8 in the stem, the blocks and the "
                "head (BF16 not with HardSwish nor depthwise operations) and no sparsity: " + ", ".join(problems),
            )
        regions = region_precisions(genome)
        block_precisions = regions["blocks"]

        progress["stage"] = "build"
        t = time.monotonic()
        try:
            model, info = build_model(genome, train_seed)
        except ModelBuildError as exc:
            raise Invalid("build", exc.code, exc.detail) from exc
        model = model.to(self.device)
        durations["build_model"] = time.monotonic() - t
        quality["model"] = {"parameters": info.parameters, "shortcuts": list(info.shortcuts)}

        progress["stage"] = "train"
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        t = time.monotonic()
        limit = self.config.train_time_limit_s
        train_started = _now()

        def must_stop() -> bool:
            return limit is not None and _now() - train_started > limit

        def time_limit_error() -> Invalid:
            peak = (
                f", peak allocated {torch.cuda.max_memory_allocated(self.device) / 1e6:.0f} MB"
                if self.device.type == "cuda"
                else ""
            )
            return Invalid(
                "train",
                "train_time_limit",
                f"training exceeded the limit of {limit} s (after {_now() - train_started:.1f} s{peak})",
            )

        try:
            report = train_model(
                model,
                self.data.x_train,
                self.data.y_train,
                log10_lr=genome.log10_lr,
                log10_wd=genome.log10_wd,
                config=self.config.train,
                seed=train_seed,
                should_stop=must_stop,
            )
        except InterruptedError as exc:  # only the time limit asks the trainer to stop
            raise time_limit_error() from exc
        except TrainingDiverged as exc:
            raise Invalid("train", "nonfinite_loss", str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            raise Invalid("train", "out_of_memory", str(exc)) from exc
        durations["train"] = time.monotonic() - t
        quality["train"] = asdict(report)
        quality.setdefault("excluded_from_gpu_seconds", {})["paused"] = float(report.extra.get("paused_s", 0.0))
        if self.device.type == "cuda":
            quality["train_memory"] = {
                "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1e6,
                "cap": self.memory_cap,
            }
        macs_unpruned = count_macs(model)
        prune_pcts = [block.prune_pct for block in genome.blocks]
        if any(prune_pcts):
            progress["stage"] = "prune"
            t = time.monotonic()
            before = evaluate_accuracy(model, self.data.x_val, self.data.y_val)
            try:
                prune_report = prune_network(model, prune_pcts)
            except PruningError as exc:
                raise Invalid("prune", exc.code, exc.detail) from exc
            after_prune = evaluate_accuracy(model, self.data.x_val, self.data.y_val)
            try:
                tune_report = train_model(
                    model,
                    self.data.x_train,
                    self.data.y_train,
                    log10_lr=genome.log10_lr,
                    log10_wd=genome.log10_wd,
                    config=finetune_config(self.config.train),
                    seed=train_seed + 2,
                    should_stop=must_stop,
                )
            except InterruptedError as exc:
                raise time_limit_error() from exc
            except TrainingDiverged as exc:
                raise Invalid("prune", "nonfinite_loss", str(exc)) from exc
            except torch.cuda.OutOfMemoryError as exc:
                raise Invalid("prune", "out_of_memory", str(exc)) from exc
            durations["prune"] = time.monotonic() - t
            quality["pruning"] = {
                **prune_report.to_dict(),
                "finetune": asdict(tune_report),
                "val_accuracy": {"before": before, "after_prune": after_prune},
            }
            quality["excluded_from_gpu_seconds"]["paused"] += float(tune_report.extra.get("paused_s", 0.0))
        quality["model"].update(
            parameters_pruned=count_parameters(model),
            macs=count_macs(model),
            macs_unpruned=macs_unpruned,
            widths=effective_widths(model),
        )
        progress["stage"] = "reference"
        accuracy_batches: dict[str, Any] = {}
        quality["torch_val_accuracy_diagnostic"] = evaluate_accuracy(
            model, self.data.x_val, self.data.y_val, info=accuracy_batches
        )

        t = time.monotonic()
        folded = fold_batchnorm(model)
        import numpy as np

        # PyTorch FP32 logits of every validation image, for the engine's parity diagnostic.
        reference_batches: dict[str, Any] = {}

        def logits(start: int, stop: int) -> torch.Tensor:
            with torch.no_grad():
                return folded(self.data.x_val[start:stop]).float().cpu()  # noqa: F821 (del depois do uso)

        ref = torch.cat(run_in_batches(logits, self.data.x_val.shape[0], info=reference_batches)).numpy()
        quality["diagnostic_batches"] = {"reference_logits": reference_batches, "torch_accuracy": accuracy_batches}
        ref_path = work / "ref_logits.npy"
        np.save(ref_path, ref)
        plan = None
        qat_net = None
        wanted_int8 = int8_regions(regions["stem"], block_precisions, regions["head"])
        if wanted_int8:
            progress["stage"] = "quantize"
            q0 = time.monotonic()
            mode = genome.int8_mode or "ptq"
            try:
                chosen = calibration_indices(self.data.y_train, self.config.calibration_examples, seed=0)
                plan = plan_quantization(
                    folded.float().eval(),
                    wanted_int8,
                    self.data.x_train[chosen],
                    mode,
                    genome.weight_granularity or "per_channel",
                    (genome.calibration or "max") if mode == "ptq" else QAT_INIT_CALIBRATION,
                    chosen,
                )
            except QuantizationError as exc:
                raise Invalid("quantize", exc.code, exc.detail) from exc
            durations["quantize"] = time.monotonic() - q0
            quality["quantization"] = plan.definition()
            if mode == "qat":
                progress["stage"] = "qat"
                q0 = time.monotonic()
                qat_net = fold_batchnorm(model)
                insert_qdq(qat_net, plan)
                try:
                    qat_report = train_model(
                        qat_net,
                        self.data.x_train,
                        self.data.y_train,
                        log10_lr=genome.log10_lr,
                        log10_wd=genome.log10_wd,
                        config=finetune_config(self.config.train),
                        seed=train_seed + 1,
                        should_stop=must_stop,
                    )
                except InterruptedError as exc:
                    raise time_limit_error() from exc
                except TrainingDiverged as exc:
                    raise Invalid("qat", "nonfinite_loss", str(exc)) from exc
                except torch.cuda.OutOfMemoryError as exc:
                    raise Invalid("qat", "out_of_memory", str(exc)) from exc
                durations["qat"] = time.monotonic() - q0
                quality["qat"] = asdict(qat_report)
                quality["quantization"]["final_input_ranges"] = {
                    name: round(float(m.input_amax), 6)
                    for name, m in qat_net.named_modules()
                    if isinstance(m, QDQLayer)
                }
                quality["excluded_from_gpu_seconds"]["paused"] += float(qat_report.extra.get("paused_s", 0.0))
        progress["stage"] = "export"
        typed = (
            typed_network(qat_net, block_precisions, regions["stem"], regions["head"], plan, prefolded=True)
            if qat_net is not None
            else typed_network(model, block_precisions, regions["stem"], regions["head"], plan)
        )
        if plan is not None:  # what the simulated Q/DQ network scores in PyTorch, next to the engine's own E
            quality["torch_val_accuracy_quantized_diagnostic"] = evaluate_accuracy(
                typed, self.data.x_val, self.data.y_val
            )
        quality["precision"] = {
            "requested_blocks": block_precisions,
            "requested_stem": regions["stem"],
            "requested_head": regions["head"],
            "casts_in_graph": typed.casts,
        }
        try:
            exported = export_network(typed, work)
        except ExportError as exc:
            raise Invalid("export", "onnx_export_failed", str(exc)) from exc
        durations["export"] = time.monotonic() - t
        kept: tuple[Any, Any] | None = None
        if keep_networks:  # parked on the CPU: the engines are built next, the parity needs them afterwards
            kept = (folded.float().cpu().eval(), typed.cpu().eval())
        del folded, typed, model, qat_net
        self._release()
        return _Onnx(
            exported=exported,
            ref_path=ref_path,
            block_precisions=block_precisions,
            regions=regions,
            genome=genome,
            networks=kept,
        )

    def _run(
        self,
        request: EvaluationRequest,
        work: Path,
        durations: dict[str, float],
        quality: dict[str, Any],
        progress: dict[str, str],
    ) -> tuple[Fitness, dict[str, tuple[str, str]]]:
        prepared = self._make_onnx(
            request.genome.expressed_genome, request.train_seed, work, durations, quality, progress
        )
        exported, ref_path = prepared.exported, prepared.ref_path
        block_precisions, regions = prepared.block_precisions, prepared.regions

        progress["stage"] = "trt_build"
        t = time.monotonic()
        plan_path = work / "engine.plan"
        try:
            engine = build_engine(exported.onnx_path, plan_path)
        except EngineBuildError as exc:
            raise Invalid(exc.stage, f"{exc.stage}_failed", exc.detail) from exc
        durations["trt_build"] = time.monotonic() - t
        (work / "inspector.json").write_text(json.dumps(engine.inspector), encoding="utf-8")
        quality["engine"] = {**engine.builder, "plan_bytes": engine.plan_bytes, **summarize_layers(engine.inspector)}
        progress["stage"] = "precision_check"
        check = check_engine_precision(
            exported.onnx_path, engine.inspector, block_precisions, regions["stem"], regions["head"]
        )
        quality["precision"].update(
            {"ok": check["ok"], "regions": check["regions"], "problems": check["problems"], "policy": check["policy"]}
        )
        (work / "precision.json").write_text(json.dumps(check, sort_keys=True), encoding="utf-8")
        if not check["ok"]:
            raise Invalid("precision_check", "precision_not_executed", "; ".join(check["problems"]))
        self._release()

        progress["stage"] = "benchmark"
        t = time.monotonic()
        try:
            bench = self._benchmark(plan_path, work / "bench.json", ref_path, quality)
        finally:
            durations["benchmark"] = time.monotonic() - t

        e = bench.get("e") or {}
        if not e.get("all_finite", False):
            raise Invalid("benchmark", "nonfinite_engine_output", "engine produced non-finite logits on validation")
        latency, memory = bench["latency"], bench["memory"]
        quality["measurement"] = {
            "e": {"unit": "fraction top-1 error", "source": "TensorRT engine, batch 1", **e},
            "l": {
                "unit": "ms",
                "definition": "median per-inference GPU time, CUDA events, batch 1",
                **{k: v for k, v in latency.items()},
            },
            "v": {"unit": "MB decimal", "definition": "peak allocator bytes above CUDA-context baseline", **memory},
            "s": {
                "unit": "MB decimal",
                "definition": "bytes of the serialized .plan / 1e6",
                "plan_bytes": engine.plan_bytes,
            },
            "parity": bench.get("quality", {}),
            "bench_version": bench["bench_version"],
        }
        try:
            fitness = Fitness(
                e=float(e["e_fraction"]),
                l_ms=float(latency["median_ms"]),
                v_mb=memory["v_bytes_allocator_peak"] / BYTES_PER_MB,
                s_mb=engine.plan_bytes / BYTES_PER_MB,
            )
        except (KeyError, ValueError) as exc:
            raise Invalid("benchmark", "invalid_measurement", str(exc)) from exc

        progress["stage"] = "publish"
        artifacts = self._publish(request, work, exported, bench, quality)
        return fitness, artifacts

    def _benchmark(
        self, plan_path: Path, bench_path: Path, ref_path: Path, quality: dict[str, Any], val_path: Path | None = None
    ) -> dict[str, Any]:
        """Isolated measurement with bounded retries; every failed try is kept in ``quality``.

        ``val_path`` is the labelled set the engine's E is computed on (the validation file unless given).
        """
        tries: list[dict[str, Any]] = []
        quality["benchmark_tries"] = tries
        budget = {
            "spawn": self.config.bench_spawn_retries,
            "crash": self.config.bench_crash_retries,
            "shared_gpu": self.config.bench_shared_retries,
            "measurement": self.config.bench_measurement_retries,
        }
        while True:
            self._wait_exclusive("benchmark", quality)
            try:
                report = run_isolated(
                    plan_path,
                    bench_path,
                    val_path=val_path or self.val_file,
                    ref_logits_path=ref_path,
                    warmup=self.config.bench_warmup,
                    iters=self.config.bench_iters,
                    timeout_s=self.config.bench_timeout_s,
                )
            except BenchProcessError as exc:
                tries.append({"kind": exc.kind, "exit_code": exc.exit_code, "detail": exc.detail[:500]})
                if budget.get(exc.kind, 0) > 0:
                    budget[exc.kind] -= 1
                    time.sleep(self.config.bench_retry_wait_s)
                    continue
                if exc.kind == "spawn":
                    raise Interrupted("benchmark", "spawn_failed", str(exc)) from exc
                # The child ran and failed. A device that no longer computes is infrastructure;
                # on a healthy device the failure belongs to this engine.
                healthy, note = self._gpu_healthy()
                tries[-1]["device_health"] = note
                if not healthy:
                    raise Interrupted("benchmark", "device_unhealthy", f"{exc}; device: {note}") from exc
                if "out of memory" in exc.detail.lower():
                    raise Invalid("benchmark", "out_of_memory", str(exc)) from exc
                code = {"crash": "benchmark_process_crashed", "timeout": "benchmark_timeout"}.get(
                    exc.kind, "benchmark_failed"
                )
                raise Invalid("benchmark", code, str(exc)) from exc
            intruders = self.gpu_guard() if self.gpu_guard is not None else []
            if intruders:
                # Another job took a GPU context while the engine was being timed: L is not trustworthy.
                tries.append({"kind": "shared_gpu", "exit_code": 0, "detail": f"{len(intruders)} other job(s)"})
                if budget["shared_gpu"] > 0:
                    budget["shared_gpu"] -= 1
                    continue
                raise Interrupted(
                    "benchmark", "gpu_shared", f"the GPU was shared during every measurement: {intruders}"
                )
            tries.append({"kind": "ok"})
            quality["gpu_exclusive_at_benchmark"] = self.gpu_guard is not None
            return report

    def _publish(
        self, request: EvaluationRequest, work: Path, exported: Any, bench: dict[str, Any], quality: dict[str, Any]
    ) -> dict[str, tuple[str, str]]:
        """Copy the attempt's artifacts under the store root atomically; return kind -> (rel_path, sha256)."""
        rel_dir = PurePosixPath("runs", request.run_id, "attempts", request.attempt_id)
        target = Path(request.artifact_root) / rel_dir
        target.mkdir(parents=True, exist_ok=True)
        files: dict[str, tuple[str, bytes]] = {
            "engine": ("engine.plan", (work / "engine.plan").read_bytes()),
            "inspector": ("inspector.json", (work / "inspector.json").read_bytes()),
            "precision": ("precision.json", (work / "precision.json").read_bytes()),
            "bench": ("bench.json", json.dumps(bench, sort_keys=True).encode("utf-8")),
            "train_log": ("train.json", json.dumps(quality.get("train", {}), sort_keys=True).encode("utf-8")),
        }
        if self.config.keep_onnx:
            files["onnx"] = ("model.onnx", exported.onnx_path.read_bytes())
            if exported.data_path is not None:
                files["onnx_data"] = ("model.onnx.data", exported.data_path.read_bytes())
        out: dict[str, tuple[str, str]] = {}
        for kind, (name, payload) in files.items():
            write_atomic(target / name, payload)
            out[kind] = (str(rel_dir / name), hashlib.sha256(payload).hexdigest())
        if out["engine"][1] != sha256_file(target / "engine.plan"):
            raise Interrupted(
                "publish",
                "artifact_write_mismatch",
                "engine.plan read back from the artifact root differs from what was written",
            )
        return out

    def _release(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
