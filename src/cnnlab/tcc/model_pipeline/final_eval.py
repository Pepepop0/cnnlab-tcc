"""Final evaluation of a finalist in two passes (TCC-01 section 6; task G9).

The protocol freezes the finalist list **and its artifacts** before the official test is opened, so the evaluation is two
functions that never share the test:

1. :func:`final_build` retrains the finalist **from scratch** (30 epochs on the full 45k training split, with the
   retraining seed of its seed block), prunes, fine-tunes, quantizes (PTQ or QAT) and exports it exactly like a search
   candidate, builds the engine **three times** (TensorRT's tactic choice differs between builds: L moves by a few
   percent and the plan size by about 1.5%, see ``wave3-gauss.md``) and measures each build on the validation split.
   L, S and V of the finalist are the medians of the three builds, all three values are kept, and the build whose L is
   the median is the **final engine**. The final engine, the other builds, the graph and the two PyTorch reference
   networks (FP32 and the simulated quantized one, for the test parity) are written and hashed. It receives no access to
   the official test and cannot open it.
2. :func:`final_test` takes the manifest of a finished build, verifies the SHA-256 of every file it will use, asks the
   :class:`TestLedger` to ``claim`` the engine (an artifact is evaluated once), and measures E, top-5, the per-class
   metrics and the PyTorch parity of that engine on the 10,000 official test images, in the engine (bench v4, CUDA
   Graph). The test is read only through :class:`TestAccess`, whose finalist grant exists only for a frozen list; the
   operation releases it after every engine of the list is built and its manifest confirmed.

The caller (the finalist job, task A12) creates the grant, catalogues the artifacts and records the test evaluation; this
module does not touch SQLite.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch

from cnnlab.tcc.model_pipeline.bench import BENCH_VERSION
from cnnlab.tcc.model_pipeline.data_access import (
    OFFICIAL_SIZES,
    CifarSource,
    DataAccessGrant,
    LoadedSplit,
    SealedSplitError,
    SplitIndices,
    grant_finalist_access,
    load_splits,
)
from cnnlab.tcc.model_pipeline.demo_data import DemoData
from cnnlab.tcc.model_pipeline.engine import EngineBuildError, build_engine, sha256_file, summarize_layers, write_atomic
from cnnlab.tcc.model_pipeline.evaluator import (
    BYTES_PER_MB,
    EngineEvaluator,
    EvaluatorConfig,
    Interrupted,
    Invalid,
)
from cnnlab.tcc.model_pipeline.precision import check_engine_precision
from cnnlab.tcc.model_pipeline.quantization import N_CALIBRATION
from cnnlab.tcc.model_pipeline.splits import DemoSplitSpec, poc_stratified_split
from cnnlab.tcc.model_pipeline.trainer import TrainConfig, normalize_images, run_in_batches
from cnnlab.tcc.science.genome import ExpressedGenome

FINAL_VERSION = "tcc-final/v2"
FINAL_EPOCHS = 30
FINAL_BUILDS = 3
NUM_CLASSES = 100


class TestAlreadyEvaluated(RuntimeError):
    """The official test was already run on this engine artifact (once per unique artifact)."""

    __test__ = False  # not a pytest class


class TestLedger(Protocol):
    """Where evaluated engines are recorded. ``claim`` raises :class:`TestAlreadyEvaluated` on a repeat; ``release``
    gives a claim back when the measurement died before producing any result (an infrastructure failure)."""

    __test__ = False

    def claim(self, engine_sha256: str) -> None: ...

    def release(self, engine_sha256: str) -> None: ...


class MemoryLedger:
    """In-memory ledger (tests, dry runs). The campaign's ledger is the ``test_evaluations`` table (UNIQUE sha)."""

    def __init__(self) -> None:
        self.claimed: list[str] = []

    def claim(self, engine_sha256: str) -> None:
        if engine_sha256 in self.claimed:
            raise TestAlreadyEvaluated(engine_sha256)
        self.claimed.append(engine_sha256)

    def release(self, engine_sha256: str) -> None:
        if engine_sha256 in self.claimed:
            self.claimed.remove(engine_sha256)


@dataclass(frozen=True)
class TestAccess:
    """The key to the official test split: a finalist grant, the source of the files and the once-only ledger."""

    __test__ = False

    grant: DataAccessGrant
    source: CifarSource
    ledger: TestLedger

    def __post_init__(self) -> None:
        if not self.grant.may_read_official_test:
            raise SealedSplitError("the final test needs a finalist grant that holds the official test split")

    def load_test(self) -> LoadedSplit:
        return load_splits(self.grant, self.source, names=["test"])["test"]


@dataclass(frozen=True)
class FinalConfig:
    """Protocol constants of the final evaluation (TCC-01 section 6); the tests lower the epochs and the builds."""

    epochs: int = FINAL_EPOCHS
    builds: int = FINAL_BUILDS
    batch_size: int = 128
    deterministic: bool = False
    calibration_examples: int = N_CALIBRATION
    bench_warmup: int = 50
    bench_iters: int = 1000
    train_time_limit_s: float | None = None
    gpu_wait_s: float = 600.0

    def __post_init__(self) -> None:
        if self.builds < 1 or self.builds % 2 == 0:
            raise ValueError("builds must be odd, so that the median is one of the builds")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")

    def evaluator_config(self) -> EvaluatorConfig:
        return EvaluatorConfig(
            train=TrainConfig(epochs=self.epochs, batch_size=self.batch_size, deterministic=self.deterministic),
            bench_warmup=self.bench_warmup,
            bench_iters=self.bench_iters,
            gpu_wait_s=self.gpu_wait_s,
            calibration_examples=self.calibration_examples,
            train_time_limit_s=self.train_time_limit_s,
        )

    def definition(self) -> dict[str, Any]:
        return {
            "version": FINAL_VERSION,
            "epochs": self.epochs,
            "builds": self.builds,
            "evaluator": self.evaluator_config().definition(),
            "bench_version": BENCH_VERSION,
            "median_of": "the builds' L, S and V; the final engine is the build with the median L",
        }


@dataclass(frozen=True)
class FinalBuild:
    """One of the engine builds: its identity and what the validation run measured on it."""

    index: int
    plan_sha256: str
    plan_bytes: int
    s_mb: float
    l_ms: float
    v_mb: float
    e_val: float
    top5_val: float
    precision_ok: bool
    build_s: float
    bench_version: str


@dataclass(frozen=True)
class FinalBuildResult:
    """Outcome of :func:`final_build`; ``to_dict()`` is the manifest that :func:`final_test` takes."""

    status: Literal["valid", "invalid", "interrupted"]
    retrain_seed: int
    label: str
    builds: tuple[FinalBuild, ...] = ()
    chosen_build: int | None = None
    l_ms: float | None = None  # median of the builds
    v_mb: float | None = None
    s_mb: float | None = None
    e_val: float | None = None  # of the final engine, on the validation split
    engine_sha256: str | None = None
    artifact_paths: Mapping[str, str] = field(default_factory=dict)  # relative to the artifact directory
    artifact_sha256: Mapping[str, str] = field(default_factory=dict)
    failure_stage: str | None = None
    failure_code: str | None = None
    failure_detail: str | None = None
    quality: Mapping[str, Any] = field(default_factory=dict)
    durations_s: Mapping[str, float] = field(default_factory=dict)
    gpu_seconds: float = 0.0
    final_version: str = FINAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinalTestResult:
    """Outcome of :func:`final_test`: the official-test metrics of one engine, or why there are none."""

    status: Literal["valid", "invalid", "interrupted"]
    engine_sha256: str | None
    test: dict[str, Any] | None = None
    failure_stage: str | None = None
    failure_code: str | None = None
    failure_detail: str | None = None
    quality: Mapping[str, Any] = field(default_factory=dict)
    durations_s: Mapping[str, float] = field(default_factory=dict)
    gpu_seconds: float = 0.0
    final_version: str = FINAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- data ----------------------------------------------------------------------------------------------------------


def final_grant(
    source: CifarSource, run_id: str, finalist_list_id: str, spec: DemoSplitSpec | None = None
) -> DataAccessGrant:
    """Finalist grant: the 45k/5k split of the official train file and **all** of the official test file.

    Only the labels of the train file are read here; no image, and not the test file.
    """
    spec = spec or DemoSplitSpec(train_per_class=None, subset_seed=0)
    _, train_labels = source.read("train")
    train_idx, val_idx = poc_stratified_split(train_labels, spec.val_fraction, spec.split_seed)
    return grant_finalist_access(
        run_id,
        finalist_list_id,
        [
            SplitIndices("train", "train", tuple(train_idx)),
            SplitIndices("val", "train", tuple(val_idx)),
            SplitIndices("test", "test", tuple(range(OFFICIAL_SIZES["test"]))),
        ],
    )


def prepare_final_data(grant: DataAccessGrant, source: CifarSource, device: torch.device) -> DemoData:
    """Train (45k) and validation (5k) tensors through the grant; the test split is not read."""
    loaded = load_splits(grant, source, names=["train", "val"])
    train, val = loaded["train"], loaded["val"]
    manifest = {
        "schema": "tcc-final-data/v1",
        "stage": "final",
        "n_train": int(train.labels.numel()),
        "n_val": int(val.labels.numel()),
        "fingerprints": grant.fingerprints(),
        "finalist_list_id": grant.finalist_list_id,
        "official_test_read": False,
    }
    return DemoData(
        x_train=normalize_images(train.images_u8.to(device)),
        y_train=train.labels.to(device),
        x_val=normalize_images(val.images_u8.to(device)),
        y_val=val.labels.to(device),
        manifest=manifest,
    )


def per_class_metrics(counts: Mapping[str, list[int]]) -> list[dict[str, Any]]:
    """Per-class support, precision, recall and F1 from the engine's confusion counts."""
    rows = []
    for index, (support, predicted, tp) in enumerate(
        zip(counts["support"], counts["predicted"], counts["true_positive"])
    ):
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class": index,
                "support": support,
                "predicted": predicted,
                "true_positive": tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


# --- pass 1: the build ---------------------------------------------------------------------------------------------------


class FinalBuilder(EngineEvaluator):
    """Retrain, compress, export and build three engines; the first half of the pipeline is the search evaluator's."""

    def __init__(
        self,
        data: DemoData,
        val_file: Path,
        config: FinalConfig,
        device: str = "cuda",
        work_root: Path | None = None,
        gpu_guard: Callable[[], list[dict[str, str]]] | None = None,
    ) -> None:
        super().__init__(
            data, val_file, config.evaluator_config(), device=device, work_root=work_root, gpu_guard=gpu_guard
        )
        self.final = config

    def run(
        self, genome: Mapping[str, Any] | ExpressedGenome, retrain_seed: int, artifact_dir: Path, label: str
    ) -> FinalBuildResult:
        started = time.monotonic()
        durations: dict[str, float] = {}
        excluded = {"gpu_wait": 0.0, "session_prepare": 0.0, "paused": 0.0}
        quality: dict[str, Any] = {
            "protocol_stage": "final",
            "final": self.final.definition(),
            "data": self.data.manifest,
            "excluded_from_gpu_seconds": excluded,
        }
        progress = {"stage": "guard"}
        work = self.work_root / label
        builds: list[FinalBuild] = []

        def spent() -> float:
            return max(
                0.0,
                time.monotonic() - started - excluded["gpu_wait"] - excluded["session_prepare"] - excluded["paused"],
            )

        def failed(status: str, stage: str, code: str, detail: str) -> FinalBuildResult:
            return FinalBuildResult(
                status=status,
                retrain_seed=retrain_seed,
                label=label,  # type: ignore[arg-type]
                builds=tuple(builds),
                failure_stage=stage,
                failure_code=code,
                failure_detail=detail[:4000],
                quality=quality,
                durations_s=durations,
                gpu_seconds=spent(),
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
            expressed = genome.to_dict() if isinstance(genome, ExpressedGenome) else dict(genome)
            return self._build(
                expressed,
                retrain_seed,
                work,
                Path(artifact_dir) / label,
                label,
                durations,
                quality,
                progress,
                builds,
                spent,
            )
        except Invalid as bad:
            return failed("invalid", bad.stage, bad.code, bad.detail)
        except Interrupted as stop:
            return failed("interrupted", stop.stage, stop.code, stop.detail)
        except Exception as exc:
            status, code, detail = self._classify(exc, progress["stage"], quality)
            return failed(status, progress["stage"], code, detail)
        finally:
            try:
                self._release()
            except Exception:
                pass
            shutil.rmtree(work, ignore_errors=True)

    def _build(
        self,
        expressed: Mapping[str, Any],
        seed: int,
        work: Path,
        target: Path,
        label: str,
        durations: dict[str, float],
        quality: dict[str, Any],
        progress: dict[str, str],
        builds: list[FinalBuild],
        spent: Callable[[], float],
    ) -> FinalBuildResult:
        prepared = self._make_onnx(expressed, seed, work, durations, quality, progress, keep_networks=True)
        onnx_path = prepared.exported.onnx_path
        plans: list[Path] = []
        benches: list[dict[str, Any]] = []
        quality["benchmark_tries"] = []
        for index in range(self.final.builds):
            progress["stage"] = f"trt_build[{index}]"
            t = time.monotonic()
            plan_path = work / f"engine{index}.plan"
            try:
                engine = build_engine(onnx_path, plan_path)
            except EngineBuildError as exc:
                raise Invalid(exc.stage, f"{exc.stage}_failed", exc.detail) from exc
            build_s = time.monotonic() - t
            durations[f"trt_build_{index}"] = build_s
            (work / f"inspector{index}.json").write_text(json.dumps(engine.inspector), encoding="utf-8")
            progress["stage"] = f"precision_check[{index}]"
            check = check_engine_precision(
                onnx_path,
                engine.inspector,
                prepared.block_precisions,
                prepared.regions["stem"],
                prepared.regions["head"],
            )
            (work / f"precision{index}.json").write_text(json.dumps(check, sort_keys=True), encoding="utf-8")
            if not check["ok"]:
                raise Invalid(
                    "precision_check", "precision_not_executed", f"build {index}: " + "; ".join(check["problems"])
                )
            self._release()
            progress["stage"] = f"benchmark[{index}]"
            scratch: dict[str, Any] = {"excluded_from_gpu_seconds": quality["excluded_from_gpu_seconds"]}
            t = time.monotonic()
            bench = self._benchmark(plan_path, work / f"bench{index}.json", prepared.ref_path, scratch)
            durations[f"benchmark_{index}"] = time.monotonic() - t
            quality["benchmark_tries"].append(scratch.get("benchmark_tries", []))
            e = bench.get("e") or {}
            if not e.get("all_finite", False):
                raise Invalid("benchmark", "nonfinite_engine_output", f"build {index}: non-finite logits on validation")
            builds.append(
                FinalBuild(
                    index=index,
                    plan_sha256=engine.plan_sha256,
                    plan_bytes=engine.plan_bytes,
                    s_mb=engine.plan_bytes / BYTES_PER_MB,
                    l_ms=float(bench["latency"]["median_ms"]),
                    v_mb=bench["memory"]["v_bytes_allocator_peak"] / BYTES_PER_MB,
                    e_val=float(e["e_fraction"]),
                    top5_val=float(e.get("top5_accuracy", float("nan"))),
                    precision_ok=True,
                    build_s=build_s,
                    bench_version=bench["bench_version"],
                )
            )
            plans.append(plan_path)
            benches.append(bench)
            if index == 0:
                quality["engine"] = {"layers": summarize_layers(engine.inspector)}
        quality["engine_builds"] = [asdict(b) for b in builds]

        # The final engine is the build with the median L (the lower one when two tie).
        order = sorted(range(len(builds)), key=lambda i: (builds[i].l_ms, i))
        chosen = order[len(order) // 2]
        progress["stage"] = "publish"
        fp32_net, simulated_net = prepared.networks
        torch.save(fp32_net, work / "reference_fp32.pt")
        torch.save(simulated_net, work / "reference_simulated.pt")
        artifacts = self._publish_build(target, work, prepared, plans, chosen, benches, quality)
        return FinalBuildResult(
            status="valid",
            retrain_seed=seed,
            label=label,
            builds=tuple(builds),
            chosen_build=chosen,
            l_ms=statistics.median(b.l_ms for b in builds),
            v_mb=statistics.median(b.v_mb for b in builds),
            s_mb=statistics.median(b.s_mb for b in builds),
            e_val=builds[chosen].e_val,
            engine_sha256=builds[chosen].plan_sha256,
            artifact_paths={k: v[0] for k, v in artifacts.items()},
            artifact_sha256={k: v[1] for k, v in artifacts.items()},
            quality=quality,
            durations_s=durations,
            gpu_seconds=spent(),
        )

    def _publish_build(
        self,
        target: Path,
        work: Path,
        prepared: Any,
        plans: list[Path],
        chosen: int,
        benches: list[dict[str, Any]],
        quality: dict[str, Any],
    ) -> dict[str, tuple[str, str]]:
        """Write and verify the artifacts; return kind -> (path relative to the artifact directory, sha256)."""
        target.mkdir(parents=True, exist_ok=True)
        files: dict[str, tuple[str, bytes]] = {
            "engine": ("engine.plan", plans[chosen].read_bytes()),
            "inspector": ("inspector.json", (work / f"inspector{chosen}.json").read_bytes()),
            "precision": ("precision.json", (work / f"precision{chosen}.json").read_bytes()),
            "bench": ("bench.json", json.dumps(benches[chosen], sort_keys=True).encode("utf-8")),
            "train_log": ("train.json", json.dumps(quality.get("train", {}), sort_keys=True).encode("utf-8")),
            "builds": ("builds.json", json.dumps(quality["engine_builds"], sort_keys=True).encode("utf-8")),
            "reference_fp32": ("reference_fp32.pt", (work / "reference_fp32.pt").read_bytes()),
            "reference_simulated": ("reference_simulated.pt", (work / "reference_simulated.pt").read_bytes()),
        }
        for index, plan in enumerate(plans):
            if index != chosen:
                files[f"engine_build{index}"] = (f"engine-build{index}.plan", plan.read_bytes())
        files["onnx"] = ("model.onnx", prepared.exported.onnx_path.read_bytes())
        if prepared.exported.data_path is not None:
            files["onnx_data"] = ("model.onnx.data", prepared.exported.data_path.read_bytes())
        out: dict[str, tuple[str, str]] = {}
        for kind, (name, payload) in files.items():
            write_atomic(target / name, payload)
            digest = hashlib.sha256(payload).hexdigest()
            if sha256_file(target / name) != digest:
                raise Interrupted(
                    "publish", "artifact_write_mismatch", f"{name} read back differs from what was written"
                )
            out[kind] = (f"{target.name}/{name}", digest)
        return out


def final_build(
    genome: Mapping[str, Any] | ExpressedGenome,
    retrain_seed: int,
    config: FinalConfig,
    data: DemoData,
    work_dir: Path,
    artifact_dir: Path,
    *,
    label: str,
    device: str = "cuda",
    gpu_guard: Callable[[], list[dict[str, str]]] | None = None,
) -> FinalBuildResult:
    """Pass 1: retrain a finalist from scratch, compress it, build its engine three times, write and hash the artifacts.

    ``data`` comes from :func:`prepare_final_data` (train 45k + validation 5k). Nothing here can reach the official
    test: it takes no :class:`TestAccess`. The artifacts go to ``artifact_dir/label``; the returned paths are relative
    to ``artifact_dir``, and ``result.to_dict()`` is the manifest for :func:`final_test`.
    """
    work_dir = Path(work_dir)
    builder = FinalBuilder(
        data, work_dir / "val.npz", config, device=device, work_root=work_dir / "work", gpu_guard=gpu_guard
    )
    return builder.run(genome, retrain_seed, Path(artifact_dir), label)


# --- pass 2: the official test ---------------------------------------------------------------------------------------------


class FinalTester(EngineEvaluator):
    """Measures one built engine on the official test split (uses only the benchmark half of the evaluator)."""

    def __init__(
        self,
        config: FinalConfig,
        device: str = "cuda",
        work_root: Path | None = None,
        gpu_guard: Callable[[], list[dict[str, str]]] | None = None,
    ) -> None:
        super().__init__(
            None,
            Path("unused"),
            config.evaluator_config(),  # type: ignore[arg-type]
            device=device,
            work_root=work_root,
            gpu_guard=gpu_guard,
        )

    def run(self, manifest: Mapping[str, Any], artifact_dir: Path, access: TestAccess) -> FinalTestResult:
        started = time.monotonic()
        durations: dict[str, float] = {}
        quality: dict[str, Any] = {
            "protocol_stage": "final",
            "excluded_from_gpu_seconds": {"gpu_wait": 0.0, "session_prepare": 0.0},
        }
        progress = {"stage": "verify"}
        engine_sha = manifest.get("engine_sha256")
        work = self.work_root / f"test-{str(engine_sha)[:16]}"
        claimed = False

        def spent() -> float:
            return max(0.0, time.monotonic() - started - quality["excluded_from_gpu_seconds"]["gpu_wait"])

        def failed(status: str, stage: str, code: str, detail: str) -> FinalTestResult:
            return FinalTestResult(
                status=status,
                engine_sha256=engine_sha,
                failure_stage=stage,  # type: ignore[arg-type]
                failure_code=code,
                failure_detail=detail[:4000],
                quality=quality,
                durations_s=durations,
                gpu_seconds=spent(),
            )

        try:
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            files = self._verified_files(manifest, Path(artifact_dir))
            self._wait_exclusive("guard", quality)
            progress["stage"] = "claim"
            access.ledger.claim(str(engine_sha))
            claimed = True
            progress["stage"] = "test"
            t = time.monotonic()
            test = self._measure_test(files, access, work, quality)
            durations["test"] = time.monotonic() - t
            return FinalTestResult(
                status="valid",
                engine_sha256=engine_sha,
                test=test,
                quality=quality,
                durations_s=durations,
                gpu_seconds=spent(),
            )
        except TestAlreadyEvaluated:
            raise  # a repeat of the one-time test is a caller's error, never a finding about the candidate
        except Invalid as bad:
            return failed("invalid", bad.stage, bad.code, bad.detail)
        except Interrupted as stop:
            if claimed:
                access.ledger.release(str(engine_sha))  # nothing was measured: the artifact may be tried again
            return failed("interrupted", stop.stage, stop.code, stop.detail)
        except Exception as exc:
            if claimed and "test_metrics" not in quality:
                access.ledger.release(str(engine_sha))
            status, code, detail = self._classify(exc, progress["stage"], quality)
            return failed(status, progress["stage"], code, detail)
        finally:
            try:
                self._release()
            except Exception:
                pass
            shutil.rmtree(work, ignore_errors=True)

    @staticmethod
    def _verified_files(manifest: Mapping[str, Any], artifact_dir: Path) -> dict[str, Path]:
        """The files the test needs, each checked against the SHA-256 of the manifest before anything is loaded."""
        if manifest.get("status") != "valid" or not manifest.get("engine_sha256"):
            raise Invalid("verify", "no_final_engine", "the manifest is not of a valid build")
        files: dict[str, Path] = {}
        for kind in ("engine", "reference_fp32", "reference_simulated"):
            relative = manifest.get("artifact_paths", {}).get(kind)
            expected = manifest.get("artifact_sha256", {}).get(kind)
            if not relative or not expected:
                raise Invalid("verify", "manifest_incomplete", f"the manifest has no {kind}")
            path = artifact_dir / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise Invalid("verify", "artifact_changed", f"{kind} ({relative}) is missing or differs from its hash")
            files[kind] = path
        if manifest["artifact_sha256"]["engine"] != manifest["engine_sha256"]:
            raise Invalid(
                "verify", "engine_hash_mismatch", "engine_sha256 differs from the hash of the engine artifact"
            )
        return files

    def _measure_test(
        self, files: Mapping[str, Path], access: TestAccess, work: Path, quality: dict[str, Any]
    ) -> dict[str, Any]:
        split = access.load_test()
        x = normalize_images(split.images_u8.to(self.device))
        n = int(split.labels.numel())
        labels = split.labels.cpu()

        def logits_of(path: Path) -> torch.Tensor:
            net = torch.load(path, weights_only=False).to(self.device).eval()  # a pickle of our own, hash-verified

            def run(start: int, stop: int) -> torch.Tensor:
                with torch.no_grad():
                    return net(x[start:stop]).float().cpu()  # noqa: F821 (del depois do uso)

            out = torch.cat(run_in_batches(run, n))
            del net
            self._release()
            return out

        ref = logits_of(files["reference_fp32"])
        simulated = logits_of(files["reference_simulated"])
        ref_path, test_path = work / "test_ref_logits.npy", work / "test.npz"
        np.save(ref_path, ref.numpy())
        np.savez(test_path, x=x.cpu().numpy().astype("float32"), y=labels.numpy())
        del x
        self._release()
        scratch: dict[str, Any] = {"excluded_from_gpu_seconds": quality["excluded_from_gpu_seconds"]}
        try:
            bench = self._benchmark(files["engine"], work / "bench_test.json", ref_path, scratch, val_path=test_path)
        finally:
            test_path.unlink(missing_ok=True)
        e = bench.get("e") or {}
        if not e.get("all_finite", False):
            raise Invalid("test", "nonfinite_engine_output", "engine produced non-finite logits on the test split")
        quality["test_metrics"] = True  # from here on the result exists: a claim is not given back
        quality["benchmark_tries"] = scratch.get("benchmark_tries", [])
        return {
            "n": int(e["n"]),
            "e_fraction": float(e["e_fraction"]),
            "errors": int(e["errors"]),
            "top1_accuracy": float(e["top1_accuracy"]),
            "top5_accuracy": float(e["top5_accuracy"]),
            "macro_precision": float(e["macro_precision"]),
            "macro_recall": float(e["macro_recall"]),
            "macro_f1": float(e["macro_f1"]),
            "per_class": per_class_metrics(e["per_class"]),
            "parity_vs_torch_fp32": (bench.get("quality") or {}).get("parity_vs_torch_fp32"),
            "torch_top1_error_fp32": float((ref.argmax(1) != labels).float().mean()),
            "torch_top1_error_simulated": float((simulated.argmax(1) != labels).float().mean()),
            "bench_version": bench["bench_version"],
            "split_fingerprint": split.fingerprint,
        }


def final_test(
    manifest: Mapping[str, Any] | FinalBuildResult,
    artifact_dir: Path,
    config: FinalConfig,
    access: TestAccess,
    work_dir: Path,
    *,
    device: str = "cuda",
    gpu_guard: Callable[[], list[dict[str, str]]] | None = None,
) -> FinalTestResult:
    """Pass 2: E, top-5, per-class metrics and PyTorch parity of an already built and verified final engine.

    ``manifest`` is ``FinalBuildResult.to_dict()`` (or the result). The SHA-256 of the engine and of the two reference
    networks is checked against it before anything is loaded; ``access.ledger`` claims the engine before the test is
    measured, so an artifact is evaluated once (a claim is given back only if the measurement died with no result).
    """
    data = manifest.to_dict() if isinstance(manifest, FinalBuildResult) else manifest
    tester = FinalTester(config, device=device, work_root=Path(work_dir) / "work", gpu_guard=gpu_guard)
    return tester.run(data, Path(artifact_dir), access)
