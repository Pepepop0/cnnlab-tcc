"""Science side of the pilot demo, in plain types: runner, evaluator and dataset identity.

:func:`build_pilot_science` is what a wiring factory of the operation front
needs (``DemoRunSpec.factory_args`` -> runner, evaluator, split fingerprints,
space hash, labels, environment). It imports neither ``operation`` nor
``storage`` nor SQLite, so the dependency keeps pointing from operation to
science. Building it reads only the official **train** file on the CPU (to
derive the index sets and their fingerprints); the GPU is first touched by
the evaluator, at the first evaluation, after its exclusivity check.

Scope is fixed here and repeated in every label: pilot stage, FP32 unpruned
slice, class-balanced subset of the 45k train split, preliminary L/V. The
official test split is described by the fingerprint of its index set only.
``train_per_class=None`` trains on the whole 45k split (label ``train-full-45k``
instead of ``train-subset``) and ``generations=None`` leaves the search without a
generation limit, so only the GPU budget ends the run (campaign settings).

Machine paths. ``cifar_root`` and ``work_dir`` may be given as arguments or,
when absent, through the environment variables ``CNNLAB_TCC_CIFAR_ROOT`` and
``CNNLAB_TCC_WORK_DIR``. The operation front copies ``factory_args`` into the
frozen plan, so the environment route keeps absolute paths out of the plan
and of its hash. Paths never enter the dataset identity or the cache context.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cnnlab.tcc.contracts import TCCJobSpec
from cnnlab.tcc.model_pipeline.bench import DEFAULT_ITERS, DEFAULT_WARMUP
from cnnlab.tcc.model_pipeline.data_access import OFFICIAL_SIZES, TorchvisionCifar100, index_fingerprint
from cnnlab.tcc.model_pipeline.demo_data import DemoData, prepare_demo_data
from cnnlab.tcc.model_pipeline.evaluator import (
    EVALUATOR_VERSION,
    EngineEvaluator,
    EvaluatorConfig,
    backend_manifest,
    other_gpu_python_jobs,
    pilot_cache_context,
)
from cnnlab.tcc.model_pipeline.one_candidate import git_revision
from cnnlab.tcc.model_pipeline.splits import DemoSplitSpec
from cnnlab.tcc.model_pipeline.trainer import TrainConfig
from cnnlab.tcc.science.operators import SLICES
from cnnlab.tcc.science.record import CacheContext
from cnnlab.tcc.science.search import RUNNER_VERSION, SearchConfig, SearchRunner

WIRING_VERSION = "tcc-pilot-wiring/v1"
SCOPE_LABELS: tuple[str, ...] = ("pilot", "fp32-unpruned-slice", "train-subset", "preliminary-L-V", "not-confirmatory")
SLICE_LABELS: dict[str, str] = {
    "fp32": "fp32-unpruned-slice",
    "fp16_block": "fp16-per-block-unpruned-slice",
    "fp16_regions": "fp16-per-region-unpruned-slice",
    "bf16_regions": "bf16-per-region-unpruned-slice",
    "int8_ptq": "int8-ptq-per-region-unpruned-slice",
    "int8_qat": "int8-qat-per-region-unpruned-slice",
    "prune_fp32": "fp32-pruned-slice",
    "compressed": "fp32-fp16-int8-pruned-slice",
    "admitted": "admitted-space-slice",
    "architecture": "fp32-architecture-slice",
}
FULL_TRAIN_LABEL = "train-full-45k"


def scope_labels(slice_name: str, full_train: bool = False) -> tuple[str, ...]:
    """Labels repeated in plan, environment and evidence; the slice and the training data vary."""
    swap = {SLICE_LABELS["fp32"]: SLICE_LABELS[slice_name]}
    if full_train:
        swap["train-subset"] = FULL_TRAIN_LABEL
    return tuple(swap.get(label, label) for label in SCOPE_LABELS)


DEFAULTS: dict[str, Any] = {
    "train_per_class": 100,
    "subset_seed": 20260929,
    "epochs": 5,
    "population": 4,
    "generations": 2,
    "bench_warmup": DEFAULT_WARMUP,
    "bench_iters": DEFAULT_ITERS,
    "gpu_guard": True,
    "gpu_wait_s": 180,
    "device": "cuda",  # "cpu" exists for tests of the wiring; measurements need the GPU
    "deterministic_training": False,
    "train_time_limit_s": None,  # longest training of one attempt, in seconds; None: no limit
    "slice": "fp32",  # search slice: "fp32" or "fp16_block"
    "code_version": None,  # None: read from git at build time
}
REQUIRED: tuple[str, ...] = ("cifar_root", "work_dir")
PATH_ENVIRONMENT: dict[str, str] = {"cifar_root": "CNNLAB_TCC_CIFAR_ROOT", "work_dir": "CNNLAB_TCC_WORK_DIR"}


@dataclass(frozen=True)
class PilotScience:
    """Everything the science front contributes to one pilot run."""

    runner: SearchRunner
    evaluator: EngineEvaluator
    data: DemoData
    dataset_id: str
    dataset_manifest: dict[str, Any]
    splits: dict[str, dict[str, Any]]  # name -> {"sha256", "n_examples", "ref"}
    space_hash: str
    runner_label: str
    evaluator_label: str
    environment: dict[str, Any]
    search_config: SearchConfig
    evaluator_config: EvaluatorConfig
    code_version: str
    arguments: dict[str, Any]
    synthetic: bool = False
    slice_name: str = "fp32"
    scope: tuple[str, ...] = SCOPE_LABELS

    def context_for(self, job: TCCJobSpec) -> CacheContext:
        return pilot_cache_context(
            self.data,
            self.evaluator_config,
            job.train_seed,
            self.code_version,
            self.space_hash,
            dict(self.environment["backend"]),
        )


def resolve_arguments(args: Mapping[str, Any]) -> dict[str, Any]:
    """Defaults filled in, unknown or missing keys refused (a typo must not become a silent default)."""
    unknown = set(args) - set(DEFAULTS) - set(REQUIRED)
    paths = {k: args.get(k) or os.environ.get(PATH_ENVIRONMENT[k]) for k in REQUIRED}
    missing = [f"{k} (or ${PATH_ENVIRONMENT[k]})" for k in REQUIRED if not paths[k]]
    if unknown or missing:
        raise ValueError(f"pilot wiring arguments: unknown {sorted(unknown)}, missing {missing}")
    merged = {**DEFAULTS, **dict(args), **paths}
    for name in (
        "train_per_class",
        "subset_seed",
        "epochs",
        "population",
        "generations",
        "bench_warmup",
        "bench_iters",
        "gpu_wait_s",
    ):
        if merged[name] is None and name in ("train_per_class", "generations"):
            continue  # None: the full 45k split / no generation limit
        if isinstance(merged[name], bool) or not isinstance(merged[name], int):
            raise ValueError(f"{name} must be an int")
    if not isinstance(merged["gpu_guard"], bool) or not isinstance(merged["deterministic_training"], bool):
        raise ValueError("gpu_guard and deterministic_training must be bool")
    limit = merged["train_time_limit_s"]
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float)) or not limit > 0):
        raise ValueError("train_time_limit_s must be None or a number greater than 0")
    if merged["device"] not in ("cuda", "cpu"):
        raise ValueError("device must be 'cuda' or 'cpu'")
    if merged["slice"] not in SLICES:
        raise ValueError(f"slice must be one of {sorted(SLICES)}")
    return merged


def split_descriptors(data: DemoData) -> dict[str, dict[str, Any]]:
    """The four registered splits; ``test`` is the fingerprint of its index set, the file is never opened."""
    fp = data.manifest["fingerprints"]
    n_train = data.manifest["n_train_subset"]
    full = bool(data.manifest.get("train_is_full_split"))
    train_ref = "full 45k train split (no subset)" if full else "demo subset of the 45k train split"
    return {
        "train": {"sha256": fp["train"], "n_examples": n_train, "ref": train_ref},
        "val": {"sha256": fp["val"], "n_examples": data.manifest["n_val"], "ref": "poc 5k validation split"},
        "test": {
            "sha256": index_fingerprint("test", range(OFFICIAL_SIZES["test"])),
            "n_examples": OFFICIAL_SIZES["test"],
            "ref": "official test file, sealed; indices only",
        },
        # FP32 slice: no PTQ, so no calibration set is used. The schema wants four splits; the
        # descriptor repeats the training subset instead of inventing a set that nothing reads.
        "calibration": {
            "sha256": fp["train"],
            "n_examples": n_train,
            "ref": "unused in the fp32 slice (no PTQ); same as the train split used",
        },
    }


def build_pilot_science(args: Mapping[str, Any]) -> PilotScience:
    """Build runner and evaluator for one pilot run from JSON-like arguments."""
    merged = resolve_arguments(args)
    work = Path(merged["work_dir"])
    spec = DemoSplitSpec(train_per_class=merged["train_per_class"], subset_seed=merged["subset_seed"])
    data = prepare_demo_data(TorchvisionCifar100(Path(merged["cifar_root"])), spec, "pilot-wiring", torch.device("cpu"))
    evaluator_config = EvaluatorConfig(
        train=TrainConfig(epochs=merged["epochs"], deterministic=merged["deterministic_training"]),
        bench_warmup=merged["bench_warmup"],
        bench_iters=merged["bench_iters"],
        gpu_wait_s=float(merged["gpu_wait_s"]),
        train_time_limit_s=(None if merged["train_time_limit_s"] is None else float(merged["train_time_limit_s"])),
    )
    search_config = SearchConfig(population=merged["population"], generations=merged["generations"])
    code_version = merged["code_version"] or git_revision()
    backend = backend_manifest()
    space = SLICES[merged["slice"]]
    full_train = merged["train_per_class"] is None
    scope = scope_labels(merged["slice"], full_train)
    space_hash = space.space_hash()

    def context_for(job: TCCJobSpec) -> CacheContext:
        return pilot_cache_context(data, evaluator_config, job.train_seed, code_version, space_hash, backend)

    evaluator = EngineEvaluator(
        data,
        work / "val.npz",
        evaluator_config,
        device=merged["device"],
        work_root=work / "attempts",
        gpu_guard=other_gpu_python_jobs if merged["gpu_guard"] else None,
    )
    recorded = {k: v for k, v in merged.items() if k not in REQUIRED}  # machine paths are not identity
    return PilotScience(
        runner=SearchRunner(search_config, space, context_for),
        evaluator=evaluator,
        data=data,
        dataset_id=f"cifar100-pilot-{data.fingerprint[:16]}",
        dataset_manifest=dict(data.manifest),
        splits=split_descriptors(data),
        space_hash=space_hash,
        runner_label=(
            f"nsga2-joint/{RUNNER_VERSION} pop={search_config.population} "
            f"gen={'unlimited' if search_config.generations is None else search_config.generations} "
            f"slice={merged['slice']}"
        ),
        evaluator_label=(
            f"{EVALUATOR_VERSION} S=.plan bytes; L,V preliminary; "
            f"train {'full split' if full_train else 'subset'} {data.manifest['n_train_subset']}"
        ),
        environment={
            "backend": backend,
            "scope": list(scope),
            "wiring": WIRING_VERSION,
            "search": search_config.definition(),
            "evaluator": evaluator_config.definition(),
            "code_version": code_version,
            "slice": merged["slice"],
            "space_slice": space.definition()["reason"],
        },
        search_config=search_config,
        evaluator_config=evaluator_config,
        code_version=code_version,
        arguments=recorded,
        slice_name=merged["slice"],
        scope=scope,
    )
