"""Wiring factory of the final phase: joins ``operation.final_stage`` to the two-pass final evaluation of the science front.

Like ``demo_wiring``, it is the one module that imports both ``operation`` and ``model_pipeline``. Usage::

    cnnlab-tcc final prepare --factory cnnlab.tcc.final_wiring:final_wiring ...

The factory is called with the plan's ``factory_args`` plus ``"stage": "final"`` and ``"finalist_list_id"``. The data
paths come from ``cifar_root`` / ``work_dir`` among them or from ``CNNLAB_TCC_CIFAR_ROOT`` / ``CNNLAB_TCC_WORK_DIR``.

* **Pass 1** (``evaluator``): ``final_build`` retrains the genome from scratch, compresses it, builds the engine three
  times and writes the artifacts under ``<artifact root>/runs/<run>/attempts/<attempt>/``. It is given no access to the
  official test. The result becomes an ordinary ``EvaluationResult`` (E on validation, medians of L, V and S, the artifacts
  by kind) and the manifest ``final_test`` needs is kept in ``quality["final"]["manifest"]``.
* **Pass 2** (``test_evaluator``): ``final_test`` verifies the files against that manifest, claims the engine through the
  operation's ledger, opens the official test through the data-access barrier and measures it in the engine.

Nothing here is imported until the factory is called, so the operation tests do not need torch. The conversion of a build
manifest into a result is a plain function (:func:`evaluation_result_of`) tested without the science front.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cnnlab.tcc.contracts import EvaluationRequest, EvaluationResult, Fitness
from cnnlab.tcc.operation.demo import DemoRunSpec
from cnnlab.tcc.operation.final_stage import FinalStageError, FinalWiring, TestAlreadyOpened, TestRequest
from cnnlab.tcc.storage.artifacts import attempt_artifact_dir

PATH_ENVIRONMENT = {"cifar_root": "CNNLAB_TCC_CIFAR_ROOT", "work_dir": "CNNLAB_TCC_WORK_DIR"}
SPEC_ONLY_ARGS = ("stage", "finalist_list_id")  # carried to this factory, not to the search wiring


def artifact_dir_of(artifact_root: Path, run_id: str, attempt_id: str) -> tuple[Path, str]:
    """(directory, label) such that ``final_build`` writes into ``<root>/runs/<run>/attempts/<attempt>``."""
    relative = PurePosixPath(attempt_artifact_dir(run_id, attempt_id))
    return Path(artifact_root) / Path(*relative.parent.parts), relative.name


def evaluation_result_of(request: EvaluationRequest, manifest: Mapping[str, Any]) -> EvaluationResult:
    """The ``EvaluationResult`` of a ``final_build`` manifest (``FinalBuildResult.to_dict()``)."""
    status = manifest["status"]
    common = dict(
        run_id=request.run_id,
        candidate_id=request.candidate_id,
        attempt_id=request.attempt_id,
        gpu_seconds=float(manifest.get("gpu_seconds") or 0.0),
        durations_s=dict(manifest.get("durations_s") or {}),
        failure_stage=manifest.get("failure_stage"),
        failure_code=manifest.get("failure_code"),
        failure_detail=manifest.get("failure_detail"),
    )
    pipeline = dict(manifest.get("quality") or {})
    if status != "valid":
        return EvaluationResult(
            status=status,
            fitness=None,
            quality={"synthetic": False, "protocol_stage": "final", "pipeline": pipeline},
            **common,
        )  # type: ignore[arg-type]
    prefix = PurePosixPath(attempt_artifact_dir(request.run_id, request.attempt_id)).parent
    builds = list(manifest.get("builds") or [])
    return EvaluationResult(
        status="valid",
        fitness=Fitness(
            e=float(manifest["e_val"]),
            l_ms=float(manifest["l_ms"]),
            v_mb=float(manifest["v_mb"]),
            s_mb=float(manifest["s_mb"]),
        ),
        artifact_paths={kind: str(prefix / rel) for kind, rel in dict(manifest["artifact_paths"]).items()},
        artifact_sha256=dict(manifest["artifact_sha256"]),
        quality={
            "synthetic": False,
            "protocol_stage": "final",
            "pipeline": pipeline,
            "measurement": {"bench_version": builds[0]["bench_version"] if builds else None},
            "final": {"builds": builds, "chosen_build": manifest.get("chosen_build"), "manifest": dict(manifest)},
        },
        **common,  # type: ignore[arg-type]
    )


def metrics_of_test_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """What the operation records for a ``final_test`` result: its metrics, plus the keys the barrier checks."""
    if result["status"] != "valid" or not result.get("test"):
        raise FinalStageError(
            f"the test pass ended {result['status']} ({result.get('failure_code')}: {result.get('failure_detail')}); "
            "no result was recorded"
        )
    test = dict(result["test"])
    return {**test, "n_examples": test["n"], "engine_sha256": result["engine_sha256"]}


class _Science:
    """The science front, imported when first needed (torch and TensorRT live behind it)."""

    def __init__(self) -> None:
        from cnnlab.tcc.model_pipeline import final_eval
        from cnnlab.tcc.model_pipeline.data_access import TorchvisionCifar100

        self.final_eval, self.TorchvisionCifar100 = final_eval, TorchvisionCifar100


class FinalBuildEvaluator:
    """Pass 1 as a ``contracts.Evaluator``: stage ``finalist``, no test access."""

    def __init__(
        self,
        science: Any,
        cifar_root: Path,
        work_dir: Path,
        finalist_list_id: str,
        train_time_limit_s: float | None,
        device: str = "cuda",
    ) -> None:
        self._science, self._cifar_root, self._work, self._list = (
            science,
            Path(cifar_root),
            Path(work_dir),
            finalist_list_id,
        )
        self._limit, self._device = train_time_limit_s, device
        self._source: Any = None
        self._data: Any = None

    def _prepared(self, run_id: str) -> Any:
        if self._data is None:
            import torch

            fe = self._science.final_eval
            self._source = self._science.TorchvisionCifar100(self._cifar_root)
            grant = fe.final_grant(self._source, run_id, self._list)
            self._data = fe.prepare_final_data(grant, self._source, torch.device(self._device))
        return self._data

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if request.stage != "finalist":
            raise FinalStageError(f"the final evaluator takes finalist requests, not {request.stage!r}")
        fe = self._science.final_eval
        directory, label = artifact_dir_of(request.artifact_root, request.run_id, request.attempt_id)
        config = fe.FinalConfig(train_time_limit_s=self._limit)
        built = fe.final_build(
            request.genome.expressed_genome,
            request.train_seed,
            config,
            self._prepared(request.run_id),
            self._work,
            directory,
            label=label,
            device=self._device,
        )
        return evaluation_result_of(request, built.to_dict())


class _LedgerAdapter:
    """The operation's ledger under the science front's contract (its own ``TestAlreadyEvaluated``)."""

    def __init__(self, ledger: Any, already_evaluated: type[Exception]) -> None:
        self._ledger, self._error = ledger, already_evaluated

    def claim(self, engine_sha256: str) -> None:
        try:
            self._ledger.claim(engine_sha256)
        except TestAlreadyOpened as exc:
            raise self._error(str(exc)) from exc

    def release(self, engine_sha256: str) -> None:
        self._ledger.release(engine_sha256)


class FinalTestEvaluator:
    """Pass 2 as a ``final_stage.TestEvaluator``: the official test, once per engine, through the barrier."""

    __test__ = False

    def __init__(
        self, science: Any, cifar_root: Path, work_dir: Path, train_time_limit_s: float | None, device: str = "cuda"
    ) -> None:
        self._science, self._cifar_root, self._work, self._limit, self._device = (
            science,
            Path(cifar_root),
            Path(work_dir),
            train_time_limit_s,
            device,
        )

    def evaluate_test(self, request: TestRequest) -> Mapping[str, Any]:
        fe = self._science.final_eval
        source = self._science.TorchvisionCifar100(self._cifar_root)
        grant = fe.final_grant(source, request.run_id, request.finalist_list_id)
        access = fe.TestAccess(grant, source, _LedgerAdapter(request.ledger, fe.TestAlreadyEvaluated))
        if not request.build_manifest:
            raise FinalStageError(f"{request.run_id}: the build pass stored no manifest for this engine")
        directory, _ = artifact_dir_of(request.artifact_root, request.run_id, request.attempt_id)
        done = fe.final_test(
            request.build_manifest,
            directory,
            fe.FinalConfig(train_time_limit_s=self._limit),
            access,
            self._work,
            device=self._device,
        )
        return metrics_of_test_result(done.to_dict())


def final_train_limit(search_limit_s: float | None, search_epochs: int | None, final_epochs: int) -> float | None:
    """Training time limit of a finalist: the plan's limit scaled by the ratio of epochs.

    Amendment of 2026-10-24 (TCC-01 does not fix a limit for the retraining): the plan's ``train_time_limit_s`` was
    set by the pilot for one search training (``search_epochs``, 5); a finalist trains ``final_epochs`` (30) from
    scratch, so the same absolute cap stopped a legitimate 30-epoch retraining. The cap keeps its purpose, catching a
    runaway training, at the same rate per epoch.
    """
    if search_limit_s is None:
        return None
    if not search_epochs or int(search_epochs) <= 0:
        raise FinalStageError("the plan's limit cannot be scaled without the search epochs (factory_args.epochs)")
    return float(search_limit_s) * final_epochs / int(search_epochs)


def final_wiring(spec: DemoRunSpec) -> FinalWiring:
    """Factory: the final evaluator, the test evaluator, the cache context of a 30-epoch training and the plan's data."""
    from cnnlab.tcc import demo_wiring
    from cnnlab.tcc.model_pipeline.evaluator import pilot_cache_context

    args = dict(spec.factory_args)
    if args.get("stage") != "final" or not args.get("finalist_list_id"):
        raise FinalStageError("final_wiring needs the plan's factory_args plus stage='final' and finalist_list_id")
    list_id = str(args.pop("finalist_list_id"))
    args.pop("stage")
    paths = {k: str(args.get(k) or os.environ.get(env, "")) for k, env in PATH_ENVIRONMENT.items()}
    missing = [f"{k} (or ${PATH_ENVIRONMENT[k]})" for k, v in paths.items() if not v]
    if missing:
        raise FinalStageError(f"final wiring: missing {missing}")
    # The plan's data and space, exactly as the search wiring built them: same factory arguments, same hashes.
    from cnnlab.tcc.model_pipeline.pilot_wiring import build_pilot_science

    science = build_pilot_science({k: v for k, v in args.items() if k in _search_keys()})
    science_front = _Science()
    limit = final_train_limit(args.get("train_time_limit_s"), args.get("epochs"), science_front.final_eval.FINAL_EPOCHS)
    config = science_front.final_eval.FinalConfig(train_time_limit_s=limit)

    def context_for(job: Any) -> Any:
        return pilot_cache_context(
            science.data,
            config.evaluator_config(),
            job.train_seed,
            science.code_version,
            science.space_hash,
            dict(science.environment["backend"]),
        )

    from cnnlab.tcc.operation.demo import DatasetRegistration
    from cnnlab.tcc.storage.sqlite_store import SplitDescriptor

    cifar, work = Path(paths["cifar_root"]), Path(paths["work_dir"])
    return FinalWiring(
        evaluator=FinalBuildEvaluator(science_front, cifar, work, list_id, limit),
        test_evaluator=FinalTestEvaluator(science_front, cifar, work, limit),
        context_for=context_for,
        dataset=DatasetRegistration(
            science.dataset_id,
            science.dataset_manifest,
            {name: SplitDescriptor(**d) for name, d in science.splits.items()},
        ),
        space_hash=science.space_hash,
        evaluator_label=f"{science.evaluator_label} | final {config.definition()['version']}",
        synthetic=False,
        environment={**science.environment, "final": config.definition()},
    )


def _search_keys() -> set[str]:
    """The arguments the search wiring understands (the plan's others are for the runner, the sentinel and the methods)."""
    from cnnlab.tcc.model_pipeline.pilot_wiring import DEFAULTS, REQUIRED

    return set(DEFAULTS) | set(REQUIRED)


__all__ = [
    "final_wiring",
    "final_train_limit",
    "evaluation_result_of",
    "metrics_of_test_result",
    "FinalBuildEvaluator",
    "FinalTestEvaluator",
]
