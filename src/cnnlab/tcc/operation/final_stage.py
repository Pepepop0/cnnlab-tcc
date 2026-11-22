"""The final phase: retrain the finalists and open the official test, once per engine (TCC-01 section 6; task A12).

Operation side only. The science front (G9) supplies two objects through a factory:

* an **Evaluator** (the ordinary ``contracts.Evaluator``, called with ``stage="finalist"``) that trains the genome from
  scratch for 30 epochs with the block's retrain seed, compresses it, exports it and builds the engine three times, and
  answers with the validation error and the *medians* of L, V and S, the engine as an ``engine`` artifact, and in
  ``quality`` the three builds (``quality["final"]["builds"]``) and ``quality["measurement"]["bench_version"]``;
* a **TestEvaluator** that runs one committed engine on the official test split and answers with its top-1 accuracy.
  It is called only through the barrier below, with a ``ledger`` it must ``claim`` the engine through *right before*
  it reads the test (and ``release`` only if the measurement died with no result), and with the manifest the build
  pass stored (``quality["final"]["manifest"]``).

**One retraining is one run** (``stage="finalist"``, the frozen list in ``runs.finalist_list_id``) with the genome of the
selection copied from its confirmatory run, the block's retrain seed (``plan.blocks[b].retrain_seed``) and the factory
arguments of the campaign plus ``"stage": "final"``. The unit of retraining is ``(seed block, expressed-genome hash)``: a
genome chosen by several roles or methods of one block is trained and tested once, and every selection that chose it
reads that result. At most 72 units; the code checks it.

**Two passes, two jobs per run** (each is a job of the queue, so one GPU job at a time, with lease and cancel):

1. ``build``: train, compress, build, measure. Nothing here reads the official test.
2. ``test``: opens the official test for the committed engine. It runs only when
   * the list is frozen, its stored hash is the hash of its selections *and* the one the operator confirmed
     (``--list-sha256``), and the plan it belongs to is frozen and intact;
   * **every** unit of the list has a valid engine on disk with the bytes the catalog records, and the manifest of those
     engines has the hash the operator confirmed (``--artifacts-sha256``): the artifacts are frozen before the test is
     opened for any of them;
   * the engine has not been opened before. The ``ledger`` marks the opening with a file created exclusively *before*
     the test is read; a mark without a recorded result (the process died in between) is refused, never repeated,
     because the test may already have been seen. The database refuses a second opening of an engine as well
     (``test_evaluations``).

**After the artifacts are frozen no engine changes.** A technical failure of a build can be repeated (``retry``) only
before the test jobs are queued; afterwards the list, the engines and their manifest hash are what they are.

``export_final`` writes the CSV that ``campaign inference`` reads (one row per selection, with the result of its unit),
with the bench version and the hashes, and a manifest.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from cnnlab.tcc.contracts import EvaluationRequest, Evaluator, GenomeRecord, TCCJobSpec
from cnnlab.tcc.operation import bench_versions
from cnnlab.tcc.operation.demo import (
    PROTOCOL_DOC,
    CodeVersionMismatchError,
    DatasetRegistration,
    DemoRunSpec,
    DemoWiring,
    StartResult,
    _work,
    environment_manifest,
    load_factory,
    require_real_wiring,
)
from cnnlab.tcc.operation.provenance import CodeVersion, capture_code_version, default_repo_root
from cnnlab.tcc.operation.service import RunnerContext, RunnerStatus
from cnnlab.tcc.science.record import CacheContext, cache_context_hash
from cnnlab.tcc.storage.canonical import canonical_sha256, genome_from_payload, sha256_hex
from cnnlab.tcc.storage.errors import ConsistencyError, NotFoundError, OfficialTestAccessError, StorageError
from cnnlab.tcc.storage.jobs import JobQueue
from cnnlab.tcc.storage.readmodel import open_read_model
from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore

FINAL_FORMAT = "tcc-final-stage/1"
MANIFEST_NAME = "manifesto.json"
ARTIFACTS_MANIFEST_NAME = "artefatos_finais.json"
MARKS_DIR = "abertura-do-teste"
SYNTHETIC_SUFFIX = "_SINTETICO"
MAX_RETRAINS = 72
BUILD, TEST = "build", "test"
MODES = (BUILD, TEST)
DEFAULT_BUDGET_GPU_S = 14_400.0  # a ceiling for one retraining; the run is one attempt, which is never cut
FINAL_STAGE_ARG = {"stage": "final"}
ROLE_ORDER = ("menor_E", "equilibrio", "H1")


class FinalStageError(StorageError):
    """The final phase refused to go on; what was written before stays."""


# -- the interface of the science front (G9) -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TestRequest:
    """Everything the test evaluator is given to open the official test for one committed engine."""

    __test__ = False  # not a pytest class

    run_id: str
    attempt_id: str
    candidate_id: str
    retrain_id: str
    engine_path: Path
    engine_sha256: str
    finalist_list_id: str
    list_sha256: str
    splits: Mapping[str, Any]  # name -> SplitDescriptor; ``test`` is present because the store released it
    artifact_root: Path
    ledger: Any = None  # ``claim(engine_sha256)`` before the test is read, ``release`` if it died without a result
    build_manifest: Mapping[str, Any] = field(default_factory=dict)  # what the build pass stored for this engine


class TestEvaluator(Protocol):
    __test__ = False

    def evaluate_test(self, request: TestRequest) -> Mapping[str, Any]:
        """Open the test split for ``request.engine_path`` once and return ``{"top1_accuracy", "n_examples",
        "engine_sha256", ...}`` (top-1 as a fraction; the SHA-256 of the engine file actually used)."""
        ...


@dataclass(frozen=True, slots=True)
class FinalWiring:
    """What the final factory returns (``factory(DemoRunSpec) -> FinalWiring``)."""

    evaluator: Evaluator
    test_evaluator: TestEvaluator
    context_for: Callable[[TCCJobSpec], CacheContext]  # the cache context of a final training (30 epochs, retrain seed)
    dataset: DatasetRegistration
    space_hash: str
    evaluator_label: str
    synthetic: bool
    environment: Mapping[str, Any] = field(default_factory=dict)


FinalFactory = Callable[[DemoRunSpec], FinalWiring]


class TestAlreadyOpened(OfficialTestAccessError):
    """The official test was already opened for this engine (or may have been: the process died after the claim)."""

    __test__ = False


class MarkLedger:
    """The once-only ledger of the official test: an exclusive file per engine next to the database.

    ``claim`` creates ``<dir>/<engine sha>.json`` exclusively and refuses when the file exists or when the database
    already holds a test result for the engine; ``release`` removes the mark of a measurement that died with no result.
    """

    def __init__(self, directory: Path, run_id: str, list_sha256: str, opened: Callable[[], set[str]]) -> None:
        self.directory, self.run_id, self.list_sha256, self._opened = Path(directory), run_id, list_sha256, opened
        self.claims: list[str] = []

    def _path(self, engine_sha256: str) -> Path:
        return self.directory / f"{engine_sha256}.json"

    def claim(self, engine_sha256: str) -> None:
        if engine_sha256 in self._opened():
            raise TestAlreadyOpened(f"engine {engine_sha256[:12]}… was already opened on the official test")
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self._path(engine_sha256), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise TestAlreadyOpened(
                f"the test was already opened for engine {engine_sha256[:12]}… (mark {self._path(engine_sha256).name}) but no result "
                "was recorded; it is not repeated. Decide by hand what happened"
            ) from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "run_id": self.run_id,
                        "engine_sha256": engine_sha256,
                        "list_sha256": self.list_sha256,
                        "opened_at": datetime.now(UTC).isoformat(),
                    },
                    sort_keys=True,
                )
            )
            fh.flush()
            os.fsync(fh.fileno())
        self.claims.append(engine_sha256)

    def release(self, engine_sha256: str) -> None:
        if engine_sha256 in self.claims:
            self.claims.remove(engine_sha256)
            self._path(engine_sha256).unlink(missing_ok=True)


# -- the list and its retraining units -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Unit:
    """One retraining: a genome of a seed block and every selection that chose it."""

    retrain_id: str
    seed_block: int
    phenotype_hash: str
    selections: tuple[Mapping[str, Any], ...]  # as the read model gives them, in selection-id order

    @property
    def run_id(self) -> str:
        return f"final-{self.retrain_id}"

    @property
    def source(self) -> Mapping[str, Any]:
        """The selection whose confirmatory candidate is copied: the first in role, then method, order."""
        return sorted(self.selections, key=lambda s: (ROLE_ORDER.index(s["role"]), s["method"], s["selection_id"]))[0]

    def job_id(self, mode: str, n: int = 1) -> str:
        return f"job-final-{mode}-{self.run_id}" + ("" if n == 1 else f"-r{n}")


def units_of(listing: Mapping[str, Any]) -> list[Unit]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for s in listing["selections"]:
        groups[(int(s["seed_block"]), s["phenotype_hash"])].append(s)
    units = [
        Unit(f"rt-b{block}-{digest[:12]}", block, digest, tuple(sorted(members, key=lambda s: s["selection_id"])))
        for (block, digest), members in sorted(groups.items())
    ]
    if len(units) > MAX_RETRAINS:
        raise FinalStageError(f"the list has {len(units)} retrainings; the protocol allows at most {MAX_RETRAINS}")
    return units


def verify_list(rm: Any, list_id: str, expected_sha256: str) -> Mapping[str, Any]:
    """The frozen list, after the checks that make it the authorization to go on; refused otherwise."""
    listing = rm.finalist_list(list_id)
    if not listing["hash_ok"]:
        raise FinalStageError(f"finalist list {list_id!r}: its stored hash is not the hash of its selections")
    if listing["list_sha256"] != expected_sha256:
        raise FinalStageError(
            f"finalist list {list_id!r} has hash {listing['list_sha256'][:16]}…, not the {expected_sha256[:16]}… given: "
            "it is not the list that was frozen and checked"
        )
    plan = rm.plan(listing["plan_id"])
    if plan["status"] != "frozen" or not plan["hash_ok"]:
        raise FinalStageError(f"plan {listing['plan_id']!r} of the list is not frozen and intact")
    return listing


# -- preparing the retrainings -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedFinal:
    unit: Unit
    run_id: str
    job_id: str
    created: bool


def final_spec(
    plan_id: str,
    plan_content: Mapping[str, Any],
    unit: Unit,
    factory: str,
    author: str,
    budget_gpu_s: float,
    list_id: str,
) -> DemoRunSpec:
    """The spec the final factory is asked about: the plan's arguments and slice, plus ``"stage": "final"``."""
    block = next(b for b in plan_content["blocks"] if b["seed_block"] == unit.seed_block)
    return DemoRunSpec(
        method=unit.source["method"],
        seed_block=unit.seed_block,
        search_seed=block["search_seed"],
        train_seed=block["retrain_seed"],
        calibration_seed=block["calibration_seed"],
        budget_gpu_s=budget_gpu_s,
        code_label=plan_content["code_freeze"]["label"],
        author=author,
        run_id=unit.run_id,
        factory=factory,
        factory_args={**plan_content["factory_args"], **FINAL_STAGE_ARG, "finalist_list_id": list_id},
        stage="confirmatory",
        plan_id=plan_id,
    )


def load_final(
    db_path: Path,
    list_id: str,
    list_sha256: str,
    factory_path: str,
    *,
    repo_root: Path | None = None,
    author: str = "Pedro Antunes",
    budget_gpu_s: float = DEFAULT_BUDGET_GPU_S,
) -> tuple[FinalWiring, CodeVersion, Path]:
    """The final wiring from its factory, the code version of the checkout, and the repository root."""
    repo = Path(repo_root) if repo_root is not None else default_repo_root()
    with open_read_model(Path(db_path)) as rm:
        listing = verify_list(rm, list_id, list_sha256)
        plan = rm.plan(listing["plan_id"])
    unit = units_of(listing)[0]
    spec = final_spec(plan["plan_id"], plan["content"], unit, factory_path, author, budget_gpu_s, list_id)
    final = load_factory(factory_path)(spec)
    if not isinstance(final, FinalWiring):
        raise FinalStageError(f"{factory_path} returned {type(final).__name__}, not a FinalWiring")
    return final, capture_code_version(plan["content"]["code_freeze"]["label"], repo), repo


def as_demo_wiring(final: FinalWiring, db_path: Path) -> DemoWiring:
    runner = FinalRunner(Path(db_path), final.context_for, final.test_evaluator)
    return DemoWiring(
        runner=runner,
        evaluator=final.evaluator,
        dataset=final.dataset,
        space_hash=final.space_hash,
        runner_label=f"final-stage/{FINAL_FORMAT}",
        evaluator_label=final.evaluator_label,
        synthetic=final.synthetic,
        environment=final.environment,
    )


def _check_plan(plan: Any, code_version: CodeVersion, repo: Path, final: FinalWiring) -> None:
    content = plan.content
    if content.get("format") != "tcc-campaign-plan/1":
        raise FinalStageError(f"plan {plan.plan_id!r} is not a campaign plan")
    if plan.status != "frozen" or content.get("pending"):
        raise FinalStageError(f"plan {plan.plan_id!r} is not frozen and complete")
    frozen = content["code_freeze"]
    if code_version.git_sha != frozen["git_sha"] or bool(code_version.dirty):
        raise CodeVersionMismatchError(
            f"the final phase runs the code the plan froze ({str(frozen['git_sha'])[:12]}), clean; this checkout is "
            f"{str(code_version.git_sha)[:12]}{' with uncommitted changes' if code_version.dirty else ''}"
        )
    if final.space_hash != content["space_hash"]:
        raise ConsistencyError("the search space of the final wiring is not the one frozen in the plan")
    if final.dataset.dataset_id != content["dataset_id"]:
        raise ConsistencyError("the dataset of the final wiring is not the one frozen in the plan")
    doc = repo / PROTOCOL_DOC
    if not doc.is_file() or f"tcc01-{sha256_hex(doc.read_bytes())[:12]}" != content["protocol_id"]:
        raise ConsistencyError("the protocol document is not the one the plan was made against")


def prepare_builds(
    store: SQLiteRunStore,
    list_id: str,
    list_sha256: str,
    final: FinalWiring,
    code_version: CodeVersion,
    *,
    factory: str,
    repo_root: Path,
    max_attempts: int = 1,
    budget_gpu_s: float = DEFAULT_BUDGET_GPU_S,
    author: str = "Pedro Antunes",
    allow_synthetic: bool = False,
) -> list[PreparedFinal]:
    """Create one finalist run and its build job per retraining unit of the frozen list; safe to repeat."""
    require_real_wiring(_shim(final), allow_synthetic=allow_synthetic)
    if max_attempts < 1:
        raise FinalStageError("max_attempts must be at least 1")
    with open_read_model(store.db_path) as rm:
        listing = verify_list(rm, list_id, list_sha256)
    plan = store.get_plan(listing["plan_id"])
    repo = Path(repo_root)
    _check_plan(plan, code_version, repo, final)
    units = units_of(listing)
    _refuse_after_freeze(store, units, "prepare")
    manifest = environment_manifest(_shim(final), code_version, repo)
    environment_id = f"env-{canonical_sha256(manifest)[:12]}"
    store.register_environment(environment_id, manifest)
    store.register_dataset(final.dataset.dataset_id, final.dataset.manifest, final.dataset.splits)
    queue = JobQueue(store)
    out: list[PreparedFinal] = []
    for unit in units:
        spec = final_spec(plan.plan_id, plan.content, unit, factory, author, budget_gpu_s, list_id)
        job = TCCJobSpec(
            run_id=unit.run_id,
            protocol_id=plan.protocol_id,
            plan_id=plan.plan_id,
            dataset_id=final.dataset.dataset_id,
            environment_id=environment_id,
            space_hash=final.space_hash,
            method=spec.method,
            stage="finalist",
            seed_block=unit.seed_block,
            search_seed=spec.search_seed,
            train_seed=spec.train_seed,
            calibration_seed=spec.calibration_seed,
            budget_gpu_s=budget_gpu_s,
        )
        existing = store.get_run(unit.run_id)
        store.create_run(job, finalist_list_id=list_id)
        job_id = unit.job_id(BUILD)
        try:
            queue.get(job_id)
        except NotFoundError:
            queue.enqueue(
                job_id,
                unit.run_id,
                options=_options(BUILD, unit, listing, code_version, manifest, final)
                | {"max_attempts": max_attempts, "final_spec": spec.to_payload()},
            )
        out.append(PreparedFinal(unit, unit.run_id, job_id, existing is None))
    return out


def _shim(final: FinalWiring) -> DemoWiring:
    return DemoWiring(
        runner=lambda ctx: "completed",
        evaluator=final.evaluator,
        dataset=final.dataset,
        space_hash=final.space_hash,  # type: ignore[arg-type,return-value]
        runner_label=f"final-stage/{FINAL_FORMAT}",
        evaluator_label=final.evaluator_label,
        synthetic=final.synthetic,
        environment=final.environment,
    )


def _options(
    mode: str,
    unit: Unit,
    listing: Mapping[str, Any],
    code_version: CodeVersion,
    manifest: Mapping[str, Any],
    final: FinalWiring,
) -> dict[str, Any]:
    return {
        "purpose": "final-stage",
        "final_mode": mode,
        "retrain_id": unit.retrain_id,
        "selection_ids": [s["selection_id"] for s in unit.selections],
        "list_id": listing["list_id"],
        "list_sha256": listing["list_sha256"],
        "code_version": code_version.to_payload(),
        "system_version": manifest["system"]["readable"],
        "runner": f"final-stage/{FINAL_FORMAT}",
        "evaluator": final.evaluator_label,
        "synthetic_evaluator": final.synthetic,
    }


# -- the artifacts of the builds, frozen before any test is opened -------------------------------------------------


def _unit_state(rm: Any, unit: Unit) -> dict[str, Any]:
    """The valid original attempt of a unit's run and its engine artifact (None where they do not exist yet)."""
    try:
        rows = rm.evaluations(unit.run_id)
    except NotFoundError:
        return {"unit": unit, "attempt": None, "engine": None}
    valid = [r for r in rows if r["status"] == "valid" and not r["cache_hit"]]
    if len(valid) != 1:
        return {"unit": unit, "attempt": None, "engine": None, "valid_attempts": len(valid)}
    attempt = valid[0]
    engines = [
        a for a in rm.artifacts(unit.run_id) if a["attempt_id"] == attempt["attempt_id"] and a["kind"] == "engine"
    ]
    return {"unit": unit, "attempt": attempt, "engine": engines[0] if len(engines) == 1 else None}


def artifacts_manifest(db_path: Path, artifact_root: Path, list_id: str, list_sha256: str) -> dict[str, Any]:
    """Every unit's engine with the bytes on disk checked against the catalog; refused while any is missing or altered.

    ``manifest_sha256`` is the hash the operator confirms before the test is opened.
    """
    with open_read_model(Path(db_path)) as rm:
        listing = verify_list(rm, list_id, list_sha256)
        states = [_unit_state(rm, u) for u in units_of(listing)]
    entries, problems = [], []
    for state in states:
        unit, attempt, engine = state["unit"], state["attempt"], state["engine"]
        if attempt is None or engine is None:
            problems.append(f"{unit.retrain_id}: no single valid attempt with an engine artifact")
            continue
        path = Path(artifact_root) / engine["rel_path"]
        if not path.is_file():
            problems.append(f"{unit.retrain_id}: the engine file {path} is missing")
            continue
        actual = sha256_hex(path.read_bytes())
        if actual != engine["sha256"]:
            problems.append(
                f"{unit.retrain_id}: the engine on disk has SHA-256 {actual[:12]}…, the catalog {engine['sha256'][:12]}…"
            )
            continue
        entries.append(
            {
                "retrain_id": unit.retrain_id,
                "run_id": unit.run_id,
                "seed_block": unit.seed_block,
                "phenotype_hash": unit.phenotype_hash,
                "attempt_id": attempt["attempt_id"],
                "engine_rel_path": engine["rel_path"],
                "engine_sha256": engine["sha256"],
                "engine_bytes": engine["size_bytes"],
                "bench_version": attempt.get("bench_version"),
                "synthetic": bool(attempt["synthetic"]),
                "validation": {
                    "e": attempt["e"],
                    "l_ms": attempt["l_ms"],
                    "v_mb": attempt["v_mb"],
                    "s_mb": attempt["s_mb"],
                },
            }
        )
    if problems:
        raise FinalStageError("the artifacts are not ready to be frozen:\n  " + "\n  ".join(problems))
    body = {
        "format": FINAL_FORMAT,
        "kind": "artifacts",
        "list_id": list_id,
        "list_sha256": list_sha256,
        "engines": entries,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def export_artifacts_manifest(
    db_path: Path, artifact_root: Path, list_id: str, list_sha256: str, dest: Path
) -> dict[str, Any]:
    dest = Path(dest)
    if dest.exists():
        raise FinalStageError(f"destination already exists: {dest} (an export never overwrites)")
    manifest = artifacts_manifest(db_path, artifact_root, list_id, list_sha256)
    dest.mkdir(parents=True)
    (dest / ARTIFACTS_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def enqueue_tests(
    store: SQLiteRunStore,
    list_id: str,
    list_sha256: str,
    artifacts_sha256: str,
    artifact_root: Path,
    final: FinalWiring,
    code_version: CodeVersion,
    repo_root: Path,
) -> list[str]:
    """Queue the test job of every unit, after the artifacts have the hash the operator confirmed."""
    manifest = artifacts_manifest(store.db_path, artifact_root, list_id, list_sha256)
    if manifest["manifest_sha256"] != artifacts_sha256:
        raise FinalStageError(
            f"the engines have manifest hash {manifest['manifest_sha256'][:16]}…, not the {artifacts_sha256[:16]}… given: "
            "the artifacts changed since they were frozen"
        )
    with open_read_model(store.db_path) as rm:
        listing = verify_list(rm, list_id, list_sha256)
        opened = {t["run_id"] for t in rm.test_evaluations()}
    queue = JobQueue(store)
    shim_manifest = environment_manifest(_shim(final), code_version, Path(repo_root))
    queued = []
    for unit in units_of(listing):
        if unit.run_id in opened:
            continue  # its engine was opened already; never again
        job_id = unit.job_id(TEST)
        try:
            queue.get(job_id)
        except NotFoundError:
            queue.enqueue(
                job_id,
                unit.run_id,
                options=_options(TEST, unit, listing, code_version, shim_manifest, final)
                | {"artifacts_sha256": artifacts_sha256},
            )
            queued.append(job_id)
    return queued


def _refuse_after_freeze(store: SQLiteRunStore, units: Sequence[Unit], what: str) -> None:
    """Once any test job exists the artifacts are frozen: no engine of the list changes any more."""
    queue = JobQueue(store)
    for unit in units:
        try:
            queue.get(unit.job_id(TEST))
        except NotFoundError:
            continue
        raise FinalStageError(
            f"cannot {what}: the artifacts were frozen when the test jobs were queued ({unit.job_id(TEST)} exists)"
        )


def requeue_failed(
    store: SQLiteRunStore,
    list_id: str,
    list_sha256: str,
    mode: str,
    *,
    reason: str,
    max_attempts: int | None = None,
) -> list[str]:
    """A new job for every unit whose last job of ``mode`` ended failed or cancelled; the reason is recorded with it.

    This is the justified repetition after a technical failure (TCC-01 section 6): it is explicit, one per call, and
    the reason stays in the job options. ``max_attempts`` raises the number of invalid attempts a build may use.
    """
    if mode not in MODES:
        raise FinalStageError(f"mode must be one of {MODES}")
    if not reason.strip():
        raise FinalStageError("a repetition needs its reason, which is recorded")
    with open_read_model(store.db_path) as rm:
        units = units_of(verify_list(rm, list_id, list_sha256))
    if mode == BUILD:
        _refuse_after_freeze(store, units, "repeat a build")
    queue = JobQueue(store)
    created = []
    for unit in units:
        last = None
        for n in range(1, 50):
            try:
                last = (n, queue.get(unit.job_id(mode, n)))
            except NotFoundError:
                break
        if last is None or last[1].state not in ("failed", "cancelled"):
            continue
        options = dict(last[1].options) | {"requeue_reason": reason, "requeued_from": last[1].job_id}
        if max_attempts is not None:
            options["max_attempts"] = max_attempts
        job_id = unit.job_id(mode, last[0] + 1)
        queue.enqueue(job_id, unit.run_id, options=options)
        created.append(job_id)
    return created


# -- the runner of both passes ----------------------------------------------------------------------------------------


class FinalRunner:
    """The runner of the final phase: the pass is in the job options (``final_mode``)."""

    def __init__(
        self, db_path: Path, context_for: Callable[[TCCJobSpec], CacheContext], test_evaluator: TestEvaluator
    ) -> None:
        self.db_path = Path(db_path)
        self.context_for = context_for
        self.test_evaluator = test_evaluator

    def __call__(self, ctx: RunnerContext) -> RunnerStatus:
        mode = ctx.options.get("final_mode")
        if mode == BUILD:
            return self._build(ctx)
        if mode == TEST:
            return self._test(ctx)
        raise FinalStageError(f"job {ctx.job_id!r} has no final_mode (build or test)")

    # -- build --------------------------------------------------------------------------------------------------
    def _build(self, ctx: RunnerContext) -> RunnerStatus:
        job, options = ctx.job, ctx.options
        with open_read_model(self.db_path) as rm:
            listing = verify_list(rm, str(options["list_id"]), str(options["list_sha256"]))
            self._check_run(rm, job, listing)
            unit = next(u for u in units_of(listing) if u.run_id == job.run_id)
            attempts = rm.evaluations(job.run_id)
            record = genome_from_payload(rm.genome_record(unit.source["source_run_id"], unit.source["candidate_id"]))
        if any(a["status"] == "valid" for a in attempts):
            return "completed"
        invalid = sum(1 for a in attempts if a["status"] == "invalid")
        if invalid >= int(options.get("max_attempts", 1)):
            raise FinalStageError(
                f"{job.run_id}: {invalid} invalid attempt(s) already, the limit is {options.get('max_attempts', 1)}; "
                "a repetition after a technical failure must be justified and allowed explicitly"
            )
        if ctx.should_stop():
            return "stopped"
        context = self.context_for(job)
        final_record = GenomeRecord(
            raw_genome=record.raw_genome,
            expressed_genome=record.expressed_genome,
            repair_log=record.repair_log,
            phenotype_hash=record.phenotype_hash,
            cache_context_hash=cache_context_hash(record.phenotype_hash, context),
            space_hash=record.space_hash,
        )
        candidate_id = f"{job.run_id}.g"
        with open_read_model(self.db_path) as rm:
            try:
                existing = rm.genome_record(job.run_id, candidate_id)
            except NotFoundError:
                existing = None
        if existing is not None and existing["cache_context_hash"] != final_record.cache_context_hash:
            # A justified repetition under another training context (amendment of 2026-10-24: the retraining limit
            # changed, and with it the code): the first candidate stays as recorded; this attempt gets its own.
            candidate_id = f"{job.run_id}.g{len(attempts) + 1}"
        ctx.store.register_candidate(
            job.run_id,
            candidate_id,
            final_record,
            {
                "origin": "final_retrain",
                "generation": 0,
                "retrain_id": unit.retrain_id,
                "source_run_id": unit.source["source_run_id"],
                "source_candidate_id": unit.source["candidate_id"],
                "selections": [
                    {k: s[k] for k in ("selection_id", "role", "method", "source_run_id", "candidate_id")}
                    for s in unit.selections
                ],
            },
        )
        attempt_id = f"{candidate_id}.a{len(attempts) + 1}"
        request = EvaluationRequest(
            run_id=job.run_id,
            candidate_id=candidate_id,
            attempt_id=attempt_id,
            stage=job.stage,
            genome=final_record,
            train_seed=job.train_seed,
            calibration_seed=job.calibration_seed,
            dataset_id=job.dataset_id,
            environment_id=job.environment_id,
            protocol_id=job.protocol_id,
            plan_id=job.plan_id,
            artifact_root=ctx.artifact_root,
        )
        result = ctx.evaluator.evaluate(request)
        ctx.store.commit_evaluation(result)
        if result.status == "interrupted":
            return "stopped"
        if result.status != "valid":
            raise FinalStageError(
                f"{job.run_id}: the retraining ended {result.status} ({result.failure_code}: {result.failure_detail})"
            )
        return "completed"

    # -- test ---------------------------------------------------------------------------------------------------
    def _test(self, ctx: RunnerContext) -> RunnerStatus:
        job, options = ctx.job, ctx.options
        list_id, list_sha256 = str(options["list_id"]), str(options["list_sha256"])
        manifest = artifacts_manifest(
            self.db_path, ctx.artifact_root, list_id, list_sha256
        )  # all of them, before any test
        if manifest["manifest_sha256"] != options.get("artifacts_sha256"):
            raise FinalStageError(
                "the engines are not the ones whose manifest hash was confirmed: the test stays closed"
            )
        with open_read_model(self.db_path) as rm:
            listing = verify_list(rm, list_id, list_sha256)
            self._check_run(rm, job, listing)
            unit = next(u for u in units_of(listing) if u.run_id == job.run_id)
            opened = [t for t in rm.test_evaluations() if t["run_id"] == job.run_id]
        entry = next(e for e in manifest["engines"] if e["run_id"] == job.run_id)
        if opened:
            if opened[0]["engine_sha256"] != entry["engine_sha256"]:
                raise OfficialTestAccessError(
                    f"{job.run_id}: the test was opened for another engine than the committed one"
                )
            return "completed"  # already done: the engine is never opened twice
        ledger = MarkLedger(self.db_path.parent / MARKS_DIR, job.run_id, list_sha256, self._opened_engines)
        splits = ctx.store.dataset_splits_for_run(job.run_id)  # the store releases ``test`` for a finalist run only
        if "test" not in splits:
            raise OfficialTestAccessError("the store did not release the official test for this run")
        request = TestRequest(
            run_id=job.run_id,
            attempt_id=entry["attempt_id"],
            candidate_id=str(entry["attempt_id"]).rsplit(".a", 1)[0],
            retrain_id=unit.retrain_id,
            engine_path=Path(ctx.artifact_root) / entry["engine_rel_path"],
            engine_sha256=entry["engine_sha256"],
            finalist_list_id=list_id,
            list_sha256=list_sha256,
            splits=splits,
            artifact_root=Path(ctx.artifact_root),
            ledger=ledger,
            build_manifest=self._build_manifest(entry["attempt_id"], job.run_id),
        )
        metrics = dict(self.test_evaluator.evaluate_test(request))
        if entry["engine_sha256"] not in ledger.claims:
            raise FinalStageError(
                f"{job.run_id}: the test evaluator answered without claiming the engine in the ledger; "
                "the result is not recorded"
            )
        self._check_metrics(metrics, entry, splits["test"].n_examples)
        selection = unit.selections[0]["selection_id"]
        ctx.store.record_test_evaluation(
            f"test-{job.run_id}",
            run_id=job.run_id,
            selection_id=selection,
            attempt_id=entry["attempt_id"],
            engine_sha256=entry["engine_sha256"],
            metrics={
                **metrics,
                "retrain_id": unit.retrain_id,
                "shared_by": [s["selection_id"] for s in unit.selections],
            },
        )
        return "completed"

    # -- shared -------------------------------------------------------------------------------------------------
    def _opened_engines(self) -> set[str]:
        with open_read_model(self.db_path) as rm:
            return {t["engine_sha256"] for t in rm.test_evaluations()}

    def _build_manifest(self, attempt_id: str, run_id: str) -> Mapping[str, Any]:
        """What the build pass stored for the engine (``quality["final"]["manifest"]``); empty for a wiring that stores none."""
        return dict(((_quality(self.db_path, run_id).get(attempt_id) or {}).get("final") or {}).get("manifest") or {})

    @staticmethod
    def _check_run(rm: Any, job: TCCJobSpec, listing: Mapping[str, Any]) -> None:
        run = next((r for r in rm.runs() if r["run_id"] == job.run_id), None)
        if run is None or run["stage"] != "finalist" or run["finalist_list_id"] != listing["list_id"]:
            raise FinalStageError(f"run {job.run_id!r} is not a finalist run of list {listing['list_id']!r}")
        if run["plan_id"] != listing["plan_id"]:
            raise FinalStageError(f"run {job.run_id!r} belongs to another plan than the list")

    @staticmethod
    def _check_metrics(metrics: Mapping[str, Any], entry: Mapping[str, Any], expected_n: int) -> None:
        top1 = metrics.get("top1_accuracy")
        if (
            isinstance(top1, bool)
            or not isinstance(top1, (int, float))
            or not math.isfinite(top1)
            or not 0.0 <= top1 <= 1.0
        ):
            raise FinalStageError(
                f"the test evaluator answered top1_accuracy={top1!r}; a fraction between 0 and 1 is needed"
            )
        if metrics.get("n_examples") != expected_n:
            raise FinalStageError(
                f"the test evaluator counted {metrics.get('n_examples')!r} examples, the split has {expected_n}"
            )
        if metrics.get("engine_sha256") != entry["engine_sha256"]:
            raise FinalStageError("the test evaluator used another engine file than the committed one")


# -- running the jobs ----------------------------------------------------------------------------------------------------


def run_pending(
    db_path: Path,
    artifact_root: Path,
    list_id: str,
    list_sha256: str,
    mode: str,
    final: FinalWiring,
    code_version: CodeVersion,
    *,
    worker_id: str,
    lease_s: float = 120.0,
    heartbeat_s: float | None = None,
    allow_synthetic: bool = False,
    allow_code_change: bool = False,
    on_event: Any = None,
    install_sigint: bool = False,
    log: Callable[[str], None] = print,
) -> list[StartResult]:
    """Run, one after the other, the queued jobs of ``mode`` of the units of the list; stops at the first failure."""
    if mode not in MODES:
        raise FinalStageError(f"mode must be one of {MODES}")
    require_real_wiring(_shim(final), allow_synthetic=allow_synthetic)
    wiring = as_demo_wiring(final, Path(db_path))
    results: list[StartResult] = []
    with SQLiteRunStore(Path(db_path), Path(artifact_root)) as store:
        with open_read_model(store.db_path) as rm:
            units = units_of(verify_list(rm, list_id, list_sha256))
        queue = JobQueue(store)
        for unit in units:
            for n in range(1, 10):
                job_id = unit.job_id(mode, n)
                try:
                    job = queue.get(job_id)
                except NotFoundError:
                    break
                if job.state in ("done", "cancelled") or (
                    job.state == "failed" and n < 9 and _has_next(queue, unit, mode, n)
                ):
                    continue
                result = _work(
                    store,
                    job,
                    wiring,
                    code_version,
                    worker_id=worker_id,
                    lease_s=lease_s,
                    heartbeat_s=heartbeat_s,
                    allow_code_change=allow_code_change,
                    allow_synthetic=allow_synthetic,
                    on_event=on_event,
                    install_sigint=install_sigint,
                )
                results.append(result)
                log(result.message)
                if result.outcome is None or result.outcome.end_state != "done":
                    return results
    return results


def _has_next(queue: JobQueue, unit: Unit, mode: str, n: int) -> bool:
    try:
        queue.get(unit.job_id(mode, n + 1))
        return True
    except NotFoundError:
        return False


def status(db_path: Path, list_id: str, list_sha256: str | None = None) -> list[dict[str, Any]]:
    """Per unit: where the build and the test stand (read-only)."""
    with open_read_model(Path(db_path)) as rm:
        listing = rm.finalist_list(list_id)
        jobs = rm.jobs()
        opened = {t["run_id"] for t in rm.test_evaluations()}
        rows = []
        for unit in units_of(listing):
            state = _unit_state(rm, unit)
            states = {
                mode: [j["state"] for j in jobs if j["job_id"].startswith(f"job-final-{mode}-{unit.run_id}")]
                for mode in MODES
            }
            rows.append(
                {
                    "retrain_id": unit.retrain_id,
                    "seed_block": unit.seed_block,
                    "selections": len(unit.selections),
                    "built": state["attempt"] is not None,
                    "build_jobs": states[BUILD],
                    "test_jobs": states[TEST],
                    "test_opened": unit.run_id in opened,
                }
            )
    return rows


# -- the export for the inference ------------------------------------------------------------------------------------------


def _csv(columns: Sequence[str], rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(names)
    for row in rows:
        writer.writerow(
            [
                "" if row.get(c) is None else (repr(row[c]) if isinstance(row.get(c), float) else row.get(c))
                for c in columns
            ]
        )
    return buffer.getvalue().encode("utf-8")


RESULT_COLUMNS = [
    "seed_block",
    "method",
    "role",
    "candidate_id",
    "test_accuracy",
    "l_ms",
    "v_mb",
    "s_mb",
    "val_error",
    "bench_version",
    "retrain_id",
    "run_id",
    "selection_id",
    "shared_with",
    "n_builds",
    "builds_json",
    "engine_sha256",
    "test_id",
    "source_run_id",
    "phenotype_hash",
    "protocol_stage",
]
MEASURED = ("test_accuracy", "l_ms", "v_mb", "s_mb", "val_error", "builds_json")


def export_final(
    db_path: Path,
    list_id: str,
    list_sha256: str,
    dest: Path,
    *,
    allow_synthetic: bool = False,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """The final results, one row per selection of the list, in the file that ``campaign inference`` reads."""
    dest = Path(dest)
    if dest.exists():
        raise FinalStageError(f"destination already exists: {dest} (an export never overwrites)")
    if not dest.parent.is_dir():
        raise FinalStageError(f"parent of the destination does not exist: {dest.parent}")
    with open_read_model(Path(db_path)) as rm:
        listing = verify_list(rm, list_id, list_sha256)
        units = units_of(listing)
        tests = {t["run_id"]: t for t in rm.test_evaluations()}
        details = {u.run_id: {d["attempt_id"]: d for d in _details(rm, u)} for u in units}  # noqa: F841 (lê e valida os detalhes)
        states = {u.run_id: _unit_state(rm, u) for u in units}
        qualities = {u.run_id: _quality(db_path, u.run_id) for u in units}
    synthetic = [u.run_id for u in units if states[u.run_id]["attempt"] and states[u.run_id]["attempt"]["synthetic"]]
    if synthetic and not allow_synthetic:
        raise FinalStageError(
            f"{len(synthetic)} retraining(s) have synthetic results (fake evaluator): they are not "
            "measurements; pass allow_synthetic to export them in labeled columns"
        )
    if synthetic and len(synthetic) != len([u for u in units if states[u.run_id]["attempt"]]):
        raise FinalStageError("synthetic and real results are never exported together")
    missing = [u.retrain_id for u in units if u.run_id not in tests]
    if missing and not allow_incomplete:
        raise FinalStageError(f"{len(missing)} retraining(s) have no official-test result yet: {missing[:4]}")
    versions = {
        u.run_id: frozenset(
            {str(states[u.run_id]["attempt"]["bench_version"])}
            if states[u.run_id]["attempt"] and states[u.run_id]["attempt"]["bench_version"]
            else ()
        )
        for u in units
    }
    bench_versions.select_runs(versions)  # a final table is measured by one benchmark version
    rows: list[dict[str, Any]] = []
    for unit in units:
        test = tests.get(unit.run_id)
        attempt = states[unit.run_id]["attempt"]
        if test is None or attempt is None:
            continue
        builds = ((qualities[unit.run_id].get(attempt["attempt_id"]) or {}).get("final") or {}).get("builds") or []
        for s in unit.selections:
            rows.append(
                {
                    "seed_block": s["seed_block"],
                    "method": s["method"],
                    "role": s["role"],
                    "candidate_id": s["candidate_id"],
                    "test_accuracy": float(test["metrics"]["top1_accuracy"]),
                    "l_ms": attempt["l_ms"],
                    "v_mb": attempt["v_mb"],
                    "s_mb": attempt["s_mb"],
                    "val_error": attempt["e"],
                    "bench_version": attempt["bench_version"],
                    "retrain_id": unit.retrain_id,
                    "run_id": unit.run_id,
                    "selection_id": s["selection_id"],
                    "shared_with": " ".join(o["selection_id"] for o in unit.selections if o is not s),
                    "n_builds": len(builds),
                    "builds_json": json.dumps(builds, sort_keys=True) if builds else None,
                    "engine_sha256": test["engine_sha256"],
                    "test_id": test["test_id"],
                    "source_run_id": s["source_run_id"],
                    "phenotype_hash": unit.phenotype_hash,
                    "protocol_stage": "final",
                }
            )
    rows.sort(key=lambda r: (r["seed_block"], r["method"], ROLE_ORDER.index(r["role"])))
    mark = bool(synthetic)
    names = [f"{c}{SYNTHETIC_SUFFIX}" if mark and c in MEASURED else c for c in RESULT_COLUMNS]
    contents = {"resultados_finais.csv": _csv(RESULT_COLUMNS, rows, names)}
    manifest: dict[str, Any] = {
        "format": FINAL_FORMAT,
        "kind": "export",
        "exported_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "list_id": list_id,
        "list_sha256": list_sha256,
        "plan_id": listing["plan_id"],
        "synthetic": mark,
        "retrainings": len(units),
        "tested": len(units) - len(missing),
        "missing_test": missing,
        "selections": len(listing["selections"]),
        "rows": len(rows),
        "bench_version": next(iter(sorted({r["bench_version"] for r in rows if r["bench_version"]})), None),
        "protocol_stage": "final",
        "note": "one row per selection; selections that chose the same genome of a block share one retraining and one test result; "
        "test_accuracy is top-1 on the official test as a fraction; L, V, S are the medians of the builds of the final engine",
        "files": [
            {"name": n, "sha256": sha256_hex(d), "bytes": len(d), "rows": len(rows), "columns": names}
            for n, d in contents.items()
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    building = dest.parent / f".{dest.name}.partial-{uuid.uuid4().hex[:8]}"
    building.mkdir()
    try:
        for name, data in {**contents, MANIFEST_NAME: manifest_bytes}.items():
            with open(building / name, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        os.rename(building, dest)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return manifest


def _details(rm: Any, unit: Unit) -> list[dict[str, Any]]:
    try:
        return rm.evaluation_details(unit.run_id)
    except NotFoundError:
        return []


def _quality(db_path: Path, run_id: str) -> dict[str, Mapping[str, Any]]:
    """``quality_json`` of the attempts of a run (the read model keeps only the fields it knows)."""
    import sqlite3

    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT attempt_id, quality_json FROM evaluation_attempts WHERE run_id = ?", (run_id,)
        ).fetchall()
    finally:
        con.close()
    return {a: json.loads(q) if q else {} for a, q in rows}
