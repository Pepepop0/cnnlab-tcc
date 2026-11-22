"""Operate an exploratory pilot-stage run of the TCC v2 with injected science.

This module never imports the science front. The caller supplies a *wiring
factory* ("package.module:function") that returns the real search runner,
the real evaluator, the dataset split fingerprints and the search-space hash.
Operation then:

* bootstraps, idempotently, the protocol (hash of `docs/TCC-01-PROTOCOLO.md`
  as it is on disk), the environment manifest, the dataset, a **frozen pilot
  plan labeled exploratory** (never confirmatory), the run and its job;
* versions every run: the environment manifest carries a readable system
  version (package version from `pyproject.toml`, "pre-release pilot demo",
  git SHA and dirty state) and the exact code identity. Other code means
  another environment, plan and run (announced with a warning when an
  unfinished run of the same base spec exists); resuming an existing job or
  run with other code is refused unless explicitly allowed, and then
  recorded in the claim event;
* runs one worker through `OperationService` (lease, heartbeat, resume,
  checkpoint, GPU meter, budget cut), stopping cooperatively on Ctrl+C;
* reports status from `ReadModel` (read-only), so watching never interferes
  with the lease;
* refuses a wiring that declares itself synthetic unless `--allow-synthetic`
  is given: fake evaluators exist for tests of the operation, never as the
  default of a demo, and their results stay labeled `quality.synthetic`.

CLI (`python -m cnnlab.tcc.operation.demo ...`)::

    prepare --db DB --artifacts DIR --spec SPEC.json
    start   --db DB --artifacts DIR --spec SPEC.json      # bootstrap if needed, then work
    start   --db DB --artifacts DIR --job-id JOB [--allow-code-change]   # resume a job
    start   --db DB --artifacts DIR --run-id RUN [--allow-code-change]   # continue a run (new job if needed)
    (prepare/start accept --allow-synthetic, only for tests of the operation itself)
    status  --db DB [--run RUN --evaluations] [--json]
    cancel  --db DB --artifacts DIR --job-id JOB --reason TEXT
    backup  --db DB --artifacts DIR --dest NEW_DIR --label TEXT

`SPEC.json` holds a `DemoRunSpec` including `"factory": "module:function"`
and optional `"factory_args"`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import signal
import sqlite3
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from cnnlab.tcc.contracts import Evaluator, EventSink, Method, TCCJobSpec
from cnnlab.tcc.operation import builder_config
from cnnlab.tcc.operation.provenance import (
    CodeVersion,
    capture_code_version,
    code_identity,
    default_repo_root,
    require_imported_code,
    tags_at_head,
)
from cnnlab.tcc.operation.sentinel_control import SentinelHooks
from cnnlab.tcc.operation.service import JobOutcome, OperationService, SearchRunner
from cnnlab.tcc.storage.backup import create_backup
from cnnlab.tcc.storage.canonical import canonical_sha256, sha256_hex
from cnnlab.tcc.storage.errors import (
    ConsistencyError,
    IdempotencyConflictError,
    JobStateError,
    NotFoundError,
    StorageError,
)
from cnnlab.tcc.storage.jobs import JobQueue, JobRecord
from cnnlab.tcc.storage.readmodel import open_read_model, version_text
from cnnlab.tcc.storage.schema import JOB_ACTIVE_STATES
from cnnlab.tcc.storage.sqlite_store import SplitDescriptor, SQLiteRunStore

DEMO_PURPOSE = "exploratory-demo"
ENVIRONMENT_FORMAT = "tcc-environment/demo-1"
SYSTEM_RELEASE = "TCC v2 pilot demo (pre-release; not the confirmatory protocol)"
PROTOCOL_DOC = Path("docs") / "TCC-01-PROTOCOLO.md"
_METHODS = ("joint", "mono", "random", "sequential")
_TERMINAL = ("done", "failed", "cancelled")
# What makes two specs "the same run asked again"; ids, priority, label and author do not.
_BASE_SPEC_KEYS = (
    "method",
    "seed_block",
    "search_seed",
    "train_seed",
    "calibration_seed",
    "budget_gpu_s",
    "factory",
    "factory_args",
)
WarningSink = Callable[[str], None]


EXIT_REFUSED = 5  # refused before any work (validation, storage or version error)


class CodeVersionMismatchError(JobStateError):
    """The job was started with other code; resuming would mix versions silently."""


class SyntheticWiringError(JobStateError):
    """A synthetic (fake) wiring was offered without the explicit opt-in."""


@dataclass(frozen=True, slots=True)
class DemoRunSpec:
    """What the operator decides for one run: an exploratory pilot run (stage `pilot`, its own plan) or a
    run of the campaign (``plan_id`` names the frozen campaign plan, ``stage`` is `confirmatory`).
    """

    method: Method
    seed_block: int
    search_seed: int
    train_seed: int
    calibration_seed: int
    budget_gpu_s: float
    code_label: str
    author: str
    run_id: str | None = None
    job_id: str | None = None
    priority: int = 0
    factory: str | None = None
    factory_args: Mapping[str, Any] = field(default_factory=dict)
    stage: str = "pilot"
    plan_id: str | None = None

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if self.stage not in ("pilot", "confirmatory"):
            raise ValueError("stage must be 'pilot' or 'confirmatory' (finalist runs are not started here)")
        if (self.stage == "confirmatory") != (self.plan_id is not None):
            raise ValueError("a confirmatory run needs the frozen plan it belongs to (plan_id), and only it has one")
        for name in ("seed_block", "search_seed", "train_seed", "calibration_seed", "priority"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int")
        if not self.code_label.strip() or not self.author.strip():
            raise ValueError("code_label and author are required (no version is invented)")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DemoRunSpec:
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown spec keys: {sorted(unknown)}")
        return cls(**{**data, "budget_gpu_s": float(data["budget_gpu_s"])})

    @classmethod
    def load(cls, path: Path) -> DemoRunSpec:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8-sig")))

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["factory_args"] = dict(self.factory_args)
        return payload


@dataclass(frozen=True, slots=True)
class DatasetRegistration:
    """Split fingerprints computed by the data loader; indices stay outside SQLite.

    All four splits are registered (the official test only by fingerprint);
    a pilot run never receives the test split (`dataset_splits_for_run`).
    """

    dataset_id: str
    manifest: Mapping[str, Any]
    splits: Mapping[str, SplitDescriptor]


@dataclass(frozen=True, slots=True)
class DemoWiring:
    """What the injected factory returns: real science objects plus their identity."""

    runner: SearchRunner
    evaluator: Evaluator
    dataset: DatasetRegistration
    space_hash: str
    runner_label: str
    evaluator_label: str
    synthetic: bool
    environment: Mapping[str, Any] = field(default_factory=dict)
    sentinel: SentinelHooks | None = None  # the sentinel network's measurement, when the run asks for it


WiringFactory = Callable[[DemoRunSpec], DemoWiring]


def require_real_wiring(wiring: DemoWiring, *, allow_synthetic: bool) -> None:
    """Fake evaluators never run by default; the opt-in is explicit and per command."""
    if wiring.synthetic and not allow_synthetic:
        raise SyntheticWiringError(
            f"wiring {wiring.evaluator_label!r} is synthetic (fake results); "
            "use a real factory, or pass allow_synthetic/--allow-synthetic to test the operation itself"
        )


@dataclass(frozen=True, slots=True)
class PreparedDemo:
    protocol_id: str
    environment_id: str
    plan_id: str
    plan_sha256: str
    job_spec: TCCJobSpec
    job: JobRecord
    code_version: CodeVersion
    warnings: tuple[str, ...] = ()  # set only by the call that created the run


def load_factory(path: str) -> WiringFactory:
    """Resolve "package.module:function" at run time (no static science import)."""
    module_name, sep, attr = path.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError("factory must look like 'package.module:function'")
    factory = getattr(importlib.import_module(module_name), attr)
    if not callable(factory):
        raise TypeError(f"{path} is not callable")
    return factory


# -- versions ----------------------------------------------------------------------


def package_version(repo_root: Path) -> str | None:
    """Version declared in the repository's `pyproject.toml` (None if unreadable)."""
    try:
        with open(Path(repo_root) / "pyproject.toml", "rb") as fh:
            return str(tomllib.load(fh)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None


def system_version(code_version: CodeVersion, repo_root: Path) -> dict[str, Any]:
    """Readable and exact identity of the software that runs; nothing is rounded up to 1.0."""
    pkg = package_version(repo_root)
    sha = code_version.git_sha[:12] if code_version.git_sha else "unknown"
    dirty = "+dirty" if code_version.dirty else ("" if code_version.dirty is False else "+unknown-state")
    readable = f"cnnlab-tcc {pkg or '?'} | {SYSTEM_RELEASE} | git {sha}{dirty} | {code_version.label}"
    return {
        "name": "cnnlab-tcc",
        "package_version": pkg,
        "release": SYSTEM_RELEASE,
        "readable": readable,
        "code_version": code_version.to_payload(),
    }


# -- bootstrap -----------------------------------------------------------------------


def _protocol(store: SQLiteRunStore, repo_root: Path) -> str:
    doc = repo_root / PROTOCOL_DOC
    if not doc.is_file():
        raise NotFoundError(f"protocol document not found: {doc}")
    digest = sha256_hex(doc.read_bytes())
    protocol_id = f"tcc01-{digest[:12]}"
    store.register_protocol(protocol_id, digest, "pilot")
    return protocol_id


def environment_manifest(wiring: DemoWiring, code_version: CodeVersion, repo_root: Path) -> dict[str, Any]:
    return {
        "format": ENVIRONMENT_FORMAT,
        "system": system_version(code_version, repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version,
        "runner": wiring.runner_label,
        "evaluator": wiring.evaluator_label,
        "synthetic_evaluator": wiring.synthetic,
        "provided_by_wiring": dict(wiring.environment),
    }


def _plan_content(spec: DemoRunSpec, wiring: DemoWiring, environment_id: str) -> dict[str, Any]:
    return {
        "purpose": DEMO_PURPOSE,
        "stage_scope": "pilot",
        "confirmatory": False,
        "note": (
            "Exploratory demo run. Not part of the eight confirmatory blocks; its results "
            "do not enter H1-H4 nor the frozen reference point."
        ),
        "blocks": [
            {
                "seed_block": spec.seed_block,
                "search_seed": spec.search_seed,
                "train_seed": spec.train_seed,
                "calibration_seed": spec.calibration_seed,
            }
        ],
        "methods": [spec.method],
        "budget_gpu_s": spec.budget_gpu_s,
        "environment_id": environment_id,
        "dataset_id": wiring.dataset.dataset_id,
        "space_hash": wiring.space_hash,
        "runner": wiring.runner_label,
        "evaluator": wiring.evaluator_label,
        "synthetic_evaluator": wiring.synthetic,
        "factory_args": dict(spec.factory_args),
        "cut_rule": "demo: stop at budget between attempts; the attempt in progress finishes and counts",
        "interruption_policy": "new attempt_id on resume; time of the interrupted attempt is kept",
    }


def _frozen_plan(store: SQLiteRunStore, protocol_id: str, content: Mapping[str, Any], author: str) -> tuple[str, str]:
    plan_id = f"demo-plan-{canonical_sha256(content)[:12]}"
    try:
        plan = store.get_plan(plan_id)
    except NotFoundError:
        store.create_plan(plan_id, protocol_id, content, author=author)
        plan = store.freeze_plan(plan_id)
    else:
        if canonical_sha256(plan.content) != canonical_sha256(content) or plan.protocol_id != protocol_id:
            raise IdempotencyConflictError(f"plan {plan_id!r} exists with other content")
        if plan.status != "frozen":
            plan = store.freeze_plan(plan_id)
    return plan_id, plan.plan_sha256


def _base_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    base = {key: payload.get(key) for key in _BASE_SPEC_KEYS}
    base["budget_gpu_s"] = None if base["budget_gpu_s"] is None else float(base["budget_gpu_s"])
    base["factory_args"] = dict(base["factory_args"] or {})
    return base


def _changed_keys(old: Mapping[str, Any], new: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Dotted paths whose values differ (what made the hash, hence the run, another one)."""
    out: list[str] = []
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        if a == b:
            continue
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            out += _changed_keys(a, b, f"{prefix}{key}.")
        else:
            out.append(f"{prefix}{key}")
    return out


def _why_another_run(
    store: SQLiteRunStore,
    other: JobRecord,
    content: Mapping[str, Any],
    manifest: Mapping[str, Any],
    code_version: CodeVersion,
) -> str:
    reasons: list[str] = []
    recorded = other.options.get("code_version", {})
    if code_identity(recorded) != code_version.identity:
        was = f"git {str(recorded.get('git_sha'))[:12]} dirty={recorded.get('dirty')}"
        now = f"git {str(code_version.git_sha)[:12]} dirty={code_version.dirty}"
        reasons.append(f"the code changed ({was} -> {now})")
    try:
        old_run = store.get_run(other.run_id)
        if old_run is not None:
            old_content = store.get_plan(old_run.plan_id).content
            old_manifest = store.get_environment(old_run.environment_id)
            env = [k for k in _changed_keys(old_manifest, manifest) if not k.startswith("system.")]
            if env:
                reasons.append(f"the environment changed ({', '.join(env)})")
            plan = [k for k in _changed_keys(old_content, content) if k != "environment_id"]
            if plan:
                reasons.append(f"the plan changed ({', '.join(plan)})")
    except StorageError:
        pass  # the explanation is best effort; the warning itself is not
    return "; ".join(reasons) if reasons else "the frozen plan differs"


def _sibling_warnings(
    store: SQLiteRunStore,
    spec: DemoRunSpec,
    run_id: str,
    content: Mapping[str, Any],
    manifest: Mapping[str, Any],
    code_version: CodeVersion,
) -> tuple[str, ...]:
    """One warning per unfinished run that the same base spec created before."""
    base = _base_spec(spec.to_payload())
    latest: dict[str, JobRecord] = {}
    for job in JobQueue(store).list_jobs():  # queue order: the last job of a run wins
        latest[job.run_id] = job
    out: list[str] = []
    for other in latest.values():
        stored = other.options.get("demo_spec")
        if other.run_id == run_id or other.state == "done" or not isinstance(stored, Mapping):
            continue
        if _base_spec(stored) != base:
            continue
        why = _why_another_run(store, other, content, manifest, code_version)
        out.append(
            f"warning: {run_id} is a NEW run, not the continuation of {other.run_id} "
            f"(job {other.job_id} is {other.state}). Same spec, but {why}. "
            f"To continue that run: start --run-id {other.run_id}"
        )
    return tuple(out)


def _campaign_plan(
    store: SQLiteRunStore, spec: DemoRunSpec, wiring: DemoWiring, code_version: CodeVersion, repo: Path
) -> tuple[str, str, str]:
    """(protocol id, plan id, plan hash) of a campaign run, after checking it against its frozen plan.

    Everything that could make a run of the campaign differ from the plan is refused here, before anything is
    written: the plan must be frozen and complete, and the method, seeds, budget, factory arguments, dataset,
    search space, protocol document and code revision must be the ones it records.
    """
    plan = store.get_plan(str(spec.plan_id))
    content = plan.content
    if content.get("format") != "tcc-campaign-plan/1":
        raise ConsistencyError(f"plan {plan.plan_id!r} is not a campaign plan")
    if plan.status != "frozen":
        raise ConsistencyError(f"plan {plan.plan_id!r} is {plan.status}: a campaign run needs a frozen plan")
    if content.get("pending"):
        raise ConsistencyError(f"plan {plan.plan_id!r} still has pending fields {content['pending']}")
    builder_config.verify((content.get("builder") or {}).get("timing_cache"), repo)  # every build shares one cache
    if spec.method not in content["methods"]:
        raise ConsistencyError(f"method {spec.method!r} is not in plan {plan.plan_id!r}")
    block = next((b for b in content["blocks"] if b["seed_block"] == spec.seed_block), None)
    if block is None:
        raise ConsistencyError(f"seed_block {spec.seed_block} is not in plan {plan.plan_id!r}")
    for name in ("search_seed", "train_seed", "calibration_seed"):
        if block[name] != getattr(spec, name):
            raise ConsistencyError(f"{name} {getattr(spec, name)} differs from the plan value {block[name]}")
    if float(content["budget_gpu_s"]) != float(spec.budget_gpu_s):
        raise ConsistencyError(f"budget {spec.budget_gpu_s} differs from the plan budget {content['budget_gpu_s']}")
    limit = content.get("max_attempt_train_s")
    if limit is not None and (
        spec.factory_args.get("train_time_limit_s") is None
        or float(spec.factory_args["train_time_limit_s"]) != float(limit)
    ):
        raise ConsistencyError(
            f"the training time limit of the run ({spec.factory_args.get('train_time_limit_s')!r}) is not the plan's "
            f"max_attempt_train_s ({limit!r}): the evaluator would not enforce what was frozen"
        )
    if dict(spec.factory_args) != dict(content["factory_args"]):
        raise ConsistencyError("the factory arguments of the run differ from the ones frozen in the plan")
    if spec.factory != content["factory"]:
        raise ConsistencyError(f"factory {spec.factory!r} differs from the plan factory {content['factory']!r}")
    if wiring.space_hash != content["space_hash"]:
        raise ConsistencyError("the search space of this wiring is not the one frozen in the plan (space_hash differs)")
    if wiring.dataset.dataset_id != content["dataset_id"]:
        raise ConsistencyError("the dataset of this wiring is not the one frozen in the plan")
    frozen = content["code_freeze"]
    if code_version.git_sha != frozen["git_sha"] or bool(code_version.dirty):
        raise CodeVersionMismatchError(
            f"the plan froze code {frozen['git_sha'][:12]}; this run is on {str(code_version.git_sha)[:12]}"
            + (" with uncommitted changes" if code_version.dirty else "")
        )
    doc = repo / PROTOCOL_DOC
    if not doc.is_file():
        raise NotFoundError(f"protocol document not found: {doc}")
    if f"tcc01-{sha256_hex(doc.read_bytes())[:12]}" != plan.protocol_id:
        raise ConsistencyError("the protocol document changed since the plan was frozen")
    return plan.protocol_id, plan.plan_id, plan.plan_sha256


def prepare_demo(
    store: SQLiteRunStore,
    spec: DemoRunSpec,
    wiring: DemoWiring,
    code_version: CodeVersion,
    *,
    repo_root: Path | None = None,
    allow_synthetic: bool = False,
) -> PreparedDemo:
    """Register everything a pilot-stage run needs and queue its job; safe to repeat.

    The code version is part of the environment, hence of the plan and of the
    default run id: the same spec with other code prepares a new run. When
    that happens while an unfinished run of the same base spec exists, the
    result carries a warning: the operator probably wanted `resume_run`. A
    synthetic wiring is refused before anything is written, unless allowed.
    """
    require_real_wiring(wiring, allow_synthetic=allow_synthetic)
    repo = Path(repo_root) if repo_root is not None else default_repo_root()
    if not wiring.space_hash:
        raise ValueError("wiring.space_hash is required")
    if not code_version.is_known:
        raise CodeVersionMismatchError(f"code version unknown ({code_version.error}); refusing to create a run")
    campaign = spec.plan_id is not None
    if campaign:  # checked before anything is written
        protocol_id, plan_id, plan_sha = _campaign_plan(store, spec, wiring, code_version, repo)
    else:
        protocol_id = _protocol(store, repo)
    manifest = environment_manifest(wiring, code_version, repo)
    environment_id = f"env-{canonical_sha256(manifest)[:12]}"
    store.register_environment(environment_id, manifest)
    store.register_dataset(wiring.dataset.dataset_id, wiring.dataset.manifest, wiring.dataset.splits)
    warnings: tuple[str, ...] = ()
    if campaign:
        if not spec.run_id:
            raise ValueError("a campaign run needs its run_id (the queue gives it)")
        run_id = spec.run_id
    else:
        content = _plan_content(spec, wiring, environment_id)
        plan_id, plan_sha = _frozen_plan(store, protocol_id, content, spec.author)
        run_id = spec.run_id or f"demo-{spec.method}-b{spec.seed_block}-{plan_sha[:8]}"
        if store.get_run(run_id) is None:
            warnings = _sibling_warnings(store, spec, run_id, content, manifest, code_version)
    job_spec = TCCJobSpec(
        run_id=run_id,
        protocol_id=protocol_id,
        plan_id=plan_id,
        dataset_id=wiring.dataset.dataset_id,
        environment_id=environment_id,
        space_hash=wiring.space_hash,
        method=spec.method,
        stage=spec.stage,  # type: ignore[arg-type]
        seed_block=spec.seed_block,
        search_seed=spec.search_seed,
        train_seed=spec.train_seed,
        calibration_seed=spec.calibration_seed,
        budget_gpu_s=spec.budget_gpu_s,
    )
    store.create_run(job_spec)
    queue = JobQueue(store)
    job_id = spec.job_id or f"job-{run_id}"
    try:
        job = queue.get(job_id)
    except NotFoundError:
        job = queue.enqueue(
            job_id,
            run_id,
            priority=spec.priority,
            options={
                "purpose": DEMO_PURPOSE,
                "code_version": code_version.to_payload(),
                "system_version": manifest["system"]["readable"],
                "runner": wiring.runner_label,
                "evaluator": wiring.evaluator_label,
                "synthetic_evaluator": wiring.synthetic,
                "demo_spec": spec.to_payload(),
            },
        )
    if job.run_id != run_id:
        raise ConsistencyError(f"job {job_id!r} belongs to run {job.run_id!r}, not {run_id!r}")
    return PreparedDemo(protocol_id, environment_id, plan_id, plan_sha, job_spec, job, code_version, warnings)


def check_wiring(job: JobRecord, wiring: DemoWiring, *, allow_synthetic: bool, allow_change: bool) -> dict[str, Any]:
    """The wiring that works on a job must be the one it was prepared with.

    A real job is never continued by a synthetic evaluator (or the reverse):
    that has no override. Other runner/evaluator labels are refused unless
    `allow_change`, and then recorded in the claim event.
    """
    require_real_wiring(wiring, allow_synthetic=allow_synthetic)
    recorded = job.options
    if "synthetic_evaluator" in recorded and bool(recorded["synthetic_evaluator"]) != wiring.synthetic:
        raise SyntheticWiringError(
            f"job {job.job_id!r} was prepared with synthetic_evaluator={recorded['synthetic_evaluator']!r}; "
            f"this wiring says {wiring.synthetic!r}. Real and synthetic results are never mixed in a run"
        )
    before = {"runner": recorded.get("runner"), "evaluator": recorded.get("evaluator")}
    now = {"runner": wiring.runner_label, "evaluator": wiring.evaluator_label}
    if any(v is not None for v in before.values()) and before != now:
        if not allow_change:
            raise CodeVersionMismatchError(f"job {job.job_id!r} was prepared with wiring {before}, now {now}")
        return {"wiring_changed_from": before, "wiring": now}
    return {}


def check_code_version(job: JobRecord, current: CodeVersion, *, allow_change: bool) -> dict[str, Any]:
    """Claim detail for this session; refuses mixed code unless `allow_change`."""
    recorded = job.options.get("code_version", {})
    detail: dict[str, Any] = {"code_version": current.to_payload()}
    if not current.is_known:
        if not allow_change:
            raise CodeVersionMismatchError(f"code version unknown ({current.error}); pass allow_code_change to proceed")
        detail["code_version_unknown"] = True
    elif code_identity(recorded) != current.identity:
        if not allow_change:
            raise CodeVersionMismatchError(
                f"job {job.job_id!r} started with code {code_identity(recorded)}, now {current.identity}"
            )
        detail["code_changed_from"] = dict(recorded)
    return detail


# -- worker ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StartResult:
    job_id: str
    run_id: str
    outcome: JobOutcome | None
    message: str


def _work(
    store: SQLiteRunStore,
    job: JobRecord,
    wiring: DemoWiring,
    code_version: CodeVersion,
    *,
    worker_id: str,
    lease_s: float,
    heartbeat_s: float | None,
    allow_code_change: bool,
    allow_synthetic: bool,
    on_event: EventSink | None,
    install_sigint: bool,
) -> StartResult:
    if job.state in _TERMINAL:
        return StartResult(job.job_id, job.run_id, None, f"job {job.job_id} is already {job.state}; nothing to do")
    detail = {
        **check_wiring(job, wiring, allow_synthetic=allow_synthetic, allow_change=allow_code_change),
        **check_code_version(job, code_version, allow_change=allow_code_change),
    }
    service = OperationService(
        store,
        wiring.runner,
        wiring.evaluator,
        worker_id=worker_id,
        lease_s=lease_s,
        heartbeat_s=heartbeat_s,
        on_event=on_event,
        session_detail=detail,
        sentinel=wiring.sentinel,
    )
    previous = None
    if install_sigint:

        def _stop(_signum: int, _frame: object) -> None:
            service.request_stop()  # cooperative: current attempt finishes and is committed
            signal.signal(signal.SIGINT, signal.default_int_handler)  # second Ctrl+C interrupts

        previous = signal.signal(signal.SIGINT, _stop)
    try:
        outcome = service.run_next(job_id=job.job_id)
    finally:
        if install_sigint and previous is not None:
            signal.signal(signal.SIGINT, previous)
    if outcome is None:
        current = JobQueue(store).get(job.job_id)
        if current.state == "running":
            why = f"lease held by {current.lease_owner} until {current.lease_expires_at}"
        elif current.state in _TERMINAL:
            why = "already finished"
        else:
            why = "another job holds the GPU lease or a cancel was requested"
        return StartResult(job.job_id, job.run_id, None, f"job {job.job_id} not claimed (state {current.state}; {why})")
    return StartResult(
        job.job_id, job.run_id, outcome, f"job {outcome.job_id} ended {outcome.end_state}: {outcome.detail}"
    )


def start_demo(
    db_path: Path,
    artifact_root: Path,
    spec: DemoRunSpec,
    factory: WiringFactory,
    *,
    worker_id: str,
    lease_s: float = 120.0,
    heartbeat_s: float | None = None,
    allow_code_change: bool = False,
    allow_synthetic: bool = False,
    repo_root: Path | None = None,
    code_version: CodeVersion | None = None,
    on_event: EventSink | None = None,
    install_sigint: bool = False,
    on_warning: WarningSink | None = None,
) -> StartResult:
    """Bootstrap if needed, then run this run's job in this process until it ends or stops."""
    code_version = code_version or capture_code_version(spec.code_label, repo_root)
    wiring = factory(spec)
    require_real_wiring(wiring, allow_synthetic=allow_synthetic)  # before the database is even created
    with SQLiteRunStore(db_path, artifact_root) as store:
        prepared = prepare_demo(store, spec, wiring, code_version, repo_root=repo_root, allow_synthetic=allow_synthetic)
        for warning in prepared.warnings:  # before hours of work, not after
            if on_warning is not None:
                on_warning(warning)
        job = JobQueue(store).get(prepared.job.job_id)
        return _work(
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


def resume_job(
    db_path: Path,
    artifact_root: Path,
    job_id: str,
    *,
    worker_id: str,
    factory: WiringFactory | None = None,
    lease_s: float = 120.0,
    heartbeat_s: float | None = None,
    allow_code_change: bool = False,
    allow_synthetic: bool = False,
    repo_root: Path | None = None,
    code_version: CodeVersion | None = None,
    on_event: EventSink | None = None,
    install_sigint: bool = False,
) -> StartResult:
    """Continue an existing job with the spec stored in it (the run is never re-derived)."""
    with SQLiteRunStore(db_path, artifact_root) as store:
        job = JobQueue(store).get(job_id)
        stored = job.options.get("demo_spec")
        if stored is None:
            raise ConsistencyError(f"job {job_id!r} was not prepared by operation.demo (no demo_spec)")
        spec = DemoRunSpec.from_mapping(stored)
        if factory is None:
            if not spec.factory:
                raise ConsistencyError(f"job {job_id!r} has no factory; pass one explicitly")
            factory = load_factory(spec.factory)
        code_version = code_version or capture_code_version(spec.code_label, repo_root)
        wiring = factory(spec)
        return _work(
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


def _next_job_id(run_id: str, taken: Sequence[str]) -> str:
    n = len(taken) + 1
    while f"job-{run_id}-r{n}" in taken:
        n += 1
    return f"job-{run_id}-r{n}"


def resume_run(
    db_path: Path,
    artifact_root: Path,
    run_id: str,
    *,
    worker_id: str,
    factory: WiringFactory | None = None,
    lease_s: float = 120.0,
    heartbeat_s: float | None = None,
    allow_code_change: bool = False,
    allow_synthetic: bool = False,
    repo_root: Path | None = None,
    code_version: CodeVersion | None = None,
    on_event: EventSink | None = None,
    install_sigint: bool = False,
) -> StartResult:
    """Continue an existing run, looked up by its id and never derived again.

    Deriving the run id from the spec depends on the code version, so any
    commit between a cancel and the resume would silently start another run.
    Here the run is fixed: its active job is resumed, or, when the last job is
    final (cancelled or failed), a new job is queued with the spec stored in
    it. Other code is refused before anything is queued, unless allowed; then
    the claim event records the change.
    """
    with SQLiteRunStore(db_path, artifact_root) as store:
        queue = JobQueue(store)
        if store.get_run(run_id) is None:
            raise NotFoundError(f"run {run_id!r} does not exist in this database")
        jobs = [j for j in queue.list_jobs() if j.run_id == run_id]
        if not jobs:
            raise ConsistencyError(f"run {run_id!r} has no job: it was not prepared by operation.demo")
        last = jobs[-1]
        stored = last.options.get("demo_spec")
        if stored is None:
            raise ConsistencyError(f"job {last.job_id!r} was not prepared by operation.demo (no demo_spec)")
        if last.state == "done":
            return StartResult(
                last.job_id, run_id, None, f"run {run_id} is already done (job {last.job_id}); nothing to do"
            )
        spec = DemoRunSpec.from_mapping(stored)  # as stored: a factory may read run_id, and it must not change
        if factory is None:
            if not spec.factory:
                raise ConsistencyError(f"job {last.job_id!r} has no factory; pass one explicitly")
            factory = load_factory(spec.factory)
        code_version = code_version or capture_code_version(spec.code_label, repo_root)
        if last.state in JOB_ACTIVE_STATES:
            job = last
            wiring = factory(replace(spec, job_id=job.job_id))
        else:
            job_id = _next_job_id(run_id, [j.job_id for j in jobs])
            spec = replace(spec, job_id=job_id)
            wiring = factory(spec)
            # Refuse before queueing: a refused resume leaves no job behind.
            check_wiring(last, wiring, allow_synthetic=allow_synthetic, allow_change=allow_code_change)
            check_code_version(last, code_version, allow_change=allow_code_change)
            options = {k: v for k, v in last.options.items() if k != "format"}
            options.update({"demo_spec": spec.to_payload(), "resumed_from_job": last.job_id})
            job = queue.enqueue(job_id, run_id, priority=last.priority, options=options)
        return _work(
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


# -- status (read-only) ------------------------------------------------------------


def demo_status(db_path: Path, *, now: datetime | None = None, events: int = 8) -> dict[str, Any]:
    """Queue and run progress through `ReadModel` (read-only; never touches a lease nor migrates).

    `runs_without_job` lists runs that no job refers to, e.g. every run of a
    schema v1 database, written straight through the `RunStore` port.
    """
    with open_read_model(db_path) as rm:
        runs = {r["run_id"]: r for r in rm.runs()}
        jobs = []
        for job in rm.jobs(now=now):
            run = runs[job["run_id"]]
            jobs.append(
                {
                    **job,
                    "run": run,
                    "progress": rm.run_progress(job["run_id"]),
                    "recent_events": rm.job_events(job["job_id"], limit=events),
                }
            )
        queued = {j["run_id"] for j in jobs}
        loose = [
            {"run": run, "progress": rm.run_progress(run_id)} for run_id, run in runs.items() if run_id not in queued
        ]
        return {
            "read_model": rm.version,
            "schema_version": rm.schema_version(),
            "has_job_queue": rm.has_job_queue,
            "jobs": jobs,
            "runs_without_job": loose,
        }


def _fmt(value: float | None, pattern: str) -> str:
    return "-" if value is None else format(value, pattern)


def _progress_lines(run: Mapping[str, Any], prog: Mapping[str, Any], *, claims: int | None = None) -> list[str]:
    counts = (
        f"  gpu {prog['active_gpu_s']:.1f}/{prog['budget_gpu_s']:.0f}s  attempts={prog['attempts']}  "
        f"unique_valid={prog['unique_valid_evaluations']}  cache_hits={prog['cache_hits']}  "
        f"candidates={prog['candidates']}"
    )
    lines = [
        f"  system {version_text(run)}",
        counts + ("" if claims is None else f"  claims={claims}"),
        f"  generations={prog['generations']}  checkpoints={prog['checkpoints']}  fronts={prog['front_snapshots']}",
    ]
    if prog["latest_checkpoint"]:
        c = prog["latest_checkpoint"]
        lines.append(f"  checkpoint gen={c['generation']} unique={c['unique_evaluations']} at {c['created_at']}")
    return lines


def format_status(status: Mapping[str, Any]) -> str:
    lines = [f"{status['read_model']} schema v{status['schema_version']}"]
    if not status.get("has_job_queue", True):
        lines.append("  no job queue in this database (schema v1): runs were written without job, lease or heartbeat")
    elif not status["jobs"]:
        lines.append("  no jobs")
    for j in status["jobs"]:
        run, prog = j["run"], j["progress"]
        lease = ""
        if j["state"] == "running":
            lease = f" lease={j['lease_owner']} live={j['lease_live']} until {j['lease_expires_at']} hb={j['heartbeat_count']}"
        lines.append(
            f"- {j['job_id']} [{j['state']}] run={j['run_id']} {run['stage']}/{run['method']}/b{run['seed_block']}{lease}"
        )
        lines.extend(_progress_lines(run, prog, claims=j["claim_count"]))
        if j["options"].get("synthetic_evaluator"):
            lines.append(f"  SYNTHETIC evaluator: {prog['synthetic_results']} results are fake (tests/demo wiring)")
        if j["final_detail"]:
            lines.append(f"  detail: {j['final_detail']}")
        for e in j["recent_events"]:
            lines.append(f"    {e['recorded_at']} {e['kind']} -> {e['to_state']} ({e['worker_id']})")
    for item in status.get("runs_without_job", ()):
        run, prog = item["run"], item["progress"]
        lines.append(f"- (no job) run={run['run_id']} {run['stage']}/{run['method']}/b{run['seed_block']}")
        lines.extend(_progress_lines(run, prog))
        if prog["synthetic_results"]:
            lines.append(f"  SYNTHETIC: {prog['synthetic_results']} results are fake (tests/demo wiring)")
    return "\n".join(lines)


def format_evaluations(rows: Sequence[Mapping[str, Any]]) -> str:
    width = max([16, *(len(r["attempt_id"]) for r in rows)])
    head = f"{'attempt':<{width}} {'gen':>3} {'status':<11} {'E':>7} {'L ms':>9} {'V MB':>9} {'S MB':>9} {'gpu s':>9}  flags"
    out = [head]
    for r in rows:
        flags = ",".join(
            f
            for f, on in (
                ("cache", r["cache_hit"]),
                ("synthetic", r["synthetic"]),
                ("preliminary", r.get("preliminary")),
                ("recovered", r["recovered"]),
            )
            if on
        )
        if r["failure_code"]:
            flags = f"{flags} {r['failure_stage']}/{r['failure_code']}".strip()
        generation = "-" if r["generation"] is None else str(r["generation"])
        out.append(
            f"{r['attempt_id']:<{width}} {generation:>3} {r['status']:<11} {_fmt(r['e'], '.4f'):>7} "
            f"{_fmt(r['l_ms'], '.4f'):>9} {_fmt(r['v_mb'], '.3f'):>9} {_fmt(r['s_mb'], '.3f'):>9} "
            f"{_fmt(r['gpu_seconds'], '.1f'):>9}  {flags}"
        )
    return "\n".join(out)


# -- CLI ---------------------------------------------------------------------------


def _print_event(event: Any) -> None:
    print(f"[{event.ts_utc}] {event.kind} {json.dumps(dict(event.payload), sort_keys=True)}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cnnlab.tcc.operation.demo", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, *, artifacts: bool = True) -> None:
        p.add_argument("--db", type=Path, required=True)
        if artifacts:
            p.add_argument("--artifacts", type=Path, required=True)

    p = sub.add_parser("prepare")
    common(p)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--factory", help="override spec.factory ('package.module:function')")
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--allow-synthetic", action="store_true", help="accept a fake wiring (tests of the operation only)")
    p = sub.add_parser("start")
    common(p)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--spec", type=Path)
    target.add_argument("--job-id")
    target.add_argument("--run-id", help="continue this run; queues a new job if the last one is final")
    p.add_argument("--factory", help="override spec.factory ('package.module:function')")
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--worker-id", default=f"demo-{platform.node()}")
    p.add_argument("--lease-s", type=float, default=120.0)
    p.add_argument("--heartbeat-s", type=float, default=None)
    p.add_argument("--allow-code-change", action="store_true")
    p.add_argument("--allow-synthetic", action="store_true", help="accept a fake wiring (tests of the operation only)")
    p = sub.add_parser("status")
    common(p, artifacts=False)
    p.add_argument("--json", action="store_true")
    p.add_argument("--run", help="with --evaluations: run to list")
    p.add_argument("--evaluations", action="store_true")
    p = sub.add_parser("cancel")
    common(p)
    p.add_argument("--job-id", required=True)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("backup")
    common(p)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    try:
        return _run_command(parser, args)
    except (StorageError, ValueError, TypeError, ImportError, AttributeError, OSError, sqlite3.Error) as exc:
        # One clean line for the operator (and for the TUI that shows this output).
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return EXIT_REFUSED


def _run_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.command in ("start", "prepare") and getattr(args, "repo_root", None) is not None:
        try:
            require_imported_code(args.repo_root)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
    if args.command == "status":
        if args.evaluations:
            if not args.run:
                parser.error("--evaluations needs --run")
            with open_read_model(args.db) as rm:
                rows = rm.evaluations(args.run)
            print(json.dumps(rows, indent=2, sort_keys=True) if args.json else format_evaluations(rows))
            return 0
        status = demo_status(args.db)
        print(json.dumps(status, indent=2, sort_keys=True) if args.json else format_status(status))
        return 0
    if args.command == "cancel":
        with SQLiteRunStore(args.db, args.artifacts) as store:
            job = JobQueue(store).request_cancel(args.job_id, reason=args.reason)
        print(f"job {job.job_id}: {job.state} (cancel requested: {job.cancel_requested})")
        return 0
    if args.command == "backup":
        result = create_backup(args.db, args.artifacts, args.dest, label=args.label)
        print(f"backup {result.path} complete={result.complete} manifest_sha256={result.manifest_sha256}")
        print(json.dumps(result.manifest["summary"], sort_keys=True))
        return 0 if result.complete else 2

    if args.command == "start" and args.run_id:
        result = resume_run(
            args.db,
            args.artifacts,
            args.run_id,
            worker_id=args.worker_id,
            factory=load_factory(args.factory) if args.factory else None,
            lease_s=args.lease_s,
            heartbeat_s=args.heartbeat_s,
            allow_code_change=args.allow_code_change,
            allow_synthetic=args.allow_synthetic,
            repo_root=args.repo_root,
            on_event=_print_event,
            install_sigint=True,
        )
    elif args.command == "start" and args.job_id:
        result = resume_job(
            args.db,
            args.artifacts,
            args.job_id,
            worker_id=args.worker_id,
            factory=load_factory(args.factory) if args.factory else None,
            lease_s=args.lease_s,
            heartbeat_s=args.heartbeat_s,
            allow_code_change=args.allow_code_change,
            allow_synthetic=args.allow_synthetic,
            repo_root=args.repo_root,
            on_event=_print_event,
            install_sigint=True,
        )
    else:
        spec = DemoRunSpec.load(args.spec)
        factory_path = args.factory or spec.factory
        if not factory_path:
            parser.error("a wiring factory is required (spec.factory or --factory)")
        factory = load_factory(factory_path)
        if args.command == "prepare":
            code_version = capture_code_version(spec.code_label, args.repo_root)
            wiring = factory(spec)
            require_real_wiring(wiring, allow_synthetic=args.allow_synthetic)  # before the database exists
            with SQLiteRunStore(args.db, args.artifacts) as store:
                prepared = prepare_demo(
                    store, spec, wiring, code_version, repo_root=args.repo_root, allow_synthetic=args.allow_synthetic
                )
            for warning in prepared.warnings:
                print(warning, flush=True)
            print(
                f"prepared run {prepared.job_spec.run_id} (stage pilot, plan {prepared.plan_id} frozen) "
                f"job {prepared.job.job_id} [{prepared.job.state}] code sha={code_version.git_sha} dirty={code_version.dirty}"
            )
            return 0
        result = start_demo(
            args.db,
            args.artifacts,
            spec,
            factory,
            worker_id=args.worker_id,
            lease_s=args.lease_s,
            heartbeat_s=args.heartbeat_s,
            allow_code_change=args.allow_code_change,
            allow_synthetic=args.allow_synthetic,
            repo_root=args.repo_root,
            on_event=_print_event,
            install_sigint=True,
            on_warning=lambda text: print(text, flush=True),
        )
    print(result.message, flush=True)
    if result.outcome is None:
        return 0 if "already" in result.message else 3
    return {"done": 0, "queued": 0, "cancelled": 0}.get(result.outcome.end_state, 4)


if __name__ == "__main__":
    raise SystemExit(main())
