"""Experiment plan document: hash definition and job-versus-plan validation.

The plan body is free-form JSON owned by the campaign, except for three
optional keys that this module understands (format `tcc-plan/1`):

* `blocks`: list of `{seed_block, search_seed, train_seed, calibration_seed,
  retrain_seed?}` with the exact seeds of each paired block;
* `methods`: list of allowed methods (subset of `joint|mono|random|sequential`);
* `budget_gpu_s`: per-run active GPU budget in seconds.

Confirmatory runs *require* the three keys, eight blocks and distinct seeds
between blocks (TCC-01 §2). When a key is present, every run must match it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, get_args

from cnnlab.tcc.contracts import Method, TCCJobSpec
from cnnlab.tcc.storage.canonical import canonical_json, sha256_hex, to_jsonable
from cnnlab.tcc.storage.errors import ConsistencyError

PLAN_FORMAT = "tcc-plan/1"
CONFIRMATORY_BLOCKS = 8
_SEED_FIELDS = ("seed_block", "search_seed", "train_seed", "calibration_seed")
_METHODS = frozenset(get_args(Method))


@dataclass(frozen=True, slots=True)
class PlanRecord:
    plan_id: str
    protocol_id: str
    version: int
    status: str
    content: Mapping[str, Any]
    plan_sha256: str
    author: str
    amends_plan_id: str | None
    amendment_reason: str | None
    created_at: str
    updated_at: str
    frozen_at: str | None


@dataclass(frozen=True, slots=True)
class PlanExport:
    """Bytes to archive next to results; `sha256(body) == plan_sha256`."""

    plan_id: str
    status: str
    frozen_at: str | None
    body: bytes
    plan_sha256: str


def plan_body(
    *,
    plan_id: str,
    protocol_id: str,
    version: int,
    author: str,
    amends_plan_id: str | None,
    amendment_reason: str | None,
    content: Mapping[str, Any],
) -> str:
    """Canonical body whose SHA-256 is the plan hash (timestamps excluded on purpose)."""
    return canonical_json(
        {
            "format": PLAN_FORMAT,
            "plan_id": plan_id,
            "protocol_id": protocol_id,
            "version": version,
            "author": author,
            "amends_plan_id": amends_plan_id,
            "amendment_reason": amendment_reason,
            "content": to_jsonable(content, _path="$.content"),
        }
    )


def plan_sha256(**kwargs: Any) -> str:
    return sha256_hex(plan_body(**kwargs))


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_plan_content(content: Mapping[str, Any]) -> None:
    """Structural checks of the keys this layer enforces; other keys are opaque."""
    if not isinstance(content, Mapping):
        raise ConsistencyError("plan content must be a JSON object")
    to_jsonable(content, _path="$.content")
    blocks = content.get("blocks")
    if blocks is not None:
        if not isinstance(blocks, (list, tuple)) or not blocks:
            raise ConsistencyError("plan.blocks must be a non-empty list")
        seen: set[int] = set()
        for block in blocks:
            if not isinstance(block, Mapping):
                raise ConsistencyError("each plan block must be an object")
            for name in _SEED_FIELDS:
                if not _is_int(block.get(name)):
                    raise ConsistencyError(f"plan block field {name!r} must be an int")
            if "retrain_seed" in block and not _is_int(block["retrain_seed"]):
                raise ConsistencyError("plan block field 'retrain_seed' must be an int")
            if block["seed_block"] in seen:
                raise ConsistencyError(f"duplicate seed_block {block['seed_block']}")
            seen.add(block["seed_block"])
    methods = content.get("methods")
    if methods is not None:
        if not isinstance(methods, (list, tuple)) or not methods:
            raise ConsistencyError("plan.methods must be a non-empty list")
        if len(set(methods)) != len(methods) or not set(methods) <= _METHODS:
            raise ConsistencyError(f"plan.methods must be distinct values of {sorted(_METHODS)}")
    budget = content.get("budget_gpu_s")
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0:
            raise ConsistencyError("plan.budget_gpu_s must be a positive number")


def _require_confirmatory_shape(content: Mapping[str, Any]) -> None:
    for key in ("blocks", "methods", "budget_gpu_s"):
        if key not in content:
            raise ConsistencyError(f"confirmatory runs need plan key {key!r}")
    blocks = content["blocks"]
    if len(blocks) != CONFIRMATORY_BLOCKS:
        raise ConsistencyError(f"confirmatory plan needs {CONFIRMATORY_BLOCKS} blocks")
    for field in ("search_seed", "train_seed"):
        values = [b[field] for b in blocks]
        if len(set(values)) != len(values):
            raise ConsistencyError(f"confirmatory plan needs distinct {field} between blocks")


def check_job_against_plan(job: TCCJobSpec, content: Mapping[str, Any]) -> None:
    """Refuse a run whose method, seeds or budget are not the ones frozen in the plan."""
    validate_plan_content(content)
    if job.stage == "confirmatory":
        _require_confirmatory_shape(content)
    methods = content.get("methods")
    if methods is not None and job.method not in methods:
        raise ConsistencyError(f"method {job.method!r} is not in the plan")
    budget = content.get("budget_gpu_s")
    if budget is not None and job.stage != "finalist" and float(budget) != float(job.budget_gpu_s):
        raise ConsistencyError(f"budget {job.budget_gpu_s} differs from plan budget {budget}")
    blocks = content.get("blocks")
    if blocks is None:
        return
    block = next((b for b in blocks if b["seed_block"] == job.seed_block), None)
    if block is None:
        raise ConsistencyError(f"seed_block {job.seed_block} is not in the plan")
    if job.stage == "finalist":
        if "retrain_seed" in block and block["retrain_seed"] != job.train_seed:
            raise ConsistencyError("finalist train_seed differs from the plan retrain_seed")
        return
    for name in ("search_seed", "train_seed", "calibration_seed"):
        if block[name] != getattr(job, name):
            raise ConsistencyError(f"{name} {getattr(job, name)} differs from plan value {block[name]}")
