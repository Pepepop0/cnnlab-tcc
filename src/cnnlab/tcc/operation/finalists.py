"""Choice of the finalists from the campaign database and its frozen plan (TCC-01 section 6; task A10a).

Reads the database through the read model (read-only, no migration, no lease) and the **frozen plan**; it never opens
the official test split and never writes, unless the caller explicitly freezes the list (:func:`freeze_list`).

What is chosen, for every confirmatory run of the plan, from the **validation** measurements of the candidates the search
evaluated (all four objectives, the non-dominated valid ones):

* ``menor_E``: the lowest top-1 error. Ties: lowest expressed-genome hash, then candidate id.
* ``equilibrio``: the smallest Euclidean distance to the origin after normalizing each objective by
  ``(f_j - m_j) / (r_j - m_j)``, with ``m_j`` the minima of the viable pilot candidates and ``r_j`` the frozen
  hypervolume reference point, both read from the plan. Ties: lowest E, then lowest genome hash, then candidate id.
* ``H1`` (only the joint runs): among the candidates with ``E <= E_mono + 0.01``, where ``E_mono`` is the lowest validation
  error of the mono run **of the same seed block**, the smallest ``L/r_L + V/r_V + S/r_S``. Ties: lowest E, then genome
  hash. With no eligible candidate (or no mono run of that block) the seed has **no H1 candidate**: a failure, listed with
  its reason, never replaced by another choice.

Genomes repeated across roles are retrained once: the unit of retraining is ``(seed block, expressed-genome hash)``,
because the four methods of a block share the retraining seed. With eight blocks, four methods and two roles each
plus the H1 candidate, the list has at most 8 x 9 = 72 entries; that bound holds by construction and is asserted.

A run that has not finished is refused (its archive is still moving) unless ``allow_unfinished``. A run with
synthetic results is refused unless ``allow_synthetic``, and its numbers then leave in ``_SINTETICO`` columns.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from cnnlab.tcc.operation.bench_versions import select_runs, version_of_rows
from cnnlab.tcc.science.nsga2 import non_dominated_fronts
from cnnlab.tcc.storage.canonical import canonical_sha256, sha256_hex
from cnnlab.tcc.storage.errors import ConsistencyError, NotFoundError, StorageError
from cnnlab.tcc.storage.readmodel import open_read_model

FINALISTS_FORMAT = "tcc-finalists/1"
MANIFEST_NAME = "manifesto.json"
SYNTHETIC_SUFFIX = "_SINTETICO"
ROLES = ("menor_E", "equilibrio", "H1")
H1_MARGIN = 0.01  # TCC-01 section 6: E <= E_mono + 0.01 (one percentage point)
MAX_RETRAINS = 72  # 2 roles x 4 methods x 8 blocks + 1 H1 x 8 blocks
EPS = 1e-12
METHOD_ORDER = ("joint", "mono", "random", "sequential")


class FinalistsError(StorageError):
    """The selection was refused; nothing was written."""


@dataclass(frozen=True, slots=True)
class Choice:
    """One role of one run, with the numbers it was chosen by."""

    run_id: str
    method: str
    seed_block: int
    role: str
    candidate_id: str
    phenotype_hash: str
    e: float
    l_ms: float
    v_mb: float
    s_mb: float
    criterion: str
    score: float | None  # distance for equilibrio, the sum for H1, None for menor_E
    synthetic: bool = False

    @property
    def retrain_key(self) -> tuple[int, str]:
        return (self.seed_block, self.phenotype_hash)

    @property
    def objectives(self) -> tuple[float, float, float, float]:
        return (self.e, self.l_ms, self.v_mb, self.s_mb)


@dataclass(frozen=True, slots=True)
class H1Failure:
    seed_block: int
    reason: str
    e_mono: float | None = None
    best_joint_e: float | None = None


@dataclass(frozen=True, slots=True)
class Retrain:
    """One retraining: a genome of a block, and every role that asked for it."""

    retrain_id: str
    seed_block: int
    phenotype_hash: str
    roles: tuple[tuple[str, str, str], ...]  # (method, role, candidate_id)


@dataclass(frozen=True, slots=True)
class Proposal:
    plan_id: str
    plan_sha256: str
    reference_point: tuple[float, float, float, float]
    viable_minima: tuple[float, float, float, float]
    choices: tuple[Choice, ...]
    h1_failures: tuple[H1Failure, ...]
    retrains: tuple[Retrain, ...]
    runs: tuple[str, ...]
    synthetic: bool = False
    unfinished: tuple[str, ...] = field(default=())

    @property
    def list_sha256(self) -> str:
        """Identity of the list of retrainings: what the test will be run for, independent of how it is written out."""
        return canonical_sha256(
            {
                "format": FINALISTS_FORMAT,
                "plan_sha256": self.plan_sha256,
                "retrains": [
                    {
                        "retrain_id": r.retrain_id,
                        "seed_block": r.seed_block,
                        "phenotype_hash": r.phenotype_hash,
                        "roles": [list(x) for x in r.roles],
                    }
                    for r in self.retrains
                ],
                "h1_failures": [{"seed_block": f.seed_block, "reason": f.reason} for f in self.h1_failures],
            }
        )


# -- the rules ------------------------------------------------------------------------------------------------


def _archive(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The valid original evaluations that no other valid evaluation dominates, in commit order."""
    valid = [r for r in rows if r["status"] == "valid" and not r["cache_hit"] and r["e"] is not None]
    points = {r["candidate_id"]: (r["e"], r["l_ms"], r["v_mb"], r["s_mb"]) for r in valid}
    front = set(non_dominated_fronts(points)[0]) if points else set()
    return [r for r in valid if r["candidate_id"] in front]


def _key(row: Mapping[str, Any]) -> tuple[float, str, str]:
    return (float(row["e"]), str(row["phenotype_hash"]), str(row["candidate_id"]))


def normalized_distance(row: Mapping[str, Any], minima: Sequence[float], reference: Sequence[float]) -> float:
    """Euclidean distance to the origin of ``(f_j - m_j) / (r_j - m_j)``."""
    values = (row["e"], row["l_ms"], row["v_mb"], row["s_mb"])
    total = 0.0
    for f, m, r in zip(values, minima, reference):
        span = r - m
        if span <= 0:
            raise FinalistsError("the reference point must lie above the viable minima in every objective")
        total += ((f - m) / span) ** 2
    return math.sqrt(total)


def h1_score(row: Mapping[str, Any], reference: Sequence[float]) -> float:
    """``L/r_L + V/r_V + S/r_S``: how small the candidate is, in units of the reference point."""
    return row["l_ms"] / reference[1] + row["v_mb"] / reference[2] + row["s_mb"] / reference[3]


def _choice(run: Mapping[str, Any], row: Mapping[str, Any], role: str, criterion: str, score: float | None) -> Choice:
    return Choice(
        run_id=run["run_id"],
        method=run["method"],
        seed_block=int(run["seed_block"]),
        role=role,
        candidate_id=row["candidate_id"],
        phenotype_hash=row["phenotype_hash"],
        e=float(row["e"]),
        l_ms=float(row["l_ms"]),
        v_mb=float(row["v_mb"]),
        s_mb=float(row["s_mb"]),
        criterion=criterion,
        score=score,
        synthetic=bool(row.get("synthetic")),
    )


def choose_for_run(
    run: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], minima: Sequence[float], reference: Sequence[float]
) -> tuple[list[Choice], list[Mapping[str, Any]]]:
    """The two roles every run has, and its archive. An empty archive has no finalist."""
    archive = _archive(rows)
    if not archive:
        return [], []
    lowest = min(archive, key=_key)
    balance = min(archive, key=lambda r: (normalized_distance(r, minima, reference), *_key(r)))
    return [
        _choice(run, lowest, "menor_E", "lowest validation top-1 error of the non-dominated valid evaluations", None),
        _choice(
            run,
            balance,
            "equilibrio",
            "smallest Euclidean distance to the origin of (f - m) / (r - m), m the viable pilot minima, r the "
            "frozen reference point",
            normalized_distance(balance, minima, reference),
        ),
    ], archive


def choose_h1(
    run: Mapping[str, Any],
    archive: Sequence[Mapping[str, Any]],
    mono_rows: Sequence[Mapping[str, Any]] | None,
    reference: Sequence[float],
) -> Choice | H1Failure:
    """The H1 candidate of a joint run, or the reason it has none."""
    block = int(run["seed_block"])
    if mono_rows is None:
        return H1Failure(block, "no mono run of this seed block to compare with")
    mono = _archive(mono_rows)
    if not mono or not archive:
        return H1Failure(block, "the mono run or the joint run has no valid evaluation")
    e_mono = min(float(r["e"]) for r in mono)
    eligible = [r for r in archive if float(r["e"]) <= e_mono + H1_MARGIN + EPS]
    if not eligible:
        return H1Failure(
            block,
            f"no non-dominated candidate with E <= E_mono + {H1_MARGIN}",
            e_mono=e_mono,
            best_joint_e=min(float(r["e"]) for r in archive),
        )
    best = min(eligible, key=lambda r: (h1_score(r, reference), *_key(r)))
    return _choice(
        run,
        best,
        "H1",
        f"smallest L/r_L + V/r_V + S/r_S among the candidates with E <= E_mono + {H1_MARGIN} (E_mono = {e_mono!r})",
        h1_score(best, reference),
    )


def group_retrains(choices: Sequence[Choice]) -> tuple[Retrain, ...]:
    """One retraining per (seed block, genome): repeated roles and methods share it."""
    groups: dict[tuple[int, str], list[Choice]] = defaultdict(list)
    for c in choices:
        groups[c.retrain_key].append(c)
    out = []
    for (block, genome), members in sorted(groups.items()):
        roles = sorted(
            ((m.method, m.role, m.candidate_id) for m in members),
            key=lambda x: (METHOD_ORDER.index(x[0]) if x[0] in METHOD_ORDER else 99, ROLES.index(x[1])),
        )
        out.append(Retrain(f"rt-b{block}-{genome[:12]}", block, genome, tuple(roles)))
    return tuple(out)


# -- reading the database ----------------------------------------------------------------------------------------


def propose(db_path: Path, plan_id: str, *, allow_unfinished: bool = False, allow_synthetic: bool = False) -> Proposal:
    """The finalists of the confirmatory runs of ``plan_id``. Read-only; the official test is never touched."""
    with open_read_model(Path(db_path)) as rm:
        plan = rm.plan(plan_id)
        if not plan["hash_ok"]:
            raise FinalistsError(f"plan {plan_id!r}: the stored hash is not the hash of its content")
        if plan["status"] != "frozen":
            raise FinalistsError(f"plan {plan_id!r} is {plan['status']}: finalists come from a frozen plan")
        content = plan["content"]
        for name in ("reference_point", "viable_minima"):
            if content.get(name) is None:
                raise FinalistsError(f"plan {plan_id!r} has no {name}: the rules of section 6 need it")
        reference = tuple(float(x) for x in content["reference_point"])
        minima = tuple(float(x) for x in content["viable_minima"])
        runs = [r for r in rm.runs() if r["plan_id"] == plan_id and r["stage"] == "confirmatory"]
        if not runs:
            raise NotFoundError(f"plan {plan_id!r} has no confirmatory run in this database")
        latest = {}
        for job in rm.jobs():
            latest[job["run_id"]] = job["state"]
        unfinished = tuple(sorted(r["run_id"] for r in runs if latest.get(r["run_id"]) not in ("done",)))
        if unfinished and not allow_unfinished:
            raise FinalistsError(
                f"{len(unfinished)} run(s) have not finished (their archive is still moving): {list(unfinished)[:4]}"
            )
        evaluations = {r["run_id"]: rm.evaluations(r["run_id"]) for r in runs}
    select_runs({k: version_of_rows(v) for k, v in evaluations.items()})  # the runs of one campaign share one bench
    synthetic = any(e["synthetic"] for rows in evaluations.values() for e in rows)
    if synthetic and not allow_synthetic:
        raise FinalistsError(
            "some runs have synthetic results (fake evaluator of the operation tests): they are not measurements; "
            "pass allow_synthetic to select among them, in labeled columns"
        )
    by_block_method = {(int(r["seed_block"]), r["method"]): r for r in runs}
    choices: list[Choice] = []
    failures: list[H1Failure] = []
    for run in sorted(runs, key=lambda r: (int(r["seed_block"]), METHOD_ORDER.index(r["method"]))):
        found, archive = choose_for_run(run, evaluations[run["run_id"]], minima, reference)
        choices += found
        if run["method"] == "joint":
            mono = by_block_method.get((int(run["seed_block"]), "mono"))
            outcome = choose_h1(run, archive, None if mono is None else evaluations[mono["run_id"]], reference)
            (choices if isinstance(outcome, Choice) else failures).append(outcome)  # type: ignore[arg-type]
    retrains = group_retrains(choices)
    if len(retrains) > MAX_RETRAINS:
        raise FinalistsError(f"{len(retrains)} retrainings: more than the {MAX_RETRAINS} the protocol allows")
    return Proposal(
        plan_id=plan_id,
        plan_sha256=plan["plan_sha256"],
        reference_point=reference,
        viable_minima=minima,  # type: ignore[arg-type]
        choices=tuple(choices),
        h1_failures=tuple(failures),
        retrains=retrains,
        runs=tuple(sorted(r["run_id"] for r in runs)),
        synthetic=synthetic,
        unfinished=unfinished,
    )


# -- export --------------------------------------------------------------------------------------------------------


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return value


def _csv(names: Sequence[str], columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(names)
    for row in rows:
        writer.writerow([_cell(row.get(c)) for c in columns])
    return buffer.getvalue().encode("utf-8")


def export_proposal(proposal: Proposal, dest: Path) -> dict[str, Any]:
    """Write the list and its manifest into the new directory ``dest``; returns the manifest."""
    dest = Path(dest)
    if dest.exists():
        raise FinalistsError(f"destination already exists: {dest} (an export never overwrites; give a new directory)")
    if not dest.parent.is_dir():
        raise FinalistsError(f"parent of the destination does not exist: {dest.parent}")
    mark = proposal.synthetic
    number = (lambda n: f"{n}{SYNTHETIC_SUFFIX}") if mark else (lambda n: n)
    first_of: dict[tuple[int, str], str] = {}
    for c in proposal.choices:
        first_of.setdefault(c.retrain_key, f"{c.method}/{c.role}")
    retrain_id = {(r.seed_block, r.phenotype_hash): r.retrain_id for r in proposal.retrains}
    rows = [
        {
            "seed_block": c.seed_block,
            "method": c.method,
            "run_id": c.run_id,
            "role": c.role,
            "candidate_id": c.candidate_id,
            "phenotype_hash": c.phenotype_hash,
            "e": c.e,
            "l_ms": c.l_ms,
            "v_mb": c.v_mb,
            "s_mb": c.s_mb,
            "score": c.score,
            "retrain_id": retrain_id[c.retrain_key],
            "shares_retrain_with": (
                "" if first_of[c.retrain_key] == f"{c.method}/{c.role}" else first_of[c.retrain_key]
            ),
            "criterion": c.criterion,
        }
        for c in proposal.choices
    ]
    cols = [
        "seed_block",
        "method",
        "run_id",
        "role",
        "candidate_id",
        "phenotype_hash",
        "e",
        "l_ms",
        "v_mb",
        "s_mb",
        "score",
        "retrain_id",
        "shares_retrain_with",
        "criterion",
    ]
    names = [number(c) if c in ("e", "l_ms", "v_mb", "s_mb", "score") else c for c in cols]
    retrain_rows = [
        {
            "retrain_id": r.retrain_id,
            "seed_block": r.seed_block,
            "phenotype_hash": r.phenotype_hash,
            "n_roles": len(r.roles),
            "roles": "; ".join(f"{m}:{role}" for m, role, _ in r.roles),
            "candidates": "; ".join(sorted({cid for _, _, cid in r.roles})),
        }
        for r in proposal.retrains
    ]
    failure_rows = [
        {
            "seed_block": f.seed_block,
            "role": "H1",
            "reason": f.reason,
            "e_mono": f.e_mono,
            "best_joint_e": f.best_joint_e,
        }
        for f in proposal.h1_failures
    ]
    contents = {
        "finalistas.csv": _csv(names, cols, rows),
        "retreinos.csv": _csv(
            ["retrain_id", "seed_block", "phenotype_hash", "n_roles", "roles", "candidates"],
            ["retrain_id", "seed_block", "phenotype_hash", "n_roles", "roles", "candidates"],
            retrain_rows,
        ),
        "h1_falhas.csv": _csv(
            ["seed_block", "role", "reason", number("e_mono"), number("best_joint_e")],
            ["seed_block", "role", "reason", "e_mono", "best_joint_e"],
            failure_rows,
        ),
    }
    manifest: dict[str, Any] = {
        "format": FINALISTS_FORMAT,
        "exported_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "plan_id": proposal.plan_id,
        "plan_sha256": proposal.plan_sha256,
        "reference_point": list(proposal.reference_point),
        "viable_minima": list(proposal.viable_minima),
        "rules": {
            "menor_E": "lowest validation top-1 error of the non-dominated valid evaluations of the run",
            "equilibrio": "smallest Euclidean distance to the origin of (f - m) / (r - m)",
            "H1": f"joint runs only: E <= E_mono + {H1_MARGIN} of the same seed block, smallest L/r_L + V/r_V + S/r_S",
            "ties": "lowest E, then expressed-genome hash, then candidate id",
            "retraining": "once per (seed block, expressed genome): roles and methods that chose it share it",
        },
        "official_test": "not opened: the selection reads validation measurements only",
        "runs": list(proposal.runs),
        "unfinished_runs": list(proposal.unfinished),
        "synthetic": proposal.synthetic,
        "n_selections": len(proposal.choices),
        "n_retrains": len(proposal.retrains),
        "max_retrains": MAX_RETRAINS,
        "h1_failures": len(proposal.h1_failures),
        "list_sha256": proposal.list_sha256,
        "files": [{"name": name, "sha256": sha256_hex(data), "bytes": len(data)} for name, data in contents.items()],
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


def freeze_list(store: Any, proposal: Proposal, *, list_id: str, author: str) -> str:
    """Freeze the selections in the database (the step that precedes opening the official test); returns the list hash.

    ``store`` is a writer on the campaign database. Refused for a synthetic or an unfinished proposal: a frozen list is
    the authorization to open the test, and it is permanent.
    """
    from cnnlab.tcc.storage.sqlite_store import FinalistSelection

    if proposal.synthetic:
        raise FinalistsError("a list of synthetic results is never frozen")
    if proposal.unfinished:
        raise FinalistsError("a list is frozen only from runs that have finished")
    selections = [
        FinalistSelection(
            selection_id=f"sel-{c.run_id}-{c.role}",
            source_run_id=c.run_id,
            candidate_id=c.candidate_id,
            role=c.role,
            criterion=c.criterion,
        )
        for c in proposal.choices
    ]
    record = store.get_plan(proposal.plan_id)
    if record.plan_sha256 != proposal.plan_sha256:
        raise ConsistencyError("the plan changed since the proposal was made")
    return store.freeze_finalist_list(
        list_id, protocol_id=record.protocol_id, plan_id=proposal.plan_id, selections=selections, author=author
    )
