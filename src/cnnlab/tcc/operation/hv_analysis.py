"""Hypervolume of every run by GPU time and by unique evaluations, per method and seed (task A7; TCC-01 sections 4 to 6).

Reads the database through the read model only (read-only, no migration, no lease) and writes CSV files in
UTF-8 into a new directory, with a manifest that carries the SHA-256 of each file, the reference point and
where it came from.

What is computed, and why it is computed from the evaluations
-------------------------------------------------------------
For each run the evaluations are taken in commit order. After each **valid, unique** evaluation (a cache hit
adds no information and is not counted) the hypervolume of everything valid evaluated so far is computed
exactly (``science/hypervolume.py``) against one reference point, the same for every method:

* ``hv`` is the official, normalized value (each objective divided by its reference coordinate, in [0, 1]);
  ``hv_raw`` is the volume in the units of the objectives; ``hv_sens`` is the normalized value against the
  sensitivity point (1.5 x the reference in L, V and S, TCC-01 section 5).
* The x axes are the **active GPU seconds** spent up to and including that evaluation (the attempts of the run
  in commit order, valid or not, so failed attempts spend time) and the **number of unique valid evaluations**.
* The curve comes from the evaluations, not from the front snapshots of the run: a run that stops on the budget
  in the middle of a generation records no snapshot for the evaluations after the last one, and the protocol
  asks for the last confirmed value at every instant. The recorded snapshots are used for a check: where a
  snapshot recorded a hypervolume against the same reference point, it must equal the curve at the same count of
  unique evaluations (``snapshot_mismatches``).

Carry-forward and the cut. On the time grid a run keeps its last confirmed value until the grid ends, and the
time it did not use is declared (``unused_gpu_s``), as the protocol says; the grid does not extend a run's
evaluations. The cut rule lets the attempt in progress at the budget finish and count, so ``hv_final`` includes it
and ``hv_at_budget`` is the last value confirmed within the budget; the two differ only when that attempt was a
valid unique evaluation that improved the front. On the evaluation grid a run has no value beyond its own count.

Candidates beyond the reference point add no volume and are listed (``hv_fora_do_ponto.csv``), as the protocol asks.

The sentinel network (TCC-01 section 3) is read from the job events of the run. Nothing is replaced or
removed: the points of the curve whose attempt a ``sentinel_drift`` event marks carry ``sentinel_drift =
true``, and every run states what its sentinel cost (``sentinel_gpu_s``, which is *not* part of its GPU budget),
how many times it was measured, how many measurements failed and how many drifts were seen.

Synthetic results (fake evaluator of the operation tests) are not measurements: a run that has them is refused
unless ``allow_synthetic``, every value column is then named ``..._SINTETICO``, and synthetic and real runs are
never exported together.
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
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from cnnlab.tcc.operation.bench_versions import select_runs, version_of_rows
from cnnlab.tcc.science.hypervolume import hypervolume, hypervolume_of, sensitivity_reference
from cnnlab.tcc.science.nsga2 import dominates
from cnnlab.tcc.storage.canonical import sha256_hex
from cnnlab.tcc.storage.errors import NotFoundError, StorageError
from cnnlab.tcc.storage.readmodel import open_read_model

HV_FORMAT = "tcc-hv-analysis/1"
MANIFEST_NAME = "manifesto.json"
SYNTHETIC_SUFFIX = "_SINTETICO"
DEFAULT_GRID_S = 600.0
METHOD_ORDER = ("joint", "mono", "random", "sequential")
HV_COLUMNS = ("hv", "hv_raw", "hv_sens")
SNAPSHOT_TOLERANCE = 1e-9


class AnalysisError(StorageError):
    """The analysis was refused; nothing was written."""


@dataclass(frozen=True, slots=True)
class RunCurve:
    """The hypervolume of one run after each unique valid evaluation, and what the run spent."""

    run_id: str
    method: str
    seed_block: int
    stage: str
    synthetic: bool
    budget_gpu_s: float
    gpu_total_s: float
    attempts: int
    valid: int
    points: tuple[Mapping[str, Any], ...]
    outside: tuple[str, ...]
    snapshot_mismatches: tuple[str, ...]
    drift_attempts: frozenset[str] = frozenset()  # attempts a sentinel drift marks
    sentinel_gpu_s: float = 0.0  # what the measurements of the sentinel took, outside the budget
    sentinel_measurements: int = 0
    sentinel_failures: int = 0
    sentinel_drifts: int = 0

    @property
    def unique(self) -> int:
        return len(self.points)


def _vector(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (float(row["e"]), float(row["l_ms"]), float(row["v_mb"]), float(row["s_mb"]))


def _inside(point: Sequence[float], reference: Sequence[float]) -> bool:
    return all(x < r for x, r in zip(point, reference))


def run_curve(
    run: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    reference: Sequence[float],
    snapshots: Sequence[Mapping[str, Any]] = (),
    snapshot_reference: Sequence[float] | None = None,
    sentinel: Sequence[Mapping[str, Any]] = (),
) -> RunCurve:
    """The curve of one run. ``evaluations`` are the read-model rows in commit order (all statuses)."""
    ref = tuple(float(x) for x in reference)
    sens = sensitivity_reference(ref)
    gpu = 0.0
    unique = 0
    archive: list[tuple[float, ...]] = []  # non-dominated, inside the box: all that shapes the volume
    seen: dict[str, tuple[float, ...]] = {}
    hv = hv_raw = hv_sens = 0.0
    points: list[dict[str, Any]] = []
    for row in evaluations:
        if row["status"] == "running":
            continue
        gpu += float(row["gpu_seconds"] or 0.0)
        if row["status"] != "valid" or row["cache_hit"] or row["e"] is None:
            continue
        unique += 1
        vector = _vector(row)
        seen[row["candidate_id"]] = vector
        if _inside(vector, ref) and not any(dominates(a, vector) or a == vector for a in archive):
            archive = [a for a in archive if not dominates(vector, a)] + [vector]
            hv_raw = hypervolume(archive, ref)
            hv = hv_raw / math.prod(ref)
            hv_sens = hypervolume(archive, sens) / math.prod(sens)
        points.append(
            {
                "step": unique,
                "attempt_id": row["attempt_id"],
                "candidate_id": row["candidate_id"],
                "generation": row["generation"],
                "gpu_s": gpu,
                "unique": unique,
                "hv": hv,
                "hv_raw": hv_raw,
                "hv_sens": hv_sens,
                "front": len(archive),
                "sentinel_drift": False,
            }
        )
    outside = tuple(sorted(cid for cid, v in seen.items() if not _inside(v, ref)))
    mismatches: list[str] = []
    if (
        snapshots
        and snapshot_reference is not None
        and all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(snapshot_reference, ref))
    ):
        by_count = {p["unique"]: p for p in points}
        for snap in snapshots:
            recorded = snap.get("hypervolume")
            at = by_count.get(snap["unique_valid_evaluations"])
            if recorded is None or at is None:
                continue
            if abs(recorded - at["hv"]) > SNAPSHOT_TOLERANCE * max(1.0, abs(recorded)):
                mismatches.append(
                    f"snapshot {snap['snapshot_id']} at {snap['unique_valid_evaluations']} unique: "
                    f"recorded {recorded!r}, from the evaluations {at['hv']!r}"
                )
    drift_attempts = frozenset(
        a for e in sentinel if e["kind"] == "sentinel_drift" for a in e["detail"].get("attempt_ids", ())
    )
    for p in points:
        p["sentinel_drift"] = p["attempt_id"] in drift_attempts
    return RunCurve(
        run_id=run["run_id"],
        method=run["method"],
        seed_block=int(run["seed_block"]),
        stage=run["stage"],
        synthetic=any(r["synthetic"] for r in evaluations),
        budget_gpu_s=float(run["budget_gpu_s"]),
        gpu_total_s=gpu,
        attempts=sum(1 for r in evaluations if r["status"] != "running"),
        valid=sum(1 for r in evaluations if r["status"] == "valid"),
        points=tuple(points),
        outside=outside,
        snapshot_mismatches=tuple(mismatches),
        drift_attempts=drift_attempts,
        sentinel_gpu_s=sum(
            float(e["detail"].get("seconds", 0.0))
            for e in sentinel
            if e["kind"] in ("sentinel_measured", "sentinel_failed")
        ),
        sentinel_measurements=sum(1 for e in sentinel if e["kind"] == "sentinel_measured"),
        sentinel_failures=sum(1 for e in sentinel if e["kind"] == "sentinel_failed"),
        sentinel_drifts=sum(1 for e in sentinel if e["kind"] == "sentinel_drift"),
    )


def value_at_gpu(curve: RunCurve, seconds: float, key: str = "hv") -> float:
    """The last confirmed value at ``seconds`` of active GPU (0 before the first evaluation)."""
    value = 0.0
    for point in curve.points:
        if point["gpu_s"] > seconds:
            break
        value = point[key]
    return value


def value_at_unique(curve: RunCurve, count: int, key: str = "hv") -> float | None:
    """The value after ``count`` unique valid evaluations; None beyond what the run evaluated."""
    if count > curve.unique:
        return None
    return curve.points[count - 1][key] if count >= 1 else 0.0


def final_values(curve: RunCurve) -> dict[str, float]:
    """The values at the end of the run (the attempt at the budget included) and within the budget."""
    last = curve.points[-1] if curve.points else {"hv": 0.0, "hv_raw": 0.0, "hv_sens": 0.0}
    return {
        "hv_final": last["hv"],
        "hv_raw_final": last["hv_raw"],
        "hv_sens_final": last["hv_sens"],
        "hv_at_budget": value_at_gpu(curve, curve.budget_gpu_s, "hv"),
        "hv_raw_at_budget": value_at_gpu(curve, curve.budget_gpu_s, "hv_raw"),
        "hv_sens_at_budget": value_at_gpu(curve, curve.budget_gpu_s, "hv_sens"),
    }


# -- reading the database -----------------------------------------------------------------------------------


def read_curves(
    db_path: Path,
    reference: Sequence[float] | None = None,
    run_ids: Sequence[str] | None = None,
    recorded_reference: Sequence[float] | None = None,
    bench_version: str | None = None,
    notes: dict[str, Any] | None = None,
) -> tuple[list[RunCurve], tuple[float, ...], str]:
    """(curves, reference point used, where it came from) for the runs of the database.

    Without an explicit ``reference`` the point comes from the frozen plan of the runs, which must agree.
    ``recorded_reference`` is the point the runs stored their snapshot hypervolume with, when it is not
    in their plan; the snapshots are checked only against the same point.

    Runs measured by different benchmark versions are refused unless ``bench_version`` names the one to keep; the
    runs left out are listed in ``notes['left_out_by_bench_version']`` and the version kept in ``notes['bench_version']``.
    """
    with open_read_model(Path(db_path)) as rm:
        runs = [r for r in rm.runs() if run_ids is None or r["run_id"] in run_ids]
        if run_ids is not None and len(runs) != len(set(run_ids)):
            missing = sorted(set(run_ids) - {r["run_id"] for r in runs})
            raise NotFoundError(f"runs not in the database: {missing}")
        if not runs:
            raise AnalysisError("the database has no runs to analyse")
        rows_of = {r["run_id"]: rm.evaluations(r["run_id"]) for r in runs}
        keep, left_out = select_runs({k: version_of_rows(v) for k, v in rows_of.items()}, bench_version)
        runs = [r for r in runs if r["run_id"] in keep]
        if notes is not None:
            notes["bench_version"] = bench_version or next(iter(version_of_rows(rows_of[keep[0]])), None)
            notes["left_out_by_bench_version"] = left_out
        plan_points: dict[str, tuple[float, ...]] = {}
        for plan_id in sorted({r["plan_id"] for r in runs}):
            point = rm.plan(plan_id)["content"].get("reference_point")
            if point is not None:
                plan_points[plan_id] = tuple(float(x) for x in point)
        if reference is not None:
            ref, origin = tuple(float(x) for x in reference), "given on the command line"
        elif (
            plan_points
            and len(set(plan_points.values())) == 1
            and len(plan_points) == len({r["plan_id"] for r in runs})
        ):
            ref = next(iter(plan_points.values()))
            origin = "the frozen plan of the runs: " + ", ".join(sorted(plan_points))
        else:
            raise AnalysisError(
                "no reference point: the plans of these runs do not all carry the same one; give it with --reference"
            )
        if len(ref) != 4 or not all(math.isfinite(x) and x > 0 for x in ref):
            raise AnalysisError("the reference point needs four finite positive numbers (E, L, V, S)")
        curves = []
        for run in runs:
            snapshots = rm.fronts(run["run_id"])
            curves.append(
                run_curve(
                    run,
                    rows_of[run["run_id"]],
                    ref,
                    snapshots,
                    recorded_reference or plan_points.get(run["plan_id"]),
                    rm.sentinel_events(run["run_id"]),
                )
            )
    return curves, ref, origin


# -- tables ----------------------------------------------------------------------------------------------------


def _label(names: Sequence[str], synthetic: bool) -> list[str]:
    return [f"{n}{SYNTHETIC_SUFFIX}" if synthetic and n in HV_COLUMNS + ("hv_raw_final",) else n for n in names]


def curve_rows(curves: Sequence[RunCurve]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": c.run_id,
            "method": c.method,
            "seed_block": c.seed_block,
            **{
                k: p[k]
                for k in (
                    "step",
                    "attempt_id",
                    "candidate_id",
                    "generation",
                    "gpu_s",
                    "unique",
                    "hv",
                    "hv_raw",
                    "hv_sens",
                    "front",
                    "sentinel_drift",
                )
            },
        }
        for c in curves
        for p in c.points
    ]


def time_grid(curves: Sequence[RunCurve], step_s: float = DEFAULT_GRID_S) -> list[float]:
    horizon = max((c.budget_gpu_s for c in curves), default=0.0)
    return [step_s * i for i in range(1, int(horizon // step_s) + 1)]


def time_grid_rows(curves: Sequence[RunCurve], grid: Sequence[float]) -> list[dict[str, Any]]:
    out = []
    for c in curves:
        for t in grid:
            ended = t >= c.gpu_total_s
            out.append(
                {
                    "run_id": c.run_id,
                    "method": c.method,
                    "seed_block": c.seed_block,
                    "gpu_s": t,
                    "hv": value_at_gpu(c, t, "hv"),
                    "hv_raw": value_at_gpu(c, t, "hv_raw"),
                    "hv_sens": value_at_gpu(c, t, "hv_sens"),
                    "run_ended": ended,
                    "unused_gpu_s": max(0.0, t - c.gpu_total_s) if ended else 0.0,
                }
            )
    return out


def evaluation_grid_rows(curves: Sequence[RunCurve]) -> list[dict[str, Any]]:
    horizon = max((c.unique for c in curves), default=0)
    out = []
    for c in curves:
        for n in range(1, horizon + 1):
            value = value_at_unique(c, n)
            if value is None:
                continue  # the run never reached this many unique evaluations: no value, nothing carried
            out.append(
                {
                    "run_id": c.run_id,
                    "method": c.method,
                    "seed_block": c.seed_block,
                    "unique": n,
                    "gpu_s": c.points[n - 1]["gpu_s"],
                    "hv": value,
                    "hv_raw": value_at_unique(c, n, "hv_raw"),
                    "hv_sens": value_at_unique(c, n, "hv_sens"),
                }
            )
    return out


def final_rows(curves: Sequence[RunCurve]) -> list[dict[str, Any]]:
    out = []
    for c in curves:
        f = final_values(c)
        out.append(
            {
                "run_id": c.run_id,
                "method": c.method,
                "seed_block": c.seed_block,
                "stage": c.stage,
                "attempts": c.attempts,
                "valid": c.valid,
                "unique_valid": c.unique,
                "gpu_total_s": c.gpu_total_s,
                "budget_gpu_s": c.budget_gpu_s,
                "unused_gpu_s": max(0.0, c.budget_gpu_s - c.gpu_total_s),
                "excess_gpu_s": max(0.0, c.gpu_total_s - c.budget_gpu_s),
                "front_size": c.points[-1]["front"] if c.points else 0,
                "beyond_reference": len(c.outside),
                "snapshot_mismatches": len(c.snapshot_mismatches),
                "sentinel_gpu_s": c.sentinel_gpu_s,
                "sentinel_measurements": c.sentinel_measurements,
                "sentinel_failures": c.sentinel_failures,
                "sentinel_drifts": c.sentinel_drifts,
                "points_in_drift": sum(1 for p in c.points if p["sentinel_drift"]),
                "hv_final": f["hv_final"],
                "hv_raw_final": f["hv_raw_final"],
                "hv_sens_final": f["hv_sens_final"],
                "hv_at_budget": f["hv_at_budget"],
                "hv_raw_at_budget": f["hv_raw_at_budget"],
                "hv_sens_at_budget": f["hv_sens_at_budget"],
            }
        )
    return out


def paired_rows(curves: Sequence[RunCurve], key: str = "hv_final") -> list[dict[str, Any]]:
    """One row per seed block, the value of each method and the paired differences joint minus each control.

    These are the numbers of H2 and H3 before any test: no p-value and no choice is made here. A block missing
    a method leaves the cells of that method (and its differences) empty.
    """
    by_block: dict[int, dict[str, float]] = defaultdict(dict)
    for c in curves:
        by_block[c.seed_block][c.method] = final_values(c)[key]
    methods = [m for m in METHOD_ORDER if any(m in v for v in by_block.values())]
    out = []
    for block in sorted(by_block):
        values = by_block[block]
        row: dict[str, Any] = {"seed_block": block, **{m: values.get(m) for m in methods}}
        for control in methods:
            if control == "joint":
                continue
            row[f"joint_menos_{control}"] = (
                values["joint"] - values[control] if "joint" in values and control in values else None
            )
        out.append(row)
    return out


def outside_rows(curves: Sequence[RunCurve]) -> list[dict[str, Any]]:
    return [
        {"run_id": c.run_id, "method": c.method, "seed_block": c.seed_block, "candidate_id": cid}
        for c in curves
        for cid in c.outside
    ]


# -- export ------------------------------------------------------------------------------------------------------


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return value


def _csv(columns: Sequence[str], rows: Sequence[Mapping[str, Any]], names: Sequence[str] | None = None) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(names or columns)
    for row in rows:
        writer.writerow([_cell(row.get(c)) for c in columns])
    return buffer.getvalue().encode("utf-8")


@dataclass(frozen=True, slots=True)
class HvExport:
    path: Path
    manifest: Mapping[str, Any]
    curves: tuple[RunCurve, ...]


def export_hv(
    db_path: Path,
    dest: Path,
    *,
    reference: Sequence[float] | None = None,
    run_ids: Sequence[str] | None = None,
    grid_s: float = DEFAULT_GRID_S,
    allow_synthetic: bool = False,
    recorded_reference: Sequence[float] | None = None,
    bench_version: str | None = None,
) -> HvExport:
    """Write the tables and the manifest of the analysis into the new directory ``dest``."""
    dest = Path(dest)
    if dest.exists():
        raise AnalysisError(f"destination already exists: {dest} (an export never overwrites; give a new directory)")
    if not dest.parent.is_dir():
        raise AnalysisError(f"parent of the destination does not exist: {dest.parent}")
    if not (isinstance(grid_s, (int, float)) and grid_s > 0 and math.isfinite(grid_s)):
        raise AnalysisError("the time grid must be a positive number of seconds")
    notes: dict[str, Any] = {}
    curves, ref, origin = read_curves(db_path, reference, run_ids, recorded_reference, bench_version, notes)
    synthetic = [c.run_id for c in curves if c.synthetic]
    if synthetic and not allow_synthetic:
        raise AnalysisError(
            f"{len(synthetic)} run(s) have synthetic results (fake evaluator of the operation tests): {synthetic[:3]}; "
            "they are not measurements. Pass allow_synthetic to analyse them in labeled columns"
        )
    if synthetic and len(synthetic) != len(curves):
        raise AnalysisError("synthetic and real runs are never analysed together")
    mark = bool(synthetic)

    grid = time_grid(curves, grid_s)
    value_names = ["hv", "hv_raw", "hv_sens"]
    tables: dict[str, tuple[list[str], list[str], list[dict[str, Any]]]] = {}

    def add(name: str, columns: list[str], rows: list[dict[str, Any]]) -> None:
        tables[name] = (columns, _label(columns, mark), rows)

    add(
        "hv_curvas.csv",
        [
            "run_id",
            "method",
            "seed_block",
            "step",
            "attempt_id",
            "candidate_id",
            "generation",
            "gpu_s",
            "unique",
            *value_names,
            "front",
            "sentinel_drift",
        ],
        curve_rows(curves),
    )
    add(
        "hv_grade_tempo.csv",
        ["run_id", "method", "seed_block", "gpu_s", *value_names, "run_ended", "unused_gpu_s"],
        time_grid_rows(curves, grid),
    )
    add(
        "hv_grade_avaliacoes.csv",
        ["run_id", "method", "seed_block", "unique", "gpu_s", *value_names],
        evaluation_grid_rows(curves),
    )
    finals = [
        "run_id",
        "method",
        "seed_block",
        "stage",
        "attempts",
        "valid",
        "unique_valid",
        "gpu_total_s",
        "budget_gpu_s",
        "unused_gpu_s",
        "excess_gpu_s",
        "front_size",
        "beyond_reference",
        "snapshot_mismatches",
        "sentinel_gpu_s",
        "sentinel_measurements",
        "sentinel_failures",
        "sentinel_drifts",
        "points_in_drift",
        "hv_final",
        "hv_raw_final",
        "hv_sens_final",
        "hv_at_budget",
        "hv_raw_at_budget",
        "hv_sens_at_budget",
    ]
    labeled = [f"{n}{SYNTHETIC_SUFFIX}" if mark and n.startswith("hv") else n for n in finals]
    tables["hv_final.csv"] = (finals, labeled, final_rows(curves))
    paired = paired_rows(curves)
    methods = [m for m in METHOD_ORDER if any(m in r for r in paired)]
    paired_cols = ["seed_block", *methods, *[f"joint_menos_{m}" for m in methods if m != "joint"]]
    tables["hv_pareado.csv"] = (
        paired_cols,
        [c if c == "seed_block" or not mark else f"{c}{SYNTHETIC_SUFFIX}" for c in paired_cols],
        paired,
    )
    sens_paired = paired_rows(curves, "hv_sens_final")  # same table at the sensitivity reference point
    tables["hv_pareado_sensibilidade.csv"] = (
        paired_cols,
        [c if c == "seed_block" or not mark else f"{c}{SYNTHETIC_SUFFIX}" for c in paired_cols],
        sens_paired,
    )
    add("hv_fora_do_ponto.csv", ["run_id", "method", "seed_block", "candidate_id"], outside_rows(curves))

    contents = {name: _csv(cols, rows, names) for name, (cols, names, rows) in tables.items()}
    manifest: dict[str, Any] = {
        "format": HV_FORMAT,
        "exported_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "database_file": Path(db_path).name,
        "reference_point": {"E": ref[0], "L_ms": ref[1], "V_mb": ref[2], "S_mb": ref[3], "origin": origin},
        "sensitivity_point": dict(zip(("E", "L_ms", "V_mb", "S_mb"), sensitivity_reference(ref))),
        "normalization": "hv = raw volume / prod(reference point), in [0, 1]; hv_raw is in E x ms x MB x MB",
        "grid_s": grid_s,
        "cut_rule": "the attempt at the budget finishes and counts: hv_final includes it, hv_at_budget does not",
        "synthetic": mark,
        "bench_version": notes.get("bench_version"),
        "left_out_by_bench_version": notes.get("left_out_by_bench_version", {}),
        "runs": [
            {
                "run_id": c.run_id,
                "method": c.method,
                "seed_block": c.seed_block,
                "unique_valid": c.unique,
                "gpu_total_s": c.gpu_total_s,
                "beyond_reference": len(c.outside),
                "sentinel_gpu_s": c.sentinel_gpu_s,
                "sentinel_measurements": c.sentinel_measurements,
                "sentinel_failures": c.sentinel_failures,
                "sentinel_drifts": c.sentinel_drifts,
            }
            for c in curves
        ],
        "sentinel_gpu_s_total": sum(c.sentinel_gpu_s for c in curves),
        "sentinel_note": "the time of the sentinel network is a control and is not part of the GPU budget of a run; it is declared here and per run (sentinel_gpu_s)",
        "snapshot_mismatches": [m for c in curves for m in c.snapshot_mismatches],
        "files": [
            {
                "name": name,
                "sha256": sha256_hex(data),
                "bytes": len(data),
                "rows": len(tables[name][2]),
                "columns": tables[name][1],
            }
            for name, data in contents.items()
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
    return HvExport(dest, manifest, tuple(curves))


def beyond_reference(objectives: Mapping[str, Sequence[float] | None], reference: Sequence[float]) -> tuple[str, ...]:
    """Ids of the valid candidates at or beyond the reference point (re-exported for the page)."""
    return hypervolume_of(objectives, reference).outside
