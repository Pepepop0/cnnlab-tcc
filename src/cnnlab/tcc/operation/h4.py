"""H4: does the cost of a network (MACs, parameters) explain its measured latency? (TCC-01 section 6; task A10c).

Read-only over the evaluations stored in a campaign database; no GPU, no test split.

**Points.** Each valid original evaluation is one point: its measured latency ``L`` (ms) and the MACs and parameters of the
network that became the engine, that is, *after* pruning. Where each number comes from is written next to it
(``macs_source``, ``parameters_source``): ``recorded`` when the evaluator stored it with the evaluation, ``recomputed``
when it was derived here, on the CPU, from the stored expressed genome (build the network, prune it with the genome's
``prune_pct``, count). The count depends on the architecture and the widths kept, not on which channels the pruning
chose, so the recomputed value is the one the engine had. ``parameters`` as the evaluator stored it before the pruning
counts the unpruned network, so it is not used for H4.

**Correlation.** Spearman's rho (ties take the mean rank) of L against MACs and against parameters, in two scopes: every
unique evaluation, and one point per *architecture with its numeric representation* (structure plus precisions; the
point takes the median L). Evaluations of one architecture are not independent, so the interval is a deterministic
cluster bootstrap that resamples whole architectures (1000 resamples, seed 0, percentile 2.5 / 97.5).

**Predictive model.** Ridge regression of ``log L`` on standardized features (log MACs, log parameters, share of fp16
and of int8 layers), with an intercept that is not penalized and ``alpha = 1``. It is validated **by group**: the
groups are the *effective architectures* (structure alone: block count, stem, activation, head and, per block, the width
kept after pruning, operator, kernel, convolutions, residual and downsample; precision, learning rate and the like are
not part of it), assigned to ``k = 5`` folds deterministically by the sorted hash of the architecture. Every point is
predicted by a model that never saw its architecture; reported out of group: R2 on ``log L``, Spearman of the
prediction, and the mean absolute error as a percentage of L. A model on log MACs alone, one on log MACs and parameters,
the full one, and the training median as a floor are compared, so what the extra features add is visible.
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
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cnnlab.tcc.operation.bench_versions import select_runs, version_of_rows
from cnnlab.tcc.science.expression import pruned_width
from cnnlab.tcc.storage.canonical import canonical_sha256, sha256_hex
from cnnlab.tcc.storage.errors import NotFoundError, StorageError
from cnnlab.tcc.storage.readmodel import open_read_model

H4_FORMAT = "tcc-h4/1"
MANIFEST_NAME = "manifesto.json"
SYNTHETIC_SUFFIX = "_SINTETICO"
FOLDS = 5
RIDGE_ALPHA = 1.0
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
MIN_POINTS = 8
RECORDED, RECOMPUTED = "recorded", "recomputed"
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "macs": ("log_macs",),
    "macs_parameters": ("log_macs", "log_parameters"),
    "full": ("log_macs", "log_parameters", "frac_fp16", "frac_int8"),
}


class H4Error(StorageError):
    """The analysis was refused; nothing was written."""


# -- architectures ------------------------------------------------------------------------------------------------


def _blocks(expressed: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = expressed.get("blocks")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise H4Error("the stored expressed genome has no blocks")
    n = expressed.get("n_blocks")
    blocks = [b for b in raw if isinstance(b, Mapping)]
    return blocks[:n] if isinstance(n, int) and 0 < n <= len(blocks) else blocks


def architecture_key(expressed: Mapping[str, Any]) -> tuple[Any, ...]:
    """Structure of the network the engine has: widths kept after pruning, no precision, no training genes."""
    blocks = _blocks(expressed)
    return (
        len(blocks),
        expressed.get("stem_channels"),
        expressed.get("activation"),
        expressed.get("head"),
        tuple(
            (
                pruned_width(int(b["channels"]), int(b.get("prune_pct") or 0)),
                b["op"],
                b["kernel"],
                b["n_convs"],
                bool(b["residual"]),
                b["downsample"],
            )
            for b in blocks
        ),
    )


def precision_key(expressed: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        expressed.get("stem_precision"),
        expressed.get("head_precision"),
        *(b.get("precision") for b in _blocks(expressed)),
    )


def architecture_hash(expressed: Mapping[str, Any]) -> str:
    return canonical_sha256(list(architecture_key(expressed)))


def representation_hash(expressed: Mapping[str, Any]) -> str:
    return canonical_sha256([list(architecture_key(expressed)), list(precision_key(expressed))])


def precision_shares(expressed: Mapping[str, Any]) -> tuple[float, float]:
    """Share of the layers (stem, blocks, head) in fp16 and in int8."""
    precisions = precision_key(expressed)
    n = len(precisions)
    return (sum(1 for p in precisions if p == "fp16") / n, sum(1 for p in precisions if p == "int8") / n)


def recompute_cost(expressed: Mapping[str, Any]) -> tuple[int, int]:
    """(MACs, parameters) of the pruned network, built on the CPU from the stored genome. Imports torch when called.

    The weights are random: the count depends on the architecture and the widths kept, not on which channels stay.
    """
    from cnnlab.tcc.model_pipeline.builder import build_model, count_macs
    from cnnlab.tcc.model_pipeline.pruning import prune_network
    from cnnlab.tcc.science.genome import ExpressedGenome

    genome = ExpressedGenome.from_dict(expressed)
    model, _ = build_model(genome, 0)
    report = prune_network(model, [b.prune_pct for b in genome.blocks])
    return int(count_macs(model)), int(report.parameters_after)


# -- points ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Point:
    run_id: str
    method: str
    seed_block: int
    candidate_id: str
    attempt_id: str
    architecture: str
    representation: str
    l_ms: float
    macs: int
    parameters: int
    macs_source: str
    parameters_source: str
    frac_fp16: float
    frac_int8: float
    synthetic: bool


@dataclass(frozen=True, slots=True)
class Collected:
    points: tuple[Point, ...]
    skipped: Mapping[str, int]  # reason -> count
    runs: tuple[str, ...]
    synthetic: bool
    bench_version: str | None = None
    left_out: Mapping[str, str | None] | None = None  # run -> its bench version, when one was chosen


def collect(
    db_path: Path,
    run_ids: Sequence[str] | None = None,
    *,
    recompute: bool = True,
    allow_synthetic: bool = False,
    cost: Callable[[Mapping[str, Any]], tuple[int, int]] | None = None,
    bench_version: str | None = None,
) -> Collected:
    """The valid original evaluations of the runs with their cost; nothing is written."""
    cost = cost or recompute_cost
    skipped: dict[str, int] = defaultdict(int)
    points: list[Point] = []
    cache: dict[str, tuple[int, int]] = {}
    with open_read_model(Path(db_path)) as rm:
        runs = [r for r in rm.runs() if run_ids is None or r["run_id"] in run_ids]
        if run_ids is not None and len(runs) != len(set(run_ids)):
            raise NotFoundError(f"runs not in the database: {sorted(set(run_ids) - {r['run_id'] for r in runs})}")
        if not runs:
            raise H4Error("the database has no runs to analyse")
        details = {r["run_id"]: rm.evaluation_details(r["run_id"]) for r in runs}
        keep, left_out = select_runs(
            {k: frozenset(str(d["bench_version"]) for d in v if d["bench_version"]) for k, v in details.items()},
            bench_version,
        )
        runs = [r for r in runs if r["run_id"] in keep]
        versions = {str(d["bench_version"]) for r in runs for d in details[r["run_id"]] if d["bench_version"]}
        genomes = {r["run_id"]: rm.expressed_genomes(r["run_id"]) for r in runs}
    synthetic = [r["run_id"] for r in runs if any(d["synthetic"] for d in details[r["run_id"]])]
    if synthetic and not allow_synthetic:
        raise H4Error(
            f"{len(synthetic)} run(s) have synthetic results (fake evaluator): {synthetic[:3]}; they are not "
            "measurements. Pass allow_synthetic to analyse them in labeled columns"
        )
    if synthetic and len(synthetic) != len(runs):
        raise H4Error("synthetic and real runs are never analysed together")
    for run in runs:
        for d in details[run["run_id"]]:
            expressed = genomes[run["run_id"]].get(d["candidate_id"])
            if expressed is None:
                skipped["genome_not_stored"] += 1
                continue
            if not d["l_ms"] or d["l_ms"] <= 0:
                skipped["no_latency"] += 1
                continue
            arch = architecture_hash(expressed)
            macs, params = d.get("macs"), d.get("parameters_pruned")
            macs_source = parameters_source = RECORDED
            if macs is None or params is None:
                if not recompute:
                    skipped["cost_not_recorded"] += 1
                    continue
                if arch not in cache:
                    cache[arch] = cost(expressed)
                rec_macs, rec_params = cache[arch]
                if macs is None:
                    macs, macs_source = rec_macs, RECOMPUTED
                if params is None:
                    params, parameters_source = rec_params, RECOMPUTED
            fp16, int8 = precision_shares(expressed)
            points.append(
                Point(
                    run["run_id"],
                    run["method"],
                    int(run["seed_block"]),
                    d["candidate_id"],
                    d["attempt_id"],
                    arch,
                    representation_hash(expressed),
                    float(d["l_ms"]),
                    int(macs),
                    int(params),
                    macs_source,
                    parameters_source,
                    fp16,
                    int8,
                    bool(d["synthetic"]),
                )
            )
    return Collected(
        tuple(points),
        dict(skipped),
        tuple(r["run_id"] for r in runs),
        bool(synthetic),
        next(iter(versions), None),
        left_out,
    )


# -- statistics -------------------------------------------------------------------------------------------------


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rho with mean ranks for ties; None when either side is constant or there are fewer than 2 points."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) != len(b):
        raise H4Error("the two series must have the same length")
    if len(a) < 2:
        return None
    ra, rb = _ranks(a) - (len(a) + 1) / 2, _ranks(b) - (len(b) + 1) / 2
    denominator = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return None if denominator == 0 else float((ra * rb).sum()) / denominator


def cluster_bootstrap_ci(
    clusters: Sequence[tuple[Sequence[float], Sequence[float]]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    """Percentile 95 % interval of Spearman's rho resampling whole clusters (architectures) with replacement."""
    if len(clusters) < 3:
        return None
    rng = np.random.default_rng(seed)
    xs = [np.asarray(c[0], dtype=float) for c in clusters]
    ys = [np.asarray(c[1], dtype=float) for c in clusters]
    values = []
    for _ in range(resamples):
        pick = rng.integers(0, len(clusters), len(clusters))
        rho = spearman(np.concatenate([xs[i] for i in pick]), np.concatenate([ys[i] for i in pick]))
        if rho is not None:
            values.append(rho)
    if not values:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def scope_points(points: Sequence[Point], scope: str) -> list[tuple[str, float, int, int]]:
    """(architecture, L, MACs, parameters) of the scope: every evaluation, or one per architecture with representation."""
    if scope == "todas_avaliacoes":
        return [(p.architecture, p.l_ms, p.macs, p.parameters) for p in points]
    if scope == "uma_por_arquitetura":
        groups: dict[str, list[Point]] = defaultdict(list)
        for p in points:
            groups[p.representation].append(p)
        out = []
        for key in sorted(groups):
            members = groups[key]
            ls = sorted(m.l_ms for m in members)
            median = (ls[len(ls) // 2] + ls[(len(ls) - 1) // 2]) / 2
            out.append((members[0].architecture, median, members[0].macs, members[0].parameters))
        return out
    raise H4Error(f"unknown scope {scope!r}")


SCOPES = ("todas_avaliacoes", "uma_por_arquitetura")


def correlations(points: Sequence[Point]) -> list[dict[str, Any]]:
    rows = []
    for scope in SCOPES:
        data = scope_points(points, scope)
        for name, index in (("macs", 2), ("parameters", 3)):
            xs = [d[index] for d in data]
            ls = [d[1] for d in data]
            by_arch: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
            for d in data:
                by_arch[d[0]][0].append(d[index])
                by_arch[d[0]][1].append(d[1])
            ci = cluster_bootstrap_ci([by_arch[k] for k in sorted(by_arch)])
            rows.append(
                {
                    "scope": scope,
                    "x": name,
                    "n": len(data),
                    "architectures": len(by_arch),
                    "spearman": spearman(xs, ls),
                    "ci95_low": None if ci is None else ci[0],
                    "ci95_high": None if ci is None else ci[1],
                }
            )
    return rows


def _features(points: Sequence[Point], names: Sequence[str]) -> np.ndarray:
    columns = {
        "log_macs": [math.log(p.macs) for p in points],
        "log_parameters": [math.log(p.parameters) for p in points],
        "frac_fp16": [p.frac_fp16 for p in points],
        "frac_int8": [p.frac_int8 for p in points],
    }
    return np.array([columns[n] for n in names], dtype=float).T


def ridge_fit_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = RIDGE_ALPHA
) -> np.ndarray:
    """Ridge on standardized features (training mean and spread), unpenalized intercept."""
    mean, spread = train_x.mean(axis=0), train_x.std(axis=0)
    spread = np.where(spread == 0, 1.0, spread)
    xs, ts = (train_x - mean) / spread, (test_x - mean) / spread
    y_mean = float(train_y.mean())
    weights = np.linalg.solve(xs.T @ xs + alpha * np.eye(xs.shape[1]), xs.T @ (train_y - y_mean))
    return y_mean + ts @ weights


def assign_folds(architectures: Sequence[str], k: int = FOLDS) -> dict[str, int]:
    """Deterministic fold of every architecture: sorted by hash, dealt round-robin."""
    unique = sorted(set(architectures))
    return {a: i % k for i, a in enumerate(unique)}


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    sse, sst = float(((y - prediction) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {
        "r2_log_l": None if sst == 0 else 1.0 - sse / sst,
        "spearman": spearman(prediction, y),
        "mae_pct": float(np.mean(np.abs(np.exp(prediction) - np.exp(y)) / np.exp(y))) * 100.0,
    }


def grouped_validation(points: Sequence[Point], k: int = FOLDS) -> dict[str, Any]:
    """Out-of-group predictions of log L by every feature set and by the training median."""
    if len(points) < MIN_POINTS:
        raise H4Error(f"at least {MIN_POINTS} points are needed for the model, got {len(points)}")
    folds = assign_folds([p.architecture for p in points], k)
    k = min(k, len(set(folds.values())))
    if k < 2:
        raise H4Error("the model needs at least two architectures to validate by group")
    fold_of = np.array([folds[p.architecture] % k for p in points])
    y = np.log(np.array([p.l_ms for p in points], dtype=float))
    predictions: dict[str, np.ndarray] = {name: np.zeros(len(points)) for name in (*FEATURE_SETS, "mediana")}
    for fold in range(k):
        test, train = fold_of == fold, fold_of != fold
        predictions["mediana"][test] = float(np.median(y[train]))
        for name, features in FEATURE_SETS.items():
            x = _features(points, features)
            predictions[name][test] = ridge_fit_predict(x[train], y[train], x[test])
    models, fold_rows = [], []
    for name, prediction in predictions.items():
        models.append(
            {
                "model": name,
                "features": " ".join(FEATURE_SETS.get(name, ())),
                "n": len(points),
                "architectures": len(set(folds)),
                "folds": k,
                **_metrics(y, prediction),
            }
        )
        for fold in range(k):
            sel = fold_of == fold
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "n": int(sel.sum()),
                    "architectures": len({p.architecture for p, s in zip(points, sel) if s}),
                    **_metrics(y[sel], prediction[sel]),
                }
            )
    prediction_rows = [
        {
            "run_id": p.run_id,
            "candidate_id": p.candidate_id,
            "attempt_id": p.attempt_id,
            "architecture": p.architecture,
            "fold": int(fold_of[i]),
            "l_ms": p.l_ms,
            **{f"pred_{name}": float(np.exp(predictions[name][i])) for name in predictions},
        }
        for i, p in enumerate(points)
    ]
    return {"models": models, "folds": fold_rows, "predictions": prediction_rows}


# -- export ------------------------------------------------------------------------------------------------------


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return value


def _csv(columns: Sequence[str], rows: Sequence[Mapping[str, Any]], names: Sequence[str] | None = None) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(names or columns)
    for row in rows:
        writer.writerow([_cell(row.get(c)) for c in columns])
    return buffer.getvalue().encode("utf-8")


def export_h4(
    db_path: Path,
    dest: Path,
    *,
    run_ids: Sequence[str] | None = None,
    recompute: bool = True,
    allow_synthetic: bool = False,
    cost: Callable[[Mapping[str, Any]], tuple[int, int]] | None = None,
    bench_version: str | None = None,
) -> dict[str, Any]:
    """Write the tables and the manifest of the analysis into the new directory ``dest``; returns the manifest."""
    dest = Path(dest)
    if dest.exists():
        raise H4Error(f"destination already exists: {dest} (an export never overwrites; give a new directory)")
    if not dest.parent.is_dir():
        raise H4Error(f"parent of the destination does not exist: {dest.parent}")
    got = collect(
        db_path, run_ids, recompute=recompute, allow_synthetic=allow_synthetic, cost=cost, bench_version=bench_version
    )
    points = list(got.points)
    if len(points) < 3:
        raise H4Error(f"{len(points)} usable point(s) (skipped: {dict(got.skipped) or 'none'}): too few to analyse")
    mark = got.synthetic

    def labeled(names: Sequence[str], pattern: tuple[str, ...]) -> list[str]:
        return [f"{n}{SYNTHETIC_SUFFIX}" if mark and any(n.startswith(p) for p in pattern) else n for n in names]

    point_cols = [
        "run_id",
        "method",
        "seed_block",
        "candidate_id",
        "attempt_id",
        "architecture",
        "representation",
        "l_ms",
        "macs",
        "macs_source",
        "parameters",
        "parameters_source",
        "frac_fp16",
        "frac_int8",
    ]
    corr = correlations(points)
    corr_cols = ["scope", "x", "n", "architectures", "spearman", "ci95_low", "ci95_high"]
    tables: dict[str, tuple[list[str], list[str], list[dict[str, Any]]]] = {
        "h4_pontos.csv": (
            point_cols,
            labeled(point_cols, ("l_ms",)),
            [{**{c: getattr(p, c) for c in point_cols}} for p in points],
        ),
        "h4_correlacao.csv": (corr_cols, labeled(corr_cols, ("spearman", "ci95")), corr),
    }
    model_note = None
    try:
        validation = grouped_validation(points)
    except H4Error as exc:
        validation, model_note = None, str(exc)
    if validation is not None:
        model_cols = ["model", "features", "n", "architectures", "folds", "r2_log_l", "spearman", "mae_pct"]
        fold_cols = ["model", "fold", "n", "architectures", "r2_log_l", "spearman", "mae_pct"]
        pred_cols = [
            "run_id",
            "candidate_id",
            "attempt_id",
            "architecture",
            "fold",
            "l_ms",
            *[c for c in validation["predictions"][0] if c.startswith("pred_")],
        ]
        tables["h4_modelo.csv"] = (model_cols, labeled(model_cols, ("r2", "spearman", "mae")), validation["models"])
        tables["h4_modelo_dobras.csv"] = (fold_cols, labeled(fold_cols, ("r2", "spearman", "mae")), validation["folds"])
        tables["h4_predicoes.csv"] = (pred_cols, labeled(pred_cols, ("l_ms", "pred_")), validation["predictions"])
    contents = {name: _csv(cols, rows, names) for name, (cols, names, rows) in tables.items()}
    sources = defaultdict(int)
    for p in points:
        sources[f"macs_{p.macs_source}"] += 1
        sources[f"parameters_{p.parameters_source}"] += 1
    manifest: dict[str, Any] = {
        "format": H4_FORMAT,
        "exported_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "database_file": Path(db_path).name,
        "runs": list(got.runs),
        "synthetic": mark,
        "bench_version": got.bench_version,
        "left_out_by_bench_version": dict(got.left_out or {}),
        "points": len(points),
        "skipped": dict(got.skipped),
        "cost_sources": dict(sources),
        "model_note": model_note,
        "method": {
            "points": "valid original evaluations; MACs and parameters of the pruned network (recorded, else recomputed on the CPU "
            "from the stored genome: build, prune with prune_pct, count)",
            "correlation": f"Spearman, mean ranks for ties; interval = cluster bootstrap by architecture, {BOOTSTRAP_RESAMPLES} "
            f"resamples, seed {BOOTSTRAP_SEED}, percentile 2.5/97.5",
            "model": f"ridge (alpha {RIDGE_ALPHA}) on standardized features, target log L; features {FEATURE_SETS}; "
            f"{FOLDS} folds by effective architecture (sorted hash dealt round-robin); the median is the floor",
            "architecture": "structure with the widths kept after pruning; precision and training genes are not part of it",
        },
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
    return manifest
