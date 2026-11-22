"""The statistics of the confirmatory campaign: H2, H3 and the H1 table (TCC-01 section 6; task A10b).

Pure computation over small tables, written here because the environment has no scipy and a test of eight pairs does
not need one. Everything is exact or says which tolerance it uses.

**Paired permutation test (H2, H3).** The unit is the seed block: eight pairs ``d_b = HV(joint, b) - HV(control, b)``.
Under the null the sign of every difference is arbitrary, so the test looks at the ``2**n`` sign assignments (256 for
eight pairs) and counts how many give a sum at least as large as the observed one (one-sided, ``joint > control``).
``p = count / 2**n``, the observed assignment included, so the smallest possible p for eight pairs is 1/256. Ties,
zero differences included, keep every sign pattern: they are counted, not dropped. Sums equal to the observed one within
a relative tolerance of 1e-12 count as ties (they are equal; the tolerance only absorbs floating-point noise).

**Holm.** The family of H2 and H3 (two tests, alpha 0.05): sorted p-values multiplied by ``m, m-1, ...``, made
monotone and capped at 1; a test is rejected when its adjusted p is at most alpha.

**Interval.** The 95 % confidence interval of the mean paired difference is obtained by *inverting the same exact
test*: it is the set of shifts ``delta`` for which the two-sided sign-flip test of ``d - delta`` does not reject at 5 %.
It is exact (no normality), consistent with the test above, and found by scanning the shifts where the p-value can
change. For eight pairs it is wide: that is the information in eight seeds.

**H1 table.** For each seed block, the retrained H1 candidate against the retrained mono champion, from a CSV of
final results (official-test top-1 accuracy and the L, V and S of the final engines): the accuracy difference in
percentage points, the relative reductions of L, V and S, and whether the block supports H1 (all three reduced and an
accuracy loss of at most 1 pp). The 2 pp and 5 pp thresholds are reported as sensitivity, not as a way to choose.
Blocks without a candidate are listed as failures, never filled in.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import shutil
import statistics
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cnnlab.tcc.storage.canonical import sha256_hex
from cnnlab.tcc.storage.errors import StorageError

INFERENCE_FORMAT = "tcc-inference/1"
MANIFEST_NAME = "manifesto.json"
SYNTHETIC_SUFFIX = "_SINTETICO"
ALPHA = 0.05
TIE_TOLERANCE = 1e-12
MAX_PAIRS = 20  # 2**20 sign patterns: beyond it the exact test is not what is wanted
H1_THRESHOLDS_PP = (1.0, 2.0, 5.0)  # the protocol's 1 pp, and the 2 and 5 pp sensitivity thresholds
H1_SEEDS_FOR_SUPPORT = 6  # descriptive support criterion of the protocol: 6 of 8 seeds
DEFAULT_TESTS: tuple[tuple[str, str, str], ...] = (("H2", "joint", "random"), ("H3", "joint", "sequential"))
METHOD_ORDER = ("joint", "mono", "random", "sequential")


class InferenceError(StorageError):
    """The analysis was refused; nothing was written."""


# -- the exact test ---------------------------------------------------------------------------------------------


def _signs(n: int) -> np.ndarray:
    if not 1 <= n <= MAX_PAIRS:
        raise InferenceError(f"the exact sign-flip test takes 1 to {MAX_PAIRS} pairs, not {n}")
    return np.array(list(itertools.product((-1.0, 1.0), repeat=n)))


def _tolerance(values: np.ndarray) -> float:
    return TIE_TOLERANCE * max(1.0, float(np.abs(values).sum()))


def sign_flip_p(differences: Sequence[float], alternative: str = "greater") -> float:
    """Exact p-value of the paired sign-flip test of the mean being 0 (see the module docstring)."""
    if alternative not in ("greater", "less", "two-sided"):
        raise InferenceError("alternative must be 'greater', 'less' or 'two-sided'")
    d = np.asarray(differences, dtype=float)
    if not np.all(np.isfinite(d)):
        raise InferenceError("the differences must be finite numbers")
    sums = _signs(len(d)) @ d
    observed, tol = float(d.sum()), _tolerance(d)
    if alternative == "greater":
        hits = sums >= observed - tol
    elif alternative == "less":
        hits = sums <= observed + tol
    else:
        hits = np.abs(sums) >= abs(observed) - tol
    return float(hits.sum()) / len(sums)


def holm(p_values: Mapping[str, float], alpha: float = ALPHA) -> dict[str, dict[str, Any]]:
    """Holm step-down adjustment of a family of tests."""
    names = sorted(p_values, key=lambda k: (p_values[k], k))
    m = len(names)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (m - rank) * p_values[name]))
        adjusted[name] = running
    return {
        name: {"p": p_values[name], "p_holm": adjusted[name], "reject": adjusted[name] <= alpha} for name in p_values
    }


def exact_interval(differences: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """The set of shifts the two-sided sign-flip test does not reject at ``1 - confidence`` (see the docstring)."""
    d = np.asarray(differences, dtype=float)
    n = len(d)
    signs = _signs(n)
    a = signs @ d  # the sum of every pattern
    s = signs.sum(axis=1)
    total = float(d.sum())
    roots: list[float] = []
    for pattern_sum, count in zip(a, s):  # |A - delta S| = |D - n delta|: the two linear branches
        if n - count != 0:
            roots.append((total - pattern_sum) / (n - count))
        if n + count != 0:
            roots.append((pattern_sum + total) / (n + count))
    roots = sorted(set(roots))
    scale = max(1.0, float(np.abs(d).max()))
    points = [roots[0] - scale, roots[-1] + scale, *roots]
    points += [(x + y) / 2 for x, y in zip(roots, roots[1:])]
    alpha = 1.0 - confidence
    accepted = []
    for delta in points:
        shifted = d - delta
        sums = signs @ shifted
        observed = abs(float(shifted.sum()))
        p = float((np.abs(sums) >= observed - _tolerance(shifted)).sum()) / len(sums)
        if p > alpha + 1e-12:
            accepted.append(delta)
    if not accepted:
        raise InferenceError("no shift is accepted: the interval is empty")
    return (float(min(accepted)), float(max(accepted)))


def describe(differences: Sequence[float]) -> dict[str, Any]:
    d = list(map(float, differences))
    return {
        "n": len(d),
        "mean": statistics.fmean(d),
        "median": statistics.median(d),
        "min": min(d),
        "max": max(d),
        "sd": statistics.stdev(d) if len(d) > 1 else None,
        "positive": sum(1 for x in d if x > 0),
        "negative": sum(1 for x in d if x < 0),
        "zero": sum(1 for x in d if x == 0),
    }


@dataclass(frozen=True, slots=True)
class PairedResult:
    hypothesis: str
    treatment: str
    control: str
    blocks: tuple[int, ...]
    differences: tuple[float, ...]
    stats: Mapping[str, Any]
    ci_low: float
    ci_high: float
    p: float
    p_holm: float
    reject: bool


def paired_tests(
    values: Mapping[str, Mapping[int, float]],
    tests: Sequence[tuple[str, str, str]] = DEFAULT_TESTS,
    *,
    alpha: float = ALPHA,
) -> list[PairedResult]:
    """The family of paired one-sided tests (``treatment > control``) with Holm over the whole family.

    ``values[method][block]`` is the value of a method in a seed block. Only the blocks both methods have are paired,
    and a block missing for one of them is reported by the caller, not skipped silently here: the list of blocks is in
    the result.
    """
    raw: dict[str, tuple[str, str, tuple[int, ...], tuple[float, ...]]] = {}
    for hypothesis, treatment, control in tests:
        for name in (treatment, control):
            if name not in values:
                raise InferenceError(f"no values for method {name!r}")
        blocks = tuple(sorted(set(values[treatment]) & set(values[control])))
        if not blocks:
            raise InferenceError(f"{treatment} and {control} have no seed block in common")
        raw[hypothesis] = (treatment, control, blocks, tuple(values[treatment][b] - values[control][b] for b in blocks))
    p_values = {h: sign_flip_p(diffs, "greater") for h, (_, _, _, diffs) in raw.items()}
    adjusted = holm(p_values, alpha)
    out = []
    for hypothesis, (treatment, control, blocks, diffs) in raw.items():
        low, high = exact_interval(diffs)
        out.append(
            PairedResult(
                hypothesis,
                treatment,
                control,
                blocks,
                diffs,
                describe(diffs),
                low,
                high,
                p_values[hypothesis],
                adjusted[hypothesis]["p_holm"],
                bool(adjusted[hypothesis]["reject"]),
            )
        )
    return out


# -- H1 ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class H1Row:
    seed_block: int
    status: str  # "ok", "sem_candidata_H1", "sem_campeao_mono"
    delta_pp: float | None = None
    acc_h1: float | None = None
    acc_mono: float | None = None
    reductions: Mapping[str, float] | None = None  # 1 - H1 / mono, for l_ms, v_mb, s_mb
    h1: Mapping[str, float] | None = None
    mono: Mapping[str, float] | None = None

    def supports(self, threshold_pp: float) -> bool:
        """All of L, V and S reduced, and no more than ``threshold_pp`` points of accuracy lost."""
        return (
            self.status == "ok"
            and self.delta_pp is not None
            and self.reductions is not None
            and all(v > 0 for v in self.reductions.values())
            and self.delta_pp >= -threshold_pp - 1e-12
        )


def h1_table(rows: Sequence[Mapping[str, Any]], expected_blocks: Sequence[int] = tuple(range(8))) -> list[H1Row]:
    """One row per seed block from the final results: the joint H1 candidate against the mono champion."""

    def pick(block: int, method: str, role: str) -> Mapping[str, Any] | None:
        found = [r for r in rows if int(r["seed_block"]) == block and r["method"] == method and r["role"] == role]
        if len(found) > 1:
            raise InferenceError(f"seed block {block}: more than one {method}/{role} in the final results")
        return found[0] if found else None

    out = []
    for block in sorted(set(expected_blocks) | {int(r["seed_block"]) for r in rows}):
        h1, mono = pick(block, "joint", "H1"), pick(block, "mono", "menor_E")
        if h1 is None:
            out.append(H1Row(block, "sem_candidata_H1"))
            continue
        if mono is None:
            out.append(H1Row(block, "sem_campeao_mono"))
            continue
        a1, a0 = float(h1["test_accuracy"]), float(mono["test_accuracy"])
        for name, a in (("H1", a1), ("mono", a0)):
            if not 0.0 <= a <= 1.0:
                raise InferenceError(f"seed block {block}: {name} accuracy {a} is not a fraction between 0 and 1")
        h, m = ({k: float(r[k]) for k in ("l_ms", "v_mb", "s_mb")} for r in (h1, mono))
        out.append(H1Row(block, "ok", 100.0 * (a1 - a0), a1, a0, {k: 1.0 - h[k] / m[k] for k in h}, h, m))
    return out


def h1_summary(rows: Sequence[H1Row]) -> dict[str, Any]:
    ok = [r for r in rows if r.status == "ok"]
    deltas = [r.delta_pp for r in ok if r.delta_pp is not None]
    summary: dict[str, Any] = {
        "blocks": len(rows),
        "with_both": len(ok),
        "failures": [{"seed_block": r.seed_block, "status": r.status} for r in rows if r.status != "ok"],
        "supporting": {f"{t:g}pp": sum(1 for r in rows if r.supports(t)) for t in H1_THRESHOLDS_PP},
        "support_criterion": f"at least {H1_SEEDS_FOR_SUPPORT} of {len(rows)} seeds (descriptive, at the 1 pp threshold)",
    }
    summary["criterion_met"] = summary["supporting"]["1pp"] >= H1_SEEDS_FOR_SUPPORT
    if deltas:
        summary["delta_pp"] = describe(deltas)
        for key in ("l_ms", "v_mb", "s_mb"):
            summary[f"reduction_{key}"] = describe([r.reductions[key] for r in ok])  # type: ignore[index]
    return summary


# -- reading and writing -------------------------------------------------------------------------------------------


def load_paired(path: Path) -> tuple[dict[str, dict[int, float]], bool]:
    """``hv_pareado.csv`` of the analysis: values[method][block]; the flag says its numbers are synthetic."""
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    synthetic = any(name.endswith(SYNTHETIC_SUFFIX) for name in header)
    values: dict[str, dict[int, float]] = {}
    for name in header:
        method = name.removesuffix(SYNTHETIC_SUFFIX)
        if method not in METHOD_ORDER:
            continue
        values[method] = {int(r["seed_block"]): float(r[name]) for r in rows if r[name] != ""}
    if not values:
        raise InferenceError(f"{path}: no method column (one of {METHOD_ORDER}) found")
    return values, synthetic


def final_results_synthetic(path: Path) -> bool:
    """Whether the final results file carries the ``_SINTETICO`` label (columns written by ``final_stage.export_final``)."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh), [])
    return any(name.endswith(SYNTHETIC_SUFFIX) for name in header)


def load_final_results(path: Path) -> list[dict[str, Any]]:
    """The final results CSV: seed_block, method, role, test_accuracy (fraction), l_ms, v_mb, s_mb.

    Columns labeled ``_SINTETICO`` are read under their plain names; ``final_results_synthetic`` says they were labeled.
    """
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = [
            {(k.removesuffix(SYNTHETIC_SUFFIX) if k else k): v for k, v in row.items()} for row in csv.DictReader(fh)
        ]
    needed = {"seed_block", "method", "role", "test_accuracy", "l_ms", "v_mb", "s_mb"}
    missing = needed - set(rows[0] if rows else needed)
    if missing:
        raise InferenceError(f"{path}: missing columns {sorted(missing)}")
    return rows


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return value


def _csv(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row.get(c)) for c in columns])
    return buffer.getvalue().encode("utf-8")


def result_rows(results: Sequence[PairedResult], mark: bool) -> list[dict[str, Any]]:
    suffix = SYNTHETIC_SUFFIX if mark else ""
    return [
        {
            "hypothesis": r.hypothesis,
            "comparison": f"{r.treatment} > {r.control}",
            "pairs": r.stats["n"],
            f"mean_diff{suffix}": r.stats["mean"],
            f"median_diff{suffix}": r.stats["median"],
            f"min_diff{suffix}": r.stats["min"],
            f"max_diff{suffix}": r.stats["max"],
            f"sd_diff{suffix}": r.stats["sd"],
            f"ci95_low{suffix}": r.ci_low,
            f"ci95_high{suffix}": r.ci_high,
            "better": r.stats["positive"],
            "worse": r.stats["negative"],
            "equal": r.stats["zero"],
            f"p{suffix}": r.p,
            f"p_holm{suffix}": r.p_holm,
            "reject_at_5pct": r.reject,
        }
        for r in results
    ]


def write_analysis(
    dest: Path,
    *,
    paired: Path | None = None,
    sensitivity: Path | None = None,
    final_results: Path | None = None,
    tests: Sequence[tuple[str, str, str]] = DEFAULT_TESTS,
    expected_blocks: Sequence[int] = tuple(range(8)),
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Write H2/H3 and/or the H1 table into the new directory ``dest`` with a manifest; returns it."""
    dest = Path(dest)
    if dest.exists():
        raise InferenceError(f"destination already exists: {dest} (an export never overwrites; give a new directory)")
    if not dest.parent.is_dir():
        raise InferenceError(f"parent of the destination does not exist: {dest.parent}")
    if paired is None and final_results is None:
        raise InferenceError("give the paired hypervolumes (--paired) and/or the final results (--final-results)")
    contents: dict[str, bytes] = {}
    synthetic = False
    sources: dict[str, str] = {}
    summary: dict[str, Any] = {}
    if paired is not None:
        for label, path in (("official", paired), ("sensitivity", sensitivity)):
            if path is None:
                continue
            values, is_synthetic = load_paired(path)
            synthetic = synthetic or is_synthetic
            sources[label] = sha256_hex(Path(path).read_bytes())
            results = paired_tests(values, tests)
            suffix = "" if label == "official" else "_sensibilidade"
            columns = [
                "hypothesis",
                "comparison",
                "pairs",
                "mean_diff",
                "median_diff",
                "min_diff",
                "max_diff",
                "sd_diff",
                "ci95_low",
                "ci95_high",
                "better",
                "worse",
                "equal",
                "p",
                "p_holm",
                "reject_at_5pct",
            ]
            mark = is_synthetic
            names = [
                f"{c}{SYNTHETIC_SUFFIX}"
                if mark and c not in ("hypothesis", "comparison", "pairs", "better", "worse", "equal", "reject_at_5pct")
                else c
                for c in columns
            ]
            rows = result_rows(results, mark)
            contents[f"h2_h3{suffix}.csv"] = _csv(names, [{n: r[n] for n in names} for r in rows])
            pair_rows = [
                {"hypothesis": r.hypothesis, "seed_block": b, f"diff{SYNTHETIC_SUFFIX if mark else ''}": d}
                for r in results
                for b, d in zip(r.blocks, r.differences)
            ]
            pair_columns = ["hypothesis", "seed_block", f"diff{SYNTHETIC_SUFFIX if mark else ''}"]
            contents[f"h2_h3_pares{suffix}.csv"] = _csv(pair_columns, pair_rows)
            summary[label] = [
                {"hypothesis": r.hypothesis, "p": r.p, "p_holm": r.p_holm, "reject": r.reject, "pairs": r.stats["n"]}
                for r in results
            ]
    if final_results is not None:
        rows_in = load_final_results(final_results)
        synthetic = synthetic or final_results_synthetic(final_results)
        sources["final_results"] = sha256_hex(Path(final_results).read_bytes())
        table = h1_table(rows_in, expected_blocks)
        columns = [
            "seed_block",
            "status",
            "delta_pp",
            "acc_h1",
            "acc_mono",
            "l_ms_h1",
            "l_ms_mono",
            "l_reduction",
            "v_mb_h1",
            "v_mb_mono",
            "v_reduction",
            "s_mb_h1",
            "s_mb_mono",
            "s_reduction",
            "supports_1pp",
            "supports_2pp",
            "supports_5pp",
        ]
        out_rows = []
        for r in table:
            row: dict[str, Any] = {
                "seed_block": r.seed_block,
                "status": r.status,
                "delta_pp": r.delta_pp,
                "acc_h1": r.acc_h1,
                "acc_mono": r.acc_mono,
            }
            if r.status == "ok":
                for key, short in (("l_ms", "l_ms"), ("v_mb", "v_mb"), ("s_mb", "s_mb")):
                    row[f"{short}_h1"], row[f"{short}_mono"] = r.h1[key], r.mono[key]  # type: ignore[index]
                row["l_reduction"], row["v_reduction"], row["s_reduction"] = (
                    r.reductions[k] for k in ("l_ms", "v_mb", "s_mb")
                )  # type: ignore[index]
            for t in (1, 2, 5):
                row[f"supports_{t}pp"] = r.supports(float(t))
            out_rows.append(row)
        contents["h1_tabela.csv"] = _csv(columns, out_rows)
        summary["h1"] = h1_summary(table)
        contents["h1_resumo.json"] = (json.dumps(summary["h1"], indent=2, sort_keys=True) + "\n").encode("utf-8")
    if synthetic and not allow_synthetic:
        raise InferenceError(
            "the paired hypervolumes are synthetic (fake evaluator): they are not measurements; "
            "pass allow_synthetic to analyse them, in labeled columns"
        )
    manifest: dict[str, Any] = {
        "format": INFERENCE_FORMAT,
        "exported_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "synthetic": synthetic,
        "alpha": ALPHA,
        "tie_tolerance": TIE_TOLERANCE,
        "method": {
            "test": "exact paired sign-flip (permutation) test, one-sided, treatment > control, 2**n sign patterns, observed included",
            "family": "Holm over the tests listed (H2 and H3)",
            "interval": "95 % interval of the mean paired difference by inversion of the exact two-sided sign-flip test",
            "h1": "retrained H1 candidate against the retrained mono champion of the same seed block, official-test accuracy",
        },
        "tests": [list(t) for t in tests],
        "sources_sha256": sources,
        "summary": summary,
        "files": [{"name": n, "sha256": sha256_hex(d), "bytes": len(d)} for n, d in contents.items()],
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


__all__ = [
    "InferenceError",
    "PairedResult",
    "H1Row",
    "sign_flip_p",
    "holm",
    "exact_interval",
    "describe",
    "paired_tests",
    "h1_table",
    "h1_summary",
    "load_paired",
    "load_final_results",
    "final_results_synthetic",
    "write_analysis",
]
