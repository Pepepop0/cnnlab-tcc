"""Exploratory slices of the campaign under cost limits, for the preliminary results of the monograph.

For each limit on L, V or S and each method: the best search accuracy found by each run, and the hypervolume of the
region below the limit (the reference point moved to the limit on that objective). Joint against each control is
compared per seed with the exact sign-flip test, Holm-corrected over the whole grid. Reads the campaign database only;
nothing here touches the official test. Run with PYTHONPATH=<frozen worktree>/src.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sqlite3
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from cnnlab.tcc.science.hypervolume import normalized_hypervolume

M = ("joint", "mono", "random", "sequential")
CONTROLS = ("mono", "random", "sequential")
INDEX = {"L": 1, "V": 2, "S": 3}
LIMITS = [
    ("L", 0.03),
    ("L", 0.04),
    ("L", 0.05),
    ("L", 0.075),
    ("L", 0.1),
    ("S", 0.5),
    ("S", 1.0),
    ("S", 2.0),
    ("S", 4.0),
    ("V", 1.0),
    ("V", 2.0),
    ("V", 4.0),
]


def method_block(run_id: str) -> tuple[str, int] | None:
    m = re.search(r"-(joint|mono|random|sequential)-b(\d+)$", run_id)
    return (m.group(1), int(m.group(2))) if m else None


def sign_flip(diffs: list[float]) -> float:
    """Exact two-sided sign-flip test on the mean: these slices have no predicted direction."""
    observed = abs(sum(diffs))
    hits = sum(
        abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12
        for signs in itertools.product((1, -1), repeat=len(diffs))
    )
    return hits / 2 ** len(diffs)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=ps.__getitem__)
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path(r"C:\Users\Pedro\Documents\cnnlab-tcc-runs"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    af = a.runs / "analise-final-2026-10-22"
    ref_info = json.loads((af / "manifesto.json").read_text(encoding="utf-8"))["reference_point"]
    con = sqlite3.connect(f"file:{a.runs / 'campanha-2026-10' / 'db' / 'tcc.sqlite3'}?mode=ro", uri=True)
    points: dict[tuple[str, int], list[tuple[float, float, float, float]]] = defaultdict(list)
    for run, e, l, v, s in con.execute("""select a.run_id, m.e_fraction, m.l_ms, m.v_mb, m.s_mb from measurements m
            join evaluation_attempts a on a.attempt_id = m.attempt_id
            where a.status = 'valid' and a.run_id like 'camp-%'"""):
        points[method_block(run)].append((e, l, v, s))  # type: ignore[index]

    reference = [float(ref_info[k]) for k in ("E", "L_ms", "V_mb", "S_mb")]

    # The slices are only as good as this: all valid points against the official final hypervolume of each run.
    worst = 0.0
    with open(af / "hv_final.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["method"], int(row["seed_block"]))
            worst = max(worst, abs(normalized_hypervolume(points[key], reference) - float(row["hv_final"])))

    grid = []
    for objective, limit in LIMITS:
        k = INDEX[objective]
        ref = list(reference)
        ref[k] = limit
        acc = {m: [max([1 - p[0] for p in points[(m, b)] if p[k] <= limit] or [0.0]) for b in range(8)] for m in M}
        hv = {m: [normalized_hypervolume(points[(m, b)], ref) for b in range(8)] for m in M}
        found = {m: sum(any(p[k] <= limit for p in points[(m, b)]) for b in range(8)) for m in M}
        count = {m: st.mean(sum(p[k] <= limit for p in points[(m, b)]) for b in range(8)) for m in M}
        entry = {
            "objective": objective,
            "limit": limit,
            "acc_mean": {m: st.mean(acc[m]) for m in M},
            "hv_mean": {m: st.mean(hv[m]) for m in M},
            "seeds_with_network": found,
            "mean_networks": count,
            "pairs": {},
        }
        for c in CONTROLS:
            d_acc = [x - y for x, y in zip(acc["joint"], acc[c])]
            d_hv = [x - y for x, y in zip(hv["joint"], hv[c])]
            entry["pairs"][c] = {
                "acc_diff": st.mean(d_acc),
                "acc_wins": sum(d > 0 for d in d_acc),
                "acc_losses": sum(d < 0 for d in d_acc),
                "acc_p": sign_flip(d_acc),
                "hv_diff": st.mean(d_hv),
                "hv_wins": sum(d > 0 for d in d_hv),
                "hv_losses": sum(d < 0 for d in d_hv),
                "hv_p": sign_flip(d_hv),
            }
        grid.append(entry)
    for measure in ("acc_p", "hv_p"):
        cells = [(i, c) for i in range(len(grid)) for c in CONTROLS]
        adjusted = holm([grid[i]["pairs"][c][measure] for i, c in cells])
        for (i, c), p in zip(cells, adjusted):
            grid[i]["pairs"][c][measure + "_holm"] = p
    out = {"reference": reference, "max_abs_hv_error_vs_official": worst, "grid": grid}
    (a.out / "recortes.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("reference", reference, "max hv error vs official", worst)
    for g in grid:
        acc = " ".join(f"{m[:3]}={100 * g['acc_mean'][m]:.1f}({g['seeds_with_network'][m]})" for m in M)
        hv = " ".join(f"{m[:3]}={g['hv_mean'][m]:.4f}" for m in M)
        pr = " ".join(
            f"{c[:3]}:acc {g['pairs'][c]['acc_wins']}/{g['pairs'][c]['acc_losses']} p={g['pairs'][c]['acc_p']:.3f}"
            f"(H{g['pairs'][c]['acc_p_holm']:.2f}) hv {g['pairs'][c]['hv_wins']}/{g['pairs'][c]['hv_losses']}"
            f" p={g['pairs'][c]['hv_p']:.3f}(H{g['pairs'][c]['hv_p_holm']:.2f})"
            for c in CONTROLS
        )
        print(f"{g['objective']}<={g['limit']}: {acc} | {hv}\n    {pr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
