"""Final analysis of the campaign for the monograph: figure data (.dat for pgfplots), numbers (JSON) and a report.

Reads only: the campaign database (read-only URI), the official exports of `campaign curves`, `campaign inference`,
`campaign h4` and `final export`, and the reference networks. Nothing here opens the official test or writes to the
database. Confirmatory numbers come from the official exports; everything else is labeled exploratory in the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

M = ("joint", "mono", "random", "sequential")
NAME = {"joint": "Conjunto", "mono": "Mono-objetivo", "random": "Aleatório", "sequential": "Sequencial"}


def rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def method_block(run_id: str) -> tuple[str, int] | None:
    m = re.search(r"-(joint|mono|random|sequential)-b(\d+)$", run_id)
    return (m.group(1), int(m.group(2))) if m else None


def write_dat(path: Path, header: list[str], data: list[list[float | str]]) -> None:
    def fmt(v: float | str) -> str:
        if isinstance(v, str):
            return v
        return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.6g}"

    path.write_text(
        " ".join(header) + "\n" + "\n".join(" ".join(fmt(v) for v in r) for r in data) + "\n", encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path(r"C:\Users\Pedro\Documents\cnnlab-tcc-runs"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    R, out = a.runs, a.out
    out.mkdir(parents=True, exist_ok=True)
    AF, Q = R / "analise-final-2026-10-22", R / "campanha-2026-10"
    N: dict = {}

    # -- volume and failures (database, read only) ------------------------------------------------------------------
    con = sqlite3.connect(f"file:{Q / 'db' / 'tcc.sqlite3'}?mode=ro", uri=True)
    att = con.execute("select run_id, status, failure_code, gpu_seconds from evaluation_attempts").fetchall()
    camp = [r for r in att if r[0].startswith("camp-")]
    final = [r for r in att if r[0].startswith("final-")]
    by_m = defaultdict(Counter)
    for run, status, code, _ in camp:
        mb = method_block(run)
        by_m[mb[0]][status if status == "valid" else (code or status)] += 1
    gpu = dict(
        con.execute(
            """select substr(run_id,1,5), sum(gpu_seconds) from gpu_segments group by substr(run_id,1,5)"""
        ).fetchall()
    )
    N["volume"] = {
        "campaign_runs": 32,
        "attempts": len(camp),
        "valid": sum(1 for r in camp if r[1] == "valid"),
        "invalid": sum(1 for r in camp if r[1] == "invalid"),
        "interrupted": sum(1 for r in camp if r[1] == "interrupted"),
        "failure_codes": dict(Counter(r[2] for r in camp if r[1] != "valid")),
        "by_method": {m: dict(by_m[m]) for m in M},
        "gpu_h_campaign": round(gpu.get("camp-", 0) / 3600, 1),
        "gpu_h_final": round(gpu.get("final", 0) / 3600, 1),
        "unique_architectures": con.execute(
            "select count(distinct phenotype_hash) from candidates where run_id like 'camp-%'"
        ).fetchone()[0],
        "final_attempts": len(final),
        "final_valid": sum(1 for r in final if r[1] == "valid"),
        "tests_opened": con.execute("select count(*) from test_evaluations").fetchone()[0],
    }

    # -- hypervolume (official exports) --------------------------------------------------------------------------------
    hvf = rows(AF / "hv_final.csv")
    N["hv"] = {}
    for m in M:
        r = [x for x in hvf if x["method"] == m]
        hv = [float(x["hv_final"]) for x in r]
        N["hv"][m] = {
            "mean": st.mean(hv),
            "sd": st.stdev(hv),
            "min": min(hv),
            "max": max(hv),
            "evals_mean": st.mean(int(x["unique_valid"]) for x in r),
            "front_mean": st.mean(int(x["front_size"]) for x in r),
            "hv_sens_mean": st.mean(float(x["hv_sens_final"]) for x in r),
        }
    N["sentinel"] = {
        "measurements": sum(int(x["sentinel_measurements"]) for x in hvf),
        "drifts": sum(int(x["sentinel_drifts"]) for x in hvf),
        "failures": sum(int(x["sentinel_failures"]) for x in hvf),
    }
    pares = rows(AF / "hv_pareado.csv")
    write_dat(
        out / "hv_diferencas.dat",
        ["bloco", "mono", "aleatorio", "sequencial"],
        [
            [
                int(p["seed_block"]),
                float(p["joint_menos_mono"]),
                float(p["joint_menos_random"]),
                float(p["joint_menos_sequential"]),
            ]
            for p in pares
        ],
    )
    N["paired"] = {
        k: {"mean": st.mean(float(p[k]) for p in pares), "wins": sum(float(p[k]) > 0 for p in pares)}
        for k in ("joint_menos_mono", "joint_menos_random", "joint_menos_sequential")
    }
    N["h2_h3"] = rows(AF / "inferencia-final" / "h2_h3.csv")
    N["h2_h3_sens"] = rows(AF / "inferencia-final" / "h2_h3_sensibilidade.csv")

    grid = rows(AF / "hv_grade_tempo.csv")
    for m in M:
        by_t = defaultdict(list)
        for g in grid:
            if g["method"] == m:
                by_t[float(g["gpu_s"])].append(float(g["hv"]))
        data = [[t / 3600, st.mean(v), min(v), max(v)] for t, v in sorted(by_t.items()) if len(v) == 8]
        write_dat(out / f"hv_curva_{m}.dat", ["t", "hv", "lo", "hi"], data)
    N["hv_quartis"] = {}
    for m in M:
        by_t = defaultdict(list)
        for g in grid:
            if g["method"] == m:
                by_t[float(g["gpu_s"])].append(float(g["hv"]))
        ts = sorted(t for t, v in by_t.items() if len(v) == 8)
        at = lambda f: st.mean(by_t[min(ts, key=lambda t: abs(t - 19800 * f))])
        N["hv_quartis"][m] = [at(0.25), at(0.5), at(0.75), at(1.0)]

    # -- selection pressure (exploratory) ------------------------------------------------------------------------------
    gen = con.execute("""select a.run_id, c.generation, 1 - m.e_fraction from measurements m
        join evaluation_attempts a on a.attempt_id = m.attempt_id
        join candidates c on c.candidate_id = a.candidate_id and c.run_id = a.run_id
        where a.status = 'valid' and a.run_id like 'camp-%'""").fetchall()
    sel = defaultdict(lambda: defaultdict(list))
    for run, g, acc in gen:
        sel[method_block(run)[0]][g].append(acc)
    gens = range(0, 5)
    write_dat(
        out / "selecao.dat",
        ["geracao"] + list(M),
        [[g] + [st.mean(sel[m][g]) * 100 if len(sel[m][g]) >= 40 else float("nan") for m in M] for g in gens],
    )
    N["selection"] = {
        m: {g: [round(st.mean(sel[m][g]) * 100, 1), len(sel[m][g])] for g in gens if sel[m][g]} for m in M
    }

    # -- final stage: official test (confirmatory where the protocol says so) ------------------------------------------
    fin = rows(Q / "resultados-finais" / "resultados_finais.csv")
    for x in fin:
        for k in ("test_accuracy", "l_ms", "v_mb", "s_mb", "val_error"):
            x[k] = float(x[k])
    units = {}
    for x in fin:
        units[x["retrain_id"]] = x
    u = list(units.values())
    acc = [x["test_accuracy"] for x in u]
    N["final"] = {
        "retrainings": len(u),
        "selections": len(fin),
        "test_max": max(acc),
        "test_median": st.median(acc),
        "test_min": min(acc),
        "val_test_gap_mean": st.mean((1 - x["val_error"]) - x["test_accuracy"] for x in u),
        "best_by_method": {
            m: st.mean(
                max([x["test_accuracy"] for x in fin if x["method"] == m and int(x["seed_block"]) == b])
                for b in range(8)
            )
            for m in M
        },
        "top": [
            {k: x[k] for k in ("seed_block", "method", "role", "test_accuracy", "l_ms", "v_mb", "s_mb")}
            for x in sorted(u, key=lambda x: -x["test_accuracy"])[:10]
        ],
    }
    for m in M:
        pts = sorted({(x["l_ms"], x["test_accuracy"] * 100, x["s_mb"]) for x in fin if x["method"] == m})
        write_dat(out / f"finalistas_{m}.dat", ["L", "acc", "S"], [list(p) for p in pts])
    refs = []
    for n in ("estilo-resnet", "estilo-mobilenetv2"):
        j = json.loads((R / "referencias-2026-10" / f"{n}.json").read_text(encoding="utf-8"))
        refs.append(
            {
                "name": n,
                "acc_val": j["acc_val"],
                "l_ms": j["l_ms"],
                "v_mb": j["v_mb"],
                "s_mb": j["s_mb"],
                "parameters": (j.get("model") or {}).get("parameters"),
            }
        )
    write_dat(out / "referencias.dat", ["L", "acc", "S"], [[r["l_ms"], r["acc_val"] * 100, r["s_mb"]] for r in refs])
    N["references"] = refs
    # finalists that dominate a reference in L, V and S and lose at most k points of accuracy
    N["vs_refs"] = {}
    for r in refs:
        cheaper = [x for x in u if x["l_ms"] < r["l_ms"] and x["v_mb"] < r["v_mb"] and x["s_mb"] < r["s_mb"]]
        best = max(cheaper, key=lambda x: x["test_accuracy"])
        N["vs_refs"][r["name"]] = {
            "cheaper_in_all_three": len(cheaper),
            "best_cheaper_acc": best["test_accuracy"],
            "gap_pp": (r["acc_val"] - best["test_accuracy"]) * 100,
            "speedup": r["l_ms"] / best["l_ms"],
            "smaller": r["s_mb"] / best["s_mb"],
            "less_memory": r["v_mb"] / best["v_mb"],
            "best": {k: best[k] for k in ("method", "seed_block", "l_ms", "v_mb", "s_mb")},
        }
    N["h1"] = json.loads((AF / "inferencia-final" / "h1_resumo.json").read_text(encoding="utf-8"))
    N["h1_tabela"] = rows(AF / "inferencia-final" / "h1_tabela.csv")

    # -- H4 (official export) ---------------------------------------------------------------------------------------------
    N["h4_modelo"] = rows(AF / "analise-h4" / "h4_modelo.csv")
    N["h4_correlacao"] = rows(AF / "analise-h4" / "h4_correlacao.csv")
    pts = rows(AF / "analise-h4" / "h4_pontos.csv")
    sample = pts[::4]
    write_dat(out / "h4_pontos.dat", ["macs", "L"], [[float(p["macs"]) / 1e6, float(p["l_ms"])] for p in sample])

    # -- search-level trade-offs (exploratory) -----------------------------------------------------------------------------
    sm = con.execute("""select a.run_id, 1 - m.e_fraction, m.l_ms, m.v_mb, m.s_mb from measurements m
        join evaluation_attempts a on a.attempt_id = m.attempt_id where a.status = 'valid' and a.run_id like 'camp-%'""").fetchall()
    lim = {"S<=0.5": (4, 0.5), "S<=1": (4, 1.0), "L<=0.03": (2, 0.03), "L<=0.05": (2, 0.05)}
    N["limits"] = {}
    for name, (k, t) in lim.items():
        N["limits"][name] = {
            m: st.mean(max([r[1] for r in sm if method_block(r[0]) == (m, b) and r[k] <= t] or [0]) for b in range(8))
            for m in M
        }
    text = (AF / "final_exploratorio.txt").read_text(encoding="utf-8", errors="replace")
    N["combined_front"] = {m: int(re.search(rf"{m}\s+(\d+) redes", text).group(1)) for m in M}
    N["combined_front_total"] = int(re.search(r"FRENTE COMBINADA: (\d+) redes", text).group(1))

    (out / "numeros.json").write_text(json.dumps(N, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(
        json.dumps(
            {k: N[k] for k in ("volume", "sentinel", "paired", "final", "vs_refs")},
            indent=1,
            ensure_ascii=False,
            default=str,
        )[:6000]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
