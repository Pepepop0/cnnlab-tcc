"""Pilot report: applies the decision rules fixed before the pilot (TCC-01 section 5; proposal, table 5).

    python tools/piloto/analisa.py --db PILOT_DB --run RUN_ID [--fidelidade FIDELIDADE.json]
                                   [--ruido RUIDO.json] --out DIR

Reads the pilot database in read-only mode and writes ``DIR/piloto.json`` (numbers and
decisions, for the frozen plan) and ``DIR/PILOTO.md`` (the same, for people). Decisions:

* training epochs of the search: Spearman rho between accuracies after 5 and 30 epochs
  >= 0.7 keeps 5 epochs, otherwise 8 (from ``fidelidade.py``);
* hypervolume reference point: r_E = 1, r_j = 1.25 x max_j over viable pilot genomes,
  and the 1.5 x sensitivity point; minima m_j for the equilibrium normalization;
* V x S correlation: reported with no automatic exclusion;
* latency noise: CV of the sentinel medians <= 2% keeps 1,000 repetitions, else 3,000
  (from the measurement-control output of the science front, when given);
* cost: GPU seconds per attempt and the attempts that fit in 5.5 h of active GPU;
* training time: distribution and a suggested per-attempt ceiling for the plan.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estatistica import pearson, percentile, spearman  # noqa: E402

from cnnlab.tcc.science.hypervolume import reference_from_pilot, sensitivity_reference  # noqa: E402
from cnnlab.tcc.storage.readmodel import open_read_model  # noqa: E402

BUDGET_S = 5.5 * 3600
RHO_MIN = 0.7
CV_MAX = 0.02
VS_FLAG = 0.95
TIME_LIMIT_FACTOR = 3.0  # suggested ceiling: 3 x the 99th percentile of training time, at least 10 min


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def _quality(db: Path, run_id: str) -> dict[str, dict[str, Any]]:
    """quality_json per attempt (peak training memory is not in the read model)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT attempt_id, quality_json FROM evaluation_attempts WHERE run_id = ?", (run_id,))
        return {a: json.loads(q) if q else {} for a, q in rows}
    finally:
        con.close()


def analyse(db: Path, run_id: str, fidelity: dict[str, Any] | None, noise: dict[str, Any] | None) -> dict[str, Any]:
    with open_read_model(db) as rm:
        attempts = rm.evaluations(run_id)
        valid = [d for d in rm.evaluation_details(run_id) if not d["synthetic"]]
    quality = _quality(db, run_id)
    statuses = Counter(a["status"] for a in attempts)
    failures = Counter(a["failure_code"] for a in attempts if a["status"] == "invalid")
    points = [(d["e"], d["l_ms"], d["v_mb"], d["s_mb"]) for d in valid]
    report: dict[str, Any] = {
        "run_id": run_id,
        "attempts": dict(statuses),
        "invalid_by_code": dict(failures),
        "invalid_rate": statuses.get("invalid", 0) / len(attempts) if attempts else None,
    }

    if points:
        reference = reference_from_pilot(points)
        report["reference_point"] = {
            "E": reference[0],
            "L_ms": reference[1],
            "V_mb": reference[2],
            "S_mb": reference[3],
        }
        sens = sensitivity_reference(reference)
        report["sensitivity_point"] = {"E": sens[0], "L_ms": sens[1], "V_mb": sens[2], "S_mb": sens[3]}
        report["viable_minima"] = {
            name: min(p[i] for p in points) for i, name in enumerate(("E", "L_ms", "V_mb", "S_mb"))
        }
        v, s = [p[2] for p in points], [p[3] for p in points]
        vs_s, vs_p = spearman(v, s), pearson(v, s)
        report["v_s_correlation"] = {
            "spearman": vs_s,
            "pearson": vs_p,
            "above_0_95": bool(vs_s is not None and vs_s > VS_FLAG),
            "decision": "report with its effect on dominance; no automatic exclusion",
        }

    gpu_all = [a["gpu_seconds"] for a in attempts if a["gpu_seconds"] is not None]
    report["gpu_seconds_per_attempt"] = _summary(gpu_all)
    report["gpu_seconds_per_valid"] = _summary([d["gpu_seconds"] for d in valid])
    if gpu_all:
        mean = sum(gpu_all) / len(gpu_all)
        report["attempts_per_run_5_5h"] = math.floor(BUDGET_S / mean) if mean > 0 else None
    stages: dict[str, list[float]] = {}
    for d in valid:
        for stage, seconds in (d["durations"] or {}).items():
            stages.setdefault(stage, []).append(seconds)
    report["stage_seconds"] = {k: _summary(v) for k, v in sorted(stages.items())}
    train = stages.get("train", [])
    if train:
        p99 = percentile(train, 99)
        report["train_time_limit_suggested_s"] = max(600, int(math.ceil(TIME_LIMIT_FACTOR * p99 / 60) * 60))
    peaks = [q["train_memory"]["peak_allocated_mb"] for q in quality.values() if q.get("train_memory")]
    report["train_peak_memory_mb"] = _summary(peaks)

    decisions: dict[str, Any] = {}
    if fidelity is not None:
        pairs = [
            (g["accuracy"]["5"], g["accuracy"]["30"])
            for g in fidelity["genomes"]
            if "5" in g["accuracy"] and "30" in g["accuracy"]
        ]
        rho = spearman([a for a, _ in pairs], [b for _, b in pairs]) if len(pairs) >= 3 else None
        report["fidelity"] = {"pairs": len(pairs), "spearman_5_30": rho}
        decisions["search_epochs"] = None if rho is None else (5 if rho >= RHO_MIN else 8)
    if noise is not None:
        cv = noise.get("cv_of_medians")
        report["latency_noise"] = noise
        decisions["latency_repetitions"] = None if cv is None else (1000 if cv <= CV_MAX else 3000)
    if "reference_point" in report:
        decisions["reference_point"] = report["reference_point"]
    if "train_time_limit_suggested_s" in report:
        decisions["train_time_limit_s"] = report["train_time_limit_suggested_s"]
    report["decisions"] = decisions
    return report


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "não medido"
    if isinstance(value, float):
        return f"{value:.{digits}f}".replace(".", ",")
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    d = report["decisions"]
    lines = [
        f"# Piloto: {report['run_id']}",
        "",
        "## Decisões pelas regras fixadas antes do piloto",
        "",
        "| Verificação | Medido | Regra | Decisão |",
        "|---|---|---|---|",
    ]
    fid = report.get("fidelity") or {}
    lines.append(
        f"| Fidelidade do treino curto | rho = {_fmt(fid.get('spearman_5_30'))} ({fid.get('pairs', 0)} pares) "
        f"| rho >= 0,7 mantém 5 épocas | {_fmt(d.get('search_epochs'))} épocas |"
    )
    noise = report.get("latency_noise") or {}
    lines.append(
        f"| Ruído da latência | CV = {_fmt(noise.get('cv_of_medians'), 4)} | CV <= 2% mantém 1.000 "
        f"| {_fmt(d.get('latency_repetitions'))} repetições |"
    )
    vs = report.get("v_s_correlation") or {}
    lines.append(
        f"| Redundância V x S | Spearman = {_fmt(vs.get('spearman'))} | > 0,95 é relatada, sem exclusão "
        f"| {'relatar' if vs.get('above_0_95') else 'nada a fazer'} |"
    )
    r = report.get("reference_point") or {}
    lines.append(
        f"| Ponto de referência | E {_fmt(r.get('E'))}, L {_fmt(r.get('L_ms'))} ms, V {_fmt(r.get('V_mb'))} MB, "
        f"S {_fmt(r.get('S_mb'))} MB | 1,25 x máximo | gravar no plano |"
    )
    lines.append(f"| Tempo máximo de treino | p99 x 3 | sugestão | {_fmt(d.get('train_time_limit_s'))} s |")
    gpu = report.get("gpu_seconds_per_attempt") or {}
    lines += [
        "",
        "## Custo",
        "",
        f"- Tentativas: {report['attempts']}; taxa de inválidas: {_fmt(report.get('invalid_rate'))}.",
        f"- GPU por tentativa: média {_fmt(gpu.get('mean'), 1)} s, mediana {_fmt(gpu.get('median'), 1)} s, "
        f"p90 {_fmt(gpu.get('p90'), 1)} s, máximo {_fmt(gpu.get('max'), 1)} s.",
        f"- Tentativas que cabem em 5,5 h de GPU ativa: {_fmt(report.get('attempts_per_run_5_5h'))}.",
        "",
        "| Etapa | Mediana (s) | p90 (s) | Máximo (s) |",
        "|---|---|---|---|",
    ]
    for stage, s in report["stage_seconds"].items():
        lines.append(f"| {stage} | {_fmt(s['median'], 1)} | {_fmt(s['p90'], 1)} | {_fmt(s['max'], 1)} |")
    if report.get("invalid_by_code"):
        lines += ["", "## Inválidas por motivo", ""] + [f"- {k}: {v}" for k, v in report["invalid_by_code"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--fidelidade", type=Path)
    parser.add_argument("--ruido", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    fidelity = json.loads(args.fidelidade.read_text(encoding="utf-8")) if args.fidelidade else None
    noise = json.loads(args.ruido.read_text(encoding="utf-8")) if args.ruido else None
    report = analyse(args.db, args.run, fidelity, noise)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "piloto.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "PILOTO.md").write_text(markdown(report), encoding="utf-8")
    print(f"escrito {args.out / 'piloto.json'} e PILOTO.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
