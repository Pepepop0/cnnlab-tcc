"""Independent, read-only audit of a pilot run: SQL rows against raw artifacts and NSGA-II against DEAP.

Deliberately imports nothing from ``cnnlab``: only the standard library
and DEAP. It opens the database with ``mode=ro`` and checks that

* every cataloged artifact exists with the recorded size and SHA-256;
* each valid measurement equals what the raw files say: ``E`` from the
  engine's validation report, ``L`` as the median of the raw latency samples,
  ``V`` from the allocator peak and ``S`` from the size of ``engine.plan``;
* the engine's error agrees with the PyTorch diagnostic accuracy (for INT8, that of the simulated Q/DQ
  network the engine runs, which differs from the FP32 network before quantization);
* no attempt is valid without a measurement, and nothing else has one;
* each stored Pareto archive is the first non-dominated front of everything
  committed before it (``deap.tools.sortNondominated``);
* each generation's survivors are a valid NSGA-II selection: whole fronts
  first, then the least crowded of the split front (ties in crowding
  distance are reported, not judged: their order is a convention);
* the run is a pilot run that never received the official test split.

Run::

    python -m cnnlab.tcc.model_pipeline.verify_run --db DB --artifacts DIR --run RUN [--out FILE]

Exit status 0 when nothing is wrong, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any

VERIFIER_VERSION = "tcc-verify-run/v1"
TOLERANCE = 1e-12
ENGINE_VS_TORCH_TOLERANCE = 0.01  # fraction; the engine is FP32/TF32, the diagnostic is PyTorch FP32
VALID_KINDS = frozenset({"engine", "bench", "train_log", "inspector", "onnx"})


def short(candidate_id: str) -> str:
    """Candidate id without the run prefix (``g001.c0004``)."""
    return candidate_id.split(".", 1)[1] if "." in candidate_id else candidate_id


def _deap() -> tuple[Any, Any]:
    from deap import base, creator, tools

    if not hasattr(creator, "TccMin4"):
        creator.create("TccMin4", base.Fitness, weights=(-1.0,) * 4)
        creator.create("TccInd", list, fitness=creator.TccMin4)
    return creator, tools


def _individuals(ids: list[str], points: dict[str, tuple[float, ...]]) -> list[Any]:
    creator, _ = _deap()
    out = []
    for cid in ids:
        ind = creator.TccInd([cid])
        ind.fitness.values = points[cid]
        out.append(ind)
    return out


def verify(db: Path, artifact_root: Path, run_id: str) -> dict[str, Any]:
    """Audit one run; the returned mapping has ``ok`` and the list of ``problems``."""
    conn = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return _verify(conn, Path(artifact_root), run_id)
    finally:
        conn.close()


def _verify(conn: sqlite3.Connection, root: Path, run_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {"verifier": VERIFIER_VERSION, "run_id": run_id, "problems": []}
    bad = out["problems"].append
    out["pragma"] = {
        "integrity_check": conn.execute("PRAGMA integrity_check").fetchone()[0],
        "foreign_key_violations": len(conn.execute("PRAGMA foreign_key_check").fetchall()),
        "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
    }
    if out["pragma"]["integrity_check"] != "ok" or out["pragma"]["foreign_key_violations"]:
        bad(f"database integrity: {out['pragma']}")
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        bad(f"run {run_id!r} does not exist")
        out["ok"] = False
        return out
    run = dict(row)
    out["run"] = {
        k: run[k]
        for k in (
            "method",
            "stage",
            "seed_block",
            "search_seed",
            "train_seed",
            "space_hash",
            "budget_gpu_s",
            "plan_id",
            "dataset_id",
            "environment_id",
            "finalist_list_id",
        )
    }
    plan = conn.execute(
        "SELECT status, content_json, plan_sha256, frozen_at FROM experiment_plans WHERE plan_id = ?", (run["plan_id"],)
    ).fetchone()
    content = json.loads(plan["content_json"])
    dataset = conn.execute("SELECT manifest_json FROM datasets WHERE dataset_id = ?", (run["dataset_id"],)).fetchone()
    data = json.loads(dataset[0]) if dataset else {}
    out["plan"] = {
        "status": plan["status"],
        "frozen_at": plan["frozen_at"],
        "plan_sha256": plan["plan_sha256"],
        "confirmatory": content.get("confirmatory"),
        "purpose": content.get("purpose"),
        "scope": content.get("scope"),
        "runner": content.get("runner"),
        "evaluator_label": content.get("evaluator") if isinstance(content.get("evaluator"), str) else None,
        "synthetic_evaluator": content.get("synthetic_evaluator"),
    }
    out["dataset"] = {
        "dataset_id": run["dataset_id"],
        "spec": data.get("spec"),
        "n_train_subset": data.get("n_train_subset"),
        "n_train_full_split": data.get("n_train_full_split"),
        "n_val": data.get("n_val"),
        "official_test_read": data.get("official_test_read"),
    }
    out["dataset_splits"] = [
        dict(r)
        for r in conn.execute(
            "SELECT split, n_examples, ref, sha256 FROM dataset_splits WHERE dataset_id = ? ORDER BY split",
            (run["dataset_id"],),
        )
    ]
    out["test_evaluations_rows"] = conn.execute(
        "SELECT COUNT(*) FROM test_evaluations WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    if run["stage"] != "pilot" or run["finalist_list_id"] is not None:
        bad(f"run stage is {run['stage']!r}; this audit is for pilot runs without a finalist list")
    if plan["status"] != "frozen" or content.get("confirmatory") is not False:
        bad("the plan must be frozen and declared not confirmatory")
    if data.get("official_test_read") is not False or out["test_evaluations_rows"]:
        bad("the run must not have read or evaluated the official test split")
    if content.get("synthetic_evaluator"):
        bad("the plan declares a synthetic evaluator")
    env_row = conn.execute(
        "SELECT manifest_json FROM environments WHERE environment_id = ?", (run["environment_id"],)
    ).fetchone()
    env = json.loads(env_row[0]) if env_row else {}
    # The direct demo stores {"science", "probe"}; the operation front stores the wiring's
    # environment under "provided_by_wiring" next to its own system version.
    science = env.get("science") or env.get("provided_by_wiring") or {}
    probe = env.get("probe", {})
    out["system"] = {
        "backend": science.get("backend"),
        "science_code_version": science.get("code_version"),
        "scope": science.get("scope"),
        "search": science.get("search"),
        "train": (science.get("evaluator") or {}).get("train"),
        "bench": (science.get("evaluator") or {}).get("bench"),
        "operation_system": (env.get("system") or {}).get("readable"),
        "operation_code_version": (env.get("system") or {}).get("code_version"),
        "python": probe.get("python") or env.get("python"),
        "platform": probe.get("platform") or env.get("platform"),
        "packages": {k: v for k, v in (probe.get("packages") or {}).items() if v},
        "gpu_at_start": (probe.get("nvidia_smi", {}).get("gpus") or [None])[0],
    }
    out["by_generation"] = [
        dict(r)
        for r in conn.execute(
            "SELECT c.generation, a.status, a.failure_stage, a.failure_code, COUNT(*) AS attempts, "
            "SUM(a.gpu_seconds) AS gpu_seconds FROM evaluation_attempts a JOIN candidates c "
            "ON c.candidate_id = a.candidate_id WHERE a.run_id = ? GROUP BY 1, 2, 3, 4 ORDER BY 1, 2",
            (run_id,),
        )
    ]

    rows = conn.execute(
        "SELECT a.attempt_id, a.candidate_id, a.status, a.gpu_seconds, a.quality_json, a.durations_json, "
        "a.failure_stage, a.failure_code, a.source_attempt_id, a.committed_at, c.generation, c.metadata_json, "
        "c.phenotype_hash, m.e_fraction, m.l_ms, m.v_mb, m.s_mb, s.e_fraction AS src_e, s.l_ms AS src_l, "
        "s.v_mb AS src_v, s.s_mb AS src_s FROM evaluation_attempts a JOIN candidates c "
        "ON c.candidate_id = a.candidate_id LEFT JOIN measurements m ON m.attempt_id = a.attempt_id "
        "LEFT JOIN measurements s ON s.attempt_id = a.source_attempt_id "
        "WHERE a.run_id = ? ORDER BY a.commit_seq",
        (run_id,),
    ).fetchall()
    points: dict[str, tuple[float, ...]] = {}
    by_gen: dict[int, list[str]] = {}
    committed: dict[str, str] = {}
    out["attempts"] = [_attempt(conn, root, r, points, by_gen, committed, bad) for r in rows]
    candidates = [
        dict(r)
        for r in conn.execute(
            "SELECT candidate_id, generation FROM candidates WHERE run_id = ? ORDER BY seq", (run_id,)
        )
    ]
    resolved = {r["candidate_id"] for r in rows if r["status"] in ("valid", "invalid")}
    ledger = conn.execute(
        "SELECT COALESCE(SUM(gpu_seconds), 0) FROM gpu_segments WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    out["counts"] = {
        "candidates": len(candidates),
        "attempts": len(rows),
        "valid": sum(r["status"] == "valid" for r in rows),
        "invalid": sum(r["status"] == "invalid" for r in rows),
        "interrupted": sum(r["status"] == "interrupted" for r in rows),
        "running": sum(r["status"] == "running" for r in rows),
        "unresolved_candidates": [short(c["candidate_id"]) for c in candidates if c["candidate_id"] not in resolved],
        "distinct_phenotypes": len({r["phenotype_hash"] for r in rows}),
        "artifacts": conn.execute("SELECT COUNT(*) FROM artifacts WHERE run_id = ?", (run_id,)).fetchone()[0],
        "gpu_ledger_s": ledger,
        "attempts_gpu_s": sum(r["gpu_seconds"] or 0 for r in rows),
    }
    if abs(out["counts"]["gpu_ledger_s"] - out["counts"]["attempts_gpu_s"]) > 1e-6:
        bad(f"GPU ledger {ledger} differs from the sum of the attempts")

    checkpoints = []
    for c in conn.execute(
        "SELECT checkpoint_id, generation, unique_evaluations, active_gpu_seconds, payload_json, "
        "created_at FROM checkpoints WHERE run_id = ? ORDER BY checkpoint_id",
        (run_id,),
    ):
        payload = json.loads(c["payload_json"])
        state = payload["rng_states"].get("nsga2", {})
        checkpoints.append(
            {
                "id": c["checkpoint_id"],
                "generation": c["generation"],
                "phase": state.get("phase"),
                "unique_evaluations": c["unique_evaluations"],
                "active_gpu_seconds": c["active_gpu_seconds"],
                "population": list(payload["population"]),
                "pending": list(payload["pending_candidates"]),
                "archive": list(payload["pareto_archive"]),
                "has_rng_state": "gene_rng" in payload["rng_states"],
                "created_at": c["created_at"],
            }
        )
    fronts = []
    for s in conn.execute("SELECT * FROM front_snapshots WHERE run_id = ? ORDER BY snapshot_id", (run_id,)):
        members = [
            m[0]
            for m in conn.execute(
                "SELECT candidate_id FROM front_members WHERE snapshot_id = ? ORDER BY candidate_id",
                (s["snapshot_id"],),
            )
        ]
        fronts.append(
            {
                "unique_valid": s["unique_valid_evaluations"],
                "active_gpu_seconds": s["active_gpu_seconds"],
                "hypervolume": s["hypervolume"],
                "members": members,
                "created_at": s["created_at"],
            }
        )
    out["selection_notes"] = _check_search(points, by_gen, committed, checkpoints, fronts, bad)
    out["checkpoints"] = [
        {
            **c,
            "population": [short(p) for p in c["population"]],
            "pending": [short(p) for p in c["pending"]],
            "archive": [short(p) for p in c["archive"]],
        }
        for c in checkpoints
    ]
    out["front_snapshots"] = [{**f, "members": [short(m) for m in f["members"]]} for f in fronts]
    out["cross_check"] = {"archives": len(fronts), "survivor_sets": sum(c["phase"] == "survived" for c in checkpoints)}
    out["ok"] = not out["problems"]
    return out


def _attempt(
    conn: sqlite3.Connection,
    root: Path,
    r: sqlite3.Row,
    points: dict[str, tuple[float, ...]],
    by_gen: dict[int, list[str]],
    committed: dict[str, str],
    bad: Any,
) -> dict[str, Any]:
    meta = json.loads(r["metadata_json"])
    quality = json.loads(r["quality_json"] or "{}")
    entry: dict[str, Any] = {
        "attempt": short(r["attempt_id"]),
        "generation": r["generation"],
        "status": r["status"],
        "origin": meta.get("origin"),
        "parents": [short(p) for p in meta.get("parents", [])],
        "crossover": meta.get("crossover"),
        "mutated": meta.get("mutated"),
        "repairs": meta.get("repairs"),
        "E": r["e_fraction"],
        "L_ms": r["l_ms"],
        "V_mb": r["v_mb"],
        "S_mb": r["s_mb"],
        "gpu_seconds": r["gpu_seconds"],
        "durations_s": json.loads(r["durations_json"] or "{}"),
        "failure_stage": r["failure_stage"],
        "failure_code": r["failure_code"],
        "cache_hit_of": r["source_attempt_id"],
        "synthetic": bool(quality.get("synthetic", False)),
        "preliminary": quality.get("preliminary"),
        "benchmark_tries": [t.get("kind") for t in quality.get("benchmark_tries", [])],
    }
    precision = quality.get("precision")
    if precision is not None:
        entry["precision"] = {
            "requested_blocks": precision.get("requested_blocks"),
            "requested_stem": precision.get("requested_stem", "fp32"),
            "requested_head": precision.get("requested_head", "fp32"),
            "checked": precision.get("ok"),
            "casts_in_graph": precision.get("casts_in_graph"),
            "executed": {name: region.get("executed") for name, region in (precision.get("regions") or {}).items()},
        }
        if r["status"] == "valid" and precision.get("ok") is not True:
            bad(f"{r['attempt_id']}: valid although the engine did not confirm the requested precision")
    measured = all(entry[k] is not None for k in ("E", "L_ms", "V_mb", "S_mb"))
    if r["status"] == "valid" and r["source_attempt_id"] is None and not measured:
        bad(f"{r['attempt_id']}: valid without the four measurements")
    if r["status"] != "valid" and any(entry[k] is not None for k in ("E", "L_ms", "V_mb", "S_mb")):
        bad(f"{r['attempt_id']}: status {r['status']} carries a measurement")
    arts = {a["kind"]: a for a in conn.execute("SELECT * FROM artifacts WHERE attempt_id = ?", (r["attempt_id"],))}
    intact = True
    for kind, a in arts.items():
        path = root / a["rel_path"]
        if not path.is_file():
            bad(f"{r['attempt_id']}: artifact {kind} is missing on disk")
            intact = False
            continue
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != a["sha256"] or len(payload) != a["size_bytes"]:
            bad(f"{r['attempt_id']}: artifact {kind} differs from its cataloged hash or size")
            intact = False
    if r["status"] != "valid":
        return entry
    by_gen.setdefault(r["generation"], []).append(r["candidate_id"])
    committed[r["candidate_id"]] = r["committed_at"]
    if r["source_attempt_id"] is not None:  # same-run cache hit: the objectives are those of the source
        if r["src_e"] is None:
            bad(f"{r['attempt_id']}: cache hit of {r['source_attempt_id']}, which has no measurement")
            return entry
        points[r["candidate_id"]] = (r["src_e"], r["src_l"], r["src_v"], r["src_s"])
        return entry
    points[r["candidate_id"]] = (entry["E"], entry["L_ms"], entry["V_mb"], entry["S_mb"])
    if entry["synthetic"]:
        bad(f"{r['attempt_id']}: result is labelled synthetic; not a measurement")
    if not VALID_KINDS <= set(arts):
        bad(f"{r['attempt_id']}: artifact kinds {sorted(arts)} lack {sorted(VALID_KINDS - set(arts))}")
        return entry
    if not intact:
        return entry
    bench = json.loads((root / arts["bench"]["rel_path"]).read_text(encoding="utf-8"))
    train = json.loads((root / arts["train_log"]["rel_path"]).read_text(encoding="utf-8"))
    samples = bench["latency_samples_ms"]
    recomputed = {
        "E": bench["e"]["e_fraction"],
        "L_ms": statistics.median(samples),
        "V_mb": bench["memory"]["v_bytes_allocator_peak"] / 1e6,
        "S_mb": arts["engine"]["size_bytes"] / 1e6,
    }
    for key, value in recomputed.items():
        if abs(value - entry[key]) > TOLERANCE:
            bad(f"{r['attempt_id']}: {key} in SQL is {entry[key]}, the raw artifacts give {value}")
    if bench["e"]["errors"] / bench["e"]["n"] != bench["e"]["e_fraction"] or not bench["e"]["all_finite"]:
        bad(f"{r['attempt_id']}: inconsistent engine validation report")
    torch_acc = quality.get("torch_val_accuracy_diagnostic")
    simulated = quality.get("torch_val_accuracy_quantized_diagnostic")  # what the engine runs, when quantized
    reference_acc = torch_acc if simulated is None else simulated
    if reference_acc is not None and abs((1 - reference_acc) - entry["E"]) > ENGINE_VS_TORCH_TOLERANCE:
        bad(
            f"{r['attempt_id']}: engine error {entry['E']} vs PyTorch {1 - reference_acc}"
            + ("" if simulated is None else " (simulated Q/DQ network)")
        )
    parity = bench.get("quality", {}).get("parity_vs_torch_fp32", {})
    entry.update(
        {
            "latency": {k: bench["latency"][k] for k in ("p5_ms", "p95_ms", "iqr_ms", "mean_ms")},
            "latency_samples": len(samples),
            "warmup": bench.get("warmup"),
            "E_n": bench["e"]["n"],
            "E_errors": bench["e"]["errors"],
            "parity_max_abs_diff": parity.get("max_abs_diff"),
            "parity_top1_agreement": parity.get("top1_agreement"),
            "parity_n": parity.get("n"),
            "parity_top1_disagreements": parity.get("top1_disagreements"),
            "torch_val_accuracy_diagnostic": torch_acc,
            "torch_val_accuracy_quantized_diagnostic": simulated,
            "train": {
                "epochs": train.get("epochs"),
                "steps": train.get("steps"),
                "loss_first": train.get("epoch_loss", [None])[0],
                "loss_last": train.get("epoch_loss", [None])[-1],
            },
            "engine_output_dtypes": quality.get("engine", {}).get("output_dtypes"),
            "memory_audit": {
                k: bench["memory"].get(k)
                for k in ("trt_allocator_peak_live_bytes", "io_buffer_bytes", "free_mem_drop_bytes_audit")
            },
            "bench_version": bench.get("bench_version"),
        }
    )
    return entry


def _first_front(ids: list[str], points: dict[str, tuple[float, ...]]) -> list[str]:
    _, tools = _deap()
    if not ids:
        return []
    layers = tools.sortNondominated(_individuals(ids, points), len(ids), first_front_only=True)
    return sorted(i[0] for i in layers[0])


def _check_search(
    points: dict[str, tuple[float, ...]],
    by_gen: dict[int, list[str]],
    committed: dict[str, str],
    checkpoints: list[dict[str, Any]],
    fronts: list[dict[str, Any]],
    bad: Any,
) -> list[dict[str, Any]]:
    """Archives against the first front computed by DEAP; survivors against the NSGA-II selection rule."""
    if not points:
        return []
    _, tools = _deap()
    measured = sorted(points)
    # Every stored archive is the first front of what had been committed when it was written.
    for label, stored_at, members in [
        (f"front snapshot {i + 1}", f["created_at"], f["members"]) for i, f in enumerate(fronts)
    ] + [(f"checkpoint {c['id']}", c["created_at"], c["archive"]) for c in checkpoints]:
        seen = [c for c in measured if committed[c] <= stored_at]
        if _first_front(seen, points) != sorted(members):
            bad(f"{label}: stored archive {sorted(members)} != first front {_first_front(seen, points)}")
    notes: list[dict[str, Any]] = []
    population: list[str] | None = None
    last_generation = None
    for cp in checkpoints:
        if cp["phase"] != "survived" or cp["generation"] == last_generation:
            continue  # a resumed session may store the survival of a generation twice
        gen = last_generation = cp["generation"]
        stored = set(cp["population"])
        if population is None:
            population = sorted(stored)
            continue
        candidates = set(population) | set(by_gen.get(gen, []))
        unmeasured = {c for c in stored if c not in points}
        if not stored - unmeasured <= candidates:
            bad(f"generation {gen}: survivors are not members of parents plus offspring")
        pool = sorted(candidates & set(points))
        size, taken = len(stored), 0
        for rank, layer in enumerate(tools.sortNondominated(_individuals(pool, points), len(pool)) if pool else []):
            if taken >= size:
                break
            ids = {i[0] for i in layer}
            if taken + len(ids) <= size:
                if not ids <= stored:
                    bad(f"generation {gen}: front {rank} must survive whole, missing {sorted(ids - stored)}")
                taken += len(ids)
                continue
            tools.emo.assignCrowdingDist(layer)
            crowd = {i[0]: i.fitness.crowding_dist for i in layer}
            chosen, rejected = ids & stored, ids - stored
            if len(chosen) != size - taken:
                bad(f"generation {gen}: {len(chosen)} taken from the split front, expected {size - taken}")
            if chosen and rejected and min(crowd[c] for c in chosen) < max(crowd[r] for r in rejected):
                bad(f"generation {gen}: a less crowded candidate was rejected: {crowd}")
            floor = min((crowd[c] for c in chosen), default=float("inf"))
            notes.append(
                {
                    "generation": gen,
                    "split_front_rank": rank,
                    "split_front": sorted(short(i) for i in ids),
                    "chosen": sorted(short(c) for c in chosen),
                    "crowding": {short(k): v for k, v in sorted(crowd.items())},
                    "tied_with_chosen": sorted(short(r) for r in rejected if crowd[r] >= floor),
                }
            )
            taken = size
        if taken < min(size, len(pool)):
            bad(f"generation {gen}: only {taken} measured survivors although {len(pool)} were available")
        population = sorted(stored)
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify(args.db, args.artifacts, args.run)
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    counts = report.get("counts", {})
    print(
        f"ok={report['ok']} problems={len(report['problems'])} candidates={counts.get('candidates')} "
        f"valid={counts.get('valid')} invalid={counts.get('invalid')} interrupted={counts.get('interrupted')}"
    )
    for problem in report["problems"]:
        print("PROBLEM:", problem)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
