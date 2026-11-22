"""Paired diagnostic: the same trained weights compiled with FP32 blocks and with FP16 blocks.

A search run cannot say what FP16 changes, because its FP32 and FP16
candidates are different networks. Here each genome is trained once (FP32)
and its weights are compiled twice, all blocks in FP32 and all blocks in
FP16 (stem and head FP32 in both). Each engine is measured ``--repeats``
times in isolated processes, alternating the two engines so that slow drift
of the machine affects both alike.

Pilot diagnostic with preliminary L/V: no sentinel, a handful of genomes, one
build per engine (S varies between builds of the same graph). Reads only the
train subset and the validation split; never the official test split.

Run::

    python -m cnnlab.tcc.model_pipeline.precision_pairs --cifar-root <dir> --out <dir outside the repo>
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cnnlab.tcc.model_pipeline.bench import BenchProcessError, run_isolated
from cnnlab.tcc.model_pipeline.builder import build_model, count_macs, count_parameters, fold_batchnorm
from cnnlab.tcc.model_pipeline.data_access import TorchvisionCifar100
from cnnlab.tcc.model_pipeline.demo_data import prepare_demo_data, write_val_file
from cnnlab.tcc.model_pipeline.engine import EngineBuildError, build_engine, export_network
from cnnlab.tcc.model_pipeline.environment import probe_environment
from cnnlab.tcc.model_pipeline.evaluator import other_gpu_python_jobs
from cnnlab.tcc.model_pipeline.one_candidate import git_revision
from cnnlab.tcc.model_pipeline.precision import check_engine_precision, typed_network
from cnnlab.tcc.model_pipeline.splits import DemoSplitSpec
from cnnlab.tcc.model_pipeline.trainer import TrainConfig, evaluate_accuracy, train_model
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.operators import FP32_SLICE
from cnnlab.tcc.science.record import phenotype_hash
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.space import SPACE_V1

PAIRS_VERSION = "tcc-precision-pairs/v1"
VARIANTS: tuple[str, ...] = ("fp32", "fp16")


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Median over the repeated measurements of one engine; every repeat is kept."""
    medians = [r["latency"]["median_ms"] for r in reports]
    peaks = [r["memory"]["v_bytes_allocator_peak"] for r in reports]
    return {
        "L_ms_repeats": medians,
        "L_ms": statistics.median(medians),
        "L_ms_range": max(medians) - min(medians),
        "V_mb_repeats": [p / 1e6 for p in peaks],
        "V_mb": statistics.median(peaks) / 1e6,
        "E": reports[0]["e"]["e_fraction"],
        "E_repeats": [r["e"]["e_fraction"] for r in reports],
        "parity_vs_torch_fp32": reports[0]["quality"].get("parity_vs_torch_fp32"),
    }


def measure_genome(
    index: int,
    genome: Any,
    data: Any,
    val_file: Path,
    work: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cuda")
    model, info = build_model(genome, args.train_seed)
    model = model.to(device)
    entry: dict[str, Any] = {
        "index": index,
        "phenotype_hash": phenotype_hash(genome),
        "activation": genome.activation,
        "blocks": [
            {
                "op": b.op,
                "channels": b.channels,
                "kernel": b.kernel,
                "n_convs": b.n_convs,
                "downsample": b.downsample,
                "residual": b.residual,
            }
            for b in genome.blocks
        ],
        "parameters": count_parameters(model),
        "macs": count_macs(model),
    }
    report = train_model(
        model,
        data.x_train,
        data.y_train,
        log10_lr=genome.log10_lr,
        log10_wd=genome.log10_wd,
        config=TrainConfig(epochs=args.epochs),
        seed=args.train_seed,
    )
    entry["train"] = {"seconds": report.duration_s, "loss_last": report.epoch_loss[-1]}
    entry["torch_val_accuracy"] = evaluate_accuracy(model, data.x_val, data.y_val)
    folded = fold_batchnorm(model)
    with torch.no_grad():
        ref = torch.cat([folded(data.x_val[i : i + 1000]).float().cpu() for i in range(0, data.x_val.shape[0], 1000)])
    ref_path = work / f"g{index}-ref.npy"
    np.save(ref_path, ref.numpy())
    plans: dict[str, Path] = {}
    entry["variants"] = {}
    for variant in VARIANTS:
        precisions = [variant] * len(genome.blocks)
        out = work / f"g{index}-{variant}"
        out.mkdir(parents=True, exist_ok=True)
        try:
            exported = export_network(typed_network(model, precisions), out)
            engine = build_engine(exported.onnx_path, out / "engine.plan")
        except (EngineBuildError, RuntimeError) as exc:
            entry["variants"][variant] = {"status": "failed", "detail": f"{type(exc).__name__}: {exc}"[:500]}
            continue
        check = check_engine_precision(exported.onnx_path, engine.inspector, precisions)
        entry["variants"][variant] = {
            "status": "built" if check["ok"] else "precision_not_executed",
            "S_mb": engine.plan_bytes / 1e6,
            "onnx_bytes": exported.total_bytes,
            "build_seconds": engine.duration_s,
            "precision_ok": check["ok"],
            "precision_problems": check["problems"],
            "executed": {name: region["executed"] for name, region in check["regions"].items()},
        }
        if check["ok"]:
            plans[variant] = out / "engine.plan"
    del folded, model
    gc.collect()
    torch.cuda.empty_cache()
    reports: dict[str, list[dict[str, Any]]] = {v: [] for v in plans}
    for repeat in range(args.repeats):
        order = list(plans) if repeat % 2 == 0 else list(reversed(list(plans)))
        for variant in order:
            try:
                bench = run_isolated(
                    plans[variant],
                    work / f"g{index}-{variant}-bench{repeat}.json",
                    val_path=val_file,
                    ref_logits_path=ref_path,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            except BenchProcessError as exc:
                entry["variants"][variant].setdefault("bench_failures", []).append(str(exc)[:300])
                continue
            bench.pop("latency_samples_ms", None)
            reports[variant].append(bench)
    for variant, measured in reports.items():
        if measured:
            entry["variants"][variant].update(summarize(measured), status="measured", repeats=len(measured))
    a, b = entry["variants"].get("fp32", {}), entry["variants"].get("fp16", {})
    if a.get("status") == b.get("status") == "measured":
        entry["fp16_over_fp32"] = {
            "L": b["L_ms"] / a["L_ms"],
            "V": b["V_mb"] / a["V_mb"],
            "S": b["S_mb"] / a["S_mb"],
            "E_difference": b["E"] - a["E"],
            "L_difference_exceeds_repeat_range": abs(b["L_ms"] - a["L_ms"]) > max(a["L_ms_range"], b["L_ms_range"]),
        }
    return entry


def run(args: argparse.Namespace) -> dict[str, Any]:
    busy = other_gpu_python_jobs()
    if busy:
        raise SystemExit(f"GPU busy, refusing to start: {busy}")
    work = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    spec = DemoSplitSpec(train_per_class=args.train_per_class, subset_seed=args.subset_seed)
    data = prepare_demo_data(TorchvisionCifar100(args.cifar_root), spec, "precision-pairs", torch.device("cuda"))
    val_file = work / "val.npz"
    write_val_file(data, val_file)
    rng = GeneRng(args.genome_seed)
    started = time.monotonic()
    genomes = []
    for index in range(args.genomes):
        genome = express(FP32_SLICE.sample(rng), SPACE_V1).genome
        genomes.append(measure_genome(index, dataclasses.replace(genome), data, val_file, work, args))
        done = genomes[-1].get("fp16_over_fp32")
        print(f"[{time.monotonic() - started:6.1f}s] genome {index}: {done}", flush=True)
    ratios = [g["fp16_over_fp32"] for g in genomes if "fp16_over_fp32" in g]
    summary: dict[str, Any] = {"pairs": len(ratios)}
    for key in ("L", "V", "S"):
        values = [r[key] for r in ratios]
        if values:
            summary[f"{key}_ratio"] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "fp16_smaller_in": sum(v < 1 for v in values),
            }
    if ratios:
        diffs = [r["E_difference"] for r in ratios]
        summary["E_difference"] = {"median": statistics.median(diffs), "min": min(diffs), "max": max(diffs)}
        summary["L_beyond_repeat_range_in"] = sum(r["L_difference_exceeds_repeat_range"] for r in ratios)
    return {
        "version": PAIRS_VERSION,
        "labels": ["pilot", "diagnostic", "preliminary-L-V", "not-confirmatory"],
        "official_test_read": False,
        "code_version": git_revision(),
        "data": data.manifest,
        "config": {
            "genomes": args.genomes,
            "genome_seed": args.genome_seed,
            "train_seed": args.train_seed,
            "epochs": args.epochs,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "iters": args.iters,
            "variants": {"fp32": "all blocks fp32", "fp16": "all blocks fp16; stem and head fp32"},
        },
        "environment": probe_environment(touch_cuda=True),
        "summary": summary,
        "genomes": genomes,
        "wall_seconds": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cifar-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="scratch and report directory, outside the repo")
    parser.add_argument("--genomes", type=int, default=5)
    parser.add_argument("--genome-seed", type=int, default=41)
    parser.add_argument("--train-seed", type=int, default=2001)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--subset-seed", type=int, default=20260929)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    report = run(args)
    target = args.out / "precision-pairs.json"
    target.write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"], indent=1))
    print(f"report: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
