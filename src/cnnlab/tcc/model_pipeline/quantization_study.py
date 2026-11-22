"""Same network, trained once per variant with deterministic seeds, measured in the engine: FP32, INT8 PTQ and QAT.

Diagnostic for tasks G3 and G4, not a search and not confirmatory: one genome
drawn from the INT8 slice, trained on the frozen training subset (so, with
deterministic training, with identical weights before quantization in every
variant), and evaluated by :class:`EngineEvaluator` as FP32, as INT8 PTQ with
each weight granularity and each calibration method, and as INT8 QAT with each
weight granularity. It shows what quantization costs in E and what it buys in
L, V and S on this engine. Only the validation split is read; the official
test file is never opened.

Run::

    python -m cnnlab.tcc.model_pipeline.quantization_study --cifar-root <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from cnnlab.tcc.contracts import EvaluationRequest
from cnnlab.tcc.model_pipeline.data_access import TorchvisionCifar100
from cnnlab.tcc.model_pipeline.demo_data import prepare_demo_data, write_val_file
from cnnlab.tcc.model_pipeline.environment import probe_environment
from cnnlab.tcc.model_pipeline.evaluator import (
    EngineEvaluator,
    EvaluatorConfig,
    other_gpu_python_jobs,
    pilot_cache_context,
)
from cnnlab.tcc.model_pipeline.one_candidate import git_revision
from cnnlab.tcc.model_pipeline.quantization import CALIBRATIONS, GRANULARITIES
from cnnlab.tcc.model_pipeline.splits import DemoSplitSpec
from cnnlab.tcc.model_pipeline.trainer import TrainConfig
from cnnlab.tcc.science.operators import INT8_PTQ_SLICE, INT8_QAT_SLICE, prepare_in_slice
from cnnlab.tcc.science.sampling import GeneRng

STUDY_VERSION = "tcc-quantization-study/v2"


def variants() -> list[dict[str, str]]:
    """The FP32 control, every INT8 PTQ granularity x calibration combination and INT8 QAT per granularity."""
    out = [{"name": "fp32", "precision": "fp32"}]
    for granularity, calibration in itertools.product(GRANULARITIES, CALIBRATIONS):
        out.append(
            {
                "name": f"int8-{granularity}-{calibration}",
                "precision": "int8",
                "weight_granularity": granularity,
                "calibration": calibration,
            }
        )
    for granularity in GRANULARITIES:
        out.append(
            {
                "name": f"int8-qat-{granularity}",
                "precision": "int8",
                "int8_mode": "qat",
                "weight_granularity": granularity,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cifar-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="artifacts and report directory, outside the repo")
    parser.add_argument("--genome-seed", type=int, default=7)
    parser.add_argument("--train-seed", type=int, default=2026)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--subset-seed", type=int, default=20260929)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    busy = other_gpu_python_jobs()
    if busy:
        print(f"GPU busy: {busy}", file=sys.stderr)
        return 2
    env = probe_environment(touch_cuda=True)
    spec = DemoSplitSpec(train_per_class=args.train_per_class, subset_seed=args.subset_seed)
    data = prepare_demo_data(TorchvisionCifar100(args.cifar_root), spec, "quantization-study", torch.device("cuda"))
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    work.mkdir(exist_ok=True)
    val_file = work / "val.npz"
    write_val_file(data, val_file)
    config = EvaluatorConfig(train=TrainConfig(epochs=args.epochs, deterministic=True))
    spaces = {"ptq": INT8_PTQ_SLICE, "qat": INT8_QAT_SLICE}
    contexts = {
        mode: pilot_cache_context(data, config, args.train_seed, git_revision(), space.space_hash())
        for mode, space in spaces.items()
    }
    evaluator = EngineEvaluator(data, val_file, config, work_root=work / "attempts", gpu_guard=other_gpu_python_jobs)
    bases = {mode: space.sample(GeneRng(args.genome_seed)) for mode, space in spaces.items()}
    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants()):
        precision = variant["precision"]
        genes = {k: v for k, v in variant.items() if k not in ("name", "precision")}
        mode = "qat" if genes.get("int8_mode") == "qat" else "ptq"
        base, space = bases[mode], spaces[mode]
        raw = dataclasses.replace(
            base,
            stem_precision=precision,
            head_precision=precision,
            **genes,
            blocks=tuple(dataclasses.replace(b, precision=precision, prune_pct=0) for b in base.blocks),
        )
        prepared = prepare_in_slice(raw, space, contexts[mode])
        request = EvaluationRequest(
            run_id="quantization-study",
            candidate_id=f"v{index}",
            attempt_id=f"v{index}.a1",
            stage="pilot",
            genome=prepared.record,
            train_seed=args.train_seed,
            calibration_seed=0,
            dataset_id="cifar100-demo",
            environment_id="local",
            protocol_id="pilot-demo",
            plan_id="pilot-demo",
            artifact_root=args.out / "artifacts",
        )
        result = evaluator.evaluate(request)
        measurement = result.quality.get("measurement", {})
        row = {
            "variant": variant["name"],
            "status": result.status,
            "failure": [result.failure_stage, result.failure_code, result.failure_detail],
            "fitness": None if result.fitness is None else asdict(result.fitness),
            "torch_val_accuracy": result.quality.get("torch_val_accuracy_diagnostic"),
            "torch_val_accuracy_quantized": result.quality.get("torch_val_accuracy_quantized_diagnostic"),
            "qat": result.quality.get("qat"),
            "parity_vs_torch_fp32": (measurement.get("parity") or {}).get("parity_vs_torch_fp32"),
            "durations_s": dict(result.durations_s),
            "precision_ok": (result.quality.get("precision") or {}).get("ok"),
            "quantization": result.quality.get("quantization"),
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("variant", "status", "fitness", "failure")}, default=str), flush=True)
    report = {
        "version": STUDY_VERSION,
        "labels": ["diagnostic", "not-confirmatory", "official test not read"],
        "official_test_read": False,
        "code_version": git_revision(),
        "genome": prepared.record.expressed_genome,
        "train": {
            "epochs": args.epochs,
            "per_class": args.train_per_class,
            "seed": args.train_seed,
            "deterministic": True,
        },
        "data_manifest": data.manifest,
        "environment": env,
        "variants": rows,
    }
    target = args.out / "quantization-study.json"
    target.write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    print(f"report: {target}")
    return 0 if all(r["status"] == "valid" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
