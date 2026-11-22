"""Evaluate one FP32 candidate end to end on the GPU and print its measured (E, L, V, S).

Milestone check for the pilot demo, not a search: the genome is drawn from
the FP32 slice with ``--genome-seed``, trained on the frozen train subset and
measured by :class:`EngineEvaluator`. Artifacts go under ``--artifacts``
(outside the repository); a JSON summary is written next to them.

Run: ``python -m cnnlab.tcc.model_pipeline.one_candidate --cifar-root <dir> --artifacts <dir>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from cnnlab.tcc.contracts import EvaluationRequest
from cnnlab.tcc.model_pipeline.data_access import TorchvisionCifar100
from cnnlab.tcc.model_pipeline.demo_data import prepare_demo_data, write_val_file
from cnnlab.tcc.model_pipeline.environment import probe_environment
from cnnlab.tcc.model_pipeline.evaluator import EngineEvaluator, EvaluatorConfig, pilot_cache_context
from cnnlab.tcc.model_pipeline.spike_trt import campaign_gpu_processes
from cnnlab.tcc.model_pipeline.splits import DemoSplitSpec
from cnnlab.tcc.model_pipeline.trainer import TrainConfig
from cnnlab.tcc.science.operators import FP32_SLICE, prepare_in_slice
from cnnlab.tcc.science.sampling import GeneRng


def git_revision() -> str:
    """Commit of this checkout plus a dirty flag; the evaluator's code version."""
    import subprocess

    from cnnlab.tcc.model_pipeline.bench import child_process_kwargs

    root = Path(__file__).resolve().parents[4]
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True, **child_process_kwargs()
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            **child_process_kwargs(),
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return rev + ("+dirty" if dirty else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cifar-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--genome-seed", type=int, default=7)
    parser.add_argument("--train-seed", type=int, default=2026)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--subset-seed", type=int, default=20260929)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true", help="deterministic cuDNN algorithms in training")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    env = probe_environment(touch_cuda=True)
    busy = campaign_gpu_processes(env["nvidia_smi"].get("compute_apps", []), __import__("os").getpid())
    if busy:
        print(f"GPU busy: {busy}", file=sys.stderr)
        return 2
    run_id = "one-candidate"
    spec = DemoSplitSpec(train_per_class=args.train_per_class, subset_seed=args.subset_seed)
    data = prepare_demo_data(TorchvisionCifar100(args.cifar_root), spec, run_id, torch.device("cuda"))
    args.artifacts.mkdir(parents=True, exist_ok=True)
    work = args.artifacts.parent / (args.artifacts.name + "-work")
    work.mkdir(parents=True, exist_ok=True)
    val_file = work / "val.npz"
    val_sha = write_val_file(data, val_file)
    config = EvaluatorConfig(train=TrainConfig(epochs=args.epochs, deterministic=args.deterministic))
    context = pilot_cache_context(data, config, args.train_seed, git_revision(), FP32_SLICE.space_hash())
    genome = FP32_SLICE.sample(GeneRng(args.genome_seed))
    prepared = prepare_in_slice(genome, FP32_SLICE, context)
    request = EvaluationRequest(
        run_id=run_id,
        candidate_id="c0",
        attempt_id="c0.a1",
        stage="pilot",
        genome=prepared.record,
        train_seed=args.train_seed,
        calibration_seed=0,
        dataset_id="cifar100-demo",
        environment_id="local",
        protocol_id="pilot-demo",
        plan_id="pilot-demo",
        artifact_root=args.artifacts,
    )
    evaluator = EngineEvaluator(data, val_file, config, work_root=work)
    result = evaluator.evaluate(request)
    summary = {
        "status": result.status,
        "fitness": None if result.fitness is None else asdict(result.fitness),
        "gpu_seconds": result.gpu_seconds,
        "durations_s": dict(result.durations_s),
        "failure": [result.failure_stage, result.failure_code, result.failure_detail],
        "artifacts": dict(result.artifact_paths),
        "artifact_sha256": dict(result.artifact_sha256),
        "expressed_genome": prepared.record.expressed_genome,
        "phenotype_hash": prepared.record.phenotype_hash,
        "val_file_sha256": val_sha,
        "code_version": context.code_version,
        "cache_context_hash": prepared.record.cache_context_hash,
        "data_manifest": data.manifest,
        "quality": dict(result.quality),
        "environment": env,
    }
    (args.artifacts.parent / (args.artifacts.name + "-summary.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: summary[k] for k in ("status", "fitness", "gpu_seconds", "durations_s", "failure")},
            indent=2,
            default=str,
        )
    )
    return 0 if result.status == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
