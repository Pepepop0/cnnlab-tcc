"""Engine probe of a search slice: does TensorRT run the precision each region asked for?

For genomes sampled from a slice (all stem x head combinations the slice
allows, blocks random) it builds the typed network, exports it, builds a
strongly typed engine, runs :func:`check_engine_precision` on the inspector
and measures the parity of the engine's logits with the FP32 logits of the
same weights in an isolated benchmark process.

Evidence of representation, not of search: weights are untrained (BN
statistics are warmed on synthetic images so that the logits depend on the
input), inputs are synthetic, labels are random, and nothing here is E, L, V
or S of a candidate. Reads no dataset and never the official test split.

``--matrix PRECISION`` replaces the random genomes by a systematic set: for
every operation x activation, a three-block network whose middle block runs
in PRECISION (its neighbours in FP32), plus a network with the stem and one
with the head in PRECISION for every activation. It answers which
operation/activation combinations TensorRT builds and runs in that type.

Run::

    python -m cnnlab.tcc.model_pipeline.precision_probe --slice fp16_regions --genomes 16 --out <dir>
    python -m cnnlab.tcc.model_pipeline.precision_probe --slice bf16_regions --matrix bf16 --out <dir>
    python -m cnnlab.tcc.model_pipeline.precision_probe --slice int8_ptq --matrix int8 --out <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cnnlab.tcc.model_pipeline.bench import BenchProcessError, run_isolated
from cnnlab.tcc.model_pipeline.builder import build_model, fold_batchnorm
from cnnlab.tcc.model_pipeline.builder_smoke import structured_inputs, warm_up_batchnorm
from cnnlab.tcc.model_pipeline.engine import EngineBuildError, build_engine, export_network
from cnnlab.tcc.model_pipeline.environment import probe_environment
from cnnlab.tcc.model_pipeline.evaluator import other_gpu_python_jobs
from cnnlab.tcc.model_pipeline.one_candidate import git_revision
from cnnlab.tcc.model_pipeline.precision import check_engine_precision, region_precisions, typed_network
from cnnlab.tcc.model_pipeline.quantization import (
    CALIBRATIONS,
    GRANULARITIES,
    calibration_indices,
    int8_regions,
    plan_quantization,
)
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.operators import PROBE_SLICES, SearchSlice
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.space import OPS, SPACE_V1

PROBE_VERSION = "tcc-precision-probe/v1"
N_IMAGES = 64


def allowed_values(space: SearchSlice, gene: str) -> tuple[str, ...]:
    """Values a slice allows for a global precision gene (its pin, or its restricted subset)."""
    pinned = dict(space.pinned_global)
    if gene in pinned:
        return (pinned[gene],)
    restricted = space.restricted_domains()
    if gene in restricted:
        return tuple(restricted[gene].values)
    return tuple(dict(space.base.global_domains())[gene].values)


def probe_genomes(space: SearchSlice, count: int, seed: int) -> list[Any]:
    """Expressed genomes cycling through every stem x head combination the slice allows."""
    combos = list(itertools.product(allowed_values(space, "stem_precision"), allowed_values(space, "head_precision")))
    rng = GeneRng(seed)
    out = []
    for index in range(count):
        stem, head = combos[index % len(combos)]
        raw = dataclasses.replace(space.sample(rng), stem_precision=stem, head_precision=head)
        assert space.contains(raw), "forced combination left the slice"
        out.append(express(raw, SPACE_V1).genome)
    return out


def probe_images(seed: int) -> torch.Tensor:
    """The only source of images: the reference logits and the engine's input are made from these."""
    return structured_inputs(seed + 1, N_IMAGES)


def matrix_genomes(space: SearchSlice, precision: str, seed: int) -> list[tuple[dict[str, Any], Any]]:
    """``(label, expressed genome)`` for the operation x activation matrix of one precision."""
    activations = tuple(dict(SPACE_V1.global_domains())["activation"].values)
    rng = GeneRng(seed)
    out: list[tuple[dict[str, Any], Any]] = []

    def make(activation: str, op: str, region: str, **int8_genes: str) -> Any:
        raw = space.sample(rng)
        pick = {"stem": "stem_precision", "head": "head_precision", "head_fc256": "head_precision"}
        blocks = []
        for i, block in enumerate(raw.blocks):
            if i < 3:
                block = dataclasses.replace(
                    block,
                    channels=32,
                    op=op,
                    kernel=3,
                    n_convs=1,
                    residual=False,
                    downsample="none",
                    precision=precision if (region == "block1" and i == 1) else "fp32",
                )
            blocks.append(block)
        raw = dataclasses.replace(
            raw,
            n_blocks=3,
            stem_channels=32,
            activation=activation,
            head="gap_fc256_linear" if region == "head_fc256" else "gap_linear",
            **int8_genes,
            stem_precision=precision if region == "stem" else "fp32",
            head_precision=precision if region in ("head", "head_fc256") else "fp32",
            blocks=tuple(blocks),
        )
        assert space.contains(raw), f"{region}/{op}/{activation} left the slice"
        assert pick.get(region) is None or getattr(raw, pick[region]) == precision
        return express(raw, SPACE_V1).genome

    # INT8 also has the weight granularity and the calibration method; they cycle so that every value shows up.
    cycle = itertools.cycle(itertools.product(GRANULARITIES, CALIBRATIONS))

    def genes() -> dict[str, str]:
        if precision != "int8":
            return {}
        granularity, calibration = next(cycle)
        return {"int8_mode": "ptq", "weight_granularity": granularity, "calibration": calibration}

    for activation in activations:
        for op in OPS:
            g = genes()
            out.append(
                (
                    {"matrix": "block1", "op": op, "activation": activation, "precision": precision, **g},
                    make(activation, op, "block1", **g),
                )
            )
        for region in ("stem", "head", "head_fc256"):
            g = genes()
            out.append(
                (
                    {"matrix": region, "op": "standard", "activation": activation, "precision": precision, **g},
                    make(activation, "standard", region, **g),
                )
            )
    return out


def probe_one(
    index: int,
    genome: Any,
    work: Path,
    val_file: Path,
    images: torch.Tensor,
    args: argparse.Namespace,
    label: dict[str, Any] | None = None,
) -> dict[str, Any]:
    regions = region_precisions(genome)
    entry: dict[str, Any] = {
        "index": index,
        "activation": genome.activation,
        "stem": regions["stem"],
        "head": regions["head"],
        "label": label,
        "blocks": [
            {
                "op": b.op,
                "precision": b.precision,
                "kernel": b.kernel,
                "channels": b.channels,
                "downsample": b.downsample,
                "residual": b.residual,
            }
            for b in genome.blocks
        ],
    }
    # The parent process stays on the CPU: only TensorRT and the isolated benchmark use the GPU.
    model, _ = build_model(genome, args.seed)
    warm_up_batchnorm(model, args.seed)
    folded = fold_batchnorm(model)
    with torch.no_grad():
        reference = folded(images).float().cpu().numpy()
    np.save(work / f"g{index}-ref.npy", reference)
    entry["reference_logits"] = {
        "abs_max": float(np.abs(reference).max()),
        "std_across_images": float(reference.std(0).mean()),
    }
    plan = None
    wanted = int8_regions(regions["stem"], regions["blocks"], regions["head"])
    if wanted:  # the synthetic probe images stand in for the 1,024 calibration examples
        plan = plan_quantization(
            folded,
            wanted,
            images,
            "ptq",
            genome.weight_granularity,
            genome.calibration,
            calibration_indices(np.arange(len(images)) % 100, len(images), args.seed),
        )
        entry["quantization"] = {
            "granularity": plan.granularity,
            "calibration": plan.calibration,
            "layers": len(plan.input_ranges),
        }
    typed = typed_network(model, regions["blocks"], regions["stem"], regions["head"], plan)
    entry["casts_in_graph"] = typed.casts
    if plan is not None:  # what the Q/DQ graph says, simulated in PyTorch: the engine has to agree with it
        with torch.no_grad():
            np.save(work / f"g{index}-qdq-ref.npy", typed(images).float().cpu().numpy())
    out = work / f"g{index}"
    out.mkdir(parents=True, exist_ok=True)
    try:
        exported = export_network(typed, out)
        engine = build_engine(exported.onnx_path, out / "engine.plan")
    except (EngineBuildError, RuntimeError) as exc:
        entry.update(status="build_failed", detail=f"{type(exc).__name__}: {exc}"[:600])
        return entry
    check = check_engine_precision(
        exported.onnx_path, engine.inspector, regions["blocks"], regions["stem"], regions["head"]
    )
    entry.update(
        status="confirmed" if check["ok"] else "precision_not_executed",
        plan_bytes=engine.plan_bytes,
        problems=check["problems"],
        regions={
            name: {
                "requested": r["requested"],
                "executed": r["executed"],
                "layers": r["layers"],
                "fused_casts": r["fused_casts"],
                "by_weights_only": r.get("by_weights_only", 0),
            }
            for name, r in check["regions"].items()
        },
        evidence=sorted({row["evidence"] for row in check["layers"] if row["evidence"]}),
    )
    try:
        bench = run_isolated(
            out / "engine.plan",
            out / "bench.json",
            val_path=val_file,
            ref_logits_path=work / f"g{index}-ref.npy",
            warmup=5,
            iters=20,
        )
        entry["parity_vs_torch_fp32"] = bench["quality"].get("parity_vs_torch_fp32")
        if plan is not None:
            qdq = run_isolated(
                out / "engine.plan",
                out / "bench-qdq.json",
                val_path=val_file,
                ref_logits_path=work / f"g{index}-qdq-ref.npy",
                warmup=1,
                iters=2,
            )
            entry["parity_vs_torch_qdq"] = qdq["quality"].get("parity_vs_torch_fp32")  # the key names the reference
        entry["engine_logits_finite"] = bool(bench["e"]["all_finite"])
    except BenchProcessError as exc:
        entry["bench_failure"] = str(exc)[:300]
    return entry


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    for entry in entries:
        by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1
    failures: dict[str, int] = {}
    for entry in entries:
        for problem in entry.get("problems", []):
            key = problem.split(": ", 1)[-1]
            failures[key[:120]] = failures.get(key[:120], 0) + 1
    diffs = [e["parity_vs_torch_fp32"]["max_abs_diff"] for e in entries if e.get("parity_vs_torch_fp32")]
    disagreements = [e["parity_vs_torch_fp32"]["top1_disagreements"] for e in entries if e.get("parity_vs_torch_fp32")]
    return {
        "genomes": len(entries),
        "by_status": by_status,
        "failure_reasons": failures,
        "parity_max_abs_diff": {"max": max(diffs), "median": sorted(diffs)[len(diffs) // 2]} if diffs else None,
        "top1_disagreements_total": sum(disagreements),
        "regions_confirmed_by_weights_only": sum(
            r["by_weights_only"] for e in entries for r in e.get("regions", {}).values()
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    busy = other_gpu_python_jobs()
    if busy:
        raise SystemExit(f"GPU busy, refusing to start: {busy}")
    space = PROBE_SLICES[args.slice]
    work = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    images = probe_images(args.seed)
    labels = np.random.default_rng(args.seed).integers(0, 100, N_IMAGES)
    val_file = work / "val.npz"
    np.savez(val_file, x=images.numpy().astype("float32"), y=labels)
    started = time.monotonic()
    entries = []
    if args.matrix:
        work_items = matrix_genomes(space, args.matrix, args.seed)
    else:
        work_items = [(None, g) for g in probe_genomes(space, args.genomes, args.seed)]
    for index, (label, genome) in enumerate(work_items):
        entries.append(probe_one(index, genome, work, val_file, images, args, label))
        last = entries[-1]
        what = (
            label
            if label
            else f"stem={last['stem']} head={last['head']} blocks={[b['precision'] for b in last['blocks']]}"
        )
        print(f"[{time.monotonic() - started:6.1f}s] genome {index} {what} -> {last['status']}", flush=True)
    return {
        "version": PROBE_VERSION,
        "slice": args.slice,
        "space_hash": space.space_hash(),
        "labels": ["representation probe", "untrained weights", "synthetic inputs", "not E/L/V/S", "not-confirmatory"],
        "official_test_read": False,
        "code_version": git_revision(),
        "seed": args.seed,
        "environment": probe_environment(touch_cuda=True),
        "summary": summarize(entries),
        "genomes": entries,
        "wall_seconds": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slice", choices=tuple(PROBE_SLICES), default="fp16_regions")
    parser.add_argument("--genomes", type=int, default=16)
    parser.add_argument("--matrix", choices=("fp16", "bf16", "int8"), help="systematic operation x activation matrix")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--out", type=Path, required=True, help="scratch and report directory, outside the repo")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    report = run(args)
    target = args.out / "precision-probe.json"
    target.write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"], indent=1))
    print(f"report: {target}")
    return 0 if report["summary"]["by_status"].get("confirmed", 0) == report["summary"]["genomes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
