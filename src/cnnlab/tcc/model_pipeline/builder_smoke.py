"""Short GPU smoke: networks from the genome builder through ONNX (dynamo) and TensorRT.

Checks only that untrained builder networks export, compile, run batch 1 and
agree with PyTorch on seeded random inputs. It also tests one hypothesis from
the wave-2A spike: BF16 conv layers exported by dynamo compile once BN is
folded in PyTorch (``fold_batchnorm``) and fail when the exporter folds it.
Nothing here is E/L/V/S: weights are random, no CIFAR-100 split is read, and
the timing smoke is not the protocol's L.

Untrained networks whose BN still has its initial running statistics give
almost input-independent logits (observed in builder smoke run 1), which makes
a parity check vacuous. BN statistics are therefore warmed up in train mode,
inputs get a per-image, per-channel offset so global pooling does not average
them to the same feature, and each forward records how much the logits vary
across inputs. With random weights the head bias still dominates: top-1
agreement is uninformative when ``distinct_argmax`` is 1.

Run: ``python -m cnnlab.tcc.model_pipeline.builder_smoke --out <dir outside repo>``.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from cnnlab.tcc.model_pipeline.builder import build_model, count_macs, fold_batchnorm
from cnnlab.tcc.model_pipeline.environment import probe_environment
from cnnlab.tcc.model_pipeline.spike_trt import (
    SpikeReport,
    StageRecord,
    _Region,
    _utc,
    _write_atomic,
    campaign_gpu_processes,
    compare_logits,
    export_onnx,
    inspect_onnx,
    probe_imports,
    trt_stages,
)
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.genome import BlockGenes, ExpressedGenome, Genome
from cnnlab.tcc.science.sampling import GeneRng, sample_genome
from cnnlab.tcc.science.space import OPS, SPACE_V1

# (layout name, precision of the whole network, fold BN in PyTorch before export)
LAYOUTS: tuple[tuple[str, str, bool], ...] = (
    ("fp32_folded", "fp32", True),
    ("bf16_folded", "bf16", True),
    ("bf16_exporter_folds", "bf16", False),
)
_GLOBALS: dict[str, Any] = {
    "stem_channels": 32,
    "activation": "relu",
    "head": "gap_fc256_linear",
    "stem_precision": "fp32",
    "head_precision": "fp32",
    "int8_mode": "ptq",
    "weight_granularity": "per_tensor",
    "calibration": "max",
    "log10_lr": -3.0,
    "log10_wd": -4.0,
    "head_dropout": 0.1,
}
_BLOCK: dict[str, Any] = {
    "channels": 32,
    "op": "standard",
    "kernel": 3,
    "n_convs": 1,
    "residual": False,
    "downsample": "none",
    "precision": "fp32",
    "prune_pct": 0,
}
IMPLEMENTED_STAGES: tuple[str, ...] = (
    "gpu_guard",
    "dependency_probe",
    "torch_build",
    "torch_forward",
    "onnx_export[dynamo]",
    "onnx_inspect[dynamo]",
    "trt_build[dynamo]",
    "trt_inspect[dynamo]",
    "trt_infer[dynamo]",
    "trt_timing_smoke[dynamo]",
)


def op_genome(op: str) -> ExpressedGenome:
    """Three blocks of ``op``: widen with projection, reduce by stride 2, identity residual."""
    blocks = [
        BlockGenes(**{**_BLOCK, "op": op, "channels": 64, "residual": True, "n_convs": 2}),
        BlockGenes(**{**_BLOCK, "op": op, "channels": 64, "downsample": "conv_s2", "kernel": 5}),
        BlockGenes(**{**_BLOCK, "op": op, "channels": 64, "residual": True, "downsample": "maxpool"}),
    ]
    genome = Genome(n_blocks=3, **_GLOBALS, blocks=(*blocks, *[BlockGenes(**_BLOCK)] * 3))
    return express(genome, SPACE_V1).genome


def smoke_genomes(sampled: int = 2) -> list[tuple[str, ExpressedGenome]]:
    genomes = [(f"op:{op}", op_genome(op)) for op in OPS]
    genomes += [
        (f"sample:{seed}", express(sample_genome(SPACE_V1, GeneRng(seed)), SPACE_V1).genome) for seed in range(sampled)
    ]
    return genomes


def structured_inputs(seed: int, n: int) -> torch.Tensor:
    """N(0,1) pixels plus a per-image, per-channel N(0, 2^2) offset (synthetic; no dataset)."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((n, 3, 32, 32), generator=generator)
    return noise + 2 * torch.randn((n, 3, 1, 1), generator=generator)


def warm_up_batchnorm(model: torch.nn.Module, seed: int, batches: int = 4, batch_size: int = 16) -> None:
    """Update BN running statistics on synthetic batches (no gradient, no optimizer)."""
    was_training = model.training
    model.train()
    with torch.no_grad():
        for i in range(batches):
            model(structured_inputs(seed + 100 + i, batch_size))
    model.train(was_training)


def logit_spread(logits: torch.Tensor) -> dict[str, Any]:
    """How input-dependent the logits are; parity on constant logits proves nothing."""
    return {
        "mean_std_across_inputs": float(logits.float().std(0).mean()),
        "distinct_argmax": len(set(logits.argmax(1).tolist())),
    }


def run_builder_smoke(out_dir: Path, seed: int = 20260928, allow_busy_gpu: bool = False) -> SpikeReport:
    started = _utc()
    out_dir.mkdir(parents=True, exist_ok=True)
    environment = probe_environment(touch_cuda=True)
    stages: list[StageRecord] = []

    def finish() -> SpikeReport:
        report = SpikeReport(started, _utc(), "cuda", seed, IMPLEMENTED_STAGES, environment, tuple(stages))
        _write_atomic(out_dir / "builder_smoke_report.json", report.to_json().encode("utf-8"))
        return report

    busy = campaign_gpu_processes(environment["nvidia_smi"].get("compute_apps", []), os.getpid())
    if busy and not allow_busy_gpu:
        stages.append(StageRecord("gpu_guard", "-", "failed", "nvidia-smi --query-compute-apps", f"GPU busy: {busy}"))
        return finish()
    stages.append(StageRecord("gpu_guard", "-", "ok", "nvidia-smi --query-compute-apps"))
    deps = probe_imports()
    stages.append(deps)
    if deps.status != "ok" or not torch.cuda.is_available():
        return finish()

    device = torch.device("cuda")
    inputs = structured_inputs(seed + 1, 8).to(device)
    for label, genome in smoke_genomes():
        model, info = build_model(genome, seed)
        warm_up_batchnorm(model, seed)
        stages.append(
            StageRecord(
                "torch_build",
                label,
                "ok",
                f"build_model({label}, seed={seed})",
                "",
                {"expressed": genome.to_dict(), "info": asdict(info), "macs": count_macs(model)},
            )
        )
        reference: torch.Tensor | None = None
        for layout, precision, prefold in LAYOUTS:
            variant = f"{label}__{layout}".replace(":", "_")  # also a file-name stem
            if precision == "bf16" and prefold is False and not label.startswith("op:"):
                continue  # the negative control runs on the per-op genomes only
            net = fold_batchnorm(model) if prefold else copy.deepcopy(model).eval()
            net = _Region(net, precision).to(device).eval()
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = torch.cat([net(x.unsqueeze(0)) for x in inputs])
            torch.cuda.synchronize()
            if reference is None:
                reference = logits
            stages.append(
                StageRecord(
                    "torch_forward",
                    variant,
                    "ok",
                    "net(x) batch 1",
                    "",
                    {
                        "all_finite": bool(torch.isfinite(logits).all()),
                        "vs_fp32_folded": compare_logits(logits, reference),
                        "logit_spread": logit_spread(logits),
                    },
                    time.perf_counter() - t0,
                )
            )
            path = out_dir / f"{variant}.dynamo.onnx"
            record = export_onnx(net, inputs[:1], path, "dynamo", variant)
            stages.append(record)
            if record.status != "ok":
                continue
            stages.append(inspect_onnx(path, variant, "dynamo"))
            stages += trt_stages(path, out_dir, variant, inputs, logits, reference, "dynamo")
            torch.cuda.empty_cache()
    return finish()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="artifact directory, outside the repository")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # the dynamo exporter prints non-ASCII marks
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    report = run_builder_smoke(args.out, args.seed, args.allow_busy_gpu)
    for record in report.stages:
        print(f"{record.variant:>36} {record.stage:<26} {record.status:<8} {record.detail[:90]}")
    print(f"report: {args.out / 'builder_smoke_report.json'}")
    return 2 if report.stages and report.stages[0].status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
