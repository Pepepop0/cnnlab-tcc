"""Hand-designed reference networks through the same final pipeline as the finalists (exploratory, no test access).

Two genomes written by hand inside the search space, in the style of a small ResNet and of a MobileNetV2, FP32 and
unpruned, go through ``final_build`` (30 epochs from scratch, 3 builds, medians of L, V, S) like the finalists. Only
the validation split is used: the official test is never granted. The point is an anchor for the finalists'
validation accuracy and cost, measured on the same engine. Run with PYTHONPATH=<frozen worktree>/src.
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
import time
from pathlib import Path

import torch

from cnnlab.tcc.model_pipeline.data_access import SplitIndices, TorchvisionCifar100, grant_search_access
from cnnlab.tcc.model_pipeline.final_eval import FinalConfig, final_build, prepare_final_data
from cnnlab.tcc.model_pipeline.splits import DemoSplitSpec, poc_stratified_split
from cnnlab.tcc.science.expression import express
from cnnlab.tcc.science.genome import Genome
from cnnlab.tcc.science.space import SPACE_V1

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "piloto"))
import trava  # noqa: E402

COMMON = {
    "activation": "relu",
    "head": "gap_linear",
    "stem_precision": "fp32",
    "head_precision": "fp32",
    "int8_mode": "ptq",
    "weight_granularity": "per_channel",
    "calibration": "entropy",
    "log10_lr": -2.75,
    "log10_wd": -4.0,
    "head_dropout": 0.2,
}


def block(channels: int, op: str, downsample: str, n_convs: int = 2, kernel: int = 3, residual: bool = True) -> dict:
    return {
        "channels": channels,
        "op": op,
        "kernel": kernel,
        "n_convs": n_convs,
        "residual": residual,
        "downsample": downsample,
        "precision": "fp32",
        "prune_pct": 0,
    }


REFERENCES = {
    "estilo-resnet": {
        "n_blocks": 6,
        "stem_channels": 64,
        "blocks": [
            block(64, "standard", "none"),
            block(128, "standard", "conv_s2"),
            block(128, "standard", "none"),
            block(256, "standard", "conv_s2"),
            block(256, "standard", "none"),
            block(256, "standard", "none"),
        ],
    },
    "estilo-mobilenetv2": {
        "n_blocks": 6,
        "stem_channels": 32,
        "blocks": [
            block(32, "mbconv_e6", "none"),
            block(64, "mbconv_e6", "conv_s2"),
            block(96, "mbconv_e6", "none"),
            block(128, "mbconv_e6", "conv_s2"),
            block(192, "mbconv_e6", "none"),
            block(256, "mbconv_e6", "none", 1),
        ],
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, required=True, help="campaign database, read only: a raw genome as template")
    ap.add_argument("--cifar-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--builds", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    template = json.loads(con.execute("select genome_json from candidates limit 1").fetchone()[0])["raw_genome"]
    trava.take("referencias", "redes de referencia (validacao, sem teste)", wait_s=60)
    try:
        source = TorchvisionCifar100(args.cifar_root)
        _, labels = source.read("train")
        spec = DemoSplitSpec(train_per_class=None, subset_seed=0)
        train_idx, val_idx = poc_stratified_split(labels, spec.val_fraction, spec.split_seed)
        grant = grant_search_access(
            "referencias",
            "pilot",
            [SplitIndices("train", "train", tuple(train_idx)), SplitIndices("val", "train", tuple(val_idx))],
        )
        data = prepare_final_data(grant, source, torch.device("cuda"))
        config = FinalConfig(epochs=args.epochs, builds=args.builds)
        for name, spec_ in REFERENCES.items():
            if args.only and name not in args.only:
                continue
            if (args.out / f"{name}.json").exists():
                print(f"{name}: ja feito, pulando")
                continue
            raw = copy.deepcopy(template)
            raw.update(COMMON)
            raw["n_blocks"], raw["stem_channels"] = spec_["n_blocks"], spec_["stem_channels"]
            raw["blocks"] = spec_["blocks"] + raw["blocks"][len(spec_["blocks"]) :]
            expressed = express(Genome.from_dict(raw), SPACE_V1).genome.to_dict()
            t0 = time.monotonic()
            built = final_build(
                expressed, 20261022, config, data, args.out / f"work-{name}", args.out / "art", label=name
            )
            row = {
                "name": name,
                "status": built.status,
                "e_val": built.e_val,
                "acc_val": None if built.e_val is None else 1 - built.e_val,
                "l_ms": built.l_ms,
                "v_mb": built.v_mb,
                "s_mb": built.s_mb,
                "model": (built.quality or {}).get("model"),
                "wall_s": round(time.monotonic() - t0),
                "failure": [built.failure_stage, built.failure_code, built.failure_detail],
                "genome": expressed,
            }
            (args.out / f"{name}.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
            print(
                f"{name}: {built.status} acc_val={row['acc_val']} L={built.l_ms} V={built.v_mb} S={built.s_mb} "
                f"({row['wall_s']} s)",
                flush=True,
            )
        return 0
    finally:
        trava.release()


if __name__ == "__main__":
    sys.exit(main())
