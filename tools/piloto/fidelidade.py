"""Fidelity of the short training (pilot): accuracy after 5 and after 30 epochs of the same genomes.

    python tools/piloto/fidelidade.py --db PILOT_DB --run RUN_ID --out fidelidade.json
                                      [--n 20] [--epochs 5 30] [--cifar-root DIR]

Takes the first ``n`` viable candidates of the pilot run (a random sample, so the first ones
are as good as any), trains each from scratch on the full 45k training split for every
epoch count, with the run's training seed, and records the validation accuracy in PyTorch.
Ranking fidelity only needs the order of the accuracies, not engine measurements.

Uses the GPU under the lock ``%TEMP%/cnnlab-gpu.lock``. The output is rewritten after
every training, so an interrupted execution resumes where it stopped. The official test
file is never read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trava  # noqa: E402

from cnnlab.tcc.model_pipeline.builder import build_model
from cnnlab.tcc.model_pipeline.data_access import TorchvisionCifar100
from cnnlab.tcc.model_pipeline.demo_data import prepare_demo_data
from cnnlab.tcc.model_pipeline.evaluator import cap_process_memory
from cnnlab.tcc.model_pipeline.splits import DemoSplitSpec
from cnnlab.tcc.model_pipeline.trainer import TrainConfig, evaluate_accuracy, train_model
from cnnlab.tcc.science.genome import ExpressedGenome
from cnnlab.tcc.storage.readmodel import open_read_model


def _take_lock(job: str) -> None:
    trava.take("Piloto (fidelidade)", job)


def _write(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def candidates(db: Path, run_id: str, n: int) -> tuple[int, list[tuple[str, dict[str, Any]]]]:
    """Training seed of the run and the first ``n`` viable candidates with their expressed genome."""
    with open_read_model(db) as rm:
        run = next(r for r in rm.runs() if r["run_id"] == run_id)
        viable = {d["candidate_id"] for d in rm.evaluation_details(run_id)}
    import sqlite3

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT candidate_id, genome_json FROM candidates WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
    finally:
        con.close()
    chosen = [(cid, json.loads(g)["expressed_genome"]) for cid, g in rows if cid in viable][:n]
    return run["train_seed"], chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--epochs", type=int, nargs="+", default=[5, 30])
    parser.add_argument("--cifar-root", type=Path, default=os.environ.get("CNNLAB_TCC_CIFAR_ROOT"))
    args = parser.parse_args(argv)
    if args.cifar_root is None:
        parser.error("give --cifar-root or set CNNLAB_TCC_CIFAR_ROOT")

    seed, chosen = candidates(args.db, args.run, args.n)
    if len(chosen) < args.n:
        print(f"aviso: só {len(chosen)} candidatas viáveis na run")
    out: dict[str, Any] = (
        json.loads(args.out.read_text(encoding="utf-8"))
        if args.out.exists()
        else {
            "run_id": args.run,
            "train_seed": seed,
            "epochs": args.epochs,
            "split": "full 45k train split, 5k validation",
            "genomes": [],
        }
    )
    done = {g["candidate_id"]: g for g in out["genomes"]}

    _take_lock(f"fidelidade {len(chosen)} genomas x {args.epochs} épocas")
    try:
        device = torch.device("cuda", 0)
        cap = cap_process_memory(device, 1024)
        data = prepare_demo_data(
            TorchvisionCifar100(args.cifar_root),
            DemoSplitSpec(train_per_class=None, subset_seed=0),
            f"piloto-fidelidade-{args.run}",
            device,
        )
        out["memory_cap"] = cap
        for cid, genome in chosen:
            entry = done.setdefault(cid, {"candidate_id": cid, "accuracy": {}, "train_s": {}, "status": {}})
            if entry not in out["genomes"]:
                out["genomes"].append(entry)
            for epochs in args.epochs:
                key = str(epochs)
                if key in entry["accuracy"] or key in entry["status"]:
                    continue
                expressed = ExpressedGenome.from_dict(genome)
                model, _ = build_model(expressed, seed)
                model = model.to(device)
                started = time.monotonic()
                try:
                    train_model(
                        model,
                        data.x_train,
                        data.y_train,
                        log10_lr=expressed.log10_lr,
                        log10_wd=expressed.log10_wd,
                        config=TrainConfig(epochs=epochs),
                        seed=seed,
                    )
                    entry["accuracy"][key] = evaluate_accuracy(model, data.x_val, data.y_val)
                except torch.cuda.OutOfMemoryError:
                    entry["status"][key] = "out_of_memory"
                entry["train_s"][key] = time.monotonic() - started
                del model
                torch.cuda.empty_cache()
                _write(args.out, out)
                print(
                    f"{cid} {epochs} épocas: {entry['accuracy'].get(key, entry['status'].get(key))} "
                    f"({entry['train_s'][key]:.0f} s)",
                    flush=True,
                )
    finally:
        trava.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
