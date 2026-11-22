"""GPU rehearsal of the final evaluation (G9) before the real final stage: real training data, FAKE official test.

Runs ``final_build`` and ``final_test`` of the frozen worktree on one campaign genome, in a scratch folder. The test
split given to ``final_test`` is random images with the official shape; the real CIFAR-100 test file is never opened
(the wrapper refuses it). Nothing is written to the campaign database. Run with PYTHONPATH=<frozen worktree>/src.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from cnnlab.tcc.model_pipeline.data_access import OFFICIAL_SIZES, TorchvisionCifar100
from cnnlab.tcc.model_pipeline.final_eval import (
    FinalConfig,
    MemoryLedger,
    TestAccess,
    final_build,
    final_grant,
    final_test,
    prepare_final_data,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "piloto"))
import trava  # noqa: E402


@dataclass
class RealTrainFakeTest:
    """Real official train file; a random stand-in for the test file, so the real one stays sealed."""

    real: TorchvisionCifar100

    def read(self, official_file: str) -> tuple[torch.Tensor, torch.Tensor]:
        if official_file == "train":
            return self.real.read("train")
        if official_file == "test":
            g = torch.Generator().manual_seed(12345)
            n = OFFICIAL_SIZES["test"]
            images = torch.randint(0, 256, (n, 32, 32, 3), dtype=torch.uint8, generator=g)
            return images, torch.arange(n, dtype=torch.int64) % 100
        raise ValueError(f"unknown official file {official_file!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--candidate", required=True, help="candidate_id of a campaign network")
    ap.add_argument("--cifar-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="scratch folder (created; must not exist)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--builds", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    row = con.execute("select genome_json from candidates where candidate_id = ?", (args.candidate,)).fetchone()
    if row is None:
        raise SystemExit(f"candidate {args.candidate} not found")
    genome = json.loads(row[0])["expressed_genome"]

    trava.take("ensaio-g9", f"final_build/final_test (teste falso) de {args.candidate}", wait_s=60)
    try:
        source = RealTrainFakeTest(TorchvisionCifar100(args.cifar_root))
        config = FinalConfig(epochs=args.epochs, builds=args.builds)
        t0 = time.monotonic()
        grant = final_grant(source, "ensaio-g9", "ensaio-g9-lista")
        data = prepare_final_data(grant, source, torch.device("cuda"))
        built = final_build(genome, 20261022, config, data, args.out / "work", args.out / "art", label="ensaio")
        print(
            f"BUILD status={built.status} builds={len(built.builds)} L={built.l_ms} V={built.v_mb} S={built.s_mb} "
            f"E_val={built.e_val} gpu_s={built.gpu_seconds:.0f} wall={time.monotonic() - t0:.0f}s"
        )
        if built.status != "valid":
            print(f"  falha: {built.failure_stage} {built.failure_code} {built.failure_detail}")
            (args.out / "build.json").write_text(json.dumps(built.to_dict(), indent=2, default=str), encoding="utf-8")
            return 1
        for b in built.builds:
            print(
                f"  build {b.index}: L={b.l_ms:.4f} S={b.s_mb:.3f} V={b.v_mb:.3f} E_val={b.e_val:.4f} precisao_ok={b.precision_ok}"
            )
        access = TestAccess(final_grant(source, "ensaio-g9", "ensaio-g9-lista"), source, MemoryLedger())
        tested = final_test(built.to_dict(), args.out / "art", config, access, args.out / "work-test")
        test = tested.test or {}
        print(f"TEST status={tested.status} keys={sorted(test)[:12]}")
        if tested.status != "valid":
            print(f"  falha: {tested.failure_stage} {tested.failure_code} {tested.failure_detail}")
        (args.out / "resultado.json").write_text(
            json.dumps({"build": built.to_dict(), "test": tested.to_dict()}, indent=2, default=str), encoding="utf-8"
        )
        return 0 if tested.status == "valid" else 1
    finally:
        trava.release()


if __name__ == "__main__":
    sys.exit(main())
