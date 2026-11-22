"""Latency noise and V audit for the pilot: the sentinel measured in separate sessions.

    python tools/piloto/ruido.py --work DIR --out ruido.json [--sessions 10] [--pause-s 30]

Each session is a fresh benchmark process on the same sentinel engine (built once). The
coefficient of variation of the session medians of L is the number the protocol's rule
reads (<= 2% keeps 1,000 repetitions, otherwise 3,000); the V audit compares, per session,
the allocator's V with the drop of free device memory. ``analisa.py --ruido`` reads the
output. Uses the GPU under the lock; the output is rewritten after every session.

``--aquecer-s N`` keeps the GPU busy with matrix products for N seconds before each session,
as a candidate's training does before its measurement; GPU clock, memory clock, temperature
and power are read with ``nvidia-smi`` right before each measurement and recorded with it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from cnnlab.tcc.model_pipeline.sentinel import measure_sentinel, sentinel_engine, v_audit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trava  # noqa: E402


def gpu_state() -> dict[str, float | None]:
    """SM and memory clocks (MHz), temperature (C) and power (W) from nvidia-smi; None if unavailable."""
    fields = ("clocks.sm", "clocks.mem", "temperature.gpu", "power.draw")
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        values = [v.strip() for v in out.split(",")]
        return {k: (float(v) if v.replace(".", "", 1).isdigit() else None) for k, v in zip(fields, values)}
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return {k: None for k in fields}


def heat(seconds: float) -> None:
    """Busy the GPU for ``seconds`` (matrix products), then release everything it allocated."""
    if seconds <= 0:
        return
    import torch

    a = torch.randn(4096, 4096, device="cuda")
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for _ in range(20):
            a = (a @ a).tanh_()
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def cv(values: list[float]) -> float | None:
    """Sample coefficient of variation; None with fewer than two values."""
    if len(values) < 2:
        return None
    return statistics.stdev(values) / statistics.fmean(values)


def summarize(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    medians = [s["l_ms"] for s in sessions]
    audits = [v_audit(s) for s in sessions]
    v = [s["v_mb"] for s in sessions]
    return {
        "sessions": sessions,
        "n_sessions": len(sessions),
        "l_session_medians_ms": medians,
        "cv_of_medians": cv(medians),
        "v_mb": v,
        "v_rel_range": (max(v) - min(v)) / statistics.fmean(v) if v else None,
        "v_audit": audits,
        "v_audit_agrees": all(a["agrees"] for a in audits) if audits else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True, help="folder for the sentinel engine")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--pause-s", type=float, default=30.0)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--aquecer-s", type=float, default=0.0, help="busy the GPU this long before each session")
    args = parser.parse_args(argv)
    trava.take("Piloto (ruido)", f"ruido: sentinela em {args.sessions} sessoes")
    try:
        _, manifest = sentinel_engine(args.work)
        sessions: list[dict[str, Any]] = []
        for i in range(args.sessions):
            if i:
                time.sleep(args.pause_s)
            heat(args.aquecer_s)
            state = gpu_state()
            sessions.append(
                {**measure_sentinel(args.work, iters=args.iters), "gpu_before": state, "heated_s": args.aquecer_s}
            )
            result = {"sentinel": manifest, **summarize(sessions)}
            tmp = args.out.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
            os.replace(tmp, args.out)
            print(
                f"sessao {i + 1}: L {sessions[-1]['l_ms']:.4f} ms (p95 {sessions[-1]['l_p95_ms']:.4f}), "
                f"V {sessions[-1]['v_mb']:.2f} MB, GPU antes {state}",
                flush=True,
            )
    finally:
        trava.release()
    print(f"CV das medianas de L: {result['cv_of_medians']}; auditoria de V ok: {result['v_audit_agrees']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
