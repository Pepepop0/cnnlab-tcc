"""V audit by stage: where the device memory of one engine goes (pilot, TCC-01 section 3).

    python tools/piloto/auditoria_v.py --out auditoria_v.json PLAN [PLAN ...]

For each ``.plan``, a fresh process reads the free device memory after each stage: CUDA
context, TensorRT runtime, engine deserialization, execution context, I/O buffers and
warm-up. TensorRT uses its default allocator here (nothing routed through PyTorch), so
the drops are not inflated by the caching allocator's segments. The bench's V counts only
what passes through the allocator; this probe shows what it leaves out (runtime libraries,
kernel code loaded with the engine) and whether that part depends on the engine.
Uses the GPU under the lock.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trava  # noqa: E402

CHILD = r"""
import json, sys
import torch, tensorrt as trt
plan = open(sys.argv[1], "rb").read()
free = lambda: torch.cuda.mem_get_info()[0]
torch.zeros(1, device="cuda"); torch.cuda.synchronize()
out = {"total": torch.cuda.mem_get_info()[1]}
f0 = free()
runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR)); f1 = free()
engine = runtime.deserialize_cuda_engine(plan); f2 = free()
context = engine.create_execution_context(); f3 = free()
names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
bufs = []
for n in names:
    shape = tuple(context.get_tensor_shape(n))
    t = torch.zeros(shape, device="cuda", dtype=torch.float32); bufs.append(t)
    context.set_tensor_address(n, t.data_ptr())
f4 = free()
stream = torch.cuda.Stream()
for _ in range(200):
    context.execute_async_v3(stream.cuda_stream)
stream.synchronize(); f5 = free()
out.update({"runtime": f0 - f1, "deserialize": f1 - f2, "context": f2 - f3, "io": f3 - f4, "warmup": f4 - f5,
            "total_drop": f0 - f5, "engine_drop": f1 - f5,
            "device_memory_size_v2": int(engine.device_memory_size_v2), "plan_bytes": len(plan)})
print(json.dumps(out))
"""


def probe(plan: Path) -> dict[str, Any]:
    proc = subprocess.run([sys.executable, "-c", CHILD, str(plan)], capture_output=True, text=True, timeout=300)
    if proc.returncode:
        return {"plan": str(plan), "error": proc.stderr[-500:]}
    line = [x for x in proc.stdout.splitlines() if x.startswith("{")][-1]
    return {"plan": str(plan), **{k: (v / 1e6 if k != "total" else v) for k, v in json.loads(line).items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("plans", type=Path, nargs="+")
    args = parser.parse_args(argv)
    trava.take("Piloto (auditoria de V)", f"auditoria de V em {len(args.plans)} engines")
    try:
        results = [probe(p) for p in args.plans]
    finally:
        trava.release()
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{len(results)} engines; resultados em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
