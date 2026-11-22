"""Environment manifest for the ``environments`` table: versions, GPU and driver.

Probing never launches GPU kernels. ``torch.cuda.get_device_properties`` does
create a CUDA context, so callers that must keep the GPU untouched pass
``touch_cuda=False`` and rely on ``nvidia-smi`` only.
"""

from __future__ import annotations

import importlib.metadata
import ntpath
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timezone
from typing import Any

PACKAGES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "numpy",
    "onnx",
    "onnxscript",
    "onnx-ir",
    "onnxruntime",
    "onnxruntime-gpu",
    "tensorrt",
    "tensorrt-cu13",
    "tensorrt-cu13-libs",
    "tensorrt-cu13-bindings",
    "polygraphy",
    "nvidia-modelopt",
    "torch-pruning",
)

_SMI_GPU_FIELDS = (
    "name",
    "driver_version",
    "memory.total",
    "memory.used",
    "pstate",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "power.draw",
)


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    """Installed distribution versions; ``None`` means not installed."""
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def nvidia_smi_snapshot(timeout_s: float = 10.0) -> dict[str, Any]:
    """GPU state and processes holding a GPU context, as reported by ``nvidia-smi``.

    Process names are reduced to their base name so the manifest carries no
    user paths.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False, "error": "nvidia-smi not found on PATH"}
    # Own hidden console: a process whose console host died cannot start console children.
    own_console: dict[str, Any] = {"stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        own_console["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        gpu = (
            subprocess.run(
                [exe, f"--query-gpu={','.join(_SMI_GPU_FIELDS)}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=True,
                **own_console,
            )
            .stdout.strip()
            .splitlines()
        )
        apps = (
            subprocess.run(
                [exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=True,
                **own_console,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    gpus = [dict(zip(_SMI_GPU_FIELDS, (v.strip() for v in line.split(",")))) for line in gpu if line]
    compute = []
    for line in apps:
        if line.count(",") < 2:
            continue
        pid, rest = line.split(",", 1)  # process names may contain commas
        name, memory = rest.rsplit(",", 1)
        compute.append(
            {"pid": pid.strip(), "process_name": ntpath.basename(name.strip()), "used_memory_mib": memory.strip()}
        )
    return {"available": True, "gpus": gpus, "compute_apps": compute}


def torch_cuda_info(touch_cuda: bool) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"importable": False, "error": str(exc)}
    cudnn = None
    if touch_cuda:  # asking cuDNN for its version initializes CUDA, so it is part of "touching"
        try:
            cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        except (RuntimeError, ValueError):  # raised by torch when no CUDA device is visible
            cudnn = None
    info: dict[str, Any] = {
        "importable": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": cudnn,
    }
    if touch_cuda and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            cuda_available=True,
            device_name=props.name,
            capability=f"sm_{props.major}{props.minor}",
            total_memory_bytes=props.total_memory,
            arch_list=torch.cuda.get_arch_list(),
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    elif touch_cuda:
        info["cuda_available"] = False
    return info


def probe_environment(touch_cuda: bool = True) -> dict[str, Any]:
    """Collect the manifest fields TCC-04 requires for ``environments``."""
    return {
        "captured_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch": torch_cuda_info(touch_cuda),
        "nvidia_smi": nvidia_smi_snapshot(),
    }
