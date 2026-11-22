"""Interface de linha de comando do CNN Lab TCC."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cnnlab", description="CNN Lab TCC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("tui", help="abre a interface no terminal")

    train_parser = subparsers.add_parser("train", help="executa treino classico")
    train_parser.add_argument("--config", required=True, type=Path, metavar="X.json")

    ga_parser = subparsers.add_parser("ga", help="executa algoritmo genetico")
    ga_parser.add_argument("--config", required=True, type=Path, metavar="X.json")

    subparsers.add_parser("doctor", help="verifica ambiente e dados")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Executa o subcomando solicitado e devolve seu codigo de saida."""
    args = _build_parser().parse_args(argv)

    if args.command == "doctor":
        return _doctor()
    if args.command == "tui":
        from cnnlab.tui.app import main as tui_main

        tui_main()
        return 0

    try:
        if args.command == "train":
            from cnnlab.config import ClassicJob

            job = ClassicJob.model_validate_json(args.config.read_text(encoding="utf-8"))
        else:
            from cnnlab.config import GAJob

            job = GAJob.model_validate_json(args.config.read_text(encoding="utf-8"))
        return _run_headless(job)
    except KeyboardInterrupt:
        print("FALHA execucao interrompida pelo usuario", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"FALHA {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _doctor() -> int:
    critical_failure = False

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        _report(False, "torch", f"nao importou ({type(exc).__name__}: {exc})")
        _report(False, "GPU CUDA", "indisponivel porque torch nao importou")
        critical_failure = True
    else:
        cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
        torch_version = getattr(torch, "__version__", "desconhecida")
        _report(
            cuda_build is not None,
            "torch",
            f"versao {torch_version}; build CUDA {cuda_build or 'ausente'}",
        )
        if cuda_build is None:
            critical_failure = True

        try:
            cuda_available = bool(torch.cuda.is_available())
            if not cuda_available:
                _report(False, "GPU CUDA", "torch.cuda.is_available() = False")
                critical_failure = True
            else:
                name = torch.cuda.get_device_name(0)
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                _report(
                    True,
                    "GPU CUDA",
                    (f"{name}; VRAM total {_gib(total_bytes)} GiB; livre {_gib(free_bytes)} GiB"),
                )
        except Exception as exc:
            _report(False, "GPU CUDA", f"consulta falhou ({type(exc).__name__}: {exc})")
            critical_failure = True

    _report(True, "Python", platform.python_version())
    for package in ("textual", "pydantic", "deap"):
        _check_import(package)

    dataset_dir = Path("data") / "cifar-100-python"
    _report(dataset_dir.is_dir(), "CIFAR-100", str(dataset_dir))

    runs_dir = Path("runs")
    run_count = sum(entry.is_dir() for entry in runs_dir.iterdir()) if runs_dir.is_dir() else 0
    _report(runs_dir.is_dir(), "runs/", f"{run_count} run(s)")

    return 1 if critical_failure else 0


def _check_import(package: str) -> None:
    try:
        module = importlib.import_module(package)
    except Exception as exc:
        _report(False, package, f"nao importou ({type(exc).__name__}: {exc})")
        return
    version = getattr(module, "__version__", None)
    _report(True, package, f"versao {version}" if version else "importou")


def _report(ok: bool, check: str, detail: str) -> None:
    print(f"{'OK' if ok else 'FALHA'} {check}: {detail}")


def _gib(byte_count: int) -> str:
    return f"{byte_count / 1024**3:.2f}"


def _run_headless(job: Any) -> int:
    import torch

    from cnnlab.events import RunFailed
    from cnnlab.worker.queue import Job
    from cnnlab.worker.runner import Runner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    queued_job = Job(id="headless", spec=job, status="running")
    queue = _HeadlessQueue(queued_job)
    failures: list[str] = []

    def on_event(event: Any) -> None:
        if isinstance(event, RunFailed):
            failures.append(event.error)

    print(f"OK iniciando: modo {job.kind}; device {device}")
    Runner(queue, on_event, device).run_one(queued_job, threading.Event())

    if queued_job.status != "done":
        detail = failures[-1] if failures else f"status final {queued_job.status}"
        print(f"FALHA run {queued_job.run_id or 'sem-id'}: {detail}", file=sys.stderr)
        return 1

    print(f"OK concluida: {queued_job.run_id}")
    return 0


class _HeadlessQueue:
    """Superficie minima de fila exigida por ``Runner.run_one``."""

    def __init__(self, job: Any) -> None:
        self.job = job

    def mark(self, job_id: str, status: str, run_id: str | None = None) -> None:
        if job_id != self.job.id:
            raise KeyError(job_id)
        self.job.status = status
        self.job.run_id = run_id


if __name__ == "__main__":
    raise SystemExit(main())
