"""The pilot, end to end, in one command (TCC-01 section 5; task A11).

``cnnlab-tcc piloto run --dir DIR --cifar-root CIFAR`` runs the four steps of the pilot in order, in a directory and a
database of their own (never the campaign's):

1. ``busca``: a random search of ``population x (generations + 1)`` genomes (64 by default) over the admitted slice,
   by the operation's own ``run`` command (a pilot run, its own plan, no budget decision hidden in it);
2. ``fidelidade``: ``tools/piloto/fidelidade.py``, the same genomes trained 5 and 30 epochs;
3. ``ruido``: ``tools/piloto/ruido.py``, the sentinel network measured in separate sessions;
4. ``analise``: ``tools/piloto/analisa.py``, the decision rules fixed before the pilot, writing ``piloto.json``, which
   ``campaign plan --pilot`` reads.

This module never imports torch and never touches the GPU itself: each step is a subprocess that takes the GPU lock of
its own tool, so one GPU job at a time still holds. A step that finds its work done is skipped, so the command can be
repeated after an interruption; the settings are stored in ``piloto.config.json`` on the first run and a repeat with
other settings is refused (start a new directory instead). The analysis is refused when the attempts were measured by
more than one benchmark version. Nothing here reads the official test split.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from cnnlab.tcc.operation import campaign
from cnnlab.tcc.operation.bench_versions import select_runs
from cnnlab.tcc.science.operators import SLICES
from cnnlab.tcc.storage.canonical import sha256_hex
from cnnlab.tcc.storage.errors import NotFoundError, StorageError
from cnnlab.tcc.storage.readmodel import open_read_model

PILOT_FORMAT = "tcc-pilot-run/1"
RUN_ID = "piloto-aleatorio"
FACTORY = "cnnlab.tcc.demo_wiring:demo_wiring"
# The slice the pilot searches: the admitted space of the campaign. Change it with --slice.
DEFAULT_SLICE = "admitted"
STEPS = ("busca", "fidelidade", "ruido", "analise")
CONFIG_NAME = "piloto.config.json"
STATE_NAME = "estado.json"
SUMMARY_NAME = "resumo.json"
SPEC_NAME = "busca.spec.json"
# Seeds of the pilot: fixed, and apart from the campaign's, whose derive from the master seed of the plan.
SEARCH_SEED, TRAIN_SEED, CALIBRATION_SEED = 20261001, 20261002, 20261003
PENDING_DECISIONS = ("search_epochs", "latency_repetitions", "reference_point", "train_time_limit_s")

Executor = Callable[[Sequence[str], Mapping[str, str]], int]


class PilotError(StorageError):
    """The pilot cannot go on as asked; nothing further was run."""


@dataclass(frozen=True, slots=True)
class PilotSettings:
    cifar_root: str
    slice: str = DEFAULT_SLICE
    population: int = 16
    generations: int = 3
    epochs: int = 5
    budget_gpu_s: float = 86_400.0  # a ceiling; the run ends when its generations do
    fidelity_n: int = 20
    fidelity_epochs: tuple[int, ...] = (5, 30)
    sessions: int = 10
    pause_s: float = 30.0
    heat_s: float = 60.0
    search_seed: int = SEARCH_SEED
    train_seed: int = TRAIN_SEED
    calibration_seed: int = CALIBRATION_SEED
    author: str = "Pedro Antunes"

    def __post_init__(self) -> None:
        if self.slice not in SLICES:
            raise PilotError(f"slice must be one of {sorted(SLICES)}, not {self.slice!r}")
        for name in ("population", "generations", "epochs", "fidelity_n", "sessions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "generations" else 1):
                raise PilotError(f"{name} must be a whole number" + ("" if name == "generations" else " of at least 1"))
        if self.population % 2:
            raise PilotError("population must be even (offspring come in pairs)")
        if not (self.fidelity_epochs and all(isinstance(e, int) and e >= 1 for e in self.fidelity_epochs)):
            raise PilotError("fidelity_epochs must be a list of epoch counts")
        for name in ("budget_gpu_s", "pause_s", "heat_s"):
            if not getattr(self, name) >= 0:
                raise PilotError(f"{name} must not be negative")

    @property
    def planned_genomes(self) -> int:
        return self.population * (self.generations + 1)

    def definition(self) -> dict[str, Any]:
        out = asdict(self)
        out["fidelity_epochs"] = list(self.fidelity_epochs)
        return out

    @classmethod
    def from_definition(cls, data: Mapping[str, Any]) -> PilotSettings:
        known = set(cls.__dataclass_fields__)
        if set(data) - known:
            raise PilotError(f"unknown settings: {sorted(set(data) - known)}")
        return cls(**{**data, "fidelity_epochs": tuple(data.get("fidelity_epochs", (5, 30)))})


def _atomic_write(path: Path, data: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(tmp, path)


def _default_executor(argv: Sequence[str], env: Mapping[str, str]) -> int:
    return subprocess.run(list(argv), env=dict(env)).returncode


@dataclass
class Pilot:
    """The pilot of one directory: its commands, what is done, and the run."""

    root: Path
    settings: PilotSettings
    repo_root: Path
    python: str = field(default_factory=lambda: sys.executable)

    # -- layout --------------------------------------------------------------------------------------------------
    @property
    def db(self) -> Path:
        return self.root / "tcc.sqlite3"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def work(self) -> Path:
        return self.root / "trabalho"

    @property
    def fidelity_file(self) -> Path:
        return self.root / "fidelidade.json"

    @property
    def noise_file(self) -> Path:
        return self.root / "ruido.json"

    @property
    def analysis_dir(self) -> Path:
        return self.root / "analise"

    @property
    def report_file(self) -> Path:
        return self.analysis_dir / "piloto.json"

    # -- the run -------------------------------------------------------------------------------------------------
    def spec(self) -> dict[str, Any]:
        s = self.settings
        return {
            "method": "random",
            "seed_block": 0,
            "search_seed": s.search_seed,
            "train_seed": s.train_seed,
            "calibration_seed": s.calibration_seed,
            "budget_gpu_s": s.budget_gpu_s,
            "code_label": "piloto",
            "author": s.author,
            "run_id": RUN_ID,
            "job_id": f"job-{RUN_ID}",
            "factory": FACTORY,
            "factory_args": {
                "slice": s.slice,
                "population": s.population,
                "generations": s.generations,
                "epochs": s.epochs,
                "train_per_class": None,
                "cifar_root": s.cifar_root,
                "work_dir": str(self.work),
            },
        }

    def _run_state(self) -> str | None:
        """State of the last job of the pilot run (``None`` before it exists)."""
        if not self.db.exists():
            return None
        try:
            with open_read_model(self.db) as rm:
                if not any(r["run_id"] == RUN_ID for r in rm.runs()):
                    return None
                states = [j["state"] for j in rm.jobs() if j["run_id"] == RUN_ID]
        except (StorageError, OSError):
            return None
        return states[-1] if states else "no job"

    def _tool(self, name: str) -> str:
        path = self.repo_root / "tools" / "piloto" / name
        if not path.is_file():
            raise PilotError(f"the pilot tool {path} does not exist")
        return str(path)

    def command(self, step: str) -> list[str]:
        s = self.settings
        if step == "busca":
            base = [
                self.python,
                "-m",
                "cnnlab.tcc.operation.cli",
                "run",
                "--db",
                str(self.db),
                "--artifacts",
                str(self.artifacts),
                "--repo-root",
                str(self.repo_root),
            ]
            return (
                [*base, "--spec", str(self.root / SPEC_NAME)]
                if self._run_state() is None
                else [*base, "--run-id", RUN_ID]
            )
        if step == "fidelidade":
            return [
                self.python,
                self._tool("fidelidade.py"),
                "--db",
                str(self.db),
                "--run",
                RUN_ID,
                "--out",
                str(self.fidelity_file),
                "--n",
                str(s.fidelity_n),
                "--epochs",
                *map(str, s.fidelity_epochs),
                "--cifar-root",
                s.cifar_root,
            ]
        if step == "ruido":
            return [
                self.python,
                self._tool("ruido.py"),
                "--work",
                str(self.root / "trabalho-ruido"),
                "--out",
                str(self.noise_file),
                "--sessions",
                str(s.sessions),
                "--pause-s",
                str(s.pause_s),
                "--aquecer-s",
                str(s.heat_s),
            ]
        if step == "analise":
            return [
                self.python,
                self._tool("analisa.py"),
                "--db",
                str(self.db),
                "--run",
                RUN_ID,
                "--fidelidade",
                str(self.fidelity_file),
                "--ruido",
                str(self.noise_file),
                "--out",
                str(self.analysis_dir),
            ]
        raise PilotError(f"unknown step {step!r}; one of {STEPS}")

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        src = str(self.repo_root / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUNBUFFERED"] = "1"
        env["CNNLAB_TCC_CIFAR_ROOT"] = self.settings.cifar_root
        env["CNNLAB_TCC_WORK_DIR"] = str(self.work)
        return env

    # -- what is done --------------------------------------------------------------------------------------------
    def _valid_candidates(self) -> int:
        try:
            with open_read_model(self.db) as rm:
                return len({d["candidate_id"] for d in rm.evaluation_details(RUN_ID)})
        except (StorageError, OSError):
            return 0

    def done(self, step: str) -> tuple[bool, str]:
        s = self.settings
        if step == "busca":
            state = self._run_state()
            return state == "done", f"job {state or 'not started'}"
        if step == "fidelidade":
            data = self._json(self.fidelity_file)
            if data is None:
                return False, "no file"
            need = min(s.fidelity_n, self._valid_candidates())
            covered = [
                g
                for g in data.get("genomes", [])
                if all(str(e) in g.get("accuracy", {}) or str(e) in g.get("status", {}) for e in s.fidelity_epochs)
            ]
            return bool(need) and len(covered) >= need, f"{len(covered)} of {need} genomes trained"
        if step == "ruido":
            data = self._json(self.noise_file)
            have = 0 if data is None else int(data.get("n_sessions", 0))
            return have >= s.sessions, f"{have} of {s.sessions} sessions"
        if step == "analise":
            if not self.report_file.exists():
                return False, "no piloto.json"
            newest_input = max(p.stat().st_mtime for p in (self.fidelity_file, self.noise_file) if p.exists())
            return self.report_file.stat().st_mtime >= newest_input, str(self.report_file)
        raise PilotError(f"unknown step {step!r}; one of {STEPS}")

    @staticmethod
    def _json(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def status(self) -> dict[str, dict[str, Any]]:
        return {step: dict(zip(("done", "detail"), self.done(step))) for step in STEPS}

    # -- guards --------------------------------------------------------------------------------------------------
    def bench_version(self) -> str:
        """The benchmark version of the pilot's attempts; refuses a mix or a run without a measured attempt."""
        with open_read_model(self.db) as rm:
            details = rm.evaluation_details(RUN_ID)
        if any(d["synthetic"] for d in details):
            raise PilotError("the pilot run has synthetic results: a pilot is measured, not simulated")
        versions = frozenset(str(d["bench_version"]) for d in details if d["bench_version"])
        select_runs({RUN_ID: versions})  # more than one version in the run: refused
        if not versions:
            raise PilotError("the pilot run has no attempt measured by the benchmark: nothing to analyse")
        return next(iter(versions))

    # -- the run -------------------------------------------------------------------------------------------------
    def run(
        self,
        executor: Executor = _default_executor,
        *,
        only: Sequence[str] | None = None,
        dry_run: bool = False,
        log: Callable[[str], None] = print,
    ) -> int:
        chosen = [s for s in STEPS if only is None or s in only]
        unknown = set(only or ()) - set(STEPS)
        if unknown:
            raise PilotError(f"unknown step(s) {sorted(unknown)}; one of {STEPS}")
        if not dry_run:
            self.root.mkdir(parents=True, exist_ok=True)
            self._write_config()
            if self._run_state() is None:
                _atomic_write(self.root / SPEC_NAME, self.spec())
        state: dict[str, Any] = self._json(self.root / STATE_NAME) or {"format": PILOT_FORMAT, "steps": {}}
        for step in chosen:
            finished, detail = self.done(step)
            if finished:
                log(f"{step}: already done ({detail})")
                continue
            if step == "analise" and not dry_run:
                version = self.bench_version()
                log(f"analise: benchmark version of the pilot attempts: {version}")
            argv = self.command(step)
            log(f"{step}: {shlex.join(argv)}")
            if dry_run:
                continue
            started = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            code = executor(argv, self.environment())
            state["steps"][step] = {
                "started_at": started,
                "exit_code": code,
                "finished_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
            _atomic_write(self.root / STATE_NAME, state)
            if code != 0:
                log(f"{step}: exit code {code}; stopped. Fix the cause and repeat the command: finished steps are kept")
                return code
            finished, detail = self.done(step)
            if not finished:
                log(f"{step}: the tool ended without finishing its work ({detail}); stopped. Repeat the command")
                return 1
            log(f"{step}: done ({detail})")
        if not dry_run and all(self.done(s)[0] for s in STEPS):
            summary = self.summarize()
            log(f"pilot report: {self.report_file} (sha256 {summary['piloto_json_sha256']})")
            if summary["pending_decisions"]:
                log(f"decisions the pilot could not take: {summary['pending_decisions']}")
            log("next: cnnlab-tcc campaign plan --settings S.json --pilot " + str(self.report_file))
        return 0

    def _write_config(self) -> None:
        path = self.root / CONFIG_NAME
        stored = self._json(path)
        mine = {"format": PILOT_FORMAT, "settings": self.settings.definition()}
        if stored is not None and stored != mine:
            raise PilotError(
                f"{path} holds other settings than the ones given; the pilot of a directory is one: "
                "repeat with the same settings or start a new directory"
            )
        if stored is None:
            _atomic_write(path, mine)

    def summarize(self) -> dict[str, Any]:
        """What the pilot produced, with the hashes of its outputs; written next to the report."""
        decisions = campaign.load_pilot(self.report_file)
        pending = [
            name
            for name, value in (
                ("search_epochs", decisions.epochs),
                ("latency_repetitions", decisions.latency_repetitions),
                ("reference_point", decisions.reference_point),
                ("viable_minima", decisions.viable_minima),
                ("train_time_limit_s", decisions.max_attempt_train_s),
            )
            if value is None
        ]
        with open_read_model(self.db) as rm:
            attempts = rm.evaluations(RUN_ID)
        counts: dict[str, int] = {}
        for a in attempts:
            counts[a["status"]] = counts.get(a["status"], 0) + 1
        summary = {
            "format": PILOT_FORMAT,
            "run_id": RUN_ID,
            "database": str(self.db),
            "settings": self.settings.definition(),
            "planned_genomes": self.settings.planned_genomes,
            "attempts": counts,
            "bench_version": self.bench_version(),
            "piloto_json": str(self.report_file),
            "piloto_json_sha256": decisions.file_sha256,
            "outputs_sha256": {p.name: sha256_hex(p.read_bytes()) for p in (self.fidelity_file, self.noise_file)},
            "pending_decisions": pending,
            "note": "the candidates of the pilot are not transferred to the confirmatory runs (TCC-01 section 5)",
        }
        _atomic_write(self.root / SUMMARY_NAME, summary)
        return summary


def open_pilot(root: Path, settings: PilotSettings | None, repo_root: Path) -> Pilot:
    """The pilot of ``root``: with ``settings`` (first run or a repeat that must agree), else the stored ones."""
    stored = Pilot._json(Path(root) / CONFIG_NAME)
    if settings is None:
        if stored is None:
            raise NotFoundError(f"{root} holds no pilot ({CONFIG_NAME} is missing)")
        settings = PilotSettings.from_definition(stored["settings"])
    return Pilot(Path(root), settings, Path(repo_root))


__all__ = ["Pilot", "PilotSettings", "PilotError", "open_pilot", "STEPS", "RUN_ID", "DEFAULT_SLICE"]
