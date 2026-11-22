"""The pilot end to end in one command (task A11). CPU only; no GPU, no torch in this process.

The four steps are subprocesses in real use. Here an executor stands in for them: it writes what each tool would write
(the database of the search with scripted, non-synthetic measurements; the fidelity and noise files), and for the last
step it runs the real ``tools/piloto/analisa.py`` on that database, so ``piloto.json`` is the real report and is read
back by the campaign planner. Nothing here is a measurement.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.contracts import Fitness
from cnnlab.tcc.operation import campaign, cli, pilot
from cnnlab.tcc.operation.demo import EXIT_REFUSED, DemoRunSpec, default_repo_root
from cnnlab.tcc.storage import JobQueue, SQLiteRunStore
from cnnlab.tcc.storage.testing import FakeEvaluator, make_genome, make_job, make_request, seed_reference_rows

BENCH = "tcc-bench/preliminary-v4"
REPO = default_repo_root()


def settings(**changes: Any) -> pilot.PilotSettings:
    return dataclasses.replace(pilot.PilotSettings(cifar_root="C:/data/cifar"), **changes)


def fixture(tmp_path: Path, **changes: Any) -> pilot.Pilot:
    return pilot.Pilot(tmp_path / "piloto", settings(**changes), REPO)


def build_search(
    p: pilot.Pilot, *, n: int = 24, bench: Sequence[str] | None = (BENCH,), synthetic: bool = False, finish: bool = True
) -> None:
    """What the ``busca`` step leaves: a pilot run of ``n`` valid candidates with scripted measurements."""
    with SQLiteRunStore(p.db, p.artifacts) as store:
        seed_reference_rows(store)
        job = make_job(pilot.RUN_ID, method="random", stage="pilot")
        store.create_run(job)
        evaluator = FakeEvaluator(gpu_seconds=2.0)
        for i in range(n):
            genome = make_genome(i)
            cid, aid = f"{pilot.RUN_ID}.c{i}", f"{pilot.RUN_ID}.c{i}.a1"
            store.register_candidate(job.run_id, cid, genome, {"generation": 0})
            store.begin_attempt(job.run_id, cid, aid)
            result = evaluator.evaluate(make_request(job, cid, aid, genome, store.artifact_root))
            quality: dict[str, Any] = {"synthetic": True} if synthetic else {}
            if bench:
                quality["measurement"] = {"bench_version": bench[i % len(bench)]}
            store.commit_evaluation(
                dataclasses.replace(
                    result,
                    fitness=Fitness(e=0.9 - 0.01 * i, l_ms=0.5 + 0.01 * i, v_mb=20.0 + i, s_mb=5.0 + 0.5 * i),
                    quality=quality,
                    durations_s={"train": 100.0 + i, "build": 10.0, "bench": 5.0},
                )
            )
        queue = JobQueue(store)
        queue.enqueue(f"job-{pilot.RUN_ID}", pilot.RUN_ID)
        if finish:
            lease = queue.claim_next("w", lease_s=60.0, job_id=f"job-{pilot.RUN_ID}")
            assert lease is not None
            queue.finish(lease, "done", detail="ok")


def write_fidelity(p: pilot.Pilot, n: int = 20) -> None:
    genomes = [
        {
            "candidate_id": f"{pilot.RUN_ID}.c{i}",
            "accuracy": {"5": 0.1 + 0.01 * i, "30": 0.3 + 0.012 * i},
            "train_s": {},
            "status": {},
        }
        for i in range(n)
    ]
    p.fidelity_file.write_text(json.dumps({"run_id": pilot.RUN_ID, "genomes": genomes}), encoding="utf-8")


def write_noise(p: pilot.Pilot, sessions: int = 10, cv: float = 0.01) -> None:
    p.noise_file.write_text(json.dumps({"n_sessions": sessions, "cv_of_medians": cv}), encoding="utf-8")


def run_analisa(argv: Sequence[str]) -> int:
    path = REPO / "tools" / "piloto" / "analisa.py"
    spec = importlib.util.spec_from_file_location("analisa_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main(list(argv)))


class Tools:
    """An executor that does what the tools do, recording every command it was given."""

    def __init__(self, p: pilot.Pilot, *, fail: str | None = None, bench: Sequence[str] | None = (BENCH,)) -> None:
        self.p, self.fail, self.bench, self.calls = p, fail, bench, []

    def __call__(self, argv: Sequence[str], env: Mapping[str, str]) -> int:
        tool = Path(argv[1]).name if argv[1] != "-m" else "busca"
        self.calls.append(tool)
        if self.fail == tool:
            return 3
        if tool == "busca":
            build_search(self.p, bench=self.bench)
        elif tool == "fidelidade.py":
            write_fidelity(self.p)
        elif tool == "ruido.py":
            write_noise(self.p)
        elif tool == "analisa.py":
            return run_analisa(argv[2:])
        return 0


# -- settings -----------------------------------------------------------------------------------------------------------


def test_the_defaults_make_64_genomes_over_a_slice_that_runs_end_to_end() -> None:
    s = settings()
    assert s.planned_genomes == 64 == s.population * (s.generations + 1) and s.slice == pilot.DEFAULT_SLICE
    assert (s.fidelity_n, s.fidelity_epochs, s.sessions) == (20, (5, 30), 10)


@pytest.mark.parametrize(
    "changes",
    [
        {"slice": "nope"},
        {"population": 15},
        {"population": 0},
        {"generations": -1},
        {"fidelity_epochs": ()},
        {"sessions": 0},
        {"pause_s": -1.0},
    ],
)
def test_nonsense_settings_are_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(pilot.PilotError):
        settings(**changes)


def test_the_spec_is_a_valid_random_pilot_run_of_its_own(tmp_path: Path) -> None:
    spec = fixture(tmp_path).spec()
    run = DemoRunSpec.from_mapping(spec)
    assert (run.method, run.stage, run.plan_id, run.run_id) == ("random", "pilot", None, pilot.RUN_ID)
    assert (run.search_seed, run.train_seed) != (0, 0) and run.factory == pilot.FACTORY
    args = spec["factory_args"]
    assert args["population"] == 16 and args["generations"] == 3 and args["train_per_class"] is None
    assert args["slice"] == pilot.DEFAULT_SLICE and args["work_dir"].endswith("trabalho")


# -- the commands -------------------------------------------------------------------------------------------------------


def test_every_step_has_the_command_of_the_documented_sequence(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    busca, fid, noise, analysis = (p.command(s) for s in pilot.STEPS)
    assert (
        busca[1:5] == ["-m", "cnnlab.tcc.operation.cli", "run", "--db"]
        and "--spec" in busca
        and "--run-id" not in busca
    )
    assert fid[1].endswith("fidelidade.py") and fid[fid.index("--epochs") + 1 : fid.index("--epochs") + 3] == [
        "5",
        "30",
    ]
    assert noise[1].endswith("ruido.py") and "--aquecer-s" in noise and noise[noise.index("--sessions") + 1] == "10"
    assert analysis[1].endswith("analisa.py") and "--fidelidade" in analysis and "--ruido" in analysis
    assert analysis[analysis.index("--out") + 1] == str(p.analysis_dir)
    env = p.environment()
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(REPO / "src")
    assert env["CNNLAB_TCC_CIFAR_ROOT"] == "C:/data/cifar" and env["CNNLAB_TCC_WORK_DIR"] == str(p.work)


def test_a_dry_run_prints_the_commands_and_writes_nothing(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    lines: list[str] = []
    assert p.run(Tools(p), dry_run=True, log=lines.append) == 0
    assert [l.split(":")[0] for l in lines] == list(pilot.STEPS) and not p.root.exists()


def test_the_search_continues_the_run_when_it_already_exists(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    build_search(p, finish=False)
    command = p.command("busca")
    assert "--run-id" in command and "--spec" not in command and command[command.index("--run-id") + 1] == pilot.RUN_ID


# -- running -------------------------------------------------------------------------------------------------------------


def test_the_whole_pilot_runs_in_order_and_ends_with_a_report_the_planner_reads(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    tools = Tools(p)
    lines: list[str] = []
    assert p.run(tools, log=lines.append) == 0
    assert tools.calls == ["busca", "fidelidade.py", "ruido.py", "analisa.py"]
    assert all(p.done(s)[0] for s in pilot.STEPS)
    decisions = campaign.load_pilot(p.report_file)
    assert (
        decisions.epochs == 5 and decisions.latency_repetitions == 1000
    )  # rho 1 and CV 1%: the rules keep 5 and 1,000
    assert decisions.reference_point is not None and decisions.viable_minima is not None
    assert decisions.max_attempt_train_s is not None
    summary = json.loads((p.root / pilot.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert (
        summary["bench_version"] == BENCH
        and summary["pending_decisions"] == []
        and summary["attempts"] == {"valid": 24}
    )
    assert summary["piloto_json_sha256"] == decisions.file_sha256 and summary["planned_genomes"] == 64
    assert any("campaign plan" in l and "--pilot" in l for l in lines)
    state = json.loads((p.root / pilot.STATE_NAME).read_text(encoding="utf-8"))
    assert all(state["steps"][s]["exit_code"] == 0 for s in pilot.STEPS)


def test_a_repeat_skips_what_is_done(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    p.run(Tools(p), log=lambda _: None)
    again = Tools(p)
    lines: list[str] = []
    assert p.run(again, log=lines.append) == 0 and again.calls == []
    assert [l for l in lines if "already done" in l].__len__() == 4


def test_a_failing_step_stops_the_pilot_and_a_repeat_resumes_there(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    broken = Tools(p, fail="ruido.py")
    lines: list[str] = []
    assert p.run(broken, log=lines.append) == 3
    assert broken.calls == ["busca", "fidelidade.py", "ruido.py"] and any("exit code 3" in l for l in lines)
    assert not p.report_file.exists() and p.done("fidelidade")[0] and not p.done("ruido")[0]
    fixed = Tools(p)
    assert p.run(fixed, log=lambda _: None) == 0 and fixed.calls == ["ruido.py", "analisa.py"]


def test_a_tool_that_ends_without_its_work_is_not_taken_as_done(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    build_search(p)
    lazy: list[str] = []

    def do_nothing(argv: Sequence[str], env: Mapping[str, str]) -> int:
        lazy.append(argv[1])
        return 0

    lines: list[str] = []
    assert p.run(do_nothing, log=lines.append) == 1 and len(lazy) == 1 and any("without finishing" in l for l in lines)


def test_only_runs_the_named_steps(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    p.run(Tools(p), log=lambda _: None)
    p.report_file.unlink()
    tools = Tools(p)
    assert p.run(tools, only=["analise"], log=lambda _: None) == 0 and tools.calls == ["analisa.py"]
    with pytest.raises(pilot.PilotError, match="unknown step"):
        p.run(tools, only=["inventada"], log=lambda _: None)


def test_the_settings_of_a_directory_are_one(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    p.run(Tools(p), only=["busca"], log=lambda _: None)
    other = pilot.Pilot(p.root, settings(population=8), REPO)
    with pytest.raises(pilot.PilotError, match="other settings"):
        other.run(Tools(other), log=lambda _: None)


# -- guards ---------------------------------------------------------------------------------------------------------------


def test_attempts_of_two_bench_versions_are_not_analysed_together(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    tools = Tools(p, bench=("tcc-bench/preliminary-v3", BENCH))
    with pytest.raises(Exception, match="more than one benchmark version"):
        p.run(tools, log=lambda _: None)
    assert tools.calls == ["busca", "fidelidade.py", "ruido.py"] and not p.report_file.exists()


def test_a_synthetic_pilot_is_refused_for_the_analysis(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    build_search(p, synthetic=True)
    write_fidelity(p)
    write_noise(p)
    with pytest.raises(pilot.PilotError, match="synthetic"):
        p.run(Tools(p), only=["analise"], log=lambda _: None)


def test_a_pilot_without_a_benchmarked_attempt_has_nothing_to_analyse(tmp_path: Path) -> None:
    p = fixture(tmp_path)
    build_search(p, bench=None)
    write_fidelity(p)
    write_noise(p)
    with pytest.raises(pilot.PilotError, match="no attempt measured by the benchmark"):
        p.run(Tools(p), only=["analise"], log=lambda _: None)


# -- the command line ---------------------------------------------------------------------------------------------------------


def test_cli_dry_run_status_and_a_repeat_with_the_stored_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "p"
    assert cli.main(["piloto", "run", "--dir", str(root), "--cifar-root", "C:/cifar", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "64 genomes by random search" in out and "fidelidade.py" in out and not root.exists()
    assert cli.main(["piloto", "run", "--dir", str(root), "--dry-run"]) == EXIT_REFUSED  # first run: where is the data?
    assert "--cifar-root" in capsys.readouterr().err
    assert cli.main(["piloto", "status", "--dir", str(root)]) == EXIT_REFUSED  # nothing there
    p = pilot.Pilot(root, settings(cifar_root="C:/cifar", slice="fp16_regions"), REPO)
    p.run(Tools(p), only=["busca"], log=lambda _: None)
    assert cli.main(["piloto", "status", "--dir", str(root)]) == 0
    status = capsys.readouterr().out
    assert "busca" in status and "done" in status and "fidelidade" in status and "pending" in status
    # a repeat without flags takes the stored settings (slice included); a flag that disagrees is refused
    assert cli.main(["piloto", "run", "--dir", str(root), "--dry-run"]) == 0
    assert "'fp16_regions'" in capsys.readouterr().out
    assert cli.main(["piloto", "run", "--dir", str(root), "--slice", "fp32", "--only", "busca"]) == EXIT_REFUSED
    assert "other settings" in capsys.readouterr().err
