"""`cnnlab-tcc`: one entry point over what the operation layer already offers.

A thin shell with no logic of its own. Each command hands its arguments to
the module that implements it::

    cnnlab-tcc run     --db DB --artifacts DIR --spec SPEC.json     # prepare if needed, then work
    cnnlab-tcc run     --db DB --artifacts DIR --job-id JOB         # resume a job
    cnnlab-tcc run     --db DB --artifacts DIR --run-id RUN         # continue a run (new job if needed)
    cnnlab-tcc status  --db DB [--run RUN --evaluations] [--json]
    cnnlab-tcc stop    --db DB --artifacts DIR --job-id JOB --reason TEXT
    cnnlab-tcc verify  --db DB --artifacts DIR                      # catalog against files, read-only
    cnnlab-tcc verify  --backup DIR                                 # a backup against its manifest
    cnnlab-tcc backup  --db DB --artifacts DIR --dest NEW_DIR --label TEXT
    cnnlab-tcc export  --db DB --run RUN --dest NEW_DIR             # CSV + manifest, read-only
    cnnlab-tcc campaign plan    --settings S.json [--pilot piloto.json] [--db DB --artifacts DIR --freeze]
    cnnlab-tcc campaign export  --db DB --plan PLAN --dest NEW_DIR  # the frozen plan and its SHA-256
    cnnlab-tcc campaign enqueue --db DB --plan PLAN --queue-dir DIR # the runs, in the order of the plan
    cnnlab-tcc campaign curves  --db DB --dest NEW_DIR [--reference E,L,V,S]  # hypervolume by GPU time, read-only
    cnnlab-tcc campaign finalists --db DB --plan PLAN --dest NEW_DIR [--freeze LIST --artifacts DIR]  # section 6
    cnnlab-tcc campaign inference --paired hv_pareado.csv [--sensitivity ...] [--final-results F.csv] --dest NEW_DIR
    cnnlab-tcc campaign h4 --db DB --dest NEW_DIR [--run RUN ...] [--no-recompute]   # latency vs MACs, parameters
    cnnlab-tcc tui    --db DB --artifacts DIR [--spec SPEC.json] [--read-only]
    cnnlab-tcc view    --db DB [streamlit options]
    cnnlab-tcc piloto run    --dir DIR --cifar-root CIFAR [--slice S] [--only STEP ...] [--dry-run]  # the whole pilot
    cnnlab-tcc piloto status --dir DIR
    cnnlab-tcc final prepare|run|artifacts|enqueue-tests|retry|export|status --db DB --list L --list-sha256 H ...   # the final phase
    cnnlab-tcc fila    --dir DIR [--repo-root PATH] [--poll-s 2] [--lease-s 120]   # queue executor

`stop` is the cooperative cancel: the worker finishes the attempt in
progress, commits it and ends the job. `verify` and `export` open no writer,
so they never migrate nor create a database. `export` refuses a destination
that exists and a run with synthetic results (unless --allow-synthetic, which
labels the columns). Interfaces are imported only when asked for.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from cnnlab.tcc.operation import campaign, demo, final_stage, finalists, h4, hv_analysis, inference, launcher, pilot
from cnnlab.tcc.operation.export import MANIFEST_NAME, export_run
from cnnlab.tcc.storage.artifacts import scan_artifacts
from cnnlab.tcc.storage.backup import verify_backup
from cnnlab.tcc.storage.errors import StorageError
from cnnlab.tcc.storage.readmodel import open_read_model
from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore

PROG = "cnnlab-tcc"
COMMANDS = ("run", "status", "stop", "verify", "backup", "export", "campaign", "tui", "view", "fila", "piloto", "final")
EXIT_PROBLEMS = 2


def _stop(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{PROG} stop", description="Cooperative cancel of a job.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    job = launcher.request_cancel(args.db, args.artifacts, args.job_id, reason=args.reason)
    print(f"job {job.job_id}: {job.state} (cancel requested: {job.cancel_requested})")
    return 0


def _export(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{PROG} export", description="Export one run to CSV, read-only.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--run", required=True, help="run id, as shown by status")
    parser.add_argument("--dest", type=Path, required=True, help="directory to create; must not exist")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="export results of a fake evaluator too, in columns labeled _SINTETICO",
    )
    args = parser.parse_args(argv)
    result = export_run(args.db, args.run, args.dest, allow_synthetic=args.allow_synthetic)
    for entry in result.manifest["files"]:
        print(f"{entry['name']}: {entry['rows']} row(s) sha256={entry['sha256']}")
    if result.manifest["synthetic"]:
        print("SYNTHETIC results exported in labeled columns; they are not measurements")
    if result.manifest["changed_during_export"]:
        print("warning: the run changed while it was read; the files may not describe one instant")
    print(f"export {result.path}: run {args.run}, {len(result.files)} file(s), manifest {MANIFEST_NAME}")
    return 0


def _fila(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{PROG} fila", description="Executor of the demo run queue.")
    parser.add_argument("--dir", type=Path, required=True, help="queue directory (holds fila.json and db/)")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--poll-s", type=float, default=2.0)
    parser.add_argument("--lease-s", type=float, default=120.0)
    parser.add_argument(
        "--allow-synthetic", action="store_true", help="let the workers run a fake wiring (tests of the operation only)"
    )
    args = parser.parse_args(argv)
    repo_root = None if args.repo_root is None else Path(args.repo_root).resolve()  # the workers get an absolute path
    demo.require_imported_code(repo_root)  # the commit it records is the code it runs
    from cnnlab.tcc.operation.scheduler import Scheduler

    def say(message: str) -> None:
        print(message, flush=True)

    return Scheduler(
        args.dir,
        repo_root=repo_root,
        poll_s=args.poll_s,
        lease_s=args.lease_s,
        allow_synthetic=args.allow_synthetic,
        log=say,
    ).run()


def _piloto(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} piloto", description="The pilot end to end, in a directory of its own."
    )
    sub = parser.add_subparsers(dest="action", required=True)
    defaults = pilot.PilotSettings(cifar_root="-")
    p = sub.add_parser(
        "run", help="run the pending steps: busca, fidelidade, ruido, analise (a finished step is skipped)"
    )
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--cifar-root", help="folder of the CIFAR-100 data (first run; stored afterwards)")
    p.add_argument("--slice", help=f"search slice (default {defaults.slice}; G7 publishes the admitted ones)")
    p.add_argument("--population", type=int, help=f"default {defaults.population}")
    p.add_argument(
        "--generations", type=int, help=f"after the first; default {defaults.generations}: 16 x (3 + 1) = 64 genomes"
    )
    p.add_argument("--epochs", type=int, help=f"training epochs of the search; default {defaults.epochs}")
    p.add_argument("--fidelity-n", type=int, help=f"default {defaults.fidelity_n}")
    p.add_argument(
        "--sessions", type=int, help=f"sessions of the sentinel noise measurement; default {defaults.sessions}"
    )
    p.add_argument("--pause-s", type=float, help=f"default {defaults.pause_s:g}")
    p.add_argument(
        "--heat-s", type=float, help=f"busy the GPU this long before each noise session; default {defaults.heat_s:g}"
    )
    p.add_argument("--only", action="append", choices=pilot.STEPS, help="run only this step (repeat for more)")
    p.add_argument("--dry-run", action="store_true", help="print the commands and write nothing")
    p.add_argument("--repo-root", type=Path, default=None)
    p = sub.add_parser("status", help="which steps are done")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args(argv)
    repo = args.repo_root or demo.default_repo_root()
    if args.action == "status":
        current = pilot.open_pilot(args.dir, None, repo)
        for step, info in current.status().items():
            print(f"{step:<11} {'done' if info['done'] else 'pending':<8} {info['detail']}")
        return 0
    stored = pilot.Pilot._json(args.dir / pilot.CONFIG_NAME)
    if args.cifar_root is None and stored is None:
        raise ValueError("--cifar-root is needed on the first run of a pilot directory")
    # a repeat takes the stored settings; a flag given again must agree with them (the pilot of a directory is one)
    base = pilot.PilotSettings.from_definition(stored["settings"]).definition() if stored else defaults.definition()
    given = {
        "cifar_root": args.cifar_root,
        "slice": args.slice,
        "population": args.population,
        "generations": args.generations,
        "epochs": args.epochs,
        "fidelity_n": args.fidelity_n,
        "sessions": args.sessions,
        "pause_s": args.pause_s,
        "heat_s": args.heat_s,
    }
    settings = pilot.PilotSettings.from_definition({**base, **{k: v for k, v in given.items() if v is not None}})
    current = pilot.open_pilot(args.dir, settings, repo)
    print(f"pilot {args.dir}: {settings.planned_genomes} genomes by random search over slice {settings.slice!r}")
    return current.run(only=args.only, dry_run=args.dry_run)


def _final(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} final",
        description="The final phase: retrain the frozen finalists and open the official test once per engine.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    def common(p: argparse.ArgumentParser, *, writer: bool = True, hash_required: bool = True) -> None:
        p.add_argument("--db", type=Path, required=True)
        if writer:
            p.add_argument("--artifacts", type=Path, required=True)
        p.add_argument("--list", dest="list_id", required=True, help="id of the frozen finalist list")
        p.add_argument(
            "--list-sha256",
            required=hash_required,
            help="hash of the frozen list, as `campaign finalists --freeze` printed it",
        )

    p = sub.add_parser("prepare", help="create one finalist run and its build job per retraining of the frozen list")
    common(p)
    p.add_argument("--factory", required=True, help="'package.module:function' returning the FinalWiring (task G9)")
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="invalid attempts a build may use (a repetition is justified: see retry)",
    )
    p.add_argument("--budget-gpu-s", type=float, default=final_stage.DEFAULT_BUDGET_GPU_S)
    p.add_argument("--allow-synthetic", action="store_true")
    p = sub.add_parser("run", help="run the queued jobs of one pass, one after the other")
    common(p)
    p.add_argument("--mode", choices=final_stage.MODES, required=True)
    p.add_argument("--factory", required=True)
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--worker-id", default="final")
    p.add_argument("--lease-s", type=float, default=120.0)
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument(
        "--allow-code-change",
        action="store_true",
        help="continue jobs prepared with other code; the change is recorded in every claim (an amendment)",
    )
    p = sub.add_parser("artifacts", help="check the engines and write the manifest whose hash opens the test")
    common(p)
    p.add_argument("--dest", type=Path, required=True)
    p = sub.add_parser("enqueue-tests", help="queue the test jobs, after confirming the hash of the frozen engines")
    common(p)
    p.add_argument("--artifacts-sha256", required=True, help="manifest hash printed by `final artifacts`")
    p.add_argument("--factory", required=True)
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--allow-synthetic", action="store_true")
    p = sub.add_parser("retry", help="a new job for the units whose last job failed, with the reason recorded")
    common(p)
    p.add_argument("--mode", choices=final_stage.MODES, required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--max-attempts", type=int, default=None)
    p = sub.add_parser("export", help="the results CSV that `campaign inference` reads, and its manifest")
    common(p, writer=False)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true", help="export what is tested so far (tests only)")
    p = sub.add_parser("status", help="where each retraining stands")
    common(p, writer=False, hash_required=False)
    args = parser.parse_args(argv)
    if args.action == "status":
        for row in final_stage.status(args.db, args.list_id):
            print(
                f"{row['retrain_id']:<28} block {row['seed_block']}  selections {row['selections']}  built {row['built']!s:<5} "
                f"test {'opened' if row['test_opened'] else 'closed'}  build jobs {row['build_jobs']}  test jobs {row['test_jobs']}"
            )
        return 0
    if args.action == "export":
        manifest = final_stage.export_final(
            args.db,
            args.list_id,
            args.list_sha256,
            args.dest,
            allow_synthetic=args.allow_synthetic,
            allow_incomplete=args.allow_incomplete,
        )
        for entry in manifest["files"]:
            print(f"{entry['name']}: {entry['rows']} row(s) sha256={entry['sha256']}")
        print(
            f"{manifest['tested']} of {manifest['retrainings']} retraining(s) tested; bench version {manifest['bench_version']}"
        )
        if manifest["synthetic"]:
            print("SYNTHETIC results exported in labeled columns; they are not measurements")
        print(f"export {args.dest}: manifest {final_stage.MANIFEST_NAME}")
        return 0
    if args.action == "artifacts":
        manifest = final_stage.export_artifacts_manifest(
            args.db, args.artifacts, args.list_id, args.list_sha256, args.dest
        )
        print(f"{len(manifest['engines'])} engine(s) checked against the catalog and the files")
        print(f"artifacts manifest sha256={manifest['manifest_sha256']}")
        print(f"export {args.dest}: {final_stage.ARTIFACTS_MANIFEST_NAME}")
        return 0
    if args.action == "retry":
        with SQLiteRunStore(args.db, args.artifacts) as store:
            created = final_stage.requeue_failed(
                store, args.list_id, args.list_sha256, args.mode, reason=args.reason, max_attempts=args.max_attempts
            )
        print(f"{len(created)} job(s) queued: {created}")
        return 0
    final, code_version, repo = final_stage.load_final(
        args.db, args.list_id, args.list_sha256, args.factory, repo_root=args.repo_root
    )
    if args.action == "prepare":
        with SQLiteRunStore(args.db, args.artifacts) as store:
            prepared = final_stage.prepare_builds(
                store,
                args.list_id,
                args.list_sha256,
                final,
                code_version,
                factory=args.factory,
                repo_root=repo,
                max_attempts=args.max_attempts,
                budget_gpu_s=args.budget_gpu_s,
                allow_synthetic=args.allow_synthetic,
            )
        print(
            f"{len(prepared)} retraining(s) of the list {args.list_id}: {sum(p.created for p in prepared)} run(s) created"
        )
        return 0
    if args.action == "enqueue-tests":
        with SQLiteRunStore(args.db, args.artifacts) as store:
            queued = final_stage.enqueue_tests(
                store, args.list_id, args.list_sha256, args.artifacts_sha256, args.artifacts, final, code_version, repo
            )
        print(f"{len(queued)} test job(s) queued")
        return 0
    results = final_stage.run_pending(
        args.db,
        args.artifacts,
        args.list_id,
        args.list_sha256,
        args.mode,
        final,
        code_version,
        worker_id=args.worker_id,
        lease_s=args.lease_s,
        allow_synthetic=args.allow_synthetic,
        allow_code_change=args.allow_code_change,
        install_sigint=True,
    )
    failed = [r for r in results if r.outcome is None or r.outcome.end_state != "done"]
    print(f"{len(results) - len(failed)} job(s) done" + (f"; stopped at {failed[0].job_id}" if failed else ""))
    return EXIT_PROBLEMS if failed else 0


def _campaign(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{PROG} campaign", description="The plan of the confirmatory campaign.")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser(
        "plan", help="show the plan for these settings; with --freeze write it to the database and freeze it"
    )
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--pilot", type=Path, help="piloto.json of tools/piloto/analisa.py: its decisions fill the plan")
    p.add_argument("--db", type=Path)
    p.add_argument("--artifacts", type=Path)
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--factory", help="override settings.factory ('package.module:function')")
    p.add_argument("--allow-synthetic", action="store_true", help="accept a fake wiring (tests of the operation only)")
    p.add_argument("--allow-pending", action="store_true", help="freeze with fields still pending (tests only)")
    p.add_argument("--allow-dirty", action="store_true", help="freeze on a dirty tree (tests only)")
    p = sub.add_parser("export", help="write the plan and its SHA-256 into a new directory (read-only)")
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--dest", type=Path, required=True)
    p = sub.add_parser("enqueue", help="put the runs of a frozen plan in the queue, in the order of the plan")
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--queue-dir", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument(
        "--allow-dirty", action="store_true", help="queue from a checkout other than the frozen one (tests only)"
    )
    p = sub.add_parser("finalists", help="choose the finalists of a frozen plan from the validation measurements")
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--dest", type=Path, required=True, help="directory to create; must not exist")
    p.add_argument("--allow-unfinished", action="store_true", help="include runs that have not finished (tests only)")
    p.add_argument(
        "--allow-synthetic", action="store_true", help="select among results of a fake evaluator, in labeled columns"
    )
    p.add_argument("--freeze", metavar="LIST_ID", help="after exporting, freeze the list in the database under this id")
    p.add_argument("--artifacts", type=Path, help="artifact root of the database (needed by --freeze)")
    p.add_argument("--author", default="Pedro Antunes", help="author recorded with a frozen list")
    p = sub.add_parser("curves", help="hypervolume by active GPU time and by unique evaluations, per method and seed")
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True, help="directory to create; must not exist")
    p.add_argument("--reference", help="E,L,V,S of the reference point (default: the frozen plan of the runs)")
    p.add_argument("--run", action="append", help="analyse only this run (repeat the option for more)")
    p.add_argument("--grid-s", type=float, default=hv_analysis.DEFAULT_GRID_S, help="step of the time grid, seconds")
    p.add_argument("--bench-version", help="keep only the runs measured by this benchmark version (a mix is refused)")
    p.add_argument(
        "--allow-synthetic", action="store_true", help="analyse results of a fake evaluator, in labeled columns"
    )
    p = sub.add_parser("h4", help="H4: Spearman of L against MACs and parameters, and the grouped predictive model")
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True, help="directory to create; must not exist")
    p.add_argument("--run", action="append", help="analyse only this run (repeat the option for more)")
    p.add_argument("--bench-version", help="keep only the runs measured by this benchmark version (a mix is refused)")
    p.add_argument(
        "--no-recompute",
        action="store_true",
        help="use only the MACs and parameters the evaluator recorded; do not rebuild networks on the CPU",
    )
    p.add_argument(
        "--allow-synthetic", action="store_true", help="analyse results of a fake evaluator, in labeled columns"
    )
    p = sub.add_parser(
        "where", help="which code this command runs, which commit it would record, and whether a plan froze it"
    )
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--db", type=Path)
    p.add_argument("--plan")
    p = sub.add_parser("inference", help="H2/H3 (exact paired test, Holm) and the H1 table from exported tables")
    p.add_argument("--paired", type=Path, help="hv_pareado.csv of `campaign curves`")
    p.add_argument(
        "--sensitivity", type=Path, help="hv_pareado_sensibilidade.csv of `campaign curves` (needs --paired)"
    )
    p.add_argument(
        "--final-results",
        type=Path,
        help="CSV of the final results: seed_block, method, role, test_accuracy, l_ms, v_mb, s_mb",
    )
    p.add_argument("--dest", type=Path, required=True, help="directory to create; must not exist")
    p.add_argument("--allow-synthetic", action="store_true", help="analyse synthetic tables, in labeled columns")
    args = parser.parse_args(argv)
    if args.action == "where":
        return _where(args)
    if args.action in ("plan", "enqueue"):
        demo.require_imported_code(args.repo_root)  # the commit recorded is the code that runs
    if args.action == "h4":
        manifest = h4.export_h4(
            args.db,
            args.dest,
            run_ids=args.run,
            recompute=not args.no_recompute,
            allow_synthetic=args.allow_synthetic,
            bench_version=args.bench_version,
        )
        print(f"{manifest['points']} point(s) from {len(manifest['runs'])} run(s); cost {manifest['cost_sources']}")
        for reason, count in manifest["skipped"].items():
            print(f"skipped {count}: {reason}")
        if manifest["model_note"]:
            print(f"model not fitted: {manifest['model_note']}")
        with open(args.dest / "h4_correlacao.csv", encoding="utf-8", newline="") as fh:
            import csv

            for row in csv.DictReader(fh):
                value = next((v for k, v in row.items() if k.startswith("spearman")), "")
                print(f"spearman {row['x']} ({row['scope']}, n={row['n']}): {value}")
        for entry in manifest["files"]:
            print(f"{entry['name']}: {entry['rows']} row(s) sha256={entry['sha256']}")
        if manifest["synthetic"]:
            print("SYNTHETIC results analysed in labeled columns; they are not measurements")
        print(f"export {args.dest}: manifest {h4.MANIFEST_NAME}")
        return 0
    if args.action == "inference":
        if args.sensitivity is not None and args.paired is None:
            raise ValueError("--sensitivity needs --paired")
        manifest = inference.write_analysis(
            args.dest,
            paired=args.paired,
            sensitivity=args.sensitivity,
            final_results=args.final_results,
            allow_synthetic=args.allow_synthetic,
        )
        for label in ("official", "sensitivity"):
            for t in manifest["summary"].get(label, ()):
                print(
                    f"{label} {t['hypothesis']}: p={t['p']:.6f} Holm={t['p_holm']:.6f} "
                    f"{'rejects' if t['reject'] else 'does not reject'} at 5% ({t['pairs']} pairs)"
                )
        h1 = manifest["summary"].get("h1")
        if h1:
            print(
                f"H1: {h1['with_both']} of {h1['blocks']} seeds with both engines; supporting {h1['supporting']}; "
                f"criterion ({h1['support_criterion']}) {'met' if h1['criterion_met'] else 'not met'}"
            )
        for entry in manifest["files"]:
            print(f"{entry['name']}: sha256={entry['sha256']}")
        if manifest["synthetic"]:
            print("SYNTHETIC results analysed in labeled columns; they are not measurements")
        print(f"export {args.dest}: manifest {inference.MANIFEST_NAME}")
        return 0
    if args.action == "finalists":
        proposal = finalists.propose(
            args.db, args.plan, allow_unfinished=args.allow_unfinished, allow_synthetic=args.allow_synthetic
        )
        manifest = finalists.export_proposal(proposal, args.dest)
        print(
            f"plan {proposal.plan_id}: {manifest['n_selections']} selections, {manifest['n_retrains']} retrainings "
            f"(at most {manifest['max_retrains']}), {manifest['h1_failures']} seed(s) without an H1 candidate"
        )
        for entry in manifest["files"]:
            print(f"{entry['name']}: sha256={entry['sha256']}")
        print(f"list sha256={manifest['list_sha256']}")
        if proposal.synthetic:
            print("SYNTHETIC results selected in labeled columns; they are not measurements")
        if args.freeze:
            if args.artifacts is None:
                raise ValueError("--freeze needs --artifacts")
            from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore

            with SQLiteRunStore(args.db, args.artifacts) as store:
                frozen = finalists.freeze_list(store, proposal, list_id=args.freeze, author=args.author)
            print(f"finalist list {args.freeze} frozen in the database, sha256={frozen}")
        print(f"export {args.dest}: manifest {finalists.MANIFEST_NAME}")
        return 0
    if args.action == "curves":
        reference = None if args.reference is None else [float(x) for x in args.reference.split(",")]
        done = hv_analysis.export_hv(
            args.db,
            args.dest,
            reference=reference,
            run_ids=args.run,
            grid_s=args.grid_s,
            allow_synthetic=args.allow_synthetic,
            bench_version=args.bench_version,
        )
        point = done.manifest["reference_point"]
        print(
            f"reference point E={point['E']} L={point['L_ms']} V={point['V_mb']} S={point['S_mb']} ({point['origin']})"
        )
        for entry in done.manifest["files"]:
            print(f"{entry['name']}: {entry['rows']} row(s) sha256={entry['sha256']}")
        if done.manifest["synthetic"]:
            print("SYNTHETIC results analysed in labeled columns; they are not measurements")
        for mismatch in done.manifest["snapshot_mismatches"]:
            print(f"warning: {mismatch}")
        print(f"export {done.path}: {len(done.curves)} run(s), manifest {hv_analysis.MANIFEST_NAME}")
        return 0
    if args.action == "export":
        exported = campaign.export_plan(args.db, args.plan, args.dest)
        print(f"plan {exported.plan_id} [{exported.status}] sha256={exported.plan_sha256}")
        print(f"export {args.dest}: {campaign.PLAN_EXPORT_NAME}, {campaign.PLAN_HASH_NAME}")
        return 0
    if args.action == "enqueue":
        plan = campaign.read_plan(args.db, args.plan)
        if not plan["hash_ok"]:
            raise StorageError(f"plan {args.plan!r}: the stored hash is not the hash of its content")
        if plan["status"] != "frozen":
            raise StorageError(f"plan {args.plan!r} is {plan['status']}: only a frozen plan can be queued")
        frozen = plan["content"]["code_freeze"]
        now = demo.capture_code_version(campaign.CODE_LABEL_PREFIX, args.repo_root)
        if (now.git_sha != frozen["git_sha"] or now.dirty) and not args.allow_dirty:
            raise StorageError(
                f"the plan froze code {frozen['git_sha'][:12]}; this checkout is {str(now.git_sha)[:12]}"
                + (" with uncommitted changes" if now.dirty else "")
                + ": queue it from the frozen revision"
            )
        total = len(plan["content"]["run_order"])
        added = campaign.enqueue(args.queue_dir, plan)
        print(
            f"queued {len(added)} of {total} runs of {args.plan}"
            + (f" ({total - len(added)} were already there)" if len(added) < total else "")
        )
        for name in added[:8]:
            print(f"  {name}")
        return 0
    return _campaign_plan(args)


def _where(args: argparse.Namespace) -> int:
    """Print which code runs and what would be recorded for it; exit 2 if it is not the checkout named."""
    import cnnlab

    imported = demo.default_repo_root()
    repo = Path(args.repo_root).resolve() if args.repo_root is not None else imported
    code = demo.capture_code_version(campaign.CODE_LABEL_PREFIX, repo)
    tags = demo.tags_at_head(repo)
    print(f"cnnlab imported from {Path(cnnlab.__file__).resolve().parent}")
    print(
        f"repository root {repo}" + ("" if repo == imported else f"  <-- NOT the root of the running code ({imported})")
    )
    print(
        f"commit {code.git_sha}  tags {list(tags) or 'none'}  uncommitted changes {'yes' if code.dirty else 'no'}"
        + (f"  ({code.error})" if code.error else "")
    )
    ok = repo == imported and code.git_sha is not None
    if args.plan:
        if args.db is None:
            raise ValueError("--plan needs --db")
        plan = campaign.read_plan(args.db, args.plan)
        frozen = plan["content"]["code_freeze"]
        same = code.git_sha == frozen["git_sha"] and not code.dirty
        print(
            f"plan {args.plan} [{plan['status']}] froze {frozen['git_sha']} tags {frozen.get('tags') or 'none'}: "
            f"this checkout {'matches' if same else 'DOES NOT match'}"
        )
        ok = ok and same
    return 0 if ok else EXIT_PROBLEMS


def _campaign_plan(args: argparse.Namespace) -> int:
    settings = campaign.CampaignSettings.load(args.settings)
    pilot = campaign.load_pilot(args.pilot) if args.pilot else None
    if args.freeze and (args.db is None or args.artifacts is None):
        raise ValueError("--freeze needs --db and --artifacts")
    code = demo.capture_code_version(campaign.CODE_LABEL_PREFIX, args.repo_root)
    wiring = demo.load_factory(args.factory or settings.factory)(campaign.representative_spec(settings))
    demo.require_real_wiring(wiring, allow_synthetic=args.allow_synthetic)
    repo = Path(args.repo_root) if args.repo_root is not None else demo.default_repo_root()
    if not args.freeze:
        plan_id, content = campaign.preview(settings, wiring, code, repo, pilot)
        print(f"preview of plan {plan_id} (nothing was written)")
        print(
            campaign.summary(
                {"plan_id": plan_id, "status": "draft", "plan_sha256": "(computed when written)", "content": content}
            )
        )
        return 0
    from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore

    with SQLiteRunStore(args.db, args.artifacts) as store:
        made = campaign.create_plan(
            store,
            settings,
            wiring,
            code,
            repo,
            freeze=True,
            allow_pending=args.allow_pending,
            allow_dirty=args.allow_dirty,
            pilot=pilot,
        )
    print(
        campaign.summary(
            {"plan_id": made.plan_id, "status": made.status, "plan_sha256": made.plan_sha256, "content": made.content}
        )
    )
    return 0


def _verify(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{PROG} verify", description="Read-only check of files against records.")
    parser.add_argument("--db", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--backup", type=Path, help="verify a backup directory against its manifest instead")
    args = parser.parse_args(argv)
    if args.backup is not None:
        if args.db is not None or args.artifacts is not None:
            parser.error("--backup is used alone")
        result = verify_backup(args.backup)
        for problem in result.problems:
            print(f"problem: {problem}")
        print(f"backup {result.path}: {'ok' if result.ok else f'{len(result.problems)} problem(s)'}")
        return 0 if result.ok else EXIT_PROBLEMS
    if args.db is None or args.artifacts is None:
        parser.error("give --db and --artifacts, or --backup")
    with open_read_model(args.db) as rm:
        schema = rm.schema_version()
        catalog = {row["rel_path"]: row["sha256"] for row in rm.artifacts()}
    scan = scan_artifacts(args.artifacts, catalog, hash_files=True)
    for label, paths in (
        ("missing", scan.missing),
        ("mismatched", scan.mismatched),
        ("orphan", scan.orphans),
        ("temporary", scan.temporaries),
    ):
        for path in paths:
            print(f"{label}: {path}")
    verified = len(catalog) - len(scan.missing) - len(scan.mismatched)
    print(
        f"schema v{schema}: {len(catalog)} cataloged, {verified} verified, {len(scan.missing)} missing, "
        f"{len(scan.mismatched)} mismatched, {len(scan.orphans)} orphan, {len(scan.temporaries)} temporary"
    )
    return 0 if scan.ok else EXIT_PROBLEMS


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else EXIT_PROBLEMS
    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        print(f"error: unknown command {command!r}; one of: {', '.join(COMMANDS)}", file=sys.stderr)
        return EXIT_PROBLEMS
    if command == "run":
        return demo.main(["start", *rest])
    if command in ("status", "backup"):
        return demo.main([command, *rest])
    if command == "tui":
        from cnnlab.tcc import demo_tui

        return demo_tui.main(rest)
    if command == "view":
        from cnnlab.tcc import demo_streamlit

        return demo_streamlit.main(rest)
    try:
        return {
            "stop": _stop,
            "export": _export,
            "campaign": _campaign,
            "fila": _fila,
            "piloto": _piloto,
            "final": _final,
        }.get(command, _verify)(rest)
    except (StorageError, OSError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return demo.EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
