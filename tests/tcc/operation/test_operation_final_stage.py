"""The final phase (task A12): retrain the finalists, freeze the artifacts, open the official test once per engine. CPU only.

The campaign is the one of the finalists tests (scripted validation measurements, synthetic evaluator); its list is frozen
straight in the store, because ``finalists.freeze_list`` refuses synthetic results by design. The final evaluator and the
test evaluator are the synthetic ones of ``operation.testing``: nothing here trains, builds or reads a test image.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # tests/tcc/operation has no package: borrow the campaign builder
from test_operation_finalists import CODE, build  # noqa: E402

from cnnlab.tcc.operation import final_stage as fs
from cnnlab.tcc.operation import finalists, inference
from cnnlab.tcc.operation.demo import CodeVersionMismatchError, default_repo_root
from cnnlab.tcc.operation.testing import SyntheticFinalEvaluator, SyntheticTestEvaluator, synthetic_final_wiring
from cnnlab.tcc.storage import JobQueue, ReadModel, SQLiteRunStore
from cnnlab.tcc.storage.errors import OfficialTestAccessError
from cnnlab.tcc.storage.sqlite_store import FinalistSelection

REPO = default_repo_root()
FACTORY = "cnnlab.tcc.operation.testing:synthetic_final_wiring"
LIST_ID = "lista-final"


@dataclasses.dataclass
class World:
    db: Path
    art: Path
    plan_id: str
    list_sha: str
    proposal: finalists.Proposal
    final: fs.FinalWiring


def make_world(tmp_path: Path, **evaluators: Any) -> World:
    db, plan_id = build(tmp_path)
    art = db.parent / "artifacts"
    proposal = finalists.propose(db, plan_id, allow_synthetic=True)
    with SQLiteRunStore(db, art) as store:
        record = store.get_plan(plan_id)
        selections = [
            FinalistSelection(f"sel-{c.run_id}-{c.role}", c.run_id, c.candidate_id, c.role, c.criterion)
            for c in proposal.choices
        ]
        sha = store.freeze_finalist_list(
            LIST_ID, protocol_id=record.protocol_id, plan_id=plan_id, selections=selections, author="test"
        )
    final = dataclasses.replace(synthetic_final_wiring(None), **evaluators)  # type: ignore[arg-type]
    return World(db, art, plan_id, sha, proposal, final)  # type: ignore[arg-type]


def prepare(w: World, **kw: Any) -> list[fs.PreparedFinal]:
    with SQLiteRunStore(w.db, w.art) as store:
        return fs.prepare_builds(
            store, LIST_ID, w.list_sha, w.final, CODE, factory=FACTORY, repo_root=REPO, allow_synthetic=True, **kw
        )


def run(w: World, mode: str, **kw: Any) -> list[Any]:
    return fs.run_pending(
        w.db,
        w.art,
        LIST_ID,
        w.list_sha,
        mode,
        w.final,
        CODE,
        worker_id="w",
        lease_s=60.0,
        allow_synthetic=True,
        log=lambda _: None,
        **kw,
    )


def build_all(w: World) -> None:
    prepare(w)
    results = run(w, fs.BUILD)
    assert results and all(r.outcome is not None and r.outcome.end_state == "done" for r in results), [
        r.message for r in results
    ]


def freeze_artifacts(w: World) -> str:
    return fs.artifacts_manifest(w.db, w.art, LIST_ID, w.list_sha)["manifest_sha256"]


def open_all(w: World) -> None:
    with SQLiteRunStore(w.db, w.art) as store:
        fs.enqueue_tests(store, LIST_ID, w.list_sha, freeze_artifacts(w), w.art, w.final, CODE, REPO)
    results = run(w, fs.TEST)
    assert all(r.outcome is not None and r.outcome.end_state == "done" for r in results), [r.message for r in results]


# -- the units ------------------------------------------------------------------------------------------------------------


def test_one_retraining_per_genome_of_a_block_and_every_selection_that_chose_it(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    with ReadModel(w.db) as rm:
        units = fs.units_of(rm.finalist_list(LIST_ID))
    assert len(units) == len(w.proposal.retrains) == 9 and sum(len(u.selections) for u in units) == 15
    shared = [u for u in units if len(u.selections) > 1]
    assert shared and all(len({s["phenotype_hash"] for s in u.selections}) == 1 for u in units)
    assert all(
        u.retrain_id == f"rt-b{u.seed_block}-{u.phenotype_hash[:12]}" for u in units
    )  # the name finalists gave them
    assert {u.retrain_id for u in units} == {r.retrain_id for r in w.proposal.retrains}


def test_more_than_72_retrainings_are_refused() -> None:
    listing = {
        "selections": [
            {
                "seed_block": i // 9,
                "phenotype_hash": f"{i:064x}",
                "selection_id": f"s{i}",
                "role": "menor_E",
                "method": "joint",
                "source_run_id": "r",
                "candidate_id": "c",
            }
            for i in range(73)
        ]
    }
    with pytest.raises(fs.FinalStageError, match="at most 72"):
        fs.units_of(listing)


# -- preparing and building ---------------------------------------------------------------------------------------------


def test_prepare_creates_one_finalist_run_per_unit_with_the_blocks_retrain_seed_and_the_frozen_list(
    tmp_path: Path,
) -> None:
    w = make_world(tmp_path)
    prepared = prepare(w)
    assert len(prepared) == 9 and all(p.created for p in prepared)
    with ReadModel(w.db) as rm:
        plan = rm.plan(w.plan_id)["content"]
        runs = {r["run_id"]: r for r in rm.runs() if r["stage"] == "finalist"}
    assert len(runs) == 9
    for p in prepared:
        run_ = runs[p.run_id]
        block = plan["blocks"][run_["seed_block"]]
        assert run_["finalist_list_id"] == LIST_ID and run_["train_seed"] == block["retrain_seed"]
        assert (run_["search_seed"], run_["calibration_seed"]) == (block["search_seed"], block["calibration_seed"])
        assert run_["plan_id"] == w.plan_id
    again = prepare(w)  # safe to repeat
    assert [p.created for p in again] == [False] * 9
    queue = JobQueue(SQLiteRunStore(w.db, w.art))
    job = queue.get(prepared[0].job_id)
    assert job.options["final_mode"] == "build" and job.options["list_sha256"] == w.list_sha
    assert job.options["final_spec"]["factory_args"]["stage"] == "final"  # the plan's arguments plus the stage


def test_the_builds_copy_the_genome_and_commit_a_valid_engine_per_retraining(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    build_all(w)
    with ReadModel(w.db) as rm:
        for u in fs.units_of(rm.finalist_list(LIST_ID)):
            [valid] = [e for e in rm.evaluations(u.run_id) if e["status"] == "valid"]
            assert valid["phenotype_hash"] == u.phenotype_hash and valid["candidate_id"] == f"{u.run_id}.g"
            [engine] = [a for a in rm.artifacts(u.run_id) if a["kind"] == "engine"]
            assert engine["attempt_id"] == valid["attempt_id"]
            source = rm.genome_record(u.source["source_run_id"], u.source["candidate_id"])
            mine = rm.genome_record(u.run_id, f"{u.run_id}.g")
            assert (
                mine["expressed_genome"] == source["expressed_genome"]
                and mine["phenotype_hash"] == source["phenotype_hash"]
            )
            assert (
                mine["cache_context_hash"] != source["cache_context_hash"]
            )  # a 30-epoch retraining is another evaluation
        assert rm.test_evaluations() == []  # the builds never touch the test
    # a repeat has nothing left to do
    assert run(w, fs.BUILD) == []


def test_a_plan_on_other_code_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    other = dataclasses.replace(CODE, git_sha="d" * 40)
    with SQLiteRunStore(w.db, w.art) as store:
        with pytest.raises(CodeVersionMismatchError, match="froze"):
            fs.prepare_builds(
                store, LIST_ID, w.list_sha, w.final, other, factory=FACTORY, repo_root=REPO, allow_synthetic=True
            )
    with ReadModel(w.db) as rm:
        assert [r for r in rm.runs() if r["stage"] == "finalist"] == []


def test_a_list_hash_that_is_not_the_frozen_one_stops_everything(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    with SQLiteRunStore(w.db, w.art) as store:
        with pytest.raises(fs.FinalStageError, match="not the list that was frozen"):
            fs.prepare_builds(
                store, LIST_ID, "0" * 64, w.final, CODE, factory=FACTORY, repo_root=REPO, allow_synthetic=True
            )
    with pytest.raises(fs.FinalStageError):
        fs.artifacts_manifest(w.db, w.art, LIST_ID, "0" * 64)
    with pytest.raises(Exception, match="does not exist"):
        with ReadModel(w.db) as rm:
            fs.verify_list(rm, "outra-lista", w.list_sha)


def test_synthetic_results_need_the_opt_in(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    with SQLiteRunStore(w.db, w.art) as store:
        with pytest.raises(Exception, match="synthetic"):
            fs.prepare_builds(store, LIST_ID, w.list_sha, w.final, CODE, factory=FACTORY, repo_root=REPO)


def test_a_technical_failure_stops_the_job_and_a_justified_repetition_makes_another_attempt(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    victim = f"final-{w.proposal.retrains[0].retrain_id}"
    w = dataclasses.replace(
        w, final=dataclasses.replace(w.final, evaluator=SyntheticFinalEvaluator(fail_runs=[victim]))
    )
    prepare(w)
    results = run(w, fs.BUILD)
    assert (
        results[-1].outcome is not None
        and results[-1].outcome.end_state == "failed"
        and "ended invalid" in results[-1].message
    )
    with SQLiteRunStore(w.db, w.art) as store:
        with pytest.raises(fs.FinalStageError, match="reason"):
            fs.requeue_failed(store, LIST_ID, w.list_sha, fs.BUILD, reason=" ")
        [again] = fs.requeue_failed(
            store, LIST_ID, w.list_sha, fs.BUILD, reason="GPU driver reset during the build", max_attempts=2
        )
        assert again.endswith("-r2") and JobQueue(store).get(again).options["requeue_reason"].startswith("GPU driver")
    results = run(w, fs.BUILD)
    assert all(r.outcome is not None and r.outcome.end_state == "done" for r in results)
    with ReadModel(w.db) as rm:
        statuses = [e["status"] for e in rm.evaluations(victim)]
    assert statuses == ["invalid", "valid"]  # the failure stays on record next to the repetition


def test_without_the_justification_of_a_second_attempt_the_limit_holds(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    victim = f"final-{w.proposal.retrains[0].retrain_id}"
    w = dataclasses.replace(
        w, final=dataclasses.replace(w.final, evaluator=SyntheticFinalEvaluator(fail_runs=[victim]))
    )
    prepare(w)
    run(w, fs.BUILD)
    with SQLiteRunStore(w.db, w.art) as store:
        fs.requeue_failed(store, LIST_ID, w.list_sha, fs.BUILD, reason="retry without raising the limit")
    results = run(w, fs.BUILD)
    assert results[0].outcome is not None and results[0].outcome.end_state == "failed" and "limit" in results[0].message


# -- the barrier of the official test ----------------------------------------------------------------------------------------


def test_the_test_stays_closed_until_every_engine_exists_and_the_operator_confirmed_their_hash(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    prepare(w)
    with pytest.raises(fs.FinalStageError, match="not ready to be frozen"):
        freeze_artifacts(w)  # nothing built yet
    build_all(w)
    manifest = fs.artifacts_manifest(w.db, w.art, LIST_ID, w.list_sha)
    assert len(manifest["engines"]) == 9 and manifest["list_sha256"] == w.list_sha
    with SQLiteRunStore(w.db, w.art) as store:
        with pytest.raises(fs.FinalStageError, match="changed since they were frozen"):
            fs.enqueue_tests(store, LIST_ID, w.list_sha, "1" * 64, w.art, w.final, CODE, REPO)
    with ReadModel(w.db) as rm:
        assert rm.test_evaluations() == []


def test_an_engine_rewritten_after_the_freeze_closes_the_test(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    build_all(w)
    digest = freeze_artifacts(w)
    with ReadModel(w.db) as rm:
        unit = fs.units_of(rm.finalist_list(LIST_ID))[0]
        [engine] = [a for a in rm.artifacts(unit.run_id) if a["kind"] == "engine"]
    (w.art / engine["rel_path"]).write_bytes(b"not the engine that was frozen")
    with pytest.raises(fs.FinalStageError, match="engine on disk has SHA-256"):
        fs.artifacts_manifest(w.db, w.art, LIST_ID, w.list_sha)
    with SQLiteRunStore(w.db, w.art) as store:
        with pytest.raises(fs.FinalStageError, match="engine on disk"):
            fs.enqueue_tests(store, LIST_ID, w.list_sha, digest, w.art, w.final, CODE, REPO)


def test_every_engine_is_opened_once_and_every_selection_reads_its_units_result(tmp_path: Path) -> None:
    tester = SyntheticTestEvaluator()
    w = make_world(tmp_path, test_evaluator=tester)
    build_all(w)
    open_all(w)
    assert len(tester.calls) == 9 and len(set(tester.calls)) == 9  # once per unique engine
    with ReadModel(w.db) as rm:
        tests = rm.test_evaluations()
    assert len(tests) == 9 and len({t["engine_sha256"] for t in tests}) == 9
    shared = next(t for t in tests if len(t["metrics"]["shared_by"]) > 1)
    assert shared["selection_id"] == shared["metrics"]["shared_by"][0]
    # nothing is opened again: not by running, not by queueing, not by asking the store
    assert run(w, fs.TEST) == []
    with SQLiteRunStore(w.db, w.art) as store:
        assert fs.enqueue_tests(store, LIST_ID, w.list_sha, freeze_artifacts(w), w.art, w.final, CODE, REPO) == []
        with pytest.raises(OfficialTestAccessError, match="already opened"):
            store.record_test_evaluation(
                "again",
                run_id=tests[0]["run_id"],
                selection_id=tests[0]["selection_id"],
                attempt_id=tests[0]["attempt_id"],
                engine_sha256=tests[0]["engine_sha256"],
                metrics={"top1_accuracy": 0.5},
            )
    assert len(tester.calls) == 9


def test_a_second_job_for_an_opened_engine_does_not_open_it_again(tmp_path: Path) -> None:
    tester = SyntheticTestEvaluator()
    w = make_world(tmp_path, test_evaluator=tester)
    build_all(w)
    open_all(w)
    with ReadModel(w.db) as rm:
        unit = fs.units_of(rm.finalist_list(LIST_ID))[0]
    with SQLiteRunStore(w.db, w.art) as store:
        queue = JobQueue(store)
        options = dict(queue.get(unit.job_id(fs.TEST)).options)
        queue.enqueue(unit.job_id(fs.TEST, 2), unit.run_id, options=options)
    results = run(w, fs.TEST)
    assert [r.outcome.end_state for r in results if r.outcome] == ["done"] and len(tester.calls) == 9


def test_a_mark_without_a_result_is_not_repeated(tmp_path: Path) -> None:
    tester = SyntheticTestEvaluator()
    w = make_world(tmp_path, test_evaluator=tester)
    build_all(w)
    manifest = fs.artifacts_manifest(w.db, w.art, LIST_ID, w.list_sha)
    first = manifest["engines"][0]
    mark = w.db.parent / fs.MARKS_DIR / f"{first['engine_sha256']}.json"
    mark.parent.mkdir(parents=True)
    mark.write_text("{}", encoding="utf-8")  # the process died after opening the test and before recording it
    with SQLiteRunStore(w.db, w.art) as store:
        fs.enqueue_tests(store, LIST_ID, w.list_sha, manifest["manifest_sha256"], w.art, w.final, CODE, REPO)
    results = run(w, fs.TEST)
    assert (
        results[0].outcome is not None
        and results[0].outcome.end_state == "failed"
        and "not repeated" in results[0].message
    )
    assert first["run_id"] not in tester.calls
    with ReadModel(w.db) as rm:
        assert first["run_id"] not in {t["run_id"] for t in rm.test_evaluations()}


def test_an_answer_about_another_engine_is_not_recorded(tmp_path: Path) -> None:
    w = make_world(tmp_path, test_evaluator=SyntheticTestEvaluator(wrong_hash=True))
    build_all(w)
    with SQLiteRunStore(w.db, w.art) as store:
        fs.enqueue_tests(store, LIST_ID, w.list_sha, freeze_artifacts(w), w.art, w.final, CODE, REPO)
    results = run(w, fs.TEST)
    assert (
        results[0].outcome is not None
        and results[0].outcome.end_state == "failed"
        and "another engine file" in results[0].message
    )
    with ReadModel(w.db) as rm:
        assert rm.test_evaluations() == []


def test_an_accuracy_that_is_not_a_fraction_is_not_recorded(tmp_path: Path) -> None:
    w = make_world(
        tmp_path, test_evaluator=SyntheticTestEvaluator(accuracy=lambda run_id: 63.2)
    )  # percent, not a fraction
    build_all(w)
    with SQLiteRunStore(w.db, w.art) as store:
        fs.enqueue_tests(store, LIST_ID, w.list_sha, freeze_artifacts(w), w.art, w.final, CODE, REPO)
    results = run(w, fs.TEST)
    assert results[0].outcome is not None and "fraction between 0 and 1" in results[0].message
    with ReadModel(w.db) as rm:
        assert rm.test_evaluations() == []


def test_a_search_run_never_receives_the_official_test(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    prepare(w)
    with SQLiteRunStore(w.db, w.art) as store:
        assert "test" not in store.dataset_splits_for_run(w.proposal.choices[0].run_id)  # a confirmatory run
        assert "test" in store.dataset_splits_for_run(f"final-{w.proposal.retrains[0].retrain_id}")
        with pytest.raises(OfficialTestAccessError):
            store.official_test_split(w.proposal.choices[0].run_id)


# -- the export -------------------------------------------------------------------------------------------------------------


def test_the_export_is_the_file_the_inference_reads(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    build_all(w)
    with pytest.raises(fs.FinalStageError, match="no official-test result yet"):
        fs.export_final(w.db, LIST_ID, w.list_sha, tmp_path / "x", allow_synthetic=True)
    open_all(w)
    with pytest.raises(fs.FinalStageError, match="synthetic"):
        fs.export_final(w.db, LIST_ID, w.list_sha, tmp_path / "x")
    dest = tmp_path / "final"
    manifest = fs.export_final(w.db, LIST_ID, w.list_sha, dest, allow_synthetic=True)
    assert (
        manifest["rows"] == 15
        and manifest["tested"] == 9
        and manifest["synthetic"]
        and manifest["protocol_stage"] == "final"
    )
    assert manifest["bench_version"] == SyntheticFinalEvaluator.BENCH and manifest["list_sha256"] == w.list_sha
    with open(dest / "resultados_finais.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 15 and "test_accuracy_SINTETICO" in rows[0] and "test_accuracy" not in rows[0]
    assert all(r["protocol_stage"] == "final" and r["bench_version"] == SyntheticFinalEvaluator.BENCH for r in rows)
    twins = [r for r in rows if r["shared_with"]]
    assert twins and all(r["engine_sha256"] and r["test_id"] for r in twins)
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run_id"], set()).add((r["test_accuracy_SINTETICO"], r["l_ms_SINTETICO"]))
    assert all(len(v) == 1 for v in by_run.values())  # the selections of one unit read one result
    assert json.loads(rows[0]["builds_json_SINTETICO"]) and rows[0]["n_builds"] == "3"
    with pytest.raises(fs.FinalStageError, match="already exists"):
        fs.export_final(w.db, LIST_ID, w.list_sha, dest, allow_synthetic=True)
    # the inference reads it: the H1 table has the seed block with both engines
    out = tmp_path / "inferencia"
    with pytest.raises(Exception, match="synthetic"):
        inference.write_analysis(out, final_results=dest / "resultados_finais.csv")
    result = inference.write_analysis(out, final_results=dest / "resultados_finais.csv", allow_synthetic=True)
    assert result["synthetic"] is True and result["summary"]["h1"]["with_both"] >= 1


def test_the_export_can_be_partial_only_when_asked_and_says_what_is_missing(tmp_path: Path) -> None:
    class FailsOnTheSecond(SyntheticTestEvaluator):
        def evaluate_test(self, request: Any) -> dict[str, Any]:
            if len(self.calls) == 1:
                self.calls.append(request.run_id)
                return {
                    "top1_accuracy": 0.5,
                    "n_examples": 10_000,
                    "engine_sha256": "0" * 64,
                }  # an engine that is not it
            return super().evaluate_test(request)

    w = make_world(tmp_path, test_evaluator=FailsOnTheSecond())
    build_all(w)
    with SQLiteRunStore(w.db, w.art) as store:
        fs.enqueue_tests(store, LIST_ID, w.list_sha, freeze_artifacts(w), w.art, w.final, CODE, REPO)
    results = run(w, fs.TEST)
    assert [r.outcome.end_state for r in results if r.outcome] == ["done", "failed"]  # it stops at the first failure
    with pytest.raises(fs.FinalStageError, match="8 retraining"):
        fs.export_final(w.db, LIST_ID, w.list_sha, tmp_path / "x", allow_synthetic=True)
    manifest = fs.export_final(
        w.db, LIST_ID, w.list_sha, tmp_path / "parcial", allow_synthetic=True, allow_incomplete=True
    )
    assert manifest["tested"] == 1 and len(manifest["missing_test"]) == 8 and 1 <= manifest["rows"] < 15


def test_the_status_shows_where_each_unit_stands(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    prepare(w)
    rows = fs.status(w.db, LIST_ID)
    assert (
        len(rows) == 9 and not any(r["built"] or r["test_opened"] for r in rows) and rows[0]["build_jobs"] == ["queued"]
    )
    build_all(w)
    assert all(r["built"] and not r["test_opened"] for r in fs.status(w.db, LIST_ID))


# -- the ledger and the freeze -----------------------------------------------------------------------------------------------


def test_the_ledger_claims_an_engine_once_and_gives_a_dead_measurement_back(tmp_path: Path) -> None:
    opened: set[str] = set()
    ledger = fs.MarkLedger(tmp_path / "marks", "run", "l" * 64, lambda: opened)
    ledger.claim("a" * 64)
    with pytest.raises(fs.TestAlreadyOpened, match="not repeated"):
        ledger.claim("a" * 64)  # a mark without a result: not repeated
    other = fs.MarkLedger(tmp_path / "marks", "run", "l" * 64, lambda: opened)
    with pytest.raises(fs.TestAlreadyOpened):
        other.claim("a" * 64)  # a fresh process sees the mark too
    ledger.release("a" * 64)  # the measurement died with no result: the claim goes back
    other.claim("a" * 64)
    opened.add("b" * 64)
    with pytest.raises(fs.TestAlreadyOpened, match="already opened"):
        other.claim("b" * 64)  # the database says so
    other.release("c" * 64)  # releasing what was never claimed is harmless


def test_a_test_evaluator_that_does_not_claim_the_engine_is_not_recorded(tmp_path: Path) -> None:
    w = make_world(tmp_path, test_evaluator=SyntheticTestEvaluator(claims=False))
    build_all(w)
    with SQLiteRunStore(w.db, w.art) as store:
        fs.enqueue_tests(store, LIST_ID, w.list_sha, freeze_artifacts(w), w.art, w.final, CODE, REPO)
    results = run(w, fs.TEST)
    assert results[0].outcome is not None and "without claiming the engine" in results[0].message
    with ReadModel(w.db) as rm:
        assert rm.test_evaluations() == []


def test_the_test_pass_receives_the_manifest_the_build_stored(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []

    class Spy(SyntheticTestEvaluator):
        def evaluate_test(self, request: Any) -> dict[str, Any]:
            seen.append(dict(request.build_manifest))
            return super().evaluate_test(request)

    w = make_world(tmp_path, test_evaluator=Spy())
    build_all(w)
    open_all(w)
    assert len(seen) == 9 and all(m["engine_sha256"] for m in seen)


def test_after_the_artifacts_are_frozen_no_engine_changes(tmp_path: Path) -> None:
    w = make_world(tmp_path)
    victim = f"final-{w.proposal.retrains[0].retrain_id}"
    w = dataclasses.replace(
        w, final=dataclasses.replace(w.final, evaluator=SyntheticFinalEvaluator(fail_runs=[victim]))
    )
    prepare(w)
    run(w, fs.BUILD)  # the victim failed; the others are built
    with SQLiteRunStore(w.db, w.art) as store:
        fs.requeue_failed(
            store, LIST_ID, w.list_sha, fs.BUILD, reason="driver reset", max_attempts=2
        )  # before the freeze: fine
    run(w, fs.BUILD)
    open_all(w)  # the freeze: test jobs exist
    with SQLiteRunStore(w.db, w.art) as store:
        with pytest.raises(fs.FinalStageError, match="artifacts were frozen"):
            fs.requeue_failed(store, LIST_ID, w.list_sha, fs.BUILD, reason="too late")
        with pytest.raises(fs.FinalStageError, match="artifacts were frozen"):
            fs.prepare_builds(
                store, LIST_ID, w.list_sha, w.final, CODE, factory=FACTORY, repo_root=REPO, allow_synthetic=True
            )


# -- the command line ------------------------------------------------------------------------------------------------------------


def test_the_final_command_runs_the_whole_phase_and_the_export_feeds_the_inference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from cnnlab.tcc.operation import cli
    from cnnlab.tcc.operation.demo import EXIT_REFUSED

    w = make_world(tmp_path)
    monkeypatch.setattr(fs, "capture_code_version", lambda label, repo: CODE)  # the plan froze a fake commit
    base = ["final"]
    common = ["--db", str(w.db), "--list", LIST_ID, "--list-sha256", w.list_sha]
    writer = [*common, "--artifacts", str(w.art)]
    factory = ["--factory", FACTORY, "--allow-synthetic"]
    assert cli.main([*base, "prepare", *writer, *factory]) == 0
    assert "9 retraining(s)" in capsys.readouterr().out
    assert cli.main([*base, "run", *writer, *factory, "--mode", "build"]) == 0
    assert "9 job(s) done" in capsys.readouterr().out
    # the test cannot be queued before the engines are frozen and their hash confirmed
    assert cli.main([*base, "enqueue-tests", *writer, *factory, "--artifacts-sha256", "1" * 64]) == EXIT_REFUSED
    assert "changed since they were frozen" in capsys.readouterr().err
    assert cli.main([*base, "artifacts", *writer, "--dest", str(tmp_path / "artefatos")]) == 0
    printed = capsys.readouterr().out
    manifest_hash = next(
        line.split("sha256=")[1] for line in printed.splitlines() if line.startswith("artifacts manifest")
    )
    assert (
        json.loads((tmp_path / "artefatos" / fs.ARTIFACTS_MANIFEST_NAME).read_text(encoding="utf-8"))["manifest_sha256"]
        == manifest_hash
    )
    assert cli.main([*base, "enqueue-tests", *writer, *factory, "--artifacts-sha256", manifest_hash]) == 0
    assert "9 test job(s) queued" in capsys.readouterr().out
    assert cli.main([*base, "retry", *writer, "--mode", "build", "--reason", "too late"]) == EXIT_REFUSED
    assert "artifacts were frozen" in capsys.readouterr().err
    assert cli.main([*base, "run", *writer, *factory, "--mode", "test"]) == 0
    assert cli.main([*base, "status", *common]) == 0
    assert "opened" in capsys.readouterr().out
    assert (
        cli.main([*base, "export", *common, "--dest", str(tmp_path / "final")]) == EXIT_REFUSED
    )  # synthetic: needs the opt-in
    capsys.readouterr()
    assert cli.main([*base, "export", *common, "--dest", str(tmp_path / "final"), "--allow-synthetic"]) == 0
    out = capsys.readouterr().out
    assert "9 of 9 retraining(s) tested" in out and "SYNTHETIC" in out
    assert (
        cli.main(
            [
                "campaign",
                "inference",
                "--final-results",
                str(tmp_path / "final" / "resultados_finais.csv"),
                "--dest",
                str(tmp_path / "inferencia"),
                "--allow-synthetic",
            ]
        )
        == 0
    )
    assert "H1:" in capsys.readouterr().out


def test_a_repetition_under_another_training_context_gets_its_own_candidate_and_the_test_still_opens(
    tmp_path: Path,
) -> None:
    # Amendment of 2026-10-24: the retraining limit (and the code) changed between the failed build and its repetition,
    # so the cache context differs; the first candidate stays as recorded and the repetition is a candidate of its own.
    w = make_world(tmp_path)
    victim = f"final-{w.proposal.retrains[0].retrain_id}"
    w = dataclasses.replace(
        w, final=dataclasses.replace(w.final, evaluator=SyntheticFinalEvaluator(fail_runs=[victim]))
    )
    prepare(w)
    run(w, fs.BUILD)
    with SQLiteRunStore(w.db, w.art) as store:
        fs.requeue_failed(store, LIST_ID, w.list_sha, fs.BUILD, reason="limit scaled by the epochs", max_attempts=2)
    original = w.final.context_for
    amended = dataclasses.replace(
        w.final,
        evaluator=SyntheticFinalEvaluator(),
        context_for=lambda job: dataclasses.replace(original(job), code_version="amended"),
    )
    w = dataclasses.replace(w, final=amended)
    results = run(w, fs.BUILD)
    assert all(r.outcome is not None and r.outcome.end_state == "done" for r in results), [r.message for r in results]
    with ReadModel(w.db) as rm:
        rows = rm.evaluations(victim)
    assert [(r["candidate_id"].rsplit(".", 1)[-1], r["status"]) for r in rows] == [("g", "invalid"), ("g2", "valid")]
    open_all(w)  # the valid attempt of the new candidate is the unit's engine; the test opens once
