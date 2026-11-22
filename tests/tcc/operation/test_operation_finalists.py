"""Choice of the finalists (TCC-01 section 6; task A10a). CPU only.

The database holds runs of a frozen campaign plan with **scripted** validation measurements (a candidate's values are
written in the test, so the choices can be worked out by hand). The evaluator is the synthetic one, so every number is
labeled synthetic and leaves in ``_SINTETICO`` columns; nothing here is measured.
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.contracts import EvaluationResult, Fitness, TCCJobSpec
from cnnlab.tcc.operation import campaign, cli, finalists
from cnnlab.tcc.operation.campaign import CampaignSettings
from cnnlab.tcc.operation.demo import EXIT_REFUSED, default_repo_root
from cnnlab.tcc.operation.provenance import CodeVersion
from cnnlab.tcc.operation.testing import synthetic_demo_wiring
from cnnlab.tcc.storage import JobQueue, ReadModel, SQLiteRunStore
from cnnlab.tcc.storage.errors import NotFoundError, OfficialTestAccessError, StorageError
from cnnlab.tcc.storage.testing import FakeEvaluator, make_genome, run_candidate

FACTORY = "cnnlab.tcc.operation.testing:synthetic_demo_wiring"
CODE = CodeVersion("campanha", "c" * 40, False, "7" * 64, (), (), 0)
REFERENCE = (1.0, 2.0, 4.0, 4.0)
MINIMA = (0.1, 0.1, 0.5, 0.5)
STAMP_SPANS = (0.9, 1.9, 3.5, 3.5)  # r - m

# (method, seed block, genome index) -> validation (E, L, V, S). Genome indexes are the identity of the expressed genome:
# the same index in two runs is the same genome.
RUNS: dict[tuple[str, int], dict[int, tuple[float, float, float, float]]] = {
    ("mono", 0): {10: (0.30, 1.0, 2.0, 2.0), 11: (0.20, 1.5, 3.0, 3.0)},
    ("joint", 0): {
        100: (0.28, 0.5, 1.0, 1.0),  # the balance: close to the minima in every objective
        2: (0.205, 1.2, 2.5, 2.5),  # lowest error; eligible for H1 (0.205 <= 0.20 + 0.01) but large
        3: (0.209, 0.9, 1.5, 1.5),  # eligible and smaller: the H1 candidate
        4: (0.215, 1.0, 2.0, 2.0),  # dominated by 3, and above E_mono + 0.01 anyway
    },
    ("random", 0): {100: (0.28, 0.5, 1.0, 1.0)},  # the same genome as the joint balance: one retraining for both
    ("sequential", 0): {200: (0.35, 0.8, 1.2, 1.2)},
    ("mono", 1): {21: (0.20, 1.5, 3.0, 3.0)},
    ("joint", 1): {20: (0.50, 0.5, 1.0, 1.0)},  # nothing within a point of the mono champion: no H1
    ("joint", 2): {30: (0.25, 1.0, 2.0, 2.0)},  # and no mono run in this block at all
}


class Scripted(FakeEvaluator):
    """The synthetic evaluator, with the objectives of each candidate written down."""

    values: dict[str, tuple[float, float, float, float]] = {}
    bench: str | None = None  # the benchmark version the attempts claim, when a test needs one

    def evaluate(self, request: Any) -> EvaluationResult:
        result = super().evaluate(request)
        e, l_ms, v_mb, s_mb = self.values[request.candidate_id]
        quality = {**result.quality, **({"measurement": {"bench_version": self.bench}} if self.bench else {})}
        return dataclasses.replace(result, fitness=Fitness(e=e, l_ms=l_ms, v_mb=v_mb, s_mb=s_mb), quality=quality)


def build(
    tmp_path: Path,
    *,
    runs: dict[tuple[str, int], dict[int, tuple[float, ...]]] | None = None,
    unfinished: tuple[str, int] | None = None,
    with_minima: bool = True,
    name: str = "camp",
    bench: dict[str, str] | None = None,
) -> tuple[Path, str]:
    settings = CampaignSettings.from_mapping(
        {
            "master_seed": 20261026,
            "factory": FACTORY,
            "factory_args": {"slice": "fp16_regions", "epochs": 5, "n_candidates": 3, "sleep_s": 0.0},
            "reference_point": REFERENCE,
            "viable_minima": MINIMA if with_minima else None,
            "max_attempt_train_s": 1800.0,
        }
    )
    db, art = tmp_path / name / "tcc.sqlite3", tmp_path / name / "artifacts"
    wiring = synthetic_demo_wiring(campaign.representative_spec(settings))
    with SQLiteRunStore(db, art) as store:
        plan = campaign.create_plan(
            store, settings, wiring, CODE, default_repo_root(), freeze=True, allow_pending=not with_minima
        )
        store.register_environment("env-test", {"gpu": "synthetic"})
        store.register_dataset(wiring.dataset.dataset_id, wiring.dataset.manifest, wiring.dataset.splits)
        queue = JobQueue(store)
        for (method, block), candidates in (runs or RUNS).items():
            seeds = plan.content["blocks"][block]
            job = TCCJobSpec(
                run_id=f"camp-{method}-b{block}",
                protocol_id=plan.content["protocol_id"],
                plan_id=plan.plan_id,
                dataset_id=wiring.dataset.dataset_id,
                environment_id="env-test",
                space_hash=wiring.space_hash,
                method=method,
                stage="confirmatory",
                seed_block=block,
                search_seed=seeds["search_seed"],  # type: ignore[arg-type]
                train_seed=seeds["train_seed"],
                calibration_seed=seeds["calibration_seed"],
                budget_gpu_s=19_800.0,
            )
            store.create_run(job)
            evaluator = Scripted(gpu_seconds=1.0)
            evaluator.bench = None if bench is None else bench[method]
            evaluator.values = {f"{job.run_id}.g{i}": v for i, v in candidates.items()}  # type: ignore[assignment]
            for index in candidates:
                run_candidate(
                    store,
                    job,
                    f"{job.run_id}.g{index}",
                    f"{job.run_id}.g{index}.a1",
                    make_genome(index, space_hash=wiring.space_hash),
                    evaluator,
                )
            queue.enqueue(f"job-{job.run_id}", job.run_id)
            if (method, block) == unfinished:
                continue  # left queued: its archive is not final
            lease = queue.claim_next("w", lease_s=60.0, job_id=f"job-{job.run_id}")
            assert lease is not None
            queue.finish(lease, "done", detail="ok")
    return db, plan.plan_id


@pytest.fixture
def campaign_db(tmp_path: Path) -> tuple[Path, str]:
    return build(tmp_path)


def pick(proposal: finalists.Proposal, method: str, block: int, role: str) -> finalists.Choice:
    [c] = [c for c in proposal.choices if (c.method, c.seed_block, c.role) == (method, block, role)]
    return c


def distance(values: tuple[float, ...]) -> float:
    return math.sqrt(sum(((f - m) / s) ** 2 for f, m, s in zip(values, MINIMA, STAMP_SPANS)))


def test_the_roles_follow_the_rules_of_section_6_worked_out_by_hand(campaign_db: tuple[Path, str]) -> None:
    db, plan_id = campaign_db
    p = finalists.propose(db, plan_id, allow_synthetic=True)
    assert p.reference_point == REFERENCE and p.viable_minima == MINIMA and p.synthetic and p.unfinished == ()

    # lowest error: the lowest of the non-dominated, not of everything
    assert pick(p, "joint", 0, "menor_E").candidate_id == "camp-joint-b0.g2"
    assert pick(p, "mono", 0, "menor_E").candidate_id == "camp-mono-b0.g11"
    # balance: the smallest distance of (f - m) / (r - m) to the origin; here the candidate near the minima everywhere
    balance = pick(p, "joint", 0, "equilibrio")
    assert balance.candidate_id == "camp-joint-b0.g100" and balance.score == pytest.approx(
        distance((0.28, 0.5, 1.0, 1.0))
    )
    assert balance.score == pytest.approx(math.sqrt(0.2**2 + (0.4 / 1.9) ** 2 + 2 * (0.5 / 3.5) ** 2))  # written out
    assert (
        pick(p, "mono", 0, "equilibrio").candidate_id == "camp-mono-b0.g10"
    )  # not the lowest error: that is another role
    # H1: among E <= E_mono + 0.01 = 0.21 (g2 and g3), the smallest L/r_L + V/r_V + S/r_S: 1.2 for g3 against 1.85 for g2
    h1 = pick(p, "joint", 0, "H1")
    assert h1.candidate_id == "camp-joint-b0.g3" and h1.score == pytest.approx(0.9 / 2 + 1.5 / 4 + 1.5 / 4)
    assert "E_mono = 0.2" in h1.criterion and "E <= E_mono + 0.01" in h1.criterion
    # a dominated candidate is never a finalist, whatever its score
    assert not any(c.candidate_id.endswith(".g4") for c in p.choices)
    # one H1 only, and only for a joint run
    assert [c.method for c in p.choices if c.role == "H1"] == ["joint"]


def test_a_seed_without_an_eligible_candidate_or_without_a_mono_run_has_no_h1_and_says_why(
    campaign_db: tuple[Path, str],
) -> None:
    p = finalists.propose(*campaign_db, allow_synthetic=True)
    failures = {f.seed_block: f for f in p.h1_failures}
    assert sorted(failures) == [1, 2] and not any(c.role == "H1" and c.seed_block in (1, 2) for c in p.choices)
    assert "no non-dominated candidate with E <= E_mono + 0.01" in failures[1].reason
    assert (failures[1].e_mono, failures[1].best_joint_e) == (0.20, 0.50)
    assert failures[2].reason == "no mono run of this seed block to compare with"
    # the two roles of those runs are still chosen: the failure of H1 takes nothing else away
    assert pick(p, "joint", 1, "menor_E").candidate_id == "camp-joint-b1.g20"


def test_the_margin_is_inclusive_and_ties_go_to_the_lowest_error_then_the_genome_hash() -> None:
    run = {"run_id": "r", "method": "joint", "seed_block": 0}

    def row(cid: str, h: str, e: float, l_ms: float = 1.0) -> dict[str, Any]:
        return {
            "candidate_id": cid,
            "phenotype_hash": h,
            "status": "valid",
            "cache_hit": False,
            "e": e,
            "l_ms": l_ms,
            "v_mb": 1.0,
            "s_mb": 1.0,
            "synthetic": False,
        }

    mono = [row("m", "z", 0.20)]
    edge = row("edge", "b", 0.21)  # exactly E_mono + 0.01: eligible
    assert finalists.choose_h1(run, [edge], mono, REFERENCE).candidate_id == "edge"  # type: ignore[union-attr]
    assert isinstance(finalists.choose_h1(run, [row("late", "b", 0.2100001)], mono, REFERENCE), finalists.H1Failure)
    # equal scores: the lowest error wins; equal error too: the lowest genome hash; then the candidate id
    a, b = row("a", "bb", 0.205), row("b", "aa", 0.205)  # both non-dominated? same vector: the hash decides
    assert finalists.choose_h1(run, [a, b], mono, REFERENCE).candidate_id == "b"  # type: ignore[union-attr]
    low_e = row("c", "zz", 0.202, l_ms=1.0)
    other = row("d", "aa", 0.204, l_ms=1.0)
    assert finalists.choose_h1(run, [other, low_e], mono, REFERENCE).candidate_id == "c"  # type: ignore[union-attr]
    # menor_E with equal errors goes by the genome hash
    rows = [row("x", "hh", 0.3), row("y", "gg", 0.3)]
    chosen, _ = finalists.choose_for_run(run, rows, MINIMA, REFERENCE)
    assert [c.candidate_id for c in chosen if c.role == "menor_E"] == ["y"]


def test_a_reference_point_under_the_minima_is_refused_not_normalized_into_nonsense() -> None:
    row = {"e": 0.3, "l_ms": 1.0, "v_mb": 1.0, "s_mb": 1.0}
    with pytest.raises(finalists.FinalistsError, match="above the viable minima"):
        finalists.normalized_distance(row, (0.1, 2.0, 0.5, 0.5), REFERENCE)  # minimum of L equals the reference of L


def test_a_genome_chosen_by_several_roles_and_methods_is_retrained_once(campaign_db: tuple[Path, str]) -> None:
    p = finalists.propose(*campaign_db, allow_synthetic=True)
    assert len(p.choices) == 9 + 4 + 2 and len(p.retrains) == 9  # 15 selections, 9 retrainings
    shared = [r for r in p.retrains if r.seed_block == 0 and len(r.roles) == 3]
    [one] = shared  # genome 100: joint equilibrio, random menor_E and random equilibrio
    assert [(m, role) for m, role, _ in one.roles] == [
        ("joint", "equilibrio"),
        ("random", "menor_E"),
        ("random", "equilibrio"),
    ]
    assert (
        one.phenotype_hash == make_genome(100).phenotype_hash and one.retrain_id == f"rt-b0-{one.phenotype_hash[:12]}"
    )
    # the same genome in another block is another retraining (another retraining seed)
    other_block = finalists.group_retrains(
        [dataclasses.replace(c, seed_block=1) if c.candidate_id == "camp-joint-b0.g100" else c for c in p.choices]
    )
    assert len(other_block) > len(p.retrains)
    assert len(p.retrains) <= finalists.MAX_RETRAINS == 72
    assert len({r.retrain_id for r in p.retrains}) == len(p.retrains)


def test_the_list_never_exceeds_the_72_the_protocol_allows() -> None:
    # 8 blocks, 4 methods x 2 roles + H1 = 9 per block, every genome distinct
    choices = []
    for block in range(8):
        for method in finalists.METHOD_ORDER:
            for role in ("menor_E", "equilibrio"):
                choices.append(
                    finalists.Choice(
                        f"r{block}{method}",
                        method,
                        block,
                        role,
                        f"c-{block}-{method}-{role}",
                        f"h-{block}-{method}-{role}",
                        0.3,
                        1.0,
                        1.0,
                        1.0,
                        "x",
                        None,
                    )
                )
        choices.append(
            finalists.Choice(
                f"r{block}joint", "joint", block, "H1", f"c-{block}-h1", f"h-{block}-h1", 0.3, 1.0, 1.0, 1.0, "x", 1.0
            )
        )
    assert len(choices) == 72 and len(finalists.group_retrains(choices)) == 72


def test_the_export_carries_the_list_with_hashes_and_labels_synthetic_numbers(
    campaign_db: tuple[Path, str], tmp_path: Path
) -> None:
    db, plan_id = campaign_db
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    p = finalists.propose(db, plan_id, allow_synthetic=True)
    dest = tmp_path / "finalistas"
    manifest = finalists.export_proposal(p, dest)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before  # read-only
    assert sorted(x.name for x in dest.iterdir()) == [
        "finalistas.csv",
        "h1_falhas.csv",
        finalists.MANIFEST_NAME,
        "retreinos.csv",
    ]
    for entry in manifest["files"]:
        assert hashlib.sha256((dest / entry["name"]).read_bytes()).hexdigest() == entry["sha256"]
    on_disk = json.loads((dest / finalists.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk["list_sha256"] == p.list_sha256 and on_disk["n_selections"] == 15 and on_disk["n_retrains"] == 9
    assert on_disk["official_test"].startswith("not opened") and on_disk["synthetic"] is True
    assert on_disk["plan_sha256"] == p.plan_sha256 and on_disk["reference_point"] == list(REFERENCE)

    with open(dest / "finalistas.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert (
        len(rows) == 15 and {"e_SINTETICO", "l_ms_SINTETICO", "score_SINTETICO"} <= set(rows[0]) and "e" not in rows[0]
    )
    assert {r["retrain_id"] for r in rows} == {r.retrain_id for r in p.retrains}
    shared = [r for r in rows if r["shares_retrain_with"]]
    assert len(shared) == 15 - 9 and {r["shares_retrain_with"] for r in shared} <= {
        "joint/equilibrio",
        "mono/menor_E",
        "mono/equilibrio",
    } | {f"{r['method']}/{r['role']}" for r in rows}
    with open(dest / "h1_falhas.csv", encoding="utf-8", newline="") as fh:
        failures = list(csv.DictReader(fh))
    assert [f["seed_block"] for f in failures] == ["1", "2"] and failures[0]["e_mono_SINTETICO"] == "0.2"
    with open(dest / "retreinos.csv", encoding="utf-8", newline="") as fh:
        assert max(int(r["n_roles"]) for r in csv.DictReader(fh)) == 3

    with pytest.raises(finalists.FinalistsError, match="already exists"):
        finalists.export_proposal(p, dest)
    # the list hash is the identity of the retrainings, not of how they were written
    assert finalists.propose(db, plan_id, allow_synthetic=True).list_sha256 == p.list_sha256


def test_what_it_refuses(tmp_path: Path, campaign_db: tuple[Path, str]) -> None:
    db, plan_id = campaign_db
    with pytest.raises(finalists.FinalistsError, match="synthetic results"):
        finalists.propose(db, plan_id)
    with pytest.raises(NotFoundError):
        finalists.propose(db, "no-such-plan", allow_synthetic=True)

    unfinished_db, unfinished_plan = build(tmp_path, unfinished=("random", 0), name="unfinished")
    with pytest.raises(finalists.FinalistsError, match="have not finished"):
        finalists.propose(unfinished_db, unfinished_plan, allow_synthetic=True)
    partial = finalists.propose(unfinished_db, unfinished_plan, allow_unfinished=True, allow_synthetic=True)
    assert partial.unfinished == ("camp-random-b0",)
    with SQLiteRunStore(unfinished_db, tmp_path / "unfinished" / "artifacts") as store:  # and it is never frozen
        with pytest.raises(finalists.FinalistsError, match="finished"):
            finalists.freeze_list(store, dataclasses.replace(partial, synthetic=False), list_id="l", author="t")

    no_minima_db, no_minima_plan = build(tmp_path, with_minima=False, name="nominima")
    with pytest.raises(finalists.FinalistsError, match="no viable_minima"):
        finalists.propose(no_minima_db, no_minima_plan, allow_synthetic=True)


def test_freezing_writes_the_list_and_still_leaves_the_official_test_sealed(
    campaign_db: tuple[Path, str], tmp_path: Path
) -> None:
    db, plan_id = campaign_db
    p = finalists.propose(db, plan_id, allow_synthetic=True)
    art = tmp_path / "camp" / "artifacts"
    with SQLiteRunStore(db, art) as store:
        with pytest.raises(finalists.FinalistsError, match="synthetic"):
            finalists.freeze_list(store, p, list_id="lista-1", author="t")  # a fake list is never frozen
        real = dataclasses.replace(p, synthetic=False)  # what a campaign with real measurements would hand over
        digest = finalists.freeze_list(store, real, list_id="lista-1", author="Pedro Antunes")
        assert finalists.freeze_list(store, real, list_id="lista-1", author="Pedro Antunes") == digest  # idempotent
        rows = store._conn.execute(
            "SELECT source_run_id, role, candidate_id FROM finalist_selections WHERE list_id = 'lista-1' ORDER BY selection_id"
        ).fetchall()
        assert len(rows) == 15 and {r[1] for r in rows} == {"menor_E", "equilibrio", "H1"}
        with pytest.raises(OfficialTestAccessError):  # the confirmatory runs still cannot read the sealed split
            store.official_test_split("camp-joint-b0")


def test_the_selection_code_cannot_open_the_test_split_nor_write_by_itself() -> None:
    tree = ast.parse(Path(finalists.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    assert not names & {"official_test_split", "record_test_evaluation", "SQLiteRunStore", "execute", "executescript"}
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "cnnlab.tcc.storage.readmodel" in imported  # it reads through the read model


def test_the_cli_exports_and_can_freeze(
    campaign_db: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, plan_id = campaign_db
    dest = tmp_path / "cli"
    base = ["campaign", "finalists", "--db", str(db), "--plan", plan_id, "--dest", str(dest)]
    assert cli.main(base) == EXIT_REFUSED and "synthetic results" in capsys.readouterr().err and not dest.exists()
    assert cli.main([*base, "--allow-synthetic"]) == 0
    out = capsys.readouterr().out
    assert "15 selections, 9 retrainings (at most 72), 2 seed(s) without an H1 candidate" in out
    assert "SYNTHETIC results selected" in out and "list sha256=" in out
    assert cli.main([*base, "--allow-synthetic"]) == EXIT_REFUSED and "already exists" in capsys.readouterr().err
    other = tmp_path / "cli2"
    refused = cli.main(
        [
            "campaign",
            "finalists",
            "--db",
            str(db),
            "--plan",
            plan_id,
            "--dest",
            str(other),
            "--allow-synthetic",
            "--freeze",
            "lista-x",
            "--artifacts",
            str(tmp_path / "camp" / "artifacts"),
        ]
    )
    assert refused == EXIT_REFUSED and "synthetic" in capsys.readouterr().err  # exported, but never frozen
    assert (other / finalists.MANIFEST_NAME).exists()
    with ReadModel(db) as rm:
        assert rm.schema_version() >= 3


def test_a_campaign_measured_by_two_bench_versions_is_not_selected_from(tmp_path: Path) -> None:
    db, plan_id = build(
        tmp_path,
        bench={
            "mono": "tcc-bench/preliminary-v3",
            "joint": "tcc-bench/preliminary-v4",
            "random": "tcc-bench/preliminary-v4",
            "sequential": "tcc-bench/preliminary-v4",
        },
    )
    with pytest.raises(StorageError, match="different benchmark versions"):
        finalists.propose(db, plan_id, allow_synthetic=True)
