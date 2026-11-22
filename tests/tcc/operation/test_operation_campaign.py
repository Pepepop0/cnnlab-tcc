"""The plan of the confirmatory campaign (task A6): seeds, order, freezing, export, queue, and the runs against it.

CPU only. The wiring is the synthetic one of the operation tests, opted into explicitly; the code version is a
fixed fake (a campaign run refuses any code but the one the plan froze), so no test depends on the state of the
checkout. Numbers in the pilot reports below are written by the tests, not measured.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.operation import campaign, cli, schedule
from cnnlab.tcc.operation.campaign import CampaignSettings
from cnnlab.tcc.operation.demo import (
    EXIT_REFUSED,
    CodeVersionMismatchError,
    DemoRunSpec,
    default_repo_root,
    prepare_demo,
    start_demo,
)
from cnnlab.tcc.operation.provenance import CodeVersion
from cnnlab.tcc.operation.testing import synthetic_demo_wiring
from cnnlab.tcc.storage import ReadModel, SQLiteRunStore
from cnnlab.tcc.storage.canonical import sha256_hex
from cnnlab.tcc.storage.errors import ConsistencyError, ImmutableRecordError, NotFoundError

FACTORY = "cnnlab.tcc.operation.testing:synthetic_demo_wiring"
FROZEN = CodeVersion("campanha", "c" * 40, False, "7" * 64, (), (), 0)
OTHER = CodeVersion("campanha", "d" * 40, False, "8" * 64, (), (), 0)
DIRTY = CodeVersion("campanha", "c" * 40, True, "9" * 64, ("src/x.py",), (), 0)
REFERENCE = (1.0, 2.5, 12.5, 5.0)
MINIMA = (0.2, 0.5, 2.0, 1.0)  # the viable minima of an imaginary pilot
REPO = default_repo_root()


def settings(**kw: Any) -> CampaignSettings:
    base: dict[str, Any] = {
        "master_seed": 20261026,
        "factory": FACTORY,
        "budget_gpu_s": 19_800.0,
        "factory_args": {"slice": "fp16_regions", "epochs": 5, "n_candidates": 3, "sleep_s": 0.0},
        "reference_point": REFERENCE,
        "viable_minima": MINIMA,
        "max_attempt_train_s": 1800.0,
        "latency_cv_max": 0.02,
        "v_audit_rel": 0.05,
    }
    base.update(kw)
    return CampaignSettings.from_mapping(base)


def wiring_of(s: CampaignSettings) -> Any:
    return synthetic_demo_wiring(campaign.representative_spec(s))


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "db" / "tcc.sqlite3", tmp_path / "artifacts"


def frozen_plan(paths: tuple[Path, Path], s: CampaignSettings | None = None, **kw: Any) -> campaign.CampaignPlan:
    s = s or settings()
    with SQLiteRunStore(*paths) as store:
        return campaign.create_plan(store, s, wiring_of(s), FROZEN, REPO, freeze=True, **kw)


# -- seeds and order ------------------------------------------------------------------------------


def test_seeds_are_derived_by_a_stated_rule_and_distinct_everywhere() -> None:
    blocks = campaign.derive_blocks(20261026)
    assert [b["seed_block"] for b in blocks] == list(range(8))
    seeds = [b[k] for b in blocks for k in campaign.SEED_KINDS]
    assert len(set(seeds)) == 32 and all(0 <= x < campaign.SEED_LIMIT for x in seeds)  # no two seeds alike, any kind
    assert campaign.derive_blocks(20261026) == blocks  # anyone can re-derive them
    assert campaign.derive_blocks(20261027) != blocks and campaign.derive_blocks(20261026, 3) == blocks[:3]
    assert blocks[0]["search_seed"] == campaign.derive_seed(20261026, "search_seed", 0)  # the rule, written out

    import hashlib

    text = "cnnlab-tcc-campaign:20261026:search_seed:0:0"
    assert blocks[0]["search_seed"] == int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) % (2**31 - 1)


def test_a_seed_collision_is_redrawn_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    real = campaign.derive_seed
    monkeypatch.setattr(campaign, "derive_seed", lambda m, k, b, a=0: 5 if a == 0 else real(m, k, b, a))
    blocks = campaign.derive_blocks(1, 2)
    seeds = [b[k] for b in blocks for k in campaign.SEED_KINDS]
    assert len(set(seeds)) == 8 and seeds.count(5) == 1  # the first draw took 5, every other one was re-drawn


def test_the_order_is_a_latin_square_so_no_method_is_always_first() -> None:
    order = campaign.run_order(8, campaign.DEFAULT_METHODS)
    assert [c["position"] for c in order] == list(range(1, 33))
    for block in range(8):
        cells = [c for c in order if c["seed_block"] == block]
        assert sorted(c["method"] for c in cells) == sorted(campaign.DEFAULT_METHODS)  # every method, once per block
        assert [c["position"] for c in cells] == list(range(block * 4 + 1, block * 4 + 5))  # blocks stay in order
    first = [order[b * 4]["method"] for b in range(8)]
    assert first == ["joint", "mono", "random", "sequential"] * 2
    counts = campaign.position_counts(order, 4)
    assert all(row == [2, 2, 2, 2] for row in counts.values())  # each method takes each position twice


# -- the plan -----------------------------------------------------------------------------------


def test_the_plan_records_what_the_protocol_asks_to_be_frozen(paths: tuple[Path, Path]) -> None:
    made = frozen_plan(paths)
    c = made.content
    assert made.frozen and made.pending == () and made.plan_id.startswith("campaign-plan-")
    assert c["format"] == "tcc-campaign-plan/1" and c["confirmatory"] is True and c["stage_scope"] == "confirmatory"
    assert len(c["blocks"]) == 8 and c["methods"] == list(campaign.DEFAULT_METHODS) and c["budget_gpu_s"] == 19_800.0
    assert len(c["run_order"]) == 32 and {k: v for k, v in c["code_freeze"].items() if k != "tags"} == {
        "git_sha": "c" * 40,
        "dirty": False,
        "dirty_fingerprint": "7" * 64,
        "label": "campanha-cccccccc",
    }
    assert isinstance(c["code_freeze"]["tags"], list)  # the git tags of the frozen commit (whatever the checkout has)
    assert c["space_hash"] == wiring_of(settings()).space_hash and c["reference_point"] == list(REFERENCE)
    assert c["max_attempt_train_s"] == 1800.0 and campaign.attempt_time_limit(c) == 1800.0
    assert c["factory_args"]["train_time_limit_s"] == 1800.0  # the key the evaluator reads: one limit, in both places
    assert c["factory_args"]["population"] == 30 and c["factory_args"]["generations"] is None  # protocol defaults
    assert c["factory_args"]["train_per_class"] is None and c["factory_args"]["reference_point"] == list(REFERENCE)
    assert c["factory_args"]["n_frozen"] == 7 and c["sequential"]["split"].startswith("half")
    assert {"unit", "checked", "attempt_in_progress", "excess_over_budget", "idle_time"} <= set(c["cut_rule"])
    assert {"attempt_ids", "interrupted_attempt_time", "resume", "code_change_on_resume"} <= set(
        c["interruption_policy"]
    )
    assert c["tolerances"]["sentinel_every_evaluations"] == 50 and c["tolerances"]["sentinel_drift_rel"] == 0.05
    assert c["seed_rule"]["master_seed"] == 20261026 and c["seed_rule"]["paired"].startswith("the four methods")
    with SQLiteRunStore(*paths) as store:
        with pytest.raises(ImmutableRecordError):
            store.update_plan(made.plan_id, {**c, "budget_gpu_s": 1.0})  # frozen: an amendment would be a new plan
        again = campaign.create_plan(store, settings(), wiring_of(settings()), FROZEN, REPO, freeze=True)
    assert (again.plan_id, again.plan_sha256) == (made.plan_id, made.plan_sha256)  # generating it again is a no-op


def test_a_plan_with_a_field_of_the_pilot_still_pending_cannot_be_frozen(paths: tuple[Path, Path]) -> None:
    s = settings(
        reference_point=None, viable_minima=None, max_attempt_train_s=None, latency_cv_max=None, v_audit_rel=None
    )
    with SQLiteRunStore(*paths) as store:
        with pytest.raises(
            ConsistencyError, match="reference_point.*viable_minima.*max_attempt_train_s.*tolerances.latency_cv_max"
        ):
            campaign.create_plan(store, s, wiring_of(s), FROZEN, REPO, freeze=True)
    assert not (paths[0].exists() and ReadModel(paths[0]).runs())  # refused: no plan, no run
    draft = campaign.create_plan(SQLiteRunStore(*paths), s, wiring_of(s), FROZEN, REPO, freeze=False)
    assert draft.status == "draft" and draft.pending[:3] == ("reference_point", "viable_minima", "max_attempt_train_s")
    assert not draft.frozen

    pending_ok = frozen_plan(paths, s, allow_pending=True)  # what only these tests may do
    assert pending_ok.frozen and "reference_point" in pending_ok.content["pending"]
    with SQLiteRunStore(*paths) as store:  # ... and a run against such a plan is refused
        spec = campaign_spec(pending_ok, "joint", 0)
        with pytest.raises(ConsistencyError, match="pending fields"):
            prepare_demo(store, spec, synthetic_demo_wiring(spec), FROZEN, repo_root=REPO, allow_synthetic=True)


def test_the_plan_is_not_frozen_on_unknown_or_dirty_code(paths: tuple[Path, Path]) -> None:
    unknown = CodeVersion("campanha", None, None, None, (), (), 0, error="no git")
    with SQLiteRunStore(*paths) as store:
        with pytest.raises(ConsistencyError, match="code version unknown"):
            campaign.create_plan(store, settings(), wiring_of(settings()), unknown, REPO, freeze=True)
        with pytest.raises(ConsistencyError, match="dirty tree"):
            campaign.create_plan(store, settings(), wiring_of(settings()), DIRTY, REPO, freeze=True)
        ok = campaign.create_plan(store, settings(), wiring_of(settings()), DIRTY, REPO, freeze=True, allow_dirty=True)
    assert ok.content["code_freeze"]["dirty"] is True  # only a test may; the plan says so


def test_settings_that_make_no_sense_are_refused() -> None:
    for bad in (
        {"factory_args": {"epochs": 5}},  # no slice
        {"factory_args": {"slice": "fp16_regions", "generations": 3}},  # sequential has no generation limit
        {"budget_gpu_s": 19_800.5},
        {"budget_gpu_s": 0},
        {"methods": ("joint", "joint")},
        {"n_blocks": 0},
        {"reference_point": (1.0, 2.0, 3.0)},
        {"reference_point": (1.0, 2.0, -3.0, 4.0)},
        {"max_attempt_train_s": 0},
        {"master_seed": True},
        {"factory": "nocolon"},
    ):
        with pytest.raises(ValueError):
            settings(**bad)
    with pytest.raises(ValueError, match="unknown campaign settings"):
        CampaignSettings.from_mapping({"master_seed": 1, "factory": FACTORY, "factory_args": {"slice": "x"}, "oops": 1})
    assert settings(factory_args={"slice": "fp16_regions", "generations": 3}, methods=("joint", "mono")).methods == (
        "joint",
        "mono",
    )  # a generation limit is only refused where sequential is part of the plan


def test_the_two_measurement_tolerances_have_defaults_and_say_why(paths: tuple[Path, Path]) -> None:
    bare = CampaignSettings.from_mapping(
        {"master_seed": 1, "factory": FACTORY, "factory_args": {"slice": "fp16_regions"}}
    )
    assert (bare.latency_cv_max, bare.v_audit_rel) == (0.02, 0.05) == (campaign.LATENCY_CV_MAX, campaign.V_AUDIT_REL)
    made = frozen_plan(paths, settings(latency_cv_max=0.02, v_audit_rel=0.05))
    t = made.content["tolerances"]
    assert (t["latency_cv_max"], t["v_audit_rel"]) == (0.02, 0.05) and "3,000 repetitions" in t["latency_cv_rule"]
    assert "sentinel drift" in t["v_audit_rule"] and "tolerances.latency_cv_max" not in made.pending
    assert settings(v_audit_rel=0.1).v_audit_rel == 0.1  # a different value is still the operator's to give


def test_the_training_time_limit_of_the_plan_reaches_the_evaluator_and_is_checked_on_every_run(
    paths: tuple[Path, Path],
) -> None:
    made = frozen_plan(paths)
    assert made.content["factory_args"]["train_time_limit_s"] == made.content["max_attempt_train_s"] == 1800.0
    plan = campaign.read_plan(paths[0], made.plan_id)
    [(_, item)] = [x for x in campaign.queue_items(plan)][:1]
    assert item.factory_extra["train_time_limit_s"] == 1800.0  # it travels with the queue item into the spec
    spec = DemoRunSpec.from_mapping(
        schedule.spec_payload(schedule.add_item(schedule.QueueState(), "x", item, tag=schedule.CAMPAIGN).items[0])
    )
    assert spec.factory_args["train_time_limit_s"] == 1800.0

    # a plan of an older generator: the limit is in the plan but not in the arguments the evaluator would read
    s = settings()
    content = campaign.build_content(s, wiring_of(s), FROZEN, made.content["protocol_id"])
    content["seed_rule"] = {**content["seed_rule"], "master_seed": 20261027}  # another plan id, same shape
    del content["factory_args"]["train_time_limit_s"]
    with SQLiteRunStore(*paths) as store:
        store.create_plan("old-plan", made.content["protocol_id"], content, author="t")
        store.freeze_plan("old-plan")
    old = campaign.CampaignPlan("old-plan", "", "frozen", content)
    with pytest.raises(ConsistencyError, match="training time limit of the run"):
        prepare(paths, campaign_spec(old, "joint", 0))
    with pytest.raises(ConsistencyError, match="training time limit"):  # nor may a run carry another limit
        prepare(
            paths,
            campaign_spec(
                made, "joint", 0, factory_args={**made.content["factory_args"], "train_time_limit_s": 99999.0}
            ),
        )

    with pytest.raises(ConsistencyError, match="they are one limit"):  # typed twice, differently
        campaign.build_content(
            settings(factory_args={"slice": "fp16_regions", "epochs": 5, "train_time_limit_s": 60.0}),
            wiring_of(s),
            FROZEN,
            "p",
        )
    same = campaign.build_content(
        settings(factory_args={"slice": "fp16_regions", "epochs": 5, "train_time_limit_s": 1800}),
        wiring_of(s),
        FROZEN,
        "p",
    )
    assert same["factory_args"]["train_time_limit_s"] == 1800.0  # typed once, in agreement: fine


def test_the_plan_asks_for_the_sentinel_and_the_run_carries_it(paths: tuple[Path, Path]) -> None:
    made = frozen_plan(paths)
    args = made.content["factory_args"]
    assert args["sentinel_every"] == 50 and args["sentinel_drift_limit"] == 0.05  # the plan's tolerances, as arguments
    assert made.content["tolerances"]["sentinel_every_evaluations"] == 50
    custom = frozen_plan(paths, settings(sentinel_every_evaluations=25, sentinel_drift_rel=0.03, master_seed=3))
    assert (
        custom.content["factory_args"]["sentinel_every"],
        custom.content["factory_args"]["sentinel_drift_limit"],
    ) == (25, 0.03)
    plan = campaign.read_plan(paths[0], made.plan_id)
    [(_, item)] = campaign.queue_items(plan)[:1]
    assert (
        item.factory_extra["sentinel_every"] == 50
    )  # it travels with the queue item, checked against the plan on every run
    with pytest.raises(ConsistencyError, match="factory arguments"):
        prepare(paths, campaign_spec(made, "joint", 0, factory_args={**args, "sentinel_every": 10}))


def test_the_plan_builds_its_wiring_with_the_arguments_its_runs_get(paths: tuple[Path, Path]) -> None:
    # 2026-10-11: the plan asked the factory for the dataset with the bare settings ({"slice": ...}), the factory filled
    # in its own 100-per-class subset, and every run (full 45k split) was then refused for a different dataset.
    s = settings()
    assert "train_per_class" not in s.factory_args
    spec = campaign.representative_spec(s)
    content = frozen_plan(paths, s).content
    for name in campaign.RUN_DEFAULT_ARGS:
        assert spec.factory_args[name] == content["factory_args"][name], name
    assert spec.factory_args["train_per_class"] is None  # the full split, as the runs train
    typed = campaign.representative_spec(settings(factory_args={**s.factory_args, "train_per_class": 50}))
    assert typed.factory_args["train_per_class"] == 50  # a value typed in the settings still wins


def test_the_plan_describes_what_the_code_does() -> None:
    from cnnlab.tcc.science.search import SearchConfig

    assert SearchConfig().max_interrupted_retries == 1  # "an interrupted attempt is retried once"
    assert "once" in campaign.INTERRUPTION_POLICY["infrastructure_retries"]
    assert SearchConfig(generations=None).max_stalled_generations == 3  # the search that ends before the budget
    assert "search_exhausted" in campaign.CUT_RULE["search_ends_before_the_budget"]
    assert campaign.CUT_RULE["checked"] == "between attempts, never inside one"
    assert campaign.INTERRUPTION_POLICY["unsupervised_session_max_h"] == 48  # TCC-01 section 4


# -- what the pilot decides -----------------------------------------------------------------------------


def pilot_report(tmp_path: Path, **decisions: Any) -> Path:
    path = tmp_path / "piloto.json"
    body = {
        "run_id": "pilot-x",
        "decisions": {
            "search_epochs": 5,
            "latency_repetitions": 1000,
            "train_time_limit_s": 2400,
            "reference_point": {"E": 1.0, "L_ms": 2.5, "V_mb": 12.5, "S_mb": 5.0},
            **decisions,
        },
        "viable_minima": {"E": 0.2, "L_ms": 0.5, "V_mb": 2.0, "S_mb": 1.0},
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_the_pilot_report_fills_the_plan_and_the_plan_records_where_it_came_from(
    paths: tuple[Path, Path], tmp_path: Path
) -> None:
    report = pilot_report(tmp_path)
    pilot = campaign.load_pilot(report)
    assert pilot.file_sha256 == sha256_hex(report.read_bytes()) and pilot.run_id == "pilot-x"
    s = settings(
        factory_args={"slice": "fp16_regions", "n_candidates": 3, "sleep_s": 0.0},
        reference_point=None,
        viable_minima=None,
        max_attempt_train_s=None,
    )  # nothing typed: all of it from the report
    made = frozen_plan(paths, s, pilot=pilot)
    c = made.content
    assert c["factory_args"]["epochs"] == 5 and c["factory_args"]["bench_iters"] == 1000
    assert c["reference_point"] == [1.0, 2.5, 12.5, 5.0] and c["max_attempt_train_s"] == 2400.0
    assert c["viable_minima"] == [0.2, 0.5, 2.0, 1.0]  # next to the reference point: finalists need both (section 6)
    assert c["pilot"] == {
        "report_sha256": pilot.file_sha256,
        "run_id": "pilot-x",
        "search_epochs": 5,
        "latency_repetitions": 1000,
        "max_attempt_train_s": 2400,
        "reference_point": [1.0, 2.5, 12.5, 5.0],
        "viable_minima": [0.2, 0.5, 2.0, 1.0],
        "timing_cache": None,
    }

    eight = campaign.apply_pilot(
        s, campaign.load_pilot(pilot_report(tmp_path, search_epochs=8, latency_repetitions=3000))
    )
    assert eight.factory_args["epochs"] == 8 and eight.factory_args["bench_iters"] == 3000


def test_a_value_typed_in_the_settings_cannot_disagree_with_the_pilot(tmp_path: Path) -> None:
    pilot = campaign.load_pilot(pilot_report(tmp_path))
    for typed, match in (
        ({"factory_args": {"slice": "fp16_regions", "epochs": 8}}, "epochs"),
        ({"reference_point": (1.0, 3.0, 12.5, 5.0)}, "reference_point"),
        ({"max_attempt_train_s": 600.0}, "max_attempt_train_s"),
    ):
        with pytest.raises(ConsistencyError, match=match):
            campaign.apply_pilot(settings(**typed), pilot)
    same = campaign.apply_pilot(settings(reference_point=(1.0, 2.5, 12.5, 5.0), max_attempt_train_s=2400.0), pilot)
    assert same.reference_point == (1.0, 2.5, 12.5, 5.0)  # agreeing is fine

    undecided = campaign.load_pilot(
        pilot_report(
            tmp_path, search_epochs=None, reference_point=None, train_time_limit_s=None, latency_repetitions=None
        )
    )
    open_ = campaign.apply_pilot(
        settings(reference_point=None, max_attempt_train_s=None, factory_args={"slice": "fp16_regions"}), undecided
    )
    assert open_.reference_point is None and "epochs" not in open_.factory_args  # a decision not taken stays pending
    with pytest.raises(ValueError, match="not a pilot report"):
        bad = tmp_path / "not-a-report.json"
        bad.write_text("{}", encoding="utf-8")
        campaign.load_pilot(bad)


# -- export (read-only) ------------------------------------------------------------------------------


def test_the_plan_is_exported_with_its_hash_without_writing_the_database(
    paths: tuple[Path, Path], tmp_path: Path
) -> None:
    made = frozen_plan(paths)
    before = paths[0].read_bytes()
    dest = tmp_path / "exportado"
    exported = campaign.export_plan(paths[0], made.plan_id, dest)
    body = (dest / campaign.PLAN_EXPORT_NAME).read_bytes()
    assert exported.plan_sha256 == made.plan_sha256 == sha256_hex(body)  # the hash is the hash of the file
    assert (dest / campaign.PLAN_HASH_NAME).read_text(
        encoding="utf-8"
    ) == f"{made.plan_sha256}  {campaign.PLAN_EXPORT_NAME}\n"
    parsed = json.loads(body)
    assert parsed["format"] == "tcc-plan/1" and parsed["plan_id"] == made.plan_id and parsed["content"] == made.content
    assert paths[0].read_bytes() == before  # read-only
    with pytest.raises(Exception, match="already exists"):
        campaign.export_plan(paths[0], made.plan_id, dest)
    with pytest.raises(NotFoundError):
        campaign.export_plan(paths[0], "no-such-plan", tmp_path / "other")
    with ReadModel(paths[0]) as rm:
        stored = rm.plan(made.plan_id)
    assert stored["hash_ok"] is True and stored["status"] == "frozen" and stored["frozen_at"]


# -- a run against the plan -------------------------------------------------------------------------------


def campaign_spec(plan: campaign.CampaignPlan, method: str, block: int, **kw: Any) -> DemoRunSpec:
    c = plan.content
    seeds = next(b for b in c["blocks"] if b["seed_block"] == block)
    base: dict[str, Any] = {
        "method": method,
        "seed_block": block,
        "search_seed": seeds["search_seed"],
        "train_seed": seeds["train_seed"],
        "calibration_seed": seeds["calibration_seed"],
        "budget_gpu_s": c["budget_gpu_s"],
        "code_label": c["code_freeze"]["label"],
        "author": "Pedro Antunes",
        "run_id": f"camp-test-{method}-b{block}",
        "factory": c["factory"],
        "factory_args": dict(c["factory_args"]),
        "stage": "confirmatory",
        "plan_id": plan.plan_id,
    }
    base.update(kw)
    return DemoRunSpec.from_mapping(base)


def prepare(paths: tuple[Path, Path], spec: DemoRunSpec, code: CodeVersion = FROZEN, repo: Path = REPO) -> Any:
    with SQLiteRunStore(*paths) as store:
        return prepare_demo(store, spec, synthetic_demo_wiring(spec), code, repo_root=repo, allow_synthetic=True)


def test_a_run_of_the_plan_is_created_confirmatory_against_the_frozen_plan(paths: tuple[Path, Path]) -> None:
    made = frozen_plan(paths)
    spec = campaign_spec(made, "mono", 3)
    prepared = prepare(paths, spec)
    assert prepared.plan_id == made.plan_id and prepared.plan_sha256 == made.plan_sha256
    assert prepared.job_spec.stage == "confirmatory" and prepared.job_spec.method == "mono"
    assert prepared.job_spec.seed_block == 3 and prepared.job_spec.run_id == "camp-test-mono-b3"
    with ReadModel(paths[0]) as rm:
        [run] = rm.runs()
    assert (run["stage"], run["plan_id"], run["plan_confirmatory"], run["plan_purpose"]) == (
        "confirmatory",
        made.plan_id,
        True,
        "confirmatory-campaign",
    )
    assert prepare(paths, spec).job.job_id == prepared.job.job_id  # preparing again is a no-op
    other = prepare(paths, campaign_spec(made, "joint", 3))  # a second cell shares the plan and the environment
    assert other.environment_id == prepared.environment_id and other.plan_id == prepared.plan_id


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"search_seed": 1}, ConsistencyError, "search_seed"),
        ({"train_seed": 1}, ConsistencyError, "train_seed"),
        ({"calibration_seed": 1}, ConsistencyError, "calibration_seed"),
        ({"budget_gpu_s": 9_900.0}, ConsistencyError, "budget"),
        ({"seed_block": 9}, ConsistencyError, "seed_block 9"),
        ({"factory": "cnnlab.tcc.demo_wiring:demo_wiring"}, ConsistencyError, "factory"),
        (
            {"factory_args": {"slice": "fp16_regions", "epochs": 8, "population": 30, "train_time_limit_s": 1800.0}},
            ConsistencyError,
            "factory arguments",
        ),
        ({"run_id": None}, ValueError, "run_id"),
    ],
)
def test_a_run_that_differs_from_the_plan_is_refused_before_anything_is_written(
    paths: tuple[Path, Path], change: dict[str, Any], error: type[Exception], match: str
) -> None:
    made = frozen_plan(paths)
    with pytest.raises(error, match=match):
        prepare(paths, campaign_spec(made, "joint", 0, **change))
    with ReadModel(paths[0]) as rm:
        assert rm.runs() == []  # the plan is there; no run, no job


def test_the_code_the_protocol_the_space_and_the_dataset_must_be_the_frozen_ones(
    paths: tuple[Path, Path], tmp_path: Path
) -> None:
    made = frozen_plan(paths)
    spec = campaign_spec(made, "joint", 0)
    with pytest.raises(CodeVersionMismatchError, match="froze code cccccccccccc.*dddddddddddd"):
        prepare(paths, spec, OTHER)
    with pytest.raises(CodeVersionMismatchError, match="uncommitted changes"):
        prepare(paths, spec, DIRTY)

    docs = tmp_path / "repo" / "docs"
    docs.mkdir(parents=True)
    shutil.copy(REPO / "docs" / "TCC-01-PROTOCOLO.md", docs / "TCC-01-PROTOCOLO.md")
    assert prepare(paths, spec, repo=tmp_path / "repo").plan_id == made.plan_id  # the same document: fine
    (docs / "TCC-01-PROTOCOLO.md").write_text("changed after the plan was frozen\n", encoding="utf-8")
    with pytest.raises(ConsistencyError, match="protocol document changed"):
        prepare(paths, campaign_spec(made, "mono", 0), repo=tmp_path / "repo")

    with SQLiteRunStore(*paths) as store:  # a wiring for another space or another dataset
        import dataclasses

        wiring = synthetic_demo_wiring(spec)
        with pytest.raises(ConsistencyError, match="space_hash differs"):
            prepare_demo(
                store,
                campaign_spec(made, "random", 1),
                dataclasses.replace(wiring, space_hash="f" * 64),
                FROZEN,
                repo_root=REPO,
                allow_synthetic=True,
            )
        moved = dataclasses.replace(wiring.dataset, dataset_id="another-dataset")
        with pytest.raises(ConsistencyError, match="dataset"):
            prepare_demo(
                store,
                campaign_spec(made, "random", 1),
                dataclasses.replace(wiring, dataset=moved),
                FROZEN,
                repo_root=REPO,
                allow_synthetic=True,
            )


def test_a_draft_plan_a_pilot_plan_and_a_method_outside_the_plan_do_not_start_a_run(paths: tuple[Path, Path]) -> None:
    s = settings(methods=("joint", "mono"))
    made = frozen_plan(paths, s)
    with pytest.raises(ConsistencyError, match="not in plan"):
        prepare(paths, campaign_spec(made, "random", 0))
    draft_settings = settings(
        reference_point=None,
        viable_minima=None,
        max_attempt_train_s=None,
        latency_cv_max=None,
        v_audit_rel=None,
        master_seed=5,
    )
    with SQLiteRunStore(*paths) as store:
        draft = campaign.create_plan(store, draft_settings, wiring_of(draft_settings), FROZEN, REPO, freeze=False)
    with pytest.raises(ConsistencyError, match="draft"):
        prepare(paths, campaign_spec(draft, "joint", 0))

    with SQLiteRunStore(*paths) as store:  # a plan that is not a campaign plan (the pilot's own, for instance)
        store.create_plan("a-pilot-plan", made.content["protocol_id"], {"purpose": "exploratory-demo"}, author="t")
        store.freeze_plan("a-pilot-plan")
    spec = campaign_spec(made, "joint", 0, plan_id="a-pilot-plan")
    with pytest.raises(ConsistencyError, match="not a campaign plan"):
        prepare(paths, spec)


def test_a_spec_cannot_be_confirmatory_without_a_plan_nor_a_plan_without_being_confirmatory() -> None:
    base = {
        "method": "joint",
        "seed_block": 0,
        "search_seed": 1,
        "train_seed": 2,
        "calibration_seed": 3,
        "budget_gpu_s": 60.0,
        "code_label": "x",
        "author": "t",
    }
    for extra in (
        {"stage": "confirmatory"},
        {"plan_id": "p"},
        {"stage": "finalist", "plan_id": "p"},
        {"stage": "nope"},
    ):
        with pytest.raises(ValueError):
            DemoRunSpec.from_mapping({**base, **extra})
    assert DemoRunSpec.from_mapping({**base, "stage": "confirmatory", "plan_id": "p"}).stage == "confirmatory"
    assert DemoRunSpec.from_mapping(base).stage == "pilot" and DemoRunSpec.from_mapping(base).plan_id is None


# -- the queue --------------------------------------------------------------------------------------------


def test_the_queue_gets_the_runs_in_the_order_of_the_plan_and_refuses_to_change_them(
    paths: tuple[Path, Path], tmp_path: Path
) -> None:
    made = frozen_plan(paths)
    plan = campaign.read_plan(paths[0], made.plan_id)
    qdir = tmp_path / "fila"
    added = campaign.enqueue(qdir, plan)
    assert len(added) == 32 and added[:4] == ["joint-b0", "mono-b0", "random-b0", "sequential-b0"]
    assert added[4:8] == ["mono-b1", "random-b1", "sequential-b1", "joint-b1"]  # the Latin square, in the queue
    state = schedule.load_state(qdir)
    assert [i.tag for i in state.items] == [schedule.CAMPAIGN] * 32 and all(i.from_plan for i in state.items)

    item = state.items[1]  # mono, block 0
    seeds = made.content["blocks"][0]
    assert (item.settings.method, item.settings.seed_block, item.settings.plan_id) == ("mono", 0, made.plan_id)
    assert (item.settings.search_seed, item.settings.train_seed, item.settings.calibration_seed) == (
        seeds["search_seed"],
        seeds["train_seed"],
        seeds["calibration_seed"],
    )
    assert schedule.run_id_of(item) == f"camp-{made.plan_id.rsplit('-', 1)[-1][:8]}-mono-b0"
    spec = DemoRunSpec.from_mapping(schedule.spec_payload(item))  # what the executor hands to the worker
    assert (spec.stage, spec.plan_id, spec.method, spec.seed_block) == ("confirmatory", made.plan_id, "mono", 0)
    assert dict(spec.factory_args) == made.content["factory_args"]  # exactly the frozen arguments
    assert spec.code_label == "campanha-cccccccc" and spec.budget_gpu_s == 19_800

    with pytest.raises(schedule.ScheduleError, match="plano congelado"):
        schedule.update_item(state, item.item_id, "novo-nome", item.settings)
    with pytest.raises(schedule.ScheduleError, match="ordem nao muda"):
        schedule.move_item(state, item.item_id, 2)
    assert schedule.next_planned(state).name == "joint-b0"  # the plan's order, not the queue's convenience

    assert campaign.enqueue(qdir, plan) == []  # queuing again adds nothing


def test_a_run_deleted_from_the_middle_is_not_put_back_at_the_end(paths: tuple[Path, Path], tmp_path: Path) -> None:
    plan = campaign.read_plan(paths[0], frozen_plan(paths).plan_id)
    qdir = tmp_path / "fila"
    campaign.enqueue(qdir, plan)
    state = schedule.load_state(qdir)
    schedule.mutate(qdir, lambda st: schedule.delete_item(st, state.items[5].item_id))  # random-b1
    with pytest.raises(schedule.ScheduleError, match="order of the plan holds"):
        campaign.enqueue(qdir, plan)
    assert len(schedule.load_state(qdir).items) == 31  # nothing was added out of order

    # The end of the plan is another matter: a run deleted from the end goes back to the end, where it belongs.
    other = tmp_path / "fila-fim"
    campaign.enqueue(other, plan)
    tail = schedule.load_state(other).items[-1]
    schedule.mutate(other, lambda st: schedule.delete_item(st, tail.item_id))
    assert campaign.enqueue(other, plan) == [tail.name]
    assert schedule.load_state(other).items[-1].name == tail.name and len(schedule.load_state(other).items) == 32


def test_only_a_frozen_plan_can_be_queued(paths: tuple[Path, Path], tmp_path: Path) -> None:
    s = settings(
        reference_point=None, viable_minima=None, max_attempt_train_s=None, latency_cv_max=None, v_audit_rel=None
    )
    with SQLiteRunStore(*paths) as store:
        draft = campaign.create_plan(store, s, wiring_of(s), FROZEN, REPO, freeze=False)
    plan = campaign.read_plan(paths[0], draft.plan_id)
    with pytest.raises(ConsistencyError, match="only a frozen plan"):
        campaign.enqueue(tmp_path / "fila", plan)
    assert not (tmp_path / "fila" / schedule.QUEUE_FILE).exists()


def test_the_campaign_items_do_not_disturb_manual_or_drawn_runs(tmp_path: Path) -> None:
    s = schedule.RunSettings()
    state = schedule.add_item(schedule.QueueState(), "sorteada", s, tag=schedule.SLEEP)
    state = schedule.add_item(
        state, "da-campanha", schedule.RunSettings(plan_id="p-1", method="mono"), tag=schedule.CAMPAIGN
    )
    state = schedule.add_item(state, "manual", s)
    assert [schedule.next_planned(state).name] == ["da-campanha"]  # drawn runs wait for what was asked for
    state = schedule.delete_item(state, state.items[1].item_id)
    assert schedule.next_planned(state).name == "manual"
    with pytest.raises(schedule.ScheduleError, match="tag invalida"):
        schedule.add_item(state, "x", s, tag="outra")


def test_a_queue_file_from_before_the_campaign_still_loads(tmp_path: Path) -> None:
    state = schedule.add_item(schedule.QueueState(), "antiga", schedule.RunSettings())
    data = schedule.state_to_json(state)
    for key in ("calibration_seed", "seed_block", "plan_id", "factory_extra", "code_label"):
        del data["items"][0]["settings"][key]
    (tmp_path / schedule.QUEUE_FILE).write_text(json.dumps(data), encoding="utf-8")
    loaded = schedule.load_state(tmp_path)
    assert loaded.items[0].settings == schedule.RunSettings() and not loaded.items[0].from_plan
    spec = schedule.spec_payload(loaded.items[0])
    assert spec["seed_block"] == 0 and spec["calibration_seed"] == 0 and "plan_id" not in spec  # as before


# -- the whole path ---------------------------------------------------------------------------------------


def test_queue_item_to_finished_run_through_the_real_operation(paths: tuple[Path, Path], tmp_path: Path) -> None:
    made = frozen_plan(paths)
    plan = campaign.read_plan(paths[0], made.plan_id)
    qdir = tmp_path / "fila"
    campaign.enqueue(qdir, plan)
    item = schedule.load_state(qdir).items[2]  # random, block 0
    spec = DemoRunSpec.from_mapping(schedule.spec_payload(item))
    result = start_demo(
        *paths, spec, synthetic_demo_wiring, worker_id="w", code_version=FROZEN, repo_root=REPO, allow_synthetic=True
    )
    assert result.outcome is not None and result.outcome.end_state == "done", result
    with ReadModel(paths[0]) as rm:
        [run] = rm.runs()
        [job] = rm.jobs()
        evaluations = rm.evaluations(run["run_id"])
    assert (run["stage"], run["method"], run["seed_block"], run["plan_id"]) == (
        "confirmatory",
        "random",
        0,
        made.plan_id,
    )
    assert run["plan_sha256"] == made.plan_sha256 and run["run_id"] == schedule.run_id_of(item)
    assert job["state"] == "done" and job["options"]["demo_spec"]["plan_id"] == made.plan_id
    assert len(evaluations) == 3 and all(e["synthetic"] for e in evaluations)  # synthetic: nothing was measured


def test_the_cli_plans_freezes_exports_and_queues(
    paths: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from cnnlab.tcc.operation import demo

    monkeypatch.setattr(demo, "capture_code_version", lambda label, root=None: FROZEN)
    s = tmp_path / "settings.json"
    s.write_text(
        json.dumps(
            {
                "master_seed": 20261026,
                "factory": FACTORY,
                "factory_args": {"slice": "fp16_regions", "n_candidates": 3, "sleep_s": 0.0},
            }
        ),
        encoding="utf-8",
    )
    pilot = pilot_report(tmp_path)
    base = ["campaign", "plan", "--settings", str(s), "--pilot", str(pilot), "--allow-synthetic"]
    db, art = paths

    assert cli.main(base) == 0  # no --freeze: nothing is written
    out = capsys.readouterr().out
    assert "(nothing was written)" in out and "32 runs" in out and "pending (from the pilot): none" in out
    assert not db.exists()

    # Without the pilot report the fields it decides are pending, and the plan is not frozen. The two tolerances
    # have defaults, so they are not among them.
    no_pilot = ["campaign", "plan", "--settings", str(s), "--allow-synthetic"]
    assert cli.main(no_pilot) == 0
    assert (
        "pending (from the pilot): reference_point, viable_minima, max_attempt_train_s, factory_args.epochs"
        in capsys.readouterr().out
    )
    refused = cli.main([*no_pilot, "--freeze", "--db", str(db), "--artifacts", str(art)])
    err = capsys.readouterr().err
    assert (
        refused == EXIT_REFUSED and "reference_point" in err and "viable_minima" in err and "latency_cv_max" not in err
    )
    assert cli.main([*base, "--freeze", "--db", str(db), "--artifacts", str(art)]) == 0
    frozen_out = capsys.readouterr().out
    plan_id = frozen_out.split()[1]
    assert "[frozen]" in frozen_out and "pending (from the pilot): none" in frozen_out

    exported = tmp_path / "plano"
    assert cli.main(["campaign", "export", "--db", str(db), "--plan", plan_id, "--dest", str(exported)]) == 0
    assert (exported / campaign.PLAN_HASH_NAME).read_text(encoding="utf-8").split()[0] in capsys.readouterr().out
    assert cli.main(["campaign", "export", "--db", str(db), "--plan", plan_id, "--dest", str(exported)]) == EXIT_REFUSED

    qdir = tmp_path / "fila"
    assert cli.main(["campaign", "enqueue", "--db", str(db), "--plan", plan_id, "--queue-dir", str(qdir)]) == 0
    assert "queued 32 of 32 runs" in capsys.readouterr().out
    assert cli.main(["campaign", "enqueue", "--db", str(db), "--plan", plan_id, "--queue-dir", str(qdir)]) == 0
    again = capsys.readouterr().out
    assert "queued 0 of 32 runs" in again and "32 were already there" in again

    monkeypatch.setattr(demo, "capture_code_version", lambda label, root=None: OTHER)  # another checkout
    other_q = tmp_path / "fila-other"
    assert (
        cli.main(["campaign", "enqueue", "--db", str(db), "--plan", plan_id, "--queue-dir", str(other_q)])
        == EXIT_REFUSED
    )
    assert "froze code" in capsys.readouterr().err and not (other_q / schedule.QUEUE_FILE).exists()
    assert Counter(i.settings.method for i in schedule.load_state(qdir).items) == {
        m: 8 for m in campaign.DEFAULT_METHODS
    }


def test_the_help_lists_the_campaign_command() -> None:
    assert "campaign" in cli.COMMANDS and "cnnlab-tcc campaign enqueue" in (cli.__doc__ or "")


def test_the_plan_can_name_the_admitted_slice_and_every_queue_item_carries_it(paths: tuple[Path, Path]) -> None:
    s = settings(factory_args={"slice": "admitted", "epochs": 5, "n_candidates": 3, "sleep_s": 0.0})
    made = frozen_plan(paths, s)
    assert made.content["factory_args"]["slice"] == "admitted"
    items = campaign.queue_items(campaign.read_plan(paths[0], made.plan_id))
    assert len(items) == 32 and {item.slice for _, item in items} == {"admitted"}  # the sequential runs too
    assert {item.method for _, item in items} == {"joint", "mono", "random", "sequential"}


# -- the builder's timing cache --------------------------------------------------------------------------------


def cache_file(tmp_path: Path, data: bytes = b"timing-cache-v1") -> tuple[Path, str]:
    import hashlib

    path = tmp_path / "timing.cache"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def undecided(**kw: Any) -> CampaignSettings:
    """Settings with nothing the pilot decides typed in, so a report can fill all of it."""
    return settings(
        factory_args={"slice": "fp16_regions", "n_candidates": 3, "sleep_s": 0.0},
        reference_point=None,
        viable_minima=None,
        max_attempt_train_s=None,
        **kw,
    )


def test_the_plan_has_a_builder_field_that_is_empty_by_default_and_blocks_nothing(paths: tuple[Path, Path]) -> None:
    made = frozen_plan(paths)
    assert made.content["builder"]["timing_cache"] is None and "builder.timing_cache" not in made.pending
    assert (
        "bytes" in made.content["builder"]["timing_cache_rule"]
        or "SHA-256" in made.content["builder"]["timing_cache_rule"]
    )
    recorded = frozen_plan(paths, settings(timing_cache={"adopted": False}, master_seed=2))
    assert recorded.content["builder"]["timing_cache"] == {"adopted": False, "path": None, "sha256": None}
    assert recorded.frozen and not recorded.pending


def test_an_adopted_cache_is_pending_until_the_pilot_names_the_file(paths: tuple[Path, Path], tmp_path: Path) -> None:
    s = settings(timing_cache={"adopted": True})
    with SQLiteRunStore(*paths) as store:
        with pytest.raises(ConsistencyError, match="builder.timing_cache"):
            campaign.create_plan(store, s, wiring_of(s), FROZEN, REPO, freeze=True)
    draft = campaign.create_plan(SQLiteRunStore(*paths), s, wiring_of(s), FROZEN, REPO, freeze=False)
    assert draft.pending == ("builder.timing_cache",)
    assert draft.content["builder"]["timing_cache"]["path"] is None
    s = undecided(timing_cache={"adopted": True})
    # the pilot report fills it, and a typed "adopted" agrees
    path, digest = cache_file(tmp_path)
    report = pilot_report(tmp_path, timing_cache={"adopted": True, "path": str(path), "sha256": digest})
    made = frozen_plan(paths, s, pilot=campaign.load_pilot(report))
    assert made.frozen and not made.pending
    assert made.content["builder"]["timing_cache"] == {"adopted": True, "path": str(path), "sha256": digest}
    assert made.content["pilot"]["timing_cache"] == made.content["builder"]["timing_cache"]


def test_the_pilot_cannot_contradict_what_the_settings_decided(tmp_path: Path) -> None:
    path, digest = cache_file(tmp_path)
    full = {"adopted": True, "path": str(path), "sha256": digest}
    pilot = campaign.load_pilot(pilot_report(tmp_path, timing_cache=full))
    with pytest.raises(ConsistencyError, match="timing_cache"):
        campaign.apply_pilot(undecided(timing_cache={"adopted": False}), pilot)
    with pytest.raises(ConsistencyError, match="timing_cache"):
        campaign.apply_pilot(undecided(timing_cache={**full, "sha256": "0" * 64}), pilot)
    assert campaign.apply_pilot(undecided(timing_cache=full), pilot).timing_cache == full
    assert campaign.apply_pilot(undecided(), pilot).timing_cache == full  # the pilot alone can bring it
    assert campaign.apply_pilot(
        undecided(timing_cache={"adopted": False}), campaign.load_pilot(pilot_report(tmp_path))
    ).timing_cache == {
        "adopted": False,
        "path": None,
        "sha256": None,
    }  # a report that says nothing leaves the settings as they are


def test_a_named_cache_must_be_there_with_those_bytes_when_the_plan_is_made_and_when_every_run_starts(
    paths: tuple[Path, Path], tmp_path: Path
) -> None:
    path, digest = cache_file(tmp_path)
    with SQLiteRunStore(*paths) as store:
        wrong = settings(timing_cache={"adopted": True, "path": str(path), "sha256": "1" * 64})
        with pytest.raises(ConsistencyError, match="has SHA-256"):
            campaign.create_plan(store, wrong, wiring_of(wrong), FROZEN, REPO, freeze=True)
        missing = settings(timing_cache={"adopted": True, "path": str(tmp_path / "no.cache"), "sha256": digest})
        with pytest.raises(ConsistencyError, match="is not there"):
            campaign.create_plan(store, missing, wiring_of(missing), FROZEN, REPO, freeze=True)
    s = settings(timing_cache={"adopted": True, "path": str(path), "sha256": digest})
    made = frozen_plan(paths, s)
    assert made.frozen and prepare(paths, campaign_spec(made, "joint", 0)).plan_id == made.plan_id
    path.write_bytes(b"timing-cache-v1 + entries TensorRT appended later")  # the cache moved on after the freeze
    with pytest.raises(ConsistencyError, match="would not share one cache"):
        prepare(paths, campaign_spec(made, "mono", 1))
    path.unlink()
    with pytest.raises(ConsistencyError, match="is not there"):
        prepare(paths, campaign_spec(made, "random", 2))


def test_a_relative_cache_path_is_resolved_against_the_repository(tmp_path: Path) -> None:
    from cnnlab.tcc.operation import builder_config

    path, digest = cache_file(tmp_path)
    entry = {"adopted": True, "path": path.name, "sha256": digest}
    assert builder_config.verify(entry, tmp_path) == path
    assert (
        builder_config.verify(None, tmp_path) is None
        and builder_config.verify({"adopted": False, "path": None, "sha256": None}, tmp_path) is None
    )


@pytest.mark.parametrize(
    "value",
    [
        {"path": "x"},
        {"adopted": "yes"},
        {"adopted": False, "path": "x", "sha256": "a" * 64},
        {"adopted": True, "path": "x"},
        {"adopted": True, "sha256": "a" * 64},
        {"adopted": True, "path": "x", "sha256": "ABC"},
        {"adopted": True, "extra": 1},
        {"adopted": True, "path": " ", "sha256": "a" * 64},
        "a/path",
    ],
)
def test_nonsense_cache_settings_are_refused(value: Any) -> None:
    with pytest.raises(ValueError):
        settings(timing_cache=value)
