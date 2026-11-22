from __future__ import annotations

import dataclasses
import json
import math
import sqlite3

import pytest

from cnnlab.tcc.storage import (
    ConsistencyError,
    IdempotencyConflictError,
    ImmutableRecordError,
    SerializationError,
    SQLiteRunStore,
)
from cnnlab.tcc.storage.canonical import sha256_hex
from cnnlab.tcc.storage.testing import (
    PLAN_ID,
    PROTOCOL_ID,
    make_job,
    seed_reference_rows,
    synthetic_plan_content,
)


@pytest.fixture
def draft(store: SQLiteRunStore) -> SQLiteRunStore:
    seed_reference_rows(store, freeze=False)
    return store


def test_draft_is_editable_and_frozen_is_immutable(draft: SQLiteRunStore) -> None:
    before = draft.get_plan(PLAN_ID)
    assert before.status == "draft" and before.frozen_at is None
    content = {**synthetic_plan_content(), "tolerances": {"fp32": 1e-4}}
    edited = draft.update_plan(PLAN_ID, content)
    assert edited.plan_sha256 != before.plan_sha256

    frozen = draft.freeze_plan(PLAN_ID)
    assert frozen.status == "frozen" and frozen.frozen_at is not None
    assert frozen.plan_sha256 == edited.plan_sha256  # freezing does not change the hash
    assert draft.freeze_plan(PLAN_ID) == frozen  # idempotent

    with pytest.raises(ImmutableRecordError):
        draft.update_plan(PLAN_ID, synthetic_plan_content())
    with pytest.raises(sqlite3.IntegrityError, match="frozen plan is immutable"):
        draft._conn.execute("UPDATE experiment_plans SET content_json = '{}' WHERE plan_id = ?", (PLAN_ID,))
    with pytest.raises(sqlite3.IntegrityError, match="frozen plan is immutable"):
        draft._conn.execute("UPDATE experiment_plans SET status = 'draft', frozen_at = NULL")
    with pytest.raises(sqlite3.IntegrityError, match="frozen plan is immutable"):
        draft._conn.execute("DELETE FROM experiment_plans WHERE plan_id = ?", (PLAN_ID,))
    assert draft.get_plan(PLAN_ID) == frozen


def test_export_hash_matches_body_and_survives_freeze(draft: SQLiteRunStore) -> None:
    exported_draft = draft.export_plan(PLAN_ID)
    draft.freeze_plan(PLAN_ID)
    exported = draft.export_plan(PLAN_ID)
    assert sha256_hex(exported.body) == exported.plan_sha256
    assert exported.body == exported_draft.body
    assert exported.status == "frozen" and exported.frozen_at
    body = json.loads(exported.body)
    assert body["content"]["blocks"][3]["search_seed"] == 1003
    assert body["protocol_id"] == PROTOCOL_ID


def test_amendment_creates_linked_version(draft: SQLiteRunStore) -> None:
    with pytest.raises(ConsistencyError, match="frozen"):
        draft.create_plan(
            "plan-v2",
            PROTOCOL_ID,
            synthetic_plan_content(),
            author="t",
            amends_plan_id=PLAN_ID,
            amendment_reason="pilot result",
        )
    draft.freeze_plan(PLAN_ID)
    with pytest.raises(ValueError):
        draft.create_plan("plan-v2", PROTOCOL_ID, synthetic_plan_content(), author="t", amends_plan_id=PLAN_ID)
    amended = draft.create_plan(
        "plan-v2",
        PROTOCOL_ID,
        synthetic_plan_content(),
        author="t",
        amends_plan_id=PLAN_ID,
        amendment_reason="pilot: 8 epochs",
    )
    assert amended.version == 2 and amended.amends_plan_id == PLAN_ID and amended.status == "draft"
    assert draft.get_plan(PLAN_ID).status == "frozen"  # the original is untouched
    with pytest.raises(IdempotencyConflictError):
        draft.create_plan("plan-v2", PROTOCOL_ID, synthetic_plan_content(), author="t")


def test_run_requires_frozen_plan(draft: SQLiteRunStore) -> None:
    with pytest.raises(ConsistencyError, match="frozen"):
        draft.create_run(make_job("r-draft"))
    draft.freeze_plan(PLAN_ID)
    draft.create_run(make_job("r-draft"))
    assert draft.run_hashes("r-draft").plan_hash == draft.get_plan(PLAN_ID).plan_sha256


def test_confirmatory_job_must_match_plan_seeds(seeded: SQLiteRunStore) -> None:
    good = make_job("r-good", seed_block=5)
    with pytest.raises(ConsistencyError, match="search_seed"):
        seeded.create_run(dataclasses.replace(good, search_seed=999))
    with pytest.raises(ConsistencyError, match="train_seed"):
        seeded.create_run(dataclasses.replace(good, train_seed=999))
    with pytest.raises(ConsistencyError, match="seed_block"):
        seeded.create_run(dataclasses.replace(good, seed_block=42))
    seeded.create_run(good)


def test_confirmatory_plan_needs_eight_distinct_blocks(store: SQLiteRunStore) -> None:
    seed_reference_rows(store, freeze=False)
    content = synthetic_plan_content()
    content["blocks"] = content["blocks"][:7]
    store.update_plan(PLAN_ID, content)
    store.freeze_plan(PLAN_ID)
    with pytest.raises(ConsistencyError, match="8 blocks"):
        store.create_run(make_job("r-conf"))
    store.create_run(make_job("r-pilot", stage="pilot"))  # pilot plans are not forced to 8 blocks


def test_plan_content_validation(draft: SQLiteRunStore) -> None:
    content = synthetic_plan_content()
    content["blocks"].append(dict(content["blocks"][0]))
    with pytest.raises(ConsistencyError, match="duplicate seed_block"):
        draft.update_plan(PLAN_ID, content)
    with pytest.raises(ConsistencyError, match="methods"):
        draft.update_plan(PLAN_ID, {**synthetic_plan_content(), "methods": ["joint", "bayesian"]})
    with pytest.raises(SerializationError):
        draft.update_plan(PLAN_ID, {**synthetic_plan_content(), "tolerances": {"fp32": math.nan}})


def test_method_outside_plan_is_rejected(store: SQLiteRunStore) -> None:
    seed_reference_rows(store, freeze=False)
    store.update_plan(PLAN_ID, {**synthetic_plan_content(), "methods": ["joint", "mono"]})
    store.freeze_plan(PLAN_ID)
    with pytest.raises(ConsistencyError, match="random"):
        store.create_run(make_job("r-random", method="random"))


def test_create_run_is_idempotent_and_cells_unique(seeded: SQLiteRunStore) -> None:
    job = make_job("r-1")
    seeded.create_run(job)
    seeded.create_run(job)
    assert seeded.get_run("r-1") == job
    with pytest.raises(IdempotencyConflictError):
        seeded.create_run(dataclasses.replace(job, method="mono"))
    with pytest.raises(ConsistencyError, match="already is run"):
        seeded.create_run(dataclasses.replace(job, run_id="r-1-bis"))
    seeded.create_run(make_job("r-1-pilot", stage="pilot"))  # other stage, other cell
