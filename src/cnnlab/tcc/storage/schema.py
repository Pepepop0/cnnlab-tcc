"""Versioned SQLite v2 schema and its migration runner.

Rules that keep the history auditable:

* a migration is never edited after release; its SHA-256 is stored in
  `schema_migrations` and a mismatch refuses to open the database;
* each migration runs inside one `BEGIN IMMEDIATE` transaction, so a failure
  leaves the previous version intact;
* a database newer than the code is refused instead of read as an older shape;
* scientific tables are append-only through triggers; plans are editable only
  while `draft`; attempts move once from `running` to a terminal status.

Constraints mirror the API checks in `sqlite_store.py` so that raw SQL (an
analysis notebook, a buggy script) cannot bypass the invariants either.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from cnnlab.tcc.storage.canonical import format_utc, sha256_hex, utc_now
from cnnlab.tcc.storage.errors import SchemaVersionError

_TS_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]*Z"


def _ts(col: str, *, nullable: bool = False) -> str:
    cond = f"{col} GLOB '{_TS_GLOB}'"
    return f"CHECK ({col} IS NULL OR {cond})" if nullable else f"CHECK ({cond})"


def _sha(col: str, *, nullable: bool = False) -> str:
    cond = f"(length({col}) = 64 AND {col} NOT GLOB '*[^0-9a-f]*')"
    return f"CHECK ({col} IS NULL OR {cond})" if nullable else f"CHECK {cond}"


def _nonempty(col: str) -> str:
    return f"CHECK (length({col}) > 0)"


def _json(col: str, *, nullable: bool = False) -> str:
    return f"CHECK ({col} IS NULL OR json_valid({col}))" if nullable else f"CHECK (json_valid({col}))"


def _raise(message: str) -> str:
    return f"SELECT RAISE(ABORT, '{message}');"


_REL_PATH_CHECK = (
    "CHECK ("
    "rel_path NOT GLOB '/*' "
    "AND instr(rel_path, '\\') = 0 "
    "AND instr(rel_path, ':') = 0 "
    "AND rel_path NOT GLOB '*//*' "
    "AND ('/' || rel_path || '/') NOT GLOB '*/../*' "
    "AND ('/' || rel_path || '/') NOT GLOB '*/./*'"
    ")"
)

# Tables whose rows never change or disappear once written.
APPEND_ONLY_TABLES: tuple[str, ...] = (
    "protocol_versions",
    "environments",
    "datasets",
    "dataset_splits",
    "finalist_lists",
    "finalist_selections",
    "runs",
    "candidates",
    "measurements",
    "artifacts",
    "gpu_segments",
    "checkpoints",
    "checkpoint_members",
    "front_snapshots",
    "front_members",
    "test_evaluations",
)


def _append_only_triggers() -> list[str]:
    out: list[str] = []
    for table in APPEND_ONLY_TABLES:
        out.append(
            f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} BEGIN {_raise(f'{table} is append-only')} END"
        )
        out.append(
            f"CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table} BEGIN {_raise(f'{table} is append-only')} END"
        )
    return out


_V1_TABLES: list[str] = [
    f"""
    CREATE TABLE protocol_versions (
        protocol_id     TEXT PRIMARY KEY {_nonempty("protocol_id")},
        document_sha256 TEXT NOT NULL {_sha("document_sha256")},
        status          TEXT NOT NULL CHECK (status IN ('pilot', 'confirmed', 'amended')),
        predecessor_id  TEXT REFERENCES protocol_versions(protocol_id),
        justification   TEXT,
        created_at      TEXT NOT NULL {_ts("created_at")},
        CHECK (status <> 'amended' OR (predecessor_id IS NOT NULL AND justification IS NOT NULL))
    ) STRICT
    """,
    f"""
    CREATE TABLE environments (
        environment_id  TEXT PRIMARY KEY {_nonempty("environment_id")},
        manifest_json   TEXT NOT NULL {_json("manifest_json")},
        manifest_sha256 TEXT NOT NULL {_sha("manifest_sha256")},
        created_at      TEXT NOT NULL {_ts("created_at")}
    ) STRICT
    """,
    f"""
    CREATE TABLE datasets (
        dataset_id      TEXT PRIMARY KEY {_nonempty("dataset_id")},
        manifest_json   TEXT NOT NULL {_json("manifest_json")},
        manifest_sha256 TEXT NOT NULL {_sha("manifest_sha256")},
        created_at      TEXT NOT NULL {_ts("created_at")}
    ) STRICT
    """,
    f"""
    CREATE TABLE dataset_splits (
        dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
        split      TEXT NOT NULL CHECK (split IN ('train', 'val', 'test', 'calibration')),
        sha256     TEXT NOT NULL {_sha("sha256")},
        n_examples INTEGER NOT NULL CHECK (n_examples > 0),
        ref        TEXT NOT NULL {_nonempty("ref")},
        PRIMARY KEY (dataset_id, split)
    ) STRICT
    """,
    f"""
    CREATE TABLE experiment_plans (
        plan_id          TEXT PRIMARY KEY {_nonempty("plan_id")},
        protocol_id      TEXT NOT NULL REFERENCES protocol_versions(protocol_id),
        version          INTEGER NOT NULL CHECK (version >= 1),
        status           TEXT NOT NULL CHECK (status IN ('draft', 'frozen')),
        content_json     TEXT NOT NULL {_json("content_json")},
        plan_sha256      TEXT NOT NULL {_sha("plan_sha256")},
        author           TEXT NOT NULL {_nonempty("author")},
        amends_plan_id   TEXT REFERENCES experiment_plans(plan_id),
        amendment_reason TEXT,
        created_at       TEXT NOT NULL {_ts("created_at")},
        updated_at       TEXT NOT NULL {_ts("updated_at")},
        frozen_at        TEXT {_ts("frozen_at", nullable=True)},
        UNIQUE (protocol_id, version),
        CHECK ((status = 'frozen') = (frozen_at IS NOT NULL)),
        CHECK ((amends_plan_id IS NULL) = (amendment_reason IS NULL))
    ) STRICT
    """,
    f"""
    CREATE TABLE finalist_lists (
        list_id      TEXT PRIMARY KEY {_nonempty("list_id")},
        protocol_id  TEXT NOT NULL REFERENCES protocol_versions(protocol_id),
        plan_id      TEXT NOT NULL REFERENCES experiment_plans(plan_id),
        n_selections INTEGER NOT NULL CHECK (n_selections >= 1),
        list_sha256  TEXT NOT NULL {_sha("list_sha256")},
        author       TEXT NOT NULL {_nonempty("author")},
        frozen_at    TEXT NOT NULL {_ts("frozen_at")}
    ) STRICT
    """,
    f"""
    CREATE TABLE runs (
        run_id           TEXT PRIMARY KEY {_nonempty("run_id")},
        protocol_id      TEXT NOT NULL REFERENCES protocol_versions(protocol_id),
        protocol_sha256  TEXT NOT NULL {_sha("protocol_sha256")},
        plan_id          TEXT NOT NULL REFERENCES experiment_plans(plan_id),
        plan_sha256      TEXT NOT NULL {_sha("plan_sha256")},
        dataset_id       TEXT NOT NULL REFERENCES datasets(dataset_id),
        environment_id   TEXT NOT NULL REFERENCES environments(environment_id),
        space_hash       TEXT NOT NULL {_nonempty("space_hash")},
        method           TEXT NOT NULL CHECK (method IN ('joint', 'mono', 'random', 'sequential')),
        stage            TEXT NOT NULL CHECK (stage IN ('pilot', 'confirmatory', 'finalist')),
        seed_block       INTEGER NOT NULL,
        search_seed      INTEGER NOT NULL,
        train_seed       INTEGER NOT NULL,
        calibration_seed INTEGER NOT NULL,
        budget_gpu_s     REAL NOT NULL CHECK (budget_gpu_s > 0),
        finalist_list_id TEXT REFERENCES finalist_lists(list_id),
        job_spec_json    TEXT NOT NULL {_json("job_spec_json")},
        job_spec_sha256  TEXT NOT NULL {_sha("job_spec_sha256")},
        created_at       TEXT NOT NULL {_ts("created_at")},
        CHECK ((stage = 'finalist') = (finalist_list_id IS NOT NULL)),
        CHECK (stage <> 'confirmatory' OR budget_gpu_s = 19800.0)
    ) STRICT
    """,
    f"""
    CREATE TABLE candidates (
        candidate_id       TEXT PRIMARY KEY {_nonempty("candidate_id")},
        run_id             TEXT NOT NULL REFERENCES runs(run_id),
        seq                INTEGER NOT NULL CHECK (seq >= 1),
        generation         INTEGER CHECK (generation IS NULL OR generation >= 0),
        genome_json        TEXT NOT NULL {_json("genome_json")},
        phenotype_hash     TEXT NOT NULL {_nonempty("phenotype_hash")},
        cache_context_hash TEXT NOT NULL {_nonempty("cache_context_hash")},
        space_hash         TEXT NOT NULL {_nonempty("space_hash")},
        metadata_json      TEXT NOT NULL {_json("metadata_json")},
        record_sha256      TEXT NOT NULL {_sha("record_sha256")},
        created_at         TEXT NOT NULL {_ts("created_at")},
        UNIQUE (run_id, candidate_id),
        UNIQUE (run_id, seq)
    ) STRICT
    """,
    f"""
    CREATE TABLE finalist_selections (
        selection_id  TEXT PRIMARY KEY {_nonempty("selection_id")},
        list_id       TEXT NOT NULL REFERENCES finalist_lists(list_id),
        source_run_id TEXT NOT NULL,
        candidate_id  TEXT NOT NULL,
        role          TEXT NOT NULL CHECK (role IN ('menor_E', 'equilibrio', 'H1')),
        criterion     TEXT NOT NULL {_nonempty("criterion")},
        FOREIGN KEY (source_run_id, candidate_id) REFERENCES candidates(run_id, candidate_id),
        UNIQUE (list_id, source_run_id, role)
    ) STRICT
    """,
    f"""
    CREATE TABLE evaluation_attempts (
        attempt_id        TEXT PRIMARY KEY {_nonempty("attempt_id")},
        run_id            TEXT NOT NULL,
        candidate_id      TEXT NOT NULL,
        attempt_number    INTEGER NOT NULL CHECK (attempt_number >= 1),
        status            TEXT NOT NULL CHECK (status IN ('running', 'valid', 'invalid', 'interrupted')),
        source_attempt_id TEXT,
        gpu_seconds       REAL CHECK (gpu_seconds IS NULL OR gpu_seconds >= 0),
        failure_stage     TEXT,
        failure_code      TEXT,
        failure_detail    TEXT,
        durations_json    TEXT {_json("durations_json", nullable=True)},
        quality_json      TEXT {_json("quality_json", nullable=True)},
        result_json       TEXT {_json("result_json", nullable=True)},
        result_sha256     TEXT {_sha("result_sha256", nullable=True)},
        recovered         INTEGER NOT NULL DEFAULT 0 CHECK (recovered IN (0, 1)),
        started_at        TEXT NOT NULL {_ts("started_at")},
        committed_at      TEXT {_ts("committed_at", nullable=True)},
        commit_seq        INTEGER UNIQUE CHECK (commit_seq IS NULL OR commit_seq >= 1),
        FOREIGN KEY (run_id, candidate_id) REFERENCES candidates(run_id, candidate_id),
        FOREIGN KEY (run_id, source_attempt_id) REFERENCES evaluation_attempts(run_id, attempt_id),
        UNIQUE (run_id, attempt_id),
        UNIQUE (candidate_id, attempt_number),
        CHECK ((status = 'running') = (committed_at IS NULL)),
        CHECK ((status = 'running') = (commit_seq IS NULL)),
        CHECK ((status = 'running') = (result_sha256 IS NULL)),
        CHECK ((result_json IS NULL) = (result_sha256 IS NULL)),
        CHECK (status = 'running' OR gpu_seconds IS NOT NULL),
        CHECK (source_attempt_id IS NULL OR status = 'valid'),
        CHECK (status <> 'valid' OR (failure_stage IS NULL AND failure_code IS NULL)),
        CHECK (status <> 'invalid' OR (failure_stage IS NOT NULL AND failure_code IS NOT NULL)),
        CHECK (recovered = 0 OR status = 'interrupted')
    ) STRICT
    """,
    f"""
    CREATE TABLE measurements (
        measurement_id TEXT PRIMARY KEY {_nonempty("measurement_id")},
        attempt_id     TEXT NOT NULL UNIQUE REFERENCES evaluation_attempts(attempt_id),
        e_fraction     REAL NOT NULL CHECK (e_fraction >= 0 AND e_fraction <= 1),
        l_ms           REAL NOT NULL CHECK (l_ms > 0),
        v_mb           REAL NOT NULL CHECK (v_mb >= 0),
        s_mb           REAL NOT NULL CHECK (s_mb > 0),
        created_at     TEXT NOT NULL {_ts("created_at")}
    ) STRICT
    """,
    f"""
    CREATE TABLE artifacts (
        artifact_id TEXT PRIMARY KEY {_nonempty("artifact_id")},
        run_id      TEXT NOT NULL,
        attempt_id  TEXT NOT NULL,
        kind        TEXT NOT NULL {_nonempty("kind")},
        rel_path    TEXT NOT NULL COLLATE NOCASE UNIQUE {_REL_PATH_CHECK},
        sha256      TEXT NOT NULL {_sha("sha256")},
        size_bytes  INTEGER NOT NULL CHECK (size_bytes >= 0),
        created_at  TEXT NOT NULL {_ts("created_at")},
        FOREIGN KEY (run_id, attempt_id) REFERENCES evaluation_attempts(run_id, attempt_id),
        UNIQUE (attempt_id, kind)
    ) STRICT
    """,
    f"""
    CREATE TABLE gpu_segments (
        segment_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT NOT NULL,
        attempt_id  TEXT NOT NULL,
        gpu_seconds REAL NOT NULL CHECK (gpu_seconds >= 0),
        source      TEXT NOT NULL CHECK (source IN ('runner', 'commit_reconcile')),
        recorded_at TEXT NOT NULL {_ts("recorded_at")},
        FOREIGN KEY (run_id, attempt_id) REFERENCES evaluation_attempts(run_id, attempt_id)
    ) STRICT
    """,
    f"""
    CREATE TABLE checkpoints (
        checkpoint_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id             TEXT NOT NULL REFERENCES runs(run_id),
        generation         INTEGER NOT NULL CHECK (generation >= 0),
        unique_evaluations INTEGER NOT NULL CHECK (unique_evaluations >= 0),
        active_gpu_seconds REAL NOT NULL CHECK (active_gpu_seconds >= 0),
        protocol_hash      TEXT NOT NULL,
        plan_hash          TEXT NOT NULL,
        space_hash         TEXT NOT NULL,
        payload_json       TEXT NOT NULL {_json("payload_json")},
        payload_sha256     TEXT NOT NULL {_sha("payload_sha256")},
        created_at         TEXT NOT NULL {_ts("created_at")},
        UNIQUE (checkpoint_id, run_id)
    ) STRICT
    """,
    """
    CREATE TABLE checkpoint_members (
        checkpoint_id INTEGER NOT NULL,
        run_id        TEXT NOT NULL,
        role          TEXT NOT NULL CHECK (role IN ('population', 'pareto_archive', 'pending')),
        position      INTEGER NOT NULL CHECK (position >= 0),
        candidate_id  TEXT NOT NULL,
        PRIMARY KEY (checkpoint_id, role, position),
        FOREIGN KEY (checkpoint_id, run_id) REFERENCES checkpoints(checkpoint_id, run_id),
        FOREIGN KEY (run_id, candidate_id) REFERENCES candidates(run_id, candidate_id)
    ) STRICT
    """,
    f"""
    CREATE TABLE front_snapshots (
        snapshot_id              INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id                   TEXT NOT NULL REFERENCES runs(run_id),
        active_gpu_seconds       REAL NOT NULL CHECK (active_gpu_seconds >= 0),
        unique_valid_evaluations INTEGER NOT NULL CHECK (unique_valid_evaluations >= 0),
        hypervolume              REAL CHECK (hypervolume IS NULL OR hypervolume >= 0),
        created_at               TEXT NOT NULL {_ts("created_at")},
        UNIQUE (snapshot_id, run_id)
    ) STRICT
    """,
    """
    CREATE TABLE front_members (
        snapshot_id  INTEGER NOT NULL,
        run_id       TEXT NOT NULL,
        candidate_id TEXT NOT NULL,
        PRIMARY KEY (snapshot_id, candidate_id),
        FOREIGN KEY (snapshot_id, run_id) REFERENCES front_snapshots(snapshot_id, run_id),
        FOREIGN KEY (run_id, candidate_id) REFERENCES candidates(run_id, candidate_id)
    ) STRICT
    """,
    f"""
    CREATE TABLE test_evaluations (
        test_id       TEXT PRIMARY KEY {_nonempty("test_id")},
        run_id        TEXT NOT NULL,
        selection_id  TEXT NOT NULL REFERENCES finalist_selections(selection_id),
        attempt_id    TEXT NOT NULL,
        engine_sha256 TEXT NOT NULL UNIQUE {_sha("engine_sha256")},
        metrics_json  TEXT NOT NULL {_json("metrics_json")},
        opened_at     TEXT NOT NULL {_ts("opened_at")},
        FOREIGN KEY (run_id, attempt_id) REFERENCES evaluation_attempts(run_id, attempt_id)
    ) STRICT
    """,
]

_V1_INDEXES: list[str] = [
    "CREATE UNIQUE INDEX runs_one_per_cell ON runs(plan_id, stage, method, seed_block) WHERE stage <> 'finalist'",
    "CREATE INDEX candidates_by_context ON candidates(run_id, cache_context_hash)",
    "CREATE UNIQUE INDEX attempts_one_running_per_candidate ON evaluation_attempts(candidate_id) "
    "WHERE status = 'running'",
    "CREATE INDEX attempts_by_run_status ON evaluation_attempts(run_id, status)",
    "CREATE INDEX gpu_segments_by_attempt ON gpu_segments(attempt_id)",
    "CREATE INDEX gpu_segments_by_run ON gpu_segments(run_id)",
    "CREATE INDEX checkpoints_by_run ON checkpoints(run_id, checkpoint_id)",
    "CREATE INDEX front_snapshots_by_run ON front_snapshots(run_id, snapshot_id)",
    "CREATE INDEX artifacts_by_attempt ON artifacts(attempt_id)",
]

_V1_TRIGGERS: list[str] = [
    # Plans: born draft; editable only while draft; identity never changes.
    f"""
    CREATE TRIGGER plans_born_draft BEFORE INSERT ON experiment_plans
    WHEN NEW.status <> 'draft'
    BEGIN {_raise("plans are created as draft")} END
    """,
    f"""
    CREATE TRIGGER plans_amend_frozen_only BEFORE INSERT ON experiment_plans
    WHEN NEW.amends_plan_id IS NOT NULL
     AND (SELECT status FROM experiment_plans WHERE plan_id = NEW.amends_plan_id) IS NOT 'frozen'
    BEGIN {_raise("only a frozen plan can be amended")} END
    """,
    f"""
    CREATE TRIGGER plans_frozen_immutable BEFORE UPDATE ON experiment_plans
    WHEN OLD.status = 'frozen'
    BEGIN {_raise("frozen plan is immutable")} END
    """,
    f"""
    CREATE TRIGGER plans_identity_immutable BEFORE UPDATE ON experiment_plans
    WHEN NEW.plan_id IS NOT OLD.plan_id
      OR NEW.protocol_id IS NOT OLD.protocol_id
      OR NEW.version IS NOT OLD.version
      OR NEW.author IS NOT OLD.author
      OR NEW.amends_plan_id IS NOT OLD.amends_plan_id
      OR NEW.amendment_reason IS NOT OLD.amendment_reason
      OR NEW.created_at IS NOT OLD.created_at
    BEGIN {_raise("plan identity is immutable")} END
    """,
    f"""
    CREATE TRIGGER plans_frozen_no_delete BEFORE DELETE ON experiment_plans
    WHEN OLD.status = 'frozen'
    BEGIN {_raise("frozen plan is immutable")} END
    """,
    # Runs reference a frozen plan and snapshot its hash and the protocol hash.
    f"""
    CREATE TRIGGER runs_require_frozen_plan BEFORE INSERT ON runs
    WHEN (SELECT status FROM experiment_plans WHERE plan_id = NEW.plan_id) IS NOT 'frozen'
      OR (SELECT plan_sha256 FROM experiment_plans WHERE plan_id = NEW.plan_id) IS NOT NEW.plan_sha256
      OR (SELECT protocol_id FROM experiment_plans WHERE plan_id = NEW.plan_id) IS NOT NEW.protocol_id
      OR (SELECT document_sha256 FROM protocol_versions WHERE protocol_id = NEW.protocol_id)
         IS NOT NEW.protocol_sha256
    BEGIN {_raise("run requires a frozen plan of the same protocol with matching hashes")} END
    """,
    f"""
    CREATE TRIGGER candidates_same_space BEFORE INSERT ON candidates
    WHEN EXISTS (SELECT 1 FROM runs WHERE run_id = NEW.run_id AND space_hash IS NOT NEW.space_hash)
    BEGIN {_raise("candidate space_hash differs from run space_hash")} END
    """,
    # Attempts: born running, one terminal transition, identity fixed, no delete.
    f"""
    CREATE TRIGGER attempts_born_running BEFORE INSERT ON evaluation_attempts
    WHEN NEW.status <> 'running'
    BEGIN {_raise("attempts are created as running")} END
    """,
    f"""
    CREATE TRIGGER attempts_terminal_immutable BEFORE UPDATE ON evaluation_attempts
    WHEN OLD.status <> 'running'
    BEGIN {_raise("terminal attempt is immutable")} END
    """,
    f"""
    CREATE TRIGGER attempts_identity_immutable BEFORE UPDATE ON evaluation_attempts
    WHEN NEW.attempt_id IS NOT OLD.attempt_id
      OR NEW.run_id IS NOT OLD.run_id
      OR NEW.candidate_id IS NOT OLD.candidate_id
      OR NEW.attempt_number IS NOT OLD.attempt_number
      OR NEW.started_at IS NOT OLD.started_at
    BEGIN {_raise("attempt identity is immutable")} END
    """,
    f"""
    CREATE TRIGGER attempts_no_delete BEFORE DELETE ON evaluation_attempts
    BEGIN {_raise("evaluation_attempts is append-only")} END
    """,
    f"""
    CREATE TRIGGER measurements_valid_original_only BEFORE INSERT ON measurements
    WHEN (SELECT status FROM evaluation_attempts WHERE attempt_id = NEW.attempt_id) IS NOT 'valid'
      OR (SELECT source_attempt_id FROM evaluation_attempts WHERE attempt_id = NEW.attempt_id)
         IS NOT NULL
    BEGIN {_raise("measurements exist only for valid, non-cached attempts")} END
    """,
    f"""
    CREATE TRIGGER gpu_segments_running_only BEFORE INSERT ON gpu_segments
    WHEN (SELECT status FROM evaluation_attempts WHERE attempt_id = NEW.attempt_id) IS NOT 'running'
    BEGIN {_raise("gpu time can only be added to a running attempt")} END
    """,
    f"""
    CREATE TRIGGER checkpoints_match_run BEFORE INSERT ON checkpoints
    WHEN (SELECT protocol_sha256 FROM runs WHERE run_id = NEW.run_id) IS NOT NEW.protocol_hash
      OR (SELECT plan_sha256 FROM runs WHERE run_id = NEW.run_id) IS NOT NEW.plan_hash
      OR (SELECT space_hash FROM runs WHERE run_id = NEW.run_id) IS NOT NEW.space_hash
    BEGIN {_raise("checkpoint hashes differ from run")} END
    """,
    f"""
    CREATE TRIGGER front_members_valid_only BEFORE INSERT ON front_members
    WHEN NOT EXISTS (
        SELECT 1 FROM evaluation_attempts
        WHERE candidate_id = NEW.candidate_id AND run_id = NEW.run_id AND status = 'valid'
    )
    BEGIN {_raise("front members need a valid attempt")} END
    """,
    # Finalists come from confirmatory runs; list size is fixed when frozen.
    f"""
    CREATE TRIGGER finalist_lists_frozen_plan BEFORE INSERT ON finalist_lists
    WHEN (SELECT status FROM experiment_plans WHERE plan_id = NEW.plan_id) IS NOT 'frozen'
    BEGIN {_raise("finalist list requires a frozen plan")} END
    """,
    f"""
    CREATE TRIGGER finalist_selections_rules BEFORE INSERT ON finalist_selections
    WHEN (SELECT stage FROM runs WHERE run_id = NEW.source_run_id) IS NOT 'confirmatory'
      OR (SELECT plan_id FROM runs WHERE run_id = NEW.source_run_id)
         IS NOT (SELECT plan_id FROM finalist_lists WHERE list_id = NEW.list_id)
      OR (SELECT count(*) FROM finalist_selections WHERE list_id = NEW.list_id)
         >= (SELECT n_selections FROM finalist_lists WHERE list_id = NEW.list_id)
      OR NOT EXISTS (
          SELECT 1 FROM evaluation_attempts
          WHERE candidate_id = NEW.candidate_id AND run_id = NEW.source_run_id AND status = 'valid'
      )
      OR (NEW.role = 'H1' AND (SELECT method FROM runs WHERE run_id = NEW.source_run_id) IS NOT 'joint')
    BEGIN {_raise("invalid finalist selection")} END
    """,
    # Official test: finalist run, selection from its frozen list, valid engine, once per engine.
    f"""
    CREATE TRIGGER test_evaluations_finalist_only BEFORE INSERT ON test_evaluations
    WHEN (SELECT stage FROM runs WHERE run_id = NEW.run_id) IS NOT 'finalist'
      OR (SELECT list_id FROM finalist_selections WHERE selection_id = NEW.selection_id)
         IS NOT (SELECT finalist_list_id FROM runs WHERE run_id = NEW.run_id)
      OR (SELECT status FROM evaluation_attempts WHERE attempt_id = NEW.attempt_id) IS NOT 'valid'
      OR NOT EXISTS (
          SELECT 1 FROM artifacts
          WHERE attempt_id = NEW.attempt_id AND sha256 = NEW.engine_sha256
      )
    BEGIN {_raise("official test only for finalist runs with a frozen list and a valid engine")} END
    """,
]


# --- migration 2: durable job queue, single-writer lease, GPU segment audit ---
#
# A job is the operational envelope of one run: it waits in the queue, is
# claimed by exactly one worker under a lease (token + wall-clock expiry),
# kept alive by heartbeats from the worker process itself, cancelled
# cooperatively, and becomes `interrupted` only after its lease expired and a
# claimer observed that inside a write transaction. Scientific tables keep
# their v1 shape; the links below only attribute attempts and GPU segments to
# the job/lease that produced them.

JOB_STATES: tuple[str, ...] = ("queued", "running", "interrupted", "done", "failed", "cancelled")
JOB_ACTIVE_STATES: tuple[str, ...] = ("queued", "running", "interrupted")
JOB_TERMINAL_STATES: tuple[str, ...] = ("done", "failed", "cancelled")
# The kinds migration 2 created the table with. Frozen: the SQL of migration 2 (and so its SHA-256) is built
# from this tuple, so it must never change; new kinds go to JOB_EVENT_KINDS and a later migration.
JOB_EVENT_KINDS_V2: tuple[str, ...] = (
    "enqueued",
    "claimed",
    "released",
    "lease_expired",
    "cancel_requested",
    "cancelled",
    "done",
    "failed",
)
# What the sentinel network (TCC-01 section 3) records about the measurement itself, added by migration 3:
# a measurement, a drift above the limit (with the stretch of attempts it marks) and a measurement that failed.
SENTINEL_EVENT_KINDS: tuple[str, ...] = ("sentinel_measured", "sentinel_drift", "sentinel_failed")
JOB_EVENT_KINDS: tuple[str, ...] = JOB_EVENT_KINDS_V2 + SENTINEL_EVENT_KINDS
# New append-only tables of migration 2; kept apart from APPEND_ONLY_TABLES so
# the SQL (and therefore the SHA-256) of migration 1 never changes.
V2_APPEND_ONLY_TABLES: tuple[str, ...] = ("job_events", "job_attempts", "job_gpu_segments")


def _in(values: Sequence[str]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


_V2_TABLES: list[str] = [
    f"""
    CREATE TABLE jobs (
        job_id              TEXT PRIMARY KEY {_nonempty("job_id")},
        run_id              TEXT NOT NULL REFERENCES runs(run_id),
        queue_seq           INTEGER NOT NULL UNIQUE CHECK (queue_seq >= 1),
        priority            INTEGER NOT NULL,
        options_json        TEXT NOT NULL {_json("options_json")},
        options_sha256      TEXT NOT NULL {_sha("options_sha256")},
        state               TEXT NOT NULL CHECK (state IN {_in(JOB_STATES)}),
        lease_owner         TEXT,
        lease_token         TEXT,
        lease_expires_at    TEXT {_ts("lease_expires_at", nullable=True)},
        heartbeat_at        TEXT {_ts("heartbeat_at", nullable=True)},
        heartbeat_count     INTEGER NOT NULL DEFAULT 0 CHECK (heartbeat_count >= 0),
        claim_count         INTEGER NOT NULL DEFAULT 0 CHECK (claim_count >= 0),
        cancel_requested_at TEXT {_ts("cancel_requested_at", nullable=True)},
        cancel_reason       TEXT,
        final_detail        TEXT,
        created_at          TEXT NOT NULL {_ts("created_at")},
        updated_at          TEXT NOT NULL {_ts("updated_at")},
        finished_at         TEXT {_ts("finished_at", nullable=True)},
        CHECK ((state = 'running') = (lease_token IS NOT NULL)),
        CHECK ((state = 'running') = (lease_owner IS NOT NULL)),
        CHECK ((state = 'running') = (lease_expires_at IS NOT NULL)),
        CHECK ((state IN {_in(JOB_TERMINAL_STATES)}) = (finished_at IS NOT NULL)),
        CHECK ((cancel_requested_at IS NULL) = (cancel_reason IS NULL)),
        CHECK (state <> 'cancelled' OR cancel_requested_at IS NOT NULL)
    ) STRICT
    """,
    f"""
    CREATE TABLE job_events (
        event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id      TEXT NOT NULL REFERENCES jobs(job_id),
        kind        TEXT NOT NULL CHECK (kind IN {_in(JOB_EVENT_KINDS_V2)}),
        from_state  TEXT CHECK (from_state IS NULL OR from_state IN {_in(JOB_STATES)}),
        to_state    TEXT NOT NULL CHECK (to_state IN {_in(JOB_STATES)}),
        lease_token TEXT,
        worker_id   TEXT,
        detail_json TEXT NOT NULL {_json("detail_json")},
        recorded_at TEXT NOT NULL {_ts("recorded_at")}
    ) STRICT
    """,
    """
    CREATE TABLE job_attempts (
        attempt_id  TEXT PRIMARY KEY,
        run_id      TEXT NOT NULL,
        job_id      TEXT NOT NULL REFERENCES jobs(job_id),
        lease_token TEXT NOT NULL CHECK (length(lease_token) > 0),
        FOREIGN KEY (run_id, attempt_id) REFERENCES evaluation_attempts(run_id, attempt_id)
    ) STRICT
    """,
    """
    CREATE TABLE job_gpu_segments (
        segment_id  INTEGER PRIMARY KEY REFERENCES gpu_segments(segment_id),
        job_id      TEXT NOT NULL REFERENCES jobs(job_id),
        lease_token TEXT NOT NULL CHECK (length(lease_token) > 0),
        mono_start  REAL NOT NULL,
        mono_end    REAL NOT NULL,
        CHECK (mono_end >= mono_start)
    ) STRICT
    """,
]

_V2_INDEXES: list[str] = [
    f"CREATE UNIQUE INDEX jobs_one_active_per_run ON jobs(run_id) WHERE state IN {_in(JOB_ACTIVE_STATES)}",
    # One GPU job at a time (TCC-01 §4, TCC-10): at most one row may be running.
    "CREATE UNIQUE INDEX jobs_single_running ON jobs(state) WHERE state = 'running'",
    "CREATE INDEX jobs_by_state_order ON jobs(state, priority, queue_seq)",
    "CREATE INDEX job_events_by_job ON job_events(job_id, event_id)",
    "CREATE INDEX job_attempts_by_job ON job_attempts(job_id)",
    "CREATE INDEX job_gpu_segments_by_job ON job_gpu_segments(job_id)",
]

_V2_TRIGGERS: list[str] = [
    f"""
    CREATE TRIGGER jobs_born_queued BEFORE INSERT ON jobs
    WHEN NEW.state <> 'queued' OR NEW.heartbeat_count <> 0 OR NEW.claim_count <> 0
      OR NEW.cancel_requested_at IS NOT NULL
    BEGIN {_raise("jobs are created queued, unclaimed and not cancelled")} END
    """,
    f"""
    CREATE TRIGGER jobs_terminal_immutable BEFORE UPDATE ON jobs
    WHEN OLD.state IN {_in(JOB_TERMINAL_STATES)}
    BEGIN {_raise("terminal job is immutable")} END
    """,
    f"""
    CREATE TRIGGER jobs_identity_immutable BEFORE UPDATE ON jobs
    WHEN NEW.job_id IS NOT OLD.job_id
      OR NEW.run_id IS NOT OLD.run_id
      OR NEW.queue_seq IS NOT OLD.queue_seq
      OR NEW.priority IS NOT OLD.priority
      OR NEW.options_json IS NOT OLD.options_json
      OR NEW.options_sha256 IS NOT OLD.options_sha256
      OR NEW.created_at IS NOT OLD.created_at
      OR NEW.heartbeat_count < OLD.heartbeat_count
      OR NEW.claim_count < OLD.claim_count
      OR (OLD.cancel_requested_at IS NOT NULL AND NEW.cancel_requested_at IS NOT OLD.cancel_requested_at)
    BEGIN {_raise("job identity and counters are immutable")} END
    """,
    f"""
    CREATE TRIGGER jobs_transitions BEFORE UPDATE ON jobs
    WHEN NOT (
        (OLD.state = 'queued' AND NEW.state IN ('queued', 'running', 'cancelled'))
     OR (OLD.state = 'running' AND NEW.state IN ('running', 'queued', 'interrupted', 'done', 'failed', 'cancelled'))
     OR (OLD.state = 'interrupted' AND NEW.state IN ('interrupted', 'running', 'cancelled'))
    )
    BEGIN {_raise("invalid job state transition")} END
    """,
    # A new lease token only comes from a claim (queued/interrupted -> running).
    f"""
    CREATE TRIGGER jobs_lease_token_only_on_claim BEFORE UPDATE ON jobs
    WHEN OLD.state = 'running' AND NEW.state = 'running' AND NEW.lease_token IS NOT OLD.lease_token
    BEGIN {_raise("lease token changes only through a new claim")} END
    """,
    f"""
    CREATE TRIGGER jobs_claim_counts BEFORE UPDATE ON jobs
    WHEN OLD.state <> 'running' AND NEW.state = 'running' AND NEW.claim_count <> OLD.claim_count + 1
    BEGIN {_raise("each claim increments claim_count")} END
    """,
    f"""
    CREATE TRIGGER jobs_no_delete BEFORE DELETE ON jobs
    BEGIN {_raise("jobs is append-only")} END
    """,
    f"""
    CREATE TRIGGER job_attempts_same_run BEFORE INSERT ON job_attempts
    WHEN (SELECT run_id FROM jobs WHERE job_id = NEW.job_id) IS NOT NEW.run_id
    BEGIN {_raise("attempt and job belong to different runs")} END
    """,
    f"""
    CREATE TRIGGER job_gpu_segments_match BEFORE INSERT ON job_gpu_segments
    WHEN (SELECT run_id FROM gpu_segments WHERE segment_id = NEW.segment_id)
         IS NOT (SELECT run_id FROM jobs WHERE job_id = NEW.job_id)
      OR abs((SELECT gpu_seconds FROM gpu_segments WHERE segment_id = NEW.segment_id)
             - (NEW.mono_end - NEW.mono_start)) > 1e-6
    BEGIN {_raise("job segment must match its ledger segment (run and duration)")} END
    """,
]


def _v2_append_only_triggers() -> list[str]:
    out: list[str] = []
    for table in V2_APPEND_ONLY_TABLES:
        out.append(
            f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} BEGIN {_raise(f'{table} is append-only')} END"
        )
        out.append(
            f"CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table} BEGIN {_raise(f'{table} is append-only')} END"
        )
    return out


# Migration 3: job_events takes the sentinel kinds. SQLite cannot change a CHECK in place, so the table is
# rebuilt: every row keeps its event_id (the AUTOINCREMENT counter continues after the last one), the index and
# the two append-only triggers are recreated, and nothing else of the database is touched.
_V3_STATEMENTS: list[str] = [
    "ALTER TABLE job_events RENAME TO job_events_v2",
    f"""
    CREATE TABLE job_events (
        event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id      TEXT NOT NULL REFERENCES jobs(job_id),
        kind        TEXT NOT NULL CHECK (kind IN {_in(JOB_EVENT_KINDS)}),
        from_state  TEXT CHECK (from_state IS NULL OR from_state IN {_in(JOB_STATES)}),
        to_state    TEXT NOT NULL CHECK (to_state IN {_in(JOB_STATES)}),
        lease_token TEXT,
        worker_id   TEXT,
        detail_json TEXT NOT NULL {_json("detail_json")},
        recorded_at TEXT NOT NULL {_ts("recorded_at")}
    ) STRICT
    """,
    "INSERT INTO job_events (event_id, job_id, kind, from_state, to_state, lease_token, worker_id, detail_json, "
    "recorded_at) SELECT event_id, job_id, kind, from_state, to_state, lease_token, worker_id, detail_json, "
    "recorded_at FROM job_events_v2 ORDER BY event_id",
    "DROP TABLE job_events_v2",
    "CREATE INDEX job_events_by_job ON job_events(job_id, event_id)",
    f"CREATE TRIGGER job_events_no_update BEFORE UPDATE ON job_events BEGIN {_raise('job_events is append-only')} END",
    f"CREATE TRIGGER job_events_no_delete BEFORE DELETE ON job_events BEGIN {_raise('job_events is append-only')} END",
]


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]

    @property
    def sha256(self) -> str:
        normalized = "\n;\n".join(" ".join(s.split()) for s in self.statements)
        return sha256_hex(normalized)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="initial_v2_schema",
        statements=tuple(_V1_TABLES + _V1_INDEXES + _V1_TRIGGERS + _append_only_triggers()),
    ),
    Migration(
        version=2,
        name="operation_job_queue_lease",
        statements=tuple(_V2_TABLES + _V2_INDEXES + _V2_TRIGGERS + _v2_append_only_triggers()),
    ),
    Migration(version=3, name="job_events_sentinel_kinds", statements=tuple(_V3_STATEMENTS)),
)
LATEST_VERSION: int = MIGRATIONS[-1].version

_BOOTSTRAP = f"""
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY CHECK (version >= 1),
    name       TEXT NOT NULL,
    sql_sha256 TEXT NOT NULL {_sha("sql_sha256")},
    applied_at TEXT NOT NULL {_ts("applied_at")}
) STRICT
"""


def _validate_sequence(migrations: Sequence[Migration]) -> None:
    expected = list(range(1, len(migrations) + 1))
    if [m.version for m in migrations] != expected:
        raise SchemaVersionError("migrations must be numbered 1..N without gaps")


def applied_migrations(conn: sqlite3.Connection) -> dict[int, str]:
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'").fetchone()
    if exists is None:
        return {}
    return {v: h for v, h in conn.execute("SELECT version, sql_sha256 FROM schema_migrations")}


def current_version(conn: sqlite3.Connection) -> int:
    applied = applied_migrations(conn)
    return max(applied) if applied else 0


def _check_known(applied: dict[int, str], migrations: Sequence[Migration]) -> None:
    known = {m.version: m for m in migrations}
    for version, digest in sorted(applied.items()):
        if version not in known:
            raise SchemaVersionError(f"database has migration {version}, newer than this code (max {len(migrations)})")
        if known[version].sha256 != digest:
            raise SchemaVersionError(f"migration {version} ({known[version].name}) differs from the applied SQL")


def migrate(conn: sqlite3.Connection, migrations: Sequence[Migration] = MIGRATIONS) -> int:
    """Apply pending migrations; re-running is a verified no-op. Returns the version."""
    _validate_sequence(migrations)
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(_BOOTSTRAP)
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    _check_known(applied_migrations(conn), migrations)

    for migration in migrations:
        conn.execute("BEGIN IMMEDIATE")
        try:
            applied = applied_migrations(conn)
            _check_known(applied, migrations)
            if migration.version in applied:
                conn.execute("COMMIT")
                continue
            for statement in migration.statements:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, sql_sha256, applied_at) VALUES (?, ?, ?, ?)",
                (migration.version, migration.name, migration.sha256, format_utc(utc_now())),
            )
            conn.execute(f"PRAGMA user_version = {int(migration.version)}")
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise SchemaVersionError(f"migration {migration.version} left foreign key violations: {violations[:5]}")
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return current_version(conn)


def schema_dump(conn: sqlite3.Connection) -> str:
    """Deterministic DDL text for export next to results (auditable outside the DB)."""
    rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    versions = conn.execute("SELECT version, name, sql_sha256 FROM schema_migrations ORDER BY version").fetchall()
    header = [f"-- migration {v} {n} sha256={h}" for v, n, h in versions]
    return "\n".join(header + [f"{sql.strip()};" for _, _, sql in rows]) + "\n"
