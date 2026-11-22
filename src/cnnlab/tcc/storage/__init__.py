"""SQLite v2 persistence for the TCC: the source of truth of the new experiment.

The core only sees the `RunStore` protocol from `cnnlab.tcc.contracts`;
`SQLiteRunStore` is the adapter injected by operation code. Read-only
analysis opens `open_reader` connections instead of this writer.
"""

from __future__ import annotations

from cnnlab.tcc.storage.artifacts import (
    ArtifactScan,
    PublishedArtifact,
    attempt_artifact_dir,
    normalize_rel_path,
    publish_bytes,
    publish_file,
    scan_artifacts,
)
from cnnlab.tcc.storage.backup import (
    ArtifactEntry,
    BackupResult,
    BackupVerification,
    RestoreReport,
    create_backup,
    rehearse_restore,
    restore_backup,
    verify_backup,
)
from cnnlab.tcc.storage.connection import open_reader, open_writer
from cnnlab.tcc.storage.errors import (
    ArtifactIntegrityError,
    AttemptStateError,
    BackupError,
    CheckpointMismatchError,
    ConsistencyError,
    IdempotencyConflictError,
    ImmutableRecordError,
    JobStateError,
    LeaseHeldError,
    LeaseLostError,
    NotFoundError,
    OfficialTestAccessError,
    SchemaVersionError,
    SerializationError,
    StorageError,
)
from cnnlab.tcc.storage.jobs import JobEvent, JobQueue, JobRecord, Lease, ReapedJob
from cnnlab.tcc.storage.plans import PlanExport, PlanRecord
from cnnlab.tcc.storage.readmodel import READ_MODEL_VERSION, ReadModel, ReadModelError, open_read_model
from cnnlab.tcc.storage.schema import LATEST_VERSION, MIGRATIONS, migrate, schema_dump
from cnnlab.tcc.storage.sqlite_store import (
    AttemptRecord,
    CandidateRecord,
    FinalistSelection,
    IntegrityReport,
    LeaseFence,
    RunHashes,
    RunHistory,
    SplitDescriptor,
    SQLiteRunStore,
)

__all__ = [
    "LATEST_VERSION",
    "READ_MODEL_VERSION",
    "ReadModel",
    "ReadModelError",
    "ArtifactEntry",
    "BackupResult",
    "BackupVerification",
    "RestoreReport",
    "create_backup",
    "rehearse_restore",
    "restore_backup",
    "verify_backup",
    "MIGRATIONS",
    "ArtifactIntegrityError",
    "ArtifactScan",
    "AttemptRecord",
    "AttemptStateError",
    "BackupError",
    "CandidateRecord",
    "CheckpointMismatchError",
    "ConsistencyError",
    "FinalistSelection",
    "IdempotencyConflictError",
    "ImmutableRecordError",
    "IntegrityReport",
    "JobEvent",
    "JobQueue",
    "JobRecord",
    "JobStateError",
    "Lease",
    "LeaseFence",
    "LeaseHeldError",
    "LeaseLostError",
    "NotFoundError",
    "OfficialTestAccessError",
    "PlanExport",
    "PlanRecord",
    "PublishedArtifact",
    "ReapedJob",
    "RunHashes",
    "RunHistory",
    "SQLiteRunStore",
    "SchemaVersionError",
    "SerializationError",
    "SplitDescriptor",
    "StorageError",
    "attempt_artifact_dir",
    "migrate",
    "normalize_rel_path",
    "open_read_model",
    "open_reader",
    "open_writer",
    "publish_bytes",
    "publish_file",
    "scan_artifacts",
    "schema_dump",
]
