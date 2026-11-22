"""Typed failures of the v2 storage layer.

Callers (runner, worker, CLI) branch on these classes instead of parsing
SQLite messages; triggers remain as a backstop for raw SQL access.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base class for every storage failure."""


class SchemaVersionError(StorageError):
    """Database schema is newer than the code or diverges from a known migration."""


class SerializationError(StorageError, ValueError):
    """A value cannot be represented as canonical JSON (NaN, non-str key, object...)."""


class NotFoundError(StorageError, LookupError):
    """A referenced run, plan, candidate, attempt or reference row does not exist."""


class IdempotencyConflictError(StorageError):
    """Same identifier was written before with different bytes."""


class ImmutableRecordError(StorageError):
    """Attempt to change a frozen plan or any append-only record."""


class AttemptStateError(StorageError):
    """Attempt lifecycle violation (terminal attempt, second running attempt...)."""


class ConsistencyError(StorageError):
    """Cross-row invariant violated (space hash, ledger, cache scope, front validity...)."""


class ArtifactIntegrityError(StorageError):
    """Artifact path is unsafe, file is missing, or its SHA-256 does not match."""


class CheckpointMismatchError(StorageError):
    """Checkpoint or job spec diverges from the run's frozen protocol/plan/space."""


class OfficialTestAccessError(StorageError):
    """Official test split requested outside a finalist run with a frozen list."""


class JobStateError(StorageError):
    """Job queue lifecycle violation (unknown job, terminal job, invalid transition...)."""


class LeaseHeldError(JobStateError):
    """Another worker holds the live lease of this run's job; writing would make two writers."""


class LeaseLostError(JobStateError):
    """The caller's lease was taken over or ended; the worker must stop writing at once."""


class BackupError(StorageError):
    """Backup or restore refused (target exists, manifest/hash mismatch, unsafe path...)."""
