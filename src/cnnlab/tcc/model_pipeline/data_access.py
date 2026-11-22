"""Data-access barrier: search and pilot never receive the official CIFAR-100 test split.

TCC-01 section 2 keeps the 10,000 official test images sealed until the
finalist list is frozen; TCC-09 section 5 asks for that protection to be
tested, not only agreed. The barrier works at two levels:

* :class:`DataAccessGrant` validates itself on construction. A ``pilot`` or
  ``confirmatory`` grant accepts only ``train``/``val``/``calibration`` index
  sets, all drawn from the official **train** file (45k/5k split and the
  1,024 PTQ examples inside the 45k), so a mislabelled test split is caught by
  its source file, not only by its name. A ``finalist`` grant may add ``test``
  and needs the id of the frozen finalist list.
* :func:`load_splits` is the only function here that reads image files, and it
  asks a :class:`CifarSource` for the official test file only when the grant
  is a finalist grant that contains ``test``.

The storage side (``SQLiteRunStore.dataset_splits_for_run``) already filters
split fingerprints by run stage; this module is the evaluator-side check and
does not import SQLite.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

import torch

from cnnlab.tcc.contracts import EvaluationRequest, Stage

SplitName = Literal["train", "val", "calibration", "test"]
OfficialFile = Literal["train", "test"]

SEARCH_SPLITS: frozenset[str] = frozenset({"train", "val", "calibration"})
REQUIRED_SEARCH_SPLITS: frozenset[str] = frozenset({"train", "val"})
SEARCH_STAGES: frozenset[str] = frozenset({"pilot", "confirmatory"})
OFFICIAL_SIZES: dict[str, int] = {"train": 50_000, "test": 10_000}
FINGERPRINT_KIND = "tcc.split-indices/v1"


class SealedSplitError(PermissionError):
    """A search/pilot path tried to reach the official test split.

    Mirrors ``storage.OfficialTestAccessError`` without importing storage.
    """


class SplitDefinitionError(ValueError):
    """Index sets that break the protocol's split rules (overlap, range, source)."""


def index_fingerprint(source_file: OfficialFile, indices: Iterable[int]) -> str:
    """SHA-256 of the *set* of indices into one official file.

    Order is irrelevant (the split is a set); the source file is part of the
    hash so the same numbers in the train and test files never collide.
    """
    ordered = sorted(indices)
    header = f"{FINGERPRINT_KIND}|{source_file}|{len(ordered)}|".encode("ascii")
    body = b"".join(i.to_bytes(8, "little", signed=False) for i in ordered)
    return hashlib.sha256(header + body).hexdigest()


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """Indices of one named split inside one official CIFAR-100 file."""

    name: str
    source_file: OfficialFile
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.name not in ("train", "val", "calibration", "test"):
            raise SplitDefinitionError(f"unknown split name {self.name!r}")
        if self.source_file not in OFFICIAL_SIZES:
            raise SplitDefinitionError(f"unknown official file {self.source_file!r}")
        indices = tuple(self.indices)
        if not indices:
            raise SplitDefinitionError(f"split {self.name!r} is empty")
        if any(type(i) is not int for i in indices):
            raise SplitDefinitionError(f"split {self.name!r} indices must be int")
        size = OFFICIAL_SIZES[self.source_file]
        if min(indices) < 0 or max(indices) >= size:
            raise SplitDefinitionError(f"split {self.name!r} has indices outside 0..{size - 1}")
        if len(set(indices)) != len(indices):
            raise SplitDefinitionError(f"split {self.name!r} repeats indices")
        object.__setattr__(self, "indices", indices)

    @property
    def fingerprint(self) -> str:
        return index_fingerprint(self.source_file, self.indices)

    @property
    def is_official_test(self) -> bool:
        return self.name == "test" or self.source_file == "test"


@dataclass(frozen=True, slots=True)
class DataAccessGrant:
    """Which index sets one run may read. Invariants are checked on construction."""

    run_id: str
    stage: Stage
    splits: Mapping[str, SplitIndices]
    finalist_list_id: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        splits = dict(self.splits)
        for key, split in splits.items():
            if key != split.name:
                raise SplitDefinitionError(f"split stored under {key!r} is named {split.name!r}")
        if self.stage in SEARCH_STAGES:
            sealed = sorted(k for k, s in splits.items() if s.is_official_test)
            if sealed:
                raise SealedSplitError(f"{self.stage} run {self.run_id!r} cannot receive official test data: {sealed}")
            missing = REQUIRED_SEARCH_SPLITS - set(splits)
            if missing:
                raise SplitDefinitionError(f"search grant needs splits {sorted(missing)}")
            if self.finalist_list_id is not None:
                raise ValueError("only finalist grants carry a finalist list")
        elif self.stage == "finalist":
            if not self.finalist_list_id:
                raise SealedSplitError("finalist access needs the id of a frozen finalist list")
            if "test" in splits and splits["test"].source_file != "test":
                raise SplitDefinitionError("the finalist test split must index the official test file")
        else:
            raise ValueError(f"unknown stage {self.stage!r}")
        _check_train_file_partition(splits)
        # Read-only view: a test split cannot be slipped in after validation.
        object.__setattr__(self, "splits", MappingProxyType(splits))

    @property
    def may_read_official_test(self) -> bool:
        return self.stage == "finalist" and "test" in self.splits

    def fingerprints(self) -> dict[str, str]:
        return {name: split.fingerprint for name, split in sorted(self.splits.items())}

    def verify_fingerprints(self, expected: Mapping[str, str]) -> None:
        """Compare with fingerprints registered for the dataset (e.g. in SQLite).

        Only the splits this grant holds are compared; an expected ``test``
        fingerprint is never a reason to load test data.
        """
        actual = self.fingerprints()
        wrong = sorted(n for n in actual if n in expected and expected[n] != actual[n])
        missing = sorted(n for n in actual if n not in expected)
        if wrong or missing:
            raise SplitDefinitionError(f"fingerprint mismatch {wrong}, unregistered {missing}")


def _check_train_file_partition(splits: Mapping[str, SplitIndices]) -> None:
    """train/val/calibration come from the official train file; val is disjoint; calibration is inside train."""
    for name in SEARCH_SPLITS & set(splits):
        if splits[name].source_file != "train":
            raise SplitDefinitionError(f"split {name!r} must index the official train file")
    train = set(splits["train"].indices) if "train" in splits else set()
    if "val" in splits and train & set(splits["val"].indices):
        raise SplitDefinitionError("train and val overlap")
    if "calibration" in splits and not set(splits["calibration"].indices) <= train:
        raise SplitDefinitionError("calibration examples must come from the train split only")


def grant_search_access(run_id: str, stage: Stage, splits: Sequence[SplitIndices]) -> DataAccessGrant:
    """Grant for search and pilot runs: train, validation and optional calibration only."""
    if stage not in SEARCH_STAGES:
        raise ValueError(f"search access is for {sorted(SEARCH_STAGES)}, not {stage!r}")
    return DataAccessGrant(run_id, stage, {s.name: s for s in splits})


def grant_finalist_access(run_id: str, finalist_list_id: str, splits: Sequence[SplitIndices]) -> DataAccessGrant:
    """Grant for the finalist job; the only one that may include the official test split."""
    return DataAccessGrant(run_id, "finalist", {s.name: s for s in splits}, finalist_list_id)


def grant_for_request(
    request: EvaluationRequest,
    splits: Sequence[SplitIndices],
    finalist_list_id: str | None = None,
) -> DataAccessGrant:
    """Map an evaluation request to its grant by stage.

    ``EvaluationRequest`` has no finalist-list field, so a finalist caller
    passes it explicitly.
    """
    if request.stage == "finalist":
        if finalist_list_id is None:
            raise SealedSplitError("finalist request without a frozen finalist list id")
        return grant_finalist_access(request.run_id, finalist_list_id, splits)
    return grant_search_access(request.run_id, request.stage, splits)


# --- reading --------------------------------------------------------------------


class CifarSource(Protocol):
    """Reads one official CIFAR-100 file as ``(uint8 NHWC images, int64 labels)``."""

    def read(self, official_file: OfficialFile) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass
class TorchvisionCifar100:
    """``torchvision`` reader over an existing download; never downloads."""

    root: Path
    _cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict, repr=False)

    def read(self, official_file: OfficialFile) -> tuple[torch.Tensor, torch.Tensor]:
        if official_file not in self._cache:
            from torchvision.datasets import CIFAR100

            dataset = CIFAR100(root=str(self.root), train=official_file == "train", download=False)
            self._cache[official_file] = (
                torch.from_numpy(dataset.data),
                torch.as_tensor(dataset.targets, dtype=torch.int64),
            )
        return self._cache[official_file]


@dataclass(frozen=True, slots=True)
class LoadedSplit:
    name: str
    images_u8: torch.Tensor  # N x 32 x 32 x 3, uint8, unnormalized
    labels: torch.Tensor
    fingerprint: str


def load_splits(
    grant: DataAccessGrant, source: CifarSource, names: Iterable[str] | None = None
) -> dict[str, LoadedSplit]:
    """Read the requested splits of ``grant``; test is readable only under a finalist grant.

    Asking a search grant for ``test`` raises :class:`SealedSplitError` before
    any file is touched.
    """
    wanted = sorted(grant.splits) if names is None else sorted(set(names))
    for name in wanted:
        if name == "test" and not grant.may_read_official_test:
            raise SealedSplitError(f"{grant.stage} run {grant.run_id!r} cannot read the official test split")
        if name not in grant.splits:
            raise KeyError(f"split {name!r} is not part of this grant")
    out: dict[str, LoadedSplit] = {}
    for official_file in ("train", "test"):
        members = [n for n in wanted if grant.splits[n].source_file == official_file]
        if not members:
            continue
        if official_file == "test" and not grant.may_read_official_test:  # defence in depth
            raise SealedSplitError("official test file requested without a finalist grant")
        images, labels = source.read(official_file)
        if images.shape[0] != OFFICIAL_SIZES[official_file] or labels.shape[0] != images.shape[0]:
            raise SplitDefinitionError(
                f"official {official_file} file has {images.shape[0]} images, expected {OFFICIAL_SIZES[official_file]}"
            )
        for name in members:
            split = grant.splits[name]
            index = torch.as_tensor(split.indices, dtype=torch.int64)
            out[name] = LoadedSplit(name, images[index], labels[index], split.fingerprint)
    return out
