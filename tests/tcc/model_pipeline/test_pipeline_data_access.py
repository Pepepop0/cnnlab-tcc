"""The official test split is unreachable from search and pilot (TCC-01 section 2, TCC-09 section 5).

A fake CIFAR source records which official files are read; no real image file
is opened by these tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from cnnlab.tcc.contracts import EvaluationRequest, GenomeRecord
from cnnlab.tcc.model_pipeline import data_access
from cnnlab.tcc.model_pipeline.data_access import (
    OFFICIAL_SIZES,
    DataAccessGrant,
    SealedSplitError,
    SplitDefinitionError,
    SplitIndices,
    TorchvisionCifar100,
    grant_finalist_access,
    grant_for_request,
    grant_search_access,
    index_fingerprint,
    load_splits,
)

SEARCH_STAGES = ("pilot", "confirmatory")


class FakeCifar:
    """Synthetic stand-in for the official files; image value encodes (file, index)."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    def read(self, official_file: str) -> tuple[torch.Tensor, torch.Tensor]:
        self.reads.append(official_file)
        n = OFFICIAL_SIZES[official_file]
        marker = torch.arange(n, dtype=torch.int64) + (1_000_000 if official_file == "test" else 0)
        images = torch.zeros((n, 1, 1, 3), dtype=torch.uint8)  # tiny: only the labels carry information
        return images, marker


def search_splits(n_train: int = 450, n_val: int = 50, n_cal: int = 32) -> list[SplitIndices]:
    train = tuple(range(0, n_train))
    return [
        SplitIndices("train", "train", train),
        SplitIndices("val", "train", tuple(range(n_train, n_train + n_val))),
        SplitIndices("calibration", "train", train[:n_cal]),
    ]


def sealed_split() -> SplitIndices:
    return SplitIndices("test", "test", tuple(range(100)))


# --- search and pilot -----------------------------------------------------------------


@pytest.mark.parametrize("stage", SEARCH_STAGES)
def test_search_grant_holds_only_train_val_calibration(stage: str) -> None:
    grant = grant_search_access("run-1", stage, search_splits())  # type: ignore[arg-type]
    assert set(grant.splits) == {"train", "val", "calibration"}
    assert not grant.may_read_official_test
    source = FakeCifar()
    loaded = load_splits(grant, source)
    assert source.reads == ["train"]
    assert {n: len(s.labels) for n, s in loaded.items()} == {"train": 450, "val": 50, "calibration": 32}
    assert all(int(s.labels.max()) < 1_000_000 for s in loaded.values())  # nothing from the test file


@pytest.mark.parametrize("stage", SEARCH_STAGES)
def test_search_grant_rejects_named_test_split(stage: str) -> None:
    with pytest.raises(SealedSplitError):
        grant_search_access("run-1", stage, [*search_splits(), sealed_split()])  # type: ignore[arg-type]


@pytest.mark.parametrize("stage", SEARCH_STAGES)
@pytest.mark.parametrize("name", ["train", "val", "calibration"])
def test_search_grant_rejects_mislabelled_test_file(stage: str, name: str) -> None:
    splits = [s for s in search_splits() if s.name != name]
    splits.append(SplitIndices(name, "test", tuple(range(10))))
    with pytest.raises(SealedSplitError):
        grant_search_access("run-1", stage, splits)  # type: ignore[arg-type]


@pytest.mark.parametrize("stage", SEARCH_STAGES)
def test_direct_construction_cannot_bypass_the_barrier(stage: str) -> None:
    splits = {s.name: s for s in [*search_splits(), sealed_split()]}
    with pytest.raises(SealedSplitError):
        DataAccessGrant("run-1", stage, splits)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DataAccessGrant("run-1", stage, {s.name: s for s in search_splits()}, finalist_list_id="list-1")  # type: ignore[arg-type]


@pytest.mark.parametrize("stage", SEARCH_STAGES)
def test_loading_test_from_search_grant_fails_before_any_read(stage: str) -> None:
    grant = grant_search_access("run-1", stage, search_splits())  # type: ignore[arg-type]
    source = FakeCifar()
    with pytest.raises(SealedSplitError):
        load_splits(grant, source, ["train", "test"])
    assert source.reads == []


def test_grant_splits_cannot_be_mutated_afterwards() -> None:
    grant = grant_search_access("run-1", "pilot", search_splits())
    with pytest.raises(TypeError):
        grant.splits["test"] = sealed_split()  # type: ignore[index]
    with pytest.raises(AttributeError):
        grant.stage = "finalist"  # type: ignore[misc]


def test_search_grant_needs_train_and_val() -> None:
    with pytest.raises(SplitDefinitionError):
        grant_search_access("run-1", "pilot", [s for s in search_splits() if s.name != "val"])
    grant = grant_search_access("run-1", "pilot", [s for s in search_splits() if s.name != "calibration"])
    assert set(grant.splits) == {"train", "val"}


def test_search_access_refuses_finalist_stage() -> None:
    with pytest.raises(ValueError):
        grant_search_access("run-1", "finalist", search_splits())


# --- split rules from the protocol -----------------------------------------------------------


def test_train_and_val_must_be_disjoint() -> None:
    train = SplitIndices("train", "train", tuple(range(100)))
    val = SplitIndices("val", "train", tuple(range(90, 120)))
    with pytest.raises(SplitDefinitionError, match="overlap"):
        grant_search_access("run-1", "pilot", [train, val])


def test_calibration_must_come_from_train_only() -> None:
    train = SplitIndices("train", "train", tuple(range(100)))
    val = SplitIndices("val", "train", tuple(range(100, 120)))
    calibration = SplitIndices("calibration", "train", (0, 1, 105))  # 105 is a validation image
    with pytest.raises(SplitDefinitionError, match="calibration"):
        grant_search_access("run-1", "pilot", [train, val, calibration])


@pytest.mark.parametrize(
    ("name", "source", "indices"),
    [
        ("train", "train", ()),
        ("val", "train", (1, 1)),
        ("val", "train", (-1,)),
        ("val", "train", (50_000,)),
        ("test", "test", (10_000,)),
        ("val", "train", (True,)),
        ("holdout", "train", (1,)),
        ("val", "extra", (1,)),
    ],
)
def test_malformed_index_sets(name: str, source: str, indices: tuple[Any, ...]) -> None:
    with pytest.raises(SplitDefinitionError):
        SplitIndices(name, source, indices)  # type: ignore[arg-type]


def test_fingerprint_is_set_based_and_file_scoped() -> None:
    assert index_fingerprint("train", [3, 1, 2]) == index_fingerprint("train", [1, 2, 3])
    assert index_fingerprint("train", [1, 2, 3]) != index_fingerprint("test", [1, 2, 3])
    assert index_fingerprint("train", [1, 2]) != index_fingerprint("train", [1, 2, 3])
    assert len(index_fingerprint("train", [0])) == 64


def test_verify_fingerprints_against_registered_values() -> None:
    grant = grant_search_access("run-1", "pilot", search_splits())
    registered = {**grant.fingerprints(), "test": "f" * 64}
    grant.verify_fingerprints(registered)  # extra test entry is ignored, never loaded
    with pytest.raises(SplitDefinitionError):
        grant.verify_fingerprints({**registered, "val": "0" * 64})
    with pytest.raises(SplitDefinitionError):
        grant.verify_fingerprints({"train": registered["train"]})


# --- finalist -----------------------------------------------------------------------------------


def test_finalist_grant_needs_frozen_list_id() -> None:
    with pytest.raises(SealedSplitError):
        grant_finalist_access("run-f", "", [*search_splits(), sealed_split()])
    with pytest.raises(SealedSplitError):
        DataAccessGrant("run-f", "finalist", {"test": sealed_split()})


def test_finalist_grant_reads_the_official_test_file() -> None:
    grant = grant_finalist_access("run-f", "list-1", [*search_splits(), sealed_split()])
    assert grant.may_read_official_test
    source = FakeCifar()
    loaded = load_splits(grant, source, ["test"])
    assert source.reads == ["test"]
    assert int(loaded["test"].labels.min()) >= 1_000_000
    assert loaded["test"].fingerprint == sealed_split().fingerprint


def test_finalist_test_must_index_the_test_file() -> None:
    with pytest.raises(SplitDefinitionError):
        grant_finalist_access("run-f", "list-1", [SplitIndices("test", "train", (1, 2))])


def test_wrong_file_size_is_rejected() -> None:
    class Short(FakeCifar):
        def read(self, official_file: str) -> tuple[torch.Tensor, torch.Tensor]:
            images, labels = super().read(official_file)
            return images[:-1], labels[:-1]

    grant = grant_search_access("run-1", "pilot", search_splits())
    with pytest.raises(SplitDefinitionError):
        load_splits(grant, Short())


# --- evaluation request mapping -------------------------------------------------------------------


def request(stage: str) -> EvaluationRequest:
    genome = GenomeRecord({}, {}, (), "a" * 64, "b" * 64, "c" * 64)
    return EvaluationRequest("run-1", "cand-1", "att-1", stage, genome, 1, 2, "ds", "env", "proto", "plan", Path("art"))  # type: ignore[arg-type]


@pytest.mark.parametrize("stage", SEARCH_STAGES)
def test_request_for_search_stage_never_gets_test(stage: str) -> None:
    grant = grant_for_request(request(stage), search_splits())
    assert grant.stage == stage and not grant.may_read_official_test
    with pytest.raises(SealedSplitError):
        grant_for_request(request(stage), [*search_splits(), sealed_split()])
    with pytest.raises(SealedSplitError):  # a finalist list id does not unlock a search request
        grant_for_request(request(stage), [*search_splits(), sealed_split()], finalist_list_id="list-1")


def test_finalist_request_needs_list_id() -> None:
    with pytest.raises(SealedSplitError):
        grant_for_request(request("finalist"), [sealed_split()])
    assert grant_for_request(request("finalist"), [sealed_split()], "list-1").may_read_official_test


# --- torchvision reader, without touching files -------------------------------------------------------


def test_torchvision_reader_never_downloads_and_maps_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    class FakeTorchvision:
        def __init__(self, root: str, train: bool, download: bool) -> None:
            calls.append({"train": train, "download": download})
            n = OFFICIAL_SIZES["train" if train else "test"]
            import numpy as np

            self.data = np.zeros((n, 32, 32, 3), dtype=np.uint8)
            self.targets = [0] * n

    import torchvision.datasets

    monkeypatch.setattr(torchvision.datasets, "CIFAR100", FakeTorchvision)
    reader = TorchvisionCifar100(tmp_path)
    grant = grant_search_access("run-1", "pilot", search_splits())
    load_splits(grant, reader)
    load_splits(grant, reader)  # cached: no second read
    assert calls == [{"train": True, "download": False}]


def test_module_has_no_storage_or_ui_imports() -> None:
    source = Path(data_access.__file__).read_text(encoding="utf-8")
    for forbidden in ("sqlite3", "cnnlab.tcc.storage", "textual", "streamlit"):
        assert forbidden not in source
