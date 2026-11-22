"""Pilot-demo data: a fixed class-balanced subset of the 45k train split plus the 5k validation split.

Reading goes through :mod:`data_access`: the grant is a ``pilot`` grant over
the official **train** file only, so this module cannot read the official
test file. The manifest freezes split policy, subset size and seed, and the
fingerprints of the index sets; it is part of the cache context.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cnnlab.tcc.model_pipeline.data_access import (
    CifarSource,
    SplitIndices,
    grant_search_access,
    load_splits,
)
from cnnlab.tcc.model_pipeline.splits import DemoSplitSpec, poc_stratified_split, stratified_subset
from cnnlab.tcc.model_pipeline.trainer import normalize_images
from cnnlab.tcc.science.hashing import digest


@dataclass(frozen=True)
class DemoData:
    """Normalized tensors on the target device plus the frozen manifest."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    manifest: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        """Content identity of the data slice, used as ``split_fingerprint`` in the cache context."""
        return digest("tcc.demo-data/v1", self.manifest)

    def to(self, device: torch.device) -> DemoData:
        """Same data and manifest with the tensors on ``device``."""
        return DemoData(
            self.x_train.to(device),
            self.y_train.to(device),
            self.x_val.to(device),
            self.y_val.to(device),
            self.manifest,
        )


def prepare_demo_data(source: CifarSource, spec: DemoSplitSpec, run_id: str, device: torch.device) -> DemoData:
    """Load the subset and validation split through a pilot grant; nothing from the test file."""
    _, train_labels = source.read("train")
    train_all, val_idx = poc_stratified_split(train_labels, spec.val_fraction, spec.split_seed)
    full = spec.train_per_class is None  # no subset: the whole split, in the order of the POC split
    subset = (
        list(train_all) if full else stratified_subset(train_all, train_labels, spec.train_per_class, spec.subset_seed)
    )
    grant = grant_search_access(
        run_id,
        "pilot",
        [
            SplitIndices("train", "train", tuple(subset)),
            SplitIndices("val", "train", tuple(val_idx)),
        ],
    )
    loaded = load_splits(grant, source)
    train, val = loaded["train"], loaded["val"]
    manifest = {
        "schema": "tcc-demo-data/v1",
        "spec": spec.definition(),
        "stage": "pilot",
        "n_train_full_split": len(train_all),
        "n_train_subset": len(subset),
        "n_val": len(val_idx),
        "fingerprints": grant.fingerprints(),
        "official_test_read": False,
    }
    if full:  # added only then, so the identity of every subset run stays as it was
        manifest["train_is_full_split"] = True
    return DemoData(
        x_train=normalize_images(train.images_u8.to(device)),
        y_train=train.labels.to(device),
        x_val=normalize_images(val.images_u8.to(device)),
        y_val=val.labels.to(device),
        manifest=manifest,
    )


def write_val_file(data: DemoData, path: Path) -> str:
    """Validation tensors for the isolated benchmark process (npz, float32 NCHW); returns SHA-256."""
    import numpy as np

    buffer = io.BytesIO()
    np.savez(buffer, x=data.x_val.detach().cpu().numpy().astype("float32"), y=data.y_val.detach().cpu().numpy())
    payload = buffer.getvalue()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return hashlib.sha256(payload).hexdigest()
