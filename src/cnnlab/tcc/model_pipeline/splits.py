"""Deterministic index sets over the official CIFAR-100 **train** file (TCC-01 section 2).

* :func:`poc_stratified_split` reproduces the POC's 45k/5k stratified split
  (``cnnlab.core.data._stratified_split``, seed 20260901, 10 % validation)
  without importing the POC; a test checks both give identical indices.
* :func:`stratified_subset` draws a fixed, class-balanced subset of the 45k
  training indices. The pilot demo trains on such a subset; its size and seed
  are part of the manifest and of the cache context, so a subset result is
  never confused with a full-train result. ``DemoSplitSpec(train_per_class=None)``
  means no subset at all: the whole 45k split, in the order of the POC split.

Everything here works on labels only. The official test file is not an input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

POC_SPLIT_SEED = 20260901
POC_VAL_FRACTION = 0.1
SPLIT_POLICY = "tcc-split/poc-stratified-v1"
SUBSET_POLICY = "tcc-subset/stratified-v1"
FULL_POLICY = "tcc-subset/none-full-45k-v1"


def poc_stratified_split(
    labels: torch.Tensor, val_fraction: float = POC_VAL_FRACTION, seed: int = POC_SPLIT_SEED
) -> tuple[list[int], list[int]]:
    """Train/validation indices exactly as the POC computes them (per-class permutation, then shuffle)."""
    labels = torch.as_tensor(labels, dtype=torch.int64)
    num_classes = int(labels.max()) + 1
    gen = torch.Generator().manual_seed(seed)
    train_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    for label in range(num_classes):
        idx = (labels == label).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        idx = idx[torch.randperm(idx.numel(), generator=gen)]
        n_val = min(max(1, round(idx.numel() * val_fraction)), idx.numel() - 1)
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = torch.cat(train_parts)
    val_idx = torch.cat(val_parts)
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=gen)]
    val_idx = val_idx[torch.randperm(val_idx.numel(), generator=gen)]
    return train_idx.tolist(), val_idx.tolist()


def stratified_subset(indices: Sequence[int], labels: torch.Tensor, per_class: int, seed: int) -> list[int]:
    """``per_class`` indices of each class from ``indices``, in a seeded order.

    Raises if a class has fewer than ``per_class`` examples, instead of
    silently returning an unbalanced subset.
    """
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    labels = torch.as_tensor(labels, dtype=torch.int64)
    pool = torch.as_tensor(list(indices), dtype=torch.int64)
    gen = torch.Generator().manual_seed(seed)
    chosen: list[torch.Tensor] = []
    for label in range(int(labels.max()) + 1):
        members = pool[labels[pool] == label]
        if members.numel() < per_class:
            raise ValueError(f"class {label} has {members.numel()} examples, fewer than {per_class}")
        chosen.append(members[torch.randperm(members.numel(), generator=gen)[:per_class]])
    subset = torch.cat(chosen)
    return subset[torch.randperm(subset.numel(), generator=gen)].tolist()


@dataclass(frozen=True, slots=True)
class DemoSplitSpec:
    """Frozen parameters of the pilot-demo data slice; goes into the manifest and cache context."""

    train_per_class: int | None  # None: the full 45k split, no subset
    subset_seed: int  # ignored (and not recorded) when train_per_class is None
    split_seed: int = POC_SPLIT_SEED
    val_fraction: float = POC_VAL_FRACTION

    def definition(self) -> dict[str, Any]:
        full = self.train_per_class is None
        return {
            "split_policy": SPLIT_POLICY,
            "split_seed": self.split_seed,
            "val_fraction": self.val_fraction,
            "subset_policy": FULL_POLICY if full else SUBSET_POLICY,
            "train_per_class": self.train_per_class,
            "subset_seed": None if full else self.subset_seed,
            "source_file": "cifar100-official-train",
        }
