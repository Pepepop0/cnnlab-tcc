"""Phenotype and cache-context identities, packaged as the shared ``GenomeRecord``.

* ``phenotype_hash`` covers the expressed genome only, so latent slots and
  unused quantization genes cannot change it.
* ``cache_context_hash`` adds everything else that determines an evaluation
  (TCC-02 section 4). It is a *content* identity: equal hashes in two runs
  mean the same evaluation recipe, but reuse is allowed only inside one run
  and never across seeds; enforcing that is the ``RunStore``'s job
  (``find_cached(run_id, ...)``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cnnlab.tcc.contracts import GenomeRecord
from cnnlab.tcc.science.expression import Expression, express
from cnnlab.tcc.science.genome import ExpressedGenome, Genome
from cnnlab.tcc.science.hashing import canonical_json, digest, is_sha256_hex
from cnnlab.tcc.science.space import SearchSpace

PHENOTYPE_HASH_KIND = "tcc.phenotype/v1"
CACHE_CONTEXT_HASH_KIND = "tcc.cache-context/v1"


@dataclass(frozen=True, slots=True)
class CacheContext:
    """Evaluation recipe besides the phenotype. Policies are JSON-native mappings."""

    split_fingerprint: str  # hash of train/val indices and the 1,024 calibration indices
    train_seed: int
    calibration_seed: int
    train_policy: Mapping[str, Any]  # epochs, batch, optimizer, schedule, augmentation, AMP
    pruning_policy: Mapping[str, Any]  # importance criterion, fine-tune epochs, coupling rule
    quantization_policy: Mapping[str, Any]  # Q/DQ placement, QAT epochs, fixed QAT init
    code_version: str  # git commit (plus dirty flag) of the evaluator
    backend: Mapping[str, Any]  # torch/ONNX/TensorRT/CUDA/driver versions and GPU
    builder_config: Mapping[str, Any]  # TensorRT builder flags, opt level, workspace
    space_hash: str

    def __post_init__(self) -> None:
        for name in ("split_fingerprint", "code_version"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("train_seed", "calibration_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if not is_sha256_hex(self.space_hash):
            raise ValueError("space_hash must be a SHA-256 hex digest")
        canonical_json(self.definition())  # fail early on non-JSON policy values

    def definition(self) -> dict[str, Any]:
        return {
            "split_fingerprint": self.split_fingerprint,
            "train_seed": self.train_seed,
            "calibration_seed": self.calibration_seed,
            "train_policy": self.train_policy,
            "pruning_policy": self.pruning_policy,
            "quantization_policy": self.quantization_policy,
            "code_version": self.code_version,
            "backend": self.backend,
            "builder_config": self.builder_config,
            "space_hash": self.space_hash,
        }


def phenotype_hash(expressed: ExpressedGenome) -> str:
    return digest(PHENOTYPE_HASH_KIND, expressed.to_dict())


def cache_context_hash(phenotype: str, context: CacheContext) -> str:
    if not is_sha256_hex(phenotype):
        raise ValueError("phenotype must be a SHA-256 hex digest")
    return digest(CACHE_CONTEXT_HASH_KIND, {"phenotype_hash": phenotype, "context": context.definition()})


@dataclass(frozen=True, slots=True)
class PreparedGenome:
    """Contract record plus the expression it came from (repairs, invalidity)."""

    record: GenomeRecord
    expression: Expression


def prepare_genome(genome: Genome, space: SearchSpace, context: CacheContext) -> PreparedGenome:
    """Repair, express and hash one raw genome into a ``GenomeRecord``."""
    space_hash = space.space_hash()
    if context.space_hash != space_hash:
        raise ValueError("cache context was built for another search space")
    expression = express(genome, space)
    phenotype = phenotype_hash(expression.genome)
    record = GenomeRecord(
        raw_genome=genome.to_dict(),
        expressed_genome=expression.genome.to_dict(),
        repair_log=expression.repair_log,
        phenotype_hash=phenotype,
        cache_context_hash=cache_context_hash(phenotype, context),
        space_hash=space_hash,
    )
    return PreparedGenome(record=record, expression=expression)
