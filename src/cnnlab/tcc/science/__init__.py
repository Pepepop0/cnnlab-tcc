"""Scientific core of TCC v2: genome v1, repair/expression and candidate identities.

No SQLite, UI or torch imports here; the evaluator and storage adapters live
elsewhere and talk through ``cnnlab.tcc.contracts``.
"""

from __future__ import annotations

from cnnlab.tcc.science.expression import (
    Expression,
    NetworkPlan,
    Repair,
    express,
    plan_network,
    pruned_width,
    repaired_genome,
)
from cnnlab.tcc.science.genome import (
    BlockGenes,
    ExpressedGenome,
    Genome,
    GenomeDomainError,
    GenomeError,
    GenomeStructureError,
    validate_expressed,
    validate_genome,
)
from cnnlab.tcc.science.record import (
    CacheContext,
    PreparedGenome,
    cache_context_hash,
    phenotype_hash,
    prepare_genome,
)
from cnnlab.tcc.science.sampling import GeneRng, sample_genome
from cnnlab.tcc.science.space import SPACE_V1, AdmissionError, AdmissionEvidence, SearchSpace

__all__ = [
    "SPACE_V1",
    "AdmissionError",
    "AdmissionEvidence",
    "BlockGenes",
    "CacheContext",
    "ExpressedGenome",
    "Expression",
    "GeneRng",
    "Genome",
    "GenomeDomainError",
    "GenomeError",
    "GenomeStructureError",
    "NetworkPlan",
    "PreparedGenome",
    "Repair",
    "SearchSpace",
    "cache_context_hash",
    "express",
    "phenotype_hash",
    "plan_network",
    "prepare_genome",
    "pruned_width",
    "repaired_genome",
    "sample_genome",
    "validate_expressed",
    "validate_genome",
]
