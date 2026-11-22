"""Genome summary of the read model: only what the database records, never completed.

The genomes below have the shape the science front stores
(`tcc-genome-expressed/v1`); they are fixtures, and the results attached to
them come from the fake evaluator and are labeled synthetic.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from cnnlab.tcc.contracts import GenomeRecord
from cnnlab.tcc.storage import NotFoundError, SQLiteRunStore, open_read_model
from cnnlab.tcc.storage.canonical import sha256_hex
from cnnlab.tcc.storage.readmodel import BLOCK_FIELDS, genome_summary, per_block
from cnnlab.tcc.storage.testing import SPACE_HASH, FakeEvaluator, make_genome, make_job, make_v1_copy, run_candidate

JOB = make_job("run-genome", method="joint", seed_block=7)


def block(precision: str, channels: int, **kw: Any) -> dict[str, Any]:
    base = {
        "channels": channels,
        "op": "standard",
        "kernel": 3,
        "n_convs": 2,
        "residual": False,
        "downsample": "avgpool",
        "precision": precision,
        "prune_pct": 0,
    }
    base.update(kw)
    return base


def expressed(precisions: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema": "tcc-genome-expressed/v1",
        "n_blocks": len(precisions),
        "stem_channels": 16,
        "stem_precision": "fp32",
        "activation": "hardswish",
        "head": "gap_fc256_linear",
        "head_precision": "fp32",
        "blocks": [block(p, 32 * (i + 1)) for i, p in enumerate(precisions)],
        "log10_lr": -2.7,
    }


def record(index: int, genome: dict[str, Any]) -> GenomeRecord:
    return GenomeRecord(
        raw_genome={**genome, "latent": index},
        expressed_genome=genome,
        repair_log=("clamped channels",) * index,
        phenotype_hash=sha256_hex(f"phenotype-g{index}"),
        cache_context_hash=sha256_hex(f"ctx-g{index}"),
        space_hash=SPACE_HASH,
    )


@pytest.fixture
def db(seeded: SQLiteRunStore) -> Path:
    seeded.create_run(JOB)
    run_candidate(seeded, JOB, "g0", "g0.a1", record(0, expressed(("fp32", "fp16", "fp16", "fp32"))), FakeEvaluator())
    seeded.register_candidate(
        JOB.run_id, "g1", record(1, expressed(("fp16",))), {"generation": 1, "origin": "crossover"}
    )
    seeded.register_candidate(JOB.run_id, "g2", make_genome(2), {"generation": 1})  # no blocks recorded
    return seeded.db_path


def test_genomes_list_what_each_candidate_is(db: Path) -> None:
    with open_read_model(db) as rm:
        first, second, bare = rm.genomes(JOB.run_id)
        with pytest.raises(NotFoundError):
            rm.genomes("no-such-run")
    assert (first["candidate_id"], first["generation"], first["n_blocks"], first["stem_channels"]) == ("g0", 0, 4, 16)
    assert (first["activation"], first["head"], first["stem_precision"], first["head_precision"]) == (
        "hardswish",
        "gap_fc256_linear",
        "fp32",
        "fp32",
    )
    assert first["genome_schema"] == "tcc-genome-expressed/v1" and first["blocks_recorded"]
    assert [tuple(b) for b in first["blocks"]] == [BLOCK_FIELDS] * 4
    assert per_block(first, "precision") == "fp32 fp16 fp16 fp32"
    assert per_block(first, "channels") == "32 64 96 128" and per_block(first, "residual") == "False False False False"
    assert (second["origin"], second["repairs"], per_block(second, "precision")) == ("crossover", 1, "fp16")

    # A genome without the recorded fields is shown as not recorded, never guessed.
    assert bare["n_blocks"] == 4 and bare["blocks"] == [] and not bare["blocks_recorded"]
    assert bare["stem_channels"] is None and bare["activation"] is None and bare["origin"] is None
    assert per_block(bare, "precision", missing="(nao registrado)") == "(nao registrado)"


def test_genomes_read_from_a_schema_v1_database(db: Path, tmp_path: Path) -> None:
    old = make_v1_copy(db, tmp_path / "v1" / "demo.sqlite")
    before = old.read_bytes()
    with open_read_model(old) as rm:
        assert rm.schema_version() == 1
        assert [per_block(g, "precision") for g in rm.genomes(JOB.run_id)][:2] == ["fp32 fp16 fp16 fp32", "fp16"]
    assert old.read_bytes() == before


def test_genome_summary_never_completes_or_validates() -> None:
    assert genome_summary(None)["blocks"] == [] and genome_summary("text")["n_blocks"] is None
    partial = genome_summary({"n_blocks": 2, "blocks": [{"precision": "fp16"}, {"channels": 8}, {"precision": "bf16"}]})
    assert [b["precision"] for b in partial["blocks"]] == ["fp16", None]  # only the active blocks
    assert (
        per_block(partial, "precision", missing="?") == "fp16 ?"
        and per_block(partial, "channels", missing="?") == "? 8"
    )
    odd = genome_summary({"n_blocks": 9, "blocks": [{"precision": "int8"}, "not a block"]})
    assert [b["precision"] for b in odd["blocks"]] == ["int8"]  # n_blocks beyond the list: show what is there
    assert genome_summary({"n_blocks": True, "blocks": "fp32"})["blocks"] == []
    changed = dataclasses.replace(record(0, expressed(("fp32",))), repair_log=())
    assert genome_summary(changed.expressed_genome)["head"] == "gap_fc256_linear"
