"""Analysis tab of the read-only page: data layer on CPU, and the tab drawn against a synthetic run."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

import cnnlab.tcc.demo_analysis as analysis
from cnnlab.tcc.demo_analysis import (
    AnalysisView,
    diversity_summary,
    evolution_rows,
    front_history_rows,
    genome_distance,
    lineage_edges,
    ranking_rows,
    similar_rows,
)

MODULE_FILE = Path(analysis.__file__)


def _genome(generation: int, phenotype: str, blocks: list[dict]) -> dict:
    return {
        "generation": generation,
        "phenotype_hash": phenotype,
        "n_blocks": len(blocks),
        "stem_channels": 32,
        "activation": "relu",
        "head": "gap_linear",
        "blocks": blocks,
        "blocks_recorded": True,
    }


def _block(channels: int, op: str = "standard", precision: str = "fp32") -> dict:
    return {
        "channels": channels,
        "op": op,
        "kernel": 3,
        "n_convs": 1,
        "residual": False,
        "downsample": "none",
        "precision": precision,
    }


def _detail(cid: str, generation: int, e: float, l: float, v: float, s: float) -> dict:
    return {
        "candidate_id": cid,
        "attempt_id": cid + ".a1",
        "generation": generation,
        "e": e,
        "l_ms": l,
        "v_mb": v,
        "s_mb": s,
        "gpu_seconds": 10.0,
        "synthetic": False,
        "accuracy": 1 - e,
        "top5_accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "torch_accuracy": 1 - e,
        "parity_disagreements": 0,
        "parameters": 1000,
        "l_p5_ms": l * 0.9,
        "l_p95_ms": l * 1.2,
        "l_iqr_ms": 0.01,
        "epoch_loss": [4.0, 3.5],
        "epoch_train_acc": [0.05, 0.1],
        "durations": {"train": 8.0, "benchmark": 2.0},
    }


@pytest.fixture
def view() -> AnalysisView:
    ids = [f"run.g000.c000{i}" for i in range(4)] + ["run.g001.c0004", "run.g001.c0005"]
    details = [
        _detail(ids[0], 0, 0.90, 0.50, 12.0, 12.0),
        _detail(ids[1], 0, 0.80, 0.20, 1.0, 0.6),
        _detail(ids[2], 0, 0.95, 0.18, 1.3, 0.6),
        _detail(ids[3], 0, 0.82, 0.40, 2.7, 3.0),
        _detail(ids[4], 1, 0.78, 0.13, 1.6, 0.9),
        _detail(ids[5], 1, 0.96, 0.21, 1.4, 0.7),
    ]
    genomes = {
        ids[0]: _genome(0, "p0", [_block(96), _block(32)]),
        ids[1]: _genome(0, "p1", [_block(32), _block(64, "mbconv_e3")]),
        ids[2]: _genome(0, "p2", [_block(64), _block(16)]),
        ids[3]: _genome(0, "p3", [_block(16), _block(256, "dw_separable")]),
        ids[4]: _genome(1, "p4", [_block(32), _block(64, "mbconv_e3", "fp16")]),
        ids[5]: _genome(1, "p5", [_block(64), _block(16), _block(32)]),
    }
    lineage = {
        c: {
            "candidate_id": c,
            "generation": g["generation"],
            "origin": "initial" if g["generation"] == 0 else "offspring",
            "parents": [] if g["generation"] == 0 else [ids[1], ids[2]],
            "crossover": False,
            "mutated": True,
        }
        for c, g in genomes.items()
    }
    populations = [{"checkpoint_id": 1, "generation": 0, "position": i, "candidate_id": ids[i]} for i in range(4)] + [
        {"checkpoint_id": 2, "generation": 1, "position": i, "candidate_id": c}
        for i, c in enumerate([ids[1], ids[2], ids[3], ids[4]])
    ]
    fronts = [
        {"active_gpu_seconds": 40.0, "candidate_ids": [ids[1], ids[2]]},
        {"active_gpu_seconds": 60.0, "candidate_ids": [ids[1], ids[2], ids[4]]},
    ]
    return AnalysisView("run", details, genomes, lineage, populations, fronts)


def test_evolution_uses_the_surviving_population_of_each_generation(view: AnalysisView) -> None:
    rows = {(r["geracao"], r["objetivo"], r["estatistica"]): r["valor"] for r in evolution_rows(view)}
    assert rows[(0, "E, erro top-1", "melhor")] == 0.80
    assert rows[(1, "E, erro top-1", "melhor")] == 0.78  # c0004 survived into generation 1
    assert abs(rows[(1, "Acurácia (%)", "melhor")] - 22.0) < 1e-9


def test_front_history_lists_who_entered_and_left(view: AnalysisView) -> None:
    first, second = front_history_rows(view)
    assert first["membros"] == 2 and first["sairam"] == "-"
    assert second["entraram"] == "g001.c0004" and second["melhor_E"] == 0.78


def test_ranking_by_accuracy_and_by_balance(view: AnalysisView) -> None:
    by_acc = ranking_rows(view, "acuracia")
    assert by_acc[0]["candidata"] == "g001.c0004" and by_acc[0]["posicao"] == 1
    by_l = ranking_rows(view, "L")
    assert [r["L_ms"] for r in by_l] == sorted(r["L_ms"] for r in by_l)
    balance = ranking_rows(view, "equilibrio")
    assert balance[-1]["candidata"] == "g000.c0000"  # worst in three of the four objectives


def test_genome_distance_is_zero_for_itself_and_counts_missing_blocks(view: AnalysisView) -> None:
    ids = list(view.genomes)
    assert genome_distance(view, ids[0], ids[0]) == 0.0
    assert genome_distance(view, ids[2], ids[5]) > 0  # one extra block
    nearest = similar_rows(view, ids[1])[0]
    assert nearest["candidata"] == "g001.c0004"  # same shape, only the precision of block 2 differs
    assert "precision fp32→fp16" in nearest["diferencas"]


def test_diversity_and_lineage(view: AnalysisView) -> None:
    summary = {r["geracao"]: r for r in diversity_summary(view)}
    assert summary[0]["candidatas"] == 4 and summary[0]["genomas_distintos"] == 4
    nodes, edges = lineage_edges(view)
    assert len(nodes) == 6 and len(edges) == 4  # two children with two parents each
    assert {e["para"] for e in edges} == {"g001.c0004", "g001.c0005"}


def test_module_never_imports_a_writer() -> None:
    tree = ast.parse(MODULE_FILE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert {m for m in imported if m.startswith("cnnlab")} == {"cnnlab.tcc.storage.readmodel"}
    assert "sqlite3" not in imported


def test_tab_draws_against_a_real_database_without_changing_it() -> None:
    """Opt-in: set TCC_REAL_DB to a run database; the file must be byte-identical afterwards."""
    import os

    db = os.environ.get("TCC_REAL_DB")
    if not db:
        pytest.skip("set TCC_REAL_DB to run against a real database")
    from streamlit.testing.v1 import AppTest

    import cnnlab.tcc.demo_streamlit as page

    before = hashlib.sha256(Path(db).read_bytes()).hexdigest()
    at = AppTest.from_file(str(Path(page.__file__)), default_timeout=120).run()
    at.text_input(key="db_path").set_value(db).run()
    assert not at.exception and not at.error
    assert hashlib.sha256(Path(db).read_bytes()).hexdigest() == before


def test_glossary_covers_every_column_with_help_and_searches() -> None:
    from cnnlab.tcc.demo_glossary import TERMS, filtered

    names = " ".join(t[1].lower() for t in TERMS)
    for term in (
        "e, erro",
        "l, latência",
        "v, memória",
        "s, tamanho",
        "acurácia",
        "top-5",
        "precisão macro",
        "recall macro",
        "f1 macro",
        "frente de pareto",
        "nsga-ii",
        "lease",
        "precisão",
    ):
        assert term in names, term
    assert len({t[1] for t in TERMS}) == len(TERMS)  # no duplicated term
    assert all(t[0] == "Métricas auxiliares" for t in filtered("recall macro"))
    assert filtered("") == list(TERMS)


def test_glossary_module_is_static() -> None:
    import cnnlab.tcc.demo_glossary as glossary

    tree = ast.parse(Path(glossary.__file__).read_text(encoding="utf-8"))
    imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in imported if m.startswith("cnnlab")} and "sqlite3" not in imported
