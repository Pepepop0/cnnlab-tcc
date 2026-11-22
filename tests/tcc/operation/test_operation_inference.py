"""Statistics of the confirmatory campaign (TCC-01 section 6; task A10b). CPU only, pure computation.

Exact answers are worked out by hand: eight equal positive differences give p = 1/256; a symmetric set gives p = 1/2.
The inputs are tables in the format the analysis exports; nothing here is measured, and synthetic files are labeled.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import pytest

from cnnlab.tcc.operation import inference
from cnnlab.tcc.operation.inference import InferenceError


def brute_p(d: list[float]) -> float:
    sums = [sum(s * x for s, x in zip(signs, d)) for signs in itertools.product((-1, 1), repeat=len(d))]
    return sum(1 for s in sums if s >= sum(d) - 1e-12) / len(sums)


def test_eight_equal_differences_give_the_smallest_p():
    assert inference.sign_flip_p([0.1] * 8) == pytest.approx(1 / 256)


def test_symmetric_differences_are_not_evidence():
    d = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]  # sum 0: patterns with sum >= 0, those equal to 0 counted too
    assert inference.sign_flip_p(d) == pytest.approx(brute_p(d))
    assert inference.sign_flip_p(d) > 0.5


def test_one_sided_directions_are_complementary():
    d = [0.3, 0.1, -0.05, 0.2, 0.4, 0.15, 0.0, 0.25]
    up, down = inference.sign_flip_p(d, "greater"), inference.sign_flip_p(d, "less")
    assert up < 0.1 and down > 0.9


@pytest.mark.parametrize(
    "d",
    [
        [0.3, 0.1, -0.05, 0.2, 0.4, 0.15, 0.0, 0.25],
        [1.0, 2.0, 3.0],
        [0.5, 0.5, 0.5, -0.5],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    ],
)
def test_p_matches_brute_force(d):
    assert inference.sign_flip_p(d) == pytest.approx(brute_p(d))


def test_zero_differences_keep_every_pattern():
    # a zero difference changes nothing about the sums, so p is the p of the rest (patterns of the zero double the counts)
    assert inference.sign_flip_p([0.0, 1.0, 2.0]) == pytest.approx(inference.sign_flip_p([1.0, 2.0]))
    assert inference.sign_flip_p([0.0] * 8) == 1.0


def test_float_noise_is_a_tie():
    base = [0.1, 0.2, 0.3, 0.4]
    noisy = [x + 1e-17 for x in base]
    assert inference.sign_flip_p(noisy) == inference.sign_flip_p(base)


def test_refuses_bad_input():
    with pytest.raises(InferenceError):
        inference.sign_flip_p([])
    with pytest.raises(InferenceError):
        inference.sign_flip_p([1.0, float("nan")])
    with pytest.raises(InferenceError):
        inference.sign_flip_p([1.0] * 21)
    with pytest.raises(InferenceError):
        inference.sign_flip_p([1.0], "bigger")


def test_holm_is_step_down_monotone_and_capped():
    out = inference.holm({"H2": 0.01, "H3": 0.03})
    assert out["H2"]["p_holm"] == pytest.approx(0.02) and out["H2"]["reject"]
    assert out["H3"]["p_holm"] == pytest.approx(0.03) and out["H3"]["reject"]
    out = inference.holm({"H2": 0.03, "H3": 0.04})
    assert out["H2"]["p_holm"] == pytest.approx(0.06) and not out["H2"]["reject"]
    assert out["H3"]["p_holm"] == pytest.approx(0.06) and not out["H3"]["reject"]  # monotone: never below the earlier
    assert inference.holm({"H2": 0.9, "H3": 0.8})["H2"]["p_holm"] == 1.0


def test_interval_of_equal_differences_collapses_toward_the_value():
    low, high = inference.exact_interval([0.2] * 8)
    assert low <= 0.2 <= high
    assert low > 0.0  # the test rejects a shift of 0 against 8 equal positives (p two-sided 2/256)
    # brute force: every delta strictly inside is accepted, every one outside is rejected
    for delta in (low - 0.01, high + 0.01):
        shifted = [x - delta for x in [0.2] * 8]
        sums = [abs(sum(s * x for s, x in zip(sg, shifted))) for sg in itertools.product((-1, 1), repeat=8)]
        assert sum(1 for s in sums if s >= abs(sum(shifted)) - 1e-12) / 256 <= 0.05


def test_interval_matches_a_dense_scan():
    d = [0.31, 0.12, -0.04, 0.22, 0.41, 0.17, 0.02, 0.26]
    low, high = inference.exact_interval(d)

    def two_sided_p(delta: float) -> float:
        shifted = [x - delta for x in d]
        sums = [abs(sum(s * x for s, x in zip(sg, shifted))) for sg in itertools.product((-1, 1), repeat=8)]
        return sum(1 for s in sums if s >= abs(sum(shifted)) - 1e-12) / 256

    accepted = [x / 1000 for x in range(-200, 700) if two_sided_p(x / 1000) > 0.05]
    assert low == pytest.approx(min(accepted), abs=0.0015) and high == pytest.approx(max(accepted), abs=0.0015)
    assert low < sum(d) / 8 < high


def test_describe_counts_signs():
    s = inference.describe([0.2, -0.1, 0.0, 0.3])
    assert (s["n"], s["positive"], s["negative"], s["zero"]) == (4, 2, 1, 1)
    assert s["mean"] == pytest.approx(0.1) and s["median"] == pytest.approx(0.1)
    assert (s["min"], s["max"]) == (-0.1, 0.3)
    assert inference.describe([1.0])["sd"] is None


def values(joint: list[float], random: list[float], sequential: list[float]) -> dict[str, dict[int, float]]:
    return {m: dict(enumerate(v)) for m, v in (("joint", joint), ("random", random), ("sequential", sequential))}


def test_family_of_h2_and_h3_with_holm():
    v = values([0.5] * 8, [0.4] * 8, [0.45, 0.5, 0.4, 0.55, 0.5, 0.45, 0.6, 0.35])
    h2, h3 = inference.paired_tests(v)
    assert (h2.hypothesis, h3.hypothesis) == ("H2", "H3")
    assert h2.p == pytest.approx(1 / 256) and h2.p_holm == pytest.approx(2 / 256) and h2.reject
    assert h3.p > 0.05 and not h3.reject
    assert h2.blocks == tuple(range(8)) and len(h2.differences) == 8
    # equal differences: only the shift equal to them is not rejected, so the interval is that single point
    assert h2.ci_low == pytest.approx(0.1) and h2.ci_high == pytest.approx(0.1)
    assert isinstance(h2.ci_low, float)


def test_only_common_blocks_are_paired_and_listed():
    v = values([0.5] * 8, [0.4] * 8, [0.3] * 8)
    del v["random"][3]
    h2, _ = inference.paired_tests(v)
    assert h2.blocks == (0, 1, 2, 4, 5, 6, 7) and h2.stats["n"] == 7
    with pytest.raises(InferenceError):
        inference.paired_tests({"joint": {0: 1.0}, "random": {1: 1.0}, "sequential": {0: 1.0}})
    with pytest.raises(InferenceError):
        inference.paired_tests({"joint": {0: 1.0}})


# -- H1 -----------------------------------------------------------------------------------------------------------


def final_row(block: int, method: str, role: str, acc: float, l: float, v: float, s: float) -> dict[str, object]:
    return {"seed_block": block, "method": method, "role": role, "test_accuracy": acc, "l_ms": l, "v_mb": v, "s_mb": s}


def h1_rows(blocks: int = 8, loss_pp: float = 0.5) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for b in range(blocks):
        rows.append(final_row(b, "mono", "menor_E", 0.60, 2.0, 40.0, 20.0))
        rows.append(final_row(b, "joint", "H1", 0.60 - loss_pp / 100, 1.0, 20.0, 10.0))
    return rows


def test_h1_table_reductions_and_support():
    table = inference.h1_table(h1_rows())
    assert len(table) == 8 and all(r.status == "ok" for r in table)
    r = table[0]
    assert r.delta_pp == pytest.approx(-0.5)
    assert r.reductions == pytest.approx({"l_ms": 0.5, "v_mb": 0.5, "s_mb": 0.5})
    assert r.supports(1.0) and r.supports(2.0)
    summary = inference.h1_summary(table)
    assert summary["supporting"] == {"1pp": 8, "2pp": 8, "5pp": 8} and summary["criterion_met"]


def test_h1_thresholds_are_sensitivity_not_a_choice():
    table = inference.h1_table(h1_rows(loss_pp=3.0))
    summary = inference.h1_summary(table)
    assert summary["supporting"] == {"1pp": 0, "2pp": 0, "5pp": 8} and not summary["criterion_met"]


def test_h1_needs_all_three_reduced():
    rows = h1_rows(blocks=2)
    rows[1]["v_mb"] = 41.0  # candidate bigger in V than the champion
    table = inference.h1_table(rows, expected_blocks=range(2))
    assert not table[0].supports(5.0) and table[1].supports(5.0)


def test_h1_missing_pieces_are_failures_never_filled():
    rows = [r for r in h1_rows(blocks=3) if not (r["seed_block"] == 1 and r["role"] == "H1")]
    rows = [r for r in rows if not (r["seed_block"] == 2 and r["role"] == "menor_E")]
    table = inference.h1_table(rows, expected_blocks=range(4))
    assert [r.status for r in table] == ["ok", "sem_candidata_H1", "sem_campeao_mono", "sem_candidata_H1"]
    summary = inference.h1_summary(table)
    assert summary["with_both"] == 1 and len(summary["failures"]) == 3
    assert summary["supporting"]["5pp"] == 1 and not summary["criterion_met"]


def test_h1_refuses_duplicates_and_accuracy_in_percent():
    rows = h1_rows(blocks=1) + [final_row(0, "joint", "H1", 0.5, 1, 1, 1)]
    with pytest.raises(InferenceError):
        inference.h1_table(rows, expected_blocks=[0])
    with pytest.raises(InferenceError):
        inference.h1_table(
            [final_row(0, "mono", "menor_E", 60.0, 1, 1, 1), final_row(0, "joint", "H1", 59.0, 1, 1, 1)],
            expected_blocks=[0],
        )


# -- files ----------------------------------------------------------------------------------------------------------


def write_paired(path: Path, joint, random, sequential, *, synthetic: bool = False) -> Path:
    suffix = "_SINTETICO" if synthetic else ""
    cols = [
        "seed_block",
        *(f"{m}{suffix}" for m in ("joint", "random", "sequential")),
        f"joint_menos_random{suffix}",
        f"joint_menos_sequential{suffix}",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for b, (j, r, s) in enumerate(zip(joint, random, sequential)):
            w.writerow([b, j, r, s, j - r, j - s])
    return path


def write_final(path: Path, rows) -> Path:
    cols = ["seed_block", "method", "role", "candidate_id", "test_accuracy", "l_ms", "v_mb", "s_mb"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({"candidate_id": "c", **r})
    return path


def test_write_analysis_files_and_manifest(tmp_path):
    paired = write_paired(tmp_path / "p.csv", [0.5] * 8, [0.4] * 8, [0.45, 0.5, 0.4, 0.55, 0.5, 0.45, 0.6, 0.35])
    sens = write_paired(tmp_path / "s.csv", [0.5] * 8, [0.45] * 8, [0.45] * 8)
    final = write_final(tmp_path / "f.csv", h1_rows())
    manifest = inference.write_analysis(tmp_path / "out", paired=paired, sensitivity=sens, final_results=final)
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names == [
        "h1_resumo.json",
        "h1_tabela.csv",
        "h2_h3.csv",
        "h2_h3_pares.csv",
        "h2_h3_pares_sensibilidade.csv",
        "h2_h3_sensibilidade.csv",
        "manifesto.json",
    ]
    assert manifest["format"] == "tcc-inference/1" and manifest["synthetic"] is False
    on_disk = json.loads((tmp_path / "out" / "manifesto.json").read_text(encoding="utf-8"))
    assert on_disk == manifest and set(on_disk["sources_sha256"]) == {"official", "sensitivity", "final_results"}
    with open(tmp_path / "out" / "h2_h3.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["hypothesis"] for r in rows] == ["H2", "H3"]
    assert float(rows[0]["p"]) == pytest.approx(1 / 256) and rows[0]["reject_at_5pct"] == "true"
    with open(tmp_path / "out" / "h1_tabela.csv", encoding="utf-8") as fh:
        h1 = list(csv.DictReader(fh))
    assert len(h1) == 8 and h1[0]["supports_1pp"] == "true" and float(h1[0]["delta_pp"]) == pytest.approx(-0.5)


def test_write_analysis_is_deterministic_apart_from_the_clock(tmp_path):
    paired = write_paired(tmp_path / "p.csv", [0.5] * 8, [0.4] * 8, [0.3] * 8)
    inference.write_analysis(tmp_path / "a", paired=paired)
    inference.write_analysis(tmp_path / "b", paired=paired)
    for name in ("h2_h3.csv", "h2_h3_pares.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_write_analysis_never_overwrites_or_leaves_partials(tmp_path):
    paired = write_paired(tmp_path / "p.csv", [0.5] * 8, [0.4] * 8, [0.3] * 8)
    (tmp_path / "out").mkdir()
    with pytest.raises(InferenceError, match="already exists"):
        inference.write_analysis(tmp_path / "out", paired=paired)
    with pytest.raises(InferenceError):
        inference.write_analysis(tmp_path / "nothing")
    assert not [p for p in tmp_path.iterdir() if ".partial" in p.name]


def test_synthetic_inputs_need_opt_in_and_are_labeled(tmp_path):
    paired = write_paired(tmp_path / "p.csv", [0.5] * 8, [0.4] * 8, [0.3] * 8, synthetic=True)
    with pytest.raises(InferenceError, match="synthetic"):
        inference.write_analysis(tmp_path / "x", paired=paired)
    assert not (tmp_path / "x").exists()
    manifest = inference.write_analysis(tmp_path / "y", paired=paired, allow_synthetic=True)
    assert manifest["synthetic"] is True
    header = (tmp_path / "y" / "h2_h3.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "p_SINTETICO" in header and "mean_diff_SINTETICO" in header and "hypothesis" in header
    assert "p" not in header


def test_final_results_need_their_columns(tmp_path):
    bad = tmp_path / "f.csv"
    bad.write_text("seed_block,method\n0,joint\n", encoding="utf-8")
    with pytest.raises(InferenceError, match="missing columns"):
        inference.load_final_results(bad)


def test_paired_file_without_methods_is_refused(tmp_path):
    bad = tmp_path / "p.csv"
    bad.write_text("seed_block,foo\n0,1\n", encoding="utf-8")
    with pytest.raises(InferenceError):
        inference.load_paired(bad)


def test_cli_inference_prints_and_refuses(tmp_path, capsys):
    from cnnlab.tcc.operation import cli
    from cnnlab.tcc.operation.demo import EXIT_REFUSED

    paired = write_paired(tmp_path / "p.csv", [0.5] * 8, [0.4] * 8, [0.3] * 8)
    final = write_final(tmp_path / "f.csv", h1_rows())
    base = ["campaign", "inference", "--paired", str(paired), "--final-results", str(final)]
    assert cli.main([*base, "--dest", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "official H2: p=0.003906" in out and "rejects at 5%" in out and "H1: 8 of 8" in out
    assert cli.main([*base, "--dest", str(tmp_path / "out")]) == EXIT_REFUSED  # never overwrites
    assert "already exists" in capsys.readouterr().err
    assert cli.main(["campaign", "inference", "--dest", str(tmp_path / "none")]) == EXIT_REFUSED
