"""Domains of search space v1 against docs/TCC-02-GENOMA.md section 2, and admission gating."""

from __future__ import annotations

import pytest

from cnnlab.tcc.science.space import (
    N_BLOCK_SLOTS,
    SPACE_V1,
    AdmissionError,
    AdmissionEvidence,
    Choice,
    Interval,
    SearchSpace,
)

SHA = "0" * 64

# Literal transcription of the TCC-02 table; code identifiers in parentheses there.
EXPECTED_GLOBAL_CHOICES = {
    "n_blocks": ("ordinal", (2, 3, 4, 5, 6)),
    "stem_channels": ("ordinal", (16, 32, 64)),
    "activation": ("categorical", ("relu", "silu", "hardswish")),
    "head": ("categorical", ("gap_linear", "gap_fc256_linear")),
    "stem_precision": ("categorical", ("fp32", "fp16", "bf16", "int8")),
    "head_precision": ("categorical", ("fp32", "fp16", "bf16", "int8")),
    "int8_mode": ("categorical", ("ptq", "qat")),
    "weight_granularity": ("categorical", ("per_tensor", "per_channel")),
    "calibration": ("categorical", ("max", "entropy", "percentile_99_99")),
}
EXPECTED_GLOBAL_INTERVALS = {
    "log10_lr": (-4.0, -2.0, "log10_exponent"),
    "log10_wd": (-6.0, -3.0, "log10_exponent"),
    "head_dropout": (0.0, 0.5, "linear"),
}
EXPECTED_BLOCK_CHOICES = {
    "channels": ("ordinal", (16, 32, 48, 64, 96, 128, 192, 256)),
    "op": ("categorical", ("standard", "grouped_g4", "dw_separable", "mbconv_e3", "mbconv_e6")),
    "kernel": ("ordinal", (3, 5, 7)),
    "n_convs": ("ordinal", (1, 2, 3)),
    "residual": ("categorical", (False, True)),
    "downsample": ("categorical", ("none", "maxpool", "avgpool", "conv_s2")),
    "precision": ("categorical", ("fp32", "fp16", "bf16", "int8")),
    "prune_pct": ("ordinal", (0, 25, 50)),
}

# Pinned on purpose: any edit to domains, constants or repair version must be a
# deliberate new space version, reviewed before confirmatory runs.
SPACE_V1_HASH = "b1f0d44b360d49f68d0a5bb5b0d97f9ff3e4c3a2eecb7e15dcdc0c9ea1129c7c"


def passing(technique: str = "fp8", scope: tuple[str, ...] = ("standard",), **overrides: object) -> AdmissionEvidence:
    """Synthetic evidence used only to exercise the machinery; nothing is admitted for real."""
    fields: dict[str, object] = dict(
        technique=technique,
        scope=scope,
        evidence_sha256=SHA,
        builds_and_runs=True,
        format_confirmed_in_engine=True,
        architectures_with_gain=2,
    )
    fields.update(overrides)
    return AdmissionEvidence(**fields)  # type: ignore[arg-type]


def test_global_domains_match_tcc02() -> None:
    domains = dict(SPACE_V1.global_domains())
    assert set(domains) == set(EXPECTED_GLOBAL_CHOICES) | set(EXPECTED_GLOBAL_INTERVALS)
    for name, (kind, values) in EXPECTED_GLOBAL_CHOICES.items():
        assert domains[name] == Choice(values, kind), name
    for name, (low, high, scale) in EXPECTED_GLOBAL_INTERVALS.items():
        assert domains[name] == Interval(low, high, scale), name


def test_block_domains_match_tcc02() -> None:
    domains = dict(SPACE_V1.block_domains())
    assert set(domains) == set(EXPECTED_BLOCK_CHOICES)
    for name, (kind, values) in EXPECTED_BLOCK_CHOICES.items():
        assert domains[name] == Choice(values, kind), name


def test_six_block_slots_and_two_to_six_active() -> None:
    assert N_BLOCK_SLOTS == 6
    assert dict(SPACE_V1.global_domains())["n_blocks"].values == (2, 3, 4, 5, 6)


def test_choice_membership_is_type_exact() -> None:
    channels = dict(SPACE_V1.block_domains())["channels"]
    residual = dict(SPACE_V1.block_domains())["residual"]
    assert channels.contains(16)
    assert not channels.contains(16.0)
    assert not residual.contains(1)
    assert not dict(SPACE_V1.global_domains())["n_blocks"].contains(True)


def test_interval_bounds_are_closed() -> None:
    lr = dict(SPACE_V1.global_domains())["log10_lr"]
    assert lr.contains(-4.0) and lr.contains(-2.0)
    assert not lr.contains(-4.000001) and not lr.contains(-1.999999)
    assert not lr.contains(float("nan")) and not lr.contains(-3)


def test_default_space_admits_no_conditional_technique() -> None:
    assert SPACE_V1.admissions == ()
    assert not SPACE_V1.has_sparsity_gene
    for target in ("standard", "grouped_g4", "dw_separable", "mbconv_e3", "mbconv_e6", "stem", "head"):
        assert SPACE_V1.precision_values(target) == ("fp32", "fp16", "bf16", "int8")
    assert SPACE_V1.definition()["admissions"] == []
    assert "sparsity" not in dict(SPACE_V1.block_domains())


def test_space_v1_hash_is_pinned() -> None:
    assert SPACE_V1.space_hash() == SPACE_V1_HASH
    assert SearchSpace().space_hash() == SPACE_V1_HASH


@pytest.mark.parametrize(
    "overrides",
    [
        {"builds_and_runs": False},
        {"format_confirmed_in_engine": False},
        {"architectures_with_gain": 1},
        {"architectures_with_gain": 1, "architectures_failed": 2},
    ],
)
def test_failed_admission_is_recordable_but_cannot_enter_space(overrides: dict[str, object]) -> None:
    evidence = passing("fp8", ("standard",), **overrides)
    assert not evidence.passed
    with pytest.raises(AdmissionError):
        SearchSpace(admissions=(evidence,))


def test_admission_requires_evidence_reference() -> None:
    with pytest.raises(AdmissionError):
        passing("fp8", ("standard",), evidence_sha256="report.md")


@pytest.mark.parametrize("target", ["dw_separable", "stem", "head"])
def test_sparse_24_cannot_be_declared_for_depthwise_stem_or_head(target: str) -> None:
    with pytest.raises(AdmissionError):
        passing("sparse_2_4", ("standard", target))


@pytest.mark.parametrize(
    "overrides",
    [
        {"technique": "int2"},
        {"scope": ()},
        {"scope": ("transformer",)},
        {"architectures_with_gain": 4},
        {"architectures_with_gain": 2, "architectures_failed": 2},
        {"representative_architectures": 5},
    ],
)
def test_malformed_admission_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(AdmissionError):
        passing(**overrides)  # type: ignore[arg-type]


def test_duplicate_admission_rejected() -> None:
    with pytest.raises(AdmissionError):
        SearchSpace(admissions=(passing("fp8", ("standard",)), passing("fp8", ("head",))))


def test_admission_extends_only_its_scope_and_changes_space_hash() -> None:
    space = SearchSpace(admissions=(passing("fp8", ("standard", "head")),))
    assert space.space_hash() != SPACE_V1_HASH
    assert "fp8" in space.precision_values("standard")
    assert "fp8" in space.precision_values("head")
    assert "fp8" not in space.precision_values("dw_separable")
    assert "fp8" not in space.precision_values("stem")
    assert "fp8" in space.block_precision_values()


def test_space_hash_ignores_admission_order() -> None:
    a = passing("fp8", ("standard",))
    b = passing("sparse_2_4", ("standard", "grouped_g4"))
    assert SearchSpace(admissions=(a, b)).space_hash() == SearchSpace(admissions=(b, a)).space_hash()
    assert SearchSpace(admissions=(b,)).has_sparsity_gene
