"""The sequential method: search an architecture, then compress it (TCC-01 section 4).

One run, one job, one GPU budget (5.5 h), two stages of half of it each (2.75 h + 2.75 h):

1. **Architecture.** NSGA-II on (E, L, V, S) over the architecture genes with the representation
   fixed to FP32 and no pruning. When its GPU time is spent (or it can no longer produce a new
   evaluation), the up-to-``n_frozen`` (seven) architectures of lowest validation error are
   frozen, one per distinct phenotype.
2. **Representation and pruning.** NSGA-II over the representation and pruning genes of those
   architectures only. The architecture genes of every individual are those of one frozen
   architecture, never mutated or crossed. The allocation is *rotating*: the proposals of this
   stage go to the frozen architectures one after the other (a counter that survives resumes), so
   each gets the same share of the new evaluations whatever the selection does to the
   population. Parents come from the population members of the same architecture, or from the
   frozen one when selection has dropped them all. The stage starts from the frozen architectures
   themselves (already evaluated, so free) plus random representations of each.

Both stages work on **one slice**, the admitted one of the joint method, which identifies the run
and its candidates (the store requires one space hash per run). Stage 1 draws and varies genomes in
a *restriction* of that slice (the architecture slice: representation pinned to FP32 and dense),
whose genomes are all inside the admitted slice. So stage 2 "accesses the same admitted set as
the joint method", and an FP32 architecture evaluated in stage 1 is a cache hit if stage 2
proposes it again.

What is architecture and what is representation. Architecture genes are the ones that shape the
network and its training recipe: ``n_blocks``, ``stem_channels``, ``activation``, ``head``, the
learning-rate, weight-decay and dropout genes, and per block ``channels``, ``op``, ``kernel``,
``n_convs``, ``residual`` and ``downsample``. Everything else is representation or pruning
(precisions, the INT8 genes, ``prune_pct``, ``sparsity``), so a gene added to the space later is
searched in stage 2 unless it is listed here. The training-recipe genes travel with the
architecture on purpose: the second stage compresses a model, it does not retune it.

Time. A stage ends between two attempts, like the budget: the attempt that crosses the stage
limit finishes and counts. Unused time of stage 1 is **not** given to stage 2 (TCC-01: "tempo não
usado numa etapa não é convertido em avaliações gratuitas"): stage 2 gets its own 2.75 h from
where it begins, still within the run's total budget, which the operation service enforces.
Proposals of stage 1 that had not been evaluated when its time ran out are dropped, not
evaluated late.

The archive and the front snapshots cover everything evaluated in both stages (the result of the
method is its non-dominated set over everything it spent its budget on).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from cnnlab.tcc.contracts import TCCJobSpec
from cnnlab.tcc.science.genome import BlockGenes, Genome
from cnnlab.tcc.science.nsga2 import rank_and_crowding, tournament
from cnnlab.tcc.science.operators import SearchSlice, crossover, mutate, slice_frozen_genes
from cnnlab.tcc.science.record import CacheContext
from cnnlab.tcc.science.sampling import GeneRng
from cnnlab.tcc.science.search import (
    RunnerContextLike,
    RunnerStatus,
    SearchConfig,
    SearchRunner,
    _State,
)
from cnnlab.tcc.science.space import GLOBAL_GENES

SEQUENTIAL_VERSION = "tcc-sequential/v1"
FROZEN_ARCHITECTURES = 7
ARCHITECTURE_GLOBAL: tuple[str, ...] = (
    "n_blocks",
    "stem_channels",
    "activation",
    "head",
    "log10_lr",
    "log10_wd",
    "head_dropout",
)
ARCHITECTURE_BLOCK: tuple[str, ...] = ("channels", "op", "kernel", "n_convs", "residual", "downsample")
# What stage 2 may move: the names ``mutate`` takes as frozen (a global name, or "block.<name>").
ARCHITECTURE_FROZEN = frozenset(ARCHITECTURE_GLOBAL) | frozenset(f"block.{n}" for n in ARCHITECTURE_BLOCK)
REPRESENTATION_GLOBAL: tuple[str, ...] = tuple(n for n in GLOBAL_GENES if n not in ARCHITECTURE_GLOBAL)
REPRESENTATION_BLOCK: tuple[str, ...] = ("precision", "prune_pct", "sparsity")


def graft(architecture: Genome, representation: Genome) -> Genome:
    """``architecture``'s architecture genes with ``representation``'s representation and pruning genes.

    Latent block slots are grafted too, so the result is fixed by the two inputs and nothing else.
    """
    globals_ = {
        name: getattr(architecture if name in ARCHITECTURE_GLOBAL else representation, name) for name in GLOBAL_GENES
    }
    blocks: tuple[BlockGenes, ...] = tuple(
        replace(rep, **{name: getattr(arch, name) for name in ARCHITECTURE_BLOCK})
        for arch, rep in zip(architecture.blocks, representation.blocks)
    )
    return Genome(**globals_, blocks=blocks)


def architecture_signature(genome: Genome) -> tuple[Any, ...]:
    """The architecture genes of the whole raw genome (latent slots included), comparable and hashable."""
    return (
        tuple(getattr(genome, name) for name in ARCHITECTURE_GLOBAL),
        tuple(tuple(getattr(b, name) for name in ARCHITECTURE_BLOCK) for b in genome.blocks),
    )


@dataclass(frozen=True, slots=True)
class SequentialConfig:
    """Stage split and the number of frozen architectures.

    ``stage_gpu_s`` is ``(stage 1, stage 2)`` in active GPU seconds; None splits the job's budget in half
    (5.5 h gives 2.75 h and 2.75 h), which is what the protocol fixes.
    """

    n_frozen: int = FROZEN_ARCHITECTURES
    stage_gpu_s: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.n_frozen, bool) or not isinstance(self.n_frozen, int) or self.n_frozen < 1:
            raise ValueError("n_frozen must be an int >= 1")
        if self.stage_gpu_s is not None:
            stages = tuple(float(x) for x in self.stage_gpu_s)
            if len(stages) != 2 or not all(math.isfinite(x) and x > 0 for x in stages):
                raise ValueError("stage_gpu_s must be two positive finite numbers (stage 1, stage 2)")
            object.__setattr__(self, "stage_gpu_s", stages)

    def budgets(self, job_budget_gpu_s: float) -> tuple[float, float]:
        return self.stage_gpu_s or (job_budget_gpu_s / 2.0, job_budget_gpu_s / 2.0)

    def definition(self) -> dict[str, Any]:
        return {
            "version": SEQUENTIAL_VERSION,
            "n_frozen": self.n_frozen,
            "stage_gpu_s": None if self.stage_gpu_s is None else list(self.stage_gpu_s),
            "architecture_genes": {"global": list(ARCHITECTURE_GLOBAL), "block": list(ARCHITECTURE_BLOCK)},
        }


def representation_is_searchable(admitted: SearchSlice) -> bool:
    """Whether some representation or pruning gene is free in the slice (else stage 2 has nothing to search)."""
    pinned_global = {name for name, _ in admitted.pinned_global}
    pinned_block = {name for name, _ in admitted.pinned_block}
    return any(n not in pinned_global for n in ("stem_precision", "head_precision")) or any(
        n not in pinned_block for n in ("precision", "prune_pct")
    )


class SequentialRunner(SearchRunner):
    """``runner(ctx)`` for ``method == "sequential"``; see the module docstring."""

    def __init__(
        self,
        config: SearchConfig,
        sequential: SequentialConfig,
        admitted: SearchSlice,
        architecture: SearchSlice,
        context_for: Callable[[TCCJobSpec], CacheContext],
    ) -> None:
        if config.generations is not None:
            raise ValueError("the sequential method is limited by GPU time per stage, not by generations")
        if architecture.base.space_hash() != admitted.base.space_hash():
            raise ValueError("the architecture slice and the admitted slice must share one base space")
        if not representation_is_searchable(admitted):
            raise ValueError(
                "the admitted slice pins every representation and pruning gene: stage 2 would have nothing to search"
            )
        if not admitted.contains(architecture.sample(GeneRng(0))):  # a probe: the operators draw from it
            raise ValueError("the architecture slice must lie inside the admitted slice")
        super().__init__(config, admitted, context_for, operator_space=architecture)
        self.seq = sequential

    # -- time ---------------------------------------------------------------------------

    def _budgets(self, ctx: RunnerContextLike) -> tuple[float, float]:
        return self.seq.budgets(ctx.job.budget_gpu_s)

    def _stage_capped(self, ctx: RunnerContextLike, state: _State) -> bool:
        stage1, stage2 = self._budgets(ctx)
        active = ctx.active_gpu_seconds()
        if state.stage == 1:
            return active >= stage1
        start = 0.0 if state.stage_start_gpu_s is None else state.stage_start_gpu_s
        return active - start >= stage2

    def _stopped(self, ctx: RunnerContextLike, state: _State) -> bool:
        return ctx.should_stop() or self._stage_capped(ctx, state)

    def _start_info(self, ctx: RunnerContextLike) -> dict[str, Any]:
        stage1, stage2 = self._budgets(ctx)
        return {
            "sequential": {**self.seq.definition(), "stage_budgets_gpu_s": [stage1, stage2]},
            "operator_slice": self.ops.reason,
        }

    # -- the two stages ------------------------------------------------------------------

    def __call__(self, ctx: RunnerContextLike) -> RunnerStatus:
        if ctx.job.method != "sequential":
            raise ValueError(f"SequentialRunner runs the sequential method, not {ctx.job.method!r}")
        status = super().__call__(ctx)
        state = self._state
        # "stopped" because a stage ran out of GPU time (the service did not ask): go on or finish.
        while status == "stopped" and not ctx.should_stop() and self._stage_capped(ctx, state):
            if state.stage == 2:
                return self._finish(ctx, state, "stage 2 used its GPU time")
            if not self._enter_stage2(ctx, state, "stage 1 used its GPU time"):
                return "completed"
            status = self._loop(ctx, state, self._rng, self._cache, False)
        return status

    def _record_of(self, ctx: RunnerContextLike, state: _State, candidate_id: str) -> Any:
        return (self._records.get(candidate_id) or self._rebuild_record(ctx, state, candidate_id))[0]

    def _freeze(self, ctx: RunnerContextLike, state: _State) -> list[str]:
        """Up to ``n_frozen`` valid candidates of lowest error, one per phenotype (ties by phenotype hash)."""
        rows = sorted(
            (f[0], self._record_of(ctx, state, cid).phenotype_hash, cid)
            for cid, f in state.fitness4.items()
            if f is not None
        )
        chosen: list[str] = []
        seen: set[str] = set()
        for _error, phenotype, candidate_id in rows:
            if phenotype in seen:
                continue
            seen.add(phenotype)
            chosen.append(candidate_id)
            if len(chosen) == self.seq.n_frozen:
                break
        return chosen

    def _enter_stage2(self, ctx: RunnerContextLike, state: _State, reason: str) -> bool:
        """Freeze the architectures and propose the first individuals of stage 2; False if none can be frozen."""
        rng, cache = self._rng, self._cache
        if state.phase != "survived":  # stage 1 stopped inside a generation: put its results on the curve
            self._record_front(ctx, state)
        frozen = self._freeze(ctx, state)
        dropped = list(state.pending)
        if not frozen:
            self._emit(
                ctx,
                "sequential_finished",
                {
                    "reason": "stage 1 produced no valid architecture to compress",
                    "generation": state.generation,
                    "active_gpu_seconds": ctx.active_gpu_seconds(),
                },
            )
            return False
        spent = ctx.active_gpu_seconds()
        state.stage, state.frozen, state.rotation, state.stage_start_gpu_s = 2, frozen, 0, spent
        state.pending, state.offspring, state.population = [], [], list(frozen)  # the baselines are already evaluated
        state.generation += 1
        for _ in range(max(0, self.config.population - len(frozen))):
            state.offspring.append(self._initial_individual(ctx, state, rng, cache))
        state.phase = "offspring"  # the loop evaluates the new individuals and lets selection pick the population
        self._checkpoint(ctx, state, rng)
        stage1, stage2 = self._budgets(ctx)
        self._emit(
            ctx,
            "stage_changed",
            {
                "from": 1,
                "to": 2,
                "reason": reason,
                "generation": state.generation,
                "active_gpu_seconds": spent,
                "stage1_budget_gpu_s": stage1,
                "stage1_unused_gpu_s": max(0.0, stage1 - spent),
                "stage2_budget_gpu_s": stage2,
                "dropped_unevaluated": dropped,
                "frozen": [
                    {
                        "candidate_id": cid,
                        "e": state.fitness4[cid][0],  # type: ignore[index]
                        "phenotype_hash": self._record_of(ctx, state, cid).phenotype_hash,
                    }
                    for cid in frozen
                ],
            },
        )
        return True

    def _finish(self, ctx: RunnerContextLike, state: _State, reason: str) -> RunnerStatus:
        if state.phase != "survived":
            self._record_front(ctx, state)
        _stage1, stage2 = self._budgets(ctx)
        spent = ctx.active_gpu_seconds()
        start = 0.0 if state.stage_start_gpu_s is None else state.stage_start_gpu_s
        self._emit(
            ctx,
            "sequential_finished",
            {
                "reason": reason,
                "generation": state.generation,
                "active_gpu_seconds": spent,
                "stage2_gpu_seconds": spent - start,
                "stage2_budget_gpu_s": stage2,
                "job_budget_gpu_s": ctx.job.budget_gpu_s,
                "unused_gpu_s": max(0.0, ctx.job.budget_gpu_s - spent),
                "pareto_archive": self._archive(state),
            },
        )
        return "completed"

    def _on_exhausted(
        self, ctx: RunnerContextLike, state: _State, rng: GeneRng, cache: CacheContext, stalled: int
    ) -> RunnerStatus | None:
        if state.stage == 1:  # nothing new to find in the architectures: go compress what there is
            return (
                None
                if self._enter_stage2(ctx, state, "stage 1 can no longer produce a new evaluation")
                else "completed"
            )
        return super()._on_exhausted(ctx, state, rng, cache, stalled)

    # -- stage 2 proposals -----------------------------------------------------------------

    def _next_architecture(self, state: _State) -> str:
        architecture_id = state.frozen[state.rotation % len(state.frozen)]
        state.rotation += 1
        return architecture_id

    def _initial_individual(self, ctx: RunnerContextLike, state: _State, rng: GeneRng, cache: CacheContext) -> str:
        architecture_id = self._next_architecture(state)
        architecture = state.genomes[architecture_id]
        # pin: the frozen architecture can leave a BF16 drawn for another activation outside the slice's scope
        genome, info = self._propose(
            ctx, state, rng, lambda: self.space.pin(graft(architecture, self.space.sample(rng))), cache
        )
        return self._register(
            ctx, state, genome, cache, {"origin": "initial_stage2", "architecture": architecture_id, **info}
        )

    def _vary(
        self, a: Genome, b: Genome, rng: GeneRng, architecture: Genome, mutation_p: float
    ) -> tuple[Genome, Genome, dict[str, Any]]:
        """Two children of ``a`` and ``b`` that keep ``architecture`` exactly; only representation moves."""
        frozen = ARCHITECTURE_FROZEN | slice_frozen_genes(self.space)
        restricted = self.space.restricted_domains()
        crossed = rng.unit() < self.config.crossover_p
        c1, c2 = crossover(a, b, rng, self.space.base) if crossed else (a, b)
        c1, c2 = graft(architecture, c1), graft(architecture, c2)  # latent slots may have mixed: restore them
        out: list[Genome] = []
        mutated: list[bool] = []
        for child in (c1, c2):
            do_mutate = rng.unit() < mutation_p or child == a or child == b
            if do_mutate:
                child = mutate(child, rng, self.space.base, self.config.sigma_rel, frozen, restricted)
            out.append(self.space.pin(child))
            mutated.append(do_mutate)
        return out[0], out[1], {"crossover": crossed, "mutated": mutated}

    def _make_offspring(self, ctx: RunnerContextLike, state: _State, rng: GeneRng, cache: CacheContext) -> None:
        if state.stage == 1:
            super()._make_offspring(ctx, state, rng, cache)
            return
        ranks = rank_and_crowding({k: state.objectives.get(k) for k in dict.fromkeys(state.population + state.frozen)})
        owner = {architecture_signature(state.genomes[fid]): fid for fid in state.frozen}
        members: dict[str, list[str]] = defaultdict(list)
        for cid in state.population:
            members[owner[architecture_signature(state.genomes[cid])]].append(cid)
        for _ in range(self.config.population // 2):
            architecture_id = self._next_architecture(state)
            architecture = state.genomes[architecture_id]
            pool = members.get(architecture_id) or [architecture_id]  # selection may have dropped them all
            a, b = tournament(pool, ranks, rng, 2)
            c1, c2, info = self._vary(state.genomes[a], state.genomes[b], rng, architecture, self.config.mutation_p)
            for child, mutated in zip((c1, c2), info["mutated"]):

                def remutate() -> Genome:
                    return self._vary(state.genomes[a], state.genomes[b], rng, architecture, 1.0)[0]

                first = [child]
                chosen, extra = self._propose(ctx, state, rng, lambda: first.pop() if first else remutate(), cache)
                state.offspring.append(
                    self._register(
                        ctx,
                        state,
                        chosen,
                        cache,
                        {
                            "origin": "offspring_stage2",
                            "architecture": architecture_id,
                            "parents": [a, b],
                            "crossover": info["crossover"],
                            "mutated": [mutated],
                            **extra,
                        },
                    )
                )
