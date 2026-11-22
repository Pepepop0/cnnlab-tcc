"""Wiring factory of the pilot demo: joins the operation front to the science front.

It is the
only module that imports both ``operation`` and ``model_pipeline``; neither
front imports the other.

Spec usage::

    "factory": "cnnlab.tcc.demo_wiring:demo_wiring",
    "factory_args": {"cifar_root": "...", "work_dir": "...", "train_per_class": 100,
                     "epochs": 5, "population": 4, "generations": 2}

Scope: pilot stage, FP32 unpruned slice, fixed subset of the train split
(or the full 45k split with ``train_per_class: null``), preliminary L/V. Not the
confirmatory protocol. The official test split is registered by the fingerprint
of its index set only.

``factory_args.reference_point`` (four numbers, E, L, V, S) is the hypervolume reference
point of TCC-01 section 5. It travels in the frozen plan with the other ``factory_args`` and
is handed to the search runner here, so every front snapshot of the run stores its
hypervolume. Without it no hypervolume is recorded.

Methods: ``joint`` (NSGA-II on E, L, V, S), ``mono`` (NSGA-II on E only) and
``random``. They share space, genome, repair and evaluation; the search runner
reads the method from the job. ``sequential`` has its own two-stage runner
(``science.sequential``): the admitted slice is the one named by ``slice`` (it must leave a
representation or pruning gene free), stage 1 searches architectures in the FP32 slice, and
``factory_args`` may carry ``n_frozen`` (default 7) and ``stage_gpu_s`` ([stage 1, stage 2];
default: half of the run budget each). It takes no generation limit.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from cnnlab.tcc.model_pipeline.pilot_wiring import PATH_ENVIRONMENT, build_pilot_science
from cnnlab.tcc.operation.demo import DatasetRegistration, DemoRunSpec, DemoWiring
from cnnlab.tcc.operation.sentinel_control import DRIFT_LIMIT, SentinelHooks
from cnnlab.tcc.science.operators import SLICES
from cnnlab.tcc.science.search import SearchRunner
from cnnlab.tcc.science.sequential import (
    FROZEN_ARCHITECTURES,
    SequentialConfig,
    SequentialRunner,
    representation_is_searchable,
)
from cnnlab.tcc.storage.sqlite_store import SplitDescriptor

# Runner label per method; the science wiring names every run "nsga2-joint".
RUNNER_LABELS = {
    "joint": "nsga2-joint",
    "mono": "nsga2-mono",
    "random": "random-search",
    "sequential": "sequential-2stage",
}
# Where stage 1 of the sequential method searches architectures: the admitted slice with the representation
# fixed to FP32 and no pruning (``SLICES['architecture']``, inside ``SLICES['admitted']``).
ARCHITECTURE_SLICE = SLICES["architecture"]


def _work_dir(args: object) -> Path:
    """The science wiring's work directory (argument or environment): where the sentinel engine is kept."""
    value = dict(args).get("work_dir") or os.environ.get(PATH_ENVIRONMENT["work_dir"])  # type: ignore[call-overload]
    if not value:
        raise ValueError(f"the sentinel needs the work directory (work_dir or ${PATH_ENVIRONMENT['work_dir']})")
    return Path(value)


def sentinel_hooks(work_dir: str | Path, every: int, limit: float = DRIFT_LIMIT) -> SentinelHooks:
    """The sentinel network of ``model_pipeline.sentinel`` as the operation's hooks (imported only when measuring)."""

    def measure() -> dict[str, object]:
        from cnnlab.tcc.model_pipeline import sentinel

        return sentinel.measure_sentinel(Path(work_dir))

    def drift(reference: object, current: object) -> dict[str, object]:
        from cnnlab.tcc.model_pipeline import sentinel

        return sentinel.drift(reference, current, limit=limit)  # type: ignore[arg-type]

    return SentinelHooks(measure=measure, drift=drift, every=every)  # type: ignore[arg-type]


def demo_wiring(spec: DemoRunSpec) -> DemoWiring:
    """Real runner and evaluator for one pilot run; ``synthetic`` is always False here."""
    if spec.method not in RUNNER_LABELS:
        raise ValueError(f"the pilot wiring implements the methods {sorted(RUNNER_LABELS)}, not {spec.method!r}")
    args = dict(spec.factory_args)
    reference = args.pop("reference_point", None)  # not a science argument: the runner needs it
    n_frozen = args.pop("n_frozen", FROZEN_ARCHITECTURES)  # nor are these two, of the sequential method
    stage_gpu_s = args.pop("stage_gpu_s", None)
    sentinel_every = args.pop("sentinel_every", None)  # the campaign plan asks for the sentinel network ...
    sentinel_limit = args.pop("sentinel_drift_limit", DRIFT_LIMIT)  # ... and says how much drift is too much
    sequential = spec.method == "sequential"
    if sequential:  # refused before the dataset is read
        if args.get("generations") is not None:
            raise ValueError("the sequential method is limited by GPU time per stage: generations must be null")
        args["generations"] = None
        if not representation_is_searchable(SLICES[args.get("slice", "fp32")]):
            raise ValueError(
                f"the sequential method needs a slice with a free representation or pruning gene to compress "
                f"with; slice {args.get('slice', 'fp32')!r} pins them all (use 'admitted', 'fp16_block' or 'fp16_regions')"
            )
    science = build_pilot_science(args)
    runner = science.runner
    if sequential or reference is not None:  # the science front built the plain runner: rebuild it with these
        config = science.search_config
        if reference is not None:
            config = replace(config, reference_point=tuple(reference))
        if sequential:
            runner = SequentialRunner(
                config,
                SequentialConfig(int(n_frozen), None if stage_gpu_s is None else tuple(stage_gpu_s)),
                SLICES[science.slice_name],
                ARCHITECTURE_SLICE,
                science.context_for,
            )
        else:
            runner = SearchRunner(config, SLICES[science.slice_name], science.context_for)
    runner_label = science.runner_label.replace("nsga2-joint/", f"{RUNNER_LABELS[spec.method]}/", 1)
    return DemoWiring(
        runner=runner,
        evaluator=science.evaluator,
        dataset=DatasetRegistration(
            science.dataset_id,
            science.dataset_manifest,
            {name: SplitDescriptor(**descriptor) for name, descriptor in science.splits.items()},
        ),
        space_hash=science.space_hash,
        runner_label=runner_label,
        evaluator_label=science.evaluator_label,
        synthetic=False,
        environment=science.environment,
        sentinel=None
        if sentinel_every is None
        else sentinel_hooks(_work_dir(spec.factory_args), int(sentinel_every), float(sentinel_limit)),
    )
