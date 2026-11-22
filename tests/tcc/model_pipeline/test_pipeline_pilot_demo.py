"""CPU checks of the pilot-demo composition: wiring arguments, bootstrap, resume rules and the SQL dump.

The GPU path is exercised by the demo command itself. Here the evaluator is
the synthetic ``FakeEvaluator`` of the storage front and the data are zeros,
so nothing below is a measurement.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest
import torch

from cnnlab.tcc.contracts import TCCEvent
from cnnlab.tcc.model_pipeline import pilot_demo
from cnnlab.tcc.model_pipeline.data_access import index_fingerprint
from cnnlab.tcc.model_pipeline.demo_data import DemoData
from cnnlab.tcc.model_pipeline.evaluator import EngineEvaluator, EvaluatorConfig, pilot_cache_context
from cnnlab.tcc.model_pipeline.pilot_demo import AttemptOpener, DemoRefused, bootstrap, dump_run, resume_state
from cnnlab.tcc.model_pipeline.pilot_wiring import (
    DEFAULTS,
    SCOPE_LABELS,
    PilotScience,
    resolve_arguments,
    split_descriptors,
)
from cnnlab.tcc.model_pipeline.trainer import TrainConfig
from cnnlab.tcc.science.operators import FP32_SLICE
from cnnlab.tcc.science.search import DirectContext, DirectHashes, DirectResume, SearchConfig, SearchRunner
from cnnlab.tcc.storage.sqlite_store import SQLiteRunStore
from cnnlab.tcc.storage.testing import FakeEvaluator

BACKEND = {"synthetic": True}


def fake_data() -> DemoData:
    x = torch.zeros(4, 3, 32, 32)
    y = torch.zeros(4, dtype=torch.int64)
    manifest = {
        "fingerprints": {"train": "a" * 64, "val": "b" * 64},
        "n_train_subset": 4,
        "n_val": 4,
        "official_test_read": False,
    }
    return DemoData(x, y, x, y, manifest)


def fake_science(tmp_path: Path, code_version: str = "test-code", generations: int = 1) -> PilotScience:
    """Real runner over the FP32 slice; the evaluator object is never called in these tests."""
    data = fake_data()
    evaluator_config = EvaluatorConfig(train=TrainConfig(epochs=1))
    search_config = SearchConfig(population=2, generations=generations)
    space_hash = FP32_SLICE.space_hash()

    def context_for(job: Any) -> Any:
        return pilot_cache_context(data, evaluator_config, job.train_seed, code_version, space_hash, BACKEND)

    return PilotScience(
        runner=SearchRunner(search_config, FP32_SLICE, context_for),
        evaluator=EngineEvaluator(data, tmp_path / "val.npz", evaluator_config, device="cpu"),
        data=data,
        dataset_id="cifar100-pilot-test",
        dataset_manifest=dict(data.manifest),
        splits=split_descriptors(data),
        space_hash=space_hash,
        runner_label="runner",
        evaluator_label="evaluator",
        environment={"backend": BACKEND},
        search_config=search_config,
        evaluator_config=evaluator_config,
        code_version=code_version,
        arguments={"cifar_root": "x", "work_dir": "y", "epochs": 1},
    )


def demo_args(**over: Any) -> argparse.Namespace:
    base = {"run_id": "run-demo-cpu", "search_seed": 1001, "train_seed": 2001, "budget_gpu_s": 600.0}
    return argparse.Namespace(**{**base, **over})


def direct_context(
    store: SQLiteRunStore,
    job: Any,
    evaluator: Any,
    resume: DirectResume,
    events: list[TCCEvent],
    should_stop: Any = lambda: False,
) -> DirectContext:
    hashes = store.run_hashes(job.run_id)
    return DirectContext(
        job=job,
        store=store,
        evaluator=AttemptOpener(evaluator, store),
        resume=resume,
        hashes=DirectHashes(hashes.protocol_hash, hashes.plan_hash, hashes.space_hash),
        artifact_root=store.artifact_root,
        emit=events.append,
        should_stop=should_stop,
        active_gpu_seconds=lambda: store.active_gpu_seconds(job.run_id),
        unique_valid_evaluations=lambda: store.unique_valid_evaluations(job.run_id),
    )


# --- wiring arguments ------------------------------------------------------------------


def test_split_descriptors_describe_test_by_index_only() -> None:
    splits = split_descriptors(fake_data())
    assert set(splits) == {"train", "val", "test", "calibration"}
    assert splits["test"]["sha256"] == index_fingerprint("test", range(10_000))
    assert "sealed" in splits["test"]["ref"]
    assert splits["calibration"]["sha256"] == splits["train"]["sha256"]  # fp32 slice: no separate PTQ set


def test_wiring_arguments_are_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CNNLAB_TCC_CIFAR_ROOT", "CNNLAB_TCC_WORK_DIR"):
        monkeypatch.delenv(name, raising=False)
    merged = resolve_arguments({"cifar_root": "d", "work_dir": "w", "epochs": 3})
    assert merged["epochs"] == 3 and merged["population"] == DEFAULTS["population"]
    assert merged["device"] == "cuda"
    with pytest.raises(ValueError, match="device"):
        resolve_arguments({"cifar_root": "d", "work_dir": "w", "device": "tpu"})
    with pytest.raises(ValueError, match="unknown"):
        resolve_arguments({"cifar_root": "d", "work_dir": "w", "epoch": 3})
    with pytest.raises(ValueError, match="missing"):
        resolve_arguments({"work_dir": "w"})
    with pytest.raises(ValueError, match="int"):
        resolve_arguments({"cifar_root": "d", "work_dir": "w", "epochs": "5"})
    assert "not-confirmatory" in SCOPE_LABELS and "pilot" in SCOPE_LABELS


def test_machine_paths_may_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operation front freezes factory_args into the plan; paths given by environment stay out of it."""
    monkeypatch.setenv("CNNLAB_TCC_CIFAR_ROOT", "data-dir")
    monkeypatch.setenv("CNNLAB_TCC_WORK_DIR", "work-dir")
    merged = resolve_arguments({"epochs": 2})
    assert (merged["cifar_root"], merged["work_dir"]) == ("data-dir", "work-dir")
    assert resolve_arguments({"cifar_root": "explicit", "epochs": 2})["cifar_root"] == "explicit"


def test_building_the_science_wiring_never_initializes_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The operation front calls the factory in ``prepare`` and ``status``; only evaluation may use the GPU."""
    from test_pipeline_training import FakeTrainFile  # same directory

    from cnnlab.tcc.model_pipeline import pilot_wiring

    def forbidden(*_: Any, **__: Any) -> None:
        raise AssertionError("CUDA was initialized while building the wiring")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(pilot_wiring, "TorchvisionCifar100", lambda root: FakeTrainFile())
    science = pilot_wiring.build_pilot_science(
        {
            "cifar_root": str(tmp_path / "no-dataset-needed"),
            "work_dir": str(tmp_path / "work"),
            "train_per_class": 2,
            "epochs": 1,
            "population": 2,
            "generations": 1,
        }
    )
    assert not (tmp_path / "work" / "val.npz").exists()
    assert science.evaluator.data.x_train.device.type == "cpu"
    assert science.data.manifest["n_train_subset"] == 200 and science.data.manifest["official_test_read"] is False
    assert science.dataset_id.startswith("cifar100-pilot-") and science.synthetic is False
    assert science.space_hash == FP32_SLICE.space_hash()
    assert "preliminary" in science.evaluator_label and "nsga2" in science.runner_label
    assert science.arguments["train_per_class"] == 2
    assert "cifar_root" not in science.arguments and "work_dir" not in science.arguments
    assert str(tmp_path) not in str(science.dataset_manifest) + str(science.environment)
    # Same arguments, same identity; another subset, another dataset.
    again = pilot_wiring.build_pilot_science(
        {
            "cifar_root": str(tmp_path / "no-dataset-needed"),
            "work_dir": str(tmp_path / "work"),
            "train_per_class": 2,
            "epochs": 1,
            "population": 2,
            "generations": 1,
        }
    )
    other = pilot_wiring.build_pilot_science(
        {
            "cifar_root": str(tmp_path / "no-dataset-needed"),
            "work_dir": str(tmp_path / "work"),
            "train_per_class": 3,
            "epochs": 1,
            "population": 2,
            "generations": 1,
        }
    )
    assert again.dataset_id == science.dataset_id != other.dataset_id


def test_campaign_settings_full_split_and_no_generation_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_pipeline_training import FakeTrainFile  # same directory

    from cnnlab.tcc.model_pipeline import pilot_wiring

    monkeypatch.setattr(pilot_wiring, "TorchvisionCifar100", lambda root: FakeTrainFile())
    base = {
        "cifar_root": str(tmp_path / "no-dataset-needed"),
        "work_dir": str(tmp_path / "work"),
        "epochs": 1,
        "population": 30,
        "device": "cpu",
    }
    subset = pilot_wiring.build_pilot_science({**base, "train_per_class": 2, "generations": 1})
    campaign = pilot_wiring.build_pilot_science({**base, "train_per_class": None, "generations": None})

    assert campaign.search_config.generations is None and campaign.search_config.population == 30
    assert "gen=unlimited" in campaign.runner_label and "gen=1" in subset.runner_label
    assert campaign.search_config.definition()["generations"] is None
    assert "train-full-45k" in campaign.scope and "train-subset" not in campaign.scope
    assert "train-subset" in subset.scope and "train-full-45k" not in subset.scope
    assert subset.scope == pilot_wiring.SCOPE_LABELS  # the pilot scope did not change
    assert campaign.scope[0] == "pilot" and "not-confirmatory" in campaign.scope
    assert "train full split 45000" in campaign.evaluator_label and "train subset" in subset.evaluator_label
    assert campaign.splits["train"]["n_examples"] == 45000 and "full 45k" in campaign.splits["train"]["ref"]
    assert "same as the train split used" in campaign.splits["calibration"]["ref"]
    assert campaign.data.manifest["train_is_full_split"] is True
    assert campaign.dataset_id != subset.dataset_id and campaign.arguments["train_per_class"] is None
    assert list(campaign.environment["scope"]) == list(campaign.scope)

    # None is for those two settings only; anything else still has to be an int.
    for name in ("epochs", "population", "subset_seed"):
        with pytest.raises(ValueError, match=f"{name} must be an int"):
            pilot_wiring.resolve_arguments({**base, name: None})
    assert pilot_wiring.resolve_arguments({**base, "generations": None})["generations"] is None


def test_busy_gpu_makes_the_attempt_interrupted_not_invalid(tmp_path: Path) -> None:
    from test_pipeline_training import cpu_data, request_for  # same directory

    config = EvaluatorConfig(train=TrainConfig(epochs=1, batch_size=16), gpu_wait_s=0.2, gpu_poll_s=0.05)
    data = cpu_data()
    evaluator = EngineEvaluator(
        data,
        tmp_path / "val.npz",
        config,
        device="cpu",
        work_root=tmp_path / "work",
        gpu_guard=lambda: [{"pid": "77", "process_name": "python.exe"}],
    )
    result = evaluator.evaluate(request_for(3, tmp_path, config, data))
    assert result.quality["gpu_waits"][0]["gave_up"] is True
    assert (result.status, result.failure_stage, result.failure_code) == (
        "interrupted",
        "guard",
        "infrastructure_gpu_busy",
    )
    assert result.fitness is None and "train" not in result.durations_s


# --- bootstrap, run, resume ------------------------------------------------------------------


def test_bootstrap_run_and_sql_dump(tmp_path: Path) -> None:
    db, root = tmp_path / "db.sqlite", tmp_path / "artifacts"
    store = SQLiteRunStore(db, root)
    science = fake_science(tmp_path)
    job = bootstrap(store, science, demo_args(), {"probe": "synthetic"})
    plan = store.get_plan(job.plan_id)
    assert plan.status == "frozen" and plan.content["confirmatory"] is False
    assert plan.content["scope"] == list(SCOPE_LABELS) and plan.content["code_version"] == "test-code"
    assert job.stage == "pilot" and "test" not in store.dataset_splits_for_run(job.run_id)

    events: list[TCCEvent] = []
    assert science.runner(direct_context(store, job, FakeEvaluator(), DirectResume(), events)) == "completed"
    store.close()
    dump = dump_run(db, job.run_id)
    assert len(dump["candidates"]) == len(dump["attempts"]) == 4
    assert all(a["status"] == "valid" and a["e_fraction"] is not None for a in dump["attempts"])
    assert [(g["generation"], g["status"], g["attempts"]) for g in dump["by_generation"]] == [
        (0, "valid", 2),
        (1, "valid", 2),
    ]
    assert dump["checkpoints"] and "payload_json" not in dump["checkpoints"][0]
    assert dump["front_snapshots"] and dump["front_members"]
    assert dump["gpu_seconds_total"] > 0 and dump["integrity_check"] == "ok"


def test_killed_attempt_is_closed_as_interrupted_and_retried_on_resume(tmp_path: Path) -> None:
    db, root = tmp_path / "db.sqlite", tmp_path / "artifacts"
    store = SQLiteRunStore(db, root)
    science = fake_science(tmp_path, generations=2)
    job = bootstrap(store, science, demo_args(), {})

    class Dies(FakeEvaluator):
        def evaluate(self, request: Any) -> Any:
            if len(self.calls) == 2:
                raise KeyboardInterrupt("process killed during the third attempt")
            return super().evaluate(request)

    with pytest.raises(KeyboardInterrupt):
        science.runner(direct_context(store, job, Dies(), DirectResume(), []))
    assert [a.status for a in store.list_attempts(job.run_id)] == ["valid", "valid", "running"]
    store.close()

    # New session on the same database: the running attempt is closed and the candidate gets attempt 2.
    store = SQLiteRunStore(db, root)
    resume = resume_state(store, science, job)
    # Unresolved: the interrupted candidate and its sibling that was never attempted.
    assert resume.checkpoint is not None
    assert resume.retry_candidate_ids == (f"{job.run_id}.g001.c0002", f"{job.run_id}.g001.c0003")
    assert resume.attempts_per_candidate[f"{job.run_id}.g001.c0002"] == 1
    events: list[TCCEvent] = []
    healthy = FakeEvaluator()
    assert science.runner(direct_context(store, job, healthy, resume, events)) == "completed"
    statuses = [(a.attempt_id.rsplit(".", 2)[-2:], a.status) for a in store.list_attempts(job.run_id)]
    assert (["c0002", "a1"], "interrupted") in statuses and (["c0002", "a2"], "valid") in statuses
    assert [e.kind for e in events].count("candidate_adopted") == 0  # checkpoint of generation 1 was current
    assert store.check_integrity(hash_files=True).ok
    assert len(store.run_history(job.run_id).candidates) == 6
    store.close()


def test_resume_refuses_other_code_or_configuration(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "db.sqlite", tmp_path / "artifacts")
    job = bootstrap(store, fake_science(tmp_path), demo_args(), {})
    with pytest.raises(DemoRefused, match="started with code"):
        resume_state(store, fake_science(tmp_path, code_version="other-code"), job)
    with pytest.raises(DemoRefused, match="search configuration"):
        resume_state(store, fake_science(tmp_path, generations=3), job)
    resume_state(store, fake_science(tmp_path), job)  # same code and configuration: accepted
    store.close()


def test_exit_codes_cover_every_status() -> None:
    assert pilot_demo.EXIT_CODES == {"completed": 0, "stopped": 0, "blocked": 3, "refused": 5}


def test_slice_argument_selects_the_search_slice_and_its_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_pipeline_training import FakeTrainFile  # same directory

    from cnnlab.tcc.model_pipeline import pilot_wiring
    from cnnlab.tcc.science.operators import FP16_BLOCK_SLICE

    monkeypatch.setattr(pilot_wiring, "TorchvisionCifar100", lambda root: FakeTrainFile())
    base = {
        "cifar_root": str(tmp_path / "unused"),
        "work_dir": str(tmp_path / "work"),
        "train_per_class": 1,
        "epochs": 1,
        "population": 2,
        "generations": 1,
        "code_version": "test",
    }
    fp32 = pilot_wiring.build_pilot_science(base)
    fp16 = pilot_wiring.build_pilot_science({**base, "slice": "fp16_block"})
    assert fp32.slice_name == "fp32" and fp32.scope == SCOPE_LABELS and fp32.space_hash == FP32_SLICE.space_hash()
    assert fp16.slice_name == "fp16_block" and fp16.space_hash == FP16_BLOCK_SLICE.space_hash()
    assert "fp16-per-block-unpruned-slice" in fp16.scope and "fp32-unpruned-slice" not in fp16.scope
    assert {"pilot", "preliminary-L-V", "not-confirmatory"} <= set(fp16.scope)
    assert fp16.environment["slice"] == "fp16_block" and "slice=fp16_block" in fp16.runner_label
    assert fp16.dataset_id == fp32.dataset_id  # same data, another search space
    with pytest.raises(ValueError, match="slice"):
        pilot_wiring.resolve_arguments({**base, "slice": "int8"})
