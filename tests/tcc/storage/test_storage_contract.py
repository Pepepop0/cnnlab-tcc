"""The SQLite adapter must satisfy the shared ports without changing them."""

from __future__ import annotations

import inspect

import pytest

from cnnlab.tcc import contracts
from cnnlab.tcc.storage import SQLiteRunStore
from cnnlab.tcc.storage.testing import FakeEvaluator

_PORT_METHODS = [
    name for name, member in vars(contracts.RunStore).items() if callable(member) and not name.startswith("_")
]


def test_port_method_list_is_complete() -> None:
    assert set(_PORT_METHODS) == {
        "create_run",
        "register_candidate",
        "find_cached",
        "commit_evaluation",
        "save_checkpoint",
        "load_checkpoint",
        "record_gpu_segment",
        "record_front",
    }


@pytest.mark.parametrize("name", _PORT_METHODS)
def test_sqlite_store_implements_port_signature(name: str) -> None:
    port = inspect.signature(getattr(contracts.RunStore, name))
    impl = inspect.signature(getattr(SQLiteRunStore, name))
    port_params = list(port.parameters.values())
    impl_params = list(impl.parameters.values())
    # Same positional parameters, in order; extras must be keyword-only with defaults.
    assert [p.name for p in impl_params[: len(port_params)]] == [p.name for p in port_params]
    for extra in impl_params[len(port_params) :]:
        assert extra.kind is inspect.Parameter.KEYWORD_ONLY and extra.default is not inspect.Parameter.empty


def test_fake_evaluator_matches_evaluator_port() -> None:
    port = inspect.signature(contracts.Evaluator.evaluate)
    impl = inspect.signature(FakeEvaluator.evaluate)
    assert list(port.parameters) == list(impl.parameters)


def test_storage_does_not_leak_into_contracts() -> None:
    source = inspect.getsource(contracts)
    assert "sqlite3" not in source and "cnnlab.tcc.storage" not in source
