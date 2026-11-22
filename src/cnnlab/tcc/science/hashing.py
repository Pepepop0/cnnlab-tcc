"""Canonical JSON and SHA-256 identities for genomes, spaces and cache contexts.

Every hash in the science layer goes through :func:`digest`, so two logically
equal payloads always produce the same bytes regardless of dict insertion
order, tuple/list choice or ``-0.0``. Types without an exact JSON form (numpy
integers, paths, sets) are rejected instead of being stringified silently.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize JSON-native data deterministically: sorted keys, no spaces, no NaN."""
    return json.dumps(
        _normalize(value, "$"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_hex(text: str) -> str:
    """SHA-256 of the UTF-8 encoding of ``text``, as 64 lowercase hex chars."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(kind: str, payload: Any) -> str:
    """Domain-separated hash: the same payload under another ``kind`` never collides."""
    if not kind:
        raise ValueError("digest kind must be a non-empty string")
    return sha256_hex(canonical_json({"kind": kind, "payload": payload}))


def is_sha256_hex(value: object) -> bool:
    """True for 64 lowercase hexadecimal characters."""
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _normalize(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if type(value) is int:
        return value
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path}: non-finite float is not canonical JSON")
        return 0.0 if number == 0.0 else number
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: mapping keys must be str, got {type(key).__name__}")
            out[key] = _normalize(item, f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_normalize(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise TypeError(f"{path}: {type(value).__name__} has no canonical JSON form")
