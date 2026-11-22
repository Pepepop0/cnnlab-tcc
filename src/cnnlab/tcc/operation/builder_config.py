"""The builder's TensorRT timing cache in the campaign plan.

If the science front adopts a timing cache for the engine builder, the builds of the campaign must all use the same
one, so the plan names it: its path and the SHA-256 of its bytes. The plan field is ``builder.timing_cache``:

* ``null``: no timing cache (the default; nothing to wait for);
* ``{"adopted": false}``: the decision not to use one, recorded;
* ``{"adopted": true, "path": null, "sha256": null}``: adopted, the file still to come from the pilot; the plan cannot
  be frozen until both are set (pending ``builder.timing_cache``);
* ``{"adopted": true, "path": P, "sha256": H}``: complete. The file at ``P`` must hash to ``H`` when the plan is made
  and again when every run of the campaign starts, so a cache rewritten in between (TensorRT appends to it) is caught.

A relative ``path`` is resolved against the repository root. This module only freezes and checks the identity of the
file; handing the path to the builder is the wiring's job.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cnnlab.tcc.storage.errors import ConsistencyError

RULE = (
    "the TensorRT timing cache the builder reads, when the science front adopts one: every build of the campaign uses "
    "the file with this SHA-256; a run refuses to start if the file at the path has other bytes"
)
_KEYS = frozenset({"adopted", "path", "sha256"})


def normalize(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The settings value as the plan stores it; ``ValueError`` for anything inconsistent."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - _KEYS or "adopted" not in value:
        raise ValueError(
            f"timing_cache is null or a mapping with 'adopted' and, optionally, 'path' and 'sha256', not {value!r}"
        )
    adopted, path, digest = value["adopted"], value.get("path"), value.get("sha256")
    if not isinstance(adopted, bool):
        raise ValueError("timing_cache.adopted must be true or false")
    if not adopted and (path is not None or digest is not None):
        raise ValueError("a timing cache that is not adopted has no path or sha256")
    if (path is None) != (digest is None):
        raise ValueError(
            "timing_cache.path and timing_cache.sha256 come together: one names the file, the other its bytes"
        )
    if path is not None and not (isinstance(path, str) and path.strip()):
        raise ValueError("timing_cache.path must be a non-empty string")
    if digest is not None and not (
        isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("timing_cache.sha256 must be 64 lowercase hexadecimal characters")
    return {"adopted": adopted, "path": path, "sha256": digest}


def pending(timing_cache: Mapping[str, Any] | None) -> list[str]:
    """Plan fields still waiting for the pilot: an adopted cache without its file."""
    if timing_cache and timing_cache["adopted"] and timing_cache["path"] is None:
        return ["builder.timing_cache"]
    return []


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(timing_cache: Mapping[str, Any] | None, repo_root: Path) -> Path | None:
    """The cache file if the plan names one, after checking it is there with the frozen bytes; else ``None``."""
    if not timing_cache or not timing_cache.get("path"):
        return None
    path = Path(timing_cache["path"])
    if not path.is_absolute():
        path = Path(repo_root) / path
    if not path.is_file():
        raise ConsistencyError(f"the timing cache the plan names is not there: {path}")
    actual = file_sha256(path)
    if actual != timing_cache["sha256"]:
        raise ConsistencyError(
            f"the timing cache at {path} has SHA-256 {actual[:16]}…, the plan froze {str(timing_cache['sha256'])[:16]}…: "
            "the builds of the campaign would not share one cache"
        )
    return path
