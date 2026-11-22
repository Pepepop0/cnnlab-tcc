"""GPU lock shared by the pilot tools: ``%TEMP%/cnnlab-gpu.lock``, one GPU job at a time.

The lock records the owner's PID. A lock whose PID no longer runs is an orphan (its
process was killed, e.g. when the session that launched it ended) and is removed instead
of blocking everyone. A lock without a PID (written by another tool) is respected as is.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

LOCK = Path(tempfile.gettempdir()) / "cnnlab-gpu.lock"


def pid_alive(pid: int) -> bool:
    """True while process ``pid`` runs (Windows: exit code still STILL_ACTIVE)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_pid(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("pid="):
            try:
                return int(line[4:].strip())
            except ValueError:
                return None
    return None


def take(owner: str, job: str, *, wait_s: float = 0.0, lock: Path = LOCK) -> None:
    """Take the lock or exit with the holder's description; an orphan lock is removed first.

    ``wait_s`` > 0 waits that long for a live holder to release it.
    """
    text = f"owner={owner}\npid={os.getpid()}\nsince={datetime.now().astimezone().isoformat()}\njob={job}\n"
    deadline = time.monotonic() + wait_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = lock.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            pid = lock_pid(held)
            if pid is not None and not pid_alive(pid):
                print(f"trava orfa removida (pid {pid} nao roda mais):\n{held}", file=sys.stderr)
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() < deadline:
                time.sleep(15)
                continue
            raise SystemExit(f"GPU travada por outro job:\n{held}") from None
        os.write(fd, text.encode("utf-8"))
        os.close(fd)
        return


def release(lock: Path = LOCK) -> None:
    """Remove the lock if this process holds it (never someone else's)."""
    try:
        held = lock.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    if lock_pid(held) == os.getpid():
        lock.unlink(missing_ok=True)
