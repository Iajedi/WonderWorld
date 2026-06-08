"""Helpers for importing modules from external baseline checkouts."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path

from eval.paths import REPO_ROOT


@contextlib.contextmanager
def checkout_sys_path(checkout: Path) -> Iterator[None]:
    """Prepend ``checkout`` and hide repo-root ``utils`` package shadowing."""
    checkout_str = str(checkout.resolve())
    repo_root = str(REPO_ROOT.resolve())
    old_path = sys.path.copy()
    stale_utils = [
        name
        for name in list(sys.modules)
        if name == "utils" or name.startswith("utils.")
    ]
    stale_diffusers = [
        name
        for name in list(sys.modules)
        if name == "diffusers" or name.startswith("diffusers.")
    ]
    for name in (*stale_utils, *stale_diffusers):
        del sys.modules[name]

    filtered: list[str] = []
    seen: set[str] = set()
    for entry in old_path:
        if not entry or entry == ".":
            continue
        try:
            resolved = str(Path(entry).resolve())
        except OSError:
            resolved = entry
        if resolved == repo_root or resolved in seen:
            continue
        seen.add(resolved)
        filtered.append(entry)

    sys.path[:] = [checkout_str, *filtered]
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name in (*stale_utils, *stale_diffusers):
            sys.modules.pop(name, None)
