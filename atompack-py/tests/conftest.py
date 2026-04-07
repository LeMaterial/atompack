# Copyright 2026 Entalpic
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_local_atompack_first_on_path() -> None:
    # The repo root contains a Rust crate named `atompack/` which Python can treat as a
    # namespace package when running tests from the monorepo root. Ensure we import the
    # actual Python package at `atompack-py/python/atompack` instead.
    package_root = Path(__file__).resolve().parents[1]
    python_src = package_root / "python"

    if python_src.is_dir():
        sys.path.insert(0, str(python_src))

    existing = sys.modules.get("atompack")
    if existing is None:
        return

    existing_file = getattr(existing, "__file__", None)
    if existing_file is None or str(python_src) not in str(existing_file):
        sys.modules.pop("atompack", None)


_ensure_local_atompack_first_on_path()
