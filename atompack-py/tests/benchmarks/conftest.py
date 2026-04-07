# Copyright 2026 Entalpic
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import ModuleType as PythonModuleType
from typing import Callable

import atompack
import numpy as np
import pytest


BENCHMARKS_ROOT = Path(__file__).resolve().parents[2] / "benchmarks"


def _load_benchmark_module(module_name: str, file_name: str) -> PythonModuleType:
    module_path = BENCHMARKS_ROOT / file_name
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_atompack_dataset(path: Path, n_molecules: int, n_atoms: int) -> None:
    db = atompack.Database(str(path), compression="none", overwrite=True)
    for _ in range(n_molecules):
        mol = atompack.Molecule.from_arrays(
            np.zeros((n_atoms, 3), dtype=np.float32),
            np.ones(n_atoms, dtype=np.uint8),
        )
        db.add_molecule(mol)
    db.flush()


@pytest.fixture
def load_benchmark_module() -> Callable[[str, str], PythonModuleType]:
    return _load_benchmark_module


@pytest.fixture
def write_atompack_dataset() -> Callable[[Path, int, int], None]:
    return _write_atompack_dataset


@pytest.fixture
def stub_tqdm(monkeypatch) -> None:
    fake_tqdm = ModuleType("tqdm")
    fake_tqdm.tqdm = lambda it, **kwargs: it
    fake_tqdm.write = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
