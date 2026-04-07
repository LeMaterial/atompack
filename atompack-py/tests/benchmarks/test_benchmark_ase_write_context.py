# Copyright 2026 Entalpic
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np


class _FakeDb:
    def __init__(self) -> None:
        self.enter_calls = 0
        self.exit_calls = 0
        self.write_calls = 0

    def __enter__(self):
        self.enter_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.exit_calls += 1

    def write(self, atoms) -> None:
        self.write_calls += 1


class _FakeAtoms:
    def __init__(self, numbers, positions, cell, pbc) -> None:
        self.numbers = numbers
        self.positions = positions
        self.cell = cell
        self.pbc = pbc
        self.calc = None


class _FakeCalc:
    def __init__(self, atoms, energy, forces) -> None:
        self.atoms = atoms
        self.energy = energy
        self.forces = forces


def test_create_ase_db_uses_single_context_for_full_write_loop(
    monkeypatch,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("benchmark_ase_write_context", "benchmark.py")
    fake_db = _FakeDb()

    monkeypatch.setattr(
        module,
        "ase",
        SimpleNamespace(db=SimpleNamespace(connect=lambda path: fake_db)),
        raising=False,
    )
    monkeypatch.setattr(module, "Atoms", _FakeAtoms, raising=False)
    monkeypatch.setattr(module, "SinglePointCalculator", _FakeCalc, raising=False)
    monkeypatch.setattr(module, "_safe_close_ase_db", lambda db: None)
    monkeypatch.setattr(module, "gc", type("_GC", (), {"collect": staticmethod(lambda: None)})())
    monkeypatch.setattr(
        module,
        "generate_molecule_stream",
        lambda n, atoms, seed, with_custom: (
            {
                "atomic_numbers": np.ones(atoms, dtype=np.uint8),
                "positions": np.zeros((atoms, 3), dtype=np.float32),
                "cell": np.eye(3, dtype=np.float64),
                "pbc": np.array([True, True, True], dtype=bool),
                "energy": 0.0,
                "forces": np.zeros((atoms, 3), dtype=np.float32),
            }
            for _ in range(n)
        ),
    )

    module.create_ase_db("/tmp/fake.aselmdb", n=3, atoms=2)

    assert fake_db.enter_calls == 1
    assert fake_db.exit_calls == 1
    assert fake_db.write_calls == 3
