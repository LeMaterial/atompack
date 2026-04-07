# Copyright 2026 Entalpic
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _FakeDb:
    def __init__(self) -> None:
        self.enter_calls = 0
        self.exit_calls = 0
        self.write_calls = 0
        self.disconnect_calls = 0

    def __enter__(self):
        self.enter_calls += 1
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.exit_calls += 1

    def write(self, atoms) -> None:
        self.write_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1


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


class _FakeTqdm:
    def __call__(self, it, **kwargs):
        return it

    def write(self, *args, **kwargs) -> None:
        pass


def test_scaling_benchmark_ase_writer_uses_single_context(
    monkeypatch,
    stub_tqdm,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("scaling_benchmark_ase_write_context", "scaling_benchmark.py")
    fake_db = _FakeDb()

    monkeypatch.setattr(
        module,
        "ase",
        SimpleNamespace(db=SimpleNamespace(connect=lambda path: fake_db)),
        raising=False,
    )
    monkeypatch.setattr(module, "Atoms", _FakeAtoms, raising=False)
    monkeypatch.setattr(module, "SinglePointCalculator", _FakeCalc, raising=False)
    monkeypatch.setattr(module, "_has_ase", True)

    module._write_ase_db(
        Path("/tmp/fake.aselmdb"),
        positions=np.zeros((2, 3, 3), dtype=np.float32),
        atomic_numbers=np.ones((2, 3), dtype=np.uint8),
        cell=np.tile(np.eye(3, dtype=np.float64)[None, :, :], (2, 1, 1)),
        pbc=np.zeros((2, 3), dtype=np.uint8),
        energy=np.zeros(2, dtype=np.float64),
        forces=np.zeros((2, 3, 3), dtype=np.float32),
        num=5,
    )

    assert fake_db.enter_calls == 1
    assert fake_db.exit_calls == 1
    assert fake_db.write_calls == 5
    assert fake_db.disconnect_calls == 1


def test_run_scaling_records_current_non_ase_backends(
    monkeypatch,
    tmp_path: Path,
    stub_tqdm,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("scaling_benchmark_current_backends", "scaling_benchmark.py")
    monkeypatch.setattr(module, "tqdm", _FakeTqdm())

    templates = module.Templates(
        molecules=[],
        positions=np.zeros((2, 3, 3), dtype=np.float32),
        atomic_numbers=np.ones((2, 3), dtype=np.uint8),
        cell=np.tile(np.eye(3, dtype=np.float64)[None, :, :], (2, 1, 1)),
        pbc=np.zeros((2, 3), dtype=np.uint8),
        energy=np.zeros(2, dtype=np.float64),
        forces=np.zeros((2, 3, 3), dtype=np.float32),
        stress=np.zeros((2, 3, 3), dtype=np.float64),
        charges=np.zeros((2, 3), dtype=np.float64),
    )

    monkeypatch.setattr(module, "_build_templates", lambda *args, **kwargs: templates)
    monkeypatch.setattr(module, "_write_atompack", lambda path, molecules, num: None)
    monkeypatch.setattr(module, "_write_packed_lmdb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_write_pickle_lmdb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_warmup", lambda db, num: None)
    monkeypatch.setattr(module.atompack.Database, "open", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "_bench_atompack_subprocess", lambda *args, **kwargs: [10.0])
    monkeypatch.setattr(module, "_bench_packed_lmdb_batch", lambda *args, **kwargs: [30.0])
    monkeypatch.setattr(module, "_bench_pickle_lmdb_batch", lambda *args, **kwargs: [40.0])
    monkeypatch.setattr(module, "_file_size_bytes", lambda path: 0)
    monkeypatch.setattr(module, "_has_ase", False)

    result = module.run_scaling(
        atoms_sizes=[3],
        thread_counts=[1],
        num_molecules=4,
        batch_size=2,
        num_trials=1,
        scratch_dir=tmp_path,
    )

    backends = {row["backend"] for row in result["results"]}
    assert backends == {"atompack", "lmdb_packed", "lmdb_pickle"}
