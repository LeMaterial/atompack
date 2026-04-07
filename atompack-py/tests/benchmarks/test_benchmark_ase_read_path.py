# Copyright 2026 Entalpic
from __future__ import annotations

import numpy as np


class _FallbackAtoms:
    def __init__(self) -> None:
        self.positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    def get_forces(self) -> np.ndarray:
        return np.array([[4.0, 5.0, 6.0]], dtype=np.float32)

    def get_potential_energy(self) -> float:
        return 7.0


class _FallbackRow:
    def __init__(self) -> None:
        self.toatoms_calls = 0

    def toatoms(self) -> _FallbackAtoms:
        self.toatoms_calls += 1
        return _FallbackAtoms()


def test_touch_ase_row_prefers_direct_row_arrays(load_benchmark_module) -> None:
    module = load_benchmark_module("benchmark_ase_read_path_direct", "benchmark.py")

    class _DirectRow:
        def __init__(self) -> None:
            self.positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
            self.forces = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
            self.energy = 7.0
            self.toatoms_calls = 0

        def toatoms(self):
            self.toatoms_calls += 1
            raise AssertionError("direct row arrays should avoid toatoms()")

    row = _DirectRow()

    module._touch_ase_row(row, 0)

    assert row.toatoms_calls == 0


def test_touch_ase_row_falls_back_to_atoms_when_results_missing(load_benchmark_module) -> None:
    module = load_benchmark_module("benchmark_ase_read_path_fallback", "benchmark.py")
    row = _FallbackRow()

    module._touch_ase_row(row, 0)

    assert row.toatoms_calls == 1
