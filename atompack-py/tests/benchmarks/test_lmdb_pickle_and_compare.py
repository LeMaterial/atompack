# Copyright 2026 Entalpic
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("lmdb")


def test_pickle_lmdb_roundtrip_with_optional_fields(tmp_path: Path, load_benchmark_module) -> None:
    atom_lmdb_pickle = load_benchmark_module("atom_lmdb_pickle", "atom_lmdb_pickle.py")
    store = atom_lmdb_pickle.PickleAtomLMDB(
        tmp_path / "pickle_store.lmdb",
        atom_lmdb_pickle.PickleLMDBConfig(codec="none"),
    )
    store.reset_dir()
    env = store.open_env(map_size=64 * 1024 * 1024, readonly=False, lock=False)

    positions = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    atomic_numbers = np.array([6, 8], dtype=np.uint8)
    cell = np.eye(3, dtype=np.float64) * 2.0
    pbc = np.array([1, 0, 1], dtype=np.uint8)
    forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    with env.begin(write=True) as txn:
        store.put_molecule(
            txn,
            idx=0,
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            energy=-1.234,
            forces=forces,
        )
    with env.begin(write=False) as txn:
        payload = store.get_payload(txn, 0)
    env.close()

    assert payload is not None
    np.testing.assert_allclose(payload["positions"], positions)
    np.testing.assert_array_equal(payload["atomic_numbers"], atomic_numbers)
    np.testing.assert_allclose(payload["cell"], cell)
    np.testing.assert_array_equal(payload["pbc"], pbc)
    np.testing.assert_allclose(payload["forces"], forces)
    assert float(payload["energy"]) == -1.234
