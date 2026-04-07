# Copyright 2026 Entalpic
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("h5py")


def test_chunked_reader_matches_naive_payloads(tmp_path: Path, load_benchmark_module) -> None:
    module = load_benchmark_module("hdf5_tuned_experiment_test", "hdf5_tuned_experiment.py")
    atom_hdf5 = load_benchmark_module("atom_hdf5_soa_for_tuned_test", "atom_hdf5_soa.py")

    cfg = atom_hdf5.AtomHdf5SoaConfig(
        atoms_per_molecule=2,
        with_props=True,
        with_cell_pbc=True,
        chunk_size=3,
    )
    store = atom_hdf5.AtomHdf5Soa(tmp_path / "data.h5", cfg)
    store.create_file(6)

    positions = np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3)
    atomic_numbers = (np.arange(6 * 2, dtype=np.uint8).reshape(6, 2) + 1)
    cell = np.arange(6 * 3 * 3, dtype=np.float64).reshape(6, 3, 3)
    pbc = (np.arange(6 * 3, dtype=np.uint8).reshape(6, 3) % 2).astype(np.uint8)
    energy = np.arange(6, dtype=np.float64) + 0.5
    forces = (positions + 100).astype(np.float32)

    with store.open_file("r+") as handle:
        store.put_batch(
            handle,
            0,
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            energy=energy,
            forces=forces,
        )

    naive_reader = store.open_reader()
    chunked_reader = module.ChunkedHdf5Reader(tmp_path / "data.h5", chunk_size=3, cache_chunks=2)
    try:
        indices = [4, 1, 3]
        expected = [store.get_payload(naive_reader, idx) for idx in indices]
        actual = chunked_reader.get_payloads_chunked(indices)

        assert [float(payload["energy"]) for payload in actual] == [4.5, 1.5, 3.5]
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got["positions"], want["positions"])
            np.testing.assert_array_equal(got["forces"], want["forces"])
            np.testing.assert_array_equal(got["cell"], want["cell"])
            np.testing.assert_array_equal(got["atomic_numbers"], want["atomic_numbers"])
            np.testing.assert_array_equal(got["pbc"], want["pbc"])
    finally:
        naive_reader.close()
        chunked_reader.close()
