# Copyright 2026 Entalpic
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("h5py")


def test_hdf5_soa_get_payload_reads_one_item(tmp_path: Path, load_benchmark_module) -> None:
    module = load_benchmark_module("atom_hdf5_soa_test_payload", "atom_hdf5_soa.py")

    cfg = module.AtomHdf5SoaConfig(
        atoms_per_molecule=2,
        with_props=True,
        with_cell_pbc=True,
        chunk_size=3,
    )
    store = module.AtomHdf5Soa(tmp_path / "data.h5", cfg)
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

    reader = store.open_reader()
    try:
        payload = store.get_payload(reader, 3)
        np.testing.assert_array_equal(payload["positions"], positions[3])
        np.testing.assert_array_equal(payload["forces"], forces[3])
        np.testing.assert_array_equal(payload["cell"], cell[3])
        np.testing.assert_array_equal(payload["pbc"], pbc[3].astype(bool))
        assert float(payload["energy"]) == 3.5
    finally:
        reader.close()


def test_hdf5_soa_get_payloads_preserves_requested_order(
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("atom_hdf5_soa_test_payloads", "atom_hdf5_soa.py")

    cfg = module.AtomHdf5SoaConfig(
        atoms_per_molecule=2,
        with_props=True,
        with_cell_pbc=False,
    )
    store = module.AtomHdf5Soa(tmp_path / "data.h5", cfg)
    store.create_file(5)

    positions = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    atomic_numbers = np.full((5, 2), 6, dtype=np.uint8)
    energy = np.arange(5, dtype=np.float64) + 1.0
    forces = (positions + 50).astype(np.float32)

    with store.open_file("r+") as handle:
        store.put_batch(
            handle,
            0,
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=None,
            pbc=None,
            energy=energy,
            forces=forces,
        )

    reader = store.open_reader()
    try:
        payloads = store.get_payloads(reader, [4, 1, 3])
        assert [float(payload["energy"]) for payload in payloads] == [5.0, 2.0, 4.0]
        np.testing.assert_array_equal(payloads[0]["positions"], positions[4])
        np.testing.assert_array_equal(payloads[1]["forces"], forces[1])
        np.testing.assert_array_equal(payloads[2]["atomic_numbers"], atomic_numbers[3])
    finally:
        reader.close()


def test_hdf5_soa_custom_properties_roundtrip(tmp_path: Path, load_benchmark_module) -> None:
    module = load_benchmark_module("atom_hdf5_soa_test_custom", "atom_hdf5_soa.py")

    cfg = module.AtomHdf5SoaConfig(
        atoms_per_molecule=3,
        with_props=True,
        with_cell_pbc=True,
        chunk_size=2,
    )
    store = module.AtomHdf5Soa(tmp_path / "custom.h5", cfg)
    store.create_file(4)

    positions = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
    atomic_numbers = np.full((4, 3), 6, dtype=np.uint8)
    energy = np.arange(4, dtype=np.float64) + 0.25
    forces = (positions + 10).astype(np.float32)
    cell = np.arange(4 * 3 * 3, dtype=np.float64).reshape(4, 3, 3)
    pbc = np.ones((4, 3), dtype=np.uint8)
    properties = {
        "bandgap": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "eigenvalues": np.arange(4 * 5, dtype=np.float64).reshape(4, 5),
        "phase": np.array(["train", "valid", "test", "holdout"]),
    }
    atom_properties = {
        "mulliken_charges": np.arange(4 * 3, dtype=np.float64).reshape(4, 3),
        "vectors": np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3),
    }

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
            properties=properties,
            atom_properties=atom_properties,
        )

    reader = store.open_reader()
    try:
        payload = store.get_payload(reader, 2)
        assert float(payload["energy"]) == pytest.approx(2.25)
        assert payload["properties"]["phase"] == "test"
        assert float(payload["properties"]["bandgap"]) == pytest.approx(3.0)
        np.testing.assert_array_equal(
            payload["properties"]["eigenvalues"],
            properties["eigenvalues"][2],
        )
        np.testing.assert_array_equal(
            payload["atom_properties"]["mulliken_charges"],
            atom_properties["mulliken_charges"][2],
        )
        np.testing.assert_array_equal(
            payload["atom_properties"]["vectors"],
            atom_properties["vectors"][2],
        )
    finally:
        reader.close()


def test_hdf5_soa_get_payloads_roundtrip_with_custom_properties(
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("atom_hdf5_soa_test_custom_payloads", "atom_hdf5_soa.py")

    cfg = module.AtomHdf5SoaConfig(
        atoms_per_molecule=3,
        with_props=True,
        with_cell_pbc=True,
        chunk_size=2,
    )
    store = module.AtomHdf5Soa(tmp_path / "custom_payloads.h5", cfg)
    store.create_file(4)

    positions = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
    atomic_numbers = np.full((4, 3), 6, dtype=np.uint8)
    energy = np.arange(4, dtype=np.float64) + 0.25
    forces = (positions + 10).astype(np.float32)
    cell = np.arange(4 * 3 * 3, dtype=np.float64).reshape(4, 3, 3)
    pbc = np.ones((4, 3), dtype=np.uint8)
    properties = {
        "bandgap": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "eigenvalues": np.arange(4 * 5, dtype=np.float64).reshape(4, 5),
        "phase": np.array(["train", "valid", "test", "holdout"]),
    }
    atom_properties = {
        "mulliken_charges": np.arange(4 * 3, dtype=np.float64).reshape(4, 3),
        "vectors": np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3),
    }

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
            properties=properties,
            atom_properties=atom_properties,
        )

    reader = store.open_reader()
    try:
        payloads = store.get_payloads(reader, [3, 0, 2])
        assert [float(payload["energy"]) for payload in payloads] == pytest.approx([3.25, 0.25, 2.25])
        assert [payload["properties"]["phase"] for payload in payloads] == ["holdout", "train", "test"]
        np.testing.assert_array_equal(payloads[0]["positions"], positions[3])
        np.testing.assert_array_equal(payloads[1]["properties"]["eigenvalues"], properties["eigenvalues"][0])
        np.testing.assert_array_equal(
            payloads[2]["atom_properties"]["mulliken_charges"],
            atom_properties["mulliken_charges"][2],
        )
    finally:
        reader.close()
