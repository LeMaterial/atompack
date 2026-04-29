# Copyright 2026 Entalpic
from __future__ import annotations

from pathlib import Path
import pickle

import atompack
import numpy as np
import pytest


def _make_molecule(energy: float) -> atompack.Molecule:
    mol = atompack.Molecule(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([6, 8], dtype=np.uint8),
    )
    mol.energy = float(energy)
    mol.forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    mol.charges = np.array([-0.1, 0.1], dtype=np.float64)
    mol.velocities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mol.cell = np.eye(3, dtype=np.float64) * 2.0
    mol.set_property("tag", "train")
    mol.set_property("ids", np.array([1, 2], dtype=np.int64))
    return mol


@pytest.mark.parametrize("compression", ["none", "lz4", "zstd"])
def test_database_roundtrip(tmp_path: Path, compression: str) -> None:
    path = tmp_path / f"molecules_{compression}.atp"

    mol1 = _make_molecule(-1.0)
    mol2 = _make_molecule(-2.0)

    db = atompack.Database(str(path), compression=compression)
    db.add_molecules([mol1, mol2])
    db.flush()

    assert path.exists()
    assert path.stat().st_size > 0

    db2 = atompack.Database.open(str(path))
    assert len(db2) == 2

    mol1_r = db2[0]
    assert mol1_r.energy == pytest.approx(-1.0)
    np.testing.assert_allclose(mol1_r.forces, mol1.forces)
    np.testing.assert_allclose(mol1_r.charges, mol1.charges)
    np.testing.assert_allclose(mol1_r.velocities, mol1.velocities)
    np.testing.assert_allclose(mol1_r.cell, mol1.cell)
    assert mol1_r.get_property("tag") == "train"
    np.testing.assert_array_equal(mol1_r.get_property("ids"), np.array([1, 2], dtype=np.int64))

    batch = db2.get_molecules([0, 1])
    assert [m.energy for m in batch] == pytest.approx([-1.0, -2.0])

    db3 = atompack.Database.open(str(path), mmap=True)
    assert len(db3) == 2
    assert db3[1].energy == pytest.approx(-2.0)


def test_database_rejects_invalid_compression(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Invalid compression"):
        atompack.Database(str(tmp_path / "bad.atp"), compression="definitely-not-a-codec")


def test_database_roundtrip_from_arrays_with_builtins(tmp_path: Path) -> None:
    path = tmp_path / "from_arrays_builtins.atp"
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    atomic_numbers = np.array([6, 8], dtype=np.uint8)
    forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    cell = np.eye(3, dtype=np.float64) * 3.0

    mol = atompack.Molecule.from_arrays(
        positions,
        atomic_numbers,
        energy=-7.5,
        forces=forces,
        cell=cell,
    )

    db = atompack.Database(str(path))
    db.add_molecule(mol)
    db.flush()

    reopened = atompack.Database.open(str(path))
    read = reopened[0]
    assert read.energy == pytest.approx(-7.5)
    np.testing.assert_allclose(read.positions, positions)
    np.testing.assert_array_equal(read.atomic_numbers, atomic_numbers)
    np.testing.assert_allclose(read.forces, forces)
    np.testing.assert_allclose(read.cell, cell)

def test_database_add_molecules_roundtrip_from_arrays_with_builtins(tmp_path: Path) -> None:
    path = tmp_path / "from_arrays_add_molecules.atp"
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    atomic_numbers = np.array([6, 8], dtype=np.uint8)
    forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    cell = np.eye(3, dtype=np.float64) * 3.0

    mol1 = atompack.Molecule.from_arrays(
        positions,
        atomic_numbers,
        energy=-7.5,
        forces=forces,
        cell=cell,
    )
    mol2 = atompack.Molecule.from_arrays(
        positions + np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
        atomic_numbers,
        energy=-8.5,
        forces=forces * 2.0,
        cell=cell * 2.0,
    )

    db = atompack.Database(str(path))
    db.add_molecules([mol1, mol2])
    db.flush()

    reopened = atompack.Database.open(str(path))
    assert len(reopened) == 2
    first = reopened[0]
    second = reopened[1]

    assert first.energy == pytest.approx(-7.5)
    np.testing.assert_allclose(first.positions, positions)
    np.testing.assert_array_equal(first.atomic_numbers, atomic_numbers)
    np.testing.assert_allclose(first.forces, forces)
    np.testing.assert_allclose(first.cell, cell)

    assert second.energy == pytest.approx(-8.5)
    np.testing.assert_allclose(second.positions, positions + np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_array_equal(second.atomic_numbers, atomic_numbers)
    np.testing.assert_allclose(second.forces, forces * 2.0)
    np.testing.assert_allclose(second.cell, cell * 2.0)


def test_database_add_arrays_batch_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "batch_arrays.atp"
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    atomic_numbers = np.array([[6, 8], [1, 8]], dtype=np.uint8)
    energy = np.array([-1.5, -2.5], dtype=np.float64)
    forces = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ],
        dtype=np.float32,
    )
    cell = np.stack([np.eye(3), np.eye(3) * 2.0]).astype(np.float64)
    pbc = np.array([[True, False, True], [False, True, False]], dtype=bool)

    db = atompack.Database(str(path))
    db.add_arrays_batch(
        positions,
        atomic_numbers,
        energy=energy,
        forces=forces,
        cell=cell,
        pbc=pbc,
        name=["mol-a", "mol-b"],
    )
    db.flush()

    reopened = atompack.Database.open(str(path))
    assert len(reopened) == 2
    first = reopened[0]
    second = reopened[1]

    np.testing.assert_allclose(first.positions, positions[0])
    np.testing.assert_array_equal(first.atomic_numbers, atomic_numbers[0])
    assert first.energy == pytest.approx(-1.5)
    np.testing.assert_allclose(first.forces, forces[0])
    np.testing.assert_allclose(first.cell, cell[0])
    assert first.pbc == (True, False, True)

    np.testing.assert_allclose(second.positions, positions[1])
    np.testing.assert_array_equal(second.atomic_numbers, atomic_numbers[1])
    assert second.energy == pytest.approx(-2.5)
    np.testing.assert_allclose(second.forces, forces[1])
    np.testing.assert_allclose(second.cell, cell[1])
    assert second.pbc == (False, True, False)


def test_database_add_arrays_batch_roundtrip_with_custom_properties(tmp_path: Path) -> None:
    path = tmp_path / "batch_arrays_custom.atp"
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    atomic_numbers = np.array([[6, 8], [1, 8]], dtype=np.uint8)
    energy = np.array([-1.5, -2.5], dtype=np.float64)
    forces = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ],
        dtype=np.float32,
    )

    db = atompack.Database(str(path))
    db.add_arrays_batch(
        positions,
        atomic_numbers,
        energy=energy,
        forces=forces,
        properties={
            "bandgap": np.array([1.5, 2.5], dtype=np.float64),
            "eigenvalues": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
            "phase": ["train", "valid"],
        },
        atom_properties={
            "mulliken_charges": np.array([[0.2, -0.2], [0.1, -0.1]], dtype=np.float64),
        },
    )
    db.flush()

    reopened = atompack.Database.open(str(path))
    first = reopened[0]
    second = reopened[1]

    assert first.energy == pytest.approx(-1.5)
    assert first.get_property("bandgap") == pytest.approx(1.5)
    np.testing.assert_allclose(
        first.get_property("eigenvalues"),
        np.array([0.1, 0.2], dtype=np.float64),
    )
    assert first.get_property("phase") == "train"
    np.testing.assert_allclose(
        reopened.get_molecules_flat([0, 1])["atom_properties"]["mulliken_charges"],
        np.array([0.2, -0.2, 0.1, -0.1], dtype=np.float64),
    )

    assert second.energy == pytest.approx(-2.5)
    assert second.get_property("bandgap") == pytest.approx(2.5)
    np.testing.assert_allclose(
        second.get_property("eigenvalues"),
        np.array([0.3, 0.4], dtype=np.float64),
    )
    assert second.get_property("phase") == "valid"


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("compression", ["none", "zstd"])
def test_database_single_item_reads_are_view_compatible(
    tmp_path: Path,
    mmap: bool,
    compression: str,
) -> None:
    path = tmp_path / f"single_item_view_{compression}_{mmap}.atp"
    mol = _make_molecule(-4.0)

    db = atompack.Database(str(path), compression=compression)
    db.add_molecule(mol)
    db.flush()

    reopened = atompack.Database.open(str(path), mmap=mmap)
    fetched = reopened[0]

    assert fetched.energy == pytest.approx(-4.0)
    np.testing.assert_allclose(fetched.positions, mol.positions)
    np.testing.assert_allclose(fetched.forces, mol.forces)
    np.testing.assert_array_equal(fetched.atomic_numbers, mol.atomic_numbers)
    assert fetched[0].atomic_number == 6
    assert fetched.get_property("tag") == "train"
    np.testing.assert_array_equal(
        fetched.get_property("ids"),
        np.array([1, 2], dtype=np.int64),
    )

    batch = reopened.get_molecules([0])[0]
    assert batch.energy == pytest.approx(fetched.energy)
    np.testing.assert_allclose(batch.positions, fetched.positions)
    np.testing.assert_allclose(batch.forces, fetched.forces)


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("compression", ["none", "zstd"])
def test_database_custom_array_properties_roundtrip_all_numeric_tags(
    tmp_path: Path,
    mmap: bool,
    compression: str,
) -> None:
    path = tmp_path / f"custom_array_tags_{compression}_{mmap}.atp"
    mol = _make_molecule(-6.0)
    mol.set_property("float_vec64", np.array([1.0, 2.0, 3.0], dtype=np.float64))
    mol.set_property("float_vec32", np.array([1.5, 2.5, 3.5], dtype=np.float32))
    mol.set_property("int_vec64", np.array([1, 2, 3], dtype=np.int64))
    mol.set_property("int_vec32", np.array([4, 5, 6], dtype=np.int32))
    mol.set_property(
        "vec3_f32",
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
    )
    mol.set_property(
        "vec3_f64",
        np.array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]], dtype=np.float64),
    )

    db = atompack.Database(str(path), compression=compression)
    db.add_molecule(mol)
    db.flush()

    reopened = atompack.Database.open(str(path), mmap=mmap)
    fetched = reopened[0]

    float_vec64 = fetched.get_property("float_vec64")
    assert float_vec64.dtype == np.float64
    assert float_vec64.shape == (3,)
    np.testing.assert_allclose(float_vec64, np.array([1.0, 2.0, 3.0], dtype=np.float64))

    float_vec32 = fetched.get_property("float_vec32")
    assert float_vec32.dtype == np.float32
    assert float_vec32.shape == (3,)
    np.testing.assert_allclose(float_vec32, np.array([1.5, 2.5, 3.5], dtype=np.float32))

    int_vec64 = fetched.get_property("int_vec64")
    assert int_vec64.dtype == np.int64
    assert int_vec64.shape == (3,)
    np.testing.assert_array_equal(int_vec64, np.array([1, 2, 3], dtype=np.int64))

    int_vec32 = fetched.get_property("int_vec32")
    assert int_vec32.dtype == np.int32
    assert int_vec32.shape == (3,)
    np.testing.assert_array_equal(int_vec32, np.array([4, 5, 6], dtype=np.int32))

    vec3_f32 = fetched.get_property("vec3_f32")
    assert vec3_f32.dtype == np.float32
    assert vec3_f32.shape == (2, 3)
    np.testing.assert_allclose(
        vec3_f32,
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
    )

    vec3_f64 = fetched.get_property("vec3_f64")
    assert vec3_f64.dtype == np.float64
    assert vec3_f64.shape == (2, 3)
    np.testing.assert_allclose(
        vec3_f64,
        np.array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]], dtype=np.float64),
    )


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("compression", ["none", "zstd"])
def test_database_single_item_mutation_is_copy_on_write(
    tmp_path: Path,
    mmap: bool,
    compression: str,
) -> None:
    path = tmp_path / f"single_item_cow_{compression}_{mmap}.atp"
    original = _make_molecule(-5.0)

    db = atompack.Database(str(path), compression=compression)
    db.add_molecule(original)
    db.flush()

    reopened = atompack.Database.open(str(path), mmap=mmap)
    fetched = reopened[0]
    fetched.energy = 123.0
    fetched.forces = np.array(
        [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]],
        dtype=np.float32,
    )
    fetched.set_property("tag", "mutated")
    fetched.set_property("new_ids", np.array([7, 8], dtype=np.int64))

    assert fetched.energy == pytest.approx(123.0)
    np.testing.assert_allclose(
        fetched.forces,
        np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]], dtype=np.float32),
    )
    assert fetched.get_property("tag") == "mutated"
    np.testing.assert_array_equal(
        fetched.get_property("new_ids"),
        np.array([7, 8], dtype=np.int64),
    )

    fresh = reopened[0]
    assert fresh.energy == pytest.approx(-5.0)
    np.testing.assert_allclose(fresh.forces, original.forces)
    assert fresh.get_property("tag") == "train"
    with pytest.raises(ValueError, match="Property 'new_ids' not found"):
        fresh.get_property("new_ids")


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("compression", ["none", "zstd"])
def test_database_single_item_pickle_materializes_owned_roundtrip(
    tmp_path: Path,
    mmap: bool,
    compression: str,
) -> None:
    path = tmp_path / f"single_item_pickle_{compression}_{mmap}.atp"
    original = _make_molecule(-6.0)

    db = atompack.Database(str(path), compression=compression)
    db.add_molecule(original)
    db.flush()

    reopened = atompack.Database.open(str(path), mmap=mmap)
    fetched = reopened[0]
    restored = pickle.loads(pickle.dumps(fetched))

    assert restored.energy == pytest.approx(-6.0)
    np.testing.assert_allclose(restored.positions, original.positions)
    np.testing.assert_allclose(restored.forces, original.forces)
    np.testing.assert_array_equal(restored.atomic_numbers, original.atomic_numbers)
    assert restored.get_property("tag") == "train"
    np.testing.assert_array_equal(
        restored.get_property("ids"),
        np.array([1, 2], dtype=np.int64),
    )

    restored.energy = 321.0
    restored.set_property("tag", "pickled")
    assert restored.energy == pytest.approx(321.0)
    assert restored.get_property("tag") == "pickled"

    fresh = reopened[0]
    assert fresh.energy == pytest.approx(-6.0)
    assert fresh.get_property("tag") == "train"


def test_database_init_refuses_existing_path_by_default(tmp_path: Path) -> None:
    path = tmp_path / "existing.atp"
    db = atompack.Database(str(path))
    db.add_molecule(_make_molecule(-3.0))
    db.flush()

    with pytest.raises(FileExistsError, match=r"Database\.open\(path, mmap=False\)"):
        atompack.Database(str(path))

    reopened = atompack.Database.open(str(path))
    assert len(reopened) == 1
    assert reopened[0].energy == pytest.approx(-3.0)


def test_database_init_overwrite_true_recreates_existing_path(tmp_path: Path) -> None:
    path = tmp_path / "overwrite.atp"
    db = atompack.Database(str(path))
    db.add_molecule(_make_molecule(-1.0))
    db.add_molecule(_make_molecule(-2.0))
    db.flush()

    db_overwrite = atompack.Database(str(path), overwrite=True)
    db_overwrite.flush()

    reopened = atompack.Database.open(str(path))
    assert len(reopened) == 0


def test_database_out_of_bounds_raises_index_error(tmp_path: Path) -> None:
    path = tmp_path / "oob.atp"
    db = atompack.Database(str(path))
    db.add_molecule(_make_molecule(-1.0))
    db.flush()

    reopened = atompack.Database.open(str(path))
    with pytest.raises(IndexError, match=r"out of bounds"):
        _ = reopened[len(reopened)]


def test_database_open_defaults_to_read_only_mmap(tmp_path: Path) -> None:
    path = tmp_path / "readonly_default.atp"
    db = atompack.Database(str(path))
    db.add_molecule(_make_molecule(-1.0))
    db.flush()

    reopened = atompack.Database.open(str(path))
    with pytest.raises(ValueError, match=r"reopen without mmap to write"):
        reopened.add_molecule(_make_molecule(-2.0))

    writable = atompack.Database.open(str(path), mmap=False)
    writable.add_molecule(_make_molecule(-2.0))
    writable.flush()

    check = atompack.Database.open(str(path))
    assert len(check) == 2



def test_database_sequence_iteration_stops_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "iter.atp"
    db = atompack.Database(str(path))
    energies = [-1.0, -2.0, -3.0]
    for energy in energies:
        db.add_molecule(_make_molecule(energy))
    db.flush()

    reopened = atompack.Database.open(str(path))
    iterated_energies = [mol.energy for mol in reopened]
    assert iterated_energies == pytest.approx(energies)


def test_get_molecules_flat_all_fields(tmp_path: Path) -> None:
    """get_molecules_flat should return all builtin and custom fields as batched arrays."""
    path = tmp_path / "flat.atp"

    mol1 = _make_molecule(-1.0)
    mol1.stress = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    mol1.pbc = (True, True, False)
    mol1.set_property("bandgap", 2.5)

    mol2 = _make_molecule(-2.0)
    mol2.stress = np.eye(3, dtype=np.float64)
    mol2.pbc = (False, False, True)
    mol2.set_property("bandgap", 3.0)

    db = atompack.Database(str(path))
    db.add_molecules([mol1, mol2])
    db.flush()

    db_r = atompack.Database.open(str(path), mmap=True)
    batch = db_r.get_molecules_flat([0, 1])

    # Core fields
    np.testing.assert_array_equal(batch["n_atoms"], [2, 2])
    assert batch["positions"].shape == (4, 3)
    assert batch["positions"].dtype == np.float32
    assert batch["atomic_numbers"].shape == (4,)
    np.testing.assert_array_equal(batch["atomic_numbers"], [6, 8, 6, 8])

    # Energy
    np.testing.assert_allclose(batch["energy"], [-1.0, -2.0])

    # Forces (per-atom, batched)
    assert batch["forces"].shape == (4, 3)
    np.testing.assert_allclose(batch["forces"][0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(batch["forces"][1], [0.4, 0.5, 0.6])

    # Charges (per-atom, batched)
    assert "charges" in batch
    np.testing.assert_allclose(batch["charges"], [-0.1, 0.1, -0.1, 0.1])

    # Velocities (per-atom, batched)
    assert "velocities" in batch
    assert batch["velocities"].shape == (4, 3)

    # Cell (per-molecule)
    assert batch["cell"].shape == (2, 3, 3)
    np.testing.assert_allclose(batch["cell"][0], np.eye(3) * 2.0)

    # Stress (per-molecule)
    assert batch["stress"].shape == (2, 3, 3)
    np.testing.assert_allclose(batch["stress"][0], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_allclose(batch["stress"][1], np.eye(3))

    # PBC (per-molecule)
    assert batch["pbc"].shape == (2, 3)
    np.testing.assert_array_equal(batch["pbc"][0], [True, True, False])
    np.testing.assert_array_equal(batch["pbc"][1], [False, False, True])

    # Custom properties in nested dict
    assert "properties" in batch
    np.testing.assert_allclose(batch["properties"]["bandgap"], [2.5, 3.0])
    assert "tag" in batch["properties"]  # string property
    np.testing.assert_array_equal(batch["properties"]["ids"], [1, 2, 1, 2])


def test_get_molecules_flat_empty(tmp_path: Path) -> None:
    """get_molecules_flat with empty indices should return empty arrays."""
    path = tmp_path / "empty_flat.atp"
    db = atompack.Database(str(path))
    db.add_molecule(_make_molecule(-1.0))
    db.flush()

    db_r = atompack.Database.open(str(path), mmap=True)
    batch = db_r.get_molecules_flat([])
    assert batch["n_atoms"].shape == (0,)
    assert batch["positions"].shape == (0, 3)
    assert batch["atomic_numbers"].shape == (0,)


def test_database_open_mmap_populate(tmp_path: Path) -> None:
    # populate=True prefaults mapped pages on Linux. On other platforms it's a
    # no-op. Either way the open path must succeed and return readable data.
    path = tmp_path / "populate.atp"
    db = atompack.Database(str(path))
    db.add_molecule(_make_molecule(-3.0))
    db.flush()

    db_r = atompack.Database.open(str(path), mmap=True, populate=True)
    assert len(db_r) == 1
    assert db_r[0].energy == pytest.approx(-3.0)


def test_database_negative_indexing_raises_index_error(tmp_path: Path) -> None:
    # Database does not support negative indexing today; a clear IndexError
    # is preferable to silent wraparound.
    path = tmp_path / "negidx.atp"
    db = atompack.Database(str(path))
    db.add_molecule(_make_molecule(-1.0))
    db.flush()

    db_r = atompack.Database.open(str(path))
    with pytest.raises((IndexError, OverflowError, ValueError)):
        _ = db_r[-1]


def test_database_empty_molecule_roundtrip(tmp_path: Path) -> None:
    # n_atoms == 0 is the SOA parser edge case; positions/atomic_numbers slices
    # become zero-length and most code paths must still work.
    path = tmp_path / "empty_mol.atp"
    mol = atompack.Molecule.from_arrays(
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0,), dtype=np.uint8),
        energy=0.0,
    )

    db = atompack.Database(str(path))
    db.add_molecule(mol)
    db.flush()

    db_r = atompack.Database.open(str(path))
    read = db_r[0]
    assert len(read) == 0
    assert read.positions.shape == (0, 3)
    assert read.atomic_numbers.shape == (0,)
    assert read.energy == pytest.approx(0.0)
