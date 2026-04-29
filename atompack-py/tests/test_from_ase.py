# Copyright 2026 Entalpic
from __future__ import annotations

from dataclasses import dataclass

import atompack
import numpy as np
import pytest

ase = pytest.importorskip("ase")


@dataclass
class FakeASEAtoms:
    positions: np.ndarray
    atomic_numbers: np.ndarray
    energy: float | None = None
    forces: np.ndarray | None = None
    charges: np.ndarray | None = None
    velocities: np.ndarray | None = None
    cell: np.ndarray | None = None
    pbc: np.ndarray | None = None
    info: dict | None = None
    arrays: dict | None = None
    calc: object | None = None

    def get_positions(self) -> np.ndarray:  # ASE API
        return self.positions

    def get_atomic_numbers(self) -> np.ndarray:  # ASE API
        return self.atomic_numbers

    def get_potential_energy(self) -> float:  # ASE API
        if self.energy is None:
            raise RuntimeError("no energy")
        return self.energy

    def get_forces(self) -> np.ndarray:  # ASE API
        if self.forces is None:
            raise RuntimeError("no forces")
        return self.forces

    def get_charges(self) -> np.ndarray:  # Optional ASE API
        if self.charges is None:
            raise RuntimeError("no charges")
        return self.charges

    def get_velocities(self) -> np.ndarray | None:  # ASE API
        return self.velocities

    def get_cell(self) -> np.ndarray:  # ASE API
        if self.cell is None:
            raise RuntimeError("no cell")
        return self.cell

    def get_stress(self, voigt=False):  # ASE API
        if self.info is None or "ase_stress" not in self.info:
            raise RuntimeError("no stress")
        stress = self.info["ase_stress"]
        if voigt:
            return stress
        return stress


@dataclass
class FakeCalc:
    results: dict


def test_from_ase_extracts_core_fields() -> None:
    atoms = FakeASEAtoms(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        atomic_numbers=np.array([6, 8], dtype=np.int64),
        energy=-7.5,
        forces=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
        charges=np.array([-0.1, 0.1], dtype=np.float64),
        velocities=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        cell=np.eye(3, dtype=np.float64) * 2.0,
        pbc=np.array([True, False, False]),
        info={
            "temperature": 300.0,
            "steps": 42,
            "method": "DFT",
            "float_vec": np.array([1.0, 2.0], dtype=np.float64),
            "float_vec32": np.array([3.0, 4.0], dtype=np.float32),
            "int_vec": np.array([1, 2], dtype=np.int64),
            "int_vec32": np.array([3, 4], dtype=np.int32),
            "vec3": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
            "vec3_f64": np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float64),
            "stress": np.array(
                [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3]], dtype=np.float64
            ),
            "unsupported": {"nested": True},
        },
    )

    mol = atompack.from_ase(atoms)

    assert mol.energy == pytest.approx(-7.5)
    np.testing.assert_allclose(mol.forces, np.array(atoms.forces, dtype=np.float32))
    np.testing.assert_allclose(mol.charges, np.array(atoms.charges, dtype=np.float64))
    np.testing.assert_allclose(mol.velocities, np.array(atoms.velocities, dtype=np.float32))
    np.testing.assert_allclose(mol.cell, np.array(atoms.cell, dtype=np.float64))

    np.testing.assert_allclose(mol.positions, np.array(atoms.positions, dtype=np.float32))
    np.testing.assert_array_equal(
        mol.atomic_numbers, np.array(atoms.atomic_numbers, dtype=np.uint8)
    )

    assert mol.get_property("temperature") == pytest.approx(300.0)
    steps = mol.get_property("steps")
    assert isinstance(steps, int)
    assert steps == 42
    assert mol.get_property("method") == "DFT"
    np.testing.assert_allclose(
        mol.get_property("float_vec"), np.array([1.0, 2.0], dtype=np.float64)
    )
    np.testing.assert_allclose(
        mol.get_property("float_vec32"), np.array([3.0, 4.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(mol.get_property("int_vec"), np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(mol.get_property("int_vec32"), np.array([3, 4], dtype=np.int32))
    np.testing.assert_allclose(
        mol.get_property("vec3"), np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        mol.get_property("vec3_f64"), np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float64)
    )
    np.testing.assert_allclose(
        mol.stress,
        np.array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3]], dtype=np.float64),
    )
    with pytest.raises(ValueError, match=r"not found"):
        mol.get_property("stress")

    assert mol.has_property("unsupported") is False


def test_from_ase_info_override_and_copy_toggle() -> None:
    atoms = FakeASEAtoms(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        atomic_numbers=np.array([1], dtype=np.int64),
        pbc=np.array([False, False, False]),
        info={"temperature": 300.0},
    )

    mol_no_copy = atompack.from_ase(atoms, copy_info=False)
    assert mol_no_copy.has_property("temperature") is False

    mol_override = atompack.from_ase(atoms, info={"temperature": 500.0})
    assert mol_override.get_property("temperature") == pytest.approx(500.0)


def test_from_ase_extracts_arrays_and_calc_results() -> None:
    atoms = FakeASEAtoms(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        atomic_numbers=np.array([6, 8], dtype=np.int64),
        pbc=np.array([True, True, False]),
        arrays={
            "positions": np.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]], dtype=np.float64),
            "numbers": np.array([1, 1], dtype=np.int64),
            "tags": np.array([1, 2], dtype=np.int32),
            "masses": np.array([12.0, 16.0], dtype=np.float64),
            "momenta": np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float64),
        },
        calc=FakeCalc(
            {
                "free_energy": -5.5,
                "magmoms": np.array([0.3, 0.4], dtype=np.float64),
                "forces": np.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]], dtype=np.float64),
            }
        ),
        info={"ase_stress": np.eye(3, dtype=np.float64) * 3.0},
    )

    mol = atompack.from_ase(atoms)

    np.testing.assert_array_equal(mol.get_property("tags"), np.array([1, 2], dtype=np.int32))
    np.testing.assert_allclose(
        mol.get_property("masses"), np.array([12.0, 16.0], dtype=np.float64)
    )
    np.testing.assert_allclose(
        mol.get_property("momenta"),
        np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float64),
    )
    assert mol.get_property("free_energy") == pytest.approx(-5.5)
    np.testing.assert_allclose(
        mol.get_property("magmoms"), np.array([0.3, 0.4], dtype=np.float64)
    )
    np.testing.assert_allclose(mol.stress, np.eye(3, dtype=np.float64) * 3.0)
    with pytest.raises(ValueError, match=r"not found"):
        mol.get_property("forces")


def test_add_ase_batch_roundtrip(tmp_path) -> None:
    path = tmp_path / "ase_batch.atp"
    atoms_list = [
        FakeASEAtoms(
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
            atomic_numbers=np.array([6, 8], dtype=np.int64),
            energy=-1.5,
            forces=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
            cell=np.eye(3, dtype=np.float64) * 2.0,
            pbc=np.array([True, False, True]),
        ),
        FakeASEAtoms(
            positions=np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64),
            atomic_numbers=np.array([1, 8], dtype=np.int64),
            energy=-2.5,
            forces=np.array([[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]], dtype=np.float64),
            cell=np.eye(3, dtype=np.float64) * 3.0,
            pbc=np.array([False, True, False]),
            arrays={"tags": np.array([7, 8], dtype=np.int32)},
            info={"temperature": 300.0},
        ),
    ]

    db = atompack.Database(str(path))
    atompack.add_ase_batch(db, atoms_list, batch_size=1)
    db.flush()

    reopened = atompack.Database.open(str(path))
    first = reopened[0]
    second = reopened[1]

    assert first.energy == pytest.approx(-1.5)
    np.testing.assert_allclose(first.forces, np.array(atoms_list[0].forces, dtype=np.float32))
    np.testing.assert_allclose(first.cell, np.array(atoms_list[0].cell, dtype=np.float64))
    assert first.pbc == (True, False, True)

    assert second.energy == pytest.approx(-2.5)
    np.testing.assert_allclose(second.forces, np.array(atoms_list[1].forces, dtype=np.float32))
    np.testing.assert_allclose(second.cell, np.array(atoms_list[1].cell, dtype=np.float64))
    assert second.pbc == (False, True, False)
    np.testing.assert_array_equal(second.get_property("tags"), np.array([7, 8], dtype=np.int32))
    assert second.get_property("temperature") == pytest.approx(300.0)


def test_to_ase_owned_maps_builtins_and_properties() -> None:
    mol = atompack.Molecule.from_arrays(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32),
        np.array([6, 8], dtype=np.uint8),
        energy=-4.5,
        forces=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        charges=np.array([-0.2, 0.2], dtype=np.float64),
        velocities=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        cell=np.eye(3, dtype=np.float64) * 3.0,
        stress=np.eye(3, dtype=np.float64) * 2.0,
        pbc=(True, False, True),
    )
    mol.set_property("temperature", 300.0)
    mol.set_property("tags", np.array([3, 4], dtype=np.int32))

    atoms = mol.to_ase()

    np.testing.assert_allclose(atoms.get_positions(), mol.positions)
    np.testing.assert_array_equal(atoms.get_atomic_numbers(), mol.atomic_numbers)
    np.testing.assert_allclose(np.asarray(atoms.get_cell()), mol.cell)
    np.testing.assert_array_equal(atoms.pbc, np.array([True, False, True], dtype=bool))
    np.testing.assert_allclose(atoms.get_velocities(), mol.velocities)
    assert atoms.calc.results["energy"] == pytest.approx(-4.5)
    np.testing.assert_allclose(atoms.calc.results["forces"], mol.forces)
    np.testing.assert_allclose(atoms.calc.results["charges"], mol.charges)
    np.testing.assert_allclose(atoms.calc.results["stress"], mol.stress)
    assert atoms.info["temperature"] == pytest.approx(300.0)
    np.testing.assert_array_equal(atoms.arrays["tags"], np.array([3, 4], dtype=np.int32))


def test_to_ase_calc_modes() -> None:
    mol = atompack.Molecule.from_arrays(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32),
        np.array([6, 8], dtype=np.uint8),
        energy=-4.5,
        forces=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        cell=np.eye(3, dtype=np.float64) * 3.0,
        pbc=(True, False, True),
    )

    no_calc = mol.to_ase(calc_mode="none")
    assert no_calc.calc is None

    nocopy = mol.to_ase(calc_mode="nocopy")
    assert nocopy.calc is not None
    assert nocopy.get_potential_energy() == pytest.approx(-4.5)
    np.testing.assert_allclose(nocopy.get_forces(), mol.forces)

    none_mode = mol.to_ase(calc_mode="none")
    assert none_mode.calc is None
    assert none_mode.info["energy"] == pytest.approx(-4.5)
    np.testing.assert_allclose(none_mode.arrays["forces"], mol.forces)


def test_to_ase_view_backed_molecule(tmp_path) -> None:
    path = tmp_path / "to_ase_view.atp"
    mol = atompack.Molecule.from_arrays(
        np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32),
        np.array([1, 8], dtype=np.uint8),
        energy=-1.25,
        forces=np.array([[0.7, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32),
        cell=np.eye(3, dtype=np.float64) * 4.0,
        stress=np.eye(3, dtype=np.float64) * 1.5,
        pbc=(False, True, False),
    )

    db = atompack.Database(str(path), compression="none")
    db.add_molecule(mol)
    db.flush()

    reopened = atompack.Database.open(str(path))
    view_mol = reopened[0]
    atoms = view_mol.to_ase()

    np.testing.assert_allclose(atoms.get_positions(), view_mol.positions)
    np.testing.assert_array_equal(atoms.get_atomic_numbers(), view_mol.atomic_numbers)
    np.testing.assert_allclose(np.asarray(atoms.get_cell()), view_mol.cell)
    np.testing.assert_array_equal(atoms.pbc, np.array([False, True, False], dtype=bool))
    assert atoms.calc.results["energy"] == pytest.approx(-1.25)
    np.testing.assert_allclose(atoms.calc.results["forces"], view_mol.forces)
    np.testing.assert_allclose(atoms.calc.results["stress"], view_mol.stress)


@pytest.mark.parametrize("mmap", [True, False])
def test_database_to_ase_batch_matches_per_molecule(tmp_path, mmap: bool) -> None:
    path = tmp_path / "to_ase_batch.atp"
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]],
            [[0.2, 0.1, 0.0], [1.2, 0.3, 0.4]],
        ],
        dtype=np.float32,
    )
    atomic_numbers = np.array([[6, 8], [1, 8]], dtype=np.uint8)
    energy = np.array([-4.5, -2.25], dtype=np.float64)
    forces = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ],
        dtype=np.float32,
    )
    velocities = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.5, 0.0], [0.5, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    cell = np.stack([np.eye(3), np.eye(3) * 2.0]).astype(np.float64)
    stress = np.stack([np.eye(3) * 1.5, np.eye(3) * 2.5]).astype(np.float64)
    charges = np.array([[-0.2, 0.2], [0.1, -0.1]], dtype=np.float64)
    pbc = np.array([[True, False, True], [False, True, False]], dtype=bool)

    db = atompack.Database(str(path), compression="zstd")
    db.add_arrays_batch(
        positions,
        atomic_numbers,
        energy=energy,
        forces=forces,
        charges=charges,
        velocities=velocities,
        cell=cell,
        stress=stress,
        pbc=pbc,
    )
    db.flush()

    reopened = atompack.Database.open(str(path), mmap=mmap)
    batch_atoms = reopened.to_ase_batch([0, 1])
    ref_atoms = [reopened[0].to_ase(), reopened[1].to_ase()]

    assert len(batch_atoms) == len(ref_atoms) == 2
    for batch, ref in zip(batch_atoms, ref_atoms):
        np.testing.assert_allclose(batch.get_positions(), ref.get_positions())
        np.testing.assert_array_equal(batch.get_atomic_numbers(), ref.get_atomic_numbers())
        np.testing.assert_allclose(np.asarray(batch.get_cell()), np.asarray(ref.get_cell()))
        np.testing.assert_array_equal(batch.pbc, ref.pbc)
        np.testing.assert_allclose(batch.get_velocities(), ref.get_velocities())
        for key in ("energy", "forces", "stress", "charges"):
            expected = ref.calc.results[key]
            actual = batch.calc.results[key]
            if np.isscalar(expected):
                assert actual == pytest.approx(expected)
            else:
                np.testing.assert_allclose(actual, expected)


def test_to_ase_batch_with_molecule_list_matches_individual() -> None:
    molecules = [
        atompack.Molecule.from_arrays(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([6, 8], dtype=np.uint8),
            energy=-1.0,
            forces=np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
            cell=np.eye(3, dtype=np.float64),
            pbc=(True, True, False),
        ),
        atompack.Molecule.from_arrays(
            np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32),
            np.array([1, 8], dtype=np.uint8),
            energy=-2.0,
            forces=np.array([[0.3, 0.0, 0.0], [0.0, 0.4, 0.0]], dtype=np.float32),
            cell=np.eye(3, dtype=np.float64) * 2.0,
            pbc=(False, True, False),
        ),
    ]

    batch_atoms = atompack.to_ase_batch(molecules)
    ref_atoms = [molecule.to_ase() for molecule in molecules]

    assert len(batch_atoms) == len(ref_atoms) == 2
    for batch, ref in zip(batch_atoms, ref_atoms):
        np.testing.assert_allclose(batch.get_positions(), ref.get_positions())
        np.testing.assert_array_equal(batch.get_atomic_numbers(), ref.get_atomic_numbers())
        assert batch.calc.results["energy"] == pytest.approx(ref.calc.results["energy"])
        np.testing.assert_allclose(batch.calc.results["forces"], ref.calc.results["forces"])


def test_to_ase_batch_nocopy_calc_mode(tmp_path) -> None:
    path = tmp_path / "to_ase_batch_nocopy.atp"
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

    db = atompack.Database(str(path), compression="zstd")
    db.add_arrays_batch(
        positions,
        atomic_numbers,
        energy=energy,
        forces=forces,
        cell=cell,
    )
    db.flush()

    reopened = atompack.Database.open(str(path), mmap=True)
    atoms_batch = reopened.to_ase_batch([0, 1], calc_mode="nocopy")

    assert len(atoms_batch) == 2
    assert atoms_batch[0].get_potential_energy() == pytest.approx(-1.5)
    np.testing.assert_allclose(atoms_batch[0].get_forces(), forces[0])
    assert atoms_batch[1].get_potential_energy() == pytest.approx(-2.5)
    np.testing.assert_allclose(atoms_batch[1].get_forces(), forces[1])


def test_to_ase_batch_none_calc_mode_preserves_results(tmp_path) -> None:
    path = tmp_path / "to_ase_batch_none.atp"
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
    charges = np.array([[-0.2, 0.2], [0.1, -0.1]], dtype=np.float64)
    stress = np.stack([np.eye(3) * 1.5, np.eye(3) * 2.5]).astype(np.float64)
    cell = np.stack([np.eye(3), np.eye(3) * 2.0]).astype(np.float64)

    db = atompack.Database(str(path), compression="zstd")
    db.add_arrays_batch(
        positions,
        atomic_numbers,
        energy=energy,
        forces=forces,
        charges=charges,
        stress=stress,
        cell=cell,
    )
    db.flush()

    reopened = atompack.Database.open(str(path), mmap=True)
    atoms_batch = reopened.to_ase_batch([0, 1], calc_mode="none")

    assert len(atoms_batch) == 2
    assert atoms_batch[0].calc is None
    assert atoms_batch[0].info["energy"] == pytest.approx(-1.5)
    np.testing.assert_allclose(atoms_batch[0].info["stress"], stress[0])
    np.testing.assert_allclose(atoms_batch[0].arrays["forces"], forces[0])
    np.testing.assert_allclose(atoms_batch[0].arrays["charges"], charges[0])
    assert atoms_batch[1].info["energy"] == pytest.approx(-2.5)
    np.testing.assert_allclose(atoms_batch[1].info["stress"], stress[1])
    np.testing.assert_allclose(atoms_batch[1].arrays["forces"], forces[1])
    np.testing.assert_allclose(atoms_batch[1].arrays["charges"], charges[1])


def test_from_ase_expands_voigt6_stress_to_3x3() -> None:
    # ASE's get_stress(voigt=True) returns a (6,) Voigt-form array; the bridge
    # must expand it to a (3,3) symmetric tensor before storing.
    voigt = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)
    atoms = FakeASEAtoms(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        atomic_numbers=np.array([6, 8], dtype=np.int64),
        info={"ase_stress": voigt},
    )

    mol = atompack.from_ase(atoms)
    assert mol.stress is not None
    assert mol.stress.shape == (3, 3)
    # Voigt order is (xx, yy, zz, yz, xz, xy); expanded matrix is symmetric.
    expected = np.array(
        [
            [voigt[0], voigt[5], voigt[4]],
            [voigt[5], voigt[1], voigt[3]],
            [voigt[4], voigt[3], voigt[2]],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(mol.stress, expected)
    np.testing.assert_allclose(mol.stress, mol.stress.T)  # symmetry sanity
