# Copyright 2026 Entalpic
from __future__ import annotations

import atompack
import numpy as np
import pytest


def _make_molecule() -> atompack.Molecule:
    return atompack.Molecule(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([6, 8], dtype=np.uint8),
    )


def test_atom_basics() -> None:
    a = atompack.Atom(0.0, 0.0, 0.0, 6)
    b = atompack.Atom(3.0, 4.0, 0.0, 8)

    assert a.position() == (0.0, 0.0, 0.0)
    assert a.atomic_number == 6
    assert a.distance_to(b) == pytest.approx(5.0)
    assert "Atom(" in repr(a)


def test_molecule_basic_views() -> None:
    mol = _make_molecule()

    assert len(mol) == 2

    positions = mol.positions
    assert positions.shape == (2, 3)
    assert positions.dtype == np.float32
    np.testing.assert_allclose(positions, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32))

    atomic_numbers = mol.atomic_numbers
    assert atomic_numbers.shape == (2,)
    assert atomic_numbers.dtype == np.uint8
    np.testing.assert_array_equal(atomic_numbers, np.array([6, 8], np.uint8))


def test_molecule_from_arrays() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    atomic_numbers = np.array([6, 8], dtype=np.uint8)

    mol = atompack.Molecule.from_arrays(positions, atomic_numbers)

    assert len(mol) == 2
    np.testing.assert_allclose(mol.positions, positions)
    np.testing.assert_array_equal(mol.atomic_numbers, atomic_numbers)


def test_molecule_from_arrays_with_builtins() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    atomic_numbers = np.array([6, 8], dtype=np.uint8)
    forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    charges = np.array([-0.1, 0.1], dtype=np.float64)
    velocities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    cell = np.eye(3, dtype=np.float64) * 2.0
    stress = np.eye(3, dtype=np.float32) * 4.0

    mol = atompack.Molecule.from_arrays(
        positions,
        atomic_numbers,
        energy=-12.5,
        forces=forces,
        charges=charges,
        velocities=velocities,
        cell=cell,
        stress=stress,
        pbc=(True, False, True),
        name="water",
    )

    np.testing.assert_allclose(mol.positions, positions)
    np.testing.assert_array_equal(mol.atomic_numbers, atomic_numbers)
    assert mol.energy == pytest.approx(-12.5)
    np.testing.assert_allclose(mol.forces, forces)
    np.testing.assert_allclose(mol.charges, charges)
    np.testing.assert_allclose(mol.velocities, velocities)
    np.testing.assert_allclose(mol.cell, cell)
    np.testing.assert_allclose(mol.stress, stress.astype(np.float64))
    assert mol.pbc == (True, False, True)


def test_molecule_from_arrays_validation() -> None:
    positions = np.zeros((2, 2), dtype=np.float32)
    atomic_numbers = np.array([6, 8], dtype=np.uint8)

    with pytest.raises(ValueError, match=r"positions must have shape"):
        atompack.Molecule.from_arrays(positions, atomic_numbers)

    positions = np.zeros((2, 3), dtype=np.float32)
    atomic_numbers = np.array([6], dtype=np.uint8)
    with pytest.raises(ValueError, match=r"Atomic numbers length"):
        atompack.Molecule.from_arrays(positions, atomic_numbers)


def test_molecule_ml_properties_roundtrip() -> None:
    mol = _make_molecule()

    mol.energy = -123.456
    assert mol.energy == pytest.approx(-123.456)

    mol.energy = None
    assert mol.energy is None

    forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    mol.forces = forces
    np.testing.assert_allclose(mol.forces, forces)

    charges = np.array([-0.1, 0.1], dtype=np.float64)
    mol.charges = charges
    np.testing.assert_allclose(mol.charges, charges)

    velocities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mol.velocities = velocities
    np.testing.assert_allclose(mol.velocities, velocities)

    cell = np.eye(3, dtype=np.float64) * 2.0
    mol.cell = cell
    np.testing.assert_allclose(mol.cell, cell)

    stress = np.eye(3, dtype=np.float64) * 4.0
    mol.stress = stress
    np.testing.assert_allclose(mol.stress, stress)

    stress32 = np.eye(3, dtype=np.float32) * 5.0
    mol.stress = stress32
    assert mol.stress.dtype == np.float64
    np.testing.assert_allclose(mol.stress, stress32.astype(np.float64))


def test_molecule_ml_property_validation() -> None:
    mol = _make_molecule()

    with pytest.raises(ValueError, match=r"Forces must have shape"):
        mol.forces = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"Forces length"):
        mol.forces = np.zeros((1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match=r"Charges length"):
        mol.charges = np.zeros((3,), dtype=np.float64)

    with pytest.raises(ValueError, match=r"Velocities must have shape"):
        mol.velocities = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"Velocities length"):
        mol.velocities = np.zeros((1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match=r"Cell must have shape"):
        mol.cell = np.zeros((3, 2), dtype=np.float64)

    with pytest.raises(ValueError, match=r"Stress must have shape"):
        mol.stress = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match=r"'stress' is a reserved field"):
        mol.set_property("stress", np.zeros((3, 3), dtype=np.int64))


def test_molecule_custom_properties() -> None:
    mol = _make_molecule()

    mol.set_property("temperature", 300.0)
    mol.set_property("steps", 42)
    mol.set_property("method", "DFT")
    mol.set_property("float_vec", np.array([1.0, 2.0, 3.0], dtype=np.float64))
    mol.set_property("float_vec32", np.array([1.5, 2.5], dtype=np.float32))
    mol.set_property("int_vec", np.array([1, 2], dtype=np.int64))
    mol.set_property("int_vec32", np.array([3, 4], dtype=np.int32))
    mol.set_property("vec3", np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    mol.set_property("vec3_f64", np.array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]], dtype=np.float64))
    mol.stress = np.eye(3, dtype=np.float64) * 3.0

    assert mol.get_property("temperature") == pytest.approx(300.0)
    steps = mol.get_property("steps")
    assert isinstance(steps, int)
    assert steps == 42
    assert mol.get_property("method") == "DFT"

    float_vec = mol.get_property("float_vec")
    assert isinstance(float_vec, np.ndarray)
    assert float_vec.dtype == np.float64
    np.testing.assert_allclose(float_vec, np.array([1.0, 2.0, 3.0], dtype=np.float64))

    float_vec32 = mol.get_property("float_vec32")
    assert isinstance(float_vec32, np.ndarray)
    assert float_vec32.dtype == np.float32
    np.testing.assert_allclose(float_vec32, np.array([1.5, 2.5], dtype=np.float32))

    int_vec = mol.get_property("int_vec")
    assert isinstance(int_vec, np.ndarray)
    assert int_vec.dtype == np.int64
    np.testing.assert_array_equal(int_vec, np.array([1, 2], dtype=np.int64))

    int_vec32 = mol.get_property("int_vec32")
    assert isinstance(int_vec32, np.ndarray)
    assert int_vec32.dtype == np.int32
    np.testing.assert_array_equal(int_vec32, np.array([3, 4], dtype=np.int32))

    vec3 = mol.get_property("vec3")
    assert isinstance(vec3, np.ndarray)
    assert vec3.dtype == np.float32
    assert vec3.shape == (2, 3)

    vec3_f64 = mol.get_property("vec3_f64")
    assert isinstance(vec3_f64, np.ndarray)
    assert vec3_f64.dtype == np.float64
    np.testing.assert_allclose(
        vec3_f64, np.array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]], dtype=np.float64)
    )

    np.testing.assert_allclose(mol.stress, np.eye(3, dtype=np.float64) * 3.0)

    assert mol.has_property("method") is True
    assert mol.has_property("stress") is False
    assert set(mol.property_keys()) >= {
        "temperature",
        "steps",
        "method",
        "float_vec",
        "float_vec32",
        "int_vec",
        "int_vec32",
        "vec3",
        "vec3_f64",
    }

    with pytest.raises(KeyError, match=r"not found"):
        mol.get_property("stress")

    with pytest.raises(KeyError, match=r"not found"):
        mol.get_property("does_not_exist")


def test_molecule_getitem_supports_int_and_str() -> None:
    mol = _make_molecule()
    mol.set_property("method", "DFT")
    mol.set_property("vec3", np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))

    atom0 = mol[0]
    assert isinstance(atom0, atompack.Atom)
    assert atom0.position() == (0.0, 0.0, 0.0)
    assert atom0.atomic_number == 6

    atom_last = mol[-1]
    assert isinstance(atom_last, atompack.Atom)
    assert atom_last.atomic_number == 8

    assert mol["method"] == "DFT"
    vec3 = mol["vec3"]
    assert isinstance(vec3, np.ndarray)
    np.testing.assert_allclose(vec3, np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))


def test_molecule_getitem_validation() -> None:
    mol = _make_molecule()
    mol.set_property("temperature", 300.0)

    with pytest.raises(IndexError, match=r"out of bounds"):
        _ = mol[2]
    with pytest.raises(KeyError, match=r"not found"):
        _ = mol["does_not_exist"]
    with pytest.raises(TypeError, match=r"integers or strings"):
        _ = mol[1.5]
