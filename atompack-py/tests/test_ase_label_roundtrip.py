from __future__ import annotations

import atompack
import numpy as np
import pytest

ase = pytest.importorskip("ase")
SinglePointCalculator = pytest.importorskip("ase.calculators.singlepoint").SinglePointCalculator


def _molecule():
    return atompack.Molecule.from_arrays(
        np.zeros((2, 3), dtype=np.float32),
        np.array([1, 8], dtype=np.uint8),
        energy=-3.0,
        forces=np.ones((2, 3), dtype=np.float32),
        charges=np.array([-0.2, 0.2], dtype=np.float64),
        stress=np.eye(3, dtype=np.float64),
    )


@pytest.mark.parametrize("copy_custom", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_no_calculator_labels_survive_roundtrip(tmp_path, copy_custom, batch):
    molecule = _molecule()
    atoms = molecule.to_ase(attach_calc=False)
    if copy_custom:
        atoms.info["source"] = "test"
    if batch:
        path = tmp_path / "roundtrip.atp"
        writer = atompack.Database(str(path))
        atompack.add_ase_batch(
            writer, [atoms, atoms], copy_info=copy_custom, copy_arrays=copy_custom, batch_size=1
        )
        writer.flush()
        reader = atompack.Database.open(str(path))
        restored = reader.get_molecules([0, 1])
    else:
        restored = [atompack.from_ase(atoms, copy_info=copy_custom, copy_arrays=copy_custom)]
    for result in restored:
        assert result.energy == -3.0
        np.testing.assert_allclose(result.forces, molecule.forces)
        np.testing.assert_allclose(result.charges, molecule.charges)
        np.testing.assert_allclose(result.stress, molecule.stress)
        for key in ("energy", "forces", "charges", "stress"):
            assert not result.has_property(key)


def test_builtin_label_precedence():
    atoms = _molecule().to_ase(attach_calc=False)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-4.0,
        forces=np.full((2, 3), 4.0),
        charges=np.array([-0.4, 0.4]),
        stress=np.eye(3) * 4,
    )
    from_calculator = atompack.from_ase(atoms)
    assert from_calculator.energy == -4.0
    np.testing.assert_allclose(from_calculator.forces, 4.0)
    np.testing.assert_allclose(from_calculator.charges, [-0.4, 0.4])
    np.testing.assert_allclose(from_calculator.stress, np.eye(3) * 4)
    explicit = atompack.from_ase(
        atoms,
        energy=-5.0,
        forces=np.full((2, 3), 5.0),
        charges=np.array([-0.5, 0.5]),
        stress=np.eye(3) * 5,
        info={"stress": np.eye(3) * 6},
    )
    assert explicit.energy == -5.0
    np.testing.assert_allclose(explicit.forces, 5.0)
    np.testing.assert_allclose(explicit.charges, [-0.5, 0.5])
    np.testing.assert_allclose(explicit.stress, np.eye(3) * 5)
    info_override = atompack.from_ase(atoms, info={"stress": np.eye(3) * 6})
    np.testing.assert_allclose(info_override.stress, np.eye(3) * 6)


def test_unused_malformed_fallbacks_do_not_override_getters_or_arguments():
    atoms = _molecule().to_ase(attach_calc=False)
    atoms.info["energy"] = "bad"
    atoms.info["stress"] = "bad"
    atoms.arrays["forces"] = np.full((2, 3), "bad")
    atoms.arrays["charges"] = np.full(2, "bad")
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-4.0,
        forces=np.full((2, 3), 4.0),
        charges=np.array([-0.4, 0.4]),
        stress=np.eye(3) * 4,
    )
    restored = atompack.from_ase(atoms)
    assert restored.energy == -4.0
    np.testing.assert_allclose(restored.forces, 4)
    np.testing.assert_allclose(restored.charges, [-0.4, 0.4])
    np.testing.assert_allclose(restored.stress, np.eye(3) * 4)
    atoms.calc = None
    overridden = atompack.from_ase(
        atoms,
        energy=-5.0,
        forces=np.full((2, 3), 5.0),
        charges=np.array([-0.5, 0.5]),
        stress=np.eye(3) * 5,
    )
    assert overridden.energy == -5.0
    np.testing.assert_allclose(overridden.stress, np.eye(3) * 5)


@pytest.mark.parametrize(
    "key,value",
    [
        ("energy", [1.0, 2.0]),
        ("energy", "bad"),
        ("forces", np.ones((2, 2))),
        ("forces", np.full((2, 3), "bad")),
        ("charges", np.ones(3)),
        ("stress", np.ones(4)),
    ],
)
def test_invalid_builtin_fallback_is_rejected(key, value):
    atoms = ase.Atoms("HO", positions=np.zeros((2, 3)))
    if key in {"energy", "stress"}:
        atoms.info[key] = value
    else:
        atoms.arrays[key] = value
    with pytest.raises((ValueError, TypeError), match=key):
        atompack.from_ase(atoms)


def test_missing_builtin_labels_stay_missing():
    molecule = atompack.from_ase(ase.Atoms("H", positions=[[0, 0, 0]]))
    assert molecule.energy is None
    assert molecule.forces is None
    assert molecule.charges is None
    assert molecule.stress is None
