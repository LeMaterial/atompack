# Copyright 2026 Entalpic
"""ASE conversion helpers for atompack."""

from __future__ import annotations

import numpy as np

from ._atompack_rs import PyAtomDatabase as Database
from ._atompack_rs import PyMolecule as Molecule

_BUILTIN_FIELDS = {
    "energy",
    "forces",
    "charges",
    "velocities",
    "cell",
    "stress",
    "pbc",
}

_ASE_RESERVED_ARRAYS = {"numbers", "positions"}
_ASE_TYPES = None
_CALC_MODES = {"singlepoint", "nocopy", "none"}


def _voigt6_to_mat3x3(stress):
    stress = np.asarray(stress, dtype=np.float64)
    if stress.shape != (6,):
        raise ValueError("Voigt stress must have shape (6,)")
    xx, yy, zz, yz, xz, xy = stress
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=np.float64)


def _get_stress(atoms):
    if not hasattr(atoms, "get_stress"):
        return None
    try:
        stress = atoms.get_stress(voigt=False)
    except TypeError:
        try:
            stress = atoms.get_stress()
        except Exception:
            return None
    except Exception:
        return None

    stress = np.asarray(stress)
    if stress.shape == (3, 3) and stress.dtype.kind == "f":
        return stress.astype(np.float64, copy=False)
    if stress.shape == (6,) and stress.dtype.kind == "f":
        return _voigt6_to_mat3x3(stress)
    return None


def _coerce_property(value, n_atoms):
    if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
        if isinstance(value, str):
            return value
        if isinstance(value, (bool, int, np.integer)):
            return int(value)
        return float(value)

    arr = np.asarray(value)
    if arr.ndim == 0 and arr.dtype.kind in {"b", "i", "u", "f"}:
        return arr.item()
    if arr.ndim == 1 and arr.shape[0] == n_atoms:
        if arr.dtype == np.float32:
            return arr.astype(np.float32, copy=False)
        if arr.dtype.kind == "f":
            return arr.astype(np.float64, copy=False)
        if arr.dtype == np.int32:
            return arr.astype(np.int32, copy=False)
        if arr.dtype.kind in {"b", "i", "u"}:
            return arr.astype(np.int64, copy=False)
    if arr.ndim == 2 and arr.shape == (n_atoms, 3) and arr.dtype.kind == "f":
        if arr.dtype == np.float32:
            return arr.astype(np.float32, copy=False)
        return arr.astype(np.float64, copy=False)
    return None


def _merge_properties(properties, builtins, values, n_atoms):
    for key, value in values.items():
        if key in _BUILTIN_FIELDS:
            # Builtin keys in atoms.info / info-override go to the builtins
            # dict (when shape/dtype matches), never into custom properties.
            # Without this guard, info["energy"] would land in both
            # builtins["energy"] (from get_potential_energy) and
            # properties["energy"], producing divergent state on round-trip.
            if key == "stress":
                arr = np.asarray(value)
                if arr.shape == (3, 3) and arr.dtype.kind == "f":
                    builtins["stress"] = arr.astype(np.float64, copy=False)
            continue
        coerced = _coerce_property(value, n_atoms)
        if coerced is not None:
            properties[key] = coerced


def _extract_ase_record(
    atoms,
    *,
    energy=None,
    forces=None,
    charges=None,
    velocities=None,
    cell=None,
    stress=None,
    copy_info=True,
    info=None,
):
    positions = np.asarray(atoms.get_positions(), dtype=np.float32)
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.uint8)
    n_atoms = len(atomic_numbers)

    builtins = {
        "energy": None,
        "forces": None,
        "charges": None,
        "velocities": None,
        "cell": None,
        "stress": None,
        "pbc": None,
    }

    if energy is not None:
        builtins["energy"] = float(energy)
    else:
        try:
            builtins["energy"] = float(atoms.get_potential_energy())
        except Exception:
            pass

    if forces is not None:
        builtins["forces"] = np.asarray(forces, dtype=np.float32)
    else:
        try:
            builtins["forces"] = np.asarray(atoms.get_forces(), dtype=np.float32)
        except Exception:
            pass

    if charges is not None:
        builtins["charges"] = np.asarray(charges, dtype=np.float64)
    elif hasattr(atoms, "get_charges"):
        try:
            value = atoms.get_charges()
            if value is not None:
                builtins["charges"] = np.asarray(value, dtype=np.float64)
        except Exception:
            pass

    if velocities is not None:
        builtins["velocities"] = np.asarray(velocities, dtype=np.float32)
    else:
        try:
            value = atoms.get_velocities()
            if value is not None:
                builtins["velocities"] = np.asarray(value, dtype=np.float32)
        except Exception:
            pass

    try:
        pbc = np.asarray(getattr(atoms, "pbc", None), dtype=bool)
        if pbc.shape == (3,):
            builtins["pbc"] = tuple(bool(x) for x in pbc)
            if cell is None and pbc.any():
                builtins["cell"] = np.asarray(atoms.get_cell(), dtype=np.float64)
    except Exception:
        pass

    if cell is not None:
        builtins["cell"] = np.asarray(cell, dtype=np.float64)
    if stress is not None:
        builtins["stress"] = np.asarray(stress, dtype=np.float64)
    else:
        builtins["stress"] = _get_stress(atoms)

    properties = {}

    arrays = getattr(atoms, "arrays", None)
    if isinstance(arrays, dict):
        for key, value in arrays.items():
            # Skip both ASE-reserved geometry keys ("positions", "numbers")
            # and atompack builtin field names. A user who stashes "forces"
            # in atoms.arrays must not have it duplicated into both
            # builtins["forces"] (from get_forces()) and properties["forces"].
            if key in _ASE_RESERVED_ARRAYS or key in _BUILTIN_FIELDS:
                continue
            coerced = _coerce_property(value, n_atoms)
            if coerced is not None:
                properties[key] = coerced

    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", None)
    if isinstance(results, dict):
        for key, value in results.items():
            if key not in _BUILTIN_FIELDS:
                coerced = _coerce_property(value, n_atoms)
                if coerced is not None:
                    properties[key] = coerced

    if copy_info and getattr(atoms, "info", None):
        _merge_properties(properties, builtins, atoms.info, n_atoms)
    if info is not None:
        _merge_properties(properties, builtins, info, n_atoms)

    return {
        "positions": positions,
        "atomic_numbers": atomic_numbers,
        "n_atoms": n_atoms,
        "builtins": builtins,
        "properties": properties,
    }


def _record_to_molecule(record):
    builtins = record["builtins"]
    mol = Molecule.from_arrays(
        record["positions"],
        record["atomic_numbers"],
        energy=builtins["energy"],
        forces=builtins["forces"],
        charges=builtins["charges"],
        velocities=builtins["velocities"],
        cell=builtins["cell"],
        stress=builtins["stress"],
        pbc=builtins["pbc"],
    )
    for key, value in record["properties"].items():
        mol.set_property(key, value)
    return mol


def _flush_fast_records(db, records):
    if not records:
        return
    builtins = records[0]["builtins"]
    positions = np.stack([record["positions"] for record in records], axis=0)
    atomic_numbers = np.stack([record["atomic_numbers"] for record in records], axis=0)
    kwargs = {}
    if builtins["energy"] is not None:
        kwargs["energy"] = np.array([r["builtins"]["energy"] for r in records], dtype=np.float64)
    if builtins["forces"] is not None:
        kwargs["forces"] = np.stack([r["builtins"]["forces"] for r in records], axis=0)
    if builtins["charges"] is not None:
        kwargs["charges"] = np.stack([r["builtins"]["charges"] for r in records], axis=0)
    if builtins["velocities"] is not None:
        kwargs["velocities"] = np.stack([r["builtins"]["velocities"] for r in records], axis=0)
    if builtins["cell"] is not None:
        kwargs["cell"] = np.stack([r["builtins"]["cell"] for r in records], axis=0)
    if builtins["stress"] is not None:
        kwargs["stress"] = np.stack([r["builtins"]["stress"] for r in records], axis=0)
    if builtins["pbc"] is not None:
        kwargs["pbc"] = np.array([r["builtins"]["pbc"] for r in records], dtype=bool)
    db.add_arrays_batch(positions, atomic_numbers, **kwargs)


def _import_ase():
    global _ASE_TYPES
    if _ASE_TYPES is None:
        try:
            from ase import Atoms
            from ase.calculators.calculator import Calculator
            from ase.calculators.singlepoint import SinglePointCalculator
        except ImportError as exc:
            raise ImportError(
                "ASE is required for Molecule.to_ase(); install it with `uv add ase`."
            ) from exc

        class NoCopySinglePointCalculator(Calculator):
            implemented_properties = ["energy", "forces", "stress", "charges"]

            def __init__(self, atoms, **results):
                Calculator.__init__(self)
                self.results = {}
                for prop, value in results.items():
                    if value is None:
                        continue
                    if prop in {"energy", "magmom", "free_energy"}:
                        self.results[prop] = value
                    else:
                        self.results[prop] = np.asarray(value, dtype=float)
                self.atoms = atoms

        _ASE_TYPES = (Atoms, SinglePointCalculator, NoCopySinglePointCalculator)
    return _ASE_TYPES


def _normalize_calc_mode(attach_calc, calc_mode):
    if calc_mode is None:
        calc_mode = "singlepoint"
    if calc_mode not in _CALC_MODES:
        raise ValueError(f"Invalid calc_mode {calc_mode!r}; expected one of {sorted(_CALC_MODES)}")
    if not attach_calc:
        return "none"
    return calc_mode


def _build_ase_atoms(payload, atoms_cls, calc_factory):
    if isinstance(payload, tuple):
        return _build_ase_atoms_from_tuple(payload, atoms_cls, calc_factory)

    kwargs = {
        "numbers": payload["numbers"],
        "positions": payload["positions"],
    }
    cell = payload.get("cell")
    if cell is not None:
        kwargs["cell"] = cell
    pbc = payload.get("pbc")
    if pbc is not None:
        kwargs["pbc"] = pbc

    atoms = atoms_cls(**kwargs)

    velocities = payload.get("velocities")
    if velocities is not None:
        atoms.set_velocities(velocities)

    calc_results = payload.get("calc")
    if calc_factory is not None:
        if calc_results:
            atoms.calc = calc_factory(atoms, **calc_results)
    elif calc_results:
        energy = calc_results.get("energy")
        if energy is not None:
            atoms.info["energy"] = float(energy)

        stress = calc_results.get("stress")
        if stress is not None:
            atoms.info["stress"] = np.asarray(stress)

        forces = calc_results.get("forces")
        if forces is not None:
            atoms.set_array("forces", np.asarray(forces))

        charges = calc_results.get("charges")
        if charges is not None:
            atoms.set_array("charges", np.asarray(charges))

    for key, value in payload.get("arrays", {}).items():
        atoms.set_array(key, np.asarray(value))
    atoms.info.update(payload.get("info", {}))

    return atoms


def _build_ase_atoms_from_tuple(payload, atoms_cls, calc_factory):
    (
        numbers,
        positions,
        cell,
        pbc,
        velocities,
        energy,
        forces,
        stress,
        charges,
        arrays,
        info,
    ) = payload

    kwargs = {
        "numbers": numbers,
        "positions": positions,
    }
    if cell is not None:
        kwargs["cell"] = cell
    if pbc is not None:
        kwargs["pbc"] = pbc

    atoms = atoms_cls(**kwargs)

    if velocities is not None:
        atoms.set_velocities(velocities)

    if calc_factory is not None:
        calc_results = {}
        if energy is not None:
            calc_results["energy"] = energy
        if forces is not None:
            calc_results["forces"] = forces
        if stress is not None:
            calc_results["stress"] = stress
        if charges is not None:
            calc_results["charges"] = charges
        if calc_results:
            atoms.calc = calc_factory(atoms, **calc_results)
    else:
        if energy is not None:
            atoms.info["energy"] = float(energy)
        if stress is not None:
            atoms.info["stress"] = np.asarray(stress)
        if forces is not None:
            atoms.set_array("forces", np.asarray(forces))
        if charges is not None:
            atoms.set_array("charges", np.asarray(charges))

    if arrays is not None:
        for key, value in arrays.items():
            atoms.set_array(key, np.asarray(value))
    if info is not None:
        atoms.info.update(info)

    return atoms


def _normalize_indices(db, indices):
    if indices is None:
        return list(range(len(db)))
    return list(indices)


def _copy_flat_properties(payload, flat, index, start, stop, copy_info, copy_arrays):
    if copy_arrays:
        arrays = {}
        for key, values in flat.get("atom_properties", {}).items():
            if key in _ASE_RESERVED_ARRAYS:
                continue
            arrays[key] = np.asarray(values[start:stop])
        if arrays:
            payload["arrays"] = arrays


def _molecule_to_ase_payload(molecule, *, copy_info, copy_arrays):
    return molecule._ase_builtin_tuple_fast(copy_info=copy_info, copy_arrays=copy_arrays)


def _db_to_ase_batch(
    db,
    indices,
    *,
    calc_mode,
    copy_info,
    copy_arrays,
    atoms_cls,
    calc_factory,
):
    if not indices:
        return []

    flat = db.get_molecules_flat(indices)
    if (copy_info or copy_arrays) and flat.get("properties"):
        return to_ase_batch(
            db.get_molecules(indices),
            calc_mode=calc_mode,
            copy_info=copy_info,
            copy_arrays=copy_arrays,
        )

    n_atoms = np.asarray(flat["n_atoms"], dtype=np.uint32)
    offsets = np.empty(len(n_atoms) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(n_atoms, dtype=np.int64, out=offsets[1:])

    positions = flat["positions"]
    atomic_numbers = flat["atomic_numbers"]
    cells = flat.get("cell")
    pbc = flat.get("pbc")
    velocities = flat.get("velocities")
    energies = flat.get("energy")
    forces = flat.get("forces")
    stress = flat.get("stress")
    charges = flat.get("charges")

    atoms_list = []
    for i in range(len(indices)):
        start = int(offsets[i])
        stop = int(offsets[i + 1])
        payload = {
            "numbers": atomic_numbers[start:stop],
            "positions": positions[start:stop],
        }
        if cells is not None:
            payload["cell"] = cells[i]
        if pbc is not None:
            payload["pbc"] = pbc[i]
        if velocities is not None:
            payload["velocities"] = velocities[start:stop]

        calc = {}
        if energies is not None:
            calc["energy"] = float(energies[i])
        if forces is not None:
            calc["forces"] = forces[start:stop]
        if stress is not None:
            calc["stress"] = stress[i]
        if charges is not None:
            calc["charges"] = charges[start:stop]
        if calc:
            payload["calc"] = calc

        _copy_flat_properties(payload, flat, i, start, stop, copy_info, copy_arrays)
        atoms_list.append(_build_ase_atoms(payload, atoms_cls, calc_factory))

    return atoms_list


def to_ase(
    molecule,
    *,
    attach_calc=True,
    calc_mode="singlepoint",
    copy_info=True,
    copy_arrays=True,
):
    """Convert an atompack molecule to ``ase.Atoms``.

    The conversion reads directly from the molecule getters, so it works for
    both owned and view-backed molecules without going through ``molecule.atoms()``.
    That keeps the path compatible with lazy SOA-backed molecules, although ASE
    object creation still requires Python/NumPy allocations.

    Mapping rules:

    - ``positions`` and ``atomic_numbers`` always become the ASE geometry.
    - ``cell`` and ``pbc`` are copied when present.
    - ``velocities`` are attached with ``atoms.set_velocities(...)``.
    - ``energy``, ``forces``, ``stress``, and ``charges`` are attached through
      an ASE calculator when ``attach_calc=True``. ``calc_mode="singlepoint"``
      preserves ASE's snapshot semantics, while ``calc_mode="nocopy"`` is
      faster but does not snapshot the atoms state.
    - Custom properties shaped like per-atom arrays are stored in
      ``atoms.arrays`` when ``copy_arrays=True``.
    - Remaining custom properties are stored in ``atoms.info`` when
      ``copy_info=True``.

    Parameters
    ----------
    molecule : atompack.Molecule
        Molecule to convert.
    attach_calc : bool, default=True
        Attach supported builtin results through an ASE calculator.
    calc_mode : {"singlepoint", "nocopy", "none"}, default="singlepoint"
        Calculator attachment mode. ``"singlepoint"`` uses ASE's standard
        snapshotting calculator, ``"nocopy"`` skips the internal atoms copy for
        higher throughput, and ``"none"`` suppresses calculator attachment.
    copy_info : bool, default=True
        Copy non-array custom properties into ``atoms.info``.
    copy_arrays : bool, default=True
        Copy per-atom custom arrays into ``atoms.arrays``.

    Returns
    -------
    ase.Atoms
        Converted ASE object.
    """
    atoms_cls, single_point_calculator_cls, nocopy_single_point_calculator_cls = _import_ase()
    calc_mode = _normalize_calc_mode(attach_calc, calc_mode)
    calc_factory = {
        "singlepoint": single_point_calculator_cls,
        "nocopy": nocopy_single_point_calculator_cls,
        "none": None,
    }[calc_mode]
    payload = _molecule_to_ase_payload(
        molecule,
        copy_info=copy_info,
        copy_arrays=copy_arrays,
    )
    return _build_ase_atoms(payload, atoms_cls, calc_factory)


def to_ase_batch(
    source,
    indices=None,
    *,
    attach_calc=True,
    calc_mode="singlepoint",
    copy_info=True,
    copy_arrays=True,
):
    """Convert many atompack molecules to ASE Atoms efficiently."""
    atoms_cls, single_point_calculator_cls, nocopy_single_point_calculator_cls = _import_ase()
    calc_mode = _normalize_calc_mode(attach_calc, calc_mode)
    calc_factory = {
        "singlepoint": single_point_calculator_cls,
        "nocopy": nocopy_single_point_calculator_cls,
        "none": None,
    }[calc_mode]
    if hasattr(source, "get_molecules_flat"):
        return _db_to_ase_batch(
            source,
            _normalize_indices(source, indices),
            calc_mode=calc_mode,
            copy_info=copy_info,
            copy_arrays=copy_arrays,
            atoms_cls=atoms_cls,
            calc_factory=calc_factory,
        )

    molecules = list(source if indices is None else (source[index] for index in indices))
    return [
        _build_ase_atoms(
            _molecule_to_ase_payload(
                molecule,
                copy_info=copy_info,
                copy_arrays=copy_arrays,
            ),
            atoms_cls,
            calc_factory,
        )
        for molecule in molecules
    ]


def _database_to_ase_batch(
    self,
    indices=None,
    *,
    attach_calc=True,
    calc_mode="singlepoint",
    copy_info=True,
    copy_arrays=True,
):
    return to_ase_batch(
        self,
        indices=indices,
        attach_calc=attach_calc,
        calc_mode=calc_mode,
        copy_info=copy_info,
        copy_arrays=copy_arrays,
    )


def _normalize_info_overrides(info, count):
    if info is None:
        return [None] * count
    if isinstance(info, dict):
        return [info] * count
    overrides = list(info)
    if len(overrides) != count:
        raise ValueError(
            f"info override length ({len(overrides)}) doesn't match atoms count ({count})"
        )
    return overrides


def _fast_key(record):
    builtins = record["builtins"]
    return (
        record["n_atoms"],
        builtins["energy"] is not None,
        builtins["forces"] is not None,
        builtins["charges"] is not None,
        builtins["velocities"] is not None,
        builtins["cell"] is not None,
        builtins["stress"] is not None,
        builtins["pbc"] is not None,
    )


def from_ase(
    atoms,
    energy=None,
    forces=None,
    charges=None,
    velocities=None,
    cell=None,
    stress=None,
    copy_info=True,
    info=None,
):
    """Convert one ASE Atoms object to an atompack Molecule."""
    return _record_to_molecule(
        _extract_ase_record(
            atoms,
            energy=energy,
            forces=forces,
            charges=charges,
            velocities=velocities,
            cell=cell,
            stress=stress,
            copy_info=copy_info,
            info=info,
        )
    )


def add_ase_batch(
    db,
    atoms_list,
    *,
    copy_info=True,
    info=None,
    batch_size=512,
):
    """Write many ASE Atoms objects efficiently, preserving supported metadata."""
    atoms_list = list(atoms_list)
    if not atoms_list:
        return

    info_overrides = _normalize_info_overrides(info, len(atoms_list))
    fast_key = None
    fast_records = []
    slow_records = []

    def flush_fast():
        nonlocal fast_key
        if fast_records:
            _flush_fast_records(db, fast_records)
            fast_records.clear()
            fast_key = None

    def flush_slow():
        if slow_records:
            db.add_molecules(slow_records)
            slow_records.clear()

    for atoms, info_override in zip(atoms_list, info_overrides):
        record = _extract_ase_record(atoms, copy_info=copy_info, info=info_override)
        if record["properties"]:
            flush_fast()
            slow_records.append(_record_to_molecule(record))
            if len(slow_records) >= batch_size:
                flush_slow()
            continue

        flush_slow()
        key = _fast_key(record)
        if fast_key != key:
            flush_fast()
            fast_key = key
        fast_records.append(record)
        if len(fast_records) >= batch_size:
            flush_fast()

    flush_fast()
    flush_slow()


Molecule.to_ase = to_ase
Database.to_ase_batch = _database_to_ase_batch
