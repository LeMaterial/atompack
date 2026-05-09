# Copyright 2026 Entalpic
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import atompack
import numpy as np
import pytest

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        os.environ.get("ATOMPACK_RUN_PERF_SMOKE") != "1",
        reason="set ATOMPACK_RUN_PERF_SMOKE=1 or run `make perf-smoke-py`",
    ),
]

N_MOLECULES = int(os.environ.get("ATOMPACK_PY_PERF_N_MOLECULES", "10000"))
ATOMS_PER_MOLECULE = int(os.environ.get("ATOMPACK_PY_PERF_ATOMS_PER_MOLECULE", "64"))
READ_SAMPLE = int(os.environ.get("ATOMPACK_PY_PERF_READ_SAMPLE", "5000"))


def _color(code: str, text: str) -> str:
    color_mode = os.environ.get("ATOMPACK_PERF_COLOR", "auto")
    if color_mode == "always":
        return f"\033[{code}m{text}\033[0m"
    if color_mode == "never" or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def _cyan(text: str) -> str:
    return _color("36;1", text)


def _green(text: str) -> str:
    return _color("32;1", text)


def _red(text: str) -> str:
    return _color("31;1", text)


def _yellow(text: str) -> str:
    return _color("33;1", text)


def _threshold(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _rate(units: int, seconds: float) -> float:
    return units / max(seconds, 1e-12)


def _format_rate(value: float) -> str:
    return f"{value:>12,.0f}"


def _print_metric(label: str, value: float, threshold_env: str, default: float) -> None:
    floor = _threshold(threshold_env, default)
    status = _green("PASS") if value >= floor else _red("FAIL")
    print(
        f"  {label:<24} "
        f"{_format_rate(value)} mol/s  "
        f"{_format_rate(value * ATOMS_PER_MOLECULE)} atoms/s  "
        f"min {_format_rate(floor)}  "
        f"{status:<13} {threshold_env}"
    )


def _print_report(
    *,
    write_mol_s: float,
    sequential_read_mol_s: float,
    shuffled_read_mol_s: float,
    flat_read_mol_s: float,
    read_sample: int,
    file_size_bytes: int,
) -> None:
    print()
    print(_cyan("Atompack Python Throughput Smoke"))
    print(
        f"  dataset: {N_MOLECULES:,} molecules x {ATOMS_PER_MOLECULE} atoms, "
        "compression=none, props=builtin+custom"
    )
    print(
        f"  reads: sample={read_sample:,}, file_size={file_size_bytes:,} bytes, extension=release"
    )
    print(f"  {_yellow('small warm-cache smoke test; not a publication benchmark')}")
    print()
    print(
        f"  {'metric':<24} {'throughput':>18}  {'atom throughput':>18}  "
        f"{'floor':>16}  {'status':<13} env override"
    )
    print(f"  {'-' * 105}")
    _print_metric("write add_arrays_batch", write_mol_s, "ATOMPACK_PY_MIN_WRITE_MOL_S", 10_000.0)
    _print_metric(
        "sequential get_molecule",
        sequential_read_mol_s,
        "ATOMPACK_PY_MIN_SEQUENTIAL_READ_MOL_S",
        50_000.0,
    )
    _print_metric(
        "shuffled get_molecule",
        shuffled_read_mol_s,
        "ATOMPACK_PY_MIN_SHUFFLED_READ_MOL_S",
        50_000.0,
    )
    _print_metric(
        "flat get_molecules_flat",
        flat_read_mol_s,
        "ATOMPACK_PY_MIN_FLAT_READ_MOL_S",
        50_000.0,
    )
    print()


def _measure_repeated(
    fn: Callable[[], object],
    *,
    units_per_call: int,
    min_seconds: float = 0.05,
    max_repeats: int = 50,
) -> float:
    fn()
    elapsed = 0.0
    repeats = 0
    last_result = None
    while elapsed < min_seconds and repeats < max_repeats:
        start = time.perf_counter()
        last_result = fn()
        elapsed += time.perf_counter() - start
        repeats += 1
    assert last_result is not None
    return _rate(units_per_call * repeats, elapsed)


def _synthetic_arrays() -> dict[str, object]:
    mol_index = np.arange(N_MOLECULES, dtype=np.float32)[:, None]
    atom_index = np.arange(ATOMS_PER_MOLECULE, dtype=np.float32)[None, :]
    phase = mol_index * 0.001 + atom_index * 0.01

    positions = np.empty((N_MOLECULES, ATOMS_PER_MOLECULE, 3), dtype=np.float32)
    positions[..., 0] = np.sin(phase * 1.3) * 10.0
    positions[..., 1] = np.cos(phase * 1.7) * 10.0
    positions[..., 2] = np.sin(phase * 2.1) * 10.0

    forces = np.empty_like(positions)
    forces[..., 0] = np.sin(phase * 0.7)
    forces[..., 1] = np.cos(phase * 0.9)
    forces[..., 2] = np.sin(phase * 1.1)

    elements = np.array([1, 6, 7, 8, 16], dtype=np.uint8)
    atomic_numbers = elements[
        (np.arange(N_MOLECULES)[:, None] + np.arange(ATOMS_PER_MOLECULE)[None, :]) % len(elements)
    ]
    energy = -1_000.0 - np.arange(N_MOLECULES, dtype=np.float64) * 1e-3
    cell = np.broadcast_to(
        np.eye(3, dtype=np.float64) * 10.0,
        (N_MOLECULES, 3, 3),
    ).copy()
    pbc = np.broadcast_to(
        np.array([True, True, True], dtype=np.bool_),
        (N_MOLECULES, 3),
    ).copy()

    properties = {
        "bandgap": np.linspace(0.0, 5.0, N_MOLECULES, dtype=np.float64),
        "formation_energy": np.sin(np.arange(N_MOLECULES, dtype=np.float64) * 0.01),
        "eigenvalues": np.stack(
            [
                np.sin(np.arange(N_MOLECULES, dtype=np.float64) * scale)
                for scale in (0.01, 0.02, 0.03, 0.04)
            ],
            axis=1,
        ),
    }
    atom_properties = {
        "mulliken_charges": np.sin(phase.astype(np.float64) * 0.5),
    }

    return {
        "positions": positions,
        "atomic_numbers": atomic_numbers,
        "energy": energy,
        "forces": forces,
        "cell": cell,
        "pbc": pbc,
        "properties": properties,
        "atom_properties": atom_properties,
    }


def _touch_molecule(molecule: atompack.Molecule, seed: int) -> None:
    _ = float(molecule.positions[seed % len(molecule.positions), seed % 3])
    _ = float(molecule.forces[seed % len(molecule.forces), seed % 3])
    _ = molecule.energy


def test_atompack_python_throughput_smoke(tmp_path: Path) -> None:
    arrays = _synthetic_arrays()
    path = tmp_path / "throughput-smoke.atp"

    start = time.perf_counter()
    db = atompack.Database(str(path), compression="none", overwrite=True)
    db.add_arrays_batch(**arrays)
    db.flush()
    write_seconds = time.perf_counter() - start
    write_mol_s = _rate(N_MOLECULES, write_seconds)
    file_size_bytes = path.stat().st_size

    db = atompack.Database.open(str(path), mmap=True)
    seq_indices = list(range(min(READ_SAMPLE, N_MOLECULES)))
    rng = np.random.default_rng(12_345)
    shuffled_indices = rng.integers(
        0,
        N_MOLECULES,
        size=min(READ_SAMPLE, N_MOLECULES),
        dtype=np.int64,
    ).tolist()

    def read_sequential_single() -> object:
        for index in seq_indices:
            _touch_molecule(db.get_molecule(index), index)
        return len(seq_indices)

    def read_shuffled_single() -> object:
        for index in shuffled_indices:
            _touch_molecule(db.get_molecule(index), index)
        return len(shuffled_indices)

    def read_flat() -> object:
        batch = db.get_molecules_flat(seq_indices)
        assert batch["positions"].shape == (len(seq_indices) * ATOMS_PER_MOLECULE, 3)
        return batch

    sequential_read_mol_s = _measure_repeated(
        read_sequential_single,
        units_per_call=len(seq_indices),
    )
    shuffled_read_mol_s = _measure_repeated(
        read_shuffled_single,
        units_per_call=len(shuffled_indices),
    )
    flat_read_mol_s = _measure_repeated(read_flat, units_per_call=len(seq_indices))

    print(
        "atompack_python_perf_smoke "
        f"n_molecules={N_MOLECULES} "
        f"atoms_per_molecule={ATOMS_PER_MOLECULE} "
        f"read_sample={len(seq_indices)} "
        f"write_mol_s={write_mol_s:.0f} "
        f"sequential_read_mol_s={sequential_read_mol_s:.0f} "
        f"shuffled_read_mol_s={shuffled_read_mol_s:.0f} "
        f"flat_read_mol_s={flat_read_mol_s:.0f} "
        f"file_size_bytes={file_size_bytes}"
    )
    _print_report(
        write_mol_s=write_mol_s,
        sequential_read_mol_s=sequential_read_mol_s,
        shuffled_read_mol_s=shuffled_read_mol_s,
        flat_read_mol_s=flat_read_mol_s,
        read_sample=len(seq_indices),
        file_size_bytes=file_size_bytes,
    )

    assert write_mol_s >= _threshold("ATOMPACK_PY_MIN_WRITE_MOL_S", 10_000.0)
    assert sequential_read_mol_s >= _threshold(
        "ATOMPACK_PY_MIN_SEQUENTIAL_READ_MOL_S",
        50_000.0,
    )
    assert shuffled_read_mol_s >= _threshold(
        "ATOMPACK_PY_MIN_SHUFFLED_READ_MOL_S",
        50_000.0,
    )
    assert flat_read_mol_s >= _threshold("ATOMPACK_PY_MIN_FLAT_READ_MOL_S", 50_000.0)
