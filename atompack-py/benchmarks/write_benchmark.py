# Copyright 2026 Entalpic
"""
Write benchmarks — throughput, storage footprint, and Atompack batch sweep.

Run as a standalone script so RAYON_NUM_THREADS is NOT forced to 1,
allowing Rayon to use all available cores for parallel compression.

Three benchmarks:

  1. Throughput across atom counts
       Fixed n_mols, sweep atom counts [16, 64, 256].
       Shows atompack advantage across molecule sizes.

  2. Storage footprint across atom counts
       Fixed n_mols, sweep atom counts [16, 64, 256].
       Reports dataset size, normalized bytes / molecule, bytes / atom,
       and ratio vs Atompack for each backend.

  3. Atompack batch-size sweep
       Fixed n_mols, sweep Atompack write batch size at selected atom counts.
       Useful when you want to see where the builtins write path becomes
       memory-traffic bound at larger molecules.

Usage:
  python write_benchmark.py                     # run both benchmarks
  python write_benchmark.py --bench 1           # throughput only
  python write_benchmark.py --bench 2           # storage footprint only
  python write_benchmark.py --bench 3           # atompack batch sweep only
  python write_benchmark.py --codec zstd:3      # opt into compressed writes
  python write_benchmark.py --scratch-dir /path/with/space
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import atompack

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atom_lmdb import lmdb_codec_fns, pack_record
from atom_hdf5_soa import AtomHdf5Soa, AtomHdf5SoaConfig, hdf5_available
from atom_lmdb_pickle import PickleAtomLMDB, PickleLMDBConfig

try:
    import lmdb as _lmdb
except ImportError:
    _lmdb = None

try:
    import ase.db
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    _has_ase = True
except ImportError:
    _has_ase = False

WRITE_BATCH_SIZE = 10_000
HDF5_SOA_CHUNK_SIZE = 256
ASE_WRITE_MAX = 5_000
DEFAULT_WRITE_CODEC = "none"
DEFAULT_SCRATCH_DIR = "/ogre/tmp"
DEFAULT_ATOMPACK_TARGET_BATCH_MIB = 16.0
DEFAULT_BATCH_SWEEP_SIZES = [256, 512, 1024, 2048, 4096, WRITE_BATCH_SIZE]
DEFAULT_WARMUP_TRIALS = 1
DEFAULT_MIN_TRIAL_SECONDS = 1.0
DEFAULT_MAX_TRIAL_REPEATS = 7
_ASE_LMDB_EXT: str | None = None
if _has_ase:
    for _ext in (".aselmdb", ".lmdb"):
        try:
            _tmp = tempfile.mktemp(suffix=_ext)
            ase.db.connect(_tmp)
            if os.path.exists(_tmp):
                os.unlink(_tmp)
            _ASE_LMDB_EXT = _ext
            break
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


def _dir_size(path: str | Path) -> int:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.iterdir() if f.is_file())


def _fmt(mol_s: float, size_bytes: int) -> str:
    return f"{mol_s:>12,.0f} mol/s  ({size_bytes / 1e6:.0f} MB)"


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))


def _median_int(values: list[int]) -> int:
    return int(round(_median([float(v) for v in values])))


def _sample_write_rates(
    *,
    trials: int,
    n_units_per_write: int,
    warmup_trials: int,
    min_trial_seconds: float,
    max_trial_repeats: int,
    write_once,
    size_of,
) -> tuple[list[float], int, list[int]]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if warmup_trials < 0:
        raise ValueError("warmup_trials must be non-negative")
    if min_trial_seconds < 0:
        raise ValueError("min_trial_seconds must be non-negative")
    if max_trial_repeats <= 0:
        raise ValueError("max_trial_repeats must be positive")

    for warmup in range(warmup_trials):
        write_once(f"warmup_{warmup}")
        gc.collect()

    rates: list[float] = []
    repeat_counts: list[int] = []
    last_size = 0
    for trial in range(trials):
        total_elapsed = 0.0
        total_units = 0
        repeats = 0
        last_path = None
        while repeats == 0 or (
            total_elapsed < min_trial_seconds and repeats < max_trial_repeats
        ):
            token = f"trial{trial}_rep{repeats}"
            t0 = time.perf_counter()
            last_path = write_once(token)
            total_elapsed += time.perf_counter() - t0
            total_units += n_units_per_write
            repeats += 1
            gc.collect()
        if total_elapsed <= 0:
            raise RuntimeError("Measured non-positive write duration")
        rates.append(total_units / total_elapsed)
        repeat_counts.append(repeats)
        if last_path is not None:
            last_size = size_of(last_path)
    return rates, last_size, repeat_counts


def _safe_close_ase_db(db: Any) -> None:
    """Close ASE backends without triggering broken destructor reopen paths."""
    if db is None:
        return

    env = getattr(db, "_env", None)
    if env is not None:
        try:
            env.close()
        except Exception:
            pass
        try:
            db._env = None
        except Exception:
            pass
    else:
        try:
            db.close()
        except Exception:
            pass

    # Some third-party ASE backends call self.close() in __del__. Shadow it
    # so garbage collection cannot reopen resources after explicit cleanup.
    try:
        setattr(db, "close", lambda: None)
    except Exception:
        pass


def _write_results_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _parse_codec_spec(codec: str) -> tuple[str, int, str]:
    codec = (codec or DEFAULT_WRITE_CODEC).strip().lower()
    if codec == "none":
        return "none", 0, "none"
    if codec == "lz4":
        return "lz4", 0, "lz4"
    if codec == "zstd":
        return "zstd", 3, "zstd:3"
    if codec.startswith("zstd:"):
        level_text = codec.split(":", 1)[1]
        try:
            level = int(level_text)
        except ValueError as exc:
            raise ValueError(f"Invalid zstd codec level: {codec}") from exc
        if level < 1:
            raise ValueError(f"Invalid zstd codec level: {codec}")
        return "zstd", level, f"zstd:{level}"
    raise ValueError(f"Unsupported write benchmark codec: {codec}")


def _estimate_lmdb_packed_map_size(
    n: int,
    atoms: int,
    *,
    with_custom: bool = False,
) -> int:
    avg_bytes = atoms * (12 + 12 + 1) + 72 + 8 + 256
    if with_custom:
        avg_bytes += 320 + atoms * 16
    return int(n * avg_bytes * 4) + 512 * (1 << 20)


def _estimate_lmdb_pickle_map_size(
    n: int,
    atoms: int,
    *,
    with_custom: bool = False,
) -> int:
    avg_bytes = atoms * (12 + 12 + 1) + 72 + 8 + 512
    if with_custom:
        avg_bytes += 320 + atoms * 16
    return int(n * avg_bytes * 4) + 512 * (1 << 20)


def _estimate_atompack_payload_bytes_per_molecule(atoms: int, *, with_custom: bool) -> int:
    total = atoms * (12 + 12 + 1) + 72 + 8 + 3
    if with_custom:
        total += 320 + atoms * 16
    return total


def _recommend_atompack_batch_size(
    atoms: int,
    *,
    with_custom: bool,
    target_batch_mib: float = DEFAULT_ATOMPACK_TARGET_BATCH_MIB,
    max_batch_size: int = WRITE_BATCH_SIZE,
) -> int:
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if target_batch_mib <= 0:
        raise ValueError("target_batch_mib must be positive")

    target_bytes = int(float(target_batch_mib) * 1024 * 1024)
    per_molecule = _estimate_atompack_payload_bytes_per_molecule(
        atoms,
        with_custom=with_custom,
    )
    if per_molecule <= 0:
        return max_batch_size
    heuristic_batch = max(1, min(max_batch_size, target_bytes // per_molecule))

    # The benchmark path allocates full temporary NumPy batches in Python and
    # then serializes whole records in Rust, so a pure payload-size heuristic
    # consistently over-batches larger molecules. Clamp the auto batch to
    # empirically tuned caps that match the faster plateau for this workload.
    if with_custom:
        calibrated_cap = 4_096 if atoms <= 16 else 1_024
    else:
        calibrated_cap = 2_048 if atoms <= 64 else 256

    return max(1, min(heuristic_batch, calibrated_cap))


def _resolve_atompack_batch_size(
    atoms: int,
    *,
    with_custom: bool,
    atompack_batch_size: int | None,
    atompack_target_batch_mib: float,
) -> int:
    if atompack_batch_size is not None:
        if atompack_batch_size <= 0:
            raise ValueError("atompack_batch_size must be positive")
        return atompack_batch_size
    return _recommend_atompack_batch_size(
        atoms,
        with_custom=with_custom,
        target_batch_mib=atompack_target_batch_mib,
    )


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_molecule_stream(
    n: int, atoms: int, seed: int = 42, with_custom: bool = False
):
    rng = np.random.RandomState(seed)
    cell = np.eye(3, dtype=np.float64) * 10.0
    pbc = np.array([True, True, True], dtype=bool)
    for _ in range(n):
        mol: dict[str, Any] = {
            "positions": rng.randn(atoms, 3).astype(np.float32),
            "atomic_numbers": rng.randint(1, 80, atoms).astype(np.uint8),
            "energy": float(rng.randn()),
            "forces": rng.randn(atoms, 3).astype(np.float32),
            "cell": cell,
            "pbc": pbc,
        }
        if with_custom:
            mol["custom"] = {
                "bandgap": float(rng.uniform(0, 5)),
                "formation_energy": float(rng.randn()),
                "eigenvalues": rng.randn(20).astype(np.float64),
                "mulliken_charges": rng.randn(atoms).astype(np.float64),
                "hirshfeld_volumes": rng.uniform(5, 30, atoms).astype(np.float64),
            }
        yield mol


def _write_atompack_builtin_batch(
    db,
    rng: np.random.RandomState,
    batch_size: int,
    atoms: int,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> None:
    positions = rng.randn(batch_size, atoms, 3).astype(np.float32)
    atomic_numbers = rng.randint(1, 80, size=(batch_size, atoms)).astype(np.uint8)
    energy = rng.randn(batch_size).astype(np.float64)
    forces = rng.randn(batch_size, atoms, 3).astype(np.float32)
    cell_batch = np.broadcast_to(cell, (batch_size, 3, 3)).copy()
    pbc_batch = np.broadcast_to(pbc, (batch_size, 3)).copy()
    db.add_arrays_batch(
        positions,
        atomic_numbers,
        energy=energy,
        forces=forces,
        cell=cell_batch,
        pbc=pbc_batch,
    )


def _write_atompack_custom_batch(
    db,
    rng: np.random.RandomState,
    batch_size: int,
    atoms: int,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> None:
    positions = rng.randn(batch_size, atoms, 3).astype(np.float32)
    atomic_numbers = rng.randint(1, 80, size=(batch_size, atoms)).astype(np.uint8)
    energy = rng.randn(batch_size).astype(np.float64)
    forces = rng.randn(batch_size, atoms, 3).astype(np.float32)
    cell_batch = np.broadcast_to(cell, (batch_size, 3, 3)).copy()
    pbc_batch = np.broadcast_to(pbc, (batch_size, 3)).copy()
    properties = {
        "bandgap": rng.uniform(0, 5, size=batch_size).astype(np.float64),
        "formation_energy": rng.randn(batch_size).astype(np.float64),
        "eigenvalues": rng.randn(batch_size, 20).astype(np.float64),
    }
    atom_properties = {
        "mulliken_charges": rng.randn(batch_size, atoms).astype(np.float64),
        "hirshfeld_volumes": rng.uniform(5, 30, size=(batch_size, atoms)).astype(np.float64),
    }
    db.add_arrays_batch(
        positions,
        atomic_numbers,
        energy=energy,
        forces=forces,
        cell=cell_batch,
        pbc=pbc_batch,
        properties=properties,
        atom_properties=atom_properties,
    )


# ---------------------------------------------------------------------------
# Write functions
# ---------------------------------------------------------------------------


def write_atompack(
    path: str,
    n: int,
    atoms: int,
    with_custom: bool = False,
    report: bool = False,
    compression: str = DEFAULT_WRITE_CODEC,
    level: int = 0,
    atompack_batch_size: int | None = None,
    atompack_target_batch_mib: float = DEFAULT_ATOMPACK_TARGET_BATCH_MIB,
) -> None:
    report_every = max(1, n // 10)
    db = atompack.Database(path, compression=compression, level=level, overwrite=True)
    batch_size_limit = _resolve_atompack_batch_size(
        atoms,
        with_custom=with_custom,
        atompack_batch_size=atompack_batch_size,
        atompack_target_batch_mib=atompack_target_batch_mib,
    )
    if not with_custom:
        rng = np.random.RandomState(42)
        cell = np.eye(3, dtype=np.float64) * 10.0
        pbc = np.array([True, True, True], dtype=bool)
        written = 0
        while written < n:
            if report and n >= 1_000_000 and written > 0 and written % report_every == 0:
                print(f"    atompack {written:,}/{n:,}", flush=True)
            batch_size = min(batch_size_limit, n - written)
            _write_atompack_builtin_batch(db, rng, batch_size, atoms, cell, pbc)
            written += batch_size
        db.flush()
        return

    rng = np.random.RandomState(42)
    cell = np.eye(3, dtype=np.float64) * 10.0
    pbc = np.array([True, True, True], dtype=bool)
    written = 0
    while written < n:
        if report and n >= 1_000_000 and written > 0 and written % report_every == 0:
            print(f"    atompack {written:,}/{n:,}", flush=True)
        batch_size = min(batch_size_limit, n - written)
        _write_atompack_custom_batch(db, rng, batch_size, atoms, cell, pbc)
        written += batch_size
    db.flush()


def write_lmdb_packed(
    path: str,
    n: int,
    atoms: int,
    with_custom: bool = False,
    report: bool = False,
    codec: str = DEFAULT_WRITE_CODEC,
) -> None:
    report_every = max(1, n // 10)
    shutil.rmtree(path, ignore_errors=True)
    Path(path).mkdir(parents=True, exist_ok=True)
    compress, _ = lmdb_codec_fns(codec)
    avg_bytes = atoms * (12 + 12 + 1) + 72 + 8 + 256
    if with_custom:
        avg_bytes += 320 + atoms * 16
    map_size = int(n * avg_bytes * 4) + 512 * (1 << 20)
    env = _lmdb.open(path, map_size=map_size)
    txn = env.begin(write=True)
    for i, m in enumerate(generate_molecule_stream(n, atoms, with_custom=with_custom)):
        if report and n >= 1_000_000 and i > 0 and i % report_every == 0:
            print(f"    lmdb_packed {i:,}/{n:,}", flush=True)
        rec = pack_record(
            m["positions"], m["atomic_numbers"], m["cell"], m["pbc"],
            m["energy"], m["forces"],
            custom=m.get("custom"),
        )
        if compress:
            rec = compress(rec)
        txn.put(struct.pack(">Q", i), rec)
        if (i + 1) % 50_000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()


def write_lmdb_pickle(
    path: str,
    n: int,
    atoms: int,
    with_custom: bool = False,
    report: bool = False,
    codec: str = DEFAULT_WRITE_CODEC,
) -> None:
    report_every = max(1, n // 10)
    shutil.rmtree(path, ignore_errors=True)
    store = PickleAtomLMDB(Path(path), PickleLMDBConfig(codec=codec))
    store.reset_dir()
    avg_bytes = atoms * (12 + 12 + 1) + 72 + 8 + 512
    if with_custom:
        avg_bytes += 320 + atoms * 16
    map_size = int(n * avg_bytes * 4) + 512 * (1 << 20)
    env = store.open_env(map_size=map_size, readonly=False, lock=True)
    txn = env.begin(write=True)
    for i, m in enumerate(generate_molecule_stream(n, atoms, with_custom=with_custom)):
        if report and n >= 1_000_000 and i > 0 and i % report_every == 0:
            print(f"    lmdb_pickle {i:,}/{n:,}", flush=True)
        store.put_molecule(
            txn, i, m["positions"], m["atomic_numbers"],
            m["cell"], m["pbc"], m["energy"], m["forces"],
            custom=m.get("custom"),
        )
        if (i + 1) % 50_000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()


def write_hdf5_soa(
    path: str,
    n: int,
    atoms: int,
    with_custom: bool = False,
    report: bool = False,
) -> None:
    if not hdf5_available():
        raise RuntimeError("h5py is required for hdf5_soa benchmarks")

    report_every = max(1, n // 10)
    cfg = AtomHdf5SoaConfig(
        atoms_per_molecule=atoms,
        with_props=True,
        with_cell_pbc=True,
        compression="none",
        chunk_size=HDF5_SOA_CHUNK_SIZE,
    )
    store = AtomHdf5Soa(Path(path), cfg)
    store.create_file(n)

    rng = np.random.RandomState(42)
    cell = np.eye(3, dtype=np.float64) * 10.0
    pbc = np.array([1, 1, 1], dtype=np.uint8)
    with store.open_file("r+") as handle:
        written = 0
        while written < n:
            if report and n >= 1_000_000 and written > 0 and written % report_every == 0:
                print(f"    hdf5_soa {written:,}/{n:,}", flush=True)
            batch_n = min(cfg.chunk_size, n - written)
            positions = rng.randn(batch_n, atoms, 3).astype(np.float32)
            atomic_numbers = rng.randint(1, 80, size=(batch_n, atoms)).astype(np.uint8)
            energy = rng.randn(batch_n).astype(np.float64)
            forces = rng.randn(batch_n, atoms, 3).astype(np.float32)
            cell_batch = np.broadcast_to(cell, (batch_n, 3, 3)).copy()
            pbc_batch = np.broadcast_to(pbc, (batch_n, 3)).copy()
            properties = None
            atom_properties = None
            if with_custom:
                properties = {
                    "bandgap": rng.uniform(0, 5, size=batch_n).astype(np.float64),
                    "formation_energy": rng.randn(batch_n).astype(np.float64),
                    "eigenvalues": rng.randn(batch_n, 20).astype(np.float64),
                }
                atom_properties = {
                    "mulliken_charges": rng.randn(batch_n, atoms).astype(np.float64),
                    "hirshfeld_volumes": rng.uniform(5, 30, size=(batch_n, atoms)).astype(np.float64),
                }
            store.put_batch(
                handle,
                written,
                positions,
                atomic_numbers,
                cell_batch,
                pbc_batch,
                energy,
                forces,
                properties=properties,
                atom_properties=atom_properties,
            )
            written += batch_n


def write_ase(
    path: str, n: int, atoms: int, with_custom: bool = False, report: bool = False
) -> None:
    if not _has_ase:
        raise RuntimeError("ASE is not available")
    report_every = max(1, n // 10)
    db = ase.db.connect(path)
    try:
        # Reuse one ASE transaction for the full write loop. This avoids the
        # per-row begin/commit overhead that dominates ASE LMDB/SQLite writes.
        with db:
            for i, m in enumerate(
                generate_molecule_stream(n, atoms, with_custom=with_custom)
            ):
                if report and n >= 1_000 and i > 0 and i % report_every == 0:
                    print(f"    ase {i:,}/{n:,}", flush=True)
                atoms_obj = Atoms(
                    numbers=m["atomic_numbers"],
                    positions=m["positions"],
                    cell=m["cell"],
                    pbc=m["pbc"],
                )
                atoms_obj.calc = SinglePointCalculator(
                    atoms_obj,
                    energy=m["energy"],
                    forces=m["forces"],
                )
                db.write(atoms_obj, data=m.get("custom"))
    finally:
        _safe_close_ase_db(db)
        gc.collect()


# ---------------------------------------------------------------------------
# Benchmark 1: throughput across atom counts
# ---------------------------------------------------------------------------


def bench_throughput(
    atom_counts: list[int],
    n_mols: int = 50_000,
    trials: int = 3,
    scratch_dir: str | None = None,
    codec: str = DEFAULT_WRITE_CODEC,
    atompack_batch_size: int | None = None,
    atompack_target_batch_mib: float = DEFAULT_ATOMPACK_TARGET_BATCH_MIB,
    warmup_trials: int = DEFAULT_WARMUP_TRIALS,
    min_trial_seconds: float = DEFAULT_MIN_TRIAL_SECONDS,
    max_trial_repeats: int = DEFAULT_MAX_TRIAL_REPEATS,
) -> list[dict[str, Any]]:
    atompack_compression, atompack_level, lmdb_codec = _parse_codec_spec(codec)
    print("=" * 60)
    print("Benchmark 1: Write throughput across atom counts")
    print(
        f"  {n_mols:,} molecules, {trials} trials, codec={lmdb_codec}, "
        f"warmup={warmup_trials}, min_trial_seconds={min_trial_seconds:g}, "
        f"max_repeats={max_trial_repeats}"
    )
    print("=" * 60)

    results: list[dict[str, Any]] = []
    for atoms in atom_counts:
        for with_custom in (False, True):
            tag = " + custom" if with_custom else ""
            resolved_batch_size = _resolve_atompack_batch_size(
                atoms,
                with_custom=with_custom,
                atompack_batch_size=atompack_batch_size,
                atompack_target_batch_mib=atompack_target_batch_mib,
            )
            print(f"\n--- {atoms} atoms{tag} ---")
            print(f"  atompack batch size: {resolved_batch_size:,}")

            with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
                # atompack
                rates, size, repeat_counts = _sample_write_rates(
                    trials=trials,
                    n_units_per_write=n_mols,
                    warmup_trials=warmup_trials,
                    min_trial_seconds=min_trial_seconds,
                    max_trial_repeats=max_trial_repeats,
                    write_once=lambda token: (
                        (lambda path: (
                            write_atompack(
                                path,
                                n_mols,
                                atoms,
                                with_custom,
                                compression=atompack_compression,
                                level=atompack_level,
                                atompack_batch_size=resolved_batch_size,
                                atompack_target_batch_mib=atompack_target_batch_mib,
                            ),
                            path,
                        )[1])(f"{tmp}/atp_{token}.atp")
                    ),
                    size_of=lambda path: Path(path).stat().st_size,
                )
                median_rate = _median(rates)
                print(
                    f"  atompack:     {_fmt(median_rate, size)}"
                    f"  [repeats≈{_median_int(repeat_counts)}]"
                )
                results.append(
                    {
                        "benchmark": "write_throughput",
                        "backend": "atompack",
                        "atoms": atoms,
                        "with_custom": with_custom,
                        "n_mols": n_mols,
                        "n_mols_requested": n_mols,
                        "trials": trials,
                        "codec": lmdb_codec,
                        "mol_s": median_rate,
                        "ci95_mol_s": _ci95(rates),
                        "size_bytes": size,
                        "atompack_batch_size": resolved_batch_size,
                        "warmup_trials": warmup_trials,
                        "min_trial_seconds": min_trial_seconds,
                        "max_trial_repeats": max_trial_repeats,
                        "trial_repeats_median": _median_int(repeat_counts),
                    }
                )

                # columnar baselines
                if hdf5_available():
                    rates, size, repeat_counts = _sample_write_rates(
                        trials=trials,
                        n_units_per_write=n_mols,
                        warmup_trials=warmup_trials,
                        min_trial_seconds=min_trial_seconds,
                        max_trial_repeats=max_trial_repeats,
                        write_once=lambda token: (
                            (lambda path: (
                                write_hdf5_soa(path, n_mols, atoms, with_custom),
                                path,
                            )[1])(f"{tmp}/hs_{token}.h5")
                        ),
                        size_of=_dir_size,
                    )
                    median_rate = _median(rates)
                    print(
                        f"  hdf5_soa:     {_fmt(median_rate, size)}"
                        f"  [repeats≈{_median_int(repeat_counts)}]"
                    )
                    results.append(
                        {
                            "benchmark": "write_throughput",
                            "backend": "hdf5_soa",
                            "atoms": atoms,
                            "with_custom": with_custom,
                            "n_mols": n_mols,
                            "n_mols_requested": n_mols,
                            "trials": trials,
                            "codec": "none",
                            "mol_s": median_rate,
                            "ci95_mol_s": _ci95(rates),
                            "size_bytes": size,
                            "warmup_trials": warmup_trials,
                            "min_trial_seconds": min_trial_seconds,
                            "max_trial_repeats": max_trial_repeats,
                            "trial_repeats_median": _median_int(repeat_counts),
                        }
                    )

                if _lmdb is not None:
                    rates, size, repeat_counts = _sample_write_rates(
                        trials=trials,
                        n_units_per_write=n_mols,
                        warmup_trials=warmup_trials,
                        min_trial_seconds=min_trial_seconds,
                        max_trial_repeats=max_trial_repeats,
                        write_once=lambda token: (
                            (lambda path: (
                                write_lmdb_packed(path, n_mols, atoms, with_custom, codec=lmdb_codec),
                                path,
                            )[1])(f"{tmp}/lp_{token}")
                        ),
                        size_of=_dir_size,
                    )
                    median_rate = _median(rates)
                    print(
                        f"  lmdb_packed:  {_fmt(median_rate, size)}"
                        f"  [repeats≈{_median_int(repeat_counts)}]"
                    )
                    results.append(
                        {
                            "benchmark": "write_throughput",
                            "backend": "lmdb_packed",
                            "atoms": atoms,
                            "with_custom": with_custom,
                            "n_mols": n_mols,
                            "n_mols_requested": n_mols,
                            "trials": trials,
                            "codec": lmdb_codec,
                            "mol_s": median_rate,
                            "ci95_mol_s": _ci95(rates),
                            "size_bytes": size,
                            "warmup_trials": warmup_trials,
                            "min_trial_seconds": min_trial_seconds,
                            "max_trial_repeats": max_trial_repeats,
                            "trial_repeats_median": _median_int(repeat_counts),
                        }
                    )

                # lmdb_pickle
                if _lmdb is not None:
                    rates, size, repeat_counts = _sample_write_rates(
                        trials=trials,
                        n_units_per_write=n_mols,
                        warmup_trials=warmup_trials,
                        min_trial_seconds=min_trial_seconds,
                        max_trial_repeats=max_trial_repeats,
                        write_once=lambda token: (
                            (lambda path: (
                                write_lmdb_pickle(path, n_mols, atoms, with_custom, codec=lmdb_codec),
                                path,
                            )[1])(f"{tmp}/lk_{token}")
                        ),
                        size_of=_dir_size,
                    )
                    median_rate = _median(rates)
                    print(
                        f"  lmdb_pickle:  {_fmt(median_rate, size)}"
                        f"  [repeats≈{_median_int(repeat_counts)}]"
                    )
                    results.append(
                        {
                            "benchmark": "write_throughput",
                            "backend": "lmdb_pickle",
                            "atoms": atoms,
                            "with_custom": with_custom,
                            "n_mols": n_mols,
                            "n_mols_requested": n_mols,
                            "trials": trials,
                            "codec": lmdb_codec,
                            "mol_s": median_rate,
                            "ci95_mol_s": _ci95(rates),
                            "size_bytes": size,
                            "warmup_trials": warmup_trials,
                            "min_trial_seconds": min_trial_seconds,
                            "max_trial_repeats": max_trial_repeats,
                            "trial_repeats_median": _median_int(repeat_counts),
                        }
                    )

                if _has_ase:
                    n_ase = min(n_mols, ASE_WRITE_MAX)
                    for key, ext in [("ase_sqlite", ".db"), ("ase_lmdb", _ASE_LMDB_EXT)]:
                        if ext is None:
                            continue
                        rates, size, repeat_counts = _sample_write_rates(
                            trials=trials,
                            n_units_per_write=n_ase,
                            warmup_trials=warmup_trials,
                            min_trial_seconds=min_trial_seconds,
                            max_trial_repeats=max_trial_repeats,
                            write_once=lambda token: (
                                (lambda path: (
                                    write_ase(path, n_ase, atoms, with_custom),
                                    path,
                                )[1])(f"{tmp}/{key}_{token}{ext}")
                            ),
                            size_of=lambda path: Path(path).stat().st_size if Path(path).is_file() else _dir_size(path),
                        )
                        median_rate = _median(rates)
                        print(
                            f"  {key:<13s}  {_fmt(median_rate, size)}"
                            f"  [N={n_ase:,}, repeats≈{_median_int(repeat_counts)}]"
                        )
                        results.append(
                            {
                                "benchmark": "write_throughput",
                                "backend": key,
                                "atoms": atoms,
                                "with_custom": with_custom,
                                "n_mols": n_ase,
                                "n_mols_requested": n_mols,
                                "trials": trials,
                                "codec": "n/a",
                                "mol_s": median_rate,
                                "ci95_mol_s": _ci95(rates),
                                "size_bytes": size,
                                "warmup_trials": warmup_trials,
                                "min_trial_seconds": min_trial_seconds,
                                "max_trial_repeats": max_trial_repeats,
                                "trial_repeats_median": _median_int(repeat_counts),
                            }
                        )
    return results


# ---------------------------------------------------------------------------
# Benchmark 2: storage footprint across atom counts
# ---------------------------------------------------------------------------


def _normalized_size_bytes(size_bytes: int, n_written: int, n_requested: int) -> int:
    if n_written <= 0 or n_requested <= 0:
        return 0
    return int(round(size_bytes * (n_requested / n_written)))


def bench_storage_footprint(
    atom_counts: list[int],
    n_mols: int = 50_000,
    scratch_dir: str | None = None,
    codec: str = DEFAULT_WRITE_CODEC,
    atompack_batch_size: int | None = None,
    atompack_target_batch_mib: float = DEFAULT_ATOMPACK_TARGET_BATCH_MIB,
) -> list[dict[str, Any]]:
    atompack_compression, atompack_level, lmdb_codec = _parse_codec_spec(codec)

    print("=" * 60)
    print("Benchmark 2: Storage footprint across atom counts")
    print(f"  {n_mols:,} requested molecules, codec={lmdb_codec}")
    print("=" * 60)

    results: list[dict[str, Any]] = []
    for atoms in atom_counts:
        for with_custom in (False, True):
            tag = " + custom" if with_custom else ""
            resolved_batch_size = _resolve_atompack_batch_size(
                atoms,
                with_custom=with_custom,
                atompack_batch_size=atompack_batch_size,
                atompack_target_batch_mib=atompack_target_batch_mib,
            )
            print(f"\n--- {atoms} atoms{tag} ---")
            scenario_rows: list[dict[str, Any]] = []

            with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
                path = f"{tmp}/atp.atp"
                write_atompack(
                    path,
                    n_mols,
                    atoms,
                    with_custom,
                    compression=atompack_compression,
                    level=atompack_level,
                    atompack_batch_size=resolved_batch_size,
                    atompack_target_batch_mib=atompack_target_batch_mib,
                )
                size = Path(path).stat().st_size
                scenario_rows.append(
                    {
                        "benchmark": "write_storage",
                        "backend": "atompack",
                        "atoms": atoms,
                        "with_custom": with_custom,
                        "n_mols": n_mols,
                        "n_mols_requested": n_mols,
                        "trials": 1,
                        "codec": lmdb_codec,
                        "mol_s": 0.0,
                        "ci95_mol_s": 0.0,
                        "size_bytes": size,
                        "normalized_size_bytes": size,
                        "bytes_per_mol": size / n_mols,
                        "bytes_per_atom": size / (n_mols * atoms),
                        "atompack_batch_size": resolved_batch_size,
                    }
                )

                if hdf5_available():
                    path = f"{tmp}/hs.h5"
                    write_hdf5_soa(path, n_mols, atoms, with_custom)
                    size = _dir_size(path)
                    scenario_rows.append(
                        {
                            "benchmark": "write_storage",
                            "backend": "hdf5_soa",
                            "atoms": atoms,
                            "with_custom": with_custom,
                            "n_mols": n_mols,
                            "n_mols_requested": n_mols,
                            "trials": 1,
                            "codec": "none",
                            "mol_s": 0.0,
                            "ci95_mol_s": 0.0,
                            "size_bytes": size,
                            "normalized_size_bytes": size,
                            "bytes_per_mol": size / n_mols,
                            "bytes_per_atom": size / (n_mols * atoms),
                        }
                    )

                if _lmdb is not None:
                    path = f"{tmp}/lp"
                    write_lmdb_packed(path, n_mols, atoms, with_custom, codec=lmdb_codec)
                    size = _dir_size(path)
                    scenario_rows.append(
                        {
                            "benchmark": "write_storage",
                            "backend": "lmdb_packed",
                            "atoms": atoms,
                            "with_custom": with_custom,
                            "n_mols": n_mols,
                            "n_mols_requested": n_mols,
                            "trials": 1,
                            "codec": lmdb_codec,
                            "mol_s": 0.0,
                            "ci95_mol_s": 0.0,
                            "size_bytes": size,
                            "normalized_size_bytes": size,
                            "bytes_per_mol": size / n_mols,
                            "bytes_per_atom": size / (n_mols * atoms),
                        }
                    )

                    path = f"{tmp}/lk"
                    write_lmdb_pickle(path, n_mols, atoms, with_custom, codec=lmdb_codec)
                    size = _dir_size(path)
                    scenario_rows.append(
                        {
                            "benchmark": "write_storage",
                            "backend": "lmdb_pickle",
                            "atoms": atoms,
                            "with_custom": with_custom,
                            "n_mols": n_mols,
                            "n_mols_requested": n_mols,
                            "trials": 1,
                            "codec": lmdb_codec,
                            "mol_s": 0.0,
                            "ci95_mol_s": 0.0,
                            "size_bytes": size,
                            "normalized_size_bytes": size,
                            "bytes_per_mol": size / n_mols,
                            "bytes_per_atom": size / (n_mols * atoms),
                        }
                    )

                if _has_ase:
                    n_ase = min(n_mols, ASE_WRITE_MAX)
                    for key, ext in [("ase_sqlite", ".db"), ("ase_lmdb", _ASE_LMDB_EXT)]:
                        if ext is None:
                            continue
                        path = f"{tmp}/{key}{ext}"
                        write_ase(path, n_ase, atoms, with_custom)
                        size = Path(path).stat().st_size if Path(path).is_file() else _dir_size(path)
                        normalized_size = _normalized_size_bytes(size, n_ase, n_mols)
                        scenario_rows.append(
                            {
                                "benchmark": "write_storage",
                                "backend": key,
                                "atoms": atoms,
                                "with_custom": with_custom,
                                "n_mols": n_ase,
                                "n_mols_requested": n_mols,
                                "trials": 1,
                                "codec": "n/a",
                                "mol_s": 0.0,
                                "ci95_mol_s": 0.0,
                                "size_bytes": size,
                                "normalized_size_bytes": normalized_size,
                                "bytes_per_mol": normalized_size / n_mols,
                                "bytes_per_atom": normalized_size / (n_mols * atoms),
                            }
                        )

            atompack_norm = next(
                row["normalized_size_bytes"] for row in scenario_rows if row["backend"] == "atompack"
            )
            for row in scenario_rows:
                row["size_ratio_vs_atompack"] = row["normalized_size_bytes"] / atompack_norm
                results.append(row)

            print(
                f"  {'backend':<13} {'dataset':>10} {'B/mol':>10} {'ratio':>8}",
                flush=True,
            )
            for row in sorted(
                scenario_rows,
                key=lambda item: (item["normalized_size_bytes"], item["backend"]),
            ):
                note = ""
                if row["n_mols"] != row["n_mols_requested"]:
                    note = f"  [scaled from {row['n_mols']:,}]"
                print(
                    f"  {row['backend']:<13} "
                    f"{row['normalized_size_bytes'] / 1e6:>8.1f} MB "
                    f"{row['bytes_per_mol']:>10.1f} "
                    f"{row['size_ratio_vs_atompack']:>8.2f}x{note}",
                    flush=True,
                )
    return results


# ---------------------------------------------------------------------------
# Benchmark 3: Atompack builtins batch-size sweep
# ---------------------------------------------------------------------------


def bench_atompack_batch_scaling(
    atom_counts: list[int],
    n_mols: int = 50_000,
    batch_sizes: list[int] | None = None,
    trials: int = 3,
    scratch_dir: str | None = None,
    codec: str = DEFAULT_WRITE_CODEC,
    atompack_target_batch_mib: float = DEFAULT_ATOMPACK_TARGET_BATCH_MIB,
    warmup_trials: int = DEFAULT_WARMUP_TRIALS,
    min_trial_seconds: float = DEFAULT_MIN_TRIAL_SECONDS,
    max_trial_repeats: int = DEFAULT_MAX_TRIAL_REPEATS,
) -> list[dict[str, Any]]:
    atompack_compression, atompack_level, lmdb_codec = _parse_codec_spec(codec)
    print("=" * 60)
    print("Benchmark 3: Atompack builtins batch-size sweep")
    print(
        f"  {n_mols:,} molecules, {trials} trials, codec={lmdb_codec}, "
        f"warmup={warmup_trials}, min_trial_seconds={min_trial_seconds:g}, "
        f"max_repeats={max_trial_repeats}"
    )
    print("=" * 60)

    if batch_sizes is None:
        batch_sizes = list(DEFAULT_BATCH_SWEEP_SIZES)

    results: list[dict[str, Any]] = []
    for atoms in atom_counts:
        recommended = _resolve_atompack_batch_size(
            atoms,
            with_custom=False,
            atompack_batch_size=None,
            atompack_target_batch_mib=atompack_target_batch_mib,
        )
        sweep_sizes = sorted(set([recommended, *batch_sizes]))
        print(f"\n--- {atoms} atoms (auto={recommended:,}) ---")
        with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
            for batch_size in sweep_sizes:
                rates, size, repeat_counts = _sample_write_rates(
                    trials=trials,
                    n_units_per_write=n_mols,
                    warmup_trials=warmup_trials,
                    min_trial_seconds=min_trial_seconds,
                    max_trial_repeats=max_trial_repeats,
                    write_once=lambda token, current_batch_size=batch_size: (
                        (lambda path: (
                            write_atompack(
                                path,
                                n_mols,
                                atoms,
                                compression=atompack_compression,
                                level=atompack_level,
                                atompack_batch_size=current_batch_size,
                                atompack_target_batch_mib=atompack_target_batch_mib,
                            ),
                            path,
                        )[1])(f"{tmp}/atp_bs{current_batch_size}_{token}.atp")
                    ),
                    size_of=lambda path: Path(path).stat().st_size,
                )
                median_rate = _median(rates)
                marker = " [auto]" if batch_size == recommended else ""
                print(
                    f"  batch={batch_size:>6,}  {_fmt(median_rate, size)}"
                    f"  ({median_rate * atoms:,.0f} atoms/s,"
                    f" repeats≈{_median_int(repeat_counts)}){marker}"
                )
                results.append(
                    {
                        "benchmark": "write_batch_scaling",
                        "backend": "atompack",
                        "atoms": atoms,
                        "with_custom": False,
                        "n_mols": n_mols,
                        "n_mols_requested": n_mols,
                        "trials": trials,
                        "codec": lmdb_codec,
                        "mol_s": median_rate,
                        "ci95_mol_s": _ci95(rates),
                        "size_bytes": size,
                        "batch_size": batch_size,
                        "batch_payload_bytes": (
                            batch_size
                            * _estimate_atompack_payload_bytes_per_molecule(
                                atoms,
                                with_custom=False,
                            )
                        ),
                        "is_auto_batch_size": batch_size == recommended,
                        "warmup_trials": warmup_trials,
                        "min_trial_seconds": min_trial_seconds,
                        "max_trial_repeats": max_trial_repeats,
                        "trial_repeats_median": _median_int(repeat_counts),
                    }
                )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    n_cores = os.cpu_count() or 1
    rayon = os.environ.get("RAYON_NUM_THREADS", f"all ({n_cores} cores)")

    parser = argparse.ArgumentParser(description="Write benchmarks")
    parser.add_argument(
        "--bench", nargs="*", type=int, default=[1, 2],
        help="Which benchmarks to run: 1=throughput, 2=storage footprint, 3=atompack batch sweep (default: 1 2)",
    )
    # Benchmark 1 options
    parser.add_argument("--atoms", nargs="*", type=int, default=[16, 64, 256])
    parser.add_argument("--n-mols", type=int, default=50_000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--warmup-trials",
        type=int,
        default=DEFAULT_WARMUP_TRIALS,
        help=(
            "Unreported warmup writes per backend before measuring "
            f"(default: {DEFAULT_WARMUP_TRIALS})."
        ),
    )
    parser.add_argument(
        "--min-trial-seconds",
        type=float,
        default=DEFAULT_MIN_TRIAL_SECONDS,
        help=(
            "Minimum wall-clock seconds to accumulate within each measured trial "
            f"before reporting throughput (default: {DEFAULT_MIN_TRIAL_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--max-trial-repeats",
        type=int,
        default=DEFAULT_MAX_TRIAL_REPEATS,
        help=(
            "Maximum repeated writes allowed to reach --min-trial-seconds "
            f"(default: {DEFAULT_MAX_TRIAL_REPEATS})."
        ),
    )
    parser.add_argument(
        "--atompack-batch-size",
        type=int,
        default=None,
        help="Override Atompack write batch size. Default: auto-sized from atoms and target MiB.",
    )
    parser.add_argument(
        "--atompack-target-batch-mib",
        type=float,
        default=DEFAULT_ATOMPACK_TARGET_BATCH_MIB,
        help=(
            "Target raw payload size per Atompack write batch when auto-sizing "
            f"(default: {DEFAULT_ATOMPACK_TARGET_BATCH_MIB:g} MiB)."
        ),
    )
    parser.add_argument(
        "--codec",
        type=str,
        default=DEFAULT_WRITE_CODEC,
        help="Write codec to benchmark. Default: none. Examples: none, lz4, zstd:3",
    )
    parser.add_argument(
        "--batch-scale-atoms",
        nargs="*",
        type=int,
        default=[64, 256],
        help="Atom counts for benchmark 3 (default: 64 256)",
    )
    parser.add_argument(
        "--batch-scale-sizes",
        nargs="*",
        type=int,
        default=None,
        help=f"Atompack batch sizes for benchmark 3 (default: {DEFAULT_BATCH_SWEEP_SIZES})",
    )
    parser.add_argument(
        "--scratch-dir", type=str, default=DEFAULT_SCRATCH_DIR,
        help=f"Directory for temporary datasets (default: {DEFAULT_SCRATCH_DIR}).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional path for structured JSON results.",
    )
    args = parser.parse_args(argv)
    if args.scratch_dir:
        Path(args.scratch_dir).mkdir(parents=True, exist_ok=True)

    print(f"Rayon threads: {rayon}\n")

    run = set(args.bench)
    all_results: list[dict[str, Any]] = []
    if 1 in run:
        all_results.extend(
            bench_throughput(
                args.atoms,
                args.n_mols,
                args.trials,
                args.scratch_dir,
                codec=args.codec,
                atompack_batch_size=args.atompack_batch_size,
                atompack_target_batch_mib=args.atompack_target_batch_mib,
                warmup_trials=args.warmup_trials,
                min_trial_seconds=args.min_trial_seconds,
                max_trial_repeats=args.max_trial_repeats,
            )
        )
        print()
    if 2 in run:
        all_results.extend(
            bench_storage_footprint(
                args.atoms,
                args.n_mols,
                args.scratch_dir,
                codec=args.codec,
                atompack_batch_size=args.atompack_batch_size,
                atompack_target_batch_mib=args.atompack_target_batch_mib,
            )
        )
        print()
    if 3 in run:
        all_results.extend(
            bench_atompack_batch_scaling(
                args.batch_scale_atoms,
                args.n_mols,
                args.batch_scale_sizes,
                args.trials,
                args.scratch_dir,
                codec=args.codec,
                atompack_target_batch_mib=args.atompack_target_batch_mib,
                warmup_trials=args.warmup_trials,
                min_trial_seconds=args.min_trial_seconds,
                max_trial_repeats=args.max_trial_repeats,
            )
        )
    if args.out is not None:
        _write_results_json(args.out, all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
