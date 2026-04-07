# Copyright 2026 Entalpic
"""Scaling benchmark: batch random-read throughput vs molecule size and thread count.

Writes synthetic datasets for each (atoms_per_molecule, backend) pair, warms the
page cache, then measures batch random reads at varying thread counts.

**Important**: atompack's rayon thread pool is initialized once per process, so each
(atoms_per_molecule, n_workers) combination for atompack is benchmarked in an isolated
subprocess to ensure RAYON_NUM_THREADS takes effect.

Run with:
    uv run --active python benchmarks/scaling_benchmark.py
    uv run --active python benchmarks/scaling_benchmark.py --atoms 16 64 256 --threads 1 4 8 16
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import atompack
import numpy as np
from tqdm import tqdm

from atom_lmdb import AtomLMDB, AtomLMDBConfig, lmdb_codec_fns, pack_record, unpack_arrays
from atom_lmdb_pickle import PickleAtomLMDB, PickleLMDBConfig

try:
    import ase.db
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    _has_ase = True
except ImportError:
    _has_ase = False

DEFAULT_ATOMS = [16, 64, 128, 256, 512]
DEFAULT_THREADS = [1, 2, 4, 8, 16, 24]
BATCH_SIZE = 100_000
NUM_TRIALS = 3
TEMPLATE_COUNT = 1024
DEFAULT_POPULATE = False
MIN_DATASET_MOLECULES = 200_000  # ensure batch is always a subset
TARGET_DATASET_BYTES = 200 * 1024 * 1024  # ~200 MB per dataset
ASE_MAX_MOLECULES = 50_000
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


def _default_num_molecules(atoms_per_mol: int) -> int:
    # positions(f32*3) + Z(u8) + forces(f32*3) + charges(f64) + energy(f64) + cell(72) + stress(72) + overhead
    bytes_per_mol = atoms_per_mol * (3 * 4 + 1 + 3 * 4 + 8) + 8 + 72 + 72 + 100
    return max(MIN_DATASET_MOLECULES, TARGET_DATASET_BYTES // bytes_per_mol)


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return 1.96 * float(np.std(arr, ddof=1)) / float(np.sqrt(arr.size))


def _collect_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    # Try to get CPU model name on Linux
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    # Total RAM
    try:
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                info["ram_gib"] = round(kib / 1024 / 1024, 1)
                break
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Dataset writers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Templates:
    molecules: list[atompack.Molecule]
    positions: np.ndarray
    atomic_numbers: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    energy: np.ndarray
    forces: np.ndarray
    stress: np.ndarray
    charges: np.ndarray


def _build_templates(count: int, atoms_per_mol: int, seed: int = 1234) -> Templates:
    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(count, atoms_per_mol, 3)).astype(np.float32)
    atomic_numbers = rng.integers(1, 16, size=(count, atoms_per_mol), dtype=np.uint8)
    forces = rng.normal(size=(count, atoms_per_mol, 3)).astype(np.float32)
    energy = rng.normal(loc=-50.0, scale=10.0, size=(count,)).astype(np.float64)
    cell = np.tile(np.eye(3, dtype=np.float64)[None, :, :], (count, 1, 1))
    pbc = np.zeros((count, 3), dtype=np.uint8)
    stress = rng.normal(size=(count, 3, 3)).astype(np.float64)
    charges = rng.normal(size=(count, atoms_per_mol)).astype(np.float64)

    molecules = []
    for i in range(count):
        mol = atompack.Molecule.from_arrays(positions[i], atomic_numbers[i])
        mol.energy = float(energy[i])
        mol.forces = forces[i]
        mol.cell = cell[i]
        mol.pbc = tuple(bool(b) for b in pbc[i])
        mol.stress = stress[i]
        mol.charges = charges[i]
        molecules.append(mol)

    return Templates(
        molecules=molecules,
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=pbc,
        energy=energy,
        forces=forces,
        stress=stress,
        charges=charges,
    )


def _write_atompack(path: Path, molecules: list[atompack.Molecule], num: int) -> None:
    db = atompack.Database(str(path), compression="zstd", level=3, overwrite=True)
    batch: list[atompack.Molecule] = []
    for i in tqdm(range(num), desc="  write atompack", unit="mol", leave=False):
        batch.append(molecules[i % len(molecules)])
        if len(batch) >= 512:
            db.add_molecules(batch)
            batch = []
    if batch:
        db.add_molecules(batch)
    db.flush()


def _write_packed_lmdb(
    path: Path,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    energy: np.ndarray,
    forces: np.ndarray,
    stress: np.ndarray,
    charges: np.ndarray,
    num: int,
) -> None:
    import lmdb  # type: ignore

    n_templates = positions.shape[0]
    atoms_per_mol = positions.shape[1]
    map_size = AtomLMDB.estimate_map_size_bytes(
        num_molecules=num,
        atoms_per_molecule=atoms_per_mol,
        with_props=True,
        with_cell_pbc=True,
        with_stress=True,
        with_charges=True,
        map_factor=4.0,
        map_slack_mib=256,
    )
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(path),
        map_size=map_size,
        subdir=True,
        readonly=False,
        lock=False,
        readahead=True,
        max_dbs=1,
        sync=True,
        metasync=True,
        meminit=False,
    )
    compress, _ = lmdb_codec_fns("zstd:3")
    txn = env.begin(write=True)
    for i in tqdm(range(num), desc="  write lmdb_packed", unit="mol", leave=False):
        t = i % n_templates
        value = pack_record(
            positions[t],
            atomic_numbers[t],
            cell[t],
            pbc[t],
            float(energy[t]),
            forces[t],
            stress[t],
            charges[t],
        )
        if compress is not None:
            value = compress(value)
        txn.put(struct.pack(">Q", i), value)
        if (i + 1) % 5000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()


def _write_pickle_lmdb(
    path: Path,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    energy: np.ndarray,
    forces: np.ndarray,
    stress: np.ndarray,
    charges: np.ndarray,
    num: int,
) -> None:
    n_templates = positions.shape[0]
    atoms_per_mol = positions.shape[1]
    store = PickleAtomLMDB(path, PickleLMDBConfig(codec="zstd:3"))
    store.reset_dir()
    map_size = PickleAtomLMDB.estimate_map_size_bytes(
        num_molecules=num,
        atoms_per_molecule=atoms_per_mol,
        with_props=True,
        with_cell_pbc=True,
        map_factor=4.0,
        map_slack_mib=256,
    )
    env = store.open_env(map_size=map_size, readonly=False, lock=False)
    txn = env.begin(write=True)
    for i in tqdm(range(num), desc="  write lmdb_pickle", unit="mol", leave=False):
        t = i % n_templates
        store.put_molecule(
            txn,
            idx=i,
            positions=positions[t],
            atomic_numbers=atomic_numbers[t],
            cell=cell[t],
            pbc=pbc[t],
            energy=float(energy[t]),
            forces=forces[t],
            stress=stress[t],
            charges=charges[t],
        )
        if (i + 1) % 5000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()


def _write_ase_db(
    path: Path,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    energy: np.ndarray,
    forces: np.ndarray,
    num: int,
) -> None:
    if not _has_ase:
        raise RuntimeError("ASE is not available")
    n_templates = positions.shape[0]
    db = ase.db.connect(str(path))
    try:
        # Reuse one ASE transaction for the full write loop.  This removes the
        # per-row begin/commit overhead in ASE sqlite and ASE LMDB backends.
        with db:
            for i in tqdm(range(num), desc=f"  write {path.suffix or 'ase'}", unit="mol", leave=False):
                t = i % n_templates
                atoms = Atoms(
                    numbers=atomic_numbers[t],
                    positions=positions[t],
                    cell=cell[t],
                    pbc=[bool(x) for x in pbc[t]],
                )
                atoms.calc = SinglePointCalculator(
                    atoms,
                    energy=float(energy[t]),
                    forces=forces[t],
                )
                db.write(atoms)
    finally:
        close = getattr(db, "disconnect", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _warmup(db: Any, num: int) -> None:
    """Sequential read to populate page cache."""
    for i in tqdm(range(num), desc="  warmup", unit="mol", leave=False):
        db.get_molecule(i)


def _bench_atompack_subprocess(
    path: str,
    indices_file: str,
    n_workers: int,
    num_trials: int,
) -> list[float]:
    """Run atompack bench in a subprocess so RAYON_NUM_THREADS is fresh."""
    script = f"""
import json, os, time, atompack
os.environ["RAYON_NUM_THREADS"] = "{n_workers}"
db = atompack.Database.open("{path}", mmap=True, populate={DEFAULT_POPULATE!r})
indices = json.loads(open("{indices_file}").read())
# Warmup: init rayon pool (mmap pages pre-faulted by PopulateRead)
db.get_molecules(indices[:200])
trials = []
for _ in range({num_trials}):
    t0 = time.perf_counter()
    db.get_molecules(indices)
    dt = time.perf_counter() - t0
    trials.append(len(indices) / dt)
print(json.dumps(trials))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Atompack subprocess failed:\n{result.stderr}")
    return json.loads(result.stdout.strip())


def _lmdb_packed_worker(args: tuple[str, str, list[int]]) -> int:
    """Multiprocessing worker for LMDB packed reads."""
    path, codec, chunk = args
    import lmdb as _lmdb  # type: ignore

    _, decompress = lmdb_codec_fns(codec)
    env = _lmdb.open(path, subdir=True, readonly=True, lock=False, readahead=False, max_dbs=1)
    txn = env.begin(write=False)
    try:
        for idx in chunk:
            value = txn.get(struct.pack(">Q", idx))
            if value is not None:
                if decompress is not None:
                    value = decompress(bytes(value))
                unpack_arrays(value)
        return len(chunk)
    finally:
        txn.abort()
        env.close()


def _lmdb_pickle_worker(args: tuple[str, str, list[int]]) -> int:
    """Multiprocessing worker for LMDB pickle reads."""
    path, codec, chunk = args
    local_store = PickleAtomLMDB(
        Path(path),
        PickleLMDBConfig(codec=codec, readahead=False),
    )
    env = local_store.open_env(map_size=1 << 30, readonly=True, lock=False)
    txn = env.begin(write=False)
    try:
        for idx in chunk:
            local_store.get_payload(txn, idx)
        return len(chunk)
    finally:
        txn.abort()
        env.close()


def _bench_packed_lmdb_batch(
    path: Path,
    codec: str,
    indices: list[int],
    n_workers: int,
    num_trials: int,
) -> list[float]:
    import multiprocessing as mp

    chunk_size = max(1, len(indices) // n_workers)
    chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]
    work = [(str(path), codec, chunk) for chunk in chunks]

    trials = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        with mp.Pool(processes=n_workers) as pool:
            sum(pool.map(_lmdb_packed_worker, work))
        dt = time.perf_counter() - t0
        trials.append(len(indices) / dt)
    return trials


def _bench_pickle_lmdb_batch(
    path: Path,
    codec: str,
    indices: list[int],
    n_workers: int,
    num_trials: int,
) -> list[float]:
    import multiprocessing as mp

    chunk_size = max(1, len(indices) // n_workers)
    chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]
    work = [(str(path), codec, chunk) for chunk in chunks]

    trials = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        with mp.Pool(processes=n_workers) as pool:
            sum(pool.map(_lmdb_pickle_worker, work))
        dt = time.perf_counter() - t0
        trials.append(len(indices) / dt)
    return trials


def _ase_worker(args: tuple[str, list[int]]) -> int:
    path, chunk = args
    db = ase.db.connect(path)
    try:
        for idx in chunk:
            row = db.get(id=idx + 1)
            row.toatoms()
        return len(chunk)
    finally:
        close = getattr(db, "disconnect", None)
        if callable(close):
            close()


def _bench_ase_batch(
    path: Path,
    indices: list[int],
    n_workers: int,
    num_trials: int,
) -> list[float]:
    import multiprocessing as mp

    chunk_size = max(1, len(indices) // n_workers)
    chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]
    work = [(str(path), chunk) for chunk in chunks]

    trials = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        with mp.Pool(processes=n_workers) as pool:
            sum(pool.map(_ase_worker, work))
        dt = time.perf_counter() - t0
        trials.append(len(indices) / dt)
    return trials


# ---------------------------------------------------------------------------
# File size collection
# ---------------------------------------------------------------------------


def _file_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_scaling(
    atoms_sizes: list[int],
    thread_counts: list[int],
    num_molecules: int | None,
    batch_size: int,
    num_trials: int,
    scratch_dir: Path,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)

    for atoms_per_mol in atoms_sizes:
        num = num_molecules or _default_num_molecules(atoms_per_mol)
        batch_size_actual = min(batch_size, num)
        indices = rng.integers(0, num, size=batch_size_actual).tolist()

        print(f"\n{'=' * 60}")
        print(
            f"atoms_per_molecule={atoms_per_mol}  num_molecules={num:,}  batch={batch_size_actual:,}"
        )
        print(f"{'=' * 60}")

        templates = _build_templates(TEMPLATE_COUNT, atoms_per_mol)

        tmp = Path(tempfile.mkdtemp(prefix=f"scaling_{atoms_per_mol}_", dir=str(scratch_dir)))
        try:
            # Write datasets
            atp_path = tmp / "data.atp"
            packed_path = tmp / "packed.lmdb"
            pickle_path = tmp / "pickle.lmdb"
            ase_sqlite_path = tmp / "ase.db"
            ase_lmdb_path = tmp / f"ase{_ASE_LMDB_EXT}" if _ASE_LMDB_EXT else None

            _write_atompack(atp_path, templates.molecules, num)
            _write_packed_lmdb(
                packed_path,
                templates.positions,
                templates.atomic_numbers,
                templates.cell,
                templates.pbc,
                templates.energy,
                templates.forces,
                templates.stress,
                templates.charges,
                num,
            )
            _write_pickle_lmdb(
                pickle_path,
                templates.positions,
                templates.atomic_numbers,
                templates.cell,
                templates.pbc,
                templates.energy,
                templates.forces,
                templates.stress,
                templates.charges,
                num,
            )
            ase_num = min(num, ASE_MAX_MOLECULES) if _has_ase else 0
            if _has_ase and ase_num > 0:
                _write_ase_db(
                    ase_sqlite_path,
                    templates.positions,
                    templates.atomic_numbers,
                    templates.cell,
                    templates.pbc,
                    templates.energy,
                    templates.forces,
                    ase_num,
                )
                if ase_lmdb_path is not None:
                    _write_ase_db(
                        ase_lmdb_path,
                        templates.positions,
                        templates.atomic_numbers,
                        templates.cell,
                        templates.pbc,
                        templates.energy,
                        templates.forces,
                        ase_num,
                    )

            # Collect file sizes
            sizes = {
                "atompack_bytes": _file_size_bytes(atp_path),
                "lmdb_packed_bytes": _file_size_bytes(packed_path),
                "lmdb_pickle_bytes": _file_size_bytes(pickle_path),
                "ase_sqlite_bytes": _file_size_bytes(ase_sqlite_path) if _has_ase else 0,
                "ase_lmdb_bytes": _file_size_bytes(ase_lmdb_path) if ase_lmdb_path is not None else 0,
            }
            print(
                f"  file sizes: atompack={sizes['atompack_bytes'] / 1e6:.1f}MB  "
                f"lmdb_packed={sizes['lmdb_packed_bytes'] / 1e6:.1f}MB  "
                f"lmdb_pickle={sizes['lmdb_pickle_bytes'] / 1e6:.1f}MB"
                + (
                    f"  ase_sqlite={sizes['ase_sqlite_bytes'] / 1e6:.1f}MB"
                    + (
                        f"  ase_lmdb={sizes['ase_lmdb_bytes'] / 1e6:.1f}MB"
                        if ase_lmdb_path is not None
                        else ""
                    )
                    if _has_ase and ase_num > 0
                    else ""
                )
            )

            # Warmup all datasets (populates OS page cache)
            print("  warming caches...")
            db = atompack.Database.open(str(atp_path), mmap=True, populate=DEFAULT_POPULATE)
            _warmup(db, num)
            del db
            if _has_ase and ase_num > 0:
                for path in [ase_sqlite_path, ase_lmdb_path]:
                    if path is None:
                        continue
                    db = ase.db.connect(str(path))
                    for i in tqdm(range(min(1_000, ase_num)), desc=f"  warmup {path.name}", unit="mol", leave=False):
                        db.get(id=i + 1).toatoms()
                    close = getattr(db, "disconnect", None)
                    if callable(close):
                        close()

            # Write indices to a temp file for subprocess calls
            indices_file = tmp / "indices.json"
            indices_file.write_text(json.dumps(indices))

            # Bench each backend at each thread count
            for n_workers in thread_counts:
                desc = f"  atompack  threads={n_workers:>2d}"
                trials = _bench_atompack_subprocess(
                    str(atp_path), str(indices_file), n_workers, num_trials
                )
                mean = fmean(trials)
                atoms_s = mean * atoms_per_mol
                tqdm.write(f"{desc}  {mean:>12,.0f} mol/s  {atoms_s:>14,.0f} atoms/s")
                results.append(
                    {
                        "atoms_per_molecule": atoms_per_mol,
                        "num_molecules": num,
                        "backend": "atompack",
                        "n_workers": n_workers,
                        "mean_mol_s": mean,
                        "mean_atoms_s": atoms_s,
                        "trials_mol_s": trials,
                        "ci95": _ci95(trials),
                        **sizes,
                    }
                )

            if _has_ase and ase_num > 0:
                ase_indices = [i % ase_num for i in indices]
                for backend, path in [("ase_sqlite", ase_sqlite_path), ("ase_lmdb", ase_lmdb_path)]:
                    if path is None:
                        continue
                    for n_workers in thread_counts:
                        desc = f"  {backend:<10s} threads={n_workers:>2d}"
                        trials = _bench_ase_batch(path, ase_indices, n_workers, num_trials)
                        mean = fmean(trials)
                        atoms_s = mean * atoms_per_mol
                        tqdm.write(f"{desc}  {mean:>12,.0f} mol/s  {atoms_s:>14,.0f} atoms/s")
                        results.append(
                            {
                                "atoms_per_molecule": atoms_per_mol,
                                "num_molecules": ase_num,
                                "backend": backend,
                                "n_workers": n_workers,
                                "mean_mol_s": mean,
                                "mean_atoms_s": atoms_s,
                                "trials_mol_s": trials,
                                "ci95": _ci95(trials),
                                **sizes,
                            }
                        )

            for n_workers in thread_counts:
                desc = f"  lmdb_packed  threads={n_workers:>2d}"
                trials = _bench_packed_lmdb_batch(
                    packed_path, "zstd:3", indices, n_workers, num_trials
                )
                mean = fmean(trials)
                atoms_s = mean * atoms_per_mol
                tqdm.write(f"{desc}  {mean:>12,.0f} mol/s  {atoms_s:>14,.0f} atoms/s")
                results.append(
                    {
                        "atoms_per_molecule": atoms_per_mol,
                        "num_molecules": num,
                        "backend": "lmdb_packed",
                        "n_workers": n_workers,
                        "mean_mol_s": mean,
                        "mean_atoms_s": atoms_s,
                        "trials_mol_s": trials,
                        "ci95": _ci95(trials),
                        **sizes,
                    }
                )

            for n_workers in thread_counts:
                desc = f"  lmdb_pickle  threads={n_workers:>2d}"
                trials = _bench_pickle_lmdb_batch(
                    pickle_path, "zstd:3", indices, n_workers, num_trials
                )
                mean = fmean(trials)
                atoms_s = mean * atoms_per_mol
                tqdm.write(f"{desc}  {mean:>12,.0f} mol/s  {atoms_s:>14,.0f} atoms/s")
                results.append(
                    {
                        "atoms_per_molecule": atoms_per_mol,
                        "num_molecules": num,
                        "backend": "lmdb_pickle",
                        "n_workers": n_workers,
                        "mean_mol_s": mean,
                        "mean_atoms_s": atoms_s,
                        "trials_mol_s": trials,
                        "ci95": _ci95(trials),
                        **sizes,
                    }
                )

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_size": batch_size,
            "num_trials": num_trials,
            "thread_counts": thread_counts,
            "atoms_sizes": atoms_sizes,
            "system": _collect_system_info(),
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scaling benchmark: batch reads vs molecule size and threads."
    )
    parser.add_argument("--atoms", type=int, nargs="+", default=DEFAULT_ATOMS)
    parser.add_argument("--threads", type=int, nargs="+", default=DEFAULT_THREADS)
    parser.add_argument("--num-molecules", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--out", type=Path, default=Path(".benchmarks/scaling_benchmark.json"))
    args = parser.parse_args(argv)

    scratch = args.out.parent / ".scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    result = run_scaling(
        args.atoms, args.threads, args.num_molecules, args.batch_size, args.num_trials, scratch
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
