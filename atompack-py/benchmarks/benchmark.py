# Copyright 2026 Entalpic
"""
Atompack benchmark.

Compares atompack against several Python-side storage baselines across the
three benchmark dimensions used in the blog post:

  - HDF5 SOA       (chunked field datasets via h5py)
  - LMDB packed     (per-record packed binary blobs)
  - LMDB pickle     (pickled dict-style records)
  - ASE sqlite/lmdb (reference baselines)
  1. Sequential read throughput     (format efficiency, best case)
  2. Compression impact             (decompression overhead, random access)
  3. Multiprocessing scaling        (shuffled __getitem__ scenario)

Datasets are generated once into a persistent scratch directory and reused
across runs. Read benchmarks sample 1% of the dataset per trial (min 10K
molecules).

Usage:
  python benchmark.py                        # run all benchmarks
  python benchmark.py --only 1 2             # run selected benchmarks
  python benchmark.py --atoms 12 64 256      # override atom counts
  python benchmark.py --n-mols 100000        # override molecule count
  python benchmark.py --scratch-dir /tmp/bm  # override scratch directory
  python benchmark.py --out results.json     # save results to JSON

Important:
  Build the atompack Python extension in release mode before trusting any
  throughput numbers (debug mode is much slower):
    make py-dev-release

ASE-backed baselines are intentionally sampled more lightly than Atompack/LMDB
by default because they are much slower and otherwise dominate suite runtime.
For benchmark 3, the random single-item access path is pathological for
`hdf5_soa` and ASE at 256 atoms, so those baselines are skipped by
default for that atom count.
The previous broader experimental suite is archived in
`benchmarks/archive/full_benchmark_suite.py`.
"""
from __future__ import annotations

import argparse
import atexit
import gc
import json
import multiprocessing as mp
import os
import pickle
import shutil
import struct
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Rayon initializes its thread pool once per process.  Set this before importing
# atompack so that every benchmark runs single-threaded (matching LMDB).
# Multiprocessing workers set it independently in their init functions.
# os.environ["RAYON_NUM_THREADS"] = "1"

import atompack

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atom_lmdb import (
    AtomLMDB,
    AtomLMDBConfig,
    env_used_bytes,
    lmdb_codec_fns,
    pack_record,
    unpack_arrays,
    unpack_custom_fields,
)
from atom_hdf5_soa import AtomHdf5Soa, AtomHdf5SoaConfig, hdf5_available
from atom_lmdb_pickle import PickleAtomLMDB, PickleLMDBConfig

try:
    import lmdb as _lmdb
except ImportError:
    _lmdb = None

try:
    import torch
except ImportError:
    torch = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

try:
    import ase.db
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    _has_ase = True
except ImportError:
    _has_ase = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKENDS = ("atompack", "hdf5_soa", "lmdb_packed", "lmdb_pickle")
ASE_BACKENDS = ("ase_sqlite", "ase_lmdb")
ALL_BACKENDS = BACKENDS + ASE_BACKENDS

ASE_MAX_MOLS = 50_000
ASE_READ_SAMPLE_DEFAULT = 10_000
# Detected ASE LMDB extension (.aselmdb if available)
_ASE_LMDB_EXT: str | None = None
if _has_ase:
    for _ext in (".aselmdb", ".lmdb"):
        try:
            import tempfile as _tf
            _p = _tf.mktemp(suffix=_ext)
            ase.db.connect(_p)
            if os.path.exists(_p):
                os.unlink(_p)
            _ASE_LMDB_EXT = _ext
            break
        except Exception:
            pass
COMPRESSION_CODEC_SPECS = (
    ("none", "none", 0),
    ("lz4", "lz4", 0),
    ("zstd:3", "zstd", 3),
)

DEFAULT_BENCHMARK_CODEC_LABEL = "none"
DEFAULT_BENCHMARK_ATP_COMPRESSION = "none"
DEFAULT_BENCHMARK_ATP_LEVEL = 0
DEFAULT_BENCHMARK_POPULATE = False
MULTIPROCESSING_SKIP_BACKENDS_BY_ATOMS: dict[int, frozenset[str]] = {
    256: frozenset({"hdf5_soa"}),
}

_LMDB_CODEC_AVAILABLE: dict[str, bool] = {}
_DATASET_PATHS_CACHE: dict[tuple[Any, ...], dict[str, str]] = {}
_ASE_DATASET_PATHS_CACHE: dict[tuple[Any, ...], dict[str, str]] = {}


def _lmdb_codec_available(codec: str) -> bool:
    if codec in _LMDB_CODEC_AVAILABLE:
        return _LMDB_CODEC_AVAILABLE[codec]
    if _lmdb is None:
        _LMDB_CODEC_AVAILABLE[codec] = False
        return False
    try:
        lmdb_codec_fns(codec)
    except Exception:
        _LMDB_CODEC_AVAILABLE[codec] = False
        return False
    _LMDB_CODEC_AVAILABLE[codec] = True
    return True


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


def _format_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"


def _ensure_scratch_has_space(path: Path, *, context: str, min_free_bytes: int = 64 << 20) -> None:
    """Fail fast with a useful message when the scratch filesystem is full."""
    usage = shutil.disk_usage(path)
    if usage.free >= min_free_bytes:
        return
    raise RuntimeError(
        f"Scratch filesystem is effectively full for {context}: "
        f"{path} has only {_format_bytes(usage.free)} free. "
        "Free space on that filesystem or use --scratch-dir on a different disk."
    )

DEFAULT_SCRATCH_ENV = "ATOMPACK_BENCHMARK_SCRATCH"


def _default_scratch_dir() -> Path:
    override = os.environ.get(DEFAULT_SCRATCH_ENV)
    if override:
        return Path(override).expanduser()
    return Path(tempfile.gettempdir()) / "atompack-benchmarks"


DEFAULT_SCRATCH = _default_scratch_dir()

# Default molecule counts per atom count — sized so datasets are large enough
# to exceed page cache and stress real I/O.
N_MOLS_BY_ATOMS: dict[int, int] = {
    12: 1_000_000,
    16: 20_000_000,
    64: 1_000_000,
    256: 1_000_000,
    512: 1_000_000,
}
READ_SAMPLE_BY_ATOMS: dict[int, int] = {
    12: 100_000,
}
SKIP_CUSTOM_BENCHMARK_ATOMS = {256}

# Custom property keys to read during benchmarks (must match generate_molecules)
CUSTOM_PROP_KEYS = (
    "bandgap",
    "formation_energy",
    "eigenvalues",
    "mulliken_charges",
    "hirshfeld_volumes",
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _n_mols_for_atoms(atoms: int, override: int | None = None) -> int:
    """Return molecule count for a given atom count."""
    if override is not None:
        return override
    return N_MOLS_BY_ATOMS.get(atoms, 1_000_000)


def _compression_n_mols_for_atoms(atoms: int, override: int | None = None) -> int:
    """Smaller dataset sizing for compression-ratio focused runs."""
    if override is not None:
        return override
    return max(10_000, _n_mols_for_atoms(atoms) // 10)


def _custom_variants_for_atoms(atoms: int) -> tuple[bool, ...]:
    """Return which custom-property variants should run for a given atom count."""
    if atoms in SKIP_CUSTOM_BENCHMARK_ATOMS:
        return (False,)
    return (False, True)


def _read_sample(
    n_mols: int,
    override: int | None = None,
    atoms: int | None = None,
) -> int:
    """Return number of molecules to read per trial."""
    if override is not None:
        return min(override, n_mols)
    if atoms is not None and atoms in READ_SAMPLE_BY_ATOMS:
        return min(READ_SAMPLE_BY_ATOMS[atoms], n_mols)
    return max(10_000, n_mols // 100)


def _ase_read_sample(
    n_ase: int,
    sample: int,
    override: int | None = None,
) -> int:
    """Return the ASE read sample size for slow baseline backends."""
    target = ASE_READ_SAMPLE_DEFAULT if override is None else override
    return min(sample, n_ase, target)


def _multiprocessing_skipped_backends(atoms: int) -> frozenset[str]:
    """Return backends skipped for pathological multiprocessing cases."""
    return MULTIPROCESSING_SKIP_BACKENDS_BY_ATOMS.get(atoms, frozenset())


def _resolve_selected_backends(backends: list[str] | None) -> frozenset[str]:
    if backends is None:
        return frozenset(ALL_BACKENDS)
    return frozenset(backends)


def generate_molecules(
    n: int,
    atoms: int,
    seed: int = 42,
    with_custom_props: bool = False,
) -> list[dict]:
    """Generate n synthetic molecules with given atom count (for small n).

    For large datasets, use generate_molecule_stream() instead.
    """
    return list(generate_molecule_stream(n, atoms, seed, with_custom_props))


def generate_molecule_stream(
    n: int,
    atoms: int,
    seed: int = 42,
    with_custom_props: bool = False,
):
    """Yield synthetic molecules one at a time (constant memory).

    When with_custom_props is True, each molecule gets realistic custom
    properties that mirror what production DFT datasets carry:
      - per-molecule: bandgap (f64 scalar), formation_energy (f64 scalar),
                      eigenvalues (f64 array of length 20)
      - per-atom:     mulliken_charges (f64 array), hirshfeld_volumes (f64 array)
    """
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
        if with_custom_props:
            mol["custom"] = {
                "bandgap": float(rng.uniform(0, 5)),
                "formation_energy": float(rng.randn()),
                "eigenvalues": rng.randn(20).astype(np.float64),
                "mulliken_charges": rng.randn(atoms).astype(np.float64),
                "hirshfeld_volumes": rng.uniform(5, 30, atoms).astype(np.float64),
            }
        yield mol


# ---------------------------------------------------------------------------
# Dataset creation (with caching)
# ---------------------------------------------------------------------------


WRITE_BATCH_SIZE = 10_000
LMDB_SOA_CHUNK_SIZE = 1024
HDF5_SOA_CHUNK_SIZE = 256


def _make_molecule(m: dict) -> "atompack.Molecule":
    mol = atompack.Molecule.from_arrays(
        m["positions"],
        m["atomic_numbers"],
        energy=m["energy"],
        forces=m["forces"],
        cell=m["cell"],
        pbc=tuple(bool(x) for x in m["pbc"]),
    )
    for key, val in m.get("custom", {}).items():
        mol.set_property(key, val)
    return mol


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


def create_atompack_db(
    path: str,
    n: int,
    atoms: int,
    compression: str = "zstd",
    level: int = 3,
    with_custom: bool = False,
    seed: int = 42,
) -> None:
    report_every = max(1, n // 10)
    db = atompack.Database(
        path,
        compression=compression,
        level=level,
        overwrite=True,
    )
    try:
        if not with_custom:
            rng = np.random.RandomState(seed)
            cell = np.eye(3, dtype=np.float64) * 10.0
            pbc = np.array([True, True, True], dtype=bool)
            written = 0
            while written < n:
                if n >= 100_000 and written > 0 and written % report_every == 0:
                    print(f"    atompack write... {written:,}/{n:,}", flush=True)
                batch_size = min(WRITE_BATCH_SIZE, n - written)
                _write_atompack_builtin_batch(db, rng, batch_size, atoms, cell, pbc)
                written += batch_size
            db.flush()
            return

        batch: list[atompack.Molecule] = []
        for i, m in enumerate(generate_molecule_stream(n, atoms, seed, True)):
            if n >= 100_000 and i > 0 and i % report_every == 0:
                print(f"    atompack write... {i:,}/{n:,}", flush=True)
            batch.append(_make_molecule(m))
            if len(batch) >= WRITE_BATCH_SIZE:
                db.add_molecules(batch)
                batch.clear()
        if batch:
            db.add_molecules(batch)
        db.flush()
    finally:
        del db
        gc.collect()


def create_lmdb_packed(
    path: str,
    n: int,
    atoms: int,
    codec: str = "zstd:3",
    with_custom: bool = False,
    seed: int = 42,
) -> None:
    report_every = max(1, n // 10)
    shutil.rmtree(path, ignore_errors=True)
    Path(path).mkdir(parents=True, exist_ok=True)
    compress, _ = lmdb_codec_fns(codec)
    # Estimate per-molecule bytes: positions(f32*3) + forces(f32*3) + Z(u8) + cell + energy
    avg_bytes = atoms * (12 + 12 + 1) + 72 + 8 + 256
    if with_custom:
        avg_bytes += 320 + atoms * 16
    map_size = int(n * avg_bytes * 4) + 512 * (1 << 20)
    env = _lmdb.open(path, map_size=map_size)
    txn = env.begin(write=True)
    for i, m in enumerate(generate_molecule_stream(n, atoms, seed, with_custom)):
        if n >= 100_000 and i > 0 and i % report_every == 0:
            print(f"    lmdb_packed write... {i:,}/{n:,}", flush=True)
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


def create_lmdb_pickle(
    path: str, n: int, atoms: int, codec: str = "zstd:3",
    with_custom: bool = False, seed: int = 42,
) -> None:
    report_every = max(1, n // 10)
    shutil.rmtree(path, ignore_errors=True)
    store = PickleAtomLMDB(Path(path), PickleLMDBConfig(codec=codec))
    store.reset_dir()
    # Pickle overhead is significant: positions(f32*3) + forces(f32*3) + Z(u8)
    # + cell(f64*9) + pbc(u8*3) + energy(f64) + pickle framing (~512B).
    # Custom props add eigenvalues(f64*20) + mulliken/hirshfeld(f64*atoms each)
    # + scalars, roughly +320 + atoms*16 bytes.
    avg_bytes = atoms * (12 + 12 + 1) + 72 + 8 + 512
    if with_custom:
        avg_bytes += 320 + atoms * 16
    map_size = int(n * avg_bytes * 4) + 512 * (1 << 20)
    env = store.open_env(map_size=map_size, readonly=False, lock=True)
    txn = env.begin(write=True)
    for i, m in enumerate(generate_molecule_stream(n, atoms, seed, with_custom)):
        if n >= 100_000 and i > 0 and i % report_every == 0:
            print(f"    lmdb_pickle write... {i:,}/{n:,}", flush=True)
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


def create_hdf5_soa(
    path: str,
    n: int,
    atoms: int,
    with_custom: bool = False,
    seed: int = 42,
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

    rng = np.random.RandomState(seed)
    cell = np.eye(3, dtype=np.float64) * 10.0
    pbc = np.array([1, 1, 1], dtype=np.uint8)
    with store.open_file("r+") as handle:
        written = 0
        while written < n:
            if n >= 100_000 and written > 0 and written % report_every == 0:
                print(f"    hdf5_soa write... {written:,}/{n:,}", flush=True)
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


def create_ase_db(
    path: str, n: int, atoms: int,
    with_custom: bool = False, seed: int = 42,
) -> None:
    """Write molecules to an ASE database (sqlite or lmdb, auto-detected by path)."""
    report_every = max(1, n // 10)
    db = ase.db.connect(path)
    try:
        # Reuse one ASE transaction for the full write loop.  This avoids the
        # per-row begin/commit overhead that dominates ASE LMDB/SQLite writes.
        with db:
            for i, m in enumerate(generate_molecule_stream(n, atoms, seed, with_custom)):
                if n >= 1_000 and i > 0 and i % report_every == 0:
                    print(f"    ase write... {i:,}/{n:,}", flush=True)
                a = Atoms(
                    numbers=m["atomic_numbers"],
                    positions=m["positions"],
                    cell=m["cell"],
                    pbc=m["pbc"],
                )
                calc = SinglePointCalculator(
                    a, energy=m["energy"], forces=m["forces"],
                )
                a.calc = calc
                db.write(a)
    finally:
        _safe_close_ase_db(db)
        gc.collect()


def ensure_ase_datasets(
    scratch: Path,
    atoms: int,
    n_mols: int,
    with_custom: bool = False,
) -> dict[str, str]:
    """Create ASE sqlite + lmdb datasets if they don't exist. Returns paths dict."""
    cache_key = (scratch.resolve(), atoms, n_mols, with_custom)
    cached = _ASE_DATASET_PATHS_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    n_ase = min(n_mols, ASE_MAX_MOLS)
    tag = _dataset_tag(atoms, n_ase, "ase", with_custom)
    base = scratch / tag
    base.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    paths["ase_sqlite"] = str(base / "data.db")
    if _ASE_LMDB_EXT:
        paths["ase_lmdb"] = str(base / f"data{_ASE_LMDB_EXT}")

    need_any = False
    for key, p in paths.items():
        if not Path(p).exists() or Path(p).stat().st_size == 0:
            need_any = True
            break

    if need_any:
        _ensure_scratch_has_space(base, context=f"ASE dataset creation for {tag}")
        print(f"  Creating ASE datasets: {n_ase:,} molecules ({atoms} atoms"
              f"{', custom' if with_custom else ''})...")
        for key, p in paths.items():
            if Path(p).exists() and Path(p).stat().st_size > 0:
                continue
            t0 = time.perf_counter()
            create_ase_db(p, n_ase, atoms, with_custom=with_custom)
            dt = time.perf_counter() - t0
            sz = os.path.getsize(p) if Path(p).is_file() else _dir_size(p)
            print(f"    {key}:  {sz/1e6:.1f} MB  ({n_ase/dt:,.0f} mol/s, {dt:.0f}s)")
    else:
        print(f"  Using cached ASE datasets: {tag}")

    paths["n_ase"] = str(n_ase)
    _ASE_DATASET_PATHS_CACHE[cache_key] = dict(paths)
    return dict(paths)


def _dir_size(path: str | Path) -> int:
    """Total bytes of all files in a directory."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.iterdir() if f.is_file())


def _atompack_dataset_count(path: str | Path) -> int | None:
    p = Path(path)
    if not p.exists() or not p.is_file() or p.stat().st_size == 0:
        return None
    try:
        db = atompack.Database.open(str(p))
        return len(db)
    except Exception:
        return None


def _lmdb_dataset_count(path: str | Path) -> int | None:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return None
    try:
        if not any(p.iterdir()):
            return None
    except OSError:
        return None
    if _lmdb is None:
        return None
    try:
        env = _lmdb.open(
            str(p),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=True,
            max_dbs=1,
        )
        try:
            return int(env.stat()["entries"])
        finally:
            env.close()
    except Exception:
        return None


def _hdf5_soa_dataset_count(path: str | Path) -> int | None:
    p = Path(path)
    if not p.exists() or not p.is_file() or p.stat().st_size == 0:
        return None
    if not hdf5_available():
        return None
    try:
        store, reader = _open_hdf5_soa_reader(str(path))
        try:
            return int(reader.meta["num_molecules"])
        finally:
            reader.close()
    except Exception:
        return None


def _is_complete_dataset(label: str, path: str | Path, expected_count: int) -> bool:
    if label == "atompack":
        count = _atompack_dataset_count(path)
    elif label == "hdf5_soa":
        count = _hdf5_soa_dataset_count(path)
    elif label in {"lmdb_packed", "lmdb_pickle"}:
        count = _lmdb_dataset_count(path)
    else:
        raise ValueError(f"Unknown dataset label: {label}")
    return count == expected_count


def _dataset_tag(atoms: int, n_mols: int, codec: str, custom: bool) -> str:
    """Unique subdirectory name for a dataset."""
    cp = "_custom" if custom else ""
    return f"a{atoms}_n{n_mols}_{codec}{cp}"


def ensure_datasets(
    scratch: Path,
    atoms: int,
    n_mols: int,
    codec_label: str,
    atp_comp: str,
    atp_level: int,
    with_custom: bool = False,
    skip_lmdb_packed: bool = False,
    skip_backends: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Create datasets if they don't exist yet. Returns dict of paths."""
    cache_key = (
        scratch.resolve(),
        atoms,
        n_mols,
        codec_label,
        atp_comp,
        atp_level,
        with_custom,
        skip_lmdb_packed,
        tuple(sorted(skip_backends)),
    )
    cached = _DATASET_PATHS_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    tag = _dataset_tag(atoms, n_mols, codec_label, with_custom)
    base = scratch / tag
    base.mkdir(parents=True, exist_ok=True)

    paths = {
        "atompack": str(base / "data.atp"),
        "hdf5_soa": str(base / "hdf5_soa.h5"),
        "lmdb_packed": str(base / "lmdb_packed"),
        "lmdb_pickle": str(base / "lmdb_pickle"),
    }

    atp_exists = (
        True if "atompack" in skip_backends
        else _is_complete_dataset("atompack", paths["atompack"], n_mols)
    )
    hs_exists = (
        True if "hdf5_soa" in skip_backends
        else _is_complete_dataset("hdf5_soa", paths["hdf5_soa"], n_mols)
    )
    lp_exists = (
        True if "lmdb_packed" in skip_backends
        else _is_complete_dataset("lmdb_packed", paths["lmdb_packed"], n_mols)
    )
    lk_exists = (
        True if "lmdb_pickle" in skip_backends
        else _is_complete_dataset("lmdb_pickle", paths["lmdb_pickle"], n_mols)
    )

    need_hs = (
        "hdf5_soa" not in skip_backends
        and codec_label == "none"
        and hdf5_available()
        and not hs_exists
    )
    need_lp = "lmdb_packed" not in skip_backends and not skip_lmdb_packed and not lp_exists
    need_lk = "lmdb_pickle" not in skip_backends and not lk_exists
    need_atp = "atompack" not in skip_backends and not atp_exists
    need_any = need_atp or need_hs or need_lp or need_lk

    if need_any:
        _ensure_scratch_has_space(base, context=f"dataset creation for {tag}")
        print(f"  Creating datasets: {n_mols:,} molecules ({atoms} atoms, {codec_label}"
              f"{', custom' if with_custom else ''})...")

        lmdb_codec = "none" if codec_label == "none" else codec_label

        if need_atp:
            t0 = time.perf_counter()
            create_atompack_db(
                paths["atompack"], n_mols, atoms, atp_comp, atp_level,
                with_custom=with_custom,
            )
            dt = time.perf_counter() - t0
            sz = os.path.getsize(paths["atompack"])
            print(f"    atompack:     {sz/1e9:.2f} GB  ({n_mols/dt:,.0f} mol/s, {dt:.0f}s)")

        if need_hs:
            t0 = time.perf_counter()
            create_hdf5_soa(paths["hdf5_soa"], n_mols, atoms, with_custom=with_custom)
            dt = time.perf_counter() - t0
            sz = os.path.getsize(paths["hdf5_soa"])
            print(f"    hdf5_soa:     {sz/1e9:.2f} GB  ({n_mols/dt:,.0f} mol/s, {dt:.0f}s)")

        if need_lp:
            if _lmdb_codec_available(lmdb_codec):
                t0 = time.perf_counter()
                create_lmdb_packed(
                    paths["lmdb_packed"],
                    n_mols,
                    atoms,
                    lmdb_codec,
                    with_custom=with_custom,
                )
                dt = time.perf_counter() - t0
                sz = _dir_size(paths["lmdb_packed"])
                print(f"    lmdb_packed:  {sz/1e9:.2f} GB  ({n_mols/dt:,.0f} mol/s, {dt:.0f}s)")
            else:
                print(f"    lmdb_packed:  SKIP (missing codec {lmdb_codec})")

        if need_lk:
            if _lmdb_codec_available(lmdb_codec):
                t0 = time.perf_counter()
                create_lmdb_pickle(
                    paths["lmdb_pickle"], n_mols, atoms, lmdb_codec,
                    with_custom=with_custom,
                )
                dt = time.perf_counter() - t0
                sz = _dir_size(paths["lmdb_pickle"])
                print(f"    lmdb_pickle:  {sz/1e9:.2f} GB  ({n_mols/dt:,.0f} mol/s, {dt:.0f}s)")
            else:
                print(f"    lmdb_pickle:  SKIP (missing codec {lmdb_codec})")
    else:
        print(f"  Using cached datasets: {tag}")

    _DATASET_PATHS_CACHE[cache_key] = dict(paths)
    return dict(paths)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _rates_stats(rates: list[float]) -> dict[str, Any]:
    arr = np.asarray(rates, dtype=np.float64)
    median = float(_median(rates))
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    ci95 = float(1.96 * std / np.sqrt(arr.size)) if arr.size >= 2 else 0.0
    return {
        "mol_s": median,
        "median_mol_s": median,
        "mean_mol_s": mean,
        "std_mol_s": std,
        "ci95_mol_s": ci95,
        "trials_mol_s": [float(x) for x in rates],
    }


def bench(
    fn,
    n: int,
    trials: int = 3,
    warmup: int = 1,
    desc: str | None = None,
) -> dict[str, Any]:
    """Run fn() `trials` times and return summary stats."""
    if desc:
        progress = _progress(
            range(trials + warmup),
            desc=desc,
            total=trials + warmup,
            leave=False,
        )
    else:
        progress = range(trials + warmup)

    results = []
    for step in progress:
        if step < warmup:
            fn()
            continue
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        results.append(n / dt)
    return _rates_stats(results)


def _progress(
    iterable,
    *,
    desc: str,
    total: int | None = None,
    leave: bool = False,
):
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        dynamic_ncols=True,
    )


def _result_with_atoms_s(
    atoms_per_molecule: float, **result: Any
) -> dict[str, Any]:
    row = dict(result)
    mol_s = float(row["mol_s"])
    median_mol_s = float(row.get("median_mol_s", mol_s))
    mean_mol_s = float(row.get("mean_mol_s", mol_s))
    std_mol_s = float(row.get("std_mol_s", 0.0))
    ci95_mol_s = float(row.get("ci95_mol_s", 0.0))
    row["atoms_s"] = mol_s * float(atoms_per_molecule)
    row["median_atoms_s"] = median_mol_s * float(atoms_per_molecule)
    row["mean_atoms_s"] = mean_mol_s * float(atoms_per_molecule)
    row["std_atoms_s"] = std_mol_s * float(atoms_per_molecule)
    row["ci95_atoms_s"] = ci95_mol_s * float(atoms_per_molecule)
    return row


def _atompack_size_and_avg_atoms(
    path: Path, n_samples: int = 200
) -> tuple[int, float]:
    db = atompack.Database.open(str(path))
    size = len(db)
    if size <= 0:
        return 0, 0.0
    sample_n = min(n_samples, size)
    rng = np.random.default_rng(9999)
    counts = []
    for idx in rng.choice(size, size=sample_n, replace=False):
        mol = db.get_molecule(int(idx))
        counts.append(len(mol.atomic_numbers))
    return size, float(np.mean(counts)) if counts else 0.0


def _fmt_rate(row: dict[str, Any]) -> str:
    text = f"{row['mol_s']:>12,.0f} mol/s"
    atoms_s = row.get("atoms_s")
    if isinstance(atoms_s, (int, float)):
        text += f"  {atoms_s:>14,.0f} atoms/s"
    return text


@lru_cache(maxsize=128)
def _make_indices(
    n_mols: int,
    sample: int,
    seed: int = 1234,
    sequential: bool = False,
) -> tuple[int, ...]:
    """Build index tuple: either sequential or shuffled random sample."""
    if sequential:
        return tuple(range(min(sample, n_mols)))
    rng = np.random.default_rng(seed)
    return tuple(int(i) for i in rng.choice(n_mols, size=min(sample, n_mols), replace=False))


# ---------------------------------------------------------------------------
# Per-backend read functions
# ---------------------------------------------------------------------------


def _touch_array_value(arr: np.ndarray, seed: int) -> float:
    """Touch one deterministic element to force array access without a reduction."""
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[seed % len(arr)])
    return float(arr[seed % arr.shape[0], seed % arr.shape[1]])


def read_atompack(
    db, indices: list[int], with_custom: bool = False
) -> None:
    """Read + touch representative fields without benchmarking NumPy reductions."""
    for i in indices:
        m = db.get_molecule(i)
        _ = float(m.positions[i % len(m.positions), i % 3])
        _ = float(m.forces[i % len(m.forces), i % 3])
        _ = m.energy
        if with_custom:
            for key in CUSTOM_PROP_KEYS:
                v = m.get_property(key)
                if isinstance(v, np.ndarray):
                    _ = _touch_array_value(v, i)


def read_lmdb_packed(
    txn, decompress, indices: list[int], with_custom: bool = False
) -> None:
    for i in indices:
        v = txn.get(struct.pack(">Q", i))
        if v:
            if decompress:
                v = decompress(bytes(v))
            _, pos, _, _, _, energy, forces, _, _ = unpack_arrays(v)
            _ = float(pos[i % len(pos), i % 3])
            _ = float(forces[i % len(forces), i % 3])
            _ = energy
            if with_custom:
                custom = unpack_custom_fields(v)
                if custom is not None:
                    for key in CUSTOM_PROP_KEYS:
                        value = custom[key]
                        if isinstance(value, np.ndarray):
                            _ = _touch_array_value(value, i)


def read_hdf5_soa(
    reader, store: AtomHdf5Soa, indices: list[int], with_custom: bool = False
) -> None:
    batch_size = max(1, int(reader.meta.get("chunk_size", 1)))
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        payloads = store.get_payloads(reader, batch)
        for i, payload in zip(batch, payloads):
            positions = payload["positions"]
            forces = payload["forces"]
            _ = float(positions[i % len(positions), i % 3])
            _ = float(forces[i % len(forces), i % 3])
            _ = payload["energy"]
            if with_custom:
                props = payload["properties"]
                atom_props = payload["atom_properties"]
                _ = props["bandgap"]
                _ = props["formation_energy"]
                _ = float(props["eigenvalues"][i % len(props["eigenvalues"])])
                _ = float(atom_props["mulliken_charges"][i % len(atom_props["mulliken_charges"])])
                _ = float(atom_props["hirshfeld_volumes"][i % len(atom_props["hirshfeld_volumes"])])


def read_lmdb_pickle(
    txn, decompress, indices: list[int], with_custom: bool = False
) -> None:
    for i in indices:
        v = txn.get(struct.pack(">Q", i))
        if v:
            raw = decompress(bytes(v)) if decompress else v
            d = pickle.loads(raw)
            _ = float(d["positions"][i % len(d["positions"]), i % 3])
            _ = float(d["forces"][i % len(d["forces"]), i % 3])
            _ = d["energy"]
            if with_custom:
                for key in CUSTOM_PROP_KEYS:
                    val = d.get(key)
                    if isinstance(val, np.ndarray):
                        _ = _touch_array_value(val, i)


def read_ase(
    db, indices: list[int], with_custom: bool = False
) -> None:
    """Read via ase.db — indices are 0-based, ASE IDs are 1-based."""
    for i in indices:
        row = db.get(id=i + 1)
        _touch_ase_row(row, i)


def _touch_ase_row(row, index: int) -> None:
    """Touch ASE row arrays directly when available, falling back to Atoms."""
    positions = getattr(row, "positions", None)
    forces = getattr(row, "forces", None)
    energy = getattr(row, "energy", None)

    if positions is not None and forces is not None and energy is not None:
        _ = float(positions[index % len(positions), index % 3])
        _ = float(forces[index % len(forces), index % 3])
        _ = float(energy)
        return

    atoms = row.toatoms()
    _ = float(atoms.positions[index % len(atoms.positions), index % 3])
    atom_forces = atoms.get_forces()
    _ = float(atom_forces[index % len(atom_forces), index % 3])
    _ = atoms.get_potential_energy()


def _open_hdf5_soa_reader(path: str):
    if not hdf5_available():
        raise RuntimeError("h5py is required for hdf5_soa benchmarks")

    bootstrap = AtomHdf5Soa(
        Path(path),
        AtomHdf5SoaConfig(
            atoms_per_molecule=1,
            with_props=True,
            with_cell_pbc=True,
        ),
    )
    reader = bootstrap.open_reader()
    meta = reader.meta
    store = AtomHdf5Soa(
        Path(path),
        AtomHdf5SoaConfig(
            atoms_per_molecule=int(meta["atoms_per_molecule"]),
            with_props=bool(meta["with_props"]),
            with_cell_pbc=bool(meta["with_cell_pbc"]),
            with_stress=bool(meta["with_stress"]),
            with_charges=bool(meta["with_charges"]),
            compression=str(meta.get("compression", "none")),
            chunk_size=int(meta["chunk_size"]),
        ),
    )
    return store, reader


# ---------------------------------------------------------------------------
# Shared benchmark runner for read benchmarks
# ---------------------------------------------------------------------------


def _bench_read(
    paths: dict[str, str],
    atoms: int,
    n_mols: int,
    sample: int,
    codec_label: str,
    with_custom: bool,
    trials: int,
    sequential: bool = False,
    benchmark_name: str = "read",
    ase_paths: dict[str, str] | None = None,
    ase_read_sample_override: int | None = None,
    selected_backends: frozenset[str] = frozenset(ALL_BACKENDS),
) -> list[dict]:
    """Run read benchmark across all backends. Returns result rows."""
    results = []
    indices = _make_indices(n_mols, sample, sequential=sequential)
    n_read = len(indices)
    readahead = sequential
    os.environ["RAYON_NUM_THREADS"] = "1"
    lmdb_codec = "none" if codec_label == "none" else codec_label

    if "atompack" in selected_backends:
        db = atompack.Database.open(
            paths["atompack"], mmap=True, populate=DEFAULT_BENCHMARK_POPULATE
        )
        stats = bench(
            lambda: read_atompack(db, indices, with_custom),
            n_read,
            trials,
            desc=f"{benchmark_name} atompack",
        )
        rate = stats["mol_s"]
        print(f"  atompack:     {rate:>12,.0f} mol/s  ({1e6/rate:.1f} µs/mol)")
        results.append(_result_with_atoms_s(
            atoms, benchmark=benchmark_name, atoms=atoms,
            backend="atompack", n_mols=n_mols, sample=n_read, **stats,
        ))
        del db
        gc.collect()

    if (
        "hdf5_soa" in selected_backends
        and codec_label == "none"
        and hdf5_available()
        and Path(paths["hdf5_soa"]).exists()
    ):
        store, reader = _open_hdf5_soa_reader(paths["hdf5_soa"])
        stats = bench(
            lambda: read_hdf5_soa(reader, store, indices, with_custom),
            n_read,
            trials,
            desc=f"{benchmark_name} hdf5_soa",
        )
        rate = stats["mol_s"]
        print(f"  hdf5_soa:     {rate:>12,.0f} mol/s  ({1e6/rate:.1f} µs/mol)")
        results.append(_result_with_atoms_s(
            atoms, benchmark=benchmark_name, atoms=atoms,
            backend="hdf5_soa", n_mols=n_mols, sample=n_read, **stats,
        ))
        reader.close()

    selected_lmdb_backends = {
        backend for backend in ("lmdb_packed", "lmdb_pickle")
        if backend in selected_backends
    }
    if selected_lmdb_backends and not _lmdb_codec_available(lmdb_codec):
        if "lmdb_packed" in selected_lmdb_backends:
            print(f"  lmdb_packed:  SKIP (missing codec {lmdb_codec})")
        if with_custom and "lmdb_pickle" in selected_lmdb_backends:
            print(f"  lmdb_pickle:  SKIP (missing codec {lmdb_codec})")
    else:
        if "lmdb_packed" in selected_lmdb_backends:
            _, decompress = lmdb_codec_fns(lmdb_codec)
            env = _lmdb.open(paths["lmdb_packed"], subdir=True, readonly=True,
                             lock=False, readahead=readahead)
            txn = env.begin(write=False)
            stats = bench(
                lambda: read_lmdb_packed(txn, decompress, indices, with_custom),
                n_read,
                trials,
                desc=f"{benchmark_name} lmdb_packed",
            )
            rate = stats["mol_s"]
            print(f"  lmdb_packed:  {rate:>12,.0f} mol/s  ({1e6/rate:.1f} µs/mol)")
            results.append(_result_with_atoms_s(
                atoms, benchmark=benchmark_name, atoms=atoms,
                backend="lmdb_packed", n_mols=n_mols, sample=n_read, **stats,
            ))
            txn.abort()
            env.close()

        if "lmdb_pickle" in selected_lmdb_backends:
            _, decompress = lmdb_codec_fns(lmdb_codec)
            env = _lmdb.open(paths["lmdb_pickle"], subdir=True, readonly=True,
                             lock=False, readahead=readahead)
            txn = env.begin(write=False)
            stats = bench(
                lambda: read_lmdb_pickle(txn, decompress, indices, with_custom),
                n_read,
                trials,
                desc=f"{benchmark_name} lmdb_pickle",
            )
            rate = stats["mol_s"]
            print(f"  lmdb_pickle:  {rate:>12,.0f} mol/s  ({1e6/rate:.1f} µs/mol)")
            results.append(_result_with_atoms_s(
                atoms, benchmark=benchmark_name, atoms=atoms,
                backend="lmdb_pickle", n_mols=n_mols, sample=n_read, **stats,
            ))
            txn.abort()
            env.close()

    # ASE backends (smaller dataset — indices capped to n_ase)
    if _has_ase and ase_paths:
        n_ase = int(ase_paths.get("n_ase", 0))
        if n_ase > 0:
            ase_sample = _ase_read_sample(n_ase, sample, ase_read_sample_override)
            ase_indices = _make_indices(n_ase, ase_sample, sequential=sequential)
            n_ase_read = len(ase_indices)

            for key, label in [("ase_sqlite", "ase_sqlite"), ("ase_lmdb", "ase_lmdb")]:
                if key not in ase_paths or label not in selected_backends:
                    continue
                ase_db = None
                try:
                    ase_db = ase.db.connect(ase_paths[key])
                    stats = bench(
                        lambda db=ase_db: read_ase(db, ase_indices, with_custom),
                        n_ase_read,
                        trials,
                        desc=f"{benchmark_name} {label}",
                    )
                except Exception as exc:
                    print(f"  {label:<13s}  SKIP ({exc})"
                          f"  [sample={n_ase_read:,} / N={n_ase:,}]")
                    continue
                finally:
                    if ase_db is not None:
                        _safe_close_ase_db(ase_db)
                rate = stats["mol_s"]
                print(f"  {label:<13s}  {rate:>12,.0f} mol/s  ({1e6/rate:.1f} µs/mol)"
                      f"  [sample={n_ase_read:,} / N={n_ase:,}]")
                results.append(_result_with_atoms_s(
                    atoms, benchmark=benchmark_name, atoms=atoms,
                    backend=label, n_mols=n_ase, sample=n_ase_read, **stats,
                ))

    return results


# ---------------------------------------------------------------------------
# Multiprocessing workers
# ---------------------------------------------------------------------------


def _atp_worker_cleanup() -> None:
    global _atp_db
    _atp_db = None


def _atp_worker_init(path: str) -> None:
    global _atp_db
    os.environ["RAYON_NUM_THREADS"] = "1"
    _atp_db = atompack.Database.open(path, mmap=True, populate=DEFAULT_BENCHMARK_POPULATE)
    atexit.register(_atp_worker_cleanup)
    for i in range(min(10, len(_atp_db))):
        _ = _atp_db.get_molecule(i)


def _atp_worker(chunk: list[int]) -> int:
    global _atp_db
    for i in chunk:
        m = _atp_db.get_molecule(i)
        _ = float(m.positions[i % len(m.positions), i % 3])
        _ = float(m.forces[i % len(m.forces), i % 3])
        _ = m.energy
    return len(chunk)


def _hdf5_soa_worker_cleanup() -> None:
    global _hs_store, _hs_reader
    if _hs_reader is not None:
        try:
            _hs_reader.close()
        except Exception:
            pass
    _hs_store = None
    _hs_reader = None


def _hdf5_soa_worker_init(path: str) -> None:
    global _hs_store, _hs_reader
    _hs_store, _hs_reader = _open_hdf5_soa_reader(path)
    atexit.register(_hdf5_soa_worker_cleanup)


def _hdf5_soa_worker(chunk: list[int]) -> int:
    global _hs_store, _hs_reader
    payloads = _hs_store.get_payloads(_hs_reader, chunk)
    for i, payload in zip(chunk, payloads):
        positions = payload["positions"]
        forces = payload["forces"]
        _ = float(positions[i % len(positions), i % 3])
        _ = float(forces[i % len(forces), i % 3])
        _ = payload["energy"]
    return len(chunk)


def _lmdb_packed_worker_cleanup() -> None:
    global _lp_env, _lp_txn, _lp_dec
    if _lp_txn is not None:
        try:
            _lp_txn.abort()
        except Exception:
            pass
    if _lp_env is not None:
        try:
            _lp_env.close()
        except Exception:
            pass
    _lp_env = None
    _lp_txn = None
    _lp_dec = None


def _lmdb_packed_worker_init(path: str, codec: str) -> None:
    global _lp_env, _lp_txn, _lp_dec
    _, _lp_dec = lmdb_codec_fns(codec)
    _lp_env = _lmdb.open(path, subdir=True, readonly=True, lock=False, readahead=False)
    _lp_txn = _lp_env.begin(write=False)
    atexit.register(_lmdb_packed_worker_cleanup)


def _lmdb_packed_worker(chunk: list[int]) -> int:
    global _lp_txn, _lp_dec
    for i in chunk:
        v = _lp_txn.get(struct.pack(">Q", i))
        if v:
            if _lp_dec:
                v = _lp_dec(bytes(v))
            _, pos, _, _, _, energy, forces, _, _ = unpack_arrays(v)
            _ = float(pos[i % len(pos), i % 3])
            _ = float(forces[i % len(forces), i % 3])
            _ = energy
    return len(chunk)


def _lmdb_pickle_worker_cleanup() -> None:
    global _lk_env, _lk_txn, _lk_dec
    if _lk_txn is not None:
        try:
            _lk_txn.abort()
        except Exception:
            pass
    if _lk_env is not None:
        try:
            _lk_env.close()
        except Exception:
            pass
    _lk_env = None
    _lk_txn = None
    _lk_dec = None


def _lmdb_pickle_worker_init(path: str, codec: str) -> None:
    global _lk_env, _lk_txn, _lk_dec
    _, _lk_dec = lmdb_codec_fns(codec)
    _lk_env = _lmdb.open(path, subdir=True, readonly=True, lock=False, readahead=False)
    _lk_txn = _lk_env.begin(write=False)
    atexit.register(_lmdb_pickle_worker_cleanup)


def _lmdb_pickle_worker(chunk: list[int]) -> int:
    global _lk_txn, _lk_dec
    for i in chunk:
        v = _lk_txn.get(struct.pack(">Q", i))
        if v:
            raw = _lk_dec(bytes(v)) if _lk_dec else v
            d = pickle.loads(raw)
            _ = float(d["positions"][i % len(d["positions"]), i % 3])
            _ = float(d["forces"][i % len(d["forces"]), i % 3])
            _ = d["energy"]
    return len(chunk)


def _ase_worker_cleanup() -> None:
    global _ase_db
    if _ase_db is not None:
        _safe_close_ase_db(_ase_db)
    _ase_db = None


def _ase_worker_init(path: str) -> None:
    global _ase_db
    _ase_db = ase.db.connect(path)
    atexit.register(_ase_worker_cleanup)


def _ase_worker(chunk: list[int]) -> int:
    global _ase_db
    for i in chunk:
        row = _ase_db.get(id=i + 1)
        _touch_ase_row(row, i)
    return len(chunk)


# ---------------------------------------------------------------------------
# Benchmark 1: Sequential read throughput
# ---------------------------------------------------------------------------


def bench_single_threaded(
    atom_counts: list[int],
    scratch: Path,
    n_mols_override: int | None = None,
    read_sample_override: int | None = None,
    ase_read_sample_override: int | None = None,
    trials: int = 5,
    selected_backends: frozenset[str] = frozenset(ALL_BACKENDS),
) -> list[dict]:
    """Sequential read throughput — best-case for page cache / readahead."""
    print("\n" + "=" * 60)
    print("Benchmark 1: Sequential read throughput")
    print("=" * 60)
    results = []

    for atoms in _progress(atom_counts, desc="benchmark1 atoms", leave=True):
        n_mols = _n_mols_for_atoms(atoms, n_mols_override)
        sample = _read_sample(n_mols, read_sample_override, atoms)

        variants = _custom_variants_for_atoms(atoms)
        for with_custom in _progress(
            variants,
            desc=f"{atoms} atoms variants",
            leave=False,
            total=len(variants),
        ):
            tag = " + custom" if with_custom else ""
            bm_name = "sequential_custom" if with_custom else "sequential"
            print(f"\n--- {atoms} atoms, {n_mols:,} mols, read {sample:,}{tag} ---")

            paths = ensure_datasets(
                scratch,
                atoms,
                n_mols,
                DEFAULT_BENCHMARK_CODEC_LABEL,
                DEFAULT_BENCHMARK_ATP_COMPRESSION,
                DEFAULT_BENCHMARK_ATP_LEVEL,
                with_custom=with_custom,
                skip_backends=frozenset(
                    backend for backend in BACKENDS if backend not in selected_backends
                ),
            )
            ase_paths = None
            if _has_ase and any(backend in selected_backends for backend in ASE_BACKENDS):
                ase_paths = ensure_ase_datasets(
                    scratch, atoms, n_mols, with_custom=with_custom,
                )
            results.extend(_bench_read(
                paths,
                atoms,
                n_mols,
                sample,
                DEFAULT_BENCHMARK_CODEC_LABEL,
                with_custom,
                trials,
                sequential=True, benchmark_name=bm_name, ase_paths=ase_paths,
                ase_read_sample_override=ase_read_sample_override,
                selected_backends=selected_backends,
            ))

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Compression impact (random access)
# ---------------------------------------------------------------------------


def bench_compression(
    atoms: int = 256,
    scratch: Path = DEFAULT_SCRATCH,
    n_mols_override: int | None = None,
    read_sample_override: int | None = None,
    ase_read_sample_override: int | None = None,
    trials: int = 5,
    selected_backends: frozenset[str] = frozenset(ALL_BACKENDS),
) -> list[dict]:
    """Compression impact with random access reads."""
    n_mols = _compression_n_mols_for_atoms(atoms, n_mols_override)
    sample = _read_sample(n_mols, read_sample_override, atoms)

    print("\n" + "=" * 60)
    print(f"Benchmark 2: Compression impact ({atoms} atoms, {n_mols:,} mols)")
    print("=" * 60)
    results = []
    codecs = [("none", "none", 0), ("lz4", "lz4", 0), ("zstd:3", "zstd", 3)]

    variants = _custom_variants_for_atoms(atoms)
    for with_custom in _progress(
        variants,
        desc=f"{atoms} atoms variants",
        leave=False,
        total=len(variants),
    ):
        tag = " + custom" if with_custom else ""
        bm_name = "compression_custom" if with_custom else "compression"

        for codec_label, atp_comp, atp_level in _progress(
            codecs,
            desc=f"{bm_name} codecs",
            leave=False,
            total=len(codecs),
        ):
            enabled_for_codec = set()
            if "atompack" in selected_backends:
                enabled_for_codec.add("atompack")
            if "hdf5_soa" in selected_backends and codec_label == "none":
                enabled_for_codec.add("hdf5_soa")
            if "lmdb_packed" in selected_backends:
                enabled_for_codec.add("lmdb_packed")
            if "lmdb_pickle" in selected_backends:
                enabled_for_codec.add("lmdb_pickle")
            if not enabled_for_codec:
                continue

            print(f"\n--- {codec_label}{tag}, read {sample:,} ---")
            paths = ensure_datasets(
                scratch, atoms, n_mols, codec_label, atp_comp, atp_level,
                with_custom=with_custom,
                skip_backends=frozenset(
                    backend for backend in BACKENDS if backend not in selected_backends
                ),
            )

            size_by_backend: dict[str, int] = {}
            if "atompack" in enabled_for_codec and Path(paths["atompack"]).exists():
                size_by_backend["atompack"] = os.path.getsize(paths["atompack"])
            if "hdf5_soa" in enabled_for_codec and Path(paths["hdf5_soa"]).exists():
                size_by_backend["hdf5_soa"] = os.path.getsize(paths["hdf5_soa"])
            if "lmdb_packed" in enabled_for_codec and Path(paths["lmdb_packed"]).exists():
                size_by_backend["lmdb_packed"] = _dir_size(paths["lmdb_packed"])
            if "lmdb_pickle" in enabled_for_codec and Path(paths["lmdb_pickle"]).exists():
                size_by_backend["lmdb_pickle"] = _dir_size(paths["lmdb_pickle"])

            parts = [
                f"{backend}={size_by_backend[backend]/1e9:.2f}GB"
                for backend in BACKENDS
                if backend in size_by_backend
            ]
            sizes = "  ".join(parts)
            if sizes:
                print(f"  file sizes:  {sizes}")

            read_results = _bench_read(
                paths, atoms, n_mols, sample, codec_label, with_custom, trials,
                benchmark_name=bm_name,
                ase_read_sample_override=ase_read_sample_override,
                selected_backends=selected_backends,
            )
            for r in read_results:
                r["codec"] = codec_label
                r["size_bytes"] = size_by_backend.get(r["backend"], 0)
            results.extend(read_results)

    return results


# ---------------------------------------------------------------------------
# Benchmark 3: Multiprocessing scaling
# ---------------------------------------------------------------------------


def _mp_bench(ctx, nw, init_fn, init_args, worker_fn, chunks, n_mols, trials, desc: str):
    """Create a pool, warmup, measure, and always tear it down cleanly."""
    pool = ctx.Pool(nw, initializer=init_fn, initargs=init_args)
    try:
        progress = _progress(range(trials + 1), desc=desc, leave=False, total=trials + 1)
        rates = []
        for step in progress:
            if step == 0:
                pool.map(worker_fn, chunks)  # warmup
                continue
            t0 = time.perf_counter()
            pool.map(worker_fn, chunks)
            dt = time.perf_counter() - t0
            rates.append(n_mols / dt)
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()
        return _rates_stats(rates)


def bench_multiprocessing(
    atom_counts: list[int],
    worker_counts: list[int],
    scratch: Path = DEFAULT_SCRATCH,
    n_mols_override: int | None = None,
    read_sample_override: int | None = None,
    ase_read_sample_override: int | None = None,
    trials: int = 3,
    selected_backends: frozenset[str] = frozenset(ALL_BACKENDS),
) -> list[dict]:
    """Measure scaling with persistent multiprocessing workers."""
    print("\n" + "=" * 60)
    print("Benchmark 3: Multiprocessing scaling")
    print("=" * 60)
    results = []
    ctx = mp.get_context("spawn")

    for atoms in _progress(atom_counts, desc="benchmark3 atoms", leave=True):
        n_mols = _n_mols_for_atoms(atoms, n_mols_override)
        sample = _read_sample(n_mols, read_sample_override, atoms)
        skipped_backends = _multiprocessing_skipped_backends(atoms)
        skipped_backends = frozenset(
            set(skipped_backends)
            | {backend for backend in ALL_BACKENDS if backend not in selected_backends}
        )
        print(f"\n--- {atoms} atoms, {n_mols:,} mols, read {sample:,} ---")
        if skipped_backends:
            skipped_labels = ", ".join(sorted(skipped_backends))
            print(
                f"  skipping pathological random-access baselines at {atoms} atoms: "
                f"{skipped_labels}"
            )

        paths = ensure_datasets(
            scratch,
            atoms,
            n_mols,
            DEFAULT_BENCHMARK_CODEC_LABEL,
            DEFAULT_BENCHMARK_ATP_COMPRESSION,
            DEFAULT_BENCHMARK_ATP_LEVEL,
            skip_backends=skipped_backends,
        )
        indices = _make_indices(n_mols, sample)

        ase_paths = None
        enabled_ase_backends = [
            label for label in ASE_BACKENDS
            if label not in skipped_backends
        ]
        if _has_ase and enabled_ase_backends:
            ase_paths = ensure_ase_datasets(scratch, atoms, n_mols)

        for nw in _progress(worker_counts, desc=f"{atoms} atoms workers", leave=False):
            chunk_size = max(1, len(indices) // nw)
            chunks = [indices[i:i + chunk_size]
                      for i in range(0, len(indices), chunk_size)]

            atp_stats = None
            if "atompack" not in skipped_backends:
                atp_stats = _mp_bench(
                    ctx, nw, _atp_worker_init, (paths["atompack"],),
                    _atp_worker, chunks, len(indices), trials,
                    desc=f"{atoms}a {nw}w atompack",
                )
            hs_stats = None
            if (
                "hdf5_soa" not in skipped_backends
                and hdf5_available()
                and Path(paths["hdf5_soa"]).exists()
            ):
                hs_stats = _mp_bench(
                    ctx, nw, _hdf5_soa_worker_init, (paths["hdf5_soa"],),
                    _hdf5_soa_worker, chunks, len(indices), trials,
                    desc=f"{atoms}a {nw}w hdf5_soa",
                )
            lp_stats = None
            if "lmdb_packed" not in skipped_backends:
                lp_stats = _mp_bench(
                    ctx, nw, _lmdb_packed_worker_init,
                    (paths["lmdb_packed"], DEFAULT_BENCHMARK_CODEC_LABEL),
                    _lmdb_packed_worker, chunks, len(indices), trials,
                    desc=f"{atoms}a {nw}w lmdb_packed",
                )
            lk_stats = None
            if "lmdb_pickle" not in skipped_backends:
                lk_stats = _mp_bench(
                    ctx, nw, _lmdb_pickle_worker_init,
                    (paths["lmdb_pickle"], DEFAULT_BENCHMARK_CODEC_LABEL),
                    _lmdb_pickle_worker, chunks, len(indices), trials,
                    desc=f"{atoms}a {nw}w lmdb_pickle",
                )

            backend_rates: list[tuple[str, dict[str, Any]]] = []
            if atp_stats is not None:
                backend_rates.append(("atompack", atp_stats))
            if hs_stats is not None:
                backend_rates.append(("hdf5_soa", hs_stats))
            if lp_stats is not None:
                backend_rates.append(("lmdb_packed", lp_stats))
            if lk_stats is not None:
                backend_rates.append(("lmdb_pickle", lk_stats))

            # ASE backends (indices capped to n_ase)
            if _has_ase and ase_paths:
                n_ase = int(ase_paths.get("n_ase", 0))
                if n_ase > 0:
                    ase_sample = _ase_read_sample(n_ase, sample, ase_read_sample_override)
                    ase_indices = _make_indices(n_ase, ase_sample)
                    ase_chunk_size = max(1, len(ase_indices) // nw)
                    ase_chunks = [ase_indices[i:i + ase_chunk_size]
                                  for i in range(0, len(ase_indices), ase_chunk_size)]
                    for key, label in [("ase_sqlite", "ase_sqlite"), ("ase_lmdb", "ase_lmdb")]:
                        if label in skipped_backends:
                            continue
                        if key not in ase_paths:
                            continue
                        stats = _mp_bench(
                            ctx, nw, _ase_worker_init, (ase_paths[key],),
                            _ase_worker, ase_chunks, len(ase_indices), trials,
                            desc=f"{atoms}a {nw}w {label}",
                        )
                        backend_rates.append((label, stats))

            parts = "  ".join(f"{b}={stats['mol_s']:>10,.0f}" for b, stats in backend_rates)
            print(f"  {nw}w:  {parts}")
            for backend, stats in backend_rates:
                n_used = n_mols
                s_used = len(indices)
                if backend.startswith("ase_") and ase_paths:
                    n_used = int(ase_paths.get("n_ase", n_mols))
                    s_used = min(sample, n_used)
                results.append(_result_with_atoms_s(
                    atoms,
                    benchmark="multiprocessing",
                    atoms=atoms,
                    workers=nw,
                    backend=backend,
                    n_mols=n_used,
                    sample=s_used,
                    **stats,
                ))

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    by_bench: dict[str, list[dict]] = {}
    for r in all_results:
        by_bench.setdefault(r["benchmark"], []).append(r)

    bench_names = {
        "sequential": "1a. Sequential read throughput",
        "sequential_custom": "1b. Sequential + custom props",
        "compression": "2a. Compression impact (random access)",
        "compression_custom": "2b. Compression + custom props (random access)",
        "multiprocessing": "3. Multiprocessing scaling",
    }

    for bench_key, title in bench_names.items():
        rows = by_bench.get(bench_key, [])
        if not rows:
            continue
        print(f"\n{title}")
        print("-" * 70)

        if bench_key in ("sequential", "sequential_custom"):
            atoms_vals = sorted(set(r["atoms"] for r in rows))
            for a in atoms_vals:
                n_info = ""
                sample_row = next((r for r in rows if r["atoms"] == a), None)
                if sample_row:
                    nm = sample_row.get("n_mols")
                    ns = sample_row.get("sample")
                    if nm:
                        n_info = f" ({nm:,} mols"
                        if ns:
                            n_info += f", read {ns:,}"
                        n_info += ")"
                print(f"  {a} atoms{n_info}:")
                for r in rows:
                    if r["atoms"] == a:
                        print(f"    {r['backend']:<14s} {_fmt_rate(r)}")
        elif bench_key in ("compression", "compression_custom"):
            codecs = sorted(
                set(r["codec"] for r in rows),
                key=lambda c: ["none", "lz4", "zstd:3"].index(c),
            )
            for c in codecs:
                print(f"  {c}:")
                for r in rows:
                    if r["codec"] == c:
                        print(f"    {r['backend']:<14s} {_fmt_rate(r)}  "
                              f"({r['size_bytes']/1e9:.2f} GB)")
        elif bench_key == "multiprocessing":
            atoms_vals = sorted(set(r["atoms"] for r in rows))
            for a in atoms_vals:
                print(f"  {a} atoms:")
                workers = sorted(
                    set(r["workers"] for r in rows if r["atoms"] == a)
                )
                for w in workers:
                    parts = []
                    for r in rows:
                        if r["atoms"] == a and r["workers"] == w:
                            parts.append(
                                f"{r['backend']}={r['mol_s']:,.0f}"
                            )
                    print(f"    {w}w: {', '.join(parts)} mol/s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Atompack benchmark suite")
    parser.add_argument(
        "--only", nargs="*", type=int, default=None,
        help="Run only these benchmark numbers (1-3)",
    )
    parser.add_argument(
        "--atoms", nargs="*", type=int, default=[12, 64, 256],
        help="Atom counts for benchmarks 1 and 3 (first value also used for 2)",
    )
    parser.add_argument(
        "--workers", nargs="*", type=int, default=[1, 2, 4, 8],
        help="Worker counts for benchmark 3",
    )
    parser.add_argument(
        "--n-mols", type=int, default=None,
        help="Override molecule count (default: auto-scale by atom count)",
    )
    parser.add_argument(
        "--read-sample", type=int, default=None,
        help="Override read sample size per trial (default: 1%% of dataset, min 10K)",
    )
    parser.add_argument(
        "--ase-read-sample", type=int, default=ASE_READ_SAMPLE_DEFAULT,
        help=(
            "Cap ASE-backed read samples per trial separately "
            f"(default: {ASE_READ_SAMPLE_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--scratch-dir", type=str, default=str(DEFAULT_SCRATCH),
        help=f"Persistent scratch directory for datasets (default: {DEFAULT_SCRATCH})",
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        choices=ALL_BACKENDS,
        default=None,
        help="Run only these backends",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    run = set(args.only) if args.only else {1, 2, 3}
    selected_backends = _resolve_selected_backends(args.backends)
    all_results: list[dict] = []

    if 1 in run:
        all_results.extend(bench_single_threaded(
            args.atoms, scratch, args.n_mols, args.read_sample, args.ase_read_sample,
            selected_backends=selected_backends,
        ))
    if 2 in run:
        # Use first atom count for compression benchmark
        comp_atoms = args.atoms[0] if args.atoms else 256
        all_results.extend(bench_compression(
            comp_atoms, scratch, args.n_mols, args.read_sample, args.ase_read_sample,
            selected_backends=selected_backends,
        ))
    if 3 in run:
        all_results.extend(bench_multiprocessing(
            args.atoms, args.workers, scratch, args.n_mols, args.read_sample,
            args.ase_read_sample,
            selected_backends=selected_backends,
        ))

    print_summary(all_results)

    output_path = args.out or (scratch / "benchmark_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
