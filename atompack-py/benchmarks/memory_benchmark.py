# Copyright 2026 Entalpic
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Set Rayon before importing atompack so child workers stay single-threaded.
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import atompack

try:
    import ase.db

    _HAS_ASE = True
except ImportError:
    _HAS_ASE = False

try:
    import lmdb  # type: ignore
except ImportError:
    lmdb = None

from atom_lmdb import lmdb_codec_fns, unpack_arrays
from atom_lmdb_pickle import PickleAtomLMDB, PickleLMDBConfig

from benchmark import (
    DEFAULT_BENCHMARK_ATP_COMPRESSION,
    DEFAULT_BENCHMARK_ATP_LEVEL,
    DEFAULT_BENCHMARK_CODEC_LABEL,
    DEFAULT_BENCHMARK_POPULATE,
    DEFAULT_SCRATCH,
    _ASE_LMDB_EXT,
    _ase_read_sample,
    _dataset_tag,
    _make_indices,
    _n_mols_for_atoms,
    _read_sample,
    _safe_close_ase_db,
    ensure_ase_datasets,
    ensure_datasets,
)


DEFAULT_WORKERS = 8
DEFAULT_TOUCHES_PER_WORKER = 32


@dataclass(frozen=True)
class Paths:
    atompack: Path | None
    lmdb_packed: Path | None
    lmdb_pickle: Path | None
    n_mols: int
    ase_sqlite: Path | None = None
    ase_lmdb: Path | None = None
    n_ase: int = 0


def _memory_snapshot_kib() -> dict[str, int]:
    snapshot = {
        "rss_kib": 0,
        "rss_anon_kib": 0,
        "rss_file_kib": 0,
        "rss_shmem_kib": 0,
        "private_clean_kib": 0,
        "private_dirty_kib": 0,
        "private_total_kib": 0,
    }
    status_key_map = {
        "VmRSS:": "rss_kib",
        "RssAnon:": "rss_anon_kib",
        "RssFile:": "rss_file_kib",
        "RssShmem:": "rss_shmem_kib",
    }
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        mapped = status_key_map.get(parts[0])
        if mapped is not None:
            snapshot[mapped] = int(parts[1])

    smaps_rollup = Path("/proc/self/smaps_rollup")
    if smaps_rollup.exists():
        smaps_key_map = {
            "Private_Clean:": "private_clean_kib",
            "Private_Dirty:": "private_dirty_kib",
        }
        for line in smaps_rollup.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            mapped = smaps_key_map.get(parts[0])
            if mapped is not None:
                snapshot[mapped] = int(parts[1])
    snapshot["private_total_kib"] = (
        snapshot["private_clean_kib"] + snapshot["private_dirty_kib"]
    )
    return snapshot


def _delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in after}


def _format_mem(mem: dict[str, int]) -> str:
    return (
        f"rss={mem['rss_kib']/1024:7.0f} MiB  "
        f"private={mem['private_total_kib']/1024:7.0f} MiB  "
        f"anon={mem['rss_anon_kib']/1024:7.0f} MiB  "
        f"file={mem['rss_file_kib']/1024:7.0f} MiB  "
        f"shmem={mem['rss_shmem_kib']/1024:7.0f} MiB"
    )


def _touch_array_value(arr: np.ndarray, seed: int) -> None:
    if arr.ndim == 0:
        _ = float(arr)
    elif arr.ndim == 1:
        _ = float(arr[seed % len(arr)])
    else:
        _ = float(arr[seed % arr.shape[0], seed % arr.shape[1]])


def _touch_molecule_arrays(positions: np.ndarray, atomic_numbers: np.ndarray, forces: np.ndarray, seed: int) -> None:
    _touch_array_value(positions, seed)
    if atomic_numbers.size:
        _ = int(atomic_numbers[seed % len(atomic_numbers)])
    _touch_array_value(forces, seed)


def _resolve_paths(
    scratch: Path,
    atoms: int,
    n_mols_override: int | None,
    *,
    create_missing: bool,
) -> Paths:
    n_mols = _n_mols_for_atoms(atoms, n_mols_override)

    base_dir = scratch / _dataset_tag(atoms, n_mols, DEFAULT_BENCHMARK_CODEC_LABEL, False)
    atompack = base_dir / "data.atp"
    lmdb_packed = base_dir / "lmdb_packed"
    lmdb_pickle = base_dir / "lmdb_pickle"

    if create_missing:
        base = ensure_datasets(
            scratch,
            atoms,
            n_mols,
            DEFAULT_BENCHMARK_CODEC_LABEL,
            DEFAULT_BENCHMARK_ATP_COMPRESSION,
            DEFAULT_BENCHMARK_ATP_LEVEL,
            with_custom=False,
            skip_lmdb_packed=False,
        )
        atompack = Path(base["atompack"])
        lmdb_packed = Path(base["lmdb_packed"])
        lmdb_pickle = Path(base["lmdb_pickle"])

    atompack_path = atompack if atompack.is_file() and atompack.stat().st_size > 0 else None
    lmdb_packed_path = lmdb_packed if lmdb_packed.is_dir() else None
    lmdb_pickle_path = lmdb_pickle if lmdb_pickle.is_dir() else None

    if atompack_path is None:
        raise FileNotFoundError(
            "Missing benchmark dataset for atompack at "
            f"{atompack}. Run benchmark.py first for the same --atoms/--n-mols, "
            "or pass --create-missing to build datasets from memory_benchmark.py."
        )

    ase_sqlite: Path | None = None
    ase_lmdb: Path | None = None
    n_ase = 0
    if _HAS_ASE:
        n_ase = min(n_mols, 50_000)
        ase_dir = scratch / _dataset_tag(atoms, n_ase, "ase", False)
        sqlite_candidate = ase_dir / "data.db"
        if sqlite_candidate.is_file() and sqlite_candidate.stat().st_size > 0:
            ase_sqlite = sqlite_candidate

        if _ASE_LMDB_EXT:
            lmdb_candidate = ase_dir / f"data{_ASE_LMDB_EXT}"
            if lmdb_candidate.exists():
                ase_lmdb = lmdb_candidate

        if create_missing and (ase_sqlite is None or (_ASE_LMDB_EXT and ase_lmdb is None)):
            ase = ensure_ase_datasets(scratch, atoms, n_mols, with_custom=False)
            ase_sqlite = Path(ase["ase_sqlite"]) if "ase_sqlite" in ase else None
            ase_lmdb = Path(ase["ase_lmdb"]) if "ase_lmdb" in ase else None
            n_ase = int(ase.get("n_ase", 0))

    return Paths(
        atompack=atompack_path,
        lmdb_packed=lmdb_packed_path,
        lmdb_pickle=lmdb_pickle_path,
        ase_sqlite=ase_sqlite,
        ase_lmdb=ase_lmdb,
        n_ase=n_ase,
        n_mols=n_mols,
    )


def _measure_stream_atompack(path: Path, indices: list[int]) -> dict[str, int]:
    before = _memory_snapshot_kib()
    db = atompack.Database.open(
        str(path), mmap=True, populate=DEFAULT_BENCHMARK_POPULATE
    )
    try:
        for idx in indices:
            mol = db.get_molecule(int(idx))
            _touch_molecule_arrays(mol.positions, mol.atomic_numbers, mol.forces, int(idx))
            _ = mol.energy
    finally:
        del db
        gc.collect()
    after = _memory_snapshot_kib()
    return _delta(before, after)


def _measure_stream_lmdb_packed(path: Path, indices: list[int]) -> dict[str, int]:
    before = _memory_snapshot_kib()
    _, decompress = lmdb_codec_fns(DEFAULT_BENCHMARK_CODEC_LABEL)
    env = lmdb.open(
        str(path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        max_dbs=1,
    )
    txn = env.begin(write=False)
    try:
        for idx in indices:
            value = txn.get(struct.pack(">Q", int(idx)))
            if value is None:
                continue
            if decompress is not None:
                value = decompress(bytes(value))
            _n, pos, z, _cell, _pbc, energy, forces, _stress, _charges = unpack_arrays(value)
            _touch_molecule_arrays(pos, z, forces, int(idx))
            _ = energy
    finally:
        txn.abort()
        env.close()
    after = _memory_snapshot_kib()
    return _delta(before, after)


def _measure_stream_lmdb_pickle(path: Path, indices: list[int]) -> dict[str, int]:
    before = _memory_snapshot_kib()
    store = PickleAtomLMDB(
        path,
        PickleLMDBConfig(
            codec=DEFAULT_BENCHMARK_CODEC_LABEL,
            readahead=False,
        ),
    )
    env = store.open_env(map_size=1 << 30, readonly=True, lock=False)
    txn = env.begin(write=False)
    try:
        for idx in indices:
            payload = store.get_payload(txn, int(idx))
            if payload is None:
                continue
            _touch_molecule_arrays(
                payload["positions"],
                payload["atomic_numbers"],
                payload["forces"],
                int(idx),
            )
            _ = payload["energy"]
    finally:
        txn.abort()
        env.close()
    after = _memory_snapshot_kib()
    return _delta(before, after)


def _measure_stream_ase(path: Path, indices: list[int]) -> dict[str, int]:
    before = _memory_snapshot_kib()
    db = ase.db.connect(str(path))
    try:
        for idx in indices:
            row = db.get(id=int(idx) + 1)
            atoms = row.toatoms()
            forces = atoms.get_forces()
            _touch_molecule_arrays(
                np.asarray(atoms.positions),
                np.asarray(atoms.numbers, dtype=np.uint8),
                np.asarray(forces),
                int(idx),
            )
            _ = atoms.get_potential_energy()
    finally:
        _safe_close_ase_db(db)
    after = _memory_snapshot_kib()
    return _delta(before, after)


def _open_worker(
    backend: str,
    paths: dict[str, str | None],
    mode: str,
    touches: int,
    ready_q: mp.Queue,
    stop_evt: mp.Event,
) -> None:
    before = _memory_snapshot_kib()
    handles: dict[str, Any] = {}

    if backend == "atompack":
        db = atompack.Database.open(
            str(paths["atompack"]), mmap=True, populate=DEFAULT_BENCHMARK_POPULATE
        )
        if mode == "keep_open_touch":
            for idx in range(touches):
                mol = db.get_molecule(idx)
                _touch_molecule_arrays(mol.positions, mol.atomic_numbers, mol.forces, idx)
        handles["db"] = db
    elif backend == "lmdb_packed":
        env = lmdb.open(
            str(paths["lmdb_packed"]),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            max_dbs=1,
        )
        txn = env.begin(write=False)
        if mode == "keep_open_touch":
            _, decompress = lmdb_codec_fns(DEFAULT_BENCHMARK_CODEC_LABEL)
            for idx in range(touches):
                value = txn.get(struct.pack(">Q", idx))
                if value is None:
                    continue
                if decompress is not None:
                    value = decompress(bytes(value))
                _n, pos, z, _cell, _pbc, _energy, forces, _stress, _charges = unpack_arrays(value)
                _touch_molecule_arrays(pos, z, forces, idx)
        handles["env"] = env
        handles["txn"] = txn
    elif backend == "lmdb_pickle":
        store = PickleAtomLMDB(
            Path(str(paths["lmdb_pickle"])),
            PickleLMDBConfig(codec=DEFAULT_BENCHMARK_CODEC_LABEL, readahead=False),
        )
        env = store.open_env(map_size=1 << 30, readonly=True, lock=False)
        txn = env.begin(write=False)
        if mode == "keep_open_touch":
            for idx in range(touches):
                payload = store.get_payload(txn, idx)
                if payload is not None:
                    _touch_molecule_arrays(
                        payload["positions"],
                        payload["atomic_numbers"],
                        payload["forces"],
                        idx,
                    )
        handles["env"] = env
        handles["txn"] = txn
    elif backend in {"ase_sqlite", "ase_lmdb"}:
        path = paths[backend]
        if path is None:
            raise RuntimeError(f"Missing path for backend {backend}")
        db = ase.db.connect(str(path))
        if mode == "keep_open_touch":
            for idx in range(touches):
                row = db.get(id=idx + 1)
                atoms = row.toatoms()
                forces = atoms.get_forces()
                _touch_molecule_arrays(
                    np.asarray(atoms.positions),
                    np.asarray(atoms.numbers, dtype=np.uint8),
                    np.asarray(forces),
                    idx,
                )
        handles["db"] = db
    else:
        raise ValueError(f"Unknown backend: {backend}")

    after = _memory_snapshot_kib()
    ready_q.put((before, after))
    stop_evt.wait(timeout=20.0)

    try:
        if backend in {"lmdb_packed", "lmdb_pickle"}:
            handles["txn"].abort()
            handles["env"].close()
        elif backend in {"ase_sqlite", "ase_lmdb"}:
            _safe_close_ase_db(handles["db"])
    except Exception:
        pass


def _measure_open_memory(
    backend: str,
    paths: Paths,
    workers: int,
    mode: str,
    touches: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    ready_q: mp.Queue = ctx.Queue()
    stop_evt = ctx.Event()
    path_map = {
        "atompack": str(paths.atompack),
        "lmdb_packed": str(paths.lmdb_packed),
        "lmdb_pickle": str(paths.lmdb_pickle),
        "ase_sqlite": str(paths.ase_sqlite) if paths.ase_sqlite else None,
        "ase_lmdb": str(paths.ase_lmdb) if paths.ase_lmdb else None,
    }
    procs = [
        ctx.Process(
            target=_open_worker,
            args=(backend, path_map, mode, touches, ready_q, stop_evt),
        )
        for _ in range(workers)
    ]
    for proc in procs:
        proc.start()
    entries = [ready_q.get() for _ in range(workers)]
    stop_evt.set()
    for proc in procs:
        proc.join()

    metrics = (
        "rss_kib",
        "rss_anon_kib",
        "rss_file_kib",
        "rss_shmem_kib",
        "private_clean_kib",
        "private_dirty_kib",
        "private_total_kib",
    )
    out: dict[str, Any] = {"backend": backend, "workers": workers, "mode": mode}
    for metric in metrics:
        before_total = sum(int(x[0].get(metric, 0)) for x in entries)
        after_total = sum(int(x[1].get(metric, 0)) for x in entries)
        delta_total = after_total - before_total
        out[metric] = delta_total
        out[f"{metric}_per_worker"] = int(round(delta_total / max(1, workers)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory benchmark for current benchmark datasets")
    parser.add_argument("--atoms", type=int, default=256)
    parser.add_argument("--n-mols", type=int, default=None)
    parser.add_argument("--read-sample", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--touches-per-worker", type=int, default=DEFAULT_TOUCHES_PER_WORKER)
    parser.add_argument("--scratch-dir", type=str, default=str(DEFAULT_SCRATCH))
    parser.add_argument(
        "--create-missing",
        action="store_true",
        help="Create missing benchmark datasets instead of requiring existing benchmark.py outputs",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if lmdb is None:
        raise RuntimeError("lmdb is required for memory_benchmark.py")

    scratch = Path(args.scratch_dir)
    paths = _resolve_paths(
        scratch,
        args.atoms,
        args.n_mols,
        create_missing=args.create_missing,
    )
    sample = _read_sample(paths.n_mols, args.read_sample, args.atoms)

    print("=" * 80)
    print("MEMORY BENCHMARK")
    print("=" * 80)
    print(f"atompack:     {paths.atompack}")
    print(f"lmdb_packed:  {paths.lmdb_packed}")
    print(f"lmdb_pickle:  {paths.lmdb_pickle}")
    if _HAS_ASE:
        print(f"ase_sqlite:   {paths.ase_sqlite}")
        print(f"ase_lmdb:     {paths.ase_lmdb}")

    results: dict[str, Any] = {
        "meta": {
            "atoms": args.atoms,
            "n_mols": paths.n_mols,
            "read_sample": sample,
            "workers": args.workers,
            "touches_per_worker": args.touches_per_worker,
            "codec": DEFAULT_BENCHMARK_CODEC_LABEL,
        }
    }

    print("\n" + "-" * 80)
    print("1. Open connection residency")
    print("-" * 80)
    open_rows: list[dict[str, Any]] = []
    backends = ["atompack"]
    if paths.lmdb_packed is not None:
        backends.append("lmdb_packed")
    if paths.lmdb_pickle is not None:
        backends.append("lmdb_pickle")
    if _HAS_ASE and paths.ase_sqlite is not None:
        backends.append("ase_sqlite")
    if _HAS_ASE and paths.ase_lmdb is not None:
        backends.append("ase_lmdb")

    for backend in backends:
        row = _measure_open_memory(
            backend,
            paths,
            workers=args.workers,
            mode="keep_open_touch",
            touches=args.touches_per_worker,
        )
        open_rows.append(row)
        print(f"{backend:<12s} {_format_mem(row)}")
    results["open_connection_memory"] = open_rows

    print("\n" + "-" * 80)
    print("2. Streaming read working set")
    print("-" * 80)
    stream_rows: list[dict[str, Any]] = []

    indices = _make_indices(paths.n_mols, sample, seed=42, sequential=False)
    assert paths.atompack is not None
    row = {"backend": "atompack", **_measure_stream_atompack(paths.atompack, indices)}
    stream_rows.append(row)
    print(f"atompack     {_format_mem(row)}")

    if paths.lmdb_packed is not None:
        row = {
            "backend": "lmdb_packed",
            **_measure_stream_lmdb_packed(paths.lmdb_packed, indices),
        }
        stream_rows.append(row)
        print(f"lmdb_packed  {_format_mem(row)}")
    else:
        print("lmdb_packed  SKIP (missing benchmark.py output)")

    if paths.lmdb_pickle is not None:
        row = {
            "backend": "lmdb_pickle",
            **_measure_stream_lmdb_pickle(paths.lmdb_pickle, indices),
        }
        stream_rows.append(row)
        print(f"lmdb_pickle  {_format_mem(row)}")
    else:
        print("lmdb_pickle  SKIP (missing benchmark.py output)")

    if _HAS_ASE and paths.n_ase > 0:
        ase_sample = _ase_read_sample(paths.n_ase, sample)
        ase_indices = _make_indices(paths.n_ase, ase_sample, seed=42, sequential=False)
        if paths.ase_sqlite is not None:
            row = {
                "backend": "ase_sqlite",
                **_measure_stream_ase(paths.ase_sqlite, ase_indices),
            }
            stream_rows.append(row)
            print(f"ase_sqlite   {_format_mem(row)}")
        if paths.ase_lmdb is not None:
            row = {
                "backend": "ase_lmdb",
                **_measure_stream_ase(paths.ase_lmdb, ase_indices),
            }
            stream_rows.append(row)
            print(f"ase_lmdb     {_format_mem(row)}")
    results["streaming_read_memory"] = stream_rows

    out_path = (
        Path(args.out)
        if args.out is not None
        else Path(__file__).with_name("memory_benchmark_results.json")
    )
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
