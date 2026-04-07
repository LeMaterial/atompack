# Copyright 2026 Entalpic
"""Atompack best-case read benchmark.

This script is intentionally narrower than ``benchmark.py`` and reuses the same
cached synthetic datasets under the same scratch directory layout.

It focuses on Atompack's best Python-visible read path:

- contiguous sequential batch reads
- native or fixed Rayon thread counts
- large batch-oriented APIs, with ``get_molecules_flat(...)`` as the default

Unlike ``benchmark.py``, this benchmark is not trying to be backend-fair. It is
meant to answer "what is Atompack's best-case read throughput when Rayon can
actually help?".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_RAYON_THREADS = os.environ.get("ATOMPACK_BESTCASE_RAYON_THREADS")
if _RAYON_THREADS:
    os.environ["RAYON_NUM_THREADS"] = _RAYON_THREADS

import atompack

from benchmark import (
    DEFAULT_BENCHMARK_ATP_COMPRESSION,
    DEFAULT_BENCHMARK_ATP_LEVEL,
    DEFAULT_BENCHMARK_CODEC_LABEL,
    DEFAULT_SCRATCH,
    _n_mols_for_atoms,
    _read_sample,
    _result_with_atoms_s,
    bench,
    ensure_datasets,
)

DEFAULT_ATOMS = [64, 256, 512]
DEFAULT_TARGET_ATOMS_PER_BATCH = [32_768, 131_072, 524_288]
DEFAULT_THREADS = ["native", "1", "4", "8"]
DEFAULT_METHODS = ["flat"]
DEFAULT_POPULATE = False
NON_ATOMPACK_BACKENDS = frozenset(
    {"hdf5_soa", "lmdb_soa", "lmdb_packed", "lmdb_pickle"}
)


def _codec_settings(compression: str, level: int) -> tuple[str, str, int]:
    if compression == "none":
        return DEFAULT_BENCHMARK_CODEC_LABEL, DEFAULT_BENCHMARK_ATP_COMPRESSION, DEFAULT_BENCHMARK_ATP_LEVEL
    if compression == "lz4":
        return "lz4", "lz4", 0
    return f"zstd:{level}", "zstd", level


def ensure_atompack_dataset(
    scratch: Path,
    atoms: int,
    n_mols: int,
    compression: str,
    level: int,
) -> Path:
    codec_label, atp_comp, atp_level = _codec_settings(compression, level)
    paths = ensure_datasets(
        scratch,
        atoms,
        n_mols,
        codec_label,
        atp_comp,
        atp_level,
        with_custom=False,
        skip_lmdb_packed=True,
        skip_backends=NON_ATOMPACK_BACKENDS,
    )
    return Path(paths["atompack"])


def _chunks(sample: int, batch_size: int):
    for start in range(0, sample, batch_size):
        yield list(range(start, min(start + batch_size, sample)))


def _floor_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _adaptive_batch_sizes(
    atoms: int,
    sample: int,
    target_atoms_per_batch: list[int],
) -> list[int]:
    sizes = []
    for target_atoms in target_atoms_per_batch:
        batch_size = max(1, target_atoms // atoms)
        batch_size = min(sample, _floor_pow2(batch_size))
        sizes.append(batch_size)
    return sorted(set(sizes))


def _touch_molecule(mol: Any, seed: int) -> None:
    pos = mol.positions
    forces = mol.forces
    _ = float(pos[0, seed % 3])
    _ = float(pos[-1, seed % 3])
    _ = float(forces[0, seed % 3])
    _ = float(forces[-1, seed % 3])
    _ = float(mol.energy)


def _touch_flat_batch(flat: Any, seed: int) -> None:
    n_atoms = flat["n_atoms"]
    if len(n_atoms) == 0:
        return
    positions = flat["positions"]
    forces = flat["forces"]
    energy = flat["energy"]
    _ = int(n_atoms[0])
    _ = int(n_atoms[-1])
    _ = float(positions[0, seed % 3])
    _ = float(positions[-1, seed % 3])
    _ = float(forces[0, seed % 3])
    _ = float(forces[-1, seed % 3])
    _ = float(energy[0])
    _ = float(energy[-1])


def _run_batch(db: Any, sample: int, batch_size: int) -> None:
    for chunk in _chunks(sample, batch_size):
        mols = db.get_molecules(chunk)
        for seed, mol in zip(chunk, mols):
            _touch_molecule(mol, seed)


def _run_flat(db: Any, sample: int, batch_size: int) -> None:
    for chunk in _chunks(sample, batch_size):
        flat = db.get_molecules_flat(chunk)
        _touch_flat_batch(flat, chunk[0])


def _worker_main(args: argparse.Namespace) -> int:
    db = atompack.Database.open(args.db_path, mmap=True, populate=args.populate)
    methods = {
        "batch": _run_batch,
        "flat": _run_flat,
    }
    results: list[dict[str, Any]] = []

    for batch_size in args.batch_sizes:
        for method_name in args.methods:
            fn = methods[method_name]
            stats = bench(
                lambda fn=fn, bs=batch_size: fn(db, args.sample, bs),
                args.sample,
                trials=args.trials,
            )
            results.append(
                {
                    "benchmark": "atompack_bestcase_read",
                    "method": method_name,
                    "batch_size": batch_size,
                    "threads": args.threads_label,
                    "access_pattern": "sequential",
                    **stats,
                }
            )

    print(json.dumps(results))
    return 0


def _spawn_worker(
    *,
    db_path: Path,
    sample: int,
    batch_sizes: list[int],
    methods: list[str],
    trials: int,
    threads_label: str,
    populate: bool,
) -> list[dict[str, Any]]:
    env = os.environ.copy()
    if threads_label == "native":
        env.pop("ATOMPACK_BESTCASE_RAYON_THREADS", None)
        env.pop("RAYON_NUM_THREADS", None)
    else:
        env["ATOMPACK_BESTCASE_RAYON_THREADS"] = threads_label

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--db-path",
        str(db_path),
        "--sample",
        str(sample),
        "--trials",
        str(trials),
        "--threads-label",
        threads_label,
        "--batch-sizes",
        *[str(x) for x in batch_sizes],
        "--methods",
        *methods,
    ]
    if populate:
        cmd.append("--populate")

    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(Path(__file__).resolve().parent),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "worker failed")
    return json.loads(proc.stdout)


def _thread_sort_key(label: str) -> tuple[int, int | str]:
    if label == "native":
        return (0, label)
    return (1, int(label))


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_atoms: dict[int, dict[str, Any]] = {}
    for row in rows:
        atoms = int(row["atoms"])
        current = best_by_atoms.get(atoms)
        if current is None or float(row["atoms_s"]) > float(current["atoms_s"]):
            best_by_atoms[atoms] = row
    return [best_by_atoms[atoms] for atoms in sorted(best_by_atoms)]


def _print_table(
    atoms: int,
    n_mols: int,
    sample: int,
    batch_sizes: list[int],
    rows: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 84)
    print(f"{atoms} atoms, {n_mols:,} molecules, sequential sample {sample:,}")
    print("batch sizes: " + ", ".join(f"{size:,}" for size in batch_sizes))
    print("=" * 84)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["threads"]), []).append(row)

    for threads in sorted(grouped, key=_thread_sort_key):
        print(f"\nthreads={threads}")
        print(
            f"  {'method':<8} {'batch':>8} {'mol/s':>12} {'atoms/s':>14} {'µs/mol':>10} {'ci95':>12}"
        )
        for row in sorted(grouped[threads], key=lambda r: (r["method"], r["batch_size"])):
            rate = float(row["mol_s"])
            print(
                f"  {row['method']:<8} {row['batch_size']:>8,} "
                f"{rate:>12,.0f} {row['atoms_s']:>14,.0f} {1e6/rate:>10.1f} {row['ci95_mol_s']:>12,.0f}"
            )


def _print_best_summary(rows: list[dict[str, Any]]) -> None:
    print("\n" + "-" * 84)
    print("Best observed throughput per atom count")
    print("-" * 84)
    for row in _best_rows(rows):
        print(
            f"  atoms={row['atoms']:>4}  method={row['method']:<5}  threads={row['threads']:<6} "
            f"batch={row['batch_size']:>6,}  {row['atoms_s']:>14,.0f} atoms/s"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--db-path", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--threads-label", type=str, default="native", help=argparse.SUPPRESS)
    parser.add_argument("--populate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--atoms", type=int, nargs="+", default=DEFAULT_ATOMS)
    parser.add_argument("--n-mols", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--target-atoms-per-batch",
        type=int,
        nargs="+",
        default=DEFAULT_TARGET_ATOMS_PER_BATCH,
    )
    parser.add_argument("--threads", nargs="+", default=DEFAULT_THREADS)
    parser.add_argument("--methods", nargs="+", choices=["batch", "flat"], default=DEFAULT_METHODS)
    parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--compression", type=str, default="none", choices=["none", "lz4", "zstd"])
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.worker:
        return _worker_main(args)

    all_results: list[dict[str, Any]] = []

    for atoms in args.atoms:
        n_mols = _n_mols_for_atoms(atoms, args.n_mols)
        sample = _read_sample(n_mols, args.sample, atoms)
        batch_sizes = (
            sorted(set(min(sample, size) for size in args.batch_sizes))
            if args.batch_sizes is not None
            else _adaptive_batch_sizes(atoms, sample, args.target_atoms_per_batch)
        )
        db_path = ensure_atompack_dataset(
            args.scratch_dir,
            atoms,
            n_mols,
            args.compression,
            args.level,
        )

        rows: list[dict[str, Any]] = []
        for threads in args.threads:
            worker_rows = _spawn_worker(
                db_path=db_path,
                sample=sample,
                batch_sizes=batch_sizes,
                methods=args.methods,
                trials=args.trials,
                threads_label=threads,
                populate=DEFAULT_POPULATE,
            )
            for row in worker_rows:
                row.update(
                    _result_with_atoms_s(
                        atoms,
                        atoms=atoms,
                        n_mols=n_mols,
                        sample=sample,
                        compression=args.compression,
                        level=args.level if args.compression == "zstd" else 0,
                        dataset_path=str(db_path),
                        **row,
                    )
                )
            rows.extend(worker_rows)
            all_results.extend(worker_rows)

        _print_table(atoms, n_mols, sample, batch_sizes, rows)

    _print_best_summary(all_results)

    if args.out is not None:
        payload = {
            "benchmark": "atompack_bestcase_read",
            "results": all_results,
            "best": _best_rows(all_results),
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved results to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
