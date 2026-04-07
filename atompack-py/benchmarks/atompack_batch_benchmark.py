# Copyright 2026 Entalpic
"""Atompack-only batch-read benchmark.

This script is intentionally narrower than benchmark.py. It focuses on the
three Atompack Python read paths:

- `get_molecule(i)` in chunked loops
- `get_molecules(indices)` with configurable batch sizes
- `get_molecules_flat(indices)` with configurable batch sizes

Unlike the broad backend suite, this benchmark is designed to:

- use Atompack only
- benchmark batch sizes explicitly
- allow native threading by default
- run each thread-count setting in an isolated subprocess so Rayon settings
  actually take effect
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_RAYON_THREADS = os.environ.get("ATOMPACK_BATCH_RAYON_THREADS")
if _RAYON_THREADS:
    os.environ["RAYON_NUM_THREADS"] = _RAYON_THREADS

import atompack

from benchmark import _n_mols_for_atoms, _read_sample, bench, create_atompack_db

DEFAULT_ATOMS = [64, 256, 512]
DEFAULT_BATCH_SIZES = [32, 128, 512, 2048]
DEFAULT_THREADS = ["native", "1", "4", "8"]
DEFAULT_CODEC = "none"
DEFAULT_LEVEL = 0
DEFAULT_POPULATE = False


def _dataset_tag(atoms: int, n_mols: int, codec: str) -> str:
    return f"batch_api_a{atoms}_n{n_mols}_{codec}"


def _is_complete_atompack(path: Path, expected_count: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        db = atompack.Database.open(str(path), mmap=True)
        ok = len(db) == expected_count
        del db
        return ok
    except Exception:
        return False


def ensure_atompack_dataset(
    scratch: Path,
    atoms: int,
    n_mols: int,
    compression: str,
    level: int,
) -> Path:
    base = scratch / _dataset_tag(atoms, n_mols, compression)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "data.atp"
    if _is_complete_atompack(path, n_mols):
        print(f"  Using cached Atompack dataset: {base.name}")
        return path

    print(
        f"  Creating Atompack dataset: {n_mols:,} molecules "
        f"({atoms} atoms, {compression})..."
    )
    create_atompack_db(
        str(path),
        n_mols,
        atoms,
        compression=compression,
        level=level,
        with_custom=False,
    )
    return path


def _chunks(indices: list[int], batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def _touch_molecule(mol: Any, seed: int) -> None:
    pos = mol.positions
    forces = mol.forces
    _ = float(pos[seed % len(pos), seed % 3])
    _ = float(forces[seed % len(forces), seed % 3])
    _ = mol.energy


def _run_loop_chunked(db: Any, indices: list[int], batch_size: int) -> None:
    for chunk in _chunks(indices, batch_size):
        for i in chunk:
            _touch_molecule(db.get_molecule(i), i)


def _run_get_molecules(db: Any, indices: list[int], batch_size: int) -> None:
    for chunk in _chunks(indices, batch_size):
        mols = db.get_molecules(chunk)
        for i, mol in zip(chunk, mols):
            _touch_molecule(mol, i)


def _run_get_molecules_flat(db: Any, indices: list[int], batch_size: int) -> None:
    for chunk in _chunks(indices, batch_size):
        flat = db.get_molecules_flat(chunk)
        positions = flat["positions"]
        forces = flat["forces"]
        energy = flat["energy"]
        n_atoms = flat["n_atoms"]
        offset = 0
        for local_i, (seed, n_atoms_i) in enumerate(zip(chunk, n_atoms)):
            n_atoms_i = int(n_atoms_i)
            _ = float(positions[offset, seed % 3])
            _ = float(forces[offset, seed % 3])
            _ = float(energy[local_i])
            offset += n_atoms_i


def _worker_main(args: argparse.Namespace) -> int:
    indices = json.loads(Path(args.indices_file).read_text())
    db = atompack.Database.open(args.db_path, mmap=True, populate=args.populate)

    methods = {
        "loop": _run_loop_chunked,
        "batch": _run_get_molecules,
        "flat": _run_get_molecules_flat,
    }
    results: list[dict[str, Any]] = []

    for batch_size in args.batch_sizes:
        for method_name, fn in methods.items():
            stats = bench(
                lambda fn=fn, bs=batch_size: fn(db, indices, bs),
                len(indices),
                trials=args.trials,
            )
            results.append(
                {
                    "method": method_name,
                    "batch_size": batch_size,
                    "threads": args.threads_label,
                    **stats,
                }
            )

    print(json.dumps(results))
    return 0


def _spawn_worker(
    db_path: Path,
    indices_file: Path,
    batch_sizes: list[int],
    trials: int,
    threads_label: str,
    populate: bool,
) -> list[dict[str, Any]]:
    env = os.environ.copy()
    if threads_label == "native":
        env.pop("ATOMPACK_BATCH_RAYON_THREADS", None)
        env.pop("RAYON_NUM_THREADS", None)
    else:
        env["ATOMPACK_BATCH_RAYON_THREADS"] = threads_label

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--db-path",
        str(db_path),
        "--indices-file",
        str(indices_file),
        "--trials",
        str(trials),
        "--threads-label",
        threads_label,
        "--batch-sizes",
        *[str(x) for x in batch_sizes],
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


def _print_table(atoms: int, n_mols: int, sample: int, rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print(f"{atoms} atoms, {n_mols:,} molecules, sample {sample:,}")
    print("=" * 72)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["threads"], []).append(row)

    for threads in grouped:
        print(f"\nthreads={threads}")
        print(f"  {'method':<8} {'batch':>7} {'mol/s':>12} {'µs/mol':>10} {'ci95':>12}")
        for row in sorted(grouped[threads], key=lambda r: (r["method"], r["batch_size"])):
            rate = row["mol_s"]
            print(
                f"  {row['method']:<8} {row['batch_size']:>7,} "
                f"{rate:>12,.0f} {1e6/rate:>10.1f} {row['ci95_mol_s']:>12,.0f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--db-path", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--indices-file", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--threads-label", type=str, default="native", help=argparse.SUPPRESS)
    parser.add_argument("--populate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--atoms", type=int, nargs="+", default=DEFAULT_ATOMS)
    parser.add_argument("--n-mols", type=int, default=None)
    parser.add_argument("--read-sample", type=int, default=None)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--threads", nargs="+", default=DEFAULT_THREADS)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/ogre/atompack-v2/benchmarks"))
    parser.add_argument("--compression", type=str, default=DEFAULT_CODEC, choices=["none", "lz4", "zstd"])
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.worker:
        return _worker_main(args)

    all_results: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)

    for atoms in args.atoms:
        n_mols = _n_mols_for_atoms(atoms, args.n_mols)
        sample = _read_sample(n_mols, args.read_sample)
        db_path = ensure_atompack_dataset(
            args.scratch_dir,
            atoms,
            n_mols,
            args.compression,
            args.level,
        )

        indices = rng.choice(n_mols, size=min(sample, n_mols), replace=False).tolist()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            Path(tmp.name).write_text(json.dumps(indices))
            indices_file = Path(tmp.name)

        try:
            rows: list[dict[str, Any]] = []
            for threads in args.threads:
                worker_rows = _spawn_worker(
                    db_path=db_path,
                    indices_file=indices_file,
                    batch_sizes=args.batch_sizes,
                    trials=args.trials,
                    threads_label=threads,
                    populate=DEFAULT_POPULATE,
                )
                for row in worker_rows:
                    row.update(
                        {
                            "atoms": atoms,
                            "n_mols": n_mols,
                            "sample": len(indices),
                            "compression": args.compression,
                            "benchmark": "atompack_batch_api",
                        }
                    )
                rows.extend(worker_rows)
                all_results.extend(worker_rows)
            _print_table(atoms, n_mols, len(indices), rows)
        finally:
            indices_file.unlink(missing_ok=True)

    if args.out is not None:
        args.out.write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved results to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
