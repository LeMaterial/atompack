# Copyright 2026 Entalpic
from __future__ import annotations

import argparse
import json
import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - handled at runtime
    h5py = None  # type: ignore[assignment]

from atom_hdf5_soa import AtomHdf5Soa, AtomHdf5SoaConfig, hdf5_available


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    n_mols: int = 200_000
    atoms: int = 64
    chunk_size: int = 256
    batch_size: int = 256
    sequential_sample: int = 50_000
    random_sample: int = 50_000
    write_trials: int = 1
    read_trials: int = 3
    seed: int = 42
    rdcc_nbytes_mib: int = 128
    rdcc_nslots: int = 1_000_003
    cache_chunks: int = 8


@dataclass(frozen=True)
class BenchmarkRow:
    benchmark: str
    reader: str
    n_mols: int
    atoms: int
    chunk_size: int
    batch_size: int
    sample: int
    mol_s: float
    median_mol_s: float
    mean_mol_s: float
    std_mol_s: float
    ci95_mol_s: float
    trials_mol_s: list[float]
    size_bytes: int


def _require_hdf5() -> None:
    if not hdf5_available() or h5py is None:
        raise RuntimeError("h5py is required for hdf5_tuned_experiment.py")


def _store_cfg(cfg: ExperimentConfig) -> AtomHdf5SoaConfig:
    return AtomHdf5SoaConfig(
        atoms_per_molecule=cfg.atoms,
        with_props=True,
        with_cell_pbc=True,
        compression="none",
        chunk_size=cfg.chunk_size,
    )


def _touch_payload(payload: dict[str, Any], seed: int) -> None:
    positions = payload["positions"]
    forces = payload["forces"]
    _ = float(positions[seed % len(positions), seed % 3])
    _ = float(forces[seed % len(forces), seed % 3])
    _ = float(payload["energy"])


def _bench(fn: Any, units: int, trials: int) -> tuple[float, list[float]]:
    rates: list[float] = []
    for _ in range(max(1, trials)):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        rates.append(units / dt)
    return float(np.median(rates)), rates


def _stats(
    benchmark: str,
    reader: str,
    cfg: ExperimentConfig,
    sample: int,
    rates: list[float],
) -> BenchmarkRow:
    mean = float(np.mean(rates))
    std = float(np.std(rates))
    ci95 = 1.96 * std / math.sqrt(len(rates)) if len(rates) > 1 else 0.0
    return BenchmarkRow(
        benchmark=benchmark,
        reader=reader,
        n_mols=cfg.n_mols,
        atoms=cfg.atoms,
        chunk_size=cfg.chunk_size,
        batch_size=cfg.batch_size,
        sample=sample,
        mol_s=float(np.median(rates)),
        median_mol_s=float(np.median(rates)),
        mean_mol_s=mean,
        std_mol_s=std,
        ci95_mol_s=ci95,
        trials_mol_s=[float(rate) for rate in rates],
        size_bytes=cfg.path.stat().st_size if cfg.path.exists() else 0,
    )


def write_dataset(cfg: ExperimentConfig) -> BenchmarkRow:
    _require_hdf5()
    cfg_store = _store_cfg(cfg)
    store = AtomHdf5Soa(cfg.path, cfg_store)

    def run_once() -> None:
        store.create_file(cfg.n_mols)
        rng = np.random.RandomState(cfg.seed)
        cell = np.eye(3, dtype=np.float64) * 10.0
        pbc = np.array([1, 1, 1], dtype=np.uint8)
        with store.open_file("r+") as handle:
            written = 0
            while written < cfg.n_mols:
                batch_n = min(cfg.chunk_size, cfg.n_mols - written)
                positions = rng.randn(batch_n, cfg.atoms, 3).astype(np.float32)
                atomic_numbers = rng.randint(1, 80, size=(batch_n, cfg.atoms)).astype(np.uint8)
                energy = rng.randn(batch_n).astype(np.float64)
                forces = rng.randn(batch_n, cfg.atoms, 3).astype(np.float32)
                cell_batch = np.broadcast_to(cell, (batch_n, 3, 3)).copy()
                pbc_batch = np.broadcast_to(pbc, (batch_n, 3)).copy()
                store.put_batch(
                    handle,
                    written,
                    positions,
                    atomic_numbers,
                    cell_batch,
                    pbc_batch,
                    energy,
                    forces,
                )
                written += batch_n

    _median, rates = _bench(run_once, cfg.n_mols, cfg.write_trials)
    return _stats("write", "hdf5_chunked_layout", cfg, cfg.n_mols, rates)


class ChunkedHdf5Reader:
    def __init__(
        self,
        path: Path,
        *,
        chunk_size: int,
        cache_chunks: int = 8,
        rdcc_nbytes_mib: int = 128,
        rdcc_nslots: int = 1_000_003,
    ) -> None:
        _require_hdf5()
        self.path = Path(path)
        self.chunk_size = int(chunk_size)
        self.cache_chunks = max(0, int(cache_chunks))
        self.file = h5py.File(
            self.path,
            "r",
            rdcc_nbytes=int(rdcc_nbytes_mib) * 1024 * 1024,
            rdcc_nslots=int(rdcc_nslots),
        )
        self.meta = {str(key): self.file.attrs[key] for key in self.file.attrs.keys()}
        self.positions = self.file["positions"]
        self.atomic_numbers = self.file["atomic_numbers"]
        self.energy = self.file["energy"]
        self.forces = self.file["forces"]
        self.cell = self.file["cell"]
        self.pbc = self.file["pbc"]
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def close(self) -> None:
        self.file.close()

    def _validate_idx(self, idx: int) -> None:
        total = int(self.meta["num_molecules"])
        if idx < 0 or idx >= total:
            raise IndexError(f"Index {idx} out of bounds for HDF5 SOA of length {total}")

    def _remember_chunk(self, chunk_id: int, chunk: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.cache_chunks <= 0:
            return chunk
        self._cache[chunk_id] = chunk
        self._cache.move_to_end(chunk_id)
        while len(self._cache) > self.cache_chunks:
            self._cache.popitem(last=False)
        return chunk

    def _load_chunk(self, chunk_id: int) -> dict[str, np.ndarray]:
        cached = self._cache.get(chunk_id)
        if cached is not None:
            self._cache.move_to_end(chunk_id)
            return cached

        total = int(self.meta["num_molecules"])
        start = chunk_id * self.chunk_size
        stop = min(total, start + self.chunk_size)
        if start >= total:
            raise IndexError(f"Chunk {chunk_id} out of range")

        chunk = {
            "positions": self.positions[start:stop],
            "atomic_numbers": self.atomic_numbers[start:stop],
            "energy": self.energy[start:stop],
            "forces": self.forces[start:stop],
            "cell": self.cell[start:stop],
            "pbc": self.pbc[start:stop],
        }
        return self._remember_chunk(chunk_id, chunk)

    def get_payloads_chunked(self, idxs: list[int]) -> list[dict[str, Any]]:
        grouped: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()
        payloads: list[dict[str, Any] | None] = [None] * len(idxs)
        for out_pos, raw_idx in enumerate(idxs):
            idx = int(raw_idx)
            self._validate_idx(idx)
            chunk_id, local = divmod(idx, self.chunk_size)
            grouped.setdefault(chunk_id, []).append((out_pos, local))

        for chunk_id, group in grouped.items():
            chunk = self._load_chunk(chunk_id)
            for out_pos, local in group:
                payloads[out_pos] = {
                    "positions": chunk["positions"][local],
                    "atomic_numbers": chunk["atomic_numbers"][local],
                    "energy": float(chunk["energy"][local]),
                    "forces": chunk["forces"][local],
                    "cell": chunk["cell"][local],
                    "pbc": chunk["pbc"][local].astype(bool, copy=False),
                }

        if any(payload is None for payload in payloads):
            raise RuntimeError("Missing payload while reconstructing chunked HDF5 batch")
        return [payload for payload in payloads if payload is not None]


def _random_indices(cfg: ExperimentConfig, sample: int) -> list[int]:
    rng = np.random.default_rng(cfg.seed + 1)
    return [int(i) for i in rng.choice(cfg.n_mols, size=min(sample, cfg.n_mols), replace=False)]


def _chunks(indices: list[int], batch_size: int) -> list[list[int]]:
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


def bench_naive_sequential(cfg: ExperimentConfig) -> BenchmarkRow:
    store = AtomHdf5Soa(cfg.path, _store_cfg(cfg))
    sample = min(cfg.sequential_sample, cfg.n_mols)
    indices = list(range(sample))

    def run_once() -> None:
        reader = store.open_reader()
        try:
            for idx in indices:
                payload = store.get_payload(reader, idx)
                _touch_payload(payload, idx)
        finally:
            reader.close()

    _median, rates = _bench(run_once, sample, cfg.read_trials)
    return _stats("sequential", "naive_single", cfg, sample, rates)


def bench_chunked_sequential(cfg: ExperimentConfig) -> BenchmarkRow:
    sample = min(cfg.sequential_sample, cfg.n_mols)
    batches = _chunks(list(range(sample)), cfg.batch_size)

    def run_once() -> None:
        reader = ChunkedHdf5Reader(
            cfg.path,
            chunk_size=cfg.chunk_size,
            cache_chunks=cfg.cache_chunks,
            rdcc_nbytes_mib=cfg.rdcc_nbytes_mib,
            rdcc_nslots=cfg.rdcc_nslots,
        )
        try:
            for batch in batches:
                payloads = reader.get_payloads_chunked(batch)
                for idx, payload in zip(batch, payloads):
                    _touch_payload(payload, idx)
        finally:
            reader.close()

    _median, rates = _bench(run_once, sample, cfg.read_trials)
    return _stats("sequential", "chunked_batch", cfg, sample, rates)


def bench_naive_random(cfg: ExperimentConfig) -> BenchmarkRow:
    store = AtomHdf5Soa(cfg.path, _store_cfg(cfg))
    sample = min(cfg.random_sample, cfg.n_mols)
    indices = _random_indices(cfg, sample)

    def run_once() -> None:
        reader = store.open_reader()
        try:
            for idx in indices:
                payload = store.get_payload(reader, idx)
                _touch_payload(payload, idx)
        finally:
            reader.close()

    _median, rates = _bench(run_once, sample, cfg.read_trials)
    return _stats("random", "naive_single", cfg, sample, rates)


def bench_chunked_random(cfg: ExperimentConfig) -> BenchmarkRow:
    sample = min(cfg.random_sample, cfg.n_mols)
    batches = _chunks(_random_indices(cfg, sample), cfg.batch_size)

    def run_once() -> None:
        reader = ChunkedHdf5Reader(
            cfg.path,
            chunk_size=cfg.chunk_size,
            cache_chunks=cfg.cache_chunks,
            rdcc_nbytes_mib=cfg.rdcc_nbytes_mib,
            rdcc_nslots=cfg.rdcc_nslots,
        )
        try:
            for batch in batches:
                payloads = reader.get_payloads_chunked(batch)
                for idx, payload in zip(batch, payloads):
                    _touch_payload(payload, idx)
        finally:
            reader.close()

    _median, rates = _bench(run_once, sample, cfg.read_trials)
    return _stats("random", "chunked_batch", cfg, sample, rates)


def run_experiment(cfg: ExperimentConfig) -> list[BenchmarkRow]:
    rows = [
        write_dataset(cfg),
        bench_naive_sequential(cfg),
        bench_chunked_sequential(cfg),
        bench_naive_random(cfg),
        bench_chunked_random(cfg),
    ]
    return rows


def _print_rows(rows: list[BenchmarkRow]) -> None:
    print(
        f"{'benchmark':>12}  {'reader':>16}  {'mol/s':>12}  "
        f"{'chunk':>7}  {'batch':>7}  {'sample':>8}  {'size GB':>8}"
    )
    for row in rows:
        print(
            f"{row.benchmark:>12}  {row.reader:>16}  {row.mol_s:>12,.0f}  "
            f"{row.chunk_size:>7}  {row.batch_size:>7}  {row.sample:>8,}  "
            f"{row.size_bytes / 1e9:>8.2f}"
        )


def _parse_args() -> tuple[ExperimentConfig, Path | None]:
    parser = argparse.ArgumentParser(
        description="Standalone tuned HDF5 experiment for write/sequential/random read throughput."
    )
    parser.add_argument("--path", type=Path, default=Path("/tmp/atompack_hdf5_tuned_experiment.h5"))
    parser.add_argument("--n-mols", type=int, default=200_000)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequential-sample", type=int, default=50_000)
    parser.add_argument("--random-sample", type=int, default=50_000)
    parser.add_argument("--write-trials", type=int, default=1)
    parser.add_argument("--read-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rdcc-nbytes-mib", type=int, default=128)
    parser.add_argument("--rdcc-nslots", type=int, default=1_000_003)
    parser.add_argument("--cache-chunks", type=int, default=8)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    return (
        ExperimentConfig(
            path=args.path,
            n_mols=args.n_mols,
            atoms=args.atoms,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            sequential_sample=args.sequential_sample,
            random_sample=args.random_sample,
            write_trials=args.write_trials,
            read_trials=args.read_trials,
            seed=args.seed,
            rdcc_nbytes_mib=args.rdcc_nbytes_mib,
            rdcc_nslots=args.rdcc_nslots,
            cache_chunks=args.cache_chunks,
        ),
        args.json_out,
    )


def main() -> None:
    cfg, json_out = _parse_args()
    rows = run_experiment(cfg)
    _print_rows(rows)

    if json_out is not None:
        json_out.write_text(
            json.dumps({"results": [asdict(row) for row in rows]}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
