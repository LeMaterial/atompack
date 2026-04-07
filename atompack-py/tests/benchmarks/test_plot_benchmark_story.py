# Copyright 2026 Entalpic
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _benchmark_payload(scale: float = 1.0) -> dict[str, list[dict]]:
    def row(benchmark: str, backend: str, *, atoms: int, mol_s: float, workers: int | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "benchmark": benchmark,
            "backend": backend,
            "atoms": atoms,
            "mol_s": mol_s * scale,
            "atoms_s": mol_s * atoms * scale,
            "ci95_atoms_s": 0.0,
            "n_mols": 1000,
            "sample": 100,
        }
        if workers is not None:
            payload["workers"] = workers
        return payload

    rows = [
        row("sequential", "atompack", atoms=16, mol_s=1800),
        row("sequential", "lmdb_packed", atoms=16, mol_s=900),
        row("sequential", "lmdb_pickle", atoms=16, mol_s=700),
        row("sequential", "hdf5_soa", atoms=16, mol_s=350),
        row("sequential", "ase_lmdb", atoms=16, mol_s=50),
        row("sequential", "ase_sqlite", atoms=16, mol_s=20),
        row("sequential", "atompack", atoms=64, mol_s=1200),
        row("sequential", "lmdb_packed", atoms=64, mol_s=500),
        row("sequential", "lmdb_pickle", atoms=64, mol_s=360),
        row("sequential", "hdf5_soa", atoms=64, mol_s=120),
        row("sequential", "ase_lmdb", atoms=64, mol_s=20),
        row("sequential", "ase_sqlite", atoms=64, mol_s=8),
    ]
    for workers, rates in {
        1: {"atompack": 900, "lmdb_packed": 360, "lmdb_pickle": 240, "hdf5_soa": 110, "ase_lmdb": 15, "ase_sqlite": 3},
        2: {"atompack": 1500, "lmdb_packed": 600, "lmdb_pickle": 430, "hdf5_soa": 125, "ase_lmdb": 18, "ase_sqlite": 4},
        4: {"atompack": 2200, "lmdb_packed": 980, "lmdb_pickle": 700, "hdf5_soa": 130, "ase_lmdb": 22, "ase_sqlite": 5},
    }.items():
        for backend, mol_s in rates.items():
            rows.append(row("multiprocessing", backend, atoms=16, workers=workers, mol_s=mol_s))
    for workers, rates in {
        1: {"atompack": 500, "lmdb_packed": 220, "lmdb_pickle": 150, "hdf5_soa": 60, "ase_lmdb": 9, "ase_sqlite": 3},
        2: {"atompack": 850, "lmdb_packed": 360, "lmdb_pickle": 250, "hdf5_soa": 62, "ase_lmdb": 10, "ase_sqlite": 4},
        4: {"atompack": 1200, "lmdb_packed": 700, "lmdb_pickle": 430, "hdf5_soa": 64, "ase_lmdb": 12, "ase_sqlite": 5},
    }.items():
        for backend, mol_s in rates.items():
            rows.append(row("multiprocessing", backend, atoms=64, workers=workers, mol_s=mol_s))
    return {"results": rows}


def _write_payload(scale: float = 1.0) -> list[dict]:
    rows = []
    for with_custom, rates_by_atoms in {
        False: {
            16: {"atompack": 240000, "atompack_ase_batch": 180000, "lmdb_packed": 140000, "lmdb_pickle": 90000, "ase_lmdb": 12000, "ase_sqlite": 5000},
            64: {"atompack": 170000, "atompack_ase_batch": 130000, "lmdb_packed": 90000, "lmdb_pickle": 52000, "ase_lmdb": 7000, "ase_sqlite": 2600},
            256: {"atompack": 95000, "atompack_ase_batch": 76000, "lmdb_packed": 41000, "lmdb_pickle": 20000, "ase_lmdb": 2500, "ase_sqlite": 800},
        },
        True: {
            16: {"atompack": 200000, "atompack_ase_batch": 145000, "lmdb_pickle": 70000, "ase_lmdb": 9000, "ase_sqlite": 3500},
            64: {"atompack": 140000, "atompack_ase_batch": 98000, "lmdb_pickle": 39000, "ase_lmdb": 5000, "ase_sqlite": 1800},
            256: {"atompack": 76000, "atompack_ase_batch": 56000, "lmdb_pickle": 15000, "ase_lmdb": 1800, "ase_sqlite": 600},
        },
    }.items():
        for atoms, rates in rates_by_atoms.items():
            for backend, mol_s in rates.items():
                rows.append(
                    {
                        "benchmark": "write_throughput",
                        "backend": backend,
                        "atoms": atoms,
                        "with_custom": with_custom,
                        "mol_s": mol_s * scale,
                        "ci95_mol_s": 0.0,
                    }
                )
    return rows


def _write_storage_payload(scale: float = 1.0) -> list[dict]:
    rows = []
    data = {
        False: {
            16: {"atompack": 1.00, "hdf5_soa": 0.88, "lmdb_packed": 1.22, "lmdb_pickle": 2.40, "ase_lmdb": 3.20, "ase_sqlite": 3.00},
            64: {"atompack": 1.00, "hdf5_soa": 0.95, "lmdb_packed": 2.10, "lmdb_pickle": 2.12, "ase_lmdb": 4.00, "ase_sqlite": 2.90},
            256: {"atompack": 1.00, "hdf5_soa": 0.99, "lmdb_packed": 1.18, "lmdb_pickle": 1.18, "ase_lmdb": 2.45, "ase_sqlite": 2.30},
        },
        True: {
            16: {"atompack": 1.00, "hdf5_soa": 0.86, "lmdb_packed": 1.24, "lmdb_pickle": 3.60, "ase_lmdb": 3.10, "ase_sqlite": 2.20},
            64: {"atompack": 1.00, "hdf5_soa": 0.95, "lmdb_packed": 1.32, "lmdb_pickle": 1.32, "ase_lmdb": 2.60, "ase_sqlite": 2.00},
            256: {"atompack": 1.00, "hdf5_soa": 0.99, "lmdb_packed": 1.12, "lmdb_pickle": 1.12, "ase_lmdb": 2.20, "ase_sqlite": 1.85},
        },
    }
    atompack_bpm = {16: 550.0, 64: 1750.0, 256: 6550.0}
    atompack_bpm_custom = {16: 1090.0, 64: 3050.0, 256: 10930.0}
    for with_custom, atoms_data in data.items():
        for atoms, backend_ratios in atoms_data.items():
            base_bpm = atompack_bpm_custom[atoms] if with_custom else atompack_bpm[atoms]
            for backend, ratio in backend_ratios.items():
                rows.append(
                    {
                        "benchmark": "write_storage",
                        "backend": backend,
                        "atoms": atoms,
                        "with_custom": with_custom,
                        "n_mols": 1000,
                        "n_mols_requested": 1000,
                        "trials": 1,
                        "codec": "none",
                        "mol_s": 0.0,
                        "ci95_mol_s": 0.0,
                        "size_bytes": base_bpm * 1000 * ratio * scale,
                        "normalized_size_bytes": base_bpm * 1000 * ratio * scale,
                        "bytes_per_mol": base_bpm * ratio * scale,
                        "bytes_per_atom": (base_bpm * ratio * scale) / atoms,
                        "size_ratio_vs_atompack": ratio,
                    }
                )
    return rows


def test_generate_story_report_smoke(tmp_path: Path, load_benchmark_module) -> None:
    pytest.importorskip("matplotlib")
    module = load_benchmark_module("plot_benchmark_story", "plot_benchmark_story.py")

    inputs = []
    for name, payload in {
        "nvme.json": _benchmark_payload(),
        "nfs.json": _benchmark_payload(scale=0.85),
        "write_nvme.json": _write_payload(),
        "write_storage.json": _write_storage_payload(),
    }.items():
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        inputs.append(path)

    out_dir = tmp_path / "story"
    manifest = module.generate_story_report(
        inputs,
        out_dir,
        title_prefix="Story",
        formats=["svg"],
        dpi=120,
    )

    assert (out_dir / "story_read_hero.svg").exists()
    assert (out_dir / "story_size_scaling.svg").exists()
    assert (out_dir / "story_random_filesystems.svg").exists()
    assert (out_dir / "story_write_overview.svg").exists()
    assert (out_dir / "story_write_storage.svg").exists()
    assert manifest["figures"][0]["status"] == "produced"
