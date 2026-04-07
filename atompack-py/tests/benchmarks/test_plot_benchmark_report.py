# Copyright 2026 Entalpic
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _benchmark_payload(scale: float = 1.0) -> dict[str, list[dict]]:
    def row(
        benchmark: str,
        backend: str,
        *,
        atoms: int,
        mol_s: float,
        workers: int | None = None,
        ci95_atoms_s: float = 0.0,
    ) -> dict[str, float | int | str]:
        payload: dict[str, float | int | str] = {
            "benchmark": benchmark,
            "backend": backend,
            "atoms": atoms,
            "mol_s": mol_s * scale,
            "atoms_s": mol_s * atoms * scale,
            "ci95_atoms_s": ci95_atoms_s,
            "sample": 100,
            "n_mols": 10_000,
        }
        if workers is not None:
            payload["workers"] = workers
        return payload

    rows = [
        row("sequential", "atompack", atoms=16, mol_s=1800),
        row("sequential", "lmdb_soa", atoms=16, mol_s=1400),
        row("sequential", "lmdb_packed", atoms=16, mol_s=900),
        row("sequential", "lmdb_pickle", atoms=16, mol_s=700),
        row("sequential", "hdf5_soa", atoms=16, mol_s=350),
        row("sequential", "ase_lmdb", atoms=16, mol_s=50),
        row("sequential", "ase_sqlite", atoms=16, mol_s=20),
        row("sequential", "atompack", atoms=64, mol_s=1200),
        row("sequential", "lmdb_soa", atoms=64, mol_s=900),
        row("sequential", "lmdb_packed", atoms=64, mol_s=500),
        row("sequential", "lmdb_pickle", atoms=64, mol_s=360),
        row("sequential", "hdf5_soa", atoms=64, mol_s=120),
        row("sequential", "ase_lmdb", atoms=64, mol_s=20),
        row("sequential", "ase_sqlite", atoms=64, mol_s=8),
    ]
    for workers, rates in {
        1: {"atompack": 900, "lmdb_soa": 500, "lmdb_packed": 360, "lmdb_pickle": 240, "hdf5_soa": 110, "ase_lmdb": 15, "ase_sqlite": 6},
        2: {"atompack": 1500, "lmdb_soa": 650, "lmdb_packed": 600, "lmdb_pickle": 430, "hdf5_soa": 125, "ase_lmdb": 18, "ase_sqlite": 7},
        4: {"atompack": 2200, "lmdb_soa": 800, "lmdb_packed": 980, "lmdb_pickle": 700, "hdf5_soa": 130, "ase_lmdb": 22, "ase_sqlite": 8},
    }.items():
        for backend, mol_s in rates.items():
            rows.append(
                row(
                    "multiprocessing",
                    backend,
                    atoms=16,
                    workers=workers,
                    mol_s=mol_s,
                    ci95_atoms_s=20.0,
                )
            )
    for workers, rates in {
        1: {"atompack": 500, "lmdb_soa": 360, "lmdb_packed": 220, "lmdb_pickle": 150, "hdf5_soa": 60, "ase_lmdb": 9, "ase_sqlite": 3},
        2: {"atompack": 850, "lmdb_soa": 420, "lmdb_packed": 360, "lmdb_pickle": 250, "hdf5_soa": 62, "ase_lmdb": 10, "ase_sqlite": 3.5},
        4: {"atompack": 1200, "lmdb_soa": 500, "lmdb_packed": 700, "lmdb_pickle": 430, "hdf5_soa": 64, "ase_lmdb": 12, "ase_sqlite": 4},
    }.items():
        for backend, mol_s in rates.items():
            rows.append(
                row(
                    "multiprocessing",
                    backend,
                    atoms=64,
                    workers=workers,
                    mol_s=mol_s,
                    ci95_atoms_s=30.0,
                )
            )
    return {"results": rows}


def _scaling_json() -> dict:
    return {
        "results": [
            {"atoms_per_molecule": 16, "num_molecules": 1000, "backend": "atompack", "n_workers": 1, "mean_mol_s": 100.0, "mean_atoms_s": 1600.0, "ci95": 5.0},
            {"atoms_per_molecule": 16, "num_molecules": 1000, "backend": "lmdb_packed", "n_workers": 1, "mean_mol_s": 80.0, "mean_atoms_s": 1280.0, "ci95": 4.0},
            {"atoms_per_molecule": 64, "num_molecules": 1000, "backend": "atompack", "n_workers": 4, "mean_mol_s": 120.0, "mean_atoms_s": 7680.0, "ci95": 6.0},
            {"atoms_per_molecule": 64, "num_molecules": 1000, "backend": "lmdb_packed", "n_workers": 4, "mean_mol_s": 90.0, "mean_atoms_s": 5760.0, "ci95": 3.0},
        ],
    }


def _memory_json() -> dict:
    return {
        "open_connection_memory": [
            {
                "backend": "atompack",
                "workers": 4,
                "rss_kib_per_worker": 2048,
                "rss_anon_kib_per_worker": 0,
                "rss_file_kib_per_worker": 2048,
                "private_total_kib_per_worker": 512,
            },
            {
                "backend": "lmdb_pickle",
                "workers": 4,
                "rss_kib_per_worker": 1024,
                "rss_anon_kib_per_worker": 256,
                "rss_file_kib_per_worker": 768,
                "private_total_kib_per_worker": 256,
            },
        ],
        "streaming_read_memory": [
            {
                "backend": "atompack",
                "rss_kib": 4096,
                "rss_anon_kib": 0,
                "rss_file_kib": 4096,
                "private_total_kib": 1024,
            },
            {
                "backend": "lmdb_pickle",
                "rss_kib": 512,
                "rss_anon_kib": 128,
                "rss_file_kib": 384,
                "private_total_kib": 256,
            },
        ],
    }


def _batch_rows() -> list[dict]:
    return [
        {"benchmark": "atompack_batch_api", "method": "loop", "batch_size": 32, "threads": "1", "mol_s": 100.0, "ci95_mol_s": 5.0, "atoms": 64, "n_mols": 1000},
        {"benchmark": "atompack_batch_api", "method": "loop", "batch_size": 128, "threads": "1", "mol_s": 140.0, "ci95_mol_s": 4.0, "atoms": 64, "n_mols": 1000},
        {"benchmark": "atompack_batch_api", "method": "batch", "batch_size": 32, "threads": "4", "mol_s": 200.0, "ci95_mol_s": 10.0, "atoms": 64, "n_mols": 1000},
        {"benchmark": "atompack_batch_api", "method": "flat", "batch_size": 32, "threads": "native", "mol_s": 300.0, "ci95_mol_s": 8.0, "atoms": 64, "n_mols": 1000},
    ]


def _write_rows() -> list[dict]:
    return [
        {"benchmark": "write_throughput", "backend": "atompack", "atoms": 16, "with_custom": False, "n_mols": 50000, "n_mols_requested": 50000, "trials": 3, "mol_s": 100000.0, "ci95_mol_s": 3000.0, "size_bytes": 1000},
        {"benchmark": "write_throughput", "backend": "lmdb_soa", "atoms": 16, "with_custom": False, "n_mols": 50000, "n_mols_requested": 50000, "trials": 3, "mol_s": 80000.0, "ci95_mol_s": 2000.0, "size_bytes": 1200},
        {"benchmark": "write_throughput", "backend": "atompack", "atoms": 16, "with_custom": True, "n_mols": 50000, "n_mols_requested": 50000, "trials": 3, "mol_s": 60000.0, "ci95_mol_s": 1500.0, "size_bytes": 1400},
        {"benchmark": "write_throughput", "backend": "lmdb_pickle", "atoms": 16, "with_custom": True, "n_mols": 50000, "n_mols_requested": 50000, "trials": 3, "mol_s": 50000.0, "ci95_mol_s": 1400.0, "size_bytes": 1500},
        {"benchmark": "write_scaling", "backend": "atompack", "atoms": 12, "with_custom": False, "n_mols": 50000, "n_mols_requested": 50000, "trials": 1, "mol_s": 120000.0, "ci95_mol_s": 0.0, "size_bytes": 2000},
        {"benchmark": "write_scaling", "backend": "lmdb_packed", "atoms": 12, "with_custom": False, "n_mols": 50000, "n_mols_requested": 50000, "trials": 1, "mol_s": 70000.0, "ci95_mol_s": 0.0, "size_bytes": 2200},
        {"benchmark": "write_batch_scaling", "backend": "atompack", "atoms": 256, "with_custom": False, "n_mols": 50000, "n_mols_requested": 50000, "trials": 3, "mol_s": 22000.0, "ci95_mol_s": 800.0, "size_bytes": 3000, "batch_size": 2048},
    ]


def test_detect_input_schema(load_benchmark_module) -> None:
    module = load_benchmark_module("plot_benchmark_report_detect", "plot_benchmark_report.py")

    assert module._detect_input_schema(_benchmark_payload()["results"]) == "benchmark"
    assert module._detect_input_schema(_scaling_json()) == "scaling"
    assert module._detect_input_schema(_memory_json()) == "memory"
    assert module._detect_input_schema(_batch_rows()) == "batch"
    assert module._detect_input_schema(_write_rows()) == "write"
    assert "blog" in module.SUPPORTED_REPORT_MODES


def test_load_inputs_normalizes_supported_schemas(tmp_path: Path, load_benchmark_module) -> None:
    module = load_benchmark_module("plot_benchmark_report_load", "plot_benchmark_report.py")

    files = {
        "nvme.json": _benchmark_payload(),
        "scaling.json": _scaling_json(),
        "memory.json": _memory_json(),
        "batch.json": _batch_rows(),
        "write.json": _write_rows(),
    }
    paths = []
    for name, payload in files.items():
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    bundle = module.load_inputs(paths)

    assert len(bundle["benchmark_rows"]) == len(_benchmark_payload()["results"])
    assert len(bundle["scaling_rows"]) == len(_scaling_json()["results"])
    assert len(bundle["memory_rows_open"]) == len(_memory_json()["open_connection_memory"])
    assert len(bundle["memory_rows_stream"]) == len(_memory_json()["streaming_read_memory"])
    assert len(bundle["batch_rows"]) == len(_batch_rows())
    assert len(bundle["write_rows"]) == len(_write_rows())
    assert bundle["benchmark_rows"][0]["scenario"] == "sequential"
    assert bundle["benchmark_rows"][0]["source_name"] == "nvme"
    assert bundle["write_rows"][0]["source_name"] == "write"
    assert any(row["scenario"] == "write_batch_scaling" for row in bundle["write_rows"])
    assert any(row["batch_size"] == 2048 for row in bundle["write_rows"])


def test_select_figures_supports_blog_mode(tmp_path: Path, load_benchmark_module) -> None:
    module = load_benchmark_module("plot_benchmark_report_select", "plot_benchmark_report.py")

    paths = []
    for name, payload in {
        "nvme.json": _benchmark_payload(),
        "nfs.json": _benchmark_payload(scale=0.85),
    }.items():
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    bundle = module.load_inputs(paths)
    figures = module._select_figures(bundle, no_batch_api=False, report_mode="blog")
    status = {figure["key"]: figure["status"] for figure in figures}

    assert status["blog_read_hero"] == "produced"
    assert status["blog_size_scaling"] == "produced"
    assert status["blog_random_filesystems"] == "produced"


def test_render_blog_report_and_manifest(tmp_path: Path, load_benchmark_module) -> None:
    pytest.importorskip("matplotlib")
    module = load_benchmark_module("plot_benchmark_report_blog", "plot_benchmark_report.py")

    input_paths = []
    for name, payload in {
        "nvme.json": _benchmark_payload(),
        "nfs.json": _benchmark_payload(scale=0.82),
    }.items():
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        input_paths.append(path)

    out_dir = tmp_path / "blog"
    manifest = module.generate_report(
        input_paths,
        out_dir,
        title_prefix="Launch",
        formats=["svg"],
        dpi=120,
        width=12.0,
        overview_width=14.0,
        no_batch_api=False,
        report_mode="blog",
    )

    assert (out_dir / "blog_read_hero.svg").exists()
    assert (out_dir / "blog_size_scaling.svg").exists()
    assert (out_dir / "blog_random_filesystems.svg").exists()
    manifest_path = out_dir / "figure_manifest.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["report_mode"] == "blog"
    assert any(item["key"] == "blog_read_hero" and item["status"] == "produced" for item in manifest_data["figures"])
    assert manifest["figures"]


def test_render_full_report_smoke(tmp_path: Path, load_benchmark_module) -> None:
    pytest.importorskip("matplotlib")
    module = load_benchmark_module("plot_benchmark_report_full", "plot_benchmark_report.py")

    files = {
        "bench.json": _benchmark_payload(),
        "scaling.json": _scaling_json(),
        "memory.json": _memory_json(),
        "batch.json": _batch_rows(),
        "write.json": _write_rows(),
    }
    input_paths = []
    for name, payload in files.items():
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        input_paths.append(path)

    out_dir = tmp_path / "full"
    manifest = module.generate_report(
        input_paths,
        out_dir,
        title_prefix="Smoke",
        formats=["svg"],
        dpi=120,
        width=10.0,
        overview_width=10.0,
        no_batch_api=False,
        report_mode="full",
    )

    assert (out_dir / "read_overview.svg").exists()
    assert (out_dir / "multiprocessing_scaling.svg").exists()
    assert (out_dir / "atom_scaling.svg").exists()
    assert (out_dir / "read_scaling.svg").exists()
    assert (out_dir / "memory_profile.svg").exists()
    assert (out_dir / "batch_api.svg").exists()
    assert (out_dir / "write_overview.svg").exists()
    assert (out_dir / "figure_manifest.json").exists()
    assert manifest["report_mode"] == "full"


def test_unsupported_untyped_json_fails_clearly(tmp_path: Path, load_benchmark_module) -> None:
    module = load_benchmark_module("plot_benchmark_report_unsupported", "plot_benchmark_report.py")

    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"backend": "atompack", "mol_s": 1.0}]), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported benchmark JSON schema"):
        module.load_inputs([path])
