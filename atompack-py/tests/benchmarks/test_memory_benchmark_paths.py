# Copyright 2026 Entalpic
from __future__ import annotations

from pathlib import Path


def test_memory_benchmark_uses_existing_benchmark_outputs(
    tmp_path: Path,
    monkeypatch,
    load_benchmark_module,
    write_atompack_dataset,
) -> None:
    benchmark_module = load_benchmark_module("benchmark", "benchmark.py")
    memory_module = load_benchmark_module("memory_benchmark_existing_paths", "memory_benchmark.py")

    atoms = 3
    n_mols = 4
    scratch = tmp_path / "scratch"
    base = scratch / benchmark_module._dataset_tag(atoms, n_mols, "none", False)
    base.mkdir(parents=True, exist_ok=True)

    atompack_path = base / "data.atp"
    write_atompack_dataset(atompack_path, n_molecules=n_mols, n_atoms=atoms)
    (base / "lmdb_packed").mkdir()
    (base / "lmdb_pickle").mkdir()

    def fail_ensure(*args, **kwargs):
        raise AssertionError("memory_benchmark should not create datasets by default")

    monkeypatch.setattr(memory_module, "ensure_datasets", fail_ensure)
    monkeypatch.setattr(memory_module, "ensure_ase_datasets", fail_ensure)
    monkeypatch.setattr(memory_module, "_HAS_ASE", False)

    paths = memory_module._resolve_paths(
        scratch=scratch,
        atoms=atoms,
        n_mols_override=n_mols,
        create_missing=False,
    )

    assert paths.atompack == atompack_path
    assert paths.lmdb_packed == base / "lmdb_packed"
    assert paths.lmdb_pickle == base / "lmdb_pickle"


def test_memory_benchmark_uses_shared_populate_default(load_benchmark_module) -> None:
    memory_module = load_benchmark_module("memory_benchmark_populate_default", "memory_benchmark.py")

    assert memory_module.DEFAULT_BENCHMARK_POPULATE is False
