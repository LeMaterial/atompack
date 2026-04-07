# Copyright 2026 Entalpic
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
def test_maintained_benchmark_entrypoints_import(stub_tqdm, load_benchmark_module) -> None:
    for file_name in (
        "benchmark.py",
        "write_benchmark.py",
        "scaling_benchmark.py",
        "memory_benchmark.py",
        "atompack_bestcase_read_benchmark.py",
    ):
        module = load_benchmark_module(file_name.replace(".py", ""), file_name)
        assert hasattr(module, "main")


def test_public_docs_reference_current_api_and_scripts() -> None:
    files = [
        ROOT / "README.md",
        ROOT / "docs" / "source" / "getting-started.rst",
        ROOT / "docs" / "source" / "performance.rst",
        ROOT / "docs" / "source" / "blog" / "atompack-release.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)

    assert "scripts/benchmark.py" not in combined
    assert "compare_real_omat_backends.py" not in combined
    assert "compare_lmdb_layouts.py" not in combined
    assert "atompack.Molecule(atoms)" not in combined
    assert "Database.open_mmap" not in combined
    assert "Database Format (v1)" not in combined

    assert "atompack-py/benchmarks/benchmark.py" in combined
    assert "Molecule.from_arrays" in combined or "atompack.Molecule(" in combined
