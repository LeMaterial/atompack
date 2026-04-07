# Copyright 2026 Entalpic
from __future__ import annotations


def test_atompack_batch_benchmark_populate_default_is_disabled(load_benchmark_module) -> None:
    module = load_benchmark_module(
        "atompack_batch_benchmark_populate_default",
        "atompack_batch_benchmark.py",
    )

    assert module.DEFAULT_POPULATE is False


def test_atompack_bestcase_benchmark_defaults_are_focused(load_benchmark_module) -> None:
    module = load_benchmark_module(
        "atompack_bestcase_read_benchmark_defaults",
        "atompack_bestcase_read_benchmark.py",
    )

    assert module.DEFAULT_POPULATE is False
    assert module.DEFAULT_METHODS == ["flat"]
    assert module.DEFAULT_TARGET_ATOMS_PER_BATCH == [32_768, 131_072, 524_288]
    assert module.NON_ATOMPACK_BACKENDS == frozenset(
        {"hdf5_soa", "lmdb_soa", "lmdb_packed", "lmdb_pickle"}
    )


def test_atompack_bestcase_adaptive_batch_sizes_scale_with_atom_count(load_benchmark_module) -> None:
    module = load_benchmark_module(
        "atompack_bestcase_read_benchmark_batch_sizes",
        "atompack_bestcase_read_benchmark.py",
    )

    assert module._adaptive_batch_sizes(64, 10_000, module.DEFAULT_TARGET_ATOMS_PER_BATCH) == [
        512,
        2048,
        8192,
    ]
    assert module._adaptive_batch_sizes(256, 10_000, module.DEFAULT_TARGET_ATOMS_PER_BATCH) == [
        128,
        512,
        2048,
    ]
    assert module._adaptive_batch_sizes(512, 10_000, module.DEFAULT_TARGET_ATOMS_PER_BATCH) == [
        64,
        256,
        1024,
    ]


def test_scaling_benchmark_populate_default_is_disabled(
    stub_tqdm,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("scaling_benchmark_populate_default", "scaling_benchmark.py")

    assert module.DEFAULT_POPULATE is False
