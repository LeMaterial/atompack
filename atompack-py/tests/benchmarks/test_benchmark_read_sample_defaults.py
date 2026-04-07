# Copyright 2026 Entalpic
from __future__ import annotations

def test_read_sample_uses_12_atom_override(load_benchmark_module) -> None:
    module = load_benchmark_module("benchmark_read_sample_defaults", "benchmark.py")

    assert module._read_sample(1_000_000, atoms=12) == 100_000
    assert module._read_sample(2_000_000, atoms=256) == 20_000


def test_256_atom_benchmark_skips_custom_variant(load_benchmark_module) -> None:
    module = load_benchmark_module("benchmark_custom_variant_defaults", "benchmark.py")

    assert module._custom_variants_for_atoms(256) == (False,)
    assert module._custom_variants_for_atoms(64) == (False, True)


def test_benchmark_populate_default_is_disabled(load_benchmark_module) -> None:
    module = load_benchmark_module("benchmark_populate_default", "benchmark.py")

    assert module.DEFAULT_BENCHMARK_POPULATE is False


def test_multiprocessing_skips_pathological_256_atom_backends(load_benchmark_module) -> None:
    module = load_benchmark_module("benchmark_mp_skip_defaults", "benchmark.py")

    assert module._multiprocessing_skipped_backends(12) == frozenset()
    assert module._multiprocessing_skipped_backends(256) == frozenset({"hdf5_soa"})


def test_mp_bench_closes_pool_on_success(load_benchmark_module) -> None:
    module = load_benchmark_module("benchmark_mp_cleanup_success", "benchmark.py")

    class _FakePool:
        def __init__(self) -> None:
            self.closed = 0
            self.joined = 0
            self.terminated = 0

        def map(self, worker_fn, chunks):
            return [worker_fn(chunk) for chunk in chunks]

        def close(self) -> None:
            self.closed += 1

        def join(self) -> None:
            self.joined += 1

        def terminate(self) -> None:
            self.terminated += 1

    pool = _FakePool()

    class _FakeContext:
        def Pool(self, *args, **kwargs):
            return pool

    stats = module._mp_bench(
        _FakeContext(),
        1,
        lambda: None,
        (),
        lambda chunk: len(chunk),
        [[1, 2, 3]],
        3,
        1,
        "test success",
    )

    assert stats["mol_s"] > 0
    assert pool.closed == 1
    assert pool.joined == 1
    assert pool.terminated == 0


def test_mp_bench_terminates_pool_on_failure(load_benchmark_module) -> None:
    module = load_benchmark_module("benchmark_mp_cleanup_failure", "benchmark.py")

    class _FakePool:
        def __init__(self) -> None:
            self.closed = 0
            self.joined = 0
            self.terminated = 0
            self.calls = 0

        def map(self, worker_fn, chunks):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("boom")
            return [worker_fn(chunk) for chunk in chunks]

        def close(self) -> None:
            self.closed += 1

        def join(self) -> None:
            self.joined += 1

        def terminate(self) -> None:
            self.terminated += 1

    pool = _FakePool()

    class _FakeContext:
        def Pool(self, *args, **kwargs):
            return pool

    try:
        module._mp_bench(
            _FakeContext(),
            1,
            lambda: None,
            (),
            lambda chunk: len(chunk),
            [[1, 2, 3]],
            3,
            1,
            "test failure",
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")

    assert pool.closed == 0
    assert pool.joined == 1
    assert pool.terminated == 1
