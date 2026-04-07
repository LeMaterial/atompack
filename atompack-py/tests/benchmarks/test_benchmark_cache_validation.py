# Copyright 2026 Entalpic
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_is_complete_dataset_detects_partial_backends(
    tmp_path: Path,
    load_benchmark_module,
    write_atompack_dataset,
) -> None:
    module = load_benchmark_module("benchmark_cache_validation", "benchmark.py")

    def fake_lmdb_count(path: Path | str) -> int | None:
        p = Path(path)
        marker = p / "count.txt"
        if not marker.exists():
            return None
        return int(marker.read_text(encoding="utf-8"))

    def fake_hdf5_count(path: Path | str) -> int | None:
        p = Path(path)
        if not p.exists():
            return None
        return int(p.read_text(encoding="utf-8"))

    module._lmdb_dataset_count = fake_lmdb_count  # type: ignore[attr-defined]
    module._hdf5_soa_dataset_count = fake_hdf5_count  # type: ignore[attr-defined]

    atompack_full = tmp_path / "full.atp"
    atompack_partial = tmp_path / "partial.atp"
    write_atompack_dataset(atompack_full, n_molecules=4, n_atoms=3)
    write_atompack_dataset(atompack_partial, n_molecules=2, n_atoms=3)

    packed_full = tmp_path / "packed_full"
    packed_partial = tmp_path / "packed_partial"
    pickle_full = tmp_path / "pickle_full"
    pickle_partial = tmp_path / "pickle_partial"
    hdf5_full = tmp_path / "hdf5_full.h5"
    hdf5_partial = tmp_path / "hdf5_partial.h5"
    packed_full.mkdir()
    packed_partial.mkdir()
    pickle_full.mkdir()
    pickle_partial.mkdir()
    (packed_full / "count.txt").write_text("4", encoding="utf-8")
    (packed_partial / "count.txt").write_text("2", encoding="utf-8")
    (pickle_full / "count.txt").write_text("4", encoding="utf-8")
    (pickle_partial / "count.txt").write_text("2", encoding="utf-8")
    hdf5_full.write_text("4", encoding="utf-8")
    hdf5_partial.write_text("2", encoding="utf-8")

    assert module._is_complete_dataset("atompack", atompack_full, 4) is True
    assert module._is_complete_dataset("atompack", atompack_partial, 4) is False
    assert module._is_complete_dataset("hdf5_soa", hdf5_full, 4) is True
    assert module._is_complete_dataset("hdf5_soa", hdf5_partial, 4) is False
    assert module._is_complete_dataset("lmdb_packed", packed_full, 4) is True
    assert module._is_complete_dataset("lmdb_packed", packed_partial, 4) is False
    assert module._is_complete_dataset("lmdb_pickle", pickle_full, 4) is True
    assert module._is_complete_dataset("lmdb_pickle", pickle_partial, 4) is False


def test_ensure_datasets_rebuilds_only_incomplete_backends(
    tmp_path: Path,
    monkeypatch,
    load_benchmark_module,
    write_atompack_dataset,
) -> None:
    module = load_benchmark_module("benchmark_cache_rebuild", "benchmark.py")

    scratch = tmp_path / "scratch"
    base = scratch / module._dataset_tag(3, 4, "none", False)
    base.mkdir(parents=True, exist_ok=True)

    def fake_lmdb_count(path: Path | str) -> int | None:
        p = Path(path)
        marker = p / "count.txt"
        if not marker.exists():
            return None
        return int(marker.read_text(encoding="utf-8"))

    write_atompack_dataset(base / "data.atp", n_molecules=2, n_atoms=3)
    (base / "lmdb_packed").mkdir()
    (base / "lmdb_pickle").mkdir()
    ((base / "lmdb_packed") / "count.txt").write_text("4", encoding="utf-8")
    ((base / "lmdb_pickle") / "count.txt").write_text("4", encoding="utf-8")

    calls: list[str] = []

    def fake_hdf5_count(path: Path | str) -> int | None:
        p = Path(path)
        if not p.exists():
            return None
        return int(p.read_text(encoding="utf-8"))

    def fake_create_atompack_db(
        path: str,
        n: int,
        atoms: int,
        compression: str = "zstd",
        level: int = 3,
        with_custom: bool = False,
        seed: int = 42,
    ) -> None:
        calls.append("atompack")
        write_atompack_dataset(Path(path), n_molecules=n, n_atoms=atoms)

    def fake_create_lmdb_packed(
        path: str,
        n: int,
        atoms: int,
        codec: str = "zstd:3",
        with_custom: bool = False,
        seed: int = 42,
    ) -> None:
        calls.append("lmdb_packed")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "count.txt").write_text(str(n), encoding="utf-8")

    def fake_create_lmdb_pickle(
        path: str,
        n: int,
        atoms: int,
        codec: str = "zstd:3",
        with_custom: bool = False,
        seed: int = 42,
    ) -> None:
        calls.append("lmdb_pickle")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "count.txt").write_text(str(n), encoding="utf-8")

    def fake_create_hdf5_soa(
        path: str,
        n: int,
        atoms: int,
        with_custom: bool = False,
        seed: int = 42,
    ) -> None:
        calls.append("hdf5_soa")
        Path(path).write_text(str(n), encoding="utf-8")

    monkeypatch.setattr(module, "create_atompack_db", fake_create_atompack_db)
    monkeypatch.setattr(module, "create_hdf5_soa", fake_create_hdf5_soa)
    monkeypatch.setattr(module, "create_lmdb_packed", fake_create_lmdb_packed)
    monkeypatch.setattr(module, "create_lmdb_pickle", fake_create_lmdb_pickle)
    monkeypatch.setattr(module, "_lmdb_dataset_count", fake_lmdb_count)
    monkeypatch.setattr(module, "_hdf5_soa_dataset_count", fake_hdf5_count)
    monkeypatch.setattr(module, "hdf5_available", lambda: True)

    paths = module.ensure_datasets(
        scratch=scratch,
        atoms=3,
        n_mols=4,
        codec_label="none",
        atp_comp="none",
        atp_level=0,
    )

    assert calls == ["atompack", "hdf5_soa"]
    assert module._is_complete_dataset("atompack", paths["atompack"], 4) is True
    assert module._is_complete_dataset("hdf5_soa", paths["hdf5_soa"], 4) is True
    assert module._is_complete_dataset("lmdb_packed", paths["lmdb_packed"], 4) is True
    assert module._is_complete_dataset("lmdb_pickle", paths["lmdb_pickle"], 4) is True


def test_bench_read_includes_hdf5_backend_when_available(
    tmp_path: Path,
    monkeypatch,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("benchmark_hdf5_read_backend", "benchmark.py")

    hdf5_path = tmp_path / "data.h5"
    hdf5_path.write_bytes(b"hdf5")

    monkeypatch.setattr(module, "hdf5_available", lambda: True)
    monkeypatch.setattr(module.atompack.Database, "open", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "read_atompack", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_open_hdf5_soa_reader",
        lambda path: (object(), SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.setattr(module, "_lmdb_codec_available", lambda codec: False)
    monkeypatch.setattr(module, "_has_ase", False)
    monkeypatch.setattr(
        module,
        "bench",
        lambda fn, n, trials, **kwargs: {
            "mol_s": 1.0,
            "median_mol_s": 1.0,
            "mean_mol_s": 1.0,
            "std_mol_s": 0.0,
            "ci95_mol_s": 0.0,
            "trials_mol_s": [1.0],
        },
    )

    rows = module._bench_read(
        paths={
            "atompack": str(tmp_path / "data.atp"),
            "hdf5_soa": str(hdf5_path),
            "lmdb_packed": str(tmp_path / "lmdb_packed"),
            "lmdb_pickle": str(tmp_path / "lmdb_pickle"),
        },
        atoms=3,
        n_mols=4,
        sample=2,
        codec_label="none",
        with_custom=False,
        trials=1,
    )

    assert any(row["backend"] == "hdf5_soa" for row in rows)


def test_bench_read_includes_hdf5_and_lmdb_packed_for_custom_rows(
    tmp_path: Path,
    monkeypatch,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("benchmark_custom_lmdb_packed", "benchmark.py")

    hdf5_path = tmp_path / "data.h5"
    hdf5_path.write_bytes(b"hdf5")

    monkeypatch.setattr(module, "hdf5_available", lambda: True)
    monkeypatch.setattr(module.atompack.Database, "open", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "read_atompack", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "read_hdf5_soa", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "read_lmdb_packed", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "read_lmdb_pickle", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_open_hdf5_soa_reader",
        lambda path: (object(), SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.setattr(module, "_lmdb_codec_available", lambda codec: True)
    monkeypatch.setattr(module, "_has_ase", False)

    class _Txn:
        def abort(self) -> None:
            return None

    class _Env:
        def begin(self, write: bool = False):
            return _Txn()

        def close(self) -> None:
            return None

    monkeypatch.setattr(module, "_lmdb", SimpleNamespace(open=lambda *args, **kwargs: _Env()))
    monkeypatch.setattr(
        module,
        "bench",
        lambda fn, n, trials, **kwargs: {
            "mol_s": 1.0,
            "median_mol_s": 1.0,
            "mean_mol_s": 1.0,
            "std_mol_s": 0.0,
            "ci95_mol_s": 0.0,
            "trials_mol_s": [1.0],
        },
    )

    rows = module._bench_read(
        paths={
            "atompack": str(tmp_path / "data.atp"),
            "hdf5_soa": str(hdf5_path),
            "lmdb_packed": str(tmp_path / "lmdb_packed"),
            "lmdb_pickle": str(tmp_path / "lmdb_pickle"),
        },
        atoms=3,
        n_mols=4,
        sample=2,
        codec_label="none",
        with_custom=True,
        trials=1,
    )

    assert any(row["backend"] == "hdf5_soa" for row in rows)
    assert any(row["backend"] == "lmdb_packed" for row in rows)


def test_ensure_datasets_reuses_in_process_cache(
    tmp_path: Path,
    monkeypatch,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("benchmark_cache_reuse", "benchmark.py")

    calls: list[str] = []

    def fake_is_complete_dataset(label: str, path: Path, expected_count: int) -> bool:
        calls.append(label)
        return True

    monkeypatch.setattr(module, "_is_complete_dataset", fake_is_complete_dataset)

    first = module.ensure_datasets(
        scratch=tmp_path / "scratch",
        atoms=3,
        n_mols=4,
        codec_label="none",
        atp_comp="none",
        atp_level=0,
    )
    assert calls == ["atompack", "hdf5_soa", "lmdb_packed", "lmdb_pickle"]

    calls.clear()
    second = module.ensure_datasets(
        scratch=tmp_path / "scratch",
        atoms=3,
        n_mols=4,
        codec_label="none",
        atp_comp="none",
        atp_level=0,
    )

    assert calls == []
    assert first == second


def test_ensure_datasets_can_skip_unselected_backends(
    tmp_path: Path,
    monkeypatch,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("benchmark_cache_skip_selected", "benchmark.py")

    calls: list[str] = []

    def fake_create_atompack_db(*args, **kwargs) -> None:
        calls.append("atompack")

    def fake_create_hdf5_soa(path: str, n: int, atoms: int, with_custom: bool = False, seed: int = 42) -> None:
        calls.append("hdf5_soa")
        Path(path).write_text(str(n), encoding="utf-8")

    def fake_create_lmdb_packed(*args, **kwargs) -> None:
        calls.append("lmdb_packed")

    def fake_create_lmdb_pickle(*args, **kwargs) -> None:
        calls.append("lmdb_pickle")

    monkeypatch.setattr(module, "create_atompack_db", fake_create_atompack_db)
    monkeypatch.setattr(module, "create_hdf5_soa", fake_create_hdf5_soa)
    monkeypatch.setattr(module, "create_lmdb_packed", fake_create_lmdb_packed)
    monkeypatch.setattr(module, "create_lmdb_pickle", fake_create_lmdb_pickle)
    monkeypatch.setattr(module, "_is_complete_dataset", lambda *args, **kwargs: False)
    monkeypatch.setattr(module, "_lmdb_codec_available", lambda codec: True)
    monkeypatch.setattr(module, "hdf5_available", lambda: True)

    module.ensure_datasets(
        scratch=tmp_path / "scratch",
        atoms=3,
        n_mols=4,
        codec_label="none",
        atp_comp="none",
        atp_level=0,
        skip_backends=frozenset({"atompack", "lmdb_packed", "lmdb_pickle"}),
    )

    assert calls == ["hdf5_soa"]


def test_bench_read_respects_selected_backends(
    tmp_path: Path,
    monkeypatch,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("benchmark_hdf5_selected_backend", "benchmark.py")

    hdf5_path = tmp_path / "data.h5"
    hdf5_path.write_bytes(b"hdf5")

    monkeypatch.setattr(module, "hdf5_available", lambda: True)
    monkeypatch.setattr(module.atompack.Database, "open", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "read_atompack", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "read_hdf5_soa", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_open_hdf5_soa_reader",
        lambda path: (object(), SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.setattr(module, "_lmdb_codec_available", lambda codec: True)
    monkeypatch.setattr(module, "_has_ase", False)
    monkeypatch.setattr(
        module,
        "bench",
        lambda fn, n, trials, **kwargs: {
            "mol_s": 1.0,
            "median_mol_s": 1.0,
            "mean_mol_s": 1.0,
            "std_mol_s": 0.0,
            "ci95_mol_s": 0.0,
            "trials_mol_s": [1.0],
        },
    )

    rows = module._bench_read(
        paths={
            "atompack": str(tmp_path / "data.atp"),
            "hdf5_soa": str(hdf5_path),
            "lmdb_packed": str(tmp_path / "lmdb_packed"),
            "lmdb_pickle": str(tmp_path / "lmdb_pickle"),
        },
        atoms=3,
        n_mols=4,
        sample=2,
        codec_label="none",
        with_custom=False,
        trials=1,
        selected_backends=frozenset({"hdf5_soa"}),
    )

    assert [row["backend"] for row in rows] == ["hdf5_soa"]


def test_make_indices_is_stable_and_cached(load_benchmark_module) -> None:
    module = load_benchmark_module("benchmark_indices_cache", "benchmark.py")

    first = module._make_indices(100, 10, seed=1234, sequential=False)
    second = module._make_indices(100, 10, seed=1234, sequential=False)
    sequential = module._make_indices(100, 10, sequential=True)

    assert isinstance(first, tuple)
    assert first is second
    assert len(first) == 10
    assert len(set(first)) == 10
    assert sequential == tuple(range(10))
