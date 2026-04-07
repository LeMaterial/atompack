# Copyright 2026 Entalpic
from __future__ import annotations

import json
from pathlib import Path


def test_write_benchmark_tables_include_hdf5_without_lmdb_soa(
    monkeypatch,
    tmp_path: Path,
    capsys,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("write_benchmark_lmdb_soa", "write_benchmark.py")

    monkeypatch.setattr(module, "_lmdb", object())
    monkeypatch.setattr(module, "_has_ase", False)
    monkeypatch.setattr(module, "hdf5_available", lambda: True)
    monkeypatch.setattr(module, "write_atompack", lambda path, n, atoms, *args, **kwargs: Path(path).write_bytes(b"atp"))
    monkeypatch.setattr(module, "write_hdf5_soa", lambda path, n, atoms, *args, **kwargs: Path(path).write_bytes(b"h5"))
    monkeypatch.setattr(module, "write_lmdb_packed", lambda path, n, atoms, *args, **kwargs: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(module, "write_lmdb_pickle", lambda path, n, atoms, *args, **kwargs: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(module, "_dir_size", lambda path: 0)

    module.bench_throughput([16], n_mols=4, trials=1, scratch_dir=str(tmp_path))
    module.bench_storage_footprint([16], n_mols=4, scratch_dir=str(tmp_path))

    out = capsys.readouterr().out
    assert "hdf5_soa" in out
    assert "lmdb_soa" not in out


def test_write_benchmark_can_emit_structured_json(
    monkeypatch,
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("write_benchmark_json", "write_benchmark.py")

    monkeypatch.setattr(module, "_lmdb", object())
    monkeypatch.setattr(module, "_has_ase", False)
    monkeypatch.setattr(module, "hdf5_available", lambda: True)
    monkeypatch.setattr(module, "write_atompack", lambda path, n, atoms, *args, **kwargs: Path(path).write_bytes(b"atp"))
    monkeypatch.setattr(module, "write_hdf5_soa", lambda path, n, atoms, *args, **kwargs: Path(path).write_bytes(b"h5"))
    monkeypatch.setattr(module, "write_lmdb_packed", lambda path, n, atoms, *args, **kwargs: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(module, "write_lmdb_pickle", lambda path, n, atoms, *args, **kwargs: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(module, "_dir_size", lambda path: 0)

    out_path = tmp_path / "write_results.json"
    rc = module.main(
        [
            "--bench", "1", "2",
            "--atoms", "16",
            "--n-mols", "4",
            "--trials", "1",
            "--scratch-dir", str(tmp_path),
            "--out", str(out_path),
        ]
    )

    assert rc == 0
    rows = json.loads(out_path.read_text(encoding="utf-8"))
    assert rows
    assert {"write_throughput", "write_storage"} <= {row["benchmark"] for row in rows}
    assert any(row["backend"] == "hdf5_soa" for row in rows)
    assert all(row["backend"] != "lmdb_soa" for row in rows)


def test_write_benchmark_defaults_to_none_codec_and_allows_override(
    monkeypatch,
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("write_benchmark_codec_defaults", "write_benchmark.py")

    throughput_calls: list[str] = []
    storage_calls: list[str] = []

    monkeypatch.setattr(
        module,
        "bench_throughput",
        lambda *args, **kwargs: throughput_calls.append(kwargs["codec"]) or [],
    )
    monkeypatch.setattr(
        module,
        "bench_storage_footprint",
        lambda *args, **kwargs: storage_calls.append(kwargs["codec"]) or [],
    )

    rc = module.main(["--bench", "1", "2"])
    assert rc == 0
    assert throughput_calls == ["none"]
    assert storage_calls == ["none"]

    throughput_calls.clear()
    storage_calls.clear()
    rc = module.main(["--bench", "1", "2", "--codec", "zstd:5"])
    assert rc == 0
    assert throughput_calls == ["zstd:5"]
    assert storage_calls == ["zstd:5"]


def test_recommend_atompack_batch_size_shrinks_for_large_atoms(load_benchmark_module) -> None:
    module = load_benchmark_module("write_benchmark_batch_size", "write_benchmark.py")

    assert module._recommend_atompack_batch_size(16, with_custom=False) == 2048
    assert module._recommend_atompack_batch_size(64, with_custom=False) == 2048
    assert module._recommend_atompack_batch_size(128, with_custom=False) == 256
    assert module._recommend_atompack_batch_size(256, with_custom=False) == 256
    assert module._recommend_atompack_batch_size(16, with_custom=True) == 4096
    assert module._recommend_atompack_batch_size(64, with_custom=True) == 1024
    assert module._recommend_atompack_batch_size(128, with_custom=True) == 1024
    assert module._recommend_atompack_batch_size(256, with_custom=True) == 1024


def test_bench_throughput_records_resolved_atompack_batch_size(
    monkeypatch,
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("write_benchmark_batch_rows", "write_benchmark.py")

    monkeypatch.setattr(module, "_lmdb", None)
    monkeypatch.setattr(module, "_has_ase", False)
    monkeypatch.setattr(module, "write_atompack", lambda path, n, atoms, *args, **kwargs: Path(path).write_bytes(b"atp"))

    rows = module.bench_throughput([256], n_mols=4, trials=1, scratch_dir=str(tmp_path))

    atompack_rows = [row for row in rows if row["backend"] == "atompack"]
    assert atompack_rows
    assert all(row["atompack_batch_size"] < module.WRITE_BATCH_SIZE for row in atompack_rows)
    assert all("trial_repeats_median" in row for row in atompack_rows)
    assert {row["with_custom"]: row["atompack_batch_size"] for row in atompack_rows} == {
        False: 256,
        True: 1024,
    }


def test_bench_throughput_includes_lmdb_packed_for_custom_rows(
    monkeypatch,
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("write_benchmark_custom_lmdb_packed", "write_benchmark.py")

    monkeypatch.setattr(module, "_lmdb", object())
    monkeypatch.setattr(module, "_has_ase", False)
    monkeypatch.setattr(module, "hdf5_available", lambda: True)
    monkeypatch.setattr(module, "write_atompack", lambda path, n, atoms, *args, **kwargs: Path(path).write_bytes(b"atp"))
    monkeypatch.setattr(module, "write_hdf5_soa", lambda path, n, atoms, *args, **kwargs: Path(path).write_bytes(b"h5"))
    monkeypatch.setattr(module, "write_lmdb_packed", lambda path, n, atoms, *args, **kwargs: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(module, "write_lmdb_pickle", lambda path, n, atoms, *args, **kwargs: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(module, "_dir_size", lambda path: 0)

    rows = module.bench_throughput([16], n_mols=4, trials=1, scratch_dir=str(tmp_path))

    assert any(
        row["backend"] == "lmdb_packed" and row["with_custom"] is True
        for row in rows
    )
    assert any(
        row["backend"] == "hdf5_soa" and row["with_custom"] is True
        for row in rows
    )


def test_write_benchmark_batch_scaling_can_emit_structured_json(
    monkeypatch,
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("write_benchmark_batch_scaling_json", "write_benchmark.py")

    monkeypatch.setattr(
        module,
        "write_atompack",
        lambda path, n, atoms, *args, **kwargs: Path(path).write_bytes(b"atp"),
    )

    out_path = tmp_path / "write_batch_scaling.json"
    rc = module.main(
        [
            "--bench", "3",
            "--batch-scale-atoms", "256",
            "--batch-scale-sizes", "256", "1024",
            "--n-mols", "4",
            "--trials", "1",
            "--scratch-dir", str(tmp_path),
            "--out", str(out_path),
        ]
    )

    assert rc == 0
    rows = json.loads(out_path.read_text(encoding="utf-8"))
    assert rows
    assert {row["benchmark"] for row in rows} == {"write_batch_scaling"}
    assert {256, 1024} <= {row["batch_size"] for row in rows}
    assert any(row.get("is_auto_batch_size") for row in rows)
    assert all("trial_repeats_median" in row for row in rows)


def test_bench_throughput_repeats_short_trials_until_min_duration(
    monkeypatch,
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("write_benchmark_repeats", "write_benchmark.py")

    monkeypatch.setattr(module, "_lmdb", None)
    monkeypatch.setattr(module, "_has_ase", False)

    def write_atompack(path, n, atoms, *args, **kwargs):
        Path(path).write_bytes(b"atp")

    clock = {"t": 0.0}

    def perf_counter() -> float:
        value = clock["t"]
        clock["t"] += 0.03
        return value

    monkeypatch.setattr(module, "write_atompack", write_atompack)
    monkeypatch.setattr(module.time, "perf_counter", perf_counter)

    rows = module.bench_throughput(
        [16],
        n_mols=4,
        trials=1,
        scratch_dir=str(tmp_path),
        warmup_trials=0,
        min_trial_seconds=0.1,
        max_trial_repeats=7,
    )

    atompack_row = next(row for row in rows if row["backend"] == "atompack" and not row["with_custom"])
    assert atompack_row["trial_repeats_median"] == 4


def test_main_passes_repeat_controls_to_benchmarks(
    monkeypatch,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module("write_benchmark_repeat_cli", "write_benchmark.py")

    throughput_calls: list[dict[str, object]] = []
    batch_scaling_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        module,
        "bench_throughput",
        lambda *args, **kwargs: throughput_calls.append(kwargs) or [],
    )
    monkeypatch.setattr(
        module,
        "bench_atompack_batch_scaling",
        lambda *args, **kwargs: batch_scaling_calls.append(kwargs) or [],
    )

    rc = module.main(
        [
            "--bench", "1", "3",
            "--warmup-trials", "2",
            "--min-trial-seconds", "0.25",
            "--max-trial-repeats", "5",
        ]
    )

    assert rc == 0
    assert throughput_calls == [
        {
            "codec": "none",
            "atompack_batch_size": None,
            "atompack_target_batch_mib": module.DEFAULT_ATOMPACK_TARGET_BATCH_MIB,
            "warmup_trials": 2,
            "min_trial_seconds": 0.25,
            "max_trial_repeats": 5,
        }
    ]
    assert batch_scaling_calls == [
        {
            "codec": "none",
            "atompack_target_batch_mib": module.DEFAULT_ATOMPACK_TARGET_BATCH_MIB,
            "warmup_trials": 2,
            "min_trial_seconds": 0.25,
            "max_trial_repeats": 5,
        }
    ]
