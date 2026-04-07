# Copyright 2026 Entalpic
from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import atompack
import numpy as np
import pytest


class _FakeLegacyMolecule:
    def __init__(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        *,
        energy: float | None = None,
        forces: np.ndarray | None = None,
        charges: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
        cell: np.ndarray | None = None,
        stress: np.ndarray | None = None,
        pbc: tuple[bool, bool, bool] | None = None,
        name: str | None = None,
        properties: dict[str, object] | None = None,
    ) -> None:
        self.positions = positions
        self.atomic_numbers = atomic_numbers
        self.energy = energy
        self.forces = forces
        self.charges = charges
        self.velocities = velocities
        self.cell = cell
        self.stress = stress
        self.pbc = pbc
        self.name = name
        self._properties = properties or {}

    def property_keys(self) -> list[str]:
        return list(self._properties.keys())

    def get_property(self, key: str):
        return self._properties[key]


class _FakeLegacyDatabase:
    _registry: dict[str, list[_FakeLegacyMolecule]] = {}
    open_calls = 0
    open_mmap_calls = 0
    getitem_calls = 0
    get_molecules_calls = 0

    def __init__(self, records: list[_FakeLegacyMolecule]) -> None:
        self._records = records

    @classmethod
    def open(cls, path: str):
        cls.open_calls += 1
        return cls(cls._registry[path])

    @classmethod
    def open_mmap(cls, path: str):
        cls.open_mmap_calls += 1
        return cls(cls._registry[path])

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> _FakeLegacyMolecule:
        type(self).getitem_calls += 1
        return self._records[index]

    def get_molecules(self, indices: list[int]) -> list[_FakeLegacyMolecule]:
        type(self).get_molecules_calls += 1
        return [self._records[index] for index in indices]


def _legacy_module() -> ModuleType:
    return SimpleNamespace(Database=_FakeLegacyDatabase)


def _register_legacy(path: Path, molecules: list[_FakeLegacyMolecule]) -> None:
    path.write_bytes(b"legacy")
    _FakeLegacyDatabase._registry[str(path)] = molecules


def _reset_legacy_db_counters() -> None:
    _FakeLegacyDatabase.open_calls = 0
    _FakeLegacyDatabase.open_mmap_calls = 0
    _FakeLegacyDatabase.getitem_calls = 0
    _FakeLegacyDatabase.get_molecules_calls = 0


def _spy_atompack_module() -> tuple[ModuleType, dict[str, int]]:
    counters = {"add_arrays_batch": 0, "add_molecules": 0}

    class _SpyDatabase:
        def __init__(self, *args, **kwargs) -> None:
            self._inner = atompack.Database(*args, **kwargs)

        @staticmethod
        def open(*args, **kwargs):
            return atompack.Database.open(*args, **kwargs)

        def add_arrays_batch(self, *args, **kwargs) -> None:
            counters["add_arrays_batch"] += 1
            self._inner.add_arrays_batch(*args, **kwargs)

        def add_molecules(self, *args, **kwargs) -> None:
            counters["add_molecules"] += 1
            self._inner.add_molecules(*args, **kwargs)

        def flush(self) -> None:
            self._inner.flush()

    return SimpleNamespace(Database=_SpyDatabase, Molecule=atompack.Molecule), counters


def _builtin_molecule(
    energy: float,
    *,
    n_atoms: int = 2,
    with_name: bool = True,
) -> _FakeLegacyMolecule:
    positions = np.arange(n_atoms * 3, dtype=np.float32).reshape(n_atoms, 3) + energy
    atomic_numbers = np.arange(1, n_atoms + 1, dtype=np.uint8)
    return _FakeLegacyMolecule(
        positions,
        atomic_numbers,
        energy=energy,
        forces=np.full((n_atoms, 3), energy, dtype=np.float32),
        charges=np.linspace(-0.2, 0.2, n_atoms, dtype=np.float64),
        velocities=np.full((n_atoms, 3), energy + 1.0, dtype=np.float32),
        cell=np.eye(3, dtype=np.float64) * (energy + 10.0),
        stress=np.eye(3, dtype=np.float64) * (energy + 20.0),
        pbc=(True, False, True),
        name=f"mol-{energy}" if with_name else None,
    )


def test_convert_file_mode_roundtrip_and_fast_path(
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    _reset_legacy_db_counters()
    module = load_benchmark_module("convert_legacy_atompack_test", "convert_legacy_atompack.py")
    source = tmp_path / "legacy.atp"
    target = tmp_path / "converted.atp"
    molecules = [_builtin_molecule(-1.0, n_atoms=2), _builtin_molecule(-2.0, n_atoms=3)]
    _register_legacy(source, molecules)

    atompack_spy, counters = _spy_atompack_module()
    exit_code = module.main(
        [
            str(source),
            "--output",
            str(target),
            "--overwrite",
            "--batch-size",
            "8",
            "--report-every",
            "0",
        ],
        legacy_module=_legacy_module(),
        atompack_module=atompack_spy,
    )

    assert exit_code == 0
    assert counters["add_arrays_batch"] == 2
    assert counters["add_molecules"] == 0
    assert _FakeLegacyDatabase.open_mmap_calls == 1
    assert _FakeLegacyDatabase.get_molecules_calls == 1
    assert _FakeLegacyDatabase.getitem_calls == 0

    reopened = atompack.Database.open(str(target))
    assert len(reopened) == 2
    first = reopened[0]
    second = reopened[1]

    assert first.energy == pytest.approx(-1.0)
    np.testing.assert_allclose(first.forces, np.full((2, 3), -1.0, dtype=np.float32))
    assert first.pbc == (True, False, True)
    assert second.energy == pytest.approx(-2.0)
    np.testing.assert_array_equal(second.atomic_numbers, np.array([1, 2, 3], dtype=np.uint8))


def test_convert_directory_mode_uses_manifest_layout(
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    _reset_legacy_db_counters()
    module = load_benchmark_module(
        "convert_legacy_atompack_manifest_test",
        "convert_legacy_atompack.py",
    )
    dataset_dir = tmp_path / "legacy_dataset"
    shard_a = dataset_dir / "part_0000.atp"
    shard_b = dataset_dir / "nested" / "part_0001.atp"
    shard_b.parent.mkdir(parents=True)
    _register_legacy(shard_a, [_builtin_molecule(-1.0)])
    _register_legacy(shard_b, [_builtin_molecule(-2.0)])
    (dataset_dir / "manifest.json").write_text(
        '{"output_files": ["part_0000.atp", "nested/part_0001.atp"]}'
    )

    output_dir = tmp_path / "converted_dataset"
    exit_code = module.main(
        [str(dataset_dir), "--output", str(output_dir), "--overwrite", "--report-every", "0"],
        legacy_module=_legacy_module(),
    )

    assert exit_code == 0
    assert (output_dir / "part_0000.atp").is_file()
    assert (output_dir / "nested" / "part_0001.atp").is_file()
    assert len(atompack.Database.open(str(output_dir / "part_0000.atp"))) == 1
    assert len(atompack.Database.open(str(output_dir / "nested" / "part_0001.atp"))) == 1
    assert _FakeLegacyDatabase.open_mmap_calls == 2
    assert _FakeLegacyDatabase.get_molecules_calls == 2
    assert _FakeLegacyDatabase.getitem_calls == 0


def test_convert_custom_properties_use_batched_arrays_write(
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    _reset_legacy_db_counters()
    module = load_benchmark_module(
        "convert_legacy_atompack_custom_test",
        "convert_legacy_atompack.py",
    )
    source = tmp_path / "legacy_custom.atp"
    target = tmp_path / "legacy_custom.soa.atp"
    molecules = [
        _FakeLegacyMolecule(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([6, 8], dtype=np.uint8),
            energy=-3.0,
            properties={
                "bandgap": 1.5,
                "tag": "train",
                "ids": np.array([1, 2], dtype=np.int64),
                "vec3": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
            },
        )
    ]
    _register_legacy(source, molecules)

    atompack_spy, counters = _spy_atompack_module()
    exit_code = module.main(
        [str(source), "--overwrite", "--report-every", "0"],
        legacy_module=_legacy_module(),
        atompack_module=atompack_spy,
    )

    assert exit_code == 0
    assert counters["add_arrays_batch"] == 1
    assert counters["add_molecules"] == 0
    assert _FakeLegacyDatabase.open_mmap_calls == 1
    assert _FakeLegacyDatabase.get_molecules_calls == 1
    assert _FakeLegacyDatabase.getitem_calls == 0

    reopened = atompack.Database.open(str(target))
    molecule = reopened[0]
    assert molecule.energy == pytest.approx(-3.0)
    assert molecule.get_property("tag") == "train"
    assert molecule.get_property("bandgap") == pytest.approx(1.5)
    np.testing.assert_array_equal(molecule.get_property("ids"), np.array([1, 2], dtype=np.int64))
    np.testing.assert_allclose(
        molecule.get_property("vec3"),
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
    )


def test_convert_mixed_builtin_and_custom_batches_still_use_batched_writes(
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    _reset_legacy_db_counters()
    module = load_benchmark_module(
        "convert_legacy_atompack_mixed_batch_test",
        "convert_legacy_atompack.py",
    )
    source = tmp_path / "legacy_mixed.atp"
    target = tmp_path / "legacy_mixed.soa.atp"
    molecules = [
        _builtin_molecule(-1.0),
        _FakeLegacyMolecule(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([6, 8], dtype=np.uint8),
            energy=-3.0,
            properties={
                "bandgap": 1.5,
                "tag": "train",
            },
        ),
    ]
    _register_legacy(source, molecules)

    atompack_spy, counters = _spy_atompack_module()
    exit_code = module.main(
        [str(source), "--overwrite", "--report-every", "0", "--batch-size", "8"],
        legacy_module=_legacy_module(),
        atompack_module=atompack_spy,
    )

    assert exit_code == 0
    assert counters["add_arrays_batch"] == 2
    assert counters["add_molecules"] == 0
    assert _FakeLegacyDatabase.open_mmap_calls == 1
    assert _FakeLegacyDatabase.get_molecules_calls == 1
    assert _FakeLegacyDatabase.getitem_calls == 0

    reopened = atompack.Database.open(str(target))
    assert len(reopened) == 2
    assert reopened[0].energy == pytest.approx(-1.0)
    assert reopened[1].get_property("tag") == "train"
    assert reopened[1].get_property("bandgap") == pytest.approx(1.5)


def test_plan_jobs_rejects_existing_output_without_overwrite(
    tmp_path: Path,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module(
        "convert_legacy_atompack_plan_jobs_test",
        "convert_legacy_atompack.py",
    )
    source = tmp_path / "legacy.atp"
    source.write_bytes(b"legacy")
    target = tmp_path / "existing.atp"
    target.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        module.plan_jobs(source, target, overwrite=False)


def test_convert_jobs_uses_parallel_shard_workers_when_modules_are_importable(
    monkeypatch: pytest.MonkeyPatch,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module(
        "convert_legacy_atompack_parallel_dispatch_test",
        "convert_legacy_atompack.py",
    )
    legacy_module = ModuleType("atompack_legacy_parallel_stub")
    legacy_module.Database = _FakeLegacyDatabase
    jobs = [
        module.ShardJob(Path("/tmp/legacy_a.atp"), Path("/tmp/out_a.atp"), "a"),
        module.ShardJob(Path("/tmp/legacy_b.atp"), Path("/tmp/out_b.atp"), "b"),
    ]
    calls: dict[str, object] = {}

    def _fake_parallel(parallel_jobs, **kwargs):
        calls["jobs"] = parallel_jobs
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(module, "_convert_jobs_parallel", _fake_parallel)

    exit_code = module.convert_jobs(
        jobs,
        legacy_module=legacy_module,
        atompack_module=atompack,
        workers=3,
    )

    assert exit_code == 0
    assert calls["jobs"] == jobs
    assert calls["legacy_module_name"] == "atompack_legacy_parallel_stub"
    assert calls["atompack_module_name"] == "atompack"
    assert calls["workers"] == 3


def test_convert_jobs_falls_back_to_sequential_when_modules_are_not_importable(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module(
        "convert_legacy_atompack_parallel_fallback_test",
        "convert_legacy_atompack.py",
    )
    jobs = [
        module.ShardJob(Path("/tmp/legacy_a.atp"), Path("/tmp/out_a.atp"), "a"),
        module.ShardJob(Path("/tmp/legacy_b.atp"), Path("/tmp/out_b.atp"), "b"),
    ]
    seen: list[module.ShardJob] = []

    def _fake_convert_shard(job, **kwargs):
        seen.append(job)
        return module.ShardStats()

    monkeypatch.setattr(module, "convert_shard", _fake_convert_shard)

    exit_code = module.convert_jobs(
        jobs,
        legacy_module=_legacy_module(),
        atompack_module=atompack,
        workers=2,
    )

    assert exit_code == 0
    assert seen == jobs
    assert "falling back to sequential conversion" in capsys.readouterr().err


def test_convert_jobs_rejects_non_positive_workers(
    load_benchmark_module,
) -> None:
    module = load_benchmark_module(
        "convert_legacy_atompack_workers_validation_test",
        "convert_legacy_atompack.py",
    )

    with pytest.raises(ValueError, match="workers"):
        module.convert_jobs(
            [],
            legacy_module=ModuleType("atompack_legacy_parallel_stub"),
            workers=0,
        )


def test_load_legacy_atompack_reports_missing_package(
    monkeypatch: pytest.MonkeyPatch,
    load_benchmark_module,
) -> None:
    module = load_benchmark_module(
        "convert_legacy_atompack_missing_package_test",
        "convert_legacy_atompack.py",
    )

    def _missing(name: str):
        raise ModuleNotFoundError(f"No module named {name}", name=name)

    monkeypatch.setattr(module.importlib, "import_module", _missing)

    with pytest.raises(SystemExit, match="atompack_legacy"):
        module.load_legacy_atompack()
