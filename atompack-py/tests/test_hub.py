# Copyright 2026 Entalpic
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import atompack
import numpy as np
import pytest


def _make_db(path: Path, energies: list[float]) -> None:
    db = atompack.Database(str(path), compression="none")
    for energy in energies:
        molecule = atompack.Molecule(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([6, 8], dtype=np.uint8),
        )
        molecule.energy = energy
        db.add_molecule(molecule)
    db.flush()


def _make_sparse_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)


def test_download_file_path_uses_hf_hub_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "exports" / "data.atp"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    calls = {"download": 0}

    class FakeHF:
        @staticmethod
        def hf_hub_download(**kwargs: object) -> str:
            calls["download"] += 1
            assert kwargs["filename"] == "exports/data.atp"
            return str(target)

        @staticmethod
        def snapshot_download(**kwargs: object) -> str:
            raise AssertionError("snapshot_download should not be called for a .atp file")

    monkeypatch.setattr(atompack.hub, "_require_hf_hub", lambda: FakeHF)

    resolved = atompack.hub.download("Org/repo", "exports/data.atp")

    assert calls["download"] == 1
    assert resolved == target


def test_download_directory_path_uses_snapshot_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_root = tmp_path / "snapshot"
    dataset_dir = snapshot_root / "exports" / "v1"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "part_0000.atp").touch()
    calls = {"snapshot": 0}

    class FakeHF:
        @staticmethod
        def hf_hub_download(**kwargs: object) -> str:
            raise AssertionError("hf_hub_download should not be called for a directory")

        @staticmethod
        def snapshot_download(**kwargs: object) -> str:
            calls["snapshot"] += 1
            assert kwargs["allow_patterns"] == [
                "exports/v1/*.atp",
                "exports/v1/**/*.atp",
                "exports/v1/manifest.json",
                "exports/v1/**/manifest.json",
            ]
            return str(snapshot_root)

    monkeypatch.setattr(atompack.hub, "_require_hf_hub", lambda: FakeHF)

    resolved = atompack.hub.download("Org/repo", "exports/v1")

    assert calls["snapshot"] == 1
    assert resolved == dataset_dir


def test_download_without_path_in_repo_returns_single_parent_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    shard_dir = snapshot_root / "omat" / "train"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "000.atp").touch()
    (shard_dir / "001.atp").touch()

    class FakeHF:
        @staticmethod
        def snapshot_download(**kwargs: object) -> str:
            assert kwargs["allow_patterns"] == [
                "*.atp",
                "**/*.atp",
                "manifest.json",
                "**/manifest.json",
            ]
            return str(snapshot_root)

    monkeypatch.setattr(atompack.hub, "_require_hf_hub", lambda: FakeHF)

    resolved = atompack.hub.download("Org/repo")

    assert resolved == shard_dir


def test_download_missing_path_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)

    class FakeHF:
        @staticmethod
        def snapshot_download(**kwargs: object) -> str:
            return str(snapshot_root)

    monkeypatch.setattr(atompack.hub, "_require_hf_hub", lambda: FakeHF)

    with pytest.raises(FileNotFoundError):
        atompack.hub.download("Org/repo", "missing/dataset")


def test_download_without_path_in_repo_raises_on_ambiguous_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    (snapshot_root / "omat" / "train").mkdir(parents=True, exist_ok=True)
    (snapshot_root / "omat" / "val").mkdir(parents=True, exist_ok=True)
    (snapshot_root / "omat" / "train" / "000.atp").touch()
    (snapshot_root / "omat" / "val" / "000.atp").touch()

    class FakeHF:
        @staticmethod
        def snapshot_download(**kwargs: object) -> str:
            return str(snapshot_root)

    monkeypatch.setattr(atompack.hub, "_require_hf_hub", lambda: FakeHF)

    with pytest.raises(ValueError, match="Multiple \\.atp files"):
        atompack.hub.download("Org/repo")


@pytest.mark.parametrize("path_in_repo", ["", "  ", "../train", "train/../val"])
def test_download_rejects_invalid_path_in_repo(path_in_repo: str) -> None:
    with pytest.raises(ValueError):
        atompack.hub.download("Org/repo", path_in_repo)


def test_upload_single_file_uses_upload_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "single.atp"
    source.touch()
    calls: dict[str, object] = {}

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            calls["token"] = token

        def create_repo(self, **kwargs: object) -> None:
            calls["create_repo"] = kwargs

        def upload_file(self, **kwargs: object) -> str:
            calls["upload_file"] = kwargs
            return "upload://single"

    class FakeHF:
        HfApi = FakeApi

    monkeypatch.setattr(atompack.hub, "_require_hf_hub", lambda: FakeHF)

    result = atompack.hub.upload(source, "Org/repo", path_in_repo="datasets")

    assert result == "upload://single"
    assert calls["create_repo"] == {
        "repo_id": "Org/repo",
        "repo_type": "dataset",
        "private": False,
        "exist_ok": True,
    }
    assert calls["upload_file"] == {
        "path_or_fileobj": str(source),
        "path_in_repo": "datasets/single.atp",
        "repo_id": "Org/repo",
        "repo_type": "dataset",
        "token": None,
    }


def test_upload_directory_uses_upload_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    nested_dir = dataset_dir / "train"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "000.atp").touch()
    (dataset_dir / "manifest.json").write_text("{}", encoding="utf-8")
    calls: dict[str, object] = {}

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            calls["token"] = token

        def create_repo(self, **kwargs: object) -> None:
            calls["create_repo"] = kwargs

        def upload_folder(self, **kwargs: object) -> str:
            calls["upload_folder"] = kwargs
            return "upload://folder"

    class FakeHF:
        HfApi = FakeApi

    monkeypatch.setattr(atompack.hub, "_require_hf_hub", lambda: FakeHF)

    result = atompack.hub.upload(dataset_dir, "Org/repo", path_in_repo="exports/v1")

    assert result == "upload://folder"
    assert calls["upload_folder"] == {
        "folder_path": str(dataset_dir),
        "allow_patterns": ["*.atp", "**/*.atp", "manifest.json", "**/manifest.json"],
        "path_in_repo": "exports/v1",
        "repo_id": "Org/repo",
        "repo_type": "dataset",
        "token": None,
    }


def test_upload_large_single_file_temporarily_disables_xet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "single.atp"
    _make_sparse_file(source, atompack.hub._XET_DISABLE_SINGLE_FILE_THRESHOLD_BYTES + 1)

    class FakeConstants:
        HF_HUB_DISABLE_XET = False

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            pass

        def create_repo(self, **kwargs: object) -> None:
            pass

        def upload_file(self, **kwargs: object) -> str:
            assert FakeHF.constants.HF_HUB_DISABLE_XET is True
            return "upload://single"

    class FakeHF:
        HfApi = FakeApi
        constants = FakeConstants

    monkeypatch.setattr(atompack.hub, "_require_hf_hub", lambda: FakeHF)

    result = atompack.hub.upload(source, "Org/repo")

    assert result == "upload://single"
    assert FakeHF.constants.HF_HUB_DISABLE_XET is False


def test_upload_large_single_file_can_force_xet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "single.atp"
    _make_sparse_file(source, atompack.hub._XET_DISABLE_SINGLE_FILE_THRESHOLD_BYTES + 1)

    class FakeConstants:
        HF_HUB_DISABLE_XET = False

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            pass

        def create_repo(self, **kwargs: object) -> None:
            pass

        def upload_file(self, **kwargs: object) -> str:
            assert FakeHF.constants.HF_HUB_DISABLE_XET is False
            return "upload://single"

    class FakeHF:
        HfApi = FakeApi
        constants = FakeConstants

    monkeypatch.setattr(atompack.hub, "_require_hf_hub", lambda: FakeHF)

    atompack.hub.upload(source, "Org/repo", use_xet=True)

    assert FakeHF.constants.HF_HUB_DISABLE_XET is False


def test_upload_directory_with_single_large_shard_temporarily_disables_xet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    _make_sparse_file(
        dataset_dir / "train" / "000.atp",
        atompack.hub._XET_DISABLE_SINGLE_FILE_THRESHOLD_BYTES + 1,
    )

    class FakeConstants:
        HF_HUB_DISABLE_XET = False

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            pass

        def create_repo(self, **kwargs: object) -> None:
            pass

        def upload_folder(self, **kwargs: object) -> str:
            assert FakeHF.constants.HF_HUB_DISABLE_XET is True
            return "upload://folder"

    class FakeHF:
        HfApi = FakeApi
        constants = FakeConstants

    monkeypatch.setattr(atompack.hub, "_require_hf_hub", lambda: FakeHF)

    atompack.hub.upload(dataset_dir, "Org/repo")

    assert FakeHF.constants.HF_HUB_DISABLE_XET is False


def test_upload_directory_with_multiple_shards_temporarily_disables_xet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    _make_sparse_file(dataset_dir / "train" / "000.atp", 16)
    _make_sparse_file(dataset_dir / "train" / "001.atp", 16)

    class FakeConstants:
        HF_HUB_DISABLE_XET = False

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            pass

        def create_repo(self, **kwargs: object) -> None:
            pass

        def upload_folder(self, **kwargs: object) -> str:
            assert FakeHF.constants.HF_HUB_DISABLE_XET is True
            return "upload://folder"

    class FakeHF:
        HfApi = FakeApi
        constants = FakeConstants

    monkeypatch.setattr(atompack.hub, "_require_hf_hub", lambda: FakeHF)

    atompack.hub.upload(dataset_dir, "Org/repo")

    assert FakeHF.constants.HF_HUB_DISABLE_XET is False


def test_upload_rejects_missing_or_empty_sources(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        atompack.hub.upload(tmp_path / "missing.atp", "Org/repo")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No \\.atp files"):
        atompack.hub.upload(empty_dir, "Org/repo")


@pytest.mark.parametrize("path_in_repo", ["", "  ", "../train", "train/../val"])
def test_upload_rejects_invalid_path_in_repo(tmp_path: Path, path_in_repo: str) -> None:
    source = tmp_path / "single.atp"
    source.touch()
    with pytest.raises(ValueError):
        atompack.hub.upload(source, "Org/repo", path_in_repo=path_in_repo)


def test_open_path_single_file_returns_reader(tmp_path: Path) -> None:
    source = tmp_path / "single.atp"
    _make_db(source, [-1.0, -2.0])

    reader = atompack.hub.open_path(source)

    assert len(reader) == 2
    assert reader[0].energy == pytest.approx(-1.0)
    assert [molecule.energy for molecule in reader.get_molecules([1, 0])] == pytest.approx(
        [-2.0, -1.0]
    )

    reader.close()
    with pytest.raises(ValueError, match="closed"):
        len(reader)


def test_open_path_directory_flattens_lexicographically(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _make_db(shard_dir / "b.atp", [-3.0])
    _make_db(shard_dir / "a.atp", [-1.0, -2.0])

    reader = atompack.hub.open_path(shard_dir)

    assert len(reader) == 3
    assert [reader[i].energy for i in range(len(reader))] == pytest.approx([-1.0, -2.0, -3.0])


def test_open_path_context_manager_closes_reader(tmp_path: Path) -> None:
    source = tmp_path / "single.atp"
    _make_db(source, [-1.0])

    with atompack.hub.open_path(source) as reader:
        assert reader[0].energy == pytest.approx(-1.0)

    with pytest.raises(ValueError, match="closed"):
        reader.get_molecule(0)


def test_open_from_hf_uses_download_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    _make_db(shard_dir / "000.atp", [-1.0])
    _make_db(shard_dir / "001.atp", [-2.0])

    monkeypatch.setattr(atompack.hub, "download", lambda *args, **kwargs: shard_dir)

    reader = atompack.hub.open("Org/repo", "omat/train")

    assert len(reader) == 2
    assert [reader[0].energy, reader[1].energy] == pytest.approx([-1.0, -2.0])


def test_reader_to_ase_batch_preserves_requested_order(tmp_path: Path) -> None:
    pytest.importorskip("ase")
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    _make_db(shard_dir / "a.atp", [-1.0, -2.0])
    _make_db(shard_dir / "b.atp", [-3.0])

    reader = atompack.hub.open_path(shard_dir)
    atoms_list = reader.to_ase_batch([2, 0])

    assert len(atoms_list) == 2
    assert atoms_list[0].get_potential_energy() == pytest.approx(-3.0)
    assert atoms_list[1].get_potential_energy() == pytest.approx(-1.0)


def test_import_atompack_does_not_require_huggingface_hub(tmp_path: Path) -> None:
    python_src = Path(__file__).resolve().parents[1] / "python"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        "def _blocked(name, *args, **kwargs):\n"
        "    if name == 'huggingface_hub' or name.startswith('huggingface_hub.'):\n"
        "        raise ModuleNotFoundError('blocked by test')\n"
        "    return _real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _blocked\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(python_src)])

    res = subprocess.run(
        [sys.executable, "-c", "import atompack; assert hasattr(atompack, 'hub')"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    assert res.stdout == ""
    assert res.stderr == ""


def test_hub_functions_raise_clear_error_without_optional_dependency(tmp_path: Path) -> None:
    python_src = Path(__file__).resolve().parents[1] / "python"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        "def _blocked(name, *args, **kwargs):\n"
        "    if name == 'huggingface_hub' or name.startswith('huggingface_hub.'):\n"
        "        raise ModuleNotFoundError('blocked by test')\n"
        "    return _real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _blocked\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(python_src)])

    res = subprocess.run(
        [sys.executable, "-c", "import atompack; atompack.hub.download('Org/repo')"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert res.returncode != 0
    assert "reinstall atompack" in res.stderr
    assert "huggingface_hub>=0.24" in res.stderr
