# Copyright 2026 Entalpic
from __future__ import annotations

from pathlib import Path

import atompack.hf_upload_cli as cli


def test_configure_hf_xet_env_sets_defaults(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "hf-xet"
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    monkeypatch.setattr(cli, "DEFAULT_HF_XET_CACHE", cache_dir)

    high_performance, resolved_cache = cli._configure_hf_xet_env()

    assert high_performance == "1"
    assert resolved_cache == str(cache_dir)
    assert cache_dir.is_dir()


def test_configure_hf_xet_env_preserves_existing_values(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "custom-cache"
    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "0")
    monkeypatch.setenv("HF_XET_CACHE", str(cache_dir))

    high_performance, resolved_cache = cli._configure_hf_xet_env()

    assert high_performance == "0"
    assert resolved_cache == str(cache_dir)
    assert cache_dir.is_dir()


def test_resolve_path_in_repo_preserves_relative_layout(tmp_path: Path) -> None:
    repo_root = tmp_path / "atompack-v3"
    source = repo_root / "omat" / "train"
    source.mkdir(parents=True)

    assert cli._resolve_path_in_repo(source, repo_root) == "omat/train"


def test_resolve_path_in_repo_returns_repo_root_for_tree_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "atompack-v3"
    repo_root.mkdir()

    assert cli._resolve_path_in_repo(repo_root, repo_root) is None


def test_resolve_path_in_repo_falls_back_to_directory_name_outside_repo_root(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "atompack-v3"
    repo_root.mkdir()
    source = tmp_path / "custom-upload"
    source.mkdir()

    assert cli._resolve_path_in_repo(source, repo_root) == "custom-upload"


def test_main_uses_inferred_repo_path(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "atompack-v3"
    source = repo_root / "omat" / "val"
    source.mkdir(parents=True)
    (source / "data.atp").touch()
    calls: dict[str, object] = {}

    def fake_upload(**kwargs: object) -> str:
        calls.update(kwargs)
        return "upload://ok"

    monkeypatch.setattr(cli.atp.hub, "upload", fake_upload)
    monkeypatch.setattr(cli, "DEFAULT_HF_XET_CACHE", tmp_path / "hf-xet")
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)

    exit_code = cli.main([str(source), "--repo-root", str(repo_root)])

    assert exit_code == 0
    assert calls == {
        "source": source.resolve(),
        "repo_id": "LeMaterial/Atompack",
        "path_in_repo": "omat/val",
        "revision": None,
        "token": None,
        "commit_message": None,
        "create_repo": True,
        "private": False,
        "use_xet": None,
    }
    assert capsys.readouterr().out == (
        f"source: {source.resolve()}\n"
        "repo_id: LeMaterial/Atompack\n"
        "path_in_repo: omat/val\n"
        "HF_XET_HIGH_PERFORMANCE: 1\n"
        f"HF_XET_CACHE: {tmp_path / 'hf-xet'}\n"
        "upload_result: upload://ok\n"
    )


def test_main_allows_explicit_path_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "data.atp").touch()
    calls: dict[str, object] = {}

    def fake_upload(**kwargs: object) -> str:
        calls.update(kwargs)
        return "upload://ok"

    monkeypatch.setattr(cli.atp.hub, "upload", fake_upload)
    monkeypatch.setattr(cli, "DEFAULT_HF_XET_CACHE", tmp_path / "hf-xet")

    exit_code = cli.main([str(source), "--path-in-repo", "manual/path"])

    assert exit_code == 0
    assert calls["path_in_repo"] == "manual/path"
    assert calls["use_xet"] is None


def test_main_passes_explicit_xet_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "data.atp").touch()
    calls: dict[str, object] = {}

    def fake_upload(**kwargs: object) -> str:
        calls.update(kwargs)
        return "upload://ok"

    monkeypatch.setattr(cli.atp.hub, "upload", fake_upload)
    monkeypatch.setattr(cli, "DEFAULT_HF_XET_CACHE", tmp_path / "hf-xet")

    exit_code = cli.main([str(source), "--no-xet"])

    assert exit_code == 0
    assert calls["use_xet"] is False
