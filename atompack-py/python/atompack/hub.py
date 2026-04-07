from __future__ import annotations

from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from ._atompack_rs import PyAtomDatabase as Database
from ._atompack_rs import PyMolecule as Molecule
from .ase_bridge import to_ase_batch as _to_ase_batch

_DIRECTORY_ALLOW_PATTERNS = ["*.atp", "**/*.atp", "manifest.json", "**/manifest.json"]
_XET_DISABLE_SINGLE_FILE_THRESHOLD_BYTES = 512 * 1024 * 1024


def _require_hf_hub() -> Any:
    try:
        import huggingface_hub  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "huggingface_hub is expected to be installed with atompack.\n"
            "Your environment looks out of sync; reinstall atompack with "
            "'atompack[hf]' or install 'huggingface_hub>=0.24'."
        ) from exc
    return huggingface_hub


def _normalize_repo_relative_path(path: str, *, name: str = "path_in_repo") -> str:
    normalized = path.strip().strip("/")
    if not normalized:
        raise ValueError(f"{name} cannot be empty")

    parts = [part for part in PurePosixPath(normalized).parts if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"Parent traversal is not allowed in {name}: {path!r}")
    if not parts:
        raise ValueError(f"{name} cannot resolve to repository root")
    return "/".join(parts)


def _normalize_optional_repo_relative_path(path: str | None) -> str | None:
    if path is None:
        return None
    return _normalize_repo_relative_path(path)


def _snapshot_allow_patterns(path_in_repo: str | None) -> list[str]:
    if path_in_repo is None:
        return list(_DIRECTORY_ALLOW_PATTERNS)

    return [
        f"{path_in_repo}/*.atp",
        f"{path_in_repo}/**/*.atp",
        f"{path_in_repo}/manifest.json",
        f"{path_in_repo}/**/manifest.json",
    ]


def _collect_atp_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".atp" else []
    return sorted(
        (candidate for candidate in path.glob("**/*.atp") if candidate.is_file()),
        key=lambda candidate: candidate.as_posix(),
    )


def _resolve_snapshot_root(snapshot_root: Path, path_in_repo: str | None) -> Path:
    if path_in_repo is not None:
        resolved = snapshot_root / path_in_repo
        if not resolved.exists():
            raise FileNotFoundError(
                f"Resolved path does not exist in downloaded snapshot: {resolved}"
            )
        if not _collect_atp_files(resolved):
            raise FileNotFoundError(f"No .atp files found under downloaded path: {resolved}")
        return resolved

    atp_files = sorted(snapshot_root.glob("**/*.atp"), key=lambda candidate: candidate.as_posix())
    if len(atp_files) == 0:
        raise FileNotFoundError(
            "No .atp files found in Hugging Face repo snapshot. "
            "Provide path_in_repo to a specific dataset file or directory."
        )
    if len(atp_files) == 1:
        return atp_files[0]

    parent_dirs = {path.parent for path in atp_files}
    if len(parent_dirs) == 1:
        return next(iter(parent_dirs))

    raise ValueError(
        "Multiple .atp files were found across multiple directories. "
        "Please set path_in_repo to a specific dataset file or directory."
    )


def _resolve_upload_path_in_repo(source: Path, path_in_repo: str | None) -> str:
    if path_in_repo is None:
        return source.name
    if path_in_repo.endswith(".atp"):
        return path_in_repo
    return f"{path_in_repo}/{source.name}"


def _upload_shard_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix == ".atp" else []
    return _collect_atp_files(source)


def _should_disable_xet_for_upload(source: Path, *, use_xet: bool | None) -> bool:
    if use_xet is not None:
        return not use_xet

    shard_files = _upload_shard_files(source)
    if not shard_files:
        return False
    if len(shard_files) > 1:
        return True
    return shard_files[0].stat().st_size >= _XET_DISABLE_SINGLE_FILE_THRESHOLD_BYTES


@contextmanager
def _temporary_xet_disable(hf: Any, *, disabled: bool):
    if not disabled:
        yield
        return

    constants = getattr(hf, "constants", None)
    if constants is None or not hasattr(constants, "HF_HUB_DISABLE_XET"):
        yield
        return

    previous = constants.HF_HUB_DISABLE_XET
    constants.HF_HUB_DISABLE_XET = True
    try:
        yield
    finally:
        constants.HF_HUB_DISABLE_XET = previous


class AtompackReader:
    """Read-only view over one atompack file or a lexicographically ordered shard set."""

    def __init__(self, paths: Sequence[Path], databases: Sequence[Database]):
        if len(paths) == 0:
            raise ValueError("AtompackReader requires at least one database")
        if len(paths) != len(databases):
            raise ValueError("paths and databases must have the same length")

        self._paths = [Path(path) for path in paths]
        self._databases = list(databases)
        self._lengths = [len(database) for database in self._databases]
        self._offsets: list[int] = []
        total = 0
        for length in self._lengths:
            self._offsets.append(total)
            total += length
        self._total_length = total
        self._closed = False

    def __enter__(self) -> AtompackReader:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __len__(self) -> int:
        self._ensure_open()
        return self._total_length

    def __getitem__(self, index: int) -> Molecule:
        return self.get_molecule(index)

    def get_molecule(self, index: int) -> Molecule:
        db_index, local_index = self._locate(index)
        return self._databases[db_index][local_index]

    def get_molecules(self, indices: list[int]) -> list[Molecule]:
        self._ensure_open()
        if not indices:
            return []

        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, index in enumerate(indices):
            db_index, local_index = self._locate(index)
            grouped.setdefault(db_index, []).append((output_index, local_index))

        molecules: list[Molecule | None] = [None] * len(indices)
        for db_index, pairs in grouped.items():
            local_indices = [local_index for _, local_index in pairs]
            batch = self._databases[db_index].get_molecules(local_indices)
            for (output_index, _), molecule in zip(pairs, batch):
                molecules[output_index] = molecule

        return [molecule for molecule in molecules if molecule is not None]

    def to_ase_batch(
        self,
        indices: list[int] | None = None,
        *,
        attach_calc: bool = True,
        calc_mode: str = "singlepoint",
        copy_info: bool = True,
        copy_arrays: bool = True,
    ) -> list[object]:
        self._ensure_open()
        selected_indices = list(range(self._total_length)) if indices is None else list(indices)
        if not selected_indices:
            return []

        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, index in enumerate(selected_indices):
            db_index, local_index = self._locate(index)
            grouped.setdefault(db_index, []).append((output_index, local_index))

        atoms_list: list[object | None] = [None] * len(selected_indices)
        for db_index, pairs in grouped.items():
            local_indices = [local_index for _, local_index in pairs]
            database = self._databases[db_index]
            batch_convert = getattr(database, "to_ase_batch", None)
            if callable(batch_convert):
                batch = batch_convert(
                    local_indices,
                    attach_calc=attach_calc,
                    calc_mode=calc_mode,
                    copy_info=copy_info,
                    copy_arrays=copy_arrays,
                )
            else:
                batch = _to_ase_batch(
                    database,
                    local_indices,
                    attach_calc=attach_calc,
                    calc_mode=calc_mode,
                    copy_info=copy_info,
                    copy_arrays=copy_arrays,
                )
            for (output_index, _), atoms in zip(pairs, batch):
                atoms_list[output_index] = atoms

        return [atoms for atoms in atoms_list if atoms is not None]

    def close(self) -> None:
        if self._closed:
            return

        for database in self._databases:
            close_fn = getattr(database, "close", None)
            if callable(close_fn):
                close_fn()

        self._databases.clear()
        self._paths.clear()
        self._lengths.clear()
        self._offsets.clear()
        self._total_length = 0
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("AtompackReader is closed")

    def _locate(self, index: int) -> tuple[int, int]:
        self._ensure_open()
        if not isinstance(index, int):
            raise TypeError("AtompackReader indices must be integers")

        normalized = index
        if normalized < 0:
            normalized += self._total_length
        if normalized < 0 or normalized >= self._total_length:
            raise IndexError(
                f"Index {index} out of bounds for reader of length {self._total_length}"
            )

        db_index = bisect_right(self._offsets, normalized) - 1
        local_index = normalized - self._offsets[db_index]
        return db_index, local_index


def download(
    repo_id: str,
    path_in_repo: str | None = None,
    *,
    revision: str | None = None,
    repo_type: str = "dataset",
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> Path:
    """Download a single .atp file or shard directory from a Hugging Face repo."""
    normalized = _normalize_optional_repo_relative_path(path_in_repo)
    hf = _require_hf_hub()
    common_kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "token": token,
        "cache_dir": cache_dir,
        "local_dir": local_dir,
        "local_files_only": local_files_only,
        "force_download": force_download,
    }

    if normalized is not None and normalized.endswith(".atp"):
        return Path(hf.hf_hub_download(filename=normalized, **common_kwargs))

    snapshot_root = Path(
        hf.snapshot_download(
            allow_patterns=_snapshot_allow_patterns(normalized),
            **common_kwargs,
        )
    )
    return _resolve_snapshot_root(snapshot_root, normalized)


def upload(
    source: str | Path,
    repo_id: str,
    path_in_repo: str | None = None,
    *,
    repo_type: str = "dataset",
    revision: str | None = None,
    token: str | None = None,
    commit_message: str | None = None,
    create_repo: bool = True,
    private: bool = False,
    use_xet: bool | None = None,
) -> Any:
    """Upload a local .atp file or shard directory to a Hugging Face repo.

    By default, large single-shard uploads temporarily disable Xet so the Hub
    can use multipart LFS upload, which is typically much faster for large
    compressed `.atp` files. Pass `use_xet=True` to force Xet, or
    `use_xet=False` to disable it explicitly.
    """
    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    normalized = _normalize_optional_repo_relative_path(path_in_repo)
    if source_path.is_file() and source_path.suffix != ".atp":
        raise ValueError(f"Expected a .atp file for upload, got: {source_path.name}")

    if source_path.is_dir():
        shard_files = _collect_atp_files(source_path)
        if not shard_files:
            raise FileNotFoundError(f"No .atp files found under directory: {source_path}")

    hf = _require_hf_hub()
    api = hf.HfApi(token=token)

    if create_repo:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )

    common_kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "token": token,
    }
    if revision is not None:
        common_kwargs["revision"] = revision
    if commit_message is not None:
        common_kwargs["commit_message"] = commit_message

    disable_xet = _should_disable_xet_for_upload(source_path, use_xet=use_xet)
    with _temporary_xet_disable(hf, disabled=disable_xet):
        if source_path.is_file():
            return api.upload_file(
                path_or_fileobj=str(source_path),
                path_in_repo=_resolve_upload_path_in_repo(source_path, normalized),
                **common_kwargs,
            )

        return api.upload_folder(
            folder_path=str(source_path),
            allow_patterns=list(_DIRECTORY_ALLOW_PATTERNS),
            **common_kwargs,
            **({"path_in_repo": normalized} if normalized is not None else {}),
        )


def open(
    repo_id: str,
    path_in_repo: str | None = None,
    *,
    revision: str | None = None,
    repo_type: str = "dataset",
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> AtompackReader:
    """Download atompack data from Hugging Face and return a read-only reader."""
    local_path = download(
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        revision=revision,
        repo_type=repo_type,
        token=token,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    return open_path(local_path)


def open_path(source: str | Path) -> AtompackReader:
    """Open a local .atp file or shard directory as a read-only reader."""
    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    if source_path.is_file():
        if source_path.suffix != ".atp":
            raise ValueError(f"Expected a .atp file, got: {source_path.name}")
        return AtompackReader(
            paths=[source_path],
            databases=[Database.open(str(source_path), mmap=True)],
        )

    shard_files = _collect_atp_files(source_path)
    if not shard_files:
        raise FileNotFoundError(f"No .atp files found under directory: {source_path}")

    return AtompackReader(
        paths=shard_files,
        databases=[Database.open(str(path), mmap=True) for path in shard_files],
    )


__all__ = [
    "AtompackReader",
    "download",
    "upload",
    "open",
    "open_path",
]
