"""Type stubs for atompack Hugging Face helpers."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, Sequence

from . import Molecule

class AtompackReader:
    """
    Read-only view over one atompack file or a lexicographically ordered shard set.

    The reader presents multiple shard files as one logical dataset, supports
    negative indexing, and preserves the input order for batched fetches.
    """

    def __enter__(self) -> AtompackReader: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...
    def __len__(self) -> int:
        """Return the total number of molecules across all opened files."""
        ...
    def __getitem__(self, index: int) -> Molecule:
        """
        Fetch one molecule by index.

        Negative indices are supported and resolve against the logical merged
        dataset, not within a single shard.
        """
        ...
    def get_molecule(self, index: int) -> Molecule:
        """
        Fetch one molecule by global index across the underlying shard set.

        Parameters
        ----------
        index : int
            Molecule index in the merged logical reader. Negative indices are
            supported.
        """
        ...
    def get_molecules(self, indices: Sequence[int]) -> list[Molecule]:
        """
        Fetch many molecules by global index while preserving input order.

        Parameters
        ----------
        indices : sequence of int
            Molecule indices in the merged logical reader. Negative indices are
            supported.
        """
        ...
    def to_ase_batch(
        self,
        indices: Sequence[int] | None = None,
        *,
        attach_calc: bool = True,
        calc_mode: str = "singlepoint",
        copy_info: bool = True,
        copy_arrays: bool = True,
    ) -> list[object]:
        """
        Convert a selection of molecules to ASE Atoms objects in one call.

        Passing ``indices=None`` converts the whole logical reader. The output
        order matches the requested index order.
        """
        ...
    def close(self) -> None:
        """Close the underlying databases and invalidate the reader."""
        ...

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
    """
    Download a single ``.atp`` file or shard directory from a Hugging Face repo.

    When ``path_in_repo`` points to one file, the returned path is that file.
    Otherwise the returned path is the downloaded directory containing the
    selected shard set.
    """
    ...

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
    """
    Upload a local ``.atp`` file or shard directory to a Hugging Face repo.

    By default, large single-shard uploads may temporarily disable Xet so the
    Hub can use multipart LFS upload. Pass ``use_xet=True`` to force Xet, or
    ``use_xet=False`` to disable it explicitly.
    """
    ...

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
    """
    Download atompack data from Hugging Face and return a read-only reader.

    This is a convenience wrapper around :func:`download` followed by
    :func:`open_path`.
    """
    ...

def open_path(source: str | Path) -> AtompackReader:
    """
    Open a local ``.atp`` file or shard directory as a read-only reader.

    Single files are opened directly. Directories are scanned recursively for
    ``.atp`` files and then exposed as one logical lexicographically ordered
    reader.
    """
    ...

__all__: list[str]
