"""Type stubs for the internal PyO3 extension surface."""

from __future__ import annotations

from typing import Any, Sequence, overload

import numpy as np

class PyAtom:
    """
    Low-level PyO3-backed atom with 3D coordinates and atomic number.

    Parameters
    ----------
    x : float
        X coordinate in Angstroms
    y : float
        Y coordinate in Angstroms
    z : float
        Z coordinate in Angstroms
    atomic_number : int
        Atomic number (1=H, 6=C, 8=O, etc.)

    Attributes
    ----------
    atomic_number : int
        The atomic number
    """

    def __init__(self, x: float, y: float, z: float, atomic_number: int) -> None: ...
    def position(self) -> tuple[float, float, float]:
        """
        Get the position as a tuple.

        Returns
        -------
        tuple of float
            ``(x, y, z)`` coordinates in Angstroms
        """
        ...

    @property
    def atomic_number(self) -> int:
        """
        Get the atomic number.

        Returns
        -------
        int
            Atomic number (1=H, 6=C, 8=O, etc.)
        """
        ...

    def distance_to(self, other: PyAtom) -> float:
        """
        Calculate distance to another atom.

        Parameters
        ----------
        other : PyAtom
            The other atom

        Returns
        -------
        float
            Distance in Angstroms
        """
        ...

    def __repr__(self) -> str: ...

class PyMolecule:
    """
    Low-level PyO3-backed molecule with optional builtin and custom properties.

    Parameters
    ----------
    positions : ndarray of float32, shape (n_atoms, 3)
        Atomic positions
    atomic_numbers : ndarray of uint8, shape (n_atoms,)
        Atomic numbers

    Attributes
    ----------
    forces : ndarray of float32, shape (n_atoms, 3), optional
        Per-atom forces
    energy : float, optional
        Total energy
    charges : ndarray of float64, shape (n_atoms,), optional
        Per-atom partial charges
    velocities : ndarray of float32, shape (n_atoms, 3), optional
        Per-atom velocities
    cell : ndarray of float64, shape (3, 3), optional
        Unit cell for periodic systems
    positions : ndarray of float32, shape (n_atoms, 3)
        Atomic positions (read-only)
    atomic_numbers : ndarray of uint8, shape (n_atoms,)
        Atomic numbers (read-only)
    """

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
    ) -> None: ...
    @staticmethod
    def from_arrays(
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
    ) -> PyMolecule:
        """
        Create a molecule from numpy arrays (fast path).

        Parameters
        ----------
        positions : ndarray of float32, shape (n_atoms, 3)
            Atomic positions (Angstroms)
        atomic_numbers : ndarray of uint8, shape (n_atoms,)
            Atomic numbers
        """
        ...

    def __len__(self) -> int:
        """
        Get the number of atoms.

        Returns
        -------
        int
            Number of atoms in the molecule
        """
        ...
    def atoms(self) -> list[PyAtom]:
        """
        Get the list of atoms.

        Returns
        -------
        list of PyAtom
            All atoms in the molecule
        """
        ...

    def to_owned(self) -> PyMolecule:
        """
        Materialize the molecule into an owned, self-contained object.

        This is useful before pickling or sending a database-fetched lazy view
        across process boundaries.
        """
        ...

    def _ase_builtin_tuple_fast(
        self,
        *,
        copy_info: bool = True,
        copy_arrays: bool = True,
    ) -> tuple[Any, ...]:
        """Return the packed ASE payload tuple used by the Python bridge."""
        ...

    def _ase_payload(
        self,
        *,
        copy_info: bool = True,
        copy_arrays: bool = True,
    ) -> tuple[Any, ...]:
        """Return the packed ASE payload tuple including custom properties."""
        ...

    @property
    def forces(self) -> np.ndarray | None:
        """
        Per-atom forces.

        Returns
        -------
        ndarray of float32, shape (n_atoms, 3) or None
            Forces on each atom, or None if not set
        """
        ...

    @forces.setter
    def forces(self, forces: np.ndarray) -> None: ...
    @property
    def energy(self) -> float | None:
        """
        Total energy.

        Returns
        -------
        float or None
            Energy value, or None if not set
        """
        ...

    @energy.setter
    def energy(self, energy: float | None) -> None: ...
    @property
    def charges(self) -> np.ndarray | None:
        """
        Per-atom partial charges.

        Returns
        -------
        ndarray of float64, shape (n_atoms,) or None
            Charges on each atom, or None if not set
        """
        ...

    @charges.setter
    def charges(self, charges: np.ndarray) -> None: ...
    @property
    def velocities(self) -> np.ndarray | None:
        """
        Per-atom velocities.

        Returns
        -------
        ndarray of float32, shape (n_atoms, 3) or None
            Velocities of each atom, or None if not set
        """
        ...

    @velocities.setter
    def velocities(self, velocities: np.ndarray) -> None: ...
    @property
    def cell(self) -> np.ndarray | None:
        """
        Unit cell for periodic systems.

        Returns
        -------
        ndarray of float64, shape (3, 3) or None
            Unit cell vectors, or None if not set
        """
        ...

    @cell.setter
    def cell(self, cell: np.ndarray) -> None: ...
    @property
    def stress(self) -> np.ndarray | None:
        """
        Virial stress tensor.

        Returns
        -------
        ndarray of float64, shape (3, 3) or None
            Stress tensor, or None if not set
        """
        ...

    @stress.setter
    def stress(self, stress: np.ndarray) -> None: ...
    @property
    def pbc(self) -> tuple[bool, bool, bool] | None:
        """
        Periodic boundary condition flags.

        Returns
        -------
        tuple of bool or None
            Periodicity along ``(x, y, z)``, or None if not set
        """
        ...

    @pbc.setter
    def pbc(self, pbc: tuple[bool, bool, bool] | None) -> None: ...
    @property
    def positions(self) -> np.ndarray:
        """
        Atomic positions (read-only).

        Returns
        -------
        ndarray of float32, shape (n_atoms, 3)
            Position of each atom in Angstroms
        """
        ...

    @property
    def atomic_numbers(self) -> np.ndarray:
        """
        Atomic numbers (read-only).

        Returns
        -------
        ndarray of uint8, shape (n_atoms,)
            Atomic number of each atom
        """
        ...

    def get_property(self, key: str) -> Any:
        """
        Get a custom property by key.

        Parameters
        ----------
        key : str
            Property key

        Returns
        -------
        Any
            Property value

        Raises
        ------
        KeyError
            If property key does not exist
        """
        ...

    def set_property(self, key: str, value: Any) -> None:
        """
        Set a custom property.

        Parameters
        ----------
        key : str
            Property key
        value : Any
            Property value
        """
        ...

    def property_keys(self) -> list[str]:
        """
        Get all property keys.

        Returns
        -------
        list of str
            All property keys
        """
        ...

    def has_property(self, key: str) -> bool:
        """
        Check if a property exists.

        Parameters
        ----------
        key : str
            Property key

        Returns
        -------
        bool
            True if property exists, False otherwise
        """
        ...

    @overload
    def __getitem__(self, index: int) -> PyAtom: ...
    @overload
    def __getitem__(self, index: str) -> Any: ...
    def __getitem__(self, index: int | str) -> PyAtom | Any: ...
    def __repr__(self) -> str: ...

class PyAtomDatabase:
    """
    Low-level PyO3-backed database for storing molecules with compression.

    Supports parallel writes and random access reads, making it useful for
    training and dataset preparation workflows.

    Parameters
    ----------
    path : str
        Path to database file
    compression : {"none", "lz4", "zstd"}, default="none"
        Compression type
    level : int, default=3
        Compression level for zstd (1-22)
    overwrite : bool, default=False
        If True, recreates the database file when it already exists.
    """

    def __init__(
        self,
        path: str,
        compression: str = "none",
        level: int = 3,
        overwrite: bool = False,
    ) -> None: ...
    @staticmethod
    def open(path: str, mmap: bool = True, populate: bool = False) -> PyAtomDatabase:
        """
        Open an existing database.

        By default this uses a memory-mapped index and is read-only. Pass
        ``mmap=False`` to reopen the database for appends.

        Parameters
        ----------
        path : str
            Path to existing database file
        mmap : bool, default=True
            If True, use a memory-mapped index and return a read-only handle.
            If False, load the index into memory and allow writes.
        populate : bool, default=False
            Only valid when ``mmap=True``. Prefaults mapped pages on Linux.
        """
        ...

    def add_molecule(self, molecule: PyMolecule) -> None:
        """
        Add a single molecule to the database.

        Parameters
        ----------
        molecule : PyMolecule
            Molecule to add
        """
        ...

    def add_molecules(self, molecules: Sequence[PyMolecule]) -> None:
        """
        Add multiple molecules in parallel.

        Parameters
        ----------
        molecules : sequence of PyMolecule
            Molecules to add
        """
        ...

    def add_arrays_batch(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        *,
        energy: np.ndarray | None = None,
        forces: np.ndarray | None = None,
        charges: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
        cell: np.ndarray | None = None,
        stress: np.ndarray | None = None,
        pbc: np.ndarray | None = None,
        name: Sequence[str] | None = None,
        properties: dict[str, Any] | None = None,
        atom_properties: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a stacked batch of molecules directly from numpy arrays.

        Custom properties can be supplied as batched columns via
        ``properties`` (per-molecule) and ``atom_properties`` (per-atom).
        """
        ...

    def get_molecule(self, index: int) -> PyMolecule:
        """
        Get a molecule by index.

        Parameters
        ----------
        index : int
            Molecule index (0-based)

        Returns
        -------
        PyMolecule
            The requested molecule
        """
        ...

    def get_molecules(self, indices: Sequence[int]) -> list[PyMolecule]:
        """
        Get multiple molecules by indices (batch read).

        Parameters
        ----------
        indices : sequence of int
            Molecule indices (0-based)

        Returns
        -------
        list of PyMolecule
            The requested molecules
        """
        ...

    def get_molecules_flat(self, indices: Sequence[int]) -> dict[str, Any]:
        """
        Get multiple molecules as contiguous batch arrays.

        Returns a mapping containing the stacked builtin arrays plus nested
        ``properties`` and ``atom_properties`` dictionaries when present.
        """
        ...

    def __len__(self) -> int:
        """
        Get the number of molecules.

        Returns
        -------
        int
            Number of molecules in the database
        """
        ...
    def __getitem__(self, index: int) -> PyMolecule: ...
    def flush(self) -> None:
        """
        Flush and save the database to disk.

        This writes the index and ensures all data is persisted.
        """
        ...
