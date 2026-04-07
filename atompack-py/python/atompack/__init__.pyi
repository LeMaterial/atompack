"""Type stubs for atompack"""

from typing import Any, Sequence, overload

import numpy as np
import numpy.typing as npt

from . import hub as hub

class Atom:
    """
    Represents a single atom with 3D coordinates and atomic number.

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

    Examples
    --------
    >>> carbon = Atom(0.0, 0.0, 0.0, 6)
    >>> oxygen = Atom(1.5, 0.0, 0.0, 8)
    >>> distance = carbon.distance_to(oxygen)
    """

    def __init__(self, x: float, y: float, z: float, atomic_number: int) -> None: ...
    def position(self) -> tuple[float, float, float]:
        """
        Get the position as a tuple.

        Returns
        -------
        tuple of float
            (x, y, z) coordinates in Angstroms
        """
        ...

    def distance_to(self, other: Atom) -> float:
        """
        Calculate distance to another atom.

        Parameters
        ----------
        other : Atom
            The other atom

        Returns
        -------
        float
            Distance in Angstroms
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

class Molecule:
    """
    A collection of atoms with optional properties (forces, energy, etc.).

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

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    >>> atomic_numbers = np.array([6, 8], dtype=np.uint8)
    >>> mol = Molecule(positions, atomic_numbers)
    >>> mol.energy = -100.5
    >>> mol.forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    >>> print(len(mol))  # 2
    """

    def __init__(
        self,
        positions: npt.NDArray[np.float32],
        atomic_numbers: npt.NDArray[np.uint8],
        *,
        energy: float | None = ...,
        forces: npt.NDArray[np.float32] | None = ...,
        charges: npt.NDArray[np.float64] | None = ...,
        velocities: npt.NDArray[np.float32] | None = ...,
        cell: npt.NDArray[np.float64] | None = ...,
        stress: npt.NDArray[np.float64] | npt.NDArray[np.float32] | None = ...,
        pbc: tuple[bool, bool, bool] | None = ...,
        name: str | None = ...,
    ) -> None: ...
    @staticmethod
    def from_arrays(
        positions: npt.NDArray[np.float32],
        atomic_numbers: npt.NDArray[np.uint8],
        *,
        energy: float | None = ...,
        forces: npt.NDArray[np.float32] | None = ...,
        charges: npt.NDArray[np.float64] | None = ...,
        velocities: npt.NDArray[np.float32] | None = ...,
        cell: npt.NDArray[np.float64] | None = ...,
        stress: npt.NDArray[np.float64] | npt.NDArray[np.float32] | None = ...,
        pbc: tuple[bool, bool, bool] | None = ...,
        name: str | None = ...,
    ) -> Molecule:
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

    def atoms(self) -> list[Atom]:
        """
        Get the list of atoms.

        Returns
        -------
        list of Atom
            All atoms in the molecule
        """
        ...

    def to_owned(self) -> Molecule:
        """
        Materialize the molecule into an owned, self-contained object.

        This is useful before pickling or sending a database-fetched lazy view
        across process boundaries.
        """
        ...

    def to_ase(
        self,
        *,
        attach_calc: bool = True,
        calc_mode: str = "singlepoint",
        copy_info: bool = True,
        copy_arrays: bool = True,
    ) -> object:
        """
        Convert the molecule to an ASE Atoms object.

        This reads directly from the molecule getters, so it works for both
        owned and view-backed molecules without going through `atoms()`.
        Geometry/species always become the ASE structure, supported builtin
        results are attached through `SinglePointCalculator`, per-atom custom
        arrays go to `atoms.arrays`, and remaining custom properties go to
        `atoms.info`.
        """
        ...

    @property
    def forces(self) -> npt.NDArray[np.float32] | None:
        """
        Per-atom forces.

        Returns
        -------
        ndarray of float32, shape (n_atoms, 3) or None
            Forces on each atom, or None if not set
        """
        ...

    @forces.setter
    def forces(self, value: npt.NDArray[np.float32]) -> None: ...
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
    def energy(self, value: float | None) -> None: ...
    @property
    def charges(self) -> npt.NDArray[np.float64] | None:
        """
        Per-atom partial charges.

        Returns
        -------
        ndarray of float64, shape (n_atoms,) or None
            Charges on each atom, or None if not set
        """
        ...

    @charges.setter
    def charges(self, value: npt.NDArray[np.float64]) -> None: ...
    @property
    def velocities(self) -> npt.NDArray[np.float32] | None:
        """
        Per-atom velocities.

        Returns
        -------
        ndarray of float32, shape (n_atoms, 3) or None
            Velocities of each atom, or None if not set
        """
        ...

    @velocities.setter
    def velocities(self, value: npt.NDArray[np.float32]) -> None: ...
    @property
    def cell(self) -> npt.NDArray[np.float64] | None:
        """
        Unit cell for periodic systems.

        Returns
        -------
        ndarray of float64, shape (3, 3) or None
            Unit cell vectors, or None if not set
        """
        ...

    @cell.setter
    def cell(self, value: npt.NDArray[np.float64]) -> None: ...
    @property
    def stress(self) -> npt.NDArray[np.float64] | None:
        """
        Virial stress tensor.

        Returns
        -------
        ndarray of float64, shape (3, 3) or None
            Stress tensor, or None if not set
        """
        ...

    @stress.setter
    def stress(self, value: npt.NDArray[np.float32] | npt.NDArray[np.float64]) -> None: ...
    @property
    def positions(self) -> npt.NDArray[np.float32]:
        """
        Atomic positions (read-only).

        Returns
        -------
        ndarray of float32, shape (n_atoms, 3)
            Position of each atom in Angstroms
        """
        ...

    @property
    def atomic_numbers(self) -> npt.NDArray[np.uint8]:
        """
        Atomic numbers (read-only).

        Returns
        -------
        ndarray of uint8, shape (n_atoms,)
            Atomic number of each atom
        """
        ...

    def get_property(self, key: str) -> float | int | str | npt.NDArray:
        """
        Get a custom property by key.

        Parameters
        ----------
        key : str
            Property key

        Returns
        -------
        float, int, str, or ndarray
            Property value

        Raises
        ------
        ValueError
            If property key does not exist
        """
        ...

    def set_property(self, key: str, value: float | int | str | npt.NDArray) -> None:
        """
        Set a custom property.

        Supported types: float, int, str, 1D float32/float64/int32/int64 arrays,
        and 2D float32/float64 arrays with shape (n, 3). Input dtype is preserved.
        The key 'stress' is reserved; use the dedicated ``stress`` property instead.

        Parameters
        ----------
        key : str
            Property key
        value : float, int, str, or ndarray
            Property value

        Raises
        ------
        ValueError
            If value type is not supported
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
    def __getitem__(self, index: int) -> Atom: ...
    @overload
    def __getitem__(self, index: str) -> float | int | str | npt.NDArray: ...
    def __getitem__(self, index: int | str) -> Atom | float | int | str | npt.NDArray:
        """
        Index atoms by integer or custom properties by key.

        Parameters
        ----------
        index : int or str
            Atom index (supports negative indices) or property key

        Returns
        -------
        Atom, float, int, str, or ndarray
            Atom for integer indices, property value for string keys

        Raises
        ------
        IndexError
            If atom index is out of bounds
        KeyError
            If property key does not exist
        TypeError
            If index type is neither int nor str
        """
        ...

class Database:
    """
    High-performance database for storing molecules with compression.

    Supports parallel writes and random access reads, making it ideal for
    ML training workflows with shuffled batches.

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
        If False, raises FileExistsError for existing paths.

    Examples
    --------
    Create and write molecules:

    >>> db = Database("molecules.atp")
    >>> db.add_molecule(mol1)
    >>> db.add_molecule(mol2)
    >>> db.flush()

    Open and read molecules:

    >>> db = Database.open("molecules.atp")
    >>> mol = db.get_molecule(0)
    >>> mol = db[0]  # Alternative indexing syntax
    >>> print(len(db))

    `Database.open(...)` is read-only by default. Reopen with
    `Database.open(path, mmap=False)` to append molecules.
    """

    def __init__(
        self,
        path: str,
        compression: str = "none",
        level: int = 3,
        overwrite: bool = False,
    ) -> None: ...
    @staticmethod
    def open(path: str, mmap: bool = True, populate: bool = False) -> Database:
        """
        Open an existing database.

        By default this uses a memory-mapped index and is read-only. Pass
        `mmap=False` to reopen the database for appends.

        Parameters
        ----------
        path : str
            Path to existing database file
        mmap : bool, default=True
            If True, use a memory-mapped index and return a read-only handle.
            If False, load the index into memory and allow writes.
        populate : bool, default=False
            Only valid when `mmap=True`. Prefaults mapped pages on Linux.

        Returns
        -------
        Database
            Opened database instance
        """
        ...
    def add_molecule(self, molecule: Molecule) -> None:
        """
        Add a single molecule to the database.

        Parameters
        ----------
        molecule : Molecule
            Molecule to add
        """
        ...

    def add_molecules(self, molecules: list[Molecule]) -> None:
        """
        Add multiple molecules in parallel.

        This method compresses molecules in parallel for better performance.

        Parameters
        ----------
        molecules : list of Molecule
            Molecules to add
        """
        ...
    def add_arrays_batch(
        self,
        positions: npt.NDArray[np.float32],
        atomic_numbers: npt.NDArray[np.uint8],
        *,
        energy: npt.NDArray[np.float64] | None = ...,
        forces: npt.NDArray[np.float32] | None = ...,
        charges: npt.NDArray[np.float64] | None = ...,
        velocities: npt.NDArray[np.float32] | None = ...,
        cell: npt.NDArray[np.float64] | None = ...,
        stress: npt.NDArray[np.float64] | None = ...,
        pbc: npt.NDArray[np.bool_] | None = ...,
        name: Sequence[str] | None = ...,
        properties: dict[str, Any] | None = ...,
        atom_properties: dict[str, Any] | None = ...,
    ) -> None:
        """
        Add a stacked batch of molecules directly from numpy arrays.

        Custom properties can be supplied as batched columns via
        ``properties`` (per-molecule) and ``atom_properties`` (per-atom).
        """
        ...

    def get_molecule(self, index: int) -> Molecule:
        """
        Get a molecule by index.

        Parameters
        ----------
        index : int
            Molecule index (0-based)

        Returns
        -------
        Molecule
            The requested molecule
        """
        ...
    def get_molecules(self, indices: list[int]) -> list[Molecule]:
        """
        Get multiple molecules by indices (batch read).

        Parameters
        ----------
        indices : list of int
            Molecule indices (0-based)

        Returns
        -------
        list of Molecule
            The requested molecules
        """
        ...
    def get_molecules_flat(self, indices: list[int]) -> dict[str, Any]:
        """
        Get multiple molecules as contiguous batch arrays.

        Returns a mapping containing the stacked builtin arrays plus nested
        ``properties`` and ``atom_properties`` dictionaries when present.
        """
        ...
    def to_ase_batch(
        self,
        indices: list[int] | None = None,
        *,
        attach_calc: bool = True,
        calc_mode: str = "singlepoint",
        copy_info: bool = True,
        copy_arrays: bool = True,
    ) -> list[object]:
        """
        Convert many molecules from the database to ASE Atoms in one call.

        This uses the database flat-read path internally and is substantially
        faster than repeated ``db[i].to_ase()`` for homogeneous batches.
        """
        ...

    def flush(self) -> None:
        """
        Flush and save the database to disk.

        This writes the index and ensures all data is persisted.
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

    def __getitem__(self, index: int) -> Molecule:
        """
        Get a molecule using indexing syntax: db[i].

        Parameters
        ----------
        index : int
            Molecule index (0-based)

        Returns
        -------
        Molecule
            The requested molecule
        """
        ...

def from_ase(
    atoms,
    energy: float | None = None,
    forces: npt.NDArray[np.float32] | None = None,
    charges: npt.NDArray[np.float64] | None = None,
    velocities: npt.NDArray[np.float32] | None = None,
    cell: npt.NDArray[np.float64] | None = None,
    stress: npt.NDArray[np.float64] | None = None,
    copy_info: bool = True,
    info: dict | None = None,
) -> Molecule:
    """
    Convert an ASE Atoms object to an atompack Molecule.

    This function extracts positions, atomic numbers, and available properties
    (forces, energy, charges, velocities, cell) from an ASE Atoms object and
    creates a corresponding atompack Molecule. Custom properties from atoms.info
    dict are also copied by default.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object to convert
    energy : float, optional
        Override energy from ASE Atoms. If None, attempts to extract from atoms.
    forces : ndarray of float32, shape (n_atoms, 3), optional
        Override forces from ASE Atoms. If None, attempts to extract from atoms.
    charges : ndarray of float64, shape (n_atoms,), optional
        Override charges from ASE Atoms. If None, attempts to extract from atoms.
    velocities : ndarray of float32, shape (n_atoms, 3), optional
        Override velocities from ASE Atoms. If None, attempts to extract from atoms.
    cell : ndarray of float64, shape (3, 3), optional
        Override cell from ASE Atoms. If None, attempts to extract from atoms.
    copy_info : bool, default=True
        If True, copies custom properties from atoms.info dict to molecule properties.
        Supports: str, int, float, 1D float/int arrays, 2D arrays with shape (n, 3).
    info : dict, optional
        Additional properties to store in the molecule. These will be added after
        copying atoms.info (if copy_info=True), so they can override atoms.info values.
        Supports the same types as copy_info.

    Returns
    -------
    Molecule
        Atompack molecule with all available properties

    Examples
    --------
    Basic usage:

    >>> from ase import Atoms
    >>> from atompack import from_ase
    >>>
    >>> # Create ASE Atoms object
    >>> ase_atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> ase_atoms.calc = ...  # Attach calculator
    >>> ase_atoms.get_forces()  # Calculate forces
    >>>
    >>> # Convert to atompack (automatically copies atoms.info properties)
    >>> mol = from_ase(ase_atoms)

    Override properties and custom info:

    >>> # Override energy with custom value
    >>> mol = from_ase(ase_atoms, energy=-100.0)
    >>>
    >>> # Add custom info and convert
    >>> ase_atoms.info['temperature'] = 300.0
    >>> ase_atoms.info['method'] = 'DFT'
    >>> mol = from_ase(ase_atoms)
    >>> print(mol.get_property('temperature'))  # 300.0
    >>> print(mol.get_property('method'))  # 'DFT'
    >>>
    >>> # Skip copying info dict
    >>> mol = from_ase(ase_atoms, copy_info=False)
    >>>
    >>> # Pass custom info dict
    >>> custom_info = {'temperature': 300.0, 'method': 'VASP', 'converged': True}
    >>> mol = from_ase(ase_atoms, info=custom_info)
    """
    ...

def to_ase(
    molecule: Molecule,
    *,
    attach_calc: bool = True,
    calc_mode: str = "singlepoint",
    copy_info: bool = True,
    copy_arrays: bool = True,
) -> object:
    """Convert an atompack Molecule to an ASE Atoms object."""
    ...

def to_ase_batch(
    source: Database | list[Molecule],
    indices: list[int] | None = None,
    *,
    attach_calc: bool = True,
    calc_mode: str = "singlepoint",
    copy_info: bool = True,
    copy_arrays: bool = True,
) -> list[object]:
    """Convert many atompack molecules to ASE Atoms efficiently."""
    ...

def add_ase_batch(
    db: Database,
    atoms_list: list[object],
    *,
    copy_info: bool = True,
    info: dict | list[dict | None] | None = None,
    batch_size: int = 512,
) -> None:
    """Write many ASE Atoms objects into an atompack database efficiently."""
    ...

__version__: str
__all__: list[str]
