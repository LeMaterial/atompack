# Copyright 2026 Entalpic
"""
Atompack: append-only molecule storage for atomistic ML datasets.

A Python API backed by a Rust storage engine for writing, reopening, and serving
molecular structures with forces, energies, charges, stress, and custom
properties. Built for dataset pipelines, random-access reads, batched array
loading, and ASE interoperability.

Examples
--------
Create a molecule and add properties:

>>> import atompack
>>> import numpy as np
>>>
>>> positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
>>> atomic_numbers = np.array([6, 8], dtype=np.uint8)
>>> mol = atompack.Molecule.from_arrays(positions, atomic_numbers)
>>> mol.energy = -123.456
>>> mol.forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

Save to database:

>>> db = atompack.Database("data.atp", overwrite=True)
>>> db.add_molecule(mol)
>>> db.flush()

Read back from database:

>>> db = atompack.Database.open("data.atp")
>>> mol = db[0]
>>> print(mol.energy)
-123.456

`Database.open(...)` is read-only by default and uses mmap. Reopen with
`Database.open(path, mmap=False)` if you want to append molecules.
"""

from typing import Any, Iterator

from . import hub
from ._atompack_rs import PyAtom as Atom
from ._atompack_rs import PyAtomDatabase as Database
from ._atompack_rs import PyMolecule as Molecule
from .ase_bridge import add_ase_batch, from_ase, to_ase, to_ase_batch


def _database_iter_batches(
    database: Database,
    batch_size: int,
    *,
    flat: bool = False,
    drop_last: bool = False,
) -> Iterator[list[Molecule] | dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    getter = database.get_molecules_flat if flat else database.get_molecules
    for start in range(0, len(database), batch_size):
        stop = min(start + batch_size, len(database))
        if drop_last and stop - start < batch_size:
            break
        yield getter(list(range(start, stop)))


Database.iter_batches = _database_iter_batches

__version__ = "0.2.1"
__all__ = [
    "Atom",
    "Molecule",
    "Database",
    "from_ase",
    "to_ase",
    "to_ase_batch",
    "add_ase_batch",
    "hub",
]
