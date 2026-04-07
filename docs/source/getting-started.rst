.. Copyright 2026 Entalpic

Getting Started
===============

Atompack is a Python package and Rust-backed file format for atomistic datasets. It stores whole
molecules together with builtin fields such as forces, energies, charges, velocities, stress,
PBC, and arbitrary custom properties for additional properties. The main workflow is:

- write molecules or stacked numpy batches to an append-only ``.atp`` file
- or use the Hugging Face Hub integration to download and open remote shard layouts
- reopen the file in read-only mmap mode for random access
- convert structures to or from ASE when needed
- publish a single file or shard directory through the Hugging Face Hub when distribution matters

Installation
------------

Install the Python package from the repository:

.. code-block:: bash

   uv pip install "git+https://github.com/Entalpic/atompack.git@main#subdirectory=atompack-py"

Quickstart
----------

Create a molecule, attach properties, write it to disk, and read it back:

.. code-block:: python

   import atompack
   import numpy as np

   positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
   atomic_numbers = np.array([6, 8], dtype=np.uint8)
   mol = atompack.Molecule.from_arrays(positions, atomic_numbers)

   mol.energy = -123.456
   mol.forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

   db = atompack.Database("data.atp", overwrite=True)
   db.add_molecule(mol)
   db.flush()

   db = atompack.Database.open("data.atp")
   mol = db[0]
   print(mol.energy)

Writable vs read-only opens:

- ``atompack.Database(path, ...)`` creates a new file
- ``atompack.Database.open(path)`` opens read-only with mmap by default
- ``atompack.Database.open(path, mmap=False)`` reopens the file for appends

Batch Writing, Simple Reading
-----------------------------

For ingestion, the batched write path is the most efficient. For reads, a simple indexed loop is
often already fast enough:

.. code-block:: python

   import atompack
   import numpy as np

   positions = np.random.rand(32, 64, 3).astype(np.float32)
   atomic_numbers = np.full((32, 64), 6, dtype=np.uint8)

   db = atompack.Database("batch.atp", overwrite=True)
   db.add_arrays_batch(positions, atomic_numbers)
   db.flush()

   db = atompack.Database.open("batch.atp")
   for i in range(4):
       mol = db[i]
       print(i, len(mol), mol.positions.shape)

Read-Only Mode (Memory-Mapped Index)
------------------------------------

Read-only mmap mode is the default for ``Database.open(...)`` and is the right choice for
read-mostly datasets:

.. code-block:: python

   db = atompack.Database.open("data.atp")  # mmap=True by default
   mol = db[0]

On Linux you can also prefault mapped pages:

.. code-block:: python

   db = atompack.Database.open("data.atp", mmap=True, populate=True)

ASE Integration
---------------

If you use ASE, you can convert individual structures with ``from_ase(...)`` and write many
structures efficiently with ``add_ase_batch(...)``:

.. code-block:: python

   import atompack
   from ase import Atoms

   ase_atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
   mol = atompack.from_ase(ase_atoms)
   restored = mol.to_ase()

   structures = [
       Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
       Atoms("CO2", positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]),
   ]

   db = atompack.Database("ase_data.atp", overwrite=True)
   atompack.add_ase_batch(db, structures, batch_size=256)
   db.flush()

``add_ase_batch(...)`` is the preferred path when you already have an iterator or list of
``ase.Atoms`` objects and want to ingest them directly into a database.

When Atompack Is A Good Fit
---------------------------

- dataset creation pipelines that append many molecules and publish immutable snapshots
- training workloads that repeatedly sample whole molecules at random
- pipelines that want both a Python API and a low-level Rust storage engine
- projects that need ASE conversion or Hub distribution without introducing a full database service

Tradeoffs
---------

- Append-only: updates and deletes require rewriting the file, but thanks to the efficient storage format, this is very fast to do.
- Read/write mode and read-only mmap mode are separate on purpose.
- The storage unit is a whole molecule; Atompack is not a query engine or column store.
- Compression is optional, but it is not the main abstraction. The main abstraction is a durable
  molecule record with direct indexing.

Next Steps
----------

- Python API reference: :doc:`autoapi/index`
- Hub upload/download helpers: :doc:`huggingface`
- Storage format and internals: :doc:`architecture`
- Benchmarks and reproducibility: :doc:`performance`
- Rust crate API docs: :doc:`rust-api`
