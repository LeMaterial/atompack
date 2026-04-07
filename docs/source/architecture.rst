.. Copyright 2026 Entalpic

Architecture
============

This page describes the current Atompack design as it exists in this repository. The short version
is that Atompack is an append-only molecule store with a Python API and a Rust storage engine. It
is built around a simple unit of storage, the molecule, and around predictable read/write modes for
dataset pipelines.

System View
-----------

Atompack sits between dataset producers and dataset consumers. The Python layer is the ergonomic
surface, while the Rust layer owns the file format, indexing, and read/write paths.

.. code-block:: text

   ASE / numpy / Python producers
               |
               v
      +---------------------+
      |  atompack Python    |
      |  - Molecule         |
      |  - Database         |
      |  - add_ase_batch    |
      |  - hub helpers      |
      +---------------------+
               |
               v
      +---------------------+
      |  Rust core          |
      |  - AtomDatabase     |
      |  - SOA record build |
      |  - trailing index   |
      |  - mmap read mode   |
      +---------------------+
               |
               v
      +---------------------+
      |  .atp file / shards |
      +---------------------+
               |
               v
   training loops / evaluation / Hub distribution

Repository Layout
-----------------

- ``atompack/``: Rust core crate with the storage engine, file format, and core data model
- ``atompack-py/``: PyO3 bindings plus the Python package
- ``docs/``: Sphinx documentation
- ``scripts/``: helper scripts such as stub generation

Core Data Model
---------------

The main domain type is ``Molecule``. A molecule stores:

- ``positions`` and ``atomic_numbers``
- builtin optional fields such as ``energy``, ``forces``, ``charges``, ``velocities``, ``cell``,
  ``stress``, and ``pbc``
- custom per-atom and per-molecule properties

``Atom`` exists as a lightweight convenience type, but the stored representation is already
structure-of-arrays oriented. In practice, Atompack is optimized for moving full molecule records
between disk, Rust, numpy, and ASE rather than for manipulating atom-by-atom objects in storage.

Component Overview
------------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Python API

      User-facing entry points such as ``Database(...)``, ``Database.open(...)``,
      ``Molecule.from_arrays(...)``, ``add_ase_batch(...)``, and ``atompack.hub``.

   .. grid-item-card:: Rust Storage Engine

      Owns the file format, crash-safe header handling, indexing, append paths, and mmap-backed
      read mode.

   .. grid-item-card:: SOA Records

      Molecules are stored as geometry plus builtin/custom property payloads in an
      array-oriented representation that matches numpy-heavy workloads.

   .. grid-item-card:: Distribution Layer

      Local files, shard directories, and Hugging Face dataset snapshots are all exposed through
      the same high-level reading model.

Python API
----------

The Python package is intentionally small and centered around a few workflows:

- ``atompack.Database(path, ...)`` creates a new file
- ``atompack.Database.open(path, mmap=True, populate=False)`` opens an existing file
- ``Molecule.from_arrays(...)`` builds a molecule directly from numpy arrays
- ``Database.add_arrays_batch(...)`` writes stacked numpy batches without creating one Python
  molecule per record
- ``Database.get_molecules_flat(indices)`` returns training-friendly stacked arrays already batched
- ``atompack.from_ase(...)`` and ``Molecule.to_ase()`` integrations to ASE
- ``atompack.hub`` uploads, downloads, and opens local or remote shard layouts through one reader
  interface allowing easy sharing through the Hugging Face Hub

Two open modes matter:

- writable mode: create a file or reopen with ``mmap=False`` when appending
- read-only mmap mode: the default for ``Database.open(...)`` and the preferred mode for serving
  static datasets


Storage Layout
--------------

The on-disk format lives in ``atompack/src/storage/`` and currently uses:

1. two 4 KiB header slots
2. a data region containing molecule records
3. a trailing index written on ``flush()``

Each header slot stores the format version, generation number, index location, molecule count,
record format, codec metadata, and a checksum. On open, Atompack reads both slots and chooses the
newest valid one.

This design gives Atompack its main operational properties:

- appends stay simple because new records are written sequentially
- ``flush()`` publishes a new index snapshot atomically enough for crash recovery
- molecule lookup is O(1) through the trailing index

.. code-block:: text

   +---------------------------------------------------------------+
   | Header slot A (4 KiB)                                         |
   | - magic + version                                             |
   | - generation                                                  |
   | - index offset / length                                       |
   | - molecule count                                              |
   | - record / codec metadata                                     |
   | - checksum                                                    |
   +---------------------------------------------------------------+
   | Header slot B (4 KiB)                                         |
   | - same fields, alternate commit target                        |
   +---------------------------------------------------------------+
   | Data region                                                   |
   | - record 0: positions, atomic_numbers, builtin/custom fields  |
   | - record 1: ...                                               |
   | - ...                                                         |
   | - record N-1                                                  |
   +---------------------------------------------------------------+
   | Trailing index                                                |
   | - count                                                       |
   | - per-record offset                                           |
   | - compressed size                                             |
   | - uncompressed size                                           |
   | - atom count                                                  |
   +---------------------------------------------------------------+

At commit time, Atompack writes the index first and then updates the newer valid header slot. On
open, it reads both header slots and chooses the highest valid generation.

Record Shape
------------

At a conceptual level, one stored molecule looks like this:

.. code-block:: text

   Molecule record
   |
   +-- positions:        (n_atoms, 3) float32
   +-- atomic_numbers:   (n_atoms,)   uint8
   +-- builtin fields:
   |   +-- energy
   |   +-- forces
   |   +-- charges
   |   +-- velocities
   |   +-- cell
   |   +-- stress
   |   +-- pbc
   |   +-- name
   |
   +-- custom atom properties
   |
   +-- custom molecule properties

Read Path
---------

When a file is opened read-only with mmap:

- Atompack validates the file header
- loads the index in memory-mapped mode
- optionally prefaults mapped pages on Linux when ``populate=True``
- fetches molecules by index without reopening or rescanning the file

For Python users, this means ``db[i]``, ``db.get_molecules(...)``, and
``db.get_molecules_flat(...)`` are all built on direct indexed access to the underlying file.

Write Path
----------

When a file is opened writable:

- new molecules are appended to the end of the file
- batch ingestion paths can serialize records from numpy arrays directly
- ``flush()`` rewrites the trailing index and advances the committed header generation

If the file contains an uncommitted tail after a crash or interrupted write, writable open will
truncate back to the last committed state before continuing.

Current Tradeoffs
-----------------

- The storage unit is the whole molecule, not a partial field projection.
- Writable mode and mmap-backed read mode are distinct operational modes.
- Updates and deletes require rewriting the dataset.
- The file format is explicit and simple, but it is specialized for atomistic ML datasets rather
  than for general-purpose tabular workloads.

Reference Points
----------------

For public APIs, the generated docs are usually the best entry point:

- :doc:`Python package API <autoapi/atompack/index>`: ``Database``, ``Molecule``, top-level helpers
- :doc:`ASE helpers <autoapi/atompack/ase_bridge/index>`: ``from_ase(...)``, ``to_ase(...)``, ``add_ase_batch(...)``
- :doc:`Hub helpers <autoapi/atompack/hub/index>`: local and Hugging Face dataset access
- :doc:`Rust API <rust-api>`: rustdoc for the core crate and bindings crate
