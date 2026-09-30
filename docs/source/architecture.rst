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

Custom Properties
-----------------

Custom properties are dataset-specific values keyed by name. They are separate from builtin fields:
``energy``, ``forces``, ``charges``, ``velocities``, ``cell``, ``stress``, ``pbc``, ``name``,
``positions``, and ``atomic_numbers`` keep their dedicated storage and API paths.

Each custom property key has one owner scope:

- molecule properties store one value for the whole molecule
- atom properties store one value per atom

The same custom key cannot exist in both scopes on one molecule, so property reads do not need a
scope argument. New custom keys default to molecule scope; new atom properties must be written
explicitly as atom properties. Overwriting an existing atom property keeps atom scope and validates
that the new value still has one leading entry per atom.

Custom values can be scalars, strings, ``None``, numeric arrays, or tensor-shaped numeric arrays.
For tensor-shaped values, Atompack preserves the dtype and the shape of each stored value. Tensor
shape is value-level metadata, not a global schema constraint: the same key may have shape
``(128,)`` on one molecule and ``(4, 32)`` on another. Atom-scoped tensor values must still have
``n_atoms`` as their first dimension; trailing dimensions are arbitrary.

This flexibility applies to per-molecule storage and retrieval. APIs that build one concatenated
array are necessarily stricter:

- ``Database.add_arrays_batch(...)`` accepts tensor custom properties as stacked ndarrays, not
  lists or tuples of differently shaped arrays. Existing ``list[str]`` molecule properties remain
  valid for batched string columns.
- ``Database.get_molecules_flat(...)`` can concatenate tensor properties only when the selected
  records have compatible shapes for that key. If shapes differ, the dataset is still valid, but
  callers should retrieve molecule records with ``db[i]`` or ``db.get_molecules(...)`` instead of
  asking for a flat representation.

ASE ingestion follows the same ownership rule. ``from_ase(...)`` copies supported custom ndarray
values as molecule properties and does not infer atom-property scope from ``atoms.arrays``,
``atoms.info``, calculator results, or ndarray shape.

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
2. a data region of framed entries: molecule records, schema changes, and group chunks
3. a metadata directory and the index, written after the last frame by ``flush()``

Each header slot stores the format version, generation number, index location, molecule count,
record format, codec metadata, and a checksum. On open, Atompack reads both slots and chooses the
newest valid one.

Every entry in the data region starts with a 20-byte frame header: kind, payload length,
uncompressed length, atom count, and a CRC32 of the header and payload. Index entries point at the
payload, so reads never see frame headers. The directory and index after the last frame are a
cache that can always be rebuilt from the frames.

This design gives Atompack its main operational properties:

- molecule lookup is O(1) through the index
- files stay compact however often ``flush()`` is called
- if a writing process stops before flushing, the records it wrote are recovered on the next open

.. code-block:: text

   +---------------------------------------------------------------+
   | Header slot A (4 KiB)                                         |
   | - magic + version, generation, flags                          |
   | - index / directory / schema locations, molecule count        |
   | - record / codec metadata, end of the data region             |
   | - checksum                                                    |
   +---------------------------------------------------------------+
   | Header slot B (4 KiB)                                         |
   | - same fields, alternate commit target                        |
   +---------------------------------------------------------------+
   | Data region (framed entries)                                  |
   | - schema: record format + schema lock                         |
   | - record 0: positions, atomic_numbers, builtin/custom fields  |
   | - record 1 ... record N-1                                     |
   | - groups chunk (one per add_groups call)                      |
   +---------------------------------------------------------------+
   | Metadata directory: location of every non-record frame        |
   +---------------------------------------------------------------+
   | Index                                                         |
   | - count                                                       |
   | - per-record offset, compressed size, uncompressed size,      |
   |   atom count                                                  |
   +---------------------------------------------------------------+

The first write after a flush marks both header slots as "writing" and then writes new frames over
the old directory and index. ``flush()`` writes a new directory and index after the last frame,
truncates anything beyond them, and commits the header. If a writing process stops before its
next flush, opening the file scans the frames up to the first incomplete or corrupted one and
rebuilds the index, schema, and directory from them.

Files at rest keep the version 2 layout, so Atompack 0.4 reads them. Older versions refuse a file
in the writing state instead of reading stale metadata. Files last written by an older version
have no framed flag; Atompack keeps appending to them after their last commit, as before.

Groups are stored as chunks: each ``add_groups`` call writes one frame with a CSR encoding (per-group
offsets, member record indices, optional role ids) plus one column per group property, and a
grouping is the concatenation of its chunks. Records are referenced, not copied, so a record can
belong to many groups. Groups are decoded on first access.

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

- new molecules are written as frames after the last committed frame
- batch ingestion paths can serialize records from numpy arrays directly
- ``flush()`` writes the index after the last frame and advances the committed header generation

If a writing process stops before flushing, the next open recovers every complete frame, and a
writable open continues after the last one.

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
