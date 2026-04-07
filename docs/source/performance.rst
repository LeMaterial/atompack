.. Copyright 2026 Entalpic

Performance
===========

This page summarizes the current benchmark results for Atompack. The main pattern is consistent
across the reported comparisons:

- Atompack is strongest on read-heavy serving paths, especially once access becomes less
  sequential.
- Atompack also leads the write-throughput slices used by the report, particularly when using the
  native batch ingestion APIs.
- Artifact size stays close to HDF5 SOA and substantially smaller than the LMDB and ASE baselines
  used in this repository.

Baselines
----------------------------

The benchmark set is intentionally mixed. It is not only comparing Atompack to one alternative
storage engine, but to the main layout families that show up in atomistic ML codebases:

- ``hdf5_soa`` is the conventional scientific-array baseline: one chunked HDF5 dataset per field
  such as positions, atomic numbers, energies, and forces. Its main trade-off is that it is very
  compact and works well for bulk array access, but shuffled single-molecule reads can force extra
  chunk traffic and more Python-side reconstruction work.
- ``lmdb_packed`` is a key-value baseline where each molecule is serialized into one compact binary
  record and stored in LMDB. That is closer to the access pattern used by training dataloaders and
  usually behaves better than HDF5 on random reads, but it pays per-record encode/decode overhead
  and is less storage-efficient than a compact array-oriented layout.
- ``lmdb_pickle`` is a common Python-first LMDB pattern where each entry is a pickled dict of
  numpy arrays and scalar metadata. It is flexible and easy to integrate into existing repos, but
  pickle framing and object reconstruction are exactly the costs that show up in the read and size
  numbers.
- ``ase_sqlite`` and ``ase_lmdb`` are included as ecosystem reference baselines. They are not
  designed first as high-throughput training stores, but ASE is still the main interchange layer in
  many atomistic and materials ML repositories: datasets are often published as ``ase.Atoms`` or
  ``ase.db`` collections, and training code frequently starts from an ASE-based reader or
  preprocessing step. That makes ASE an important baseline for practical compatibility, even when
  its database backends trade away throughput for generality and broad tool support.

The ASE results answer a different question from the HDF5 and LMDB baselines. They
show the cost of staying on the most common ecosystem path end to end, while the HDF5 and LMDB
results show the more specialized storage trade-offs that practitioners already use when they start
optimizing training throughput.

Read Performance
----------------

For a representative NVMe slice at 64 atoms per molecule, the comparison below uses:

- sequential read throughput
- the single-worker ``multiprocessing`` slice as the random or shuffled read proxy

.. figure:: ../../atompack_story_report/story_read_hero.svg
   :alt: Atompack read throughput benchmark hero figure

For that slice, Atompack reaches about:

- ``646k mol/s`` on sequential reads
- ``446k mol/s`` on the single-worker random or shuffled read path

Relative to the same benchmark slice, Atompack is:

- ``1.37x`` faster than HDF5 SOA on sequential reads and ``24.0x`` faster on the random or
  shuffled path
- ``3.32x`` faster than LMDB Packed on sequential reads and ``2.81x`` faster on the random or
  shuffled path
- ``5.18x`` faster than LMDB Pickle on sequential reads and ``3.82x`` faster on the random or
  shuffled path

.. list-table::
   :header-rows: 1

   * - backend
     - sequential read (mol/s)
     - random/shuffled read (mol/s)
   * - atompack
     - 646,261
     - 445,830
   * - hdf5_soa
     - 470,417
     - 18,569
   * - lmdb_packed
     - 194,467
     - 158,871
   * - lmdb_pickle
     - 124,706
     - 116,579
   * - ase_lmdb
     - 4,637
     - 4,620
   * - ase_sqlite
     - 1,790
     - 1,803

The “random/shuffled” column above comes from the single-worker ``multiprocessing`` benchmark
slice used for this benchmark report, not from a separate point-query microbenchmark.

Scaling And Filesystems
-----------------------

The benchmark results also cover atom-count scaling and behavior on shared filesystems:

.. figure:: ../../atompack_story_report/story_size_scaling.svg
   :alt: Atompack scaling figure across atom counts

.. figure:: ../../atompack_story_report/story_random_filesystems.svg
   :alt: Atompack random read behavior across filesystems


Across NVMe, NFS, GPFS, and Lustre, Atompack keeps a clear lead on the single-worker
random/shuffled read slice:

- vs LMDB Packed: about ``2.6x`` to ``2.9x`` faster
- vs LMDB Pickle: about ``3.8x`` to ``4.8x`` faster
- vs HDF5 SOA: about ``15.7x`` to ``31.6x`` faster

That consistency matters for shared-storage training setups where local-NVMe numbers are not
representative of the final deployment environment.

Write Throughput
----------------

For write throughput, the comparison below uses the current NVMe write benchmark slices:

.. figure:: ../../atompack_story_report/story_write_overview.svg
   :alt: Atompack write throughput overview

For the 64-atom slice, Atompack leads the plotted backends both with builtin fields only and with
additional custom properties:

.. list-table::
   :header-rows: 1

   * - backend
     - write builtins (mol/s)
     - write with custom props (mol/s)
   * - atompack
     - 105,473
     - 77,193
   * - hdf5_soa
     - 91,431
     - 57,198
   * - lmdb_packed
     - 37,573
     - 26,323
   * - lmdb_pickle
     - 24,967
     - 18,477
   * - ase_sqlite
     - 1,417
     - 1,398
   * - ase_lmdb
     - 919
     - 755

This is the path where Atompack's native batch ingestion matters most:

- ``Database.add_arrays_batch(...)`` for stacked numpy inputs
- ``atompack.add_ase_batch(...)`` for iterables of ``ase.Atoms``

Storage Efficiency
------------------

For storage footprint, the comparison below reports normalized artifact size for the same write
benchmark slices:

.. figure:: ../../atompack_story_report/story_write_storage.svg
   :alt: Atompack write storage efficiency comparison

For the 64-atom slice:

.. list-table::
   :header-rows: 1

   * - backend
     - size ratio vs atompack (builtins)
     - size ratio vs atompack (custom)
   * - hdf5_soa
     - 0.96x
     - 0.95x
   * - atompack
     - 1.00x
     - 1.00x
   * - lmdb_packed
     - 2.34x
     - 1.35x
   * - lmdb_pickle
     - 2.35x
     - 1.35x
   * - ase_sqlite
     - 3.05x
     - 2.08x
   * - ase_lmdb
     - 4.69x
     - 2.69x

The practical takeaway is not that Atompack is always the absolute smallest representation. The
more important result is that it stays in the compact-storage regime while pairing that with much
stronger read behavior.

Reproducing The Benchmarks
--------------------------

.. code-block:: bash

   uv run --project atompack-py --no-sync python atompack-py/benchmarks/benchmark.py \
     --out /tmp/atompack-bench/benchmark.json

   uv run --project atompack-py --no-sync python atompack-py/benchmarks/scaling_benchmark.py \
     --out /tmp/atompack-bench/scaling.json

   uv run --project atompack-py --no-sync python atompack-py/benchmarks/write_benchmark.py \
     --out /tmp/atompack-bench/write.json

For running quick microbenchmarks or inspecting the code, you can also run this binary directly:

.. code-block:: bash

   cargo run -p atompack --release --bin atompack-bench -- --help

Practical Guidance
------------------

- Use ``Database.open(path)`` for read-mostly datasets. It defaults to mmap-backed read-only mode.
- Reopen with ``Database.open(path, mmap=False)`` when you need to append more molecules.
- Prefer ``add_arrays_batch(...)`` or ``add_ase_batch(...)`` when ingesting large datasets from
  existing array or ASE pipelines.
- Prefer ``db[i]`` or ``db.get_molecules(...)`` for straightforward read paths, and use
  ``get_molecules_flat(...)`` when you specifically want already-stacked training batches.
- Compression is available when artifact size matters, but it should be treated as a workload
  tuning knob rather than as the main reason to adopt Atompack.
