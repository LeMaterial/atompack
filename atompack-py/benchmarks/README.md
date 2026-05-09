# Benchmarks

This directory contains the current Atompack benchmarking scripts and the
helper modules they share.

This directory is for the Python benchmark suite. The Rust-side benchmark harness
is separate and now lives at `atompack/src/bin/atompack-bench.rs`, while the Rust
example surface is intentionally small (`atompack/examples/basic_usage.rs`).

The main backends compared here are:

- `atompack`
- `hdf5_soa`
- `lmdb_soa`
- `lmdb_packed`
- `lmdb_pickle`
- in some scripts, ASE-backed datasets as well

Important:

- Build the Python extension in release mode before trusting throughput numbers.
- From the repo root, use `make py-dev-release`.
- Debug builds are much slower and will distort both read and write results.
- If you only want a Rust storage benchmark and do not need Python/backend
  comparisons, run:

```bash
cargo run -p atompack --release --bin atompack-bench -- --help
```

## Main Scripts

### `benchmark.py`

This is the broad benchmark suite.

It covers:

- sequential read throughput
- compression impact
- multiprocessing scaling

Use this when you want one script that gives a wide performance picture across
the blog-post workloads and backends.

Typical usage:

```bash
uv run --no-sync --project atompack-py python atompack-py/benchmarks/benchmark.py
uv run --no-sync --project atompack-py python atompack-py/benchmarks/benchmark.py --only 1 3
uv run --no-sync --project atompack-py python atompack-py/benchmarks/benchmark.py --scratch-dir /tmp/atompack-bench
```

Notes:

- `db[i]` now exercises Atompack's lazy-view single-molecule read path, with
  mutation materializing on demand via copy-on-write.
- The maintained tensor-pipeline comparison is `get_molecules_flat(...)`
  versus the `db[i]` / `get_molecule(i)` loop, which cleanly separates the
  batch-ingestion path from the per-sample dataset path.
- Pickling a lazy molecule materializes it into an owned, self-contained
  molecule first. `Molecule.to_owned()` is available when you want to force
  that boundary explicitly.
- `get_molecules_flat(...)` remains the fastest tensor-batch path when the
  consumer can work with concatenated arrays directly.
- `hdf5_soa` reflects the usual chunked-dataset HDF5 layout used for
  fixed-shape materials data; install with `pip install atompack-db[benchmarks]`
  to pull in the `h5py` dependency.
- `lmdb_packed` now participates in the synthetic custom-property variants in
  this suite too; `lmdb_soa` remains the builtins-only helper baseline.
- In this suite, `hdf5_soa` is intentionally read one molecule at a time so it
  matches the same `Dataset.__getitem__`-style access pattern as the other
  per-sample backends.
- In benchmark 3, `lmdb_soa` is skipped by default for every atom count because
  the shuffled single-item multiprocessing path defeats chunk locality and
  makes it disproportionately slow.
- In benchmark 3, `hdf5_soa` and ASE backends are also skipped by default at
  `256` atoms because that workload becomes pathologically slow and dominates
  suite runtime.
- The previous broader experimental suite, including tensor-pipeline, OMAT,
  memory, and dataset-scale benchmarks, is archived in
  `benchmarks/archive/full_benchmark_suite.py`.

### `write_benchmark.py`

This is the focused write benchmark.

It exists to answer write-specific questions more cleanly than the full suite.

It covers:

- write throughput across atom counts
- write scaling as dataset size grows
- Atompack batch-size scaling at fixed atom counts
- `hdf5_soa` and `lmdb_soa` write baselines
- ASE-backed write baselines

Use this when you want to measure ingestion speed without the extra machinery of
the broader suite.

Typical usage:

```bash
uv run --no-sync --project atompack-py python atompack-py/benchmarks/write_benchmark.py
uv run --no-sync --project atompack-py python atompack-py/benchmarks/write_benchmark.py --bench 1
uv run --no-sync --project atompack-py python atompack-py/benchmarks/write_benchmark.py --codec zstd:3
uv run --no-sync --project atompack-py python atompack-py/benchmarks/write_benchmark.py --bench 2 --sizes 50000 500000 5000000
uv run --no-sync --project atompack-py python atompack-py/benchmarks/write_benchmark.py --bench 3 --batch-scale-atoms 64 256 --batch-scale-sizes 256 512 1024 2048 4096 10000
uv run --no-sync --project atompack-py python atompack-py/benchmarks/write_benchmark.py --scratch-dir /tmp/atompack-bench
uv run --no-sync --project atompack-py python atompack-py/benchmarks/write_benchmark.py --out atompack-py/benchmarks/write_results.json
```

Notes:

- Benchmark datasets default to a temp-backed `atompack-benchmarks` directory;
  override with `--scratch-dir ...` or `ATOMPACK_BENCHMARK_SCRATCH` when you
  want a different filesystem.
- This script defaults to `--codec none` so raw write throughput is measured unless you explicitly opt into compression.
- Pass `--codec lz4` or `--codec zstd:3` when you want compressed-write numbers.
- Atompack now auto-sizes its write batch by atom count unless you pass
  `--atompack-batch-size ...`.
- The auto batch now also applies calibrated per-workload caps so large-molecule
  writes avoid the over-batching that showed up in the 128- and 256-atom NVMe
  runs.
- `--bench 3` exposes the Atompack builtins batch-size sweep directly so you
  can see where throughput peaks for larger molecules.
- AtomPack write batches now go through `add_arrays_batch(...)` for both
  builtins and the benchmark's fixed synthetic custom schema.
- `hdf5_soa` now participates in the focused write benchmark's synthetic
  custom-property rows too.
- `lmdb_packed` now participates in the custom-property write rows too, using
  the fixed synthetic custom schema from the benchmark generator.
- Pass `--out ...` to save structured JSON rows for publication/reporting.

### `convert_legacy_atompack.py`

This is the legacy-format conversion CLI.

It exists to migrate legacy `atompack_legacy` datasets into the current SOA
layout without changing the Rust reader.

It covers:

- single legacy shard conversion
- dataset-directory conversion via `manifest.json`
- batched rewrite through `add_arrays_batch(...)` for builtin and legacy custom properties
- full-fidelity fallback only when a legacy property cannot be represented by the batch writer

Typical usage:

```bash
uv run --no-sync --project atompack-py python atompack-py/benchmarks/convert_legacy_atompack.py /ogre/atompack-v2/mptrj/data.atp
uv run --no-sync --project atompack-py python atompack-py/benchmarks/convert_legacy_atompack.py /ogre/atompack-v2/mptrj --output /ogre/atompack-v2/mptrj_soa
```

Notes:

- The renamed legacy package must be importable as `atompack_legacy`.
- Converted outputs default to `--compression none`; pass `--compression zstd --level 3` if you want compressed SOA output.
- File inputs default to `<stem>.soa.atp` beside the source shard.
- Directory inputs default to a sibling `<input_name>_soa/` tree.
- Pass `--workers N` to convert different shards in parallel with separate worker processes.

### `memory_benchmark.py`

This is the focused memory benchmark.

It measures:

- open-connection memory cost
- memory scaling with worker count
- streaming-read working set

Use this when you care about RSS / private memory behavior rather than raw
throughput.

Typical usage:

```bash
uv run --no-sync --project atompack-py python atompack-py/benchmarks/memory_benchmark.py
```

Notes:

- This script currently targets pre-existing benchmark datasets.
- Pass `--create-missing` if you want it to build missing datasets itself.
- It is useful for memory investigations, but it is more environment-specific
  than the synthetic throughput scripts.

### `scaling_benchmark.py`

This is the focused read-scaling benchmark.

It measures:

- batch random-read throughput vs atom count
- scaling vs worker / thread count
- ASE-backed read scaling baselines

Use this when you want a more targeted view of read scalability than the full
suite provides.

Typical usage:

```bash
uv run --no-sync --project atompack-py python atompack-py/benchmarks/scaling_benchmark.py
uv run --no-sync --project atompack-py python atompack-py/benchmarks/scaling_benchmark.py --atoms 16 64 256 --threads 1 4 8 16
```

### `atompack_batch_benchmark.py`

This is the focused Atompack batch-read benchmark.

It measures:

- `get_molecule(i)` in chunked loops
- `get_molecules(indices)` across configurable batch sizes
- `get_molecules_flat(indices)` across configurable batch sizes
- native-threaded and fixed-thread Rayon settings in isolated subprocesses

Use this when you want to understand Atompack's own batch APIs without LMDB or
ASE baselines skewing the picture.

Typical usage:

```bash
uv run --no-sync --project atompack-py python atompack-py/benchmarks/atompack_batch_benchmark.py
uv run --no-sync --project atompack-py python atompack-py/benchmarks/atompack_batch_benchmark.py --atoms 64 256 --batch-sizes 32 128 512 2048
uv run --no-sync --project atompack-py python atompack-py/benchmarks/atompack_batch_benchmark.py --threads native 1 4 8
```

### `atompack_bestcase_read_benchmark.py`

This is the focused Atompack best-case read benchmark.

It measures:

- sequential contiguous batch reads only
- native-threaded and fixed-thread Rayon settings in isolated subprocesses
- the fastest Python-visible read path, `get_molecules_flat(...)`, by default
- adaptive batch sizes by atom count by default, so larger systems use smaller
  molecule batches

Use this when you want Atompack's ceiling for read throughput rather than a
backend-fair comparison.

It reuses the same cached synthetic datasets as `benchmark.py`, so if you
already ran the broad suite against the same scratch directory and compression
settings, it should hit the existing Atompack dataset instead of building a new
one.

Typical usage:

```bash
uv run --no-sync --project atompack-py python atompack-py/benchmarks/atompack_bestcase_read_benchmark.py
uv run --no-sync --project atompack-py python atompack-py/benchmarks/atompack_bestcase_read_benchmark.py --atoms 64 256 --threads native 4 8 16
uv run --no-sync --project atompack-py python atompack-py/benchmarks/atompack_bestcase_read_benchmark.py --atoms 64 256 --batch-sizes 512 2048 8192
uv run --no-sync --project atompack-py python atompack-py/benchmarks/atompack_bestcase_read_benchmark.py --methods flat batch --out atompack-py/benchmarks/atompack_bestcase_read.json
```

### `plot_benchmark_report.py`

This is the publication/reporting utility for benchmark JSON outputs.

It reads structured results from the maintained benchmark scripts and generates:

- GitHub-ready overview figures
- Hugging Face blog figures and caption notes
- LinkedIn Figure 1 highlight text
- a manifest of generated assets

Supported inputs:

- `benchmark.py --out ...`
- `scaling_benchmark.py --out ...`
- `memory_benchmark.py` JSON output
- `atompack_batch_benchmark.py --out ...`
- `write_benchmark.py --out ...`

Typical usage:

```bash
uv run --group docs --project atompack-py python atompack-py/benchmarks/plot_benchmark_report.py --inputs benchmark.json scaling.json memory.json batch.json --out-dir atompack-py/benchmarks/report
uv run --group docs --project atompack-py python atompack-py/benchmarks/plot_benchmark_report.py --inputs benchmark.json memory.json --out-dir atompack-py/benchmarks/report --no-batch-api
```

Notes:

- This script uses `matplotlib` from the `docs` dependency group.
- `write_benchmark.py --out ...` is supported and feeds the write-overview figure.
- Optional figures are skipped automatically when their source JSON is not provided.

## Helper Modules

### `atom_lmdb.py`

Helper code for the custom "packed LMDB" comparison backend.

It defines:

- record packing and unpacking
- codec helpers
- LMDB layout utilities

### `atom_lmdb_pickle.py`

Helper code for the "pickle LMDB" comparison backend.

It defines:

- a more Python-idiomatic LMDB payload format
- codec handling
- LMDB setup helpers

These two files are support modules, not standalone benchmark entrypoints.

### `atom_hdf5_soa.py`

Helper code for a fixed-shape SOA layout stored inside HDF5.

It defines:

- a conventional HDF5 dataset-per-field layout
- chunked molecule-axis storage for `positions`, `forces`, `energy`, and related fields
- metadata plus read helpers for single-molecule access and chunk-aware batch reconstruction

The maintained baseline keeps the layout conventional, but applies the HDF5-side
tuning that matters most for dataloader-like access:

- an HDF5 chunk size chosen for the HDF5 read/write tradeoff rather than reused from LMDB
- reader-side raw chunk cache sizing through `h5py` open options
- batch reads that regroup requested indices by chunk before rebuilding payloads

This backend depends on `h5py`, which is shipped as the `[benchmarks]` extra:
install with `pip install atompack-db[benchmarks]` (or `make py-test-benchmarks`
which now uses both the `dev` and `benchmarks` extras).

### `atom_lmdb_soa.py`

Helper code for a chunked SOA layout stored inside LMDB.

It defines:

- a fixed-size molecule SOA LMDB layout
- chunked field storage by key (`positions`, `forces`, `atomic_numbers`, ...)
- metadata and read helpers for single and batched payload access

## Choosing a Script

Use:

- `benchmark.py` for the broad performance picture
- `write_benchmark.py` for ingestion speed
- `memory_benchmark.py` for RSS / open-memory behavior
- `scaling_benchmark.py` for read scaling and batch random-read throughput
- `plot_benchmark_report.py` to turn saved JSON results into publication figures and reusable text

## Archived Benchmarks

Older experiments live under `atompack-py/benchmarks/archive/`.

Those files can still be useful as references, but the current scripts in this
directory are the maintained entrypoints.
