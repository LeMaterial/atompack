.. Copyright 2026 Entalpic

Contributing
============

This repository is a Rust storage engine with Python bindings. Most user-facing changes should
include updates to tests and documentation.

Repository Layout
-----------------

- `atompack/`: Rust core crate (storage engine, on-disk format, core data model).
- `atompack-py/`: PyO3 bindings crate plus the Python package.
- `docs/`: Sphinx documentation.
- `scripts/`: helper scripts (benchmark runner, stub generation/checks).

Development Setup
-----------------

Prerequisites:

- Rust toolchain (stable)
- `uv` for Python tooling

Common Commands
---------------

From the repository root:

.. code-block:: bash

   make ci          # Rust fmt/clippy/test + Python fmt/lint/test
   make test        # Rust + Python tests
   make rust-test   # Rust tests only
   make py-test     # Python tests only (builds extension first)

Throughput Smoke Tests
----------------------

Atompack has opt-in release-mode throughput smoke tests for changes that might affect storage,
batch reads, Python bindings, or ingestion performance:

.. code-block:: bash

   make perf-smoke        # Rust + Python release throughput smoke tests
   make perf-smoke-rust   # Rust storage smoke test only
   make perf-smoke-py     # Python integration smoke test only

These tests use a small 64-atom synthetic dataset and are designed to finish quickly, usually well
under one minute on a warm development machine. They print a color-coded summary table with write,
sequential-read, shuffled-read, and Python flat-batch throughput numbers, then compare them against
conservative regression floors.

Treat these numbers as smoke-test telemetry, not as publication benchmark results. The figures in
:doc:`performance` come from larger benchmark scripts with explicit baselines, filesystem context,
and reporting settings. The smoke tests are intentionally easier and exist to catch large
performance regressions before running the full benchmark suite.

The failure floors can be overridden when running on unusually small or large machines:

.. code-block:: bash

   ATOMPACK_RUST_MIN_SEQ_READ_MOL_S=100000 make perf-smoke-rust
   ATOMPACK_PY_MIN_SHUFFLED_READ_MOL_S=100000 make perf-smoke-py

The make targets default to color with ``ATOMPACK_PERF_COLOR=always``. Run
``make ATOMPACK_PERF_COLOR=never perf-smoke`` when you need plain text logs.

Docs
----

Build the Sphinx docs and mount Rust rustdoc under the same HTML output:

.. code-block:: bash

   make docs
   python -m http.server 8000 -d docs/build/html

Then open:

- `http://localhost:8000/` (Sphinx)
- `http://localhost:8000/rustdoc/atompack/index.html` (Rust core rustdoc)

Python Bindings And Stubs
-------------------------

The Python API is implemented as a compiled extension module (`atompack._atompack_rs`). The type
stubs for the extension are maintained by hand in
`atompack-py/python/atompack/_atompack_rs.pyi`. If you change the PyO3 surface in
`atompack-py/src/lib.rs`, update the stubs accordingly.

Style And Docs Expectations
---------------------------

- Rust: keep `pub` API documented with rustdoc (`///`) and run `cargo fmt`.
- Python: keep user-facing functions documented and run the linters/tests via `make ci-py`.
- If you change storage semantics or the on-disk format, update :doc:`architecture` and add or
  extend recovery/corruption tests in `atompack/src/storage/mod.rs`.
