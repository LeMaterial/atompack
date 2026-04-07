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
