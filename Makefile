SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

UV ?= uv
# Force a repo-local uv cache so builds work in restricted environments (e.g. CI sandboxes).
override UV_CACHE_DIR := $(CURDIR)/.uv-cache

.PHONY: help \
	rust-fmt rust-fmt-check rust-lint rust-test \
	py-sync py-fmt py-fmt-check py-lint py-lint-fix py-dev py-dev-release py-test py-test-benchmarks \
	docs-sync docs-build docs \
	fmt fmt-check lint test \
	ci-rust ci-py ci

help:
	@echo "Atompack dev commands:"
	@echo "  make fmt             Format Rust + Python"
	@echo "  make fmt-check        Check formatting"
	@echo "  make lint             Run linters"
	@echo "  make test             Run default tests (Rust + Python core suite)"
	@echo "  make ci               Run all CI checks"
	@echo ""
	@echo "Language-specific:"
	@echo "  make rust-fmt         cargo fmt --all"
	@echo "  make rust-lint        cargo clippy --workspace --all-targets -- -D warnings"
	@echo "  make rust-test        cargo test --workspace"
	@echo "  make py-fmt           uv ruff format python (atompack-py)"
	@echo "  make py-lint          uv ruff check python (atompack-py)"
	@echo "  make py-test          uv pytest core suite (atompack-py/tests without benchmark tooling)"
	@echo "  make py-test-benchmarks  uv pytest benchmark tooling suite (manual only)"
	@echo "  make py-dev           uv maturin develop (atompack-py)"
	@echo "  make py-dev-release   uv maturin develop -r (atompack-py)"
	@echo ""
	@echo "Docs:"
	@echo "  make docs-sync        Install docs deps (uv, atompack-py docs group)"
	@echo "  make docs-build       Build Sphinx HTML into docs/build/html"
	@echo "  make docs             Build Sphinx + Rust rustdoc and mount under docs/build/html/rustdoc"

rust-fmt:
	cargo fmt --all

rust-fmt-check:
	cargo fmt --all --check

rust-lint:
	cargo clippy --workspace --all-targets -- -D warnings

rust-test:
	cargo test --workspace

py-sync:
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found; install from https://docs.astral.sh/uv/" && exit 1)
	cd atompack-py && UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync --extra dev --locked

py-fmt:
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found; install from https://docs.astral.sh/uv/" && exit 1)
	cd atompack-py && UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --extra dev --locked ruff format python

py-fmt-check:
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found; install from https://docs.astral.sh/uv/" && exit 1)
	cd atompack-py && UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --extra dev --locked ruff format --check python

py-lint:
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found; install from https://docs.astral.sh/uv/" && exit 1)
	cd atompack-py && UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --extra dev --locked ruff check python

py-lint-fix:
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found; install from https://docs.astral.sh/uv/" && exit 1)
	cd atompack-py && UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --extra dev --locked ruff check --fix python

py-dev:
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found; install from https://docs.astral.sh/uv/" && exit 1)
	cd atompack-py && UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --extra dev --locked --with "maturin>=1.4,<2.0" maturin develop

py-dev-release:
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found; install from https://docs.astral.sh/uv/" && exit 1)
	cd atompack-py && UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --extra dev --locked --with "maturin>=1.4,<2.0" maturin develop -r

py-test: py-dev
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found; install from https://docs.astral.sh/uv/" && exit 1)
	cd atompack-py && UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --extra dev --locked pytest tests --ignore=tests/benchmarks

py-test-benchmarks: py-dev
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found; install from https://docs.astral.sh/uv/" && exit 1)
	cd atompack-py && UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --extra dev --extra benchmarks --locked pytest tests/benchmarks

docs-sync:
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found; install from https://docs.astral.sh/uv/" && exit 1)
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync --project atompack-py --group docs --locked

docs-build: docs-sync
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found; install from https://docs.astral.sh/uv/" && exit 1)
	# Avoid serving stale pages (e.g. removed .rst files) from prior builds.
	rm -rf docs/build/html docs/build/doctrees
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --project atompack-py --group docs --locked sphinx-build -M html docs/source docs/build

# One-stop target: build Sphinx docs and copy Rust rustdoc under the same HTML output.
docs: docs-build
	cargo doc --workspace --no-deps
	rm -rf docs/build/html/rustdoc
	mkdir -p docs/build/html/rustdoc
	cp -r target/doc/* docs/build/html/rustdoc/

fmt: rust-fmt py-fmt

fmt-check: rust-fmt-check py-fmt-check

lint: rust-lint py-lint

test: rust-test py-test

ci-rust: rust-fmt-check rust-lint rust-test

ci-py: py-fmt-check py-lint py-test

ci: ci-rust ci-py
