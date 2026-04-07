// Copyright 2026 Entalpic
//! # Atompack
//!
//! A library for storing atomistic datasets with:
//! - append-only writes
//! - direct indexed reads
//! - low-overhead mmap-backed serving paths
//! - optional compression when artifact size matters

// Re-export main types
pub mod atom;
pub mod compression;
pub mod storage;

pub use atom::{Atom, Molecule};
pub use compression::decompress as decompress_bytes;
pub use storage::{AtomDatabase, SharedMmapBytes};

/// Result type used throughout atompack
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in atompack
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),
}
