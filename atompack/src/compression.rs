// Copyright 2026 Entalpic
//! Compression utilities for atom data
//!
//! This module handles compressing and decompressing atom data efficiently.

use crate::{Error, Result};

/// Compression algorithm to use
///
/// ## Rust Concepts:
/// - `enum`: Defines a type that can be one of several variants (like union/tagged union)
/// - Each variant can hold different data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionType {
    /// No compression (fastest, largest size)
    #[default]
    None,
    /// LZ4 compression (very fast, good compression)
    Lz4,
    /// Zstandard compression (slower, better compression)
    /// The number is the compression level (1-22, higher = more compression)
    Zstd(i32),
}

/// Compress raw bytes using the specified algorithm
///
/// ## Rust Concepts:
/// - `&[u8]`: Slice of bytes (borrowed view into an array/Vec)
/// - `Vec<u8>`: Owned vector of bytes
/// - `match`: Pattern matching (like switch but more powerful)
/// - `?`: Propagate errors up (if error occurs, return early with that error)
pub fn compress(data: &[u8], compression: CompressionType) -> Result<Vec<u8>> {
    match compression {
        CompressionType::None => {
            // Just copy the data
            Ok(data.to_vec())
        }
        CompressionType::Lz4 => {
            // LZ4 compression - use high compression mode
            let compressed = lz4::block::compress(
                data,
                Some(lz4::block::CompressionMode::HIGHCOMPRESSION(12)),
                false,
            )
            .map_err(|e| Error::Compression(format!("LZ4 compression failed: {}", e)))?;
            Ok(compressed)
        }
        CompressionType::Zstd(level) => {
            // Zstandard compression
            zstd::bulk::compress(data, level)
                .map_err(|e| Error::Compression(format!("Zstd compression failed: {}", e)))
        }
    }
}

/// Decompress bytes that were compressed with the specified algorithm
pub fn decompress(
    compressed: &[u8],
    compression: CompressionType,
    expected_size: Option<usize>,
) -> Result<Vec<u8>> {
    match compression {
        CompressionType::None => Ok(compressed.to_vec()),
        CompressionType::Lz4 => {
            // LZ4 decompression - if we have expected size, use it as a hint
            let max_size = expected_size.unwrap_or(compressed.len() * 100); // Conservative estimate
            lz4::block::decompress(compressed, Some(max_size as i32))
                .map_err(|e| Error::Compression(format!("LZ4 decompression failed: {}", e)))
        }
        CompressionType::Zstd(_) => zstd::bulk::decompress(
            compressed,
            expected_size.unwrap_or(compressed.len() * 10), // Estimate if not provided
        )
        .map_err(|e| Error::Compression(format!("Zstd decompression failed: {}", e))),
    }
}

/// Convert a slice of atoms to raw bytes
///
/// ## Rust Concepts:
/// - `bytemuck::cast_slice`: Zero-copy cast from one type to another
/// - This is safe because Atom has `#[repr(C)]` - predictable layout
/// - Generic function: `<T>` means this works for any type T that meets the bounds
pub fn atoms_to_bytes<T: bytemuck::Pod>(atoms: &[T]) -> &[u8] {
    bytemuck::cast_slice(atoms)
}

/// Convert raw bytes back to a slice of atoms
///
/// ## Rust Concepts:
/// - `try_cast_slice`: Safe conversion that returns an error if alignment is wrong
/// - This ensures we don't create invalid data
pub fn bytes_to_atoms<T: bytemuck::Pod>(bytes: &[u8]) -> Result<&[T]> {
    bytemuck::try_cast_slice(bytes).map_err(|e| Error::InvalidData(format!("Cast failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Atom;

    #[test]
    fn test_compression_roundtrip() {
        let data = b"Hello, World! This is test data for compression.";

        for compression in &[
            CompressionType::None,
            CompressionType::Lz4,
            CompressionType::Zstd(3),
        ] {
            let compressed = compress(data, *compression).unwrap();
            let decompressed = decompress(&compressed, *compression, Some(data.len())).unwrap();
            assert_eq!(data.as_slice(), decompressed.as_slice());
        }
    }

    #[test]
    fn test_atom_compression() {
        let atoms = vec![
            Atom::new(1.0, 2.0, 3.0, 6),
            Atom::new(4.0, 5.0, 6.0, 8),
            Atom::new(7.0, 8.0, 9.0, 1),
        ];

        // Convert atoms to bytes
        let bytes = atoms_to_bytes(&atoms);

        // Compress
        let compressed = compress(bytes, CompressionType::Zstd(3)).unwrap();
        println!(
            "Original: {} bytes, Compressed: {} bytes",
            bytes.len(),
            compressed.len()
        );

        // Decompress
        let decompressed =
            decompress(&compressed, CompressionType::Zstd(3), Some(bytes.len())).unwrap();

        // Convert back to atoms
        let recovered: &[Atom] = bytes_to_atoms(&decompressed).unwrap();

        assert_eq!(atoms.len(), recovered.len());
        for (a, b) in atoms.iter().zip(recovered.iter()) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.z, b.z);
            assert_eq!(a.atomic_number, b.atomic_number);
        }
    }
}
