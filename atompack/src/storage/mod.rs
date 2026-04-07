// Copyright 2026 Entalpic
//! # Atompack Database Storage
//!
//! Append-only storage for atomistic datasets with fast indexed reads and an
//! mmap-backed read path. Molecules are stored as self-contained records; an
//! optional codec can be applied per record when size matters.
//!
//! ## File layout
//!
//! ```text
//! ┌──────────────────────────────────────┐
//! │ Header slot A  (4096 bytes)          │  ← crash-safe: two slots alternate
//! │ Header slot B  (4096 bytes)          │     on each flush (generation counter)
//! ├──────────────────────────────────────┤
//! │ Record 0  (SOA molecule record)      │
//! │ Record 1  (SOA molecule record)      │
//! │ ...                                  │
//! │ Record N-1                           │
//! ├──────────────────────────────────────┤
//! │ Index  [count:u64][entries...]       │  ← rewritten on flush at end of file
//! └──────────────────────────────────────┘
//! ```

use crate::atom::PropertyValue;
use crate::compression::{CompressionType, compress, decompress};
use crate::{Error, Molecule, Result};
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod header;
mod index;
mod soa;

use self::header::{Header, encode_header_slot, read_best_header};
use self::index::{IndexStorage, MoleculeIndex, decode_index, encode_index};
use self::soa::{arr, deserialize_molecule_soa, serialize_molecule_soa};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"ATPK";
/// Bump only for incompatible layout changes (not crate version).
const FILE_FORMAT_VERSION: u32 = 2;
const RECORD_FORMAT_SOA: u32 = 2;

// Section kind tags (inside each SOA record)
const KIND_BUILTIN: u8 = 0;
const KIND_ATOM_PROP: u8 = 1;
const KIND_MOL_PROP: u8 = 2;

// Type tags for section payloads
const TYPE_FLOAT: u8 = 0; // f64 scalar
const TYPE_INT: u8 = 1; // i64 scalar
const TYPE_STRING: u8 = 2; // utf8 bytes
const TYPE_F64_ARRAY: u8 = 3;
const TYPE_VEC3_F32: u8 = 4; // Vec<[f32; 3]>
const TYPE_I64_ARRAY: u8 = 5;
const TYPE_F32_ARRAY: u8 = 6;
const TYPE_VEC3_F64: u8 = 7; // Vec<[f64; 3]>
const TYPE_I32_ARRAY: u8 = 8;
const TYPE_BOOL3: u8 = 9; // [bool; 3]
const TYPE_MAT3X3_F64: u8 = 10; // [[f64; 3]; 3]

// Two redundant page-aligned header slots for crash safety.
const HEADER_SLOT_SIZE: usize = 4096;
const HEADER_REGION_SIZE: u64 = (HEADER_SLOT_SIZE as u64) * 2;
const HEADER_SLOT_A_OFFSET: u64 = 0;
const HEADER_SLOT_B_OFFSET: u64 = HEADER_SLOT_SIZE as u64;

// Index: [u64 count][count × MoleculeIndexEntry]
const INDEX_PREFIX_SIZE: usize = 8;
const INDEX_ENTRY_SIZE: usize = 8 + 4 + 4 + 4; // offset:u64 + compressed:u32 + uncompressed:u32 + n_atoms:u32

// ---------------------------------------------------------------------------
// 4. SharedMmapBytes & AtomDatabase (public API)
// ---------------------------------------------------------------------------

/// Ref-counted slice into a memory-mapped file. Keeps the mmap alive as long
/// as any view exists.
#[derive(Debug, Clone)]
pub struct SharedMmapBytes {
    mmap: Arc<Mmap>,
    start: usize,
    end: usize,
}

impl SharedMmapBytes {
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap[self.start..self.end]
    }
}

pub struct AtomDatabase {
    path: PathBuf,
    compression: CompressionType,
    generation: u64,
    record_format: u32,
    /// File offset up to which data has been committed (flushed). Anything
    /// beyond this is an uncommitted tail from a previous crash.
    committed_end: u64,
    truncate_tail_on_next_write: bool,
    index: IndexStorage,
    file: Option<File>,
    data_mmap: Option<Arc<Mmap>>,
}

impl AtomDatabase {
    // -- Creation & opening --------------------------------------------------

    /// Create a new empty database.
    pub fn create<P: AsRef<Path>>(path: P, compression: CompressionType) -> Result<Self> {
        Self::create_with_format(path, compression)
    }

    /// Alias for `create` (kept for backward compatibility).
    pub fn create_soa<P: AsRef<Path>>(path: P, compression: CompressionType) -> Result<Self> {
        Self::create(path, compression)
    }

    fn create_with_format<P: AsRef<Path>>(path: P, compression: CompressionType) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::create(&path)?;
        let header = Header {
            generation: 0,
            data_start: HEADER_REGION_SIZE,
            num_molecules: 0,
            compression,
            record_format: RECORD_FORMAT_SOA,
            index_offset: 0,
            index_len: 0,
        };
        let slot = encode_header_slot(header);
        file.write_all(&slot)?;
        file.write_all(&slot)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);

        Ok(Self {
            path,
            compression,
            generation: 0,
            record_format: RECORD_FORMAT_SOA,
            committed_end: HEADER_REGION_SIZE,
            truncate_tail_on_next_write: false,
            index: IndexStorage::InMemory(Vec::new()),
            file: None,
            data_mmap: None,
        })
    }

    // -- Opening -------------------------------------------------------------

    /// Open an existing database (loads index into memory, read-write).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_options(path, false, false)
    }

    /// Open read-only with memory-mapped index (lower memory, read-only).
    pub fn open_mmap<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_options(path, true, false)
    }

    /// Open read-only with memory-mapped index and pre-faulted pages
    /// (eliminates per-read page faults at the cost of upfront I/O).
    pub fn open_mmap_populate<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_options(path, true, true)
    }

    fn open_with_options<P: AsRef<Path>>(path: P, use_mmap: bool, populate: bool) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;

        // Determine file format version from the first 8 bytes: [magic][u32 version LE].
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)?;
        if &prefix[0..4] != MAGIC {
            return Err(Error::InvalidData("Invalid file format".into()));
        }
        let version = u32::from_le_bytes(arr(&prefix[4..8])?);

        if version != FILE_FORMAT_VERSION {
            return Err(Error::InvalidData(format!(
                "Unsupported file format version {} (expected {})",
                version, FILE_FORMAT_VERSION
            )));
        }

        Self::open_v1(path, file, use_mmap, populate)
    }

    fn open_v1(path: PathBuf, mut file: File, use_mmap: bool, populate: bool) -> Result<Self> {
        // Read the best valid header slot (crash-safe).
        let header = read_best_header(&mut file)?;
        if header.record_format != RECORD_FORMAT_SOA {
            return Err(Error::InvalidData(format!(
                "Unsupported legacy record format {}. Atompack no longer supports bincode databases.",
                header.record_format
            )));
        }

        let committed_end = if header.index_offset == 0 || header.index_len == 0 {
            header.data_start
        } else {
            header
                .index_offset
                .checked_add(header.index_len)
                .ok_or_else(|| Error::InvalidData("Index end overflow".into()))?
        };

        let file_size = file.metadata()?.len();
        let truncate_tail_on_next_write = !use_mmap && file_size > committed_end;

        // Read or memory-map the index
        // When mmap mode is requested, create a single mmap for both index and data access
        let data_mmap = if use_mmap {
            let mmap_file = File::open(&path)?;
            let mmap = unsafe { Mmap::map(&mmap_file)? };
            #[cfg(target_os = "linux")]
            if populate {
                let _ = mmap.advise(memmap2::Advice::PopulateRead);
            }
            Some(Arc::new(mmap))
        } else {
            None
        };

        let index = if header.index_offset > 0 {
            if use_mmap {
                IndexStorage::MemoryMapped {
                    mmap: unsafe { Mmap::map(&File::open(&path)?)? },
                    index_offset: header.index_offset,
                    count: header.num_molecules as usize,
                }
            } else {
                file.seek(SeekFrom::Start(header.index_offset))?;
                let mut index_bytes = vec![0u8; header.index_len as usize];
                file.read_exact(&mut index_bytes)?;
                let vec = decode_index(&index_bytes)?;
                IndexStorage::InMemory(vec)
            }
        } else {
            IndexStorage::InMemory(Vec::new())
        };

        Ok(Self {
            path,
            compression: header.compression,
            generation: header.generation,
            record_format: header.record_format,
            committed_end,
            truncate_tail_on_next_write,
            index,
            file: Some(file),
            data_mmap,
        })
    }

    fn truncate_uncommitted_tail_if_needed(&mut self) -> Result<()> {
        if !self.truncate_tail_on_next_write {
            return Ok(());
        }

        let file = OpenOptions::new().write(true).open(&self.path)?;
        let file_size = file.metadata()?.len();

        if file_size > self.committed_end {
            file.set_len(self.committed_end)?;
            file.sync_all()?;
        }

        self.truncate_tail_on_next_write = false;
        self.file = None;
        Ok(())
    }

    // -- Writing -------------------------------------------------------------

    /// Add a single molecule.
    pub fn add_molecule(&mut self, molecule: &Molecule) -> Result<()> {
        self.add_molecules(&[molecule])
    }

    /// Add multiple molecules. Serialization and compression run in parallel
    /// (rayon); the compressed blobs are then appended sequentially.
    pub fn add_molecules(&mut self, molecules: &[&Molecule]) -> Result<()> {
        if molecules.is_empty() {
            return Ok(());
        }

        let serialized: Vec<(Vec<u8>, u32)> = molecules
            .par_iter()
            .map(|mol| {
                let bytes = serialize_molecule_soa(mol)?;
                let num_atoms = mol.len() as u32;
                Ok((bytes, num_atoms))
            })
            .collect::<Result<Vec<_>>>()?;

        self.append_owned_soa_records(serialized)
    }

    /// Add pre-serialized SOA records, compressing in parallel and appending to the file.
    ///
    /// Each entry is `(soa_bytes, num_atoms)`. The bytes must be valid SOA-encoded molecule
    /// records (the same format `serialize_molecule_soa` produces). This skips serialization
    /// entirely — useful when the caller already has SOA bytes (e.g. from a View or
    /// direct numpy-to-SOA construction).
    pub fn add_raw_soa_records(&mut self, records: &[(&[u8], u32)]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        self.append_soa_records(records)
    }

    #[doc(hidden)]
    pub fn add_owned_soa_records(&mut self, records: Vec<(Vec<u8>, u32)>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        self.append_owned_soa_records(records)
    }

    fn append_soa_records(&mut self, records: &[(&[u8], u32)]) -> Result<()> {
        if matches!(&self.index, IndexStorage::MemoryMapped { .. }) {
            return Err(Error::InvalidData(
                "Cannot add molecules to a database opened with a memory-mapped index (read-only); reopen without mmap to write."
                    .into(),
            ));
        }

        self.truncate_uncommitted_tail_if_needed()?;

        let compression = self.compression;

        // Step 1: Compress all records in parallel.
        let compressed_records: Vec<(Vec<u8>, u32, u32)> = records
            .par_iter()
            .map(|(bytes, num_atoms)| {
                let uncompressed_size = bytes.len() as u32;
                let compressed = compress(bytes, compression)?;
                Ok((compressed, uncompressed_size, *num_atoms))
            })
            .collect::<Result<Vec<_>>>()?;

        // Step 2: Write all compressed data sequentially (file I/O must be sequential)
        let mut file = OpenOptions::new().append(true).open(&self.path)?;

        let mut offset = file.seek(SeekFrom::End(0))?;
        let mut new_indices = Vec::with_capacity(compressed_records.len());

        for (compressed_data, uncompressed_size, num_atoms) in compressed_records {
            file.write_all(&compressed_data)?;

            new_indices.push(MoleculeIndex {
                offset,
                compressed_size: compressed_data.len() as u32,
                uncompressed_size,
                num_atoms,
            });

            offset += compressed_data.len() as u64;
        }

        file.flush()?;
        self.index.extend(new_indices)?;

        Ok(())
    }

    fn append_owned_soa_records(&mut self, records: Vec<(Vec<u8>, u32)>) -> Result<()> {
        if matches!(&self.index, IndexStorage::MemoryMapped { .. }) {
            return Err(Error::InvalidData(
                "Cannot add molecules to a database opened with a memory-mapped index (read-only); reopen without mmap to write."
                    .into(),
            ));
        }

        self.truncate_uncommitted_tail_if_needed()?;

        let compression = self.compression;

        let compressed_records: Vec<(Vec<u8>, u32, u32)> = records
            .into_par_iter()
            .map(|(bytes, num_atoms)| {
                let uncompressed_size = bytes.len() as u32;
                let compressed = compress(&bytes, compression)?;
                Ok((compressed, uncompressed_size, num_atoms))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut file = OpenOptions::new().append(true).open(&self.path)?;

        let mut offset = file.seek(SeekFrom::End(0))?;
        let mut new_indices = Vec::with_capacity(compressed_records.len());

        for (compressed_data, uncompressed_size, num_atoms) in compressed_records {
            file.write_all(&compressed_data)?;

            new_indices.push(MoleculeIndex {
                offset,
                compressed_size: compressed_data.len() as u32,
                uncompressed_size,
                num_atoms,
            });

            offset += compressed_data.len() as u64;
        }

        file.flush()?;
        self.index.extend(new_indices)?;

        Ok(())
    }

    // -- Reading -------------------------------------------------------------

    /// Read a single molecule by index (seek + decompress + deserialize).
    pub fn get_molecule(&mut self, index: usize) -> Result<Molecule> {
        let mol_index = self
            .index
            .get(index)
            .ok_or_else(|| Error::InvalidData(format!("Index {} out of bounds", index)))?;

        if self.file.is_none() {
            self.file = Some(File::open(&self.path)?);
        }
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| Error::InvalidData("File handle missing after open".into()))?;

        file.seek(SeekFrom::Start(mol_index.offset))?;
        let mut compressed = vec![0u8; mol_index.compressed_size as usize];
        file.read_exact(&mut compressed)?;

        let decompressed = decompress(
            &compressed,
            self.compression,
            Some(mol_index.uncompressed_size as usize),
        )?;
        deserialize_molecule_soa(&decompressed)
    }

    /// Read multiple molecules in parallel.
    pub fn get_molecules(&self, indices: &[usize]) -> Result<Vec<Molecule>> {
        let raw = self.get_raw_bytes(indices)?;
        raw.into_par_iter()
            .map(|bytes| deserialize_molecule_soa(&bytes))
            .collect()
    }

    /// Atom count for a molecule (from the index, no I/O).
    pub fn num_atoms(&self, index: usize) -> Option<u32> {
        self.index.get(index).map(|e| e.num_atoms)
    }

    /// Decompressed raw SOA bytes without deserializing (for fast-path parsers).
    pub fn get_raw_bytes(&self, indices: &[usize]) -> Result<Vec<Vec<u8>>> {
        let index_refs = self.resolve_indices(indices)?;
        self.decompress_entries_parallel(&index_refs)
    }

    /// Like `get_raw_bytes` but also returns atom counts per molecule.
    pub fn read_decompress_parallel(&self, indices: &[usize]) -> Result<(Vec<Vec<u8>>, Vec<u32>)> {
        let index_refs = self.resolve_indices(indices)?;
        let n_atoms: Vec<u32> = index_refs.iter().map(|e| e.num_atoms).collect();
        let raw_bytes = self.decompress_entries_parallel(&index_refs)?;
        Ok((raw_bytes, n_atoms))
    }

    /// Validate indices and collect the corresponding `MoleculeIndex` entries.
    fn resolve_indices(&self, indices: &[usize]) -> Result<Vec<MoleculeIndex>> {
        indices
            .iter()
            .map(|&i| {
                self.index
                    .get(i)
                    .ok_or_else(|| Error::InvalidData(format!("Index {} out of bounds", i)))
            })
            .collect()
    }

    /// Decompress multiple records in parallel (mmap or per-thread file handles).
    fn decompress_entries_parallel(&self, entries: &[MoleculeIndex]) -> Result<Vec<Vec<u8>>> {
        let compression = self.compression;

        if let Some(ref mmap) = self.data_mmap {
            entries
                .par_iter()
                .map(|e| {
                    let start = e.offset as usize;
                    let end = start + e.compressed_size as usize;
                    decompress(
                        &mmap[start..end],
                        compression,
                        Some(e.uncompressed_size as usize),
                    )
                })
                .collect()
        } else {
            let path = self.path.clone();
            entries
                .par_iter()
                .map(|e| {
                    let mut file = File::open(&path)?;
                    file.seek(SeekFrom::Start(e.offset))?;
                    let mut compressed = vec![0u8; e.compressed_size as usize];
                    file.read_exact(&mut compressed)?;
                    decompress(&compressed, compression, Some(e.uncompressed_size as usize))
                })
                .collect()
        }
    }

    // -- Flush ---------------------------------------------------------------

    /// Write the index to the end of the file and update one header slot.
    /// Until flush is called, added molecules are recoverable from the file
    /// but not visible to readers that open the database.
    pub fn flush(&mut self) -> Result<()> {
        if matches!(&self.index, IndexStorage::MemoryMapped { .. }) {
            return Err(Error::InvalidData(
                "Cannot flush a database opened with a memory-mapped index (read-only); reopen without mmap to write."
                    .into(),
            ));
        }

        self.truncate_uncommitted_tail_if_needed()?;

        let index_vec = match &self.index {
            IndexStorage::InMemory(vec) => vec.as_slice(),
            IndexStorage::MemoryMapped { .. } => unreachable!(),
        };

        let index_bytes = encode_index(index_vec);

        let mut file = OpenOptions::new().write(true).open(&self.path)?;
        let index_offset = file.seek(SeekFrom::End(0))?;
        file.write_all(&index_bytes)?;
        file.flush()?;
        // Ensure the new index is on disk before publishing it in the header.
        file.sync_data()?;

        let new_generation = self.generation.saturating_add(1);
        let header = Header {
            generation: new_generation,
            data_start: HEADER_REGION_SIZE,
            num_molecules: self.index.len() as u64,
            compression: self.compression,
            record_format: self.record_format,
            index_offset,
            index_len: index_bytes.len() as u64,
        };

        let slot_offset = if new_generation.is_multiple_of(2) {
            HEADER_SLOT_A_OFFSET
        } else {
            HEADER_SLOT_B_OFFSET
        };
        let slot = encode_header_slot(header);
        file.seek(SeekFrom::Start(slot_offset))?;
        file.write_all(&slot)?;
        file.flush()?;
        file.sync_all()?;

        self.generation = new_generation;
        self.committed_end = index_offset
            .checked_add(index_bytes.len() as u64)
            .ok_or_else(|| Error::InvalidData("Index end overflow".into()))?;
        self.truncate_tail_on_next_write = false;
        Ok(())
    }

    // -- Accessors -----------------------------------------------------------

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn compression(&self) -> CompressionType {
        self.compression
    }

    /// Compressed bytes for a molecule from the mmap (None if no mmap).
    pub fn get_compressed_slice(&self, index: usize) -> Option<&[u8]> {
        let mol_index = self.index.get(index)?;
        let mmap = self.data_mmap.as_ref()?;
        let start = mol_index.offset as usize;
        let end = start + mol_index.compressed_size as usize;
        if end <= mmap.len() {
            Some(&mmap[start..end])
        } else {
            None
        }
    }

    /// Ref-counted handle to a molecule's compressed bytes in the mmap.
    pub fn get_shared_mmap_bytes(&self, index: usize) -> Option<SharedMmapBytes> {
        let mol_index = self.index.get(index)?;
        let mmap = self.data_mmap.as_ref()?;
        let start = mol_index.offset as usize;
        let end = start + mol_index.compressed_size as usize;
        if end <= mmap.len() {
            Some(SharedMmapBytes {
                mmap: Arc::clone(mmap),
                start,
                end,
            })
        } else {
            None
        }
    }

    /// Uncompressed size hint (for pre-allocating decompress buffers).
    pub fn uncompressed_size(&self, index: usize) -> Option<u32> {
        self.index.get(index).map(|e| e.uncompressed_size)
    }
}

// ---------------------------------------------------------------------------
// 6. Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Atom;
    use tempfile::NamedTempFile;

    fn molecule_from_atoms(atoms: Vec<Atom>) -> Molecule {
        Molecule::from_atoms(atoms)
    }

    #[test]
    fn test_database_create_and_add() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();

        let mut db = AtomDatabase::create(path, CompressionType::Zstd(3)).unwrap();

        let mol1 = molecule_from_atoms(vec![
            Atom::new(0.0, 0.0, 0.0, 8),
            Atom::new(1.0, 0.0, 0.0, 1),
        ]);

        let mol2 = molecule_from_atoms(vec![
            Atom::new(0.0, 0.0, 0.0, 6),
            Atom::new(1.0, 0.0, 0.0, 6),
            Atom::new(2.0, 0.0, 0.0, 6),
        ]);

        db.add_molecules(&[&mol1, &mol2]).unwrap();
        db.flush().unwrap();

        assert_eq!(db.len(), 2);

        let retrieved = db.get_molecule(0).unwrap();
        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved.atomic_numbers[0], 8);
    }

    #[test]
    fn test_database_open() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        // Create and write
        {
            let mut db = AtomDatabase::create(&path, CompressionType::Lz4).unwrap();
            let mol = molecule_from_atoms(vec![Atom::new(1.0, 2.0, 3.0, 6)]);
            db.add_molecule(&mol).unwrap();
            db.flush().unwrap();
        }

        // Reopen and read
        {
            let mut db = AtomDatabase::open(&path).unwrap();
            assert_eq!(db.len(), 1);
            let mol = db.get_molecule(0).unwrap();
            assert_eq!(mol.positions[0], [1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn test_database_with_forces_and_energy() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        // Create molecule with forces and energy
        let mut mol = molecule_from_atoms(vec![
            Atom::new(0.0, 0.0, 0.0, 6),
            Atom::new(1.0, 0.0, 0.0, 8),
        ]);
        mol.forces = Some(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        mol.energy = Some(-123.456);

        // Write to database
        {
            let mut db = AtomDatabase::create(&path, CompressionType::Zstd(3)).unwrap();
            db.add_molecule(&mol).unwrap();
            db.flush().unwrap();
        }

        // Read back and verify
        {
            let mut db = AtomDatabase::open(&path).unwrap();
            let retrieved = db.get_molecule(0).unwrap();

            // Check forces are preserved
            assert!(retrieved.forces.is_some());
            let forces = retrieved.forces.unwrap();
            assert_eq!(forces.len(), 2);
            assert_eq!(forces[0], [0.1, 0.2, 0.3]);
            assert_eq!(forces[1], [0.4, 0.5, 0.6]);

            // Check energy is preserved
            assert!(retrieved.energy.is_some());
            assert_eq!(retrieved.energy.unwrap(), -123.456);
        }
    }

    #[test]
    fn test_header_slot_recovery() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mol1 = molecule_from_atoms(vec![Atom::new(0.0, 0.0, 0.0, 6)]);
        let mol2 = molecule_from_atoms(vec![Atom::new(1.0, 2.0, 3.0, 8)]);

        // Two flushes so both header slots contain a valid committed state.
        {
            let mut db = AtomDatabase::create(&path, CompressionType::Zstd(3)).unwrap();
            db.add_molecule(&mol1).unwrap();
            db.flush().unwrap(); // gen=1 (slot B)
            db.add_molecule(&mol2).unwrap();
            db.flush().unwrap(); // gen=2 (slot A)
        }

        // Corrupt the latest header slot (slot A). Open should fall back to slot B.
        {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let pos = HEADER_SLOT_A_OFFSET + (HEADER_SLOT_SIZE as u64) - 1;
            file.seek(SeekFrom::Start(pos)).unwrap();
            let mut byte = [0u8; 1];
            file.read_exact(&mut byte).unwrap();
            byte[0] ^= 0xFF;
            file.seek(SeekFrom::Start(pos)).unwrap();
            file.write_all(&byte).unwrap();
            file.flush().unwrap();
        }

        let mut db = AtomDatabase::open(&path).unwrap();
        assert_eq!(db.len(), 1);
        let mol = db.get_molecule(0).unwrap();
        assert_eq!(mol.atomic_numbers[0], 6);
    }

    #[test]
    fn test_truncate_uncommitted_tail_on_write() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let compression = CompressionType::None;
        let mol1 = molecule_from_atoms(vec![Atom::new(0.0, 0.0, 0.0, 6)]);

        // Create a committed database state with a single molecule.
        {
            let mut db = AtomDatabase::create(&path, compression).unwrap();
            db.add_molecule(&mol1).unwrap();
            db.flush().unwrap();
        }

        let committed_size = std::fs::metadata(&path).unwrap().len();

        // Simulate an uncommitted tail (e.g., crash during a write) by appending garbage.
        {
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(&[0u8; 123]).unwrap();
            file.flush().unwrap();
        }

        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            committed_size + 123
        );

        // Opening + writing should truncate the uncommitted tail before appending new data.
        let mut db = AtomDatabase::open(&path).unwrap();
        let mol2 = molecule_from_atoms(vec![Atom::new(1.0, 2.0, 3.0, 8)]);
        let mol2_bytes = serialize_molecule_soa(&mol2).unwrap();
        let mol2_compressed = compress(&mol2_bytes, compression).unwrap();
        db.add_molecule(&mol2).unwrap();

        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            committed_size + (mol2_compressed.len() as u64)
        );
    }

    #[test]
    fn test_soa_round_trip_all_fields() {
        use crate::atom::PropertyValue;

        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut mol = molecule_from_atoms(vec![
            Atom::new(1.0, 2.0, 3.0, 6),
            Atom::new(4.0, 5.0, 6.0, 8),
        ]);
        mol.name = Some("water".to_string());
        mol.energy = Some(-42.5);
        mol.forces = Some(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        mol.charges = Some(vec![-0.5, 0.5]);
        mol.velocities = Some(vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        mol.cell = Some([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
        mol.pbc = Some([true, true, false]);
        mol.stress = Some([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        // atom_properties
        mol.atom_properties.insert(
            "mulliken".to_string(),
            PropertyValue::FloatArray(vec![0.1, 0.2]),
        );
        mol.atom_properties
            .insert("spin".to_string(), PropertyValue::Int32Array(vec![1, -1]));

        // properties
        mol.properties
            .insert("bandgap".to_string(), PropertyValue::Float(2.5));
        mol.properties.insert(
            "formula".to_string(),
            PropertyValue::String("CO".to_string()),
        );
        mol.properties.insert(
            "eigenvalues".to_string(),
            PropertyValue::FloatArray(vec![1.0, 2.0, 3.0]),
        );

        // Write and read back
        {
            let mut db = AtomDatabase::create(&path, CompressionType::Zstd(3)).unwrap();
            db.add_molecule(&mol).unwrap();
            db.flush().unwrap();
        }

        let mut db = AtomDatabase::open(&path).unwrap();
        let r = db.get_molecule(0).unwrap();

        // Verify all fields
        assert_eq!(r.name.as_deref(), Some("water"));
        assert_eq!(r.len(), 2);
        assert_eq!(r.positions[0], [1.0, 2.0, 3.0]);
        assert_eq!(r.atomic_numbers[0], 6);
        assert_eq!(r.positions[1], [4.0, 5.0, 6.0]);
        assert_eq!(r.atomic_numbers[1], 8);
        assert_eq!(r.energy, Some(-42.5));
        assert_eq!(
            r.forces.as_ref().unwrap(),
            &[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        );
        assert_eq!(r.charges.as_ref().unwrap(), &[-0.5, 0.5]);
        assert_eq!(
            r.velocities.as_ref().unwrap(),
            &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        );
        assert_eq!(
            r.cell.unwrap(),
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        );
        assert_eq!(r.pbc, Some([true, true, false]));
        assert_eq!(
            r.stress.unwrap(),
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        );

        // atom_properties
        match r.atom_properties.get("mulliken").unwrap() {
            PropertyValue::FloatArray(v) => assert_eq!(v, &[0.1, 0.2]),
            _ => panic!("wrong type for mulliken"),
        }
        match r.atom_properties.get("spin").unwrap() {
            PropertyValue::Int32Array(v) => assert_eq!(v, &[1, -1]),
            _ => panic!("wrong type for spin"),
        }

        // properties
        match r.properties.get("bandgap").unwrap() {
            PropertyValue::Float(v) => assert_eq!(*v, 2.5),
            _ => panic!("wrong type for bandgap"),
        }
        match r.properties.get("formula").unwrap() {
            PropertyValue::String(v) => assert_eq!(v, "CO"),
            _ => panic!("wrong type for formula"),
        }
        match r.properties.get("eigenvalues").unwrap() {
            PropertyValue::FloatArray(v) => assert_eq!(v, &[1.0, 2.0, 3.0]),
            _ => panic!("wrong type for eigenvalues"),
        }
    }

    #[test]
    fn test_soa_round_trip_minimal() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mol = molecule_from_atoms(vec![Atom::new(1.0, 2.0, 3.0, 6)]);
        {
            let mut db = AtomDatabase::create(&path, CompressionType::None).unwrap();
            db.add_molecule(&mol).unwrap();
            db.flush().unwrap();
        }

        let mut db = AtomDatabase::open(&path).unwrap();
        let r = db.get_molecule(0).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r.positions[0], [1.0, 2.0, 3.0]);
        assert_eq!(r.atomic_numbers[0], 6);
        assert!(r.name.is_none());
        assert!(r.energy.is_none());
        assert!(r.forces.is_none());
        assert!(r.charges.is_none());
        assert!(r.velocities.is_none());
        assert!(r.cell.is_none());
        assert!(r.pbc.is_none());
        assert!(r.stress.is_none());
        assert!(r.atom_properties.is_empty());
        assert!(r.properties.is_empty());
    }

    #[test]
    fn test_soa_all_property_value_types() {
        use crate::atom::PropertyValue;

        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut mol = molecule_from_atoms(vec![Atom::new(0.0, 0.0, 0.0, 1)]);

        // Test every PropertyValue variant in atom_properties
        mol.atom_properties
            .insert("f64arr".to_string(), PropertyValue::FloatArray(vec![1.5]));
        mol.atom_properties.insert(
            "vec3f32".to_string(),
            PropertyValue::Vec3Array(vec![[1.0, 2.0, 3.0]]),
        );
        mol.atom_properties
            .insert("i64arr".to_string(), PropertyValue::IntArray(vec![42]));
        mol.atom_properties.insert(
            "f32arr".to_string(),
            PropertyValue::Float32Array(vec![3.125]),
        );
        mol.atom_properties.insert(
            "vec3f64".to_string(),
            PropertyValue::Vec3ArrayF64(vec![[1.0, 2.0, 3.0]]),
        );
        mol.atom_properties
            .insert("i32arr".to_string(), PropertyValue::Int32Array(vec![-7]));

        // Test every PropertyValue variant in properties
        mol.properties
            .insert("scalar_f".to_string(), PropertyValue::Float(99.9));
        mol.properties
            .insert("scalar_i".to_string(), PropertyValue::Int(-100));
        mol.properties.insert(
            "str_val".to_string(),
            PropertyValue::String("hello".to_string()),
        );
        mol.properties.insert(
            "f64arr".to_string(),
            PropertyValue::FloatArray(vec![1.0, 2.0]),
        );
        mol.properties.insert(
            "vec3f32".to_string(),
            PropertyValue::Vec3Array(vec![[7.0, 8.0, 9.0]]),
        );
        mol.properties
            .insert("i64arr".to_string(), PropertyValue::IntArray(vec![10, 20]));
        mol.properties.insert(
            "f32arr".to_string(),
            PropertyValue::Float32Array(vec![0.5, 1.5]),
        );
        mol.properties.insert(
            "vec3f64".to_string(),
            PropertyValue::Vec3ArrayF64(vec![[4.0, 5.0, 6.0]]),
        );
        mol.properties.insert(
            "i32arr".to_string(),
            PropertyValue::Int32Array(vec![100, -200]),
        );

        {
            let mut db = AtomDatabase::create(&path, CompressionType::Lz4).unwrap();
            db.add_molecule(&mol).unwrap();
            db.flush().unwrap();
        }

        let mut db = AtomDatabase::open(&path).unwrap();
        let r = db.get_molecule(0).unwrap();

        // Verify atom_properties
        assert_eq!(r.atom_properties.len(), 6);
        match r.atom_properties.get("f64arr").unwrap() {
            PropertyValue::FloatArray(v) => assert_eq!(v, &[1.5]),
            other => panic!("expected FloatArray, got {:?}", other),
        }
        match r.atom_properties.get("vec3f32").unwrap() {
            PropertyValue::Vec3Array(v) => assert_eq!(v, &[[1.0, 2.0, 3.0]]),
            other => panic!("expected Vec3Array, got {:?}", other),
        }
        match r.atom_properties.get("i64arr").unwrap() {
            PropertyValue::IntArray(v) => assert_eq!(v, &[42]),
            other => panic!("expected IntArray, got {:?}", other),
        }
        match r.atom_properties.get("f32arr").unwrap() {
            PropertyValue::Float32Array(v) => assert_eq!(v, &[3.125f32]),
            other => panic!("expected Float32Array, got {:?}", other),
        }
        match r.atom_properties.get("vec3f64").unwrap() {
            PropertyValue::Vec3ArrayF64(v) => assert_eq!(v, &[[1.0, 2.0, 3.0]]),
            other => panic!("expected Vec3ArrayF64, got {:?}", other),
        }
        match r.atom_properties.get("i32arr").unwrap() {
            PropertyValue::Int32Array(v) => assert_eq!(v, &[-7]),
            other => panic!("expected Int32Array, got {:?}", other),
        }

        // Verify properties
        assert_eq!(r.properties.len(), 9);
        match r.properties.get("scalar_f").unwrap() {
            PropertyValue::Float(v) => assert_eq!(*v, 99.9),
            other => panic!("expected Float, got {:?}", other),
        }
        match r.properties.get("scalar_i").unwrap() {
            PropertyValue::Int(v) => assert_eq!(*v, -100),
            other => panic!("expected Int, got {:?}", other),
        }
        match r.properties.get("str_val").unwrap() {
            PropertyValue::String(v) => assert_eq!(v, "hello"),
            other => panic!("expected String, got {:?}", other),
        }
    }
}
