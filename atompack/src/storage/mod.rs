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
//! │ Header slot B  (4096 bytes)          │     (generation counter)
//! ├──────────────────────────────────────┤
//! │ Frame: schema                        │  ← data region: framed entries only
//! │ Frame: record 0 .. record N-1        │     (see `frames`)
//! │ Frame: groups chunk, ...             │
//! ├──────────────────────────────────────┤  ← data_end
//! │ Metadata directory                   │  ← cache written by flush, replaced
//! │ Index  [count:u64][entries...]       │     by the next write session
//! └──────────────────────────────────────┘
//! ```
//!
//! Writing after a flush first marks both header slots as "writing", then
//! overwrites the cache with new frames; the next flush writes a new cache
//! and commits the header. Files are always compact, and a crash at any point
//! is recovered on open by scanning the frames. Files without the framed flag
//! (written by older versions) keep appending after their last commit.

use crate::compression::{CompressionType, compress, decompress};
use crate::{Error, Molecule, Result};
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod dtypes;
mod extensions;
mod frames;
mod header;
mod index;
mod schema;
mod soa;

use self::dtypes::arr;
use self::extensions::Extensions;
pub use self::extensions::{GroupColumn, Grouping};
use self::frames::{
    FRAME_GROUPS, FRAME_HEADER_SIZE, FRAME_RECORD, FRAME_SCHEMA, MetaFrame, encode_schema_payload,
    frame_header, scan_frames,
};
use self::header::{Header, encode_header_slot, read_best_header};
use self::index::{IndexStorage, MoleculeIndex, decode_index, encode_index};
use self::schema::{
    SchemaEntry, SchemaLock, decode_schema_lock, encode_schema_lock, merge_schema_lock,
    record_schema, schema_from_molecule, validate_schema_lock_for_record_format,
};
use self::soa::{
    deserialize_molecule_soa, minimum_record_format_for_molecule, serialize_molecule_soa,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"ATPK";
/// Bump only for incompatible layout changes (not crate version).
const FILE_FORMAT_VERSION: u32 = 2;
const RECORD_FORMAT_SOA_V2: u32 = 2;
const RECORD_FORMAT_SOA_V3: u32 = 3;
const RECORD_FORMAT_SOA: u32 = RECORD_FORMAT_SOA_V2;

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
const TYPE_FLOAT32: u8 = 11; // f32 scalar
const TYPE_MAT3X3_F32: u8 = 12; // [[f32; 3]; 3]
const TYPE_NONE: u8 = 13; // explicit null property
const TYPE_TENSOR_F32: u8 = 14; // [ndim:u8][dims:u32...][f32...]
const TYPE_TENSOR_F64: u8 = 15; // [ndim:u8][dims:u32...][f64...]
const TYPE_TENSOR_I32: u8 = 16; // [ndim:u8][dims:u32...][i32...]
const TYPE_TENSOR_I64: u8 = 17; // [ndim:u8][dims:u32...][i64...]

// Header flags.
/// Every entry after the header region is framed (see `frames`).
const HEADER_FRAMED: u32 = 1;
/// A write session is in progress: the metadata cache is being replaced.
const HEADER_WRITING: u32 = 2;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatabaseSchemaSection {
    pub kind: u8,
    pub key: String,
    pub type_tag: u8,
    pub per_atom: bool,
    pub elem_bytes: usize,
    pub slot_bytes: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DatabaseSchema {
    pub positions_type: Option<u8>,
    pub sections: Vec<DatabaseSchemaSection>,
}

fn database_schema_from_lock(lock: &SchemaLock) -> DatabaseSchema {
    let sections = lock
        .sections
        .iter()
        .map(|((kind, key), entry)| DatabaseSchemaSection {
            kind: *kind,
            key: key.clone(),
            type_tag: entry.type_tag,
            per_atom: entry.per_atom,
            elem_bytes: entry.elem_bytes,
            slot_bytes: entry.slot_bytes,
        })
        .collect();
    DatabaseSchema {
        positions_type: lock.positions_type,
        sections,
    }
}

fn schema_lock_from_database_schema(schema: DatabaseSchema) -> SchemaLock {
    let sections = schema
        .sections
        .into_iter()
        .map(|section| {
            (
                (section.kind, section.key),
                SchemaEntry {
                    type_tag: section.type_tag,
                    per_atom: section.per_atom,
                    elem_bytes: section.elem_bytes,
                    slot_bytes: section.slot_bytes,
                },
            )
        })
        .collect();
    SchemaLock {
        positions_type: schema.positions_type,
        sections,
    }
}

pub struct AtomDatabase {
    path: PathBuf,
    compression: CompressionType,
    generation: u64,
    record_format: u32,
    /// Files created by this version frame every entry; older files keep
    /// appending after their last commit.
    framed: bool,
    /// Where the next frame is written: the end of the frames, or the end of
    /// the last commit for unframed files.
    write_end: u64,
    /// Frames were written since the last commit.
    writing: bool,
    index: IndexStorage,
    schema_lock: Option<SchemaLock>,
    /// (offset, len) of the latest schema blob, inside its frame.
    schema_span: (u64, u64),
    file: Option<File>,
    data_mmap: Option<Arc<Mmap>>,
    /// Metadata frames (groups, ...).
    extensions: Extensions,
}

enum AppendSchema<'a> {
    Infer(Vec<(&'a [u8], Option<u8>)>),
    Locked(SchemaLock),
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
        let db = Self {
            path: path.as_ref().to_path_buf(),
            compression,
            generation: 0,
            record_format: RECORD_FORMAT_SOA,
            framed: true,
            write_end: HEADER_REGION_SIZE,
            writing: false,
            index: IndexStorage::InMemory(Vec::new()),
            schema_lock: None,
            schema_span: (0, 0),
            file: None,
            data_mmap: None,
            extensions: Extensions::default(),
        };
        let mut file = File::create(&db.path)?;
        let slot = encode_header_slot(db.header(HEADER_FRAMED, (0, 0), (0, 0)));
        file.write_all(&slot)?;
        file.write_all(&slot)?;
        file.sync_all()?;
        Ok(db)
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
        if header.record_format != RECORD_FORMAT_SOA_V2
            && header.record_format != RECORD_FORMAT_SOA_V3
        {
            return Err(Error::InvalidData(format!(
                "Unsupported record format {}.",
                header.record_format
            )));
        }

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

        if header.flags & HEADER_WRITING != 0 {
            return Self::recover(path, file, header, data_mmap);
        }

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

        let schema_lock = if header.schema_offset > 0 && header.schema_len > 0 {
            file.seek(SeekFrom::Start(header.schema_offset))?;
            let mut schema_bytes = vec![0u8; header.schema_len as usize];
            file.read_exact(&mut schema_bytes)?;
            Some(decode_schema_lock(&schema_bytes)?)
        } else {
            None
        };

        let extensions = Extensions::open(
            &mut file,
            (header.extensions_offset, header.extensions_len),
            header.data_start,
        )?;
        let framed = header.flags & HEADER_FRAMED != 0;
        let write_end = if framed {
            header.data_end
        } else if header.index_len > 0 {
            header.index_offset + header.index_len
        } else {
            header.data_start
        };

        Ok(Self {
            path,
            compression: header.compression,
            generation: header.generation,
            record_format: header.record_format,
            framed,
            write_end,
            writing: false,
            index,
            schema_lock,
            schema_span: (header.schema_offset, header.schema_len),
            file: Some(file),
            data_mmap,
            extensions,
        })
    }

    /// Open a file whose write session was interrupted: rebuild the index,
    /// schema and metadata directory from its frames. The session stays open,
    /// so the next flush commits everything recovered.
    fn recover(
        path: PathBuf,
        file: File,
        header: Header,
        data_mmap: Option<Arc<Mmap>>,
    ) -> Result<Self> {
        let scan = scan_frames(&path, header.data_start)?;
        let (record_format, schema_lock, schema_span) = match scan.schema {
            Some(frame) => {
                let payload = read_section(data_mmap.as_ref(), &path, frame.offset, frame.len)?;
                let (format, blob) = payload
                    .split_first_chunk::<4>()
                    .ok_or_else(|| Error::InvalidData("Schema frame too small".into()))?;
                (
                    u32::from_le_bytes(*format),
                    Some(decode_schema_lock(blob)?),
                    (frame.offset + 4, frame.len - 4),
                )
            }
            None => (header.record_format, None, (0, 0)),
        };
        Ok(Self {
            path,
            compression: header.compression,
            generation: header.generation,
            record_format,
            framed: true,
            write_end: scan.end,
            writing: true,
            index: IndexStorage::InMemory(scan.index),
            schema_lock,
            schema_span,
            file: Some(file),
            data_mmap,
            extensions: Extensions::from_frames(scan.metadata),
        })
    }

    /// Header describing the current state, with the given flags and the
    /// (offset, len) of the metadata directory and index.
    fn header(&self, flags: u32, directory: (u64, u64), index: (u64, u64)) -> Header {
        Header {
            generation: self.generation,
            data_start: HEADER_REGION_SIZE,
            num_molecules: self.index.len() as u64,
            compression: self.compression,
            record_format: self.record_format,
            schema_offset: self.schema_span.0,
            schema_len: self.schema_span.1,
            index_offset: index.0,
            index_len: index.1,
            extensions_offset: directory.0,
            extensions_len: directory.1,
            flags,
            data_end: self.write_end,
        }
    }

    /// Write `header` into the slot for its generation.
    fn write_header(file: &mut File, header: Header) -> Result<()> {
        let slot_offset = if header.generation.is_multiple_of(2) {
            HEADER_SLOT_A_OFFSET
        } else {
            HEADER_SLOT_B_OFFSET
        };
        file.seek(SeekFrom::Start(slot_offset))?;
        file.write_all(&encode_header_slot(header))?;
        Ok(())
    }

    /// Start a write session. For framed files, both header slots are marked
    /// as writing before anything after `write_end` can be overwritten: a
    /// crash is then recovered from the frames, and older versions, which
    /// cannot recover, refuse the file instead of reading stale metadata.
    fn begin_write(&mut self) -> Result<()> {
        self.ensure_writable()?;
        if self.writing {
            return Ok(());
        }
        if self.framed {
            let mut file = OpenOptions::new().write(true).open(&self.path)?;
            for _ in 0..2 {
                self.generation += 1;
                let mut header = self.header(HEADER_FRAMED | HEADER_WRITING, (0, 0), (0, 0));
                // Invalid for older versions: they require an index when count > 0.
                header.num_molecules = u64::MAX;
                Self::write_header(&mut file, header)?;
            }
            file.sync_data()?;
        }
        self.writing = true;
        Ok(())
    }

    /// Write frames at `write_end`. Returns each frame's payload offset.
    fn write_frames<'a>(
        &mut self,
        frames: impl IntoIterator<Item = ([u8; FRAME_HEADER_SIZE], &'a [u8])>,
    ) -> Result<Vec<u64>> {
        self.begin_write()?;
        let file = OpenOptions::new().write(true).open(&self.path)?;
        let mut out = BufWriter::with_capacity(1 << 22, file);
        out.seek(SeekFrom::Start(self.write_end))?;
        let mut end = self.write_end;
        let mut offsets = Vec::new();
        for (header, payload) in frames {
            out.write_all(&header)?;
            out.write_all(payload)?;
            offsets.push(end + FRAME_HEADER_SIZE as u64);
            end += (FRAME_HEADER_SIZE + payload.len()) as u64;
        }
        out.flush()?;
        self.write_end = end;
        Ok(offsets)
    }

    fn write_metadata_frame(&mut self, kind: u32, payload: &[u8]) -> Result<MetaFrame> {
        let offset = self.write_frames([(frame_header(kind, payload, 0, 0), payload)])?[0];
        Ok(MetaFrame {
            kind,
            offset,
            len: payload.len() as u64,
        })
    }

    fn rebuild_schema_lock(&self) -> Result<SchemaLock> {
        let mut lock = SchemaLock::default();
        let compression = self.compression;
        let positions_type_hint = self.positions_type();

        if let Some(ref mmap) = self.data_mmap {
            for index in 0..self.index.len() {
                let entry = self
                    .index
                    .get(index)
                    .ok_or_else(|| Error::InvalidData(format!("Index {} out of bounds", index)))?;
                let start = entry.offset as usize;
                let end = start + entry.compressed_size as usize;
                let bytes = decompress(
                    &mmap[start..end],
                    compression,
                    Some(entry.uncompressed_size as usize),
                )?;
                let record = record_schema(&bytes, self.record_format, positions_type_hint)?;
                merge_schema_lock(&mut lock, &record)?;
            }
            return Ok(lock);
        }

        let mut file = File::open(&self.path)?;
        for index in 0..self.index.len() {
            let entry = self
                .index
                .get(index)
                .ok_or_else(|| Error::InvalidData(format!("Index {} out of bounds", index)))?;
            file.seek(SeekFrom::Start(entry.offset))?;
            let mut compressed = vec![0u8; entry.compressed_size as usize];
            file.read_exact(&mut compressed)?;
            let bytes = decompress(
                &compressed,
                compression,
                Some(entry.uncompressed_size as usize),
            )?;
            let record = record_schema(&bytes, self.record_format, positions_type_hint)?;
            merge_schema_lock(&mut lock, &record)?;
        }
        Ok(lock)
    }

    fn can_promote_record_format(&self) -> bool {
        self.record_format == RECORD_FORMAT_SOA_V2
            && self.index.is_empty()
            && self.schema_lock.is_none()
    }

    fn resolved_record_format_for_schema(&self, schema: &SchemaLock) -> Result<u32> {
        match validate_schema_lock_for_record_format(self.record_format, schema) {
            Ok(()) => Ok(self.record_format),
            Err(_) if self.can_promote_record_format() => {
                validate_schema_lock_for_record_format(RECORD_FORMAT_SOA_V3, schema)?;
                Ok(RECORD_FORMAT_SOA_V3)
            }
            Err(current_err) => Err(current_err),
        }
    }

    fn infer_schema<'a, I>(&self, records: I) -> Result<(u32, SchemaLock)>
    where
        I: IntoIterator<Item = (&'a [u8], Option<u8>)>,
    {
        let mut lock = match &self.schema_lock {
            Some(lock) => lock.clone(),
            None if self.index.is_empty() => SchemaLock::default(),
            None => self.rebuild_schema_lock()?,
        };
        let mut record_format = self.record_format;
        let can_promote = self.can_promote_record_format();

        for (bytes, positions_type_hint) in records {
            let hint = positions_type_hint.or(lock.positions_type);
            let record = match record_schema(bytes, record_format, hint) {
                Ok(record) => record,
                Err(_) if can_promote && record_format == RECORD_FORMAT_SOA_V2 => {
                    let record = record_schema(bytes, RECORD_FORMAT_SOA_V3, hint)?;
                    record_format = RECORD_FORMAT_SOA_V3;
                    record
                }
                Err(current_err) => return Err(current_err),
            };
            merge_schema_lock(&mut lock, &record)?;
        }
        Ok((record_format, lock))
    }

    fn merge_schema(&self, incoming: &SchemaLock) -> Result<(u32, SchemaLock)> {
        let record_format = self.resolved_record_format_for_schema(incoming)?;
        let mut lock = match &self.schema_lock {
            Some(lock) => lock.clone(),
            None if self.index.is_empty() => SchemaLock::default(),
            None => self.rebuild_schema_lock()?,
        };
        merge_schema_lock(&mut lock, incoming)?;
        Ok((record_format, lock))
    }

    fn ensure_writable(&self) -> Result<()> {
        if self.data_mmap.is_some() {
            return Err(Error::InvalidData(
                "Cannot write to a database opened with mmap (read-only); reopen without mmap to write."
                    .into(),
            ));
        }
        Ok(())
    }

    /// Resolve the schema for the next records; a changed schema is written
    /// as a frame before those records so that recovery can decode them.
    fn prepare_append(&mut self, schema: AppendSchema<'_>) -> Result<()> {
        self.ensure_writable()?;
        let (record_format, lock) = match schema {
            AppendSchema::Infer(records) => self.infer_schema(records)?,
            AppendSchema::Locked(schema) => self.merge_schema(&schema)?,
        };
        if record_format != self.record_format || self.schema_lock.as_ref() != Some(&lock) {
            let payload = encode_schema_payload(record_format, &lock)?;
            let frame = self.write_metadata_frame(FRAME_SCHEMA, &payload)?;
            self.schema_span = (frame.offset + 4, frame.len - 4);
        }
        self.record_format = record_format;
        self.schema_lock = Some(lock);
        Ok(())
    }

    /// Compress records in parallel and append them as frames.
    fn write_records<T: AsRef<[u8]> + Sync>(&mut self, records: &[(T, u32)]) -> Result<()> {
        let compression = self.compression;
        let framed: Vec<([u8; FRAME_HEADER_SIZE], Vec<u8>)> = records
            .par_iter()
            .map(|(bytes, num_atoms)| {
                let bytes = bytes.as_ref();
                let payload = compress(bytes, compression)?;
                let header = frame_header(FRAME_RECORD, &payload, bytes.len() as u32, *num_atoms);
                Ok((header, payload))
            })
            .collect::<Result<_>>()?;

        let offsets = self.write_frames(
            framed
                .iter()
                .map(|(header, payload)| (*header, payload.as_slice())),
        )?;
        let entries = offsets
            .into_iter()
            .zip(records.iter().zip(&framed))
            .map(
                |(offset, ((bytes, num_atoms), (_, payload)))| MoleculeIndex {
                    offset,
                    compressed_size: payload.len() as u32,
                    uncompressed_size: bytes.as_ref().len() as u32,
                    num_atoms: *num_atoms,
                },
            )
            .collect();
        self.index.extend(entries)
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

        let target_format = if self.can_promote_record_format()
            && molecules.iter().any(|molecule| {
                minimum_record_format_for_molecule(molecule) == RECORD_FORMAT_SOA_V3
            }) {
            RECORD_FORMAT_SOA_V3
        } else {
            self.record_format
        };

        let serialized: Vec<(Vec<u8>, u32, SchemaLock)> = molecules
            .par_iter()
            .map(|mol| {
                let bytes = serialize_molecule_soa(mol, target_format)?;
                let num_atoms = mol.len() as u32;
                Ok((bytes, num_atoms, schema_from_molecule(mol)?))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut batch_schema = SchemaLock::default();
        let mut records = Vec::with_capacity(serialized.len());
        for (bytes, num_atoms, schema) in serialized {
            merge_schema_lock(&mut batch_schema, &schema)?;
            records.push((bytes, num_atoms));
        }

        self.append_owned_soa_records_prevalidated(records, batch_schema)
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
        self.append_soa_records(
            records
                .iter()
                .map(|(bytes, num_atoms)| (*bytes, *num_atoms, None)),
        )
    }

    #[doc(hidden)]
    pub fn add_raw_soa_records_with_positions_type(
        &mut self,
        records: &[(&[u8], u32, u8)],
    ) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        self.append_soa_records(
            records.iter().map(|(bytes, num_atoms, positions_type)| {
                (*bytes, *num_atoms, Some(*positions_type))
            }),
        )
    }

    #[doc(hidden)]
    pub fn add_raw_soa_records_with_schema(
        &mut self,
        records: &[(&[u8], u32)],
        schema: DatabaseSchema,
    ) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        self.append_raw_soa_records_prevalidated(records, schema_lock_from_database_schema(schema))
    }

    #[doc(hidden)]
    pub fn add_owned_soa_records(&mut self, records: Vec<(Vec<u8>, u32, u8)>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        self.append_owned_soa_records(records)
    }

    #[doc(hidden)]
    pub fn add_owned_soa_records_with_schema(
        &mut self,
        records: Vec<(Vec<u8>, u32)>,
        schema: DatabaseSchema,
    ) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        self.append_owned_soa_records_prevalidated(
            records,
            schema_lock_from_database_schema(schema),
        )
    }

    fn append_soa_records<'a, I>(&mut self, records: I) -> Result<()>
    where
        I: IntoIterator<Item = (&'a [u8], u32, Option<u8>)>,
    {
        let records: Vec<(&[u8], u32, Option<u8>)> = records.into_iter().collect();
        if records.is_empty() {
            return Ok(());
        }

        self.prepare_append(AppendSchema::Infer(
            records
                .iter()
                .map(|(bytes, _, positions_type_hint)| (*bytes, *positions_type_hint))
                .collect(),
        ))?;
        let borrowed = records
            .iter()
            .map(|(bytes, num_atoms, _)| (*bytes, *num_atoms))
            .collect::<Vec<_>>();
        self.write_records(&borrowed)
    }

    fn append_owned_soa_records(&mut self, records: Vec<(Vec<u8>, u32, u8)>) -> Result<()> {
        self.prepare_append(AppendSchema::Infer(
            records
                .iter()
                .map(|(bytes, _, positions_type)| (bytes.as_slice(), Some(*positions_type)))
                .collect(),
        ))?;

        let records: Vec<(Vec<u8>, u32)> = records
            .into_iter()
            .map(|(bytes, num_atoms, _positions_type)| (bytes, num_atoms))
            .collect();
        self.write_records(&records)
    }

    fn append_owned_soa_records_prevalidated(
        &mut self,
        records: Vec<(Vec<u8>, u32)>,
        batch_schema: SchemaLock,
    ) -> Result<()> {
        self.prepare_append(AppendSchema::Locked(batch_schema))?;
        self.write_records(&records)
    }

    fn append_raw_soa_records_prevalidated(
        &mut self,
        records: &[(&[u8], u32)],
        batch_schema: SchemaLock,
    ) -> Result<()> {
        self.prepare_append(AppendSchema::Locked(batch_schema))?;
        self.write_records(records)
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
        deserialize_molecule_soa(&decompressed, self.record_format, self.positions_type())
    }

    /// Read multiple molecules in parallel.
    pub fn get_molecules(&self, indices: &[usize]) -> Result<Vec<Molecule>> {
        let raw = self.get_raw_bytes(indices)?;
        raw.into_par_iter()
            .map(|bytes| {
                deserialize_molecule_soa(&bytes, self.record_format, self.positions_type())
            })
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

    /// Commit everything written since the last flush: write the metadata
    /// directory and the index after the last frame, drop anything beyond
    /// them, then publish them in the header. Does nothing when there is
    /// nothing new.
    pub fn flush(&mut self) -> Result<()> {
        self.ensure_writable()?;
        if !self.writing {
            return Ok(());
        }
        let index_bytes = match &self.index {
            IndexStorage::InMemory(entries) => encode_index(entries),
            IndexStorage::MemoryMapped { .. } => {
                unreachable!("writable databases load their index")
            }
        };
        let directory = self.extensions.encode_directory().unwrap_or_default();

        let mut file = OpenOptions::new().write(true).open(&self.path)?;
        let data_end = self.write_end;
        file.seek(SeekFrom::Start(data_end))?;
        file.write_all(&directory)?;
        file.write_all(&index_bytes)?;
        let index_offset = data_end + directory.len() as u64;
        let end = index_offset + index_bytes.len() as u64;
        file.set_len(end)?;
        // The metadata must be on disk before the header points at it.
        file.sync_data()?;

        self.generation += 1;
        let directory_span = match directory.len() as u64 {
            0 => (0, 0),
            len => (data_end, len),
        };
        let flags = if self.framed { HEADER_FRAMED } else { 0 };
        let header = self.header(
            flags,
            directory_span,
            (index_offset, index_bytes.len() as u64),
        );
        Self::write_header(&mut file, header)?;
        file.sync_all()?;

        self.writing = false;
        if !self.framed {
            self.write_end = end;
        }
        Ok(())
    }

    // -- Groups --------------------------------------------------------------

    /// All groupings by name. Decoded from the file on first access.
    pub fn groups(&self) -> Result<&BTreeMap<String, Grouping>> {
        let (mmap, path) = (self.data_mmap.as_ref(), &self.path);
        self.extensions.groups(self.len(), |offset, len| {
            read_section(mmap, path, offset, len)
        })
    }

    /// Append groups to the grouping `name`, creating it if needed. Members
    /// must reference existing records. Written immediately as one frame and
    /// committed by the next `flush`.
    pub fn add_groups(&mut self, name: &str, groups: Grouping) -> Result<()> {
        self.ensure_writable()?;
        let (mmap, path) = (self.data_mmap.as_ref(), &self.path);
        let chunk =
            self.extensions
                .encode_groups_chunk(name, &groups, self.len(), |offset, len| {
                    read_section(mmap, path, offset, len)
                })?;
        let frame = self.write_metadata_frame(FRAME_GROUPS, &chunk)?;
        self.extensions.push_groups(name, groups, frame);
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

    pub fn record_format(&self) -> u32 {
        self.record_format
    }

    pub fn positions_type(&self) -> Option<u8> {
        self.schema_lock
            .as_ref()
            .and_then(|lock| lock.positions_type)
    }

    pub fn schema_info(&self) -> Option<DatabaseSchema> {
        self.schema_lock.as_ref().map(database_schema_from_lock)
    }

    #[doc(hidden)]
    pub fn record_format_for_schema(&self, schema: DatabaseSchema) -> Result<u32> {
        self.resolved_record_format_for_schema(&schema_lock_from_database_schema(schema))
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

/// Read a committed section, from the mmap when the database has one.
fn read_section(mmap: Option<&Arc<Mmap>>, path: &Path, offset: u64, len: u64) -> Result<Vec<u8>> {
    if let Some(mmap) = mmap {
        let range = offset as usize..(offset + len) as usize;
        return mmap
            .get(range)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| Error::InvalidData("Section out of bounds".into()));
    }
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0u8; len as usize];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// 6. Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Atom, FloatArrayData, FloatScalarData, Mat3Data, Vec3Data};
    use tempfile::NamedTempFile;

    fn adler32_for_test(bytes: &[u8]) -> u32 {
        const MOD_ADLER: u32 = 65_521;
        let mut a: u32 = 1;
        let mut b: u32 = 0;
        for &byte in bytes {
            a = (a + (byte as u32)) % MOD_ADLER;
            b = (b + a) % MOD_ADLER;
        }
        (b << 16) | a
    }

    fn encode_legacy_v2_header_slot(header: Header) -> [u8; HEADER_SLOT_SIZE] {
        let mut slot = [0u8; HEADER_SLOT_SIZE];
        slot[0..4].copy_from_slice(MAGIC);
        slot[4..8].copy_from_slice(&FILE_FORMAT_VERSION.to_le_bytes());
        slot[8..16].copy_from_slice(&header.generation.to_le_bytes());
        slot[16..24].copy_from_slice(&header.data_start.to_le_bytes());
        slot[24..32].copy_from_slice(&header.index_offset.to_le_bytes());
        slot[32..40].copy_from_slice(&header.index_len.to_le_bytes());
        slot[40..48].copy_from_slice(&header.num_molecules.to_le_bytes());

        let (compression_type, compression_level) = match header.compression {
            CompressionType::None => (0u8, 0i32),
            CompressionType::Lz4 => (1u8, 0i32),
            CompressionType::Zstd(level) => (2u8, level),
        };
        slot[48] = compression_type;
        slot[52..56].copy_from_slice(&compression_level.to_le_bytes());
        slot[56..60].copy_from_slice(&header.record_format.to_le_bytes());

        let checksum = adler32_for_test(&slot[..HEADER_SLOT_SIZE - 4]);
        slot[HEADER_SLOT_SIZE - 4..HEADER_SLOT_SIZE].copy_from_slice(&checksum.to_le_bytes());
        slot
    }

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
            assert_eq!(mol.atom(0).unwrap().position(), [1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn test_database_open_legacy_v2_header_layout() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let header = Header {
            generation: 0,
            data_start: HEADER_REGION_SIZE,
            num_molecules: 0,
            compression: CompressionType::None,
            record_format: RECORD_FORMAT_SOA_V2,
            schema_offset: 0,
            schema_len: 0,
            index_offset: 0,
            index_len: 0,
            extensions_offset: 0,
            extensions_len: 0,
            flags: 0,
            data_end: 0,
        };
        let slot = encode_legacy_v2_header_slot(header);
        let mut file = File::create(&path).unwrap();
        file.write_all(&slot).unwrap();
        file.write_all(&slot).unwrap();
        file.flush().unwrap();
        file.sync_all().unwrap();
        drop(file);

        let db = AtomDatabase::open(&path).unwrap();
        assert_eq!(db.len(), 0);
        assert_eq!(db.record_format(), RECORD_FORMAT_SOA_V2);
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
        mol.forces = Some(Vec3Data::F32(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]));
        mol.energy = Some(FloatScalarData::F64(-123.456));

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
            assert_eq!(
                forces,
                Vec3Data::F32(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            );

            // Check energy is preserved
            assert!(retrieved.energy.is_some());
            assert_eq!(retrieved.energy.unwrap(), FloatScalarData::F64(-123.456));
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

        // Corrupt the latest header slot. The other slot was marked as writing
        // when the second session started, so open recovers from the frames.
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
        assert_eq!(db.len(), 2);
        assert_eq!(db.get_molecule(0).unwrap().atomic_numbers[0], 6);
        assert_eq!(db.get_molecule(1).unwrap().atomic_numbers[0], 8);
    }

    #[test]
    fn test_garbage_after_commit_is_dropped_by_next_flush() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();
        let mol1 = molecule_from_atoms(vec![Atom::new(0.0, 0.0, 0.0, 6)]);
        let mol2 = molecule_from_atoms(vec![Atom::new(1.0, 2.0, 3.0, 8)]);

        let reference = NamedTempFile::new().unwrap();
        let mut db = AtomDatabase::create(reference.path(), CompressionType::None).unwrap();
        db.add_molecules(&[&mol1, &mol2]).unwrap();
        db.flush().unwrap();

        let mut db = AtomDatabase::create(&path, CompressionType::None).unwrap();
        db.add_molecule(&mol1).unwrap();
        db.flush().unwrap();
        // Simulate an interrupted write from another process.
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[0u8; 123]).unwrap();

        let mut db = AtomDatabase::open(&path).unwrap();
        db.add_molecule(&mol2).unwrap();
        db.flush().unwrap();

        let size = |p: &Path| std::fs::metadata(p).unwrap().len();
        assert_eq!(size(&path), size(reference.path()));
        let mut db = AtomDatabase::open(&path).unwrap();
        assert_eq!(db.len(), 2);
        assert_eq!(db.get_molecule(1).unwrap().atomic_numbers[0], 8);
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
        mol.energy = Some(FloatScalarData::F64(-42.5));
        mol.forces = Some(Vec3Data::F32(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]));
        mol.charges = Some(FloatArrayData::F64(vec![-0.5, 0.5]));
        mol.velocities = Some(Vec3Data::F32(vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        mol.cell = Some(Mat3Data::F64([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]));
        mol.pbc = Some([true, true, false]);
        mol.stress = Some(Mat3Data::F64([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]));

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
        assert_eq!(r.atom(0).unwrap().position(), [1.0, 2.0, 3.0]);
        assert_eq!(r.atomic_numbers[0], 6);
        assert_eq!(r.atom(1).unwrap().position(), [4.0, 5.0, 6.0]);
        assert_eq!(r.atomic_numbers[1], 8);
        assert_eq!(r.energy, Some(FloatScalarData::F64(-42.5)));
        assert_eq!(
            r.forces.as_ref().unwrap(),
            &Vec3Data::F32(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        );
        assert_eq!(
            r.charges.as_ref().unwrap(),
            &FloatArrayData::F64(vec![-0.5, 0.5])
        );
        assert_eq!(
            r.velocities.as_ref().unwrap(),
            &Vec3Data::F32(vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        );
        assert_eq!(
            r.cell.unwrap(),
            Mat3Data::F64([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        );
        assert_eq!(r.pbc, Some([true, true, false]));
        assert_eq!(
            r.stress.unwrap(),
            Mat3Data::F64([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
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
        assert_eq!(r.atom(0).unwrap().position(), [1.0, 2.0, 3.0]);
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
        use crate::atom::{PropertyValue, TensorData};

        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut mol = molecule_from_atoms(vec![Atom::new(0.0, 0.0, 0.0, 1)]);

        // Test every PropertyValue variant in atom_properties
        mol.atom_properties.insert(
            "atom_f64arr".to_string(),
            PropertyValue::FloatArray(vec![1.5]),
        );
        mol.atom_properties.insert(
            "atom_vec3f32".to_string(),
            PropertyValue::Vec3Array(vec![[1.0, 2.0, 3.0]]),
        );
        mol.atom_properties
            .insert("atom_i64arr".to_string(), PropertyValue::IntArray(vec![42]));
        mol.atom_properties.insert(
            "atom_f32arr".to_string(),
            PropertyValue::Float32Array(vec![3.125]),
        );
        mol.atom_properties.insert(
            "atom_vec3f64".to_string(),
            PropertyValue::Vec3ArrayF64(vec![[1.0, 2.0, 3.0]]),
        );
        mol.atom_properties.insert(
            "atom_i32arr".to_string(),
            PropertyValue::Int32Array(vec![-7]),
        );
        mol.atom_properties.insert(
            "atom_tensor".to_string(),
            PropertyValue::Tensor(TensorData::F32 {
                shape: vec![1, 2],
                values: vec![0.25, 0.75],
            }),
        );

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
        mol.properties
            .insert("none_val".to_string(), PropertyValue::None);
        mol.properties.insert(
            "tensor_val".to_string(),
            PropertyValue::Tensor(TensorData::I64 {
                shape: vec![2, 2],
                values: vec![1, 2, 3, 4],
            }),
        );

        {
            let mut db = AtomDatabase::create(&path, CompressionType::Lz4).unwrap();
            db.add_molecule(&mol).unwrap();
            db.flush().unwrap();
        }

        let mut db = AtomDatabase::open(&path).unwrap();
        let r = db.get_molecule(0).unwrap();

        // Verify atom_properties
        assert_eq!(r.atom_properties.len(), 7);
        match r.atom_properties.get("atom_f64arr").unwrap() {
            PropertyValue::FloatArray(v) => assert_eq!(v, &[1.5]),
            other => panic!("expected FloatArray, got {:?}", other),
        }
        match r.atom_properties.get("atom_vec3f32").unwrap() {
            PropertyValue::Vec3Array(v) => assert_eq!(v, &[[1.0, 2.0, 3.0]]),
            other => panic!("expected Vec3Array, got {:?}", other),
        }
        match r.atom_properties.get("atom_i64arr").unwrap() {
            PropertyValue::IntArray(v) => assert_eq!(v, &[42]),
            other => panic!("expected IntArray, got {:?}", other),
        }
        match r.atom_properties.get("atom_f32arr").unwrap() {
            PropertyValue::Float32Array(v) => assert_eq!(v, &[3.125f32]),
            other => panic!("expected Float32Array, got {:?}", other),
        }
        match r.atom_properties.get("atom_vec3f64").unwrap() {
            PropertyValue::Vec3ArrayF64(v) => assert_eq!(v, &[[1.0, 2.0, 3.0]]),
            other => panic!("expected Vec3ArrayF64, got {:?}", other),
        }
        match r.atom_properties.get("atom_i32arr").unwrap() {
            PropertyValue::Int32Array(v) => assert_eq!(v, &[-7]),
            other => panic!("expected Int32Array, got {:?}", other),
        }
        match r.atom_properties.get("atom_tensor").unwrap() {
            PropertyValue::Tensor(TensorData::F32 { shape, values }) => {
                assert_eq!(shape, &[1, 2]);
                assert_eq!(values, &[0.25, 0.75]);
            }
            other => panic!("expected TensorData::F32, got {:?}", other),
        }

        // Verify properties
        assert_eq!(r.properties.len(), 11);
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
        match r.properties.get("none_val").unwrap() {
            PropertyValue::None => {}
            other => panic!("expected None, got {:?}", other),
        }
        match r.properties.get("tensor_val").unwrap() {
            PropertyValue::Tensor(TensorData::I64 { shape, values }) => {
                assert_eq!(shape, &[2, 2]);
                assert_eq!(values, &[1, 2, 3, 4]);
            }
            other => panic!("expected TensorData::I64, got {:?}", other),
        }
    }

    #[test]
    fn test_schema_lock_rejects_position_dtype_mismatch() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();
        let mut db = AtomDatabase::create(&path, CompressionType::None).unwrap();

        let mol_f64 = Molecule::new_f64(vec![[0.0, 0.0, 0.0]], vec![6]).unwrap();
        let mol_f32 = Molecule::new(vec![[1.0, 1.0, 1.0]], vec![8]).unwrap();

        db.add_molecule(&mol_f64).unwrap();
        let err = db.add_molecule(&mol_f32).unwrap_err();
        assert!(format!("{}", err).contains("Position dtype mismatch"));
    }

    #[test]
    fn test_empty_database_stays_v2_for_v2_compatible_first_write() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();
        let mut db = AtomDatabase::create(&path, CompressionType::None).unwrap();

        let mut mol = Molecule::new(vec![[0.0, 0.0, 0.0]], vec![6]).unwrap();
        mol.energy = Some(FloatScalarData::F64(-1.0));
        mol.forces = Some(Vec3Data::F32(vec![[0.1, 0.2, 0.3]]));

        assert_eq!(db.record_format(), RECORD_FORMAT_SOA_V2);
        db.add_molecule(&mol).unwrap();
        assert_eq!(db.record_format(), RECORD_FORMAT_SOA_V2);
    }

    #[test]
    fn test_empty_database_promotes_to_v3_when_first_write_requires_it() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();
        let mut db = AtomDatabase::create(&path, CompressionType::None).unwrap();

        let mut mol = Molecule::new_f64(vec![[0.0, 0.0, 0.0]], vec![6]).unwrap();
        mol.forces = Some(Vec3Data::F64(vec![[0.1, 0.2, 0.3]]));

        assert_eq!(db.record_format(), RECORD_FORMAT_SOA_V2);
        db.add_molecule(&mol).unwrap();
        assert_eq!(db.record_format(), RECORD_FORMAT_SOA_V3);
    }

    #[test]
    fn test_schema_lock_rejects_custom_shape_mismatch() {
        use crate::atom::PropertyValue;

        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();
        let mut db = AtomDatabase::create(&path, CompressionType::None).unwrap();

        let mut mol1 = Molecule::new(vec![[0.0, 0.0, 0.0]], vec![6]).unwrap();
        mol1.set_property(
            "spectrum".to_string(),
            PropertyValue::FloatArray(vec![1.0, 2.0]),
        );

        let mut mol2 = Molecule::new(vec![[1.0, 1.0, 1.0]], vec![8]).unwrap();
        mol2.set_property("spectrum".to_string(), PropertyValue::FloatArray(vec![3.0]));

        db.add_molecule(&mol1).unwrap();
        let err = db.add_molecule(&mol2).unwrap_err();
        assert!(format!("{}", err).contains("Schema mismatch for section 'spectrum'"));
    }

    #[test]
    fn test_add_owned_soa_records_rejects_v2_incompatible_builtin_dtype() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();
        let mut db = AtomDatabase::create(&path, CompressionType::None).unwrap();

        let legacy = Molecule::new(vec![[0.0, 0.0, 0.0]], vec![6]).unwrap();
        db.add_molecule(&legacy).unwrap();
        assert_eq!(db.record_format(), RECORD_FORMAT_SOA_V2);

        let mut mol = molecule_from_atoms(vec![Atom::new(0.0, 0.0, 0.0, 6)]);
        mol.cell = Some(Mat3Data::F32([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]));

        let bytes = serialize_molecule_soa(&mol, RECORD_FORMAT_SOA_V3).unwrap();
        let err = db
            .add_owned_soa_records(vec![(bytes, mol.len() as u32, TYPE_VEC3_F32)])
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("record format 2 does not support float32 cell")
        );
    }

    #[test]
    fn test_schema_lock_allows_late_optional_builtin() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();
        let mut db = AtomDatabase::create(&path, CompressionType::None).unwrap();

        let mol1 = Molecule::new_f64(vec![[0.0, 0.0, 0.0]], vec![6]).unwrap();
        let mut mol2 = Molecule::new_f64(vec![[1.0, 1.0, 1.0]], vec![8]).unwrap();
        mol2.forces = Some(Vec3Data::F64(vec![[0.1, 0.2, 0.3]]));

        db.add_molecule(&mol1).unwrap();
        db.add_molecule(&mol2).unwrap();
        db.flush().unwrap();

        let retrieved = db.get_molecule(1).unwrap();
        assert_eq!(retrieved.forces, Some(Vec3Data::F64(vec![[0.1, 0.2, 0.3]])));
    }
}
