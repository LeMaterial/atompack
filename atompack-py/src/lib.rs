// Copyright 2026 Entalpic
//! Python bindings for atompack
//!
//! This module exposes the atompack library to Python using PyO3.

#![allow(clippy::useless_conversion)]
// PyO3 0.22 macro-generated code triggers this lint; safe to suppress until PyO3 upgrade.
#![allow(unsafe_op_in_unsafe_fn)]

use atompack::{
    Atom, AtomDatabase, Molecule, SharedMmapBytes, atom::PropertyValue,
    compression::CompressionType,
};
use numpy::{Element, PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::exceptions::{PyFileExistsError, PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use pyo3::{IntoPyObject, IntoPyObjectExt};
use std::borrow::Cow;
use std::path::PathBuf;

/// Wrapper to send raw pointers across rayon threads.
/// SAFETY: Caller must ensure non-overlapping writes.
#[derive(Clone, Copy)]
struct RawBuf<T> {
    ptr: *mut T,
}
unsafe impl<T> Send for RawBuf<T> {}
unsafe impl<T> Sync for RawBuf<T> {}

impl<T> RawBuf<T> {
    fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
        }
    }

    #[inline(always)]
    unsafe fn at(&self, offset: usize) -> *mut T {
        unsafe { self.ptr.add(offset) }
    }
}

fn invalid_data(message: impl Into<String>) -> atompack::Error {
    atompack::Error::InvalidData(message.into())
}

fn slice_to_array<const N: usize>(bytes: &[u8], label: &str) -> atompack::Result<[u8; N]> {
    bytes
        .try_into()
        .map_err(|_| invalid_data(format!("{} truncated", label)))
}

fn py_slice_to_array<const N: usize>(bytes: &[u8], label: &str) -> PyResult<[u8; N]> {
    bytes
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{} truncated", label)))
}

fn read_bytes_at<'a>(
    data: &'a [u8],
    pos: &mut usize,
    len: usize,
    label: &str,
) -> atompack::Result<&'a [u8]> {
    let start = *pos;
    let end = start
        .checked_add(len)
        .ok_or_else(|| invalid_data(format!("{} offset overflow", label)))?;
    let bytes = data
        .get(start..end)
        .ok_or_else(|| invalid_data(format!("{} truncated", label)))?;
    *pos = end;
    Ok(bytes)
}

fn read_u8_at(data: &[u8], pos: &mut usize, label: &str) -> atompack::Result<u8> {
    Ok(*read_bytes_at(data, pos, 1, label)?
        .first()
        .ok_or_else(|| invalid_data(format!("{} truncated", label)))?)
}

fn read_u16_le_at(data: &[u8], pos: &mut usize, label: &str) -> atompack::Result<u16> {
    let bytes: [u8; 2] = read_bytes_at(data, pos, 2, label)?
        .try_into()
        .map_err(|_| invalid_data(format!("{} truncated", label)))?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32_le_at(data: &[u8], pos: &mut usize, label: &str) -> atompack::Result<u32> {
    let bytes: [u8; 4] = read_bytes_at(data, pos, 4, label)?
        .try_into()
        .map_err(|_| invalid_data(format!("{} truncated", label)))?;
    Ok(u32::from_le_bytes(bytes))
}

// Section kind tags (must match atompack/src/storage/mod.rs)
const KIND_BUILTIN: u8 = 0;
const KIND_ATOM_PROP: u8 = 1;
const KIND_MOL_PROP: u8 = 2;

// Type tags (must match atompack/src/storage/mod.rs)
const TYPE_FLOAT: u8 = 0;
const TYPE_INT: u8 = 1;
const TYPE_STRING: u8 = 2;
const TYPE_F64_ARRAY: u8 = 3;
const TYPE_VEC3_F32: u8 = 4;
const TYPE_I64_ARRAY: u8 = 5;
const TYPE_F32_ARRAY: u8 = 6;
const TYPE_VEC3_F64: u8 = 7;
const TYPE_I32_ARRAY: u8 = 8;
const TYPE_BOOL3: u8 = 9;
const TYPE_MAT3X3_F64: u8 = 10;

/// A single parsed section reference (zero-copy into decompressed bytes).
#[derive(Clone)]
struct SectionRef<'a> {
    kind: u8,
    key: &'a str,
    type_tag: u8,
    payload: &'a [u8],
}

/// Per-molecule extracted data (references into decompressed bytes).
struct MolData<'a> {
    n_atoms: usize,
    positions_bytes: &'a [u8],      // n_atoms * 12
    atomic_numbers_bytes: &'a [u8], // n_atoms
    sections: Vec<SectionRef<'a>>,
}

#[derive(Clone)]
struct SectionSchema {
    kind: u8,
    key: String,
    type_tag: u8,
    per_atom: bool,
    elem_bytes: usize,
    slot_bytes: usize,
}

/// Parse SOA format bytes into MolData without allocation.
///
/// Layout:
///   [n_atoms:u32][positions:n*12][atomic_numbers:n]
///   [n_sections:u16]
///   per section: [kind:u8][key_len:u8][key][type_tag:u8][payload_len:u32][payload]
fn parse_mol_fast_soa(bytes: &[u8]) -> atompack::Result<MolData<'_>> {
    let mut pos = 0usize;
    let n_atoms = read_u32_le_at(bytes, &mut pos, "SOA n_atoms")? as usize;
    let positions_len = n_atoms
        .checked_mul(12)
        .ok_or_else(|| invalid_data("SOA positions byte length overflow"))?;
    let positions_bytes = read_bytes_at(bytes, &mut pos, positions_len, "SOA positions")?;
    let atomic_numbers_bytes = read_bytes_at(bytes, &mut pos, n_atoms, "SOA atomic_numbers")?;
    let n_sections = read_u16_le_at(bytes, &mut pos, "SOA n_sections")? as usize;

    let mut sections = Vec::with_capacity(n_sections);
    for _ in 0..n_sections {
        let kind = read_u8_at(bytes, &mut pos, "SOA section kind")?;
        let key_len = read_u8_at(bytes, &mut pos, "SOA section key length")? as usize;
        let key_bytes = read_bytes_at(bytes, &mut pos, key_len, "SOA section key")?;
        let key = std::str::from_utf8(key_bytes)
            .map_err(|_| invalid_data("Invalid UTF-8 in SOA section key"))?;
        let type_tag = read_u8_at(bytes, &mut pos, "SOA section type tag")?;
        let payload_len = read_u32_le_at(bytes, &mut pos, "SOA section payload length")? as usize;
        let payload = read_bytes_at(bytes, &mut pos, payload_len, "SOA section payload")?;
        sections.push(SectionRef {
            kind,
            key,
            type_tag,
            payload,
        });
    }

    Ok(MolData {
        n_atoms,
        positions_bytes,
        atomic_numbers_bytes,
        sections,
    })
}

fn section_schema_from_ref(
    section: &SectionRef<'_>,
    n_atoms: usize,
) -> atompack::Result<SectionSchema> {
    let per_atom = is_per_atom(section.kind, section.key, section.type_tag);
    let elem_bytes = match section.type_tag {
        TYPE_STRING => 0,
        tag if per_atom => {
            let elem_bytes = type_tag_elem_bytes(tag);
            if elem_bytes == 0 {
                return Err(invalid_data(format!(
                    "Unsupported per-atom section type tag {} for key '{}'",
                    tag, section.key
                )));
            }
            elem_bytes
        }
        TYPE_FLOAT | TYPE_INT => 8,
        TYPE_BOOL3 => 3,
        TYPE_MAT3X3_F64 => 72,
        _ => section.payload.len(),
    };
    let slot_bytes = if section.type_tag == TYPE_STRING {
        0
    } else if per_atom {
        elem_bytes
    } else {
        section.payload.len()
    };

    validate_section_payload(section, per_atom, elem_bytes, slot_bytes, n_atoms)?;

    Ok(SectionSchema {
        kind: section.kind,
        key: section.key.to_string(),
        type_tag: section.type_tag,
        per_atom,
        elem_bytes,
        slot_bytes,
    })
}

fn validate_section_payload(
    section: &SectionRef<'_>,
    per_atom: bool,
    elem_bytes: usize,
    slot_bytes: usize,
    n_atoms: usize,
) -> atompack::Result<()> {
    match section.type_tag {
        TYPE_STRING => {
            std::str::from_utf8(section.payload)
                .map_err(|_| invalid_data(format!("Invalid UTF-8 in section '{}'", section.key)))?;
            if per_atom {
                return Err(invalid_data(format!(
                    "String section '{}' cannot be per-atom in flat extraction",
                    section.key
                )));
            }
        }
        TYPE_FLOAT | TYPE_INT | TYPE_BOOL3 | TYPE_MAT3X3_F64 => {
            if section.payload.len() != slot_bytes {
                return Err(invalid_data(format!(
                    "Section '{}' has invalid payload length {} (expected {})",
                    section.key,
                    section.payload.len(),
                    slot_bytes
                )));
            }
        }
        _ if per_atom => {
            let expected = n_atoms.checked_mul(elem_bytes).ok_or_else(|| {
                invalid_data(format!("Section '{}' payload length overflow", section.key))
            })?;
            if section.payload.len() != expected {
                return Err(invalid_data(format!(
                    "Section '{}' has invalid payload length {} (expected {})",
                    section.key,
                    section.payload.len(),
                    expected
                )));
            }
        }
        _ => {
            if slot_bytes != 0 && section.payload.len() != slot_bytes {
                return Err(invalid_data(format!(
                    "Section '{}' has invalid payload length {} (expected {})",
                    section.key,
                    section.payload.len(),
                    slot_bytes
                )));
            }
            if elem_bytes != 0 && !section.payload.len().is_multiple_of(elem_bytes) {
                return Err(invalid_data(format!(
                    "Section '{}' has invalid payload length {} for element size {}",
                    section.key,
                    section.payload.len(),
                    elem_bytes
                )));
            }
        }
    }
    Ok(())
}

/// Element size in bytes for a given type tag. Returns 0 for variable-length types.
fn type_tag_elem_bytes(tag: u8) -> usize {
    match tag {
        TYPE_FLOAT => 8,
        TYPE_INT => 8,
        TYPE_STRING => 0, // variable
        TYPE_F64_ARRAY => 8,
        TYPE_VEC3_F32 => 12,
        TYPE_I64_ARRAY => 8,
        TYPE_F32_ARRAY => 4,
        TYPE_VEC3_F64 => 24,
        TYPE_I32_ARRAY => 4,
        TYPE_BOOL3 => 3,
        TYPE_MAT3X3_F64 => 72,
        _ => 0,
    }
}

/// Whether a section with the given kind/key/type_tag is per-atom (vs per-molecule).
fn is_per_atom(kind: u8, key: &str, _type_tag: u8) -> bool {
    match kind {
        KIND_ATOM_PROP => true,
        KIND_MOL_PROP => false,
        KIND_BUILTIN => matches!(key, "forces" | "charges" | "velocities"),
        _ => false,
    }
}

/// Lightweight section descriptor — stores byte offsets into the parent `bytes` buffer.
/// Key is NOT parsed eagerly; use `key()` to read it lazily from `bytes`.
#[derive(Clone, Copy)]
struct LazySection {
    kind: u8,
    key_start: usize,
    key_len: u8,
    type_tag: u8,
    payload_start: usize,
    payload_len: usize,
}

/// Byte-offset pair for a known builtin section (payload_start, payload_len, type_tag).
type BuiltinSlot = (usize, usize, u8);

enum SoaBytes {
    Owned(Vec<u8>),
    Shared(SharedMmapBytes),
}

impl SoaBytes {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Shared(bytes) => bytes.as_slice(),
        }
    }
}

impl std::ops::Deref for SoaBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

struct SoaMoleculeView {
    bytes: SoaBytes,
    n_atoms: usize,
    positions_start: usize,
    atomic_numbers_start: usize,
    // Known builtins — zero-alloc, set during from_bytes
    forces: Option<BuiltinSlot>,
    energy: Option<BuiltinSlot>,
    cell: Option<BuiltinSlot>,
    stress: Option<BuiltinSlot>,
    charges: Option<BuiltinSlot>,
    velocities: Option<BuiltinSlot>,
    pbc: Option<BuiltinSlot>,
    name: Option<BuiltinSlot>,
    // Custom properties — lazy, no String parsing until accessed
    custom_sections: Vec<LazySection>,
}

impl SoaMoleculeView {
    /// Pure-Rust parser — no Python dependency, safe to call from rayon threads.
    fn from_storage_inner(bytes: SoaBytes) -> atompack::Result<Self> {
        if bytes.len() < 6 {
            return Err(invalid_data("SOA record too small"));
        }

        let n_atoms = u32::from_le_bytes(slice_to_array(&bytes[0..4], "SOA atom count")?) as usize;
        let mut pos = 4usize;
        let positions_start = pos;
        pos = n_atoms
            .checked_mul(12)
            .and_then(|n| pos.checked_add(n))
            .ok_or_else(|| invalid_data("SOA positions overflow"))?;
        if pos > bytes.len() {
            return Err(invalid_data("SOA record truncated at positions"));
        }

        let atomic_numbers_start = pos;
        pos = pos
            .checked_add(n_atoms)
            .ok_or_else(|| invalid_data("SOA atomic_numbers overflow"))?;
        if pos + 2 > bytes.len() {
            return Err(invalid_data("SOA record truncated at atomic_numbers"));
        }

        let n_sections =
            u16::from_le_bytes(slice_to_array(&bytes[pos..pos + 2], "SOA section count")?) as usize;
        pos += 2;

        let mut forces = None;
        let mut energy = None;
        let mut cell = None;
        let mut stress = None;
        let mut charges = None;
        let mut velocities = None;
        let mut pbc = None;
        let mut name = None;
        let mut custom_sections = Vec::new();

        for _ in 0..n_sections {
            if pos + 2 > bytes.len() {
                return Err(invalid_data("SOA section header truncated"));
            }
            let kind = bytes[pos];
            pos += 1;
            let key_len = bytes[pos] as usize;
            pos += 1;
            if pos + key_len > bytes.len() {
                return Err(invalid_data("SOA section key truncated"));
            }
            let key_start = pos;
            pos += key_len;
            if pos + 5 > bytes.len() {
                return Err(invalid_data("SOA section header truncated"));
            }
            let type_tag = bytes[pos];
            pos += 1;
            let payload_len = u32::from_le_bytes(slice_to_array(
                &bytes[pos..pos + 4],
                "SOA section payload length",
            )?) as usize;
            pos += 4;
            let payload_start = pos;
            pos = pos
                .checked_add(payload_len)
                .ok_or_else(|| invalid_data("SOA section payload overflow"))?;
            if pos > bytes.len() {
                return Err(invalid_data("SOA section payload truncated"));
            }

            let key_bytes = &bytes[key_start..key_start + key_len];
            if kind == KIND_BUILTIN {
                let slot = (payload_start, payload_len, type_tag);
                match key_bytes {
                    b"forces" => forces = Some(slot),
                    b"energy" => energy = Some(slot),
                    b"cell" => cell = Some(slot),
                    b"stress" => stress = Some(slot),
                    b"charges" => charges = Some(slot),
                    b"velocities" => velocities = Some(slot),
                    b"pbc" => pbc = Some(slot),
                    b"name" => name = Some(slot),
                    _ => {
                        custom_sections.push(LazySection {
                            kind,
                            key_start,
                            key_len: key_len as u8,
                            type_tag,
                            payload_start,
                            payload_len,
                        });
                    }
                }
            } else {
                custom_sections.push(LazySection {
                    kind,
                    key_start,
                    key_len: key_len as u8,
                    type_tag,
                    payload_start,
                    payload_len,
                });
            }
        }

        Ok(Self {
            bytes,
            n_atoms,
            positions_start,
            atomic_numbers_start,
            forces,
            energy,
            cell,
            stress,
            charges,
            velocities,
            pbc,
            name,
            custom_sections,
        })
    }

    fn from_bytes_inner(bytes: Vec<u8>) -> atompack::Result<Self> {
        Self::from_storage_inner(SoaBytes::Owned(bytes))
    }

    fn from_shared_bytes_inner(bytes: SharedMmapBytes) -> atompack::Result<Self> {
        Self::from_storage_inner(SoaBytes::Shared(bytes))
    }

    /// Thin wrapper for call sites that need PyResult.
    fn from_bytes(bytes: Vec<u8>) -> PyResult<Self> {
        Self::from_bytes_inner(bytes).map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    fn positions_bytes(&self) -> &[u8] {
        &self.bytes[self.positions_start..self.positions_start + self.n_atoms * 12]
    }

    fn atomic_numbers_bytes(&self) -> &[u8] {
        &self.bytes[self.atomic_numbers_start..self.atomic_numbers_start + self.n_atoms]
    }

    #[inline]
    fn builtin_payload(&self, slot: BuiltinSlot) -> &[u8] {
        &self.bytes[slot.0..slot.0 + slot.1]
    }

    fn lazy_section_key(&self, s: &LazySection) -> PyResult<&str> {
        std::str::from_utf8(&self.bytes[s.key_start..s.key_start + s.key_len as usize])
            .map_err(|_| PyValueError::new_err("Invalid UTF-8 in section key"))
    }

    fn lazy_section_payload(&self, s: &LazySection) -> &[u8] {
        &self.bytes[s.payload_start..s.payload_start + s.payload_len]
    }

    fn find_custom_section(&self, kind: u8, key: &str) -> PyResult<Option<&LazySection>> {
        for section in &self.custom_sections {
            if section.kind == kind && self.lazy_section_key(section)? == key {
                return Ok(Some(section));
            }
        }
        Ok(None)
    }

    fn property_keys(&self) -> PyResult<Vec<String>> {
        self.custom_sections
            .iter()
            .filter(|s| s.kind == KIND_MOL_PROP)
            .map(|s| Ok(self.lazy_section_key(s)?.to_string()))
            .collect()
    }

    fn atom_at(&self, index: usize) -> PyResult<Option<Atom>> {
        if index >= self.n_atoms {
            return Ok(None);
        }
        let pos = &self.positions_bytes()[index * 12..(index + 1) * 12];
        Ok(Some(Atom::new(
            f32::from_le_bytes(py_slice_to_array(&pos[0..4], "atom x")?),
            f32::from_le_bytes(py_slice_to_array(&pos[4..8], "atom y")?),
            f32::from_le_bytes(py_slice_to_array(&pos[8..12], "atom z")?),
            self.atomic_numbers_bytes()[index],
        )))
    }

    fn energy(&self) -> PyResult<Option<f64>> {
        match self.energy {
            Some(slot) => Ok(Some(read_f64_scalar(self.builtin_payload(slot))?)),
            None => Ok(None),
        }
    }

    fn pbc(&self) -> PyResult<Option<(bool, bool, bool)>> {
        match self.pbc {
            Some(slot) => {
                let payload = self.builtin_payload(slot);
                if payload.len() != 3 {
                    return Err(PyValueError::new_err("Invalid pbc payload length"));
                }
                Ok(Some((payload[0] != 0, payload[1] != 0, payload[2] != 0)))
            }
            None => Ok(None),
        }
    }

    fn materialize(&self) -> PyResult<Molecule> {
        let positions: Vec<[f32; 3]> = (0..self.n_atoms)
            .map(|i| {
                let pos = &self.positions_bytes()[i * 12..(i + 1) * 12];
                Ok([
                    f32::from_le_bytes(py_slice_to_array(&pos[0..4], "position x")?),
                    f32::from_le_bytes(py_slice_to_array(&pos[4..8], "position y")?),
                    f32::from_le_bytes(py_slice_to_array(&pos[8..12], "position z")?),
                ])
            })
            .collect::<PyResult<_>>()?;
        let atomic_numbers = self.atomic_numbers_bytes().to_vec();
        let mut molecule =
            Molecule::new(positions, atomic_numbers).map_err(PyValueError::new_err)?;

        // Builtins
        if let Some(slot) = self.charges {
            molecule.charges = Some(decode_f64_array(self.builtin_payload(slot))?);
        }
        if let Some(slot) = self.cell {
            molecule.cell = Some(decode_mat3x3_f64(self.builtin_payload(slot))?);
        }
        if let Some(slot) = self.energy {
            molecule.energy = Some(read_f64_scalar(self.builtin_payload(slot))?);
        }
        if let Some(slot) = self.forces {
            molecule.forces = Some(decode_vec3_f32(self.builtin_payload(slot))?);
        }
        if let Some(slot) = self.name {
            let payload = self.builtin_payload(slot);
            molecule.name = Some(
                std::str::from_utf8(payload)
                    .map_err(|_| PyValueError::new_err("Invalid UTF-8 in name"))?
                    .to_string(),
            );
        }
        if let Some(slot) = self.pbc {
            let payload = self.builtin_payload(slot);
            if payload.len() != 3 {
                return Err(PyValueError::new_err("Invalid pbc payload length"));
            }
            molecule.pbc = Some([payload[0] != 0, payload[1] != 0, payload[2] != 0]);
        }
        if let Some(slot) = self.stress {
            molecule.stress = Some(decode_mat3x3_f64(self.builtin_payload(slot))?);
        }
        if let Some(slot) = self.velocities {
            molecule.velocities = Some(decode_vec3_f32(self.builtin_payload(slot))?);
        }

        // Custom properties (lazy — key parsed here only)
        for s in &self.custom_sections {
            let key = self.lazy_section_key(s)?.to_string();
            let payload = self.lazy_section_payload(s);
            match s.kind {
                KIND_ATOM_PROP => {
                    molecule
                        .atom_properties
                        .insert(key, decode_property_value(s.type_tag, payload)?);
                }
                KIND_MOL_PROP => {
                    molecule
                        .properties
                        .insert(key, decode_property_value(s.type_tag, payload)?);
                }
                _ => {}
            }
        }

        Ok(molecule)
    }
}

fn read_f64_scalar(payload: &[u8]) -> PyResult<f64> {
    if payload.len() != 8 {
        return Err(PyValueError::new_err("Invalid f64 payload length"));
    }
    Ok(f64::from_le_bytes(py_slice_to_array(
        payload,
        "f64 payload",
    )?))
}

fn read_i64_scalar(payload: &[u8]) -> PyResult<i64> {
    if payload.len() != 8 {
        return Err(PyValueError::new_err("Invalid i64 payload length"));
    }
    Ok(i64::from_le_bytes(py_slice_to_array(
        payload,
        "i64 payload",
    )?))
}

fn decode_f64_array(payload: &[u8]) -> PyResult<Vec<f64>> {
    if !payload.len().is_multiple_of(8) {
        return Err(PyValueError::new_err("Invalid f64 array payload length"));
    }
    payload
        .chunks_exact(8)
        .map(|chunk| {
            Ok(f64::from_le_bytes(py_slice_to_array(
                chunk,
                "f64 array chunk",
            )?))
        })
        .collect()
}

fn decode_i64_array(payload: &[u8]) -> PyResult<Vec<i64>> {
    if !payload.len().is_multiple_of(8) {
        return Err(PyValueError::new_err("Invalid i64 array payload length"));
    }
    payload
        .chunks_exact(8)
        .map(|chunk| {
            Ok(i64::from_le_bytes(py_slice_to_array(
                chunk,
                "i64 array chunk",
            )?))
        })
        .collect()
}

fn decode_i32_array(payload: &[u8]) -> PyResult<Vec<i32>> {
    if !payload.len().is_multiple_of(4) {
        return Err(PyValueError::new_err("Invalid i32 array payload length"));
    }
    payload
        .chunks_exact(4)
        .map(|chunk| {
            Ok(i32::from_le_bytes(py_slice_to_array(
                chunk,
                "i32 array chunk",
            )?))
        })
        .collect()
}

fn decode_f32_array(payload: &[u8]) -> PyResult<Vec<f32>> {
    if !payload.len().is_multiple_of(4) {
        return Err(PyValueError::new_err("Invalid f32 array payload length"));
    }
    payload
        .chunks_exact(4)
        .map(|chunk| {
            Ok(f32::from_le_bytes(py_slice_to_array(
                chunk,
                "f32 array chunk",
            )?))
        })
        .collect()
}

fn decode_vec3_f32(payload: &[u8]) -> PyResult<Vec<[f32; 3]>> {
    if !payload.len().is_multiple_of(12) {
        return Err(PyValueError::new_err("Invalid vec3<f32> payload length"));
    }
    payload
        .chunks_exact(12)
        .map(|chunk| {
            Ok([
                f32::from_le_bytes(py_slice_to_array(&chunk[0..4], "vec3<f32> x")?),
                f32::from_le_bytes(py_slice_to_array(&chunk[4..8], "vec3<f32> y")?),
                f32::from_le_bytes(py_slice_to_array(&chunk[8..12], "vec3<f32> z")?),
            ])
        })
        .collect()
}

fn decode_vec3_f64(payload: &[u8]) -> PyResult<Vec<[f64; 3]>> {
    if !payload.len().is_multiple_of(24) {
        return Err(PyValueError::new_err("Invalid vec3<f64> payload length"));
    }
    payload
        .chunks_exact(24)
        .map(|chunk| {
            Ok([
                f64::from_le_bytes(py_slice_to_array(&chunk[0..8], "vec3<f64> x")?),
                f64::from_le_bytes(py_slice_to_array(&chunk[8..16], "vec3<f64> y")?),
                f64::from_le_bytes(py_slice_to_array(&chunk[16..24], "vec3<f64> z")?),
            ])
        })
        .collect()
}

fn decode_mat3x3_f64(payload: &[u8]) -> PyResult<[[f64; 3]; 3]> {
    if payload.len() != 72 {
        return Err(PyValueError::new_err("Invalid mat3x3<f64> payload length"));
    }
    Ok([
        [
            f64::from_le_bytes(py_slice_to_array(&payload[0..8], "mat3x3<f64> [0][0]")?),
            f64::from_le_bytes(py_slice_to_array(&payload[8..16], "mat3x3<f64> [0][1]")?),
            f64::from_le_bytes(py_slice_to_array(&payload[16..24], "mat3x3<f64> [0][2]")?),
        ],
        [
            f64::from_le_bytes(py_slice_to_array(&payload[24..32], "mat3x3<f64> [1][0]")?),
            f64::from_le_bytes(py_slice_to_array(&payload[32..40], "mat3x3<f64> [1][1]")?),
            f64::from_le_bytes(py_slice_to_array(&payload[40..48], "mat3x3<f64> [1][2]")?),
        ],
        [
            f64::from_le_bytes(py_slice_to_array(&payload[48..56], "mat3x3<f64> [2][0]")?),
            f64::from_le_bytes(py_slice_to_array(&payload[56..64], "mat3x3<f64> [2][1]")?),
            f64::from_le_bytes(py_slice_to_array(&payload[64..72], "mat3x3<f64> [2][2]")?),
        ],
    ])
}

fn decode_property_value(type_tag: u8, payload: &[u8]) -> PyResult<PropertyValue> {
    Ok(match type_tag {
        TYPE_FLOAT => PropertyValue::Float(read_f64_scalar(payload)?),
        TYPE_INT => PropertyValue::Int(read_i64_scalar(payload)?),
        TYPE_STRING => PropertyValue::String(
            std::str::from_utf8(payload)
                .map_err(|_| PyValueError::new_err("Invalid UTF-8 in string property"))?
                .to_string(),
        ),
        TYPE_F64_ARRAY => PropertyValue::FloatArray(decode_f64_array(payload)?),
        TYPE_VEC3_F32 => PropertyValue::Vec3Array(decode_vec3_f32(payload)?),
        TYPE_I64_ARRAY => PropertyValue::IntArray(decode_i64_array(payload)?),
        TYPE_F32_ARRAY => PropertyValue::Float32Array(decode_f32_array(payload)?),
        TYPE_VEC3_F64 => PropertyValue::Vec3ArrayF64(decode_vec3_f64(payload)?),
        TYPE_I32_ARRAY => PropertyValue::Int32Array(decode_i32_array(payload)?),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported property type tag {}",
                type_tag
            )));
        }
    })
}

mod database;
mod molecule;

use database::PyAtomDatabase;
use molecule::{PyAtom, PyMolecule};

#[pyfunction]
fn _molecule_from_pickle_bytes(payload: &Bound<'_, PyBytes>) -> PyResult<PyMolecule> {
    let molecule: Molecule = bincode::deserialize(payload.as_bytes())
        .map_err(|e| PyValueError::new_err(format!("Failed to deserialize molecule: {}", e)))?;
    Ok(PyMolecule::from_owned(molecule))
}

/// The Python module
///
/// This function defines what's available when you `import atompack` in Python
#[pymodule]
fn _atompack_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAtom>()?;
    m.add_class::<PyMolecule>()?;
    m.add_class::<PyAtomDatabase>()?;
    m.add_function(pyo3::wrap_pyfunction!(_molecule_from_pickle_bytes, m)?)?;
    Ok(())
}
