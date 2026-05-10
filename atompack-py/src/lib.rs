// Copyright 2026 Entalpic
//! Python bindings for atompack
//!
//! This module exposes the atompack library to Python using PyO3.

#![allow(clippy::useless_conversion)]
// PyO3 0.26 macro-generated code triggers this lint; safe to suppress.
#![allow(unsafe_op_in_unsafe_fn)]

use atompack::{
    Atom, AtomDatabase, FloatArrayData, FloatScalarData, Mat3Data, Molecule, SharedMmapBytes,
    Vec3Data, atom::PropertyValue, compression::CompressionType,
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
const TYPE_FLOAT32: u8 = 11;
const TYPE_MAT3X3_F32: u8 = 12;

const RECORD_FORMAT_SOA_V2: u32 = 2;
const RECORD_FORMAT_SOA_V3: u32 = 3;

mod py_dtypes;
mod soa;

pub(crate) use self::py_dtypes::{
    PyFloatArray1, PyFloatArray2, PyFloatArray3, PyIntArray1, parse_float_array_field,
    parse_mat3_field, parse_positions_field, parse_property_value, parse_vec3_field,
};
pub(crate) use self::soa::{
    LazySection, SectionRef, SectionSchema, SoaContext, SoaMoleculeView, is_per_atom,
    parse_mol_fast_soa, read_f64_scalar, read_i64_scalar, section_schema_from_ref,
    type_tag_elem_bytes, validate_section_payload,
};

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
