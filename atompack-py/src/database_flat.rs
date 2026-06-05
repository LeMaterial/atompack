use super::*;
use crate::molecule::{
    cast_or_decode_f32, cast_or_decode_f64, cast_or_decode_i32, cast_or_decode_i64,
    pyarray1_from_cow, pyarray2_from_cow,
};

enum FlatPositions {
    F32(Vec<f32>),
    F64(Vec<u8>),
}

type TensorSectionPayloads = Vec<Option<Vec<u8>>>;

fn missing_tensor_error(key: &str, index: usize) -> PyErr {
    PyValueError::new_err(format!(
        "Tensor property '{}' is missing for selected molecule {}; it cannot be \
         flat-batched/concatenated. Retrieve molecules individually with get_molecule(s) instead.",
        key, index
    ))
}

fn incompatible_tensor_shapes_error(key: &str, left: &[usize], right: &[usize]) -> PyErr {
    PyValueError::new_err(format!(
        "Tensor property '{}' has incompatible shapes {:?} and {:?}; it cannot be \
         flat-batched/concatenated. Retrieve molecules individually with get_molecule(s) instead.",
        key, left, right
    ))
}

fn incompatible_atom_tensor_suffix_error(key: &str, left: &[usize], right: &[usize]) -> PyErr {
    PyValueError::new_err(format!(
        "Atom tensor property '{}' has incompatible per-atom tensor suffix shapes {:?} and {:?}; \
         it cannot be flat-batched/concatenated. Retrieve molecules individually with \
         get_molecule(s) instead.",
        key, left, right
    ))
}

fn invalid_atom_tensor_shape_error(
    key: &str,
    shape: &[usize],
    first_dim: Option<usize>,
    n_atoms: usize,
) -> PyErr {
    PyValueError::new_err(format!(
        "Atom tensor property '{}' has shape {:?}; first dimension {:?} does not match \
         atom count {}. It cannot be flat-batched/concatenated. Retrieve molecules \
         individually with get_molecule(s) instead.",
        key, shape, first_dim, n_atoms
    ))
}

fn tensor_array_from_bytes<'py>(
    py: Python<'py>,
    type_tag: u8,
    bytes: Vec<u8>,
    shape: &[usize],
) -> PyResult<Py<PyAny>> {
    Ok(match type_tag {
        TYPE_TENSOR_F32 => pyarray1_from_cow(py, cast_or_decode_f32(&bytes)?)
            .reshape(shape)?
            .into_any()
            .unbind(),
        TYPE_TENSOR_F64 => pyarray1_from_cow(py, cast_or_decode_f64(&bytes)?)
            .reshape(shape)?
            .into_any()
            .unbind(),
        TYPE_TENSOR_I32 => pyarray1_from_cow(py, cast_or_decode_i32(&bytes)?)
            .reshape(shape)?
            .into_any()
            .unbind(),
        TYPE_TENSOR_I64 => pyarray1_from_cow(py, cast_or_decode_i64(&bytes)?)
            .reshape(shape)?
            .into_any()
            .unbind(),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported tensor type tag {}",
                type_tag
            )));
        }
    })
}

fn flat_tensor_array<'py>(
    py: Python<'py>,
    section: &SectionSchema,
    payloads: &[Option<Vec<u8>>],
    n_atoms_vec: &[u32],
    total_atoms: usize,
) -> PyResult<Py<PyAny>> {
    let mut data = Vec::new();

    if section.per_atom {
        let mut expected_suffix: Option<Vec<usize>> = None;
        for (index, payload) in payloads.iter().enumerate() {
            let payload = payload
                .as_ref()
                .ok_or_else(|| missing_tensor_error(&section.key, index))?;
            let (shape, data_offset) =
                crate::soa::tensor_shape_from_payload(section.type_tag, payload.as_slice())
                    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
            let n_atoms = n_atoms_vec[index] as usize;
            let Some((&first_dim, suffix)) = shape.split_first() else {
                return Err(invalid_atom_tensor_shape_error(
                    &section.key,
                    &shape,
                    None,
                    n_atoms,
                ));
            };
            if first_dim != n_atoms {
                return Err(invalid_atom_tensor_shape_error(
                    &section.key,
                    &shape,
                    Some(first_dim),
                    n_atoms,
                ));
            }
            if let Some(expected) = &expected_suffix {
                if expected.as_slice() != suffix {
                    return Err(incompatible_atom_tensor_suffix_error(
                        &section.key,
                        expected,
                        suffix,
                    ));
                }
            } else {
                expected_suffix = Some(suffix.to_vec());
            }
            data.extend_from_slice(&payload[data_offset..]);
        }
        let suffix = expected_suffix.unwrap_or_default();
        let mut output_shape = Vec::with_capacity(1 + suffix.len());
        output_shape.push(total_atoms);
        output_shape.extend(suffix);
        tensor_array_from_bytes(py, section.type_tag, data, &output_shape)
    } else {
        let mut expected_shape: Option<Vec<usize>> = None;
        for (index, payload) in payloads.iter().enumerate() {
            let payload = payload
                .as_ref()
                .ok_or_else(|| missing_tensor_error(&section.key, index))?;
            let (shape, data_offset) =
                crate::soa::tensor_shape_from_payload(section.type_tag, payload.as_slice())
                    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
            if let Some(expected) = &expected_shape {
                if expected != &shape {
                    return Err(incompatible_tensor_shapes_error(
                        &section.key,
                        expected,
                        &shape,
                    ));
                }
            } else {
                expected_shape = Some(shape.clone());
            }
            data.extend_from_slice(&payload[data_offset..]);
        }
        let tensor_shape = expected_shape.unwrap_or_default();
        let mut output_shape = Vec::with_capacity(1 + tensor_shape.len());
        output_shape.push(payloads.len());
        output_shape.extend(tensor_shape);
        tensor_array_from_bytes(py, section.type_tag, data, &output_shape)
    }
}

pub(super) fn get_molecules_flat_soa_impl<'py>(
    inner: &AtomDatabase,
    py: Python<'py>,
    indices: Vec<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = py
        .detach(|| -> std::result::Result<_, atompack::Error> {
            use rayon::prelude::*;

            let n_mols = indices.len();
            if n_mols == 0 {
                return Ok(None);
            }

            let n_atoms_vec: Vec<u32> = indices
                .iter()
                .map(|&i| {
                    inner.num_atoms(i).ok_or_else(|| {
                        atompack::Error::InvalidData(format!("Index {} out of bounds", i))
                    })
                })
                .collect::<atompack::Result<_>>()?;

            let mut offsets = Vec::with_capacity(n_mols + 1);
            offsets.push(0usize);
            for &n in &n_atoms_vec {
                let next = offsets
                    .last()
                    .copied()
                    .ok_or_else(|| invalid_data("missing offset seed"))?
                    .checked_add(n as usize)
                    .ok_or_else(|| invalid_data("atom offset overflow"))?;
                offsets.push(next);
            }
            let total_atoms = offsets
                .last()
                .copied()
                .ok_or_else(|| invalid_data("missing final atom offset"))?;

            let compression = inner.compression();
            let use_mmap = inner.get_compressed_slice(0).is_some();
            let ctx = SoaContext::from_database(inner)?;
            let positions_type = ctx.positions_type();
            let schema_info = inner.schema_info();
            let raw_bytes_owned: Option<Vec<Vec<u8>>>;
            let schema: Vec<SectionSchema>;
            let use_ordered_schema: bool;

            let ordered_schema_from_first = |bytes: &[u8]| -> atompack::Result<Vec<SectionSchema>> {
                let first_md = parse_mol_fast_soa(bytes, ctx)?;
                let n = first_md.n_atoms;
                first_md
                    .sections
                    .iter()
                    .map(|s| section_schema_from_ref(s, n))
                    .collect::<atompack::Result<_>>()
            };

            let schema_matches_ordered =
                |ordered: &[SectionSchema], from_lock: &[SectionSchema]| {
                    if ordered.len() != from_lock.len() {
                        return false;
                    }
                    let ordered_lookup: std::collections::HashMap<(u8, &str), &SectionSchema> =
                        ordered
                            .iter()
                            .map(|entry| ((entry.kind, entry.key.as_str()), entry))
                            .collect();
                    from_lock.iter().all(|entry| {
                        ordered_lookup
                            .get(&(entry.kind, entry.key.as_str()))
                            .is_some_and(|candidate| {
                                candidate.type_tag == entry.type_tag
                                    && candidate.per_atom == entry.per_atom
                                    && candidate.elem_bytes == entry.elem_bytes
                                    && candidate.slot_bytes == entry.slot_bytes
                            })
                    })
                };

            if let Some(schema_info) = schema_info {
                let schema_from_lock: Vec<SectionSchema> = schema_info
                    .sections
                    .into_iter()
                    .map(|section| SectionSchema {
                        kind: section.kind,
                        key: section.key,
                        type_tag: section.type_tag,
                        per_atom: section.per_atom,
                        elem_bytes: section.elem_bytes,
                        slot_bytes: section.slot_bytes,
                    })
                    .collect();
                if use_mmap {
                    if compression == CompressionType::None {
                        let shared = inner.get_shared_mmap_bytes(indices[0]).ok_or_else(|| {
                            invalid_data(format!("Missing mmap bytes for molecule {}", indices[0]))
                        })?;
                        let ordered = ordered_schema_from_first(shared.as_slice())?;
                        use_ordered_schema = schema_matches_ordered(&ordered, &schema_from_lock);
                        schema = if use_ordered_schema {
                            ordered
                        } else {
                            schema_from_lock
                        };
                        raw_bytes_owned = None;
                    } else {
                        let compressed =
                            inner.get_compressed_slice(indices[0]).ok_or_else(|| {
                                invalid_data(format!(
                                    "Missing compressed bytes for molecule {}",
                                    indices[0]
                                ))
                            })?;
                        let uncompressed_size =
                            inner.uncompressed_size(indices[0]).ok_or_else(|| {
                                invalid_data(format!(
                                    "Missing uncompressed size for molecule {}",
                                    indices[0]
                                ))
                            })? as usize;
                        let first_bytes = atompack::decompress_bytes(
                            compressed,
                            compression,
                            Some(uncompressed_size),
                        )?;
                        let ordered = ordered_schema_from_first(&first_bytes)?;
                        use_ordered_schema = schema_matches_ordered(&ordered, &schema_from_lock);
                        schema = if use_ordered_schema {
                            ordered
                        } else {
                            schema_from_lock
                        };
                        raw_bytes_owned = None;
                    }
                } else {
                    let (raw_bytes, _) = inner.read_decompress_parallel(&indices)?;
                    let ordered = ordered_schema_from_first(&raw_bytes[0])?;
                    use_ordered_schema = schema_matches_ordered(&ordered, &schema_from_lock);
                    schema = if use_ordered_schema {
                        ordered
                    } else {
                        schema_from_lock
                    };
                    raw_bytes_owned = Some(raw_bytes);
                }
            } else if use_mmap {
                if compression == CompressionType::None {
                    let shared = inner.get_shared_mmap_bytes(indices[0]).ok_or_else(|| {
                        invalid_data(format!("Missing mmap bytes for molecule {}", indices[0]))
                    })?;
                    schema = ordered_schema_from_first(shared.as_slice())?;
                    use_ordered_schema = true;
                } else {
                    let compressed = inner.get_compressed_slice(indices[0]).ok_or_else(|| {
                        invalid_data(format!(
                            "Missing compressed bytes for molecule {}",
                            indices[0]
                        ))
                    })?;
                    let uncompressed_size =
                        inner.uncompressed_size(indices[0]).ok_or_else(|| {
                            invalid_data(format!(
                                "Missing uncompressed size for molecule {}",
                                indices[0]
                            ))
                        })? as usize;
                    let first_bytes = atompack::decompress_bytes(
                        compressed,
                        compression,
                        Some(uncompressed_size),
                    )?;
                    schema = ordered_schema_from_first(&first_bytes)?;
                    use_ordered_schema = true;
                }
                raw_bytes_owned = None;
            } else {
                let (raw_bytes, _) = inner.read_decompress_parallel(&indices)?;
                schema = ordered_schema_from_first(&raw_bytes[0])?;
                use_ordered_schema = true;
                raw_bytes_owned = Some(raw_bytes);
            }

            let schema_keys: Vec<(u8, &[u8])> = if use_ordered_schema {
                schema
                    .iter()
                    .map(|entry| (entry.kind, entry.key.as_bytes()))
                    .collect()
            } else {
                Vec::new()
            };
            let mut schema_lookup: std::collections::HashMap<
                u8,
                std::collections::HashMap<String, usize>,
            > = std::collections::HashMap::new();
            if !use_ordered_schema {
                for (index, entry) in schema.iter().enumerate() {
                    schema_lookup
                        .entry(entry.kind)
                        .or_default()
                        .insert(entry.key.clone(), index);
                }
            }
            let positions_stride = ctx.layout.positions_stride;
            let mut positions = match positions_type {
                TYPE_VEC3_F32 => FlatPositions::F32(vec![0f32; total_atoms * 3]),
                TYPE_VEC3_F64 => FlatPositions::F64(vec![0u8; total_atoms * positions_stride]),
                _ => unreachable!(),
            };
            let mut atomic_numbers = vec![0u8; total_atoms];

            let mut section_buffers: Vec<Vec<u8>> = schema
                .iter()
                .map(|s| {
                    if s.slot_bytes == 0 || is_tensor_type_tag(s.type_tag) {
                        Vec::new()
                    } else if s.per_atom {
                        vec![0u8; total_atoms * s.elem_bytes]
                    } else {
                        vec![0u8; n_mols * s.slot_bytes]
                    }
                })
                .collect();

            let mut string_sections: Vec<Option<Vec<Option<String>>>> = schema
                .iter()
                .map(|s| {
                    if matches!(s.type_tag, TYPE_STRING | TYPE_NONE) {
                        Some(vec![None; n_mols])
                    } else {
                        None
                    }
                })
                .collect();
            let mut tensor_sections: Vec<Option<TensorSectionPayloads>> = schema
                .iter()
                .map(|s| {
                    if is_tensor_type_tag(s.type_tag) {
                        Some(vec![None; n_mols])
                    } else {
                        None
                    }
                })
                .collect();

            let pos_buf_f32 = match &mut positions {
                FlatPositions::F32(values) => Some(RawBuf::new(values)),
                FlatPositions::F64(_) => None,
            };
            let pos_buf_f64 = match &mut positions {
                FlatPositions::F32(_) => None,
                FlatPositions::F64(values) => Some(RawBuf::new(values)),
            };
            let z_buf = RawBuf::new(&mut atomic_numbers);
            let sec_bufs: Vec<RawBuf<u8>> = section_buffers
                .iter_mut()
                .map(|buf| {
                    if buf.is_empty() {
                        RawBuf {
                            ptr: std::ptr::null_mut(),
                        }
                    } else {
                        RawBuf::new(buf)
                    }
                })
                .collect();

            let string_mutexes: Vec<Option<std::sync::Mutex<&mut Vec<Option<String>>>>> =
                string_sections
                    .iter_mut()
                    .map(|opt| opt.as_mut().map(std::sync::Mutex::new))
                    .collect();
            let tensor_mutexes: Vec<Option<std::sync::Mutex<&mut TensorSectionPayloads>>> =
                tensor_sections
                    .iter_mut()
                    .map(|opt| opt.as_mut().map(std::sync::Mutex::new))
                    .collect();

            let process_mol = |i: usize, mol_bytes: &[u8]| -> atompack::Result<()> {
                let md = parse_mol_fast_soa(mol_bytes, ctx)?;
                let atom_off = offsets[i];
                let n = md.n_atoms;

                unsafe {
                    match positions_type {
                        TYPE_VEC3_F32 => {
                            let pos_buf = pos_buf_f32
                                .as_ref()
                                .ok_or_else(|| invalid_data("missing f32 position buffer"))?;
                            std::ptr::copy_nonoverlapping(
                                md.positions_bytes.as_ptr(),
                                pos_buf.at(atom_off * 3) as *mut u8,
                                n * 12,
                            );
                        }
                        TYPE_VEC3_F64 => {
                            let pos_buf = pos_buf_f64
                                .as_ref()
                                .ok_or_else(|| invalid_data("missing f64 position buffer"))?;
                            std::ptr::copy_nonoverlapping(
                                md.positions_bytes.as_ptr(),
                                pos_buf.at(atom_off * positions_stride),
                                n * positions_stride,
                            );
                        }
                        other => {
                            return Err(invalid_data(format!(
                                "Unsupported positions type tag {}",
                                other
                            )));
                        }
                    }
                    std::ptr::copy_nonoverlapping(
                        md.atomic_numbers_bytes.as_ptr(),
                        z_buf.at(atom_off),
                        n,
                    );
                }

                if use_ordered_schema {
                    if md.sections.len() != schema.len() {
                        return Err(invalid_data(format!(
                            "SOA schema mismatch for molecule {}: expected {} sections, got {}",
                            i,
                            schema.len(),
                            md.sections.len()
                        )));
                    }
                    for (section_idx, sec) in md.sections.iter().enumerate() {
                        let schema_entry = &schema[section_idx];
                        let expected_key = &schema_keys[section_idx];

                        if sec.kind != expected_key.0 || sec.key.as_bytes() != expected_key.1 {
                            return Err(invalid_data(format!(
                                "SOA schema order mismatch at molecule {} for section '{}'",
                                i, sec.key
                            )));
                        }

                        if sec.type_tag != schema_entry.type_tag {
                            return Err(invalid_data(format!(
                                "SOA schema mismatch at molecule {} for section '{}'",
                                i, sec.key
                            )));
                        }

                        if is_tensor_type_tag(schema_entry.type_tag) {
                            let _ = crate::soa::tensor_shape_from_payload(
                                schema_entry.type_tag,
                                sec.payload,
                            )?;
                        } else if schema_entry.per_atom {
                            let expected =
                                n.checked_mul(schema_entry.elem_bytes).ok_or_else(|| {
                                    invalid_data(format!(
                                        "Section '{}' payload length overflow",
                                        sec.key
                                    ))
                                })?;
                            if sec.payload.len() != expected {
                                return Err(invalid_data(format!(
                                    "Section '{}' has invalid payload length {} (expected {})",
                                    sec.key,
                                    sec.payload.len(),
                                    expected
                                )));
                            }
                        } else if schema_entry.slot_bytes != 0
                            && sec.payload.len() != schema_entry.slot_bytes
                        {
                            return Err(invalid_data(format!(
                                "Section '{}' has invalid payload length {} (expected {})",
                                sec.key,
                                sec.payload.len(),
                                schema_entry.slot_bytes
                            )));
                        }

                        if is_tensor_type_tag(schema_entry.type_tag) {
                            if let Some(ref mtx) = tensor_mutexes[section_idx] {
                                let mut guard = mtx
                                    .lock()
                                    .map_err(|_| invalid_data("tensor section mutex poisoned"))?;
                                guard[i] = Some(sec.payload.to_vec());
                            }
                        } else if schema_entry.slot_bytes == 0 {
                            if schema_entry.type_tag == TYPE_NONE {
                                continue;
                            }
                            if let Some(ref mtx) = string_mutexes[section_idx] {
                                let val = Some(
                                    std::str::from_utf8(sec.payload)
                                        .map_err(|_| {
                                            invalid_data(format!(
                                                "Invalid UTF-8 in section '{}'",
                                                sec.key
                                            ))
                                        })?
                                        .to_string(),
                                );
                                let mut guard = mtx
                                    .lock()
                                    .map_err(|_| invalid_data("string section mutex poisoned"))?;
                                guard[i] = val;
                            }
                        } else {
                            let buf = &sec_bufs[section_idx];
                            let offset = if schema_entry.per_atom {
                                atom_off * schema_entry.elem_bytes
                            } else {
                                i * schema_entry.slot_bytes
                            };
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    sec.payload.as_ptr(),
                                    buf.at(offset),
                                    sec.payload.len(),
                                );
                            }
                        }
                    }
                } else {
                    for sec in &md.sections {
                        let section_idx = schema_lookup
                            .get(&sec.kind)
                            .and_then(|entries| entries.get(sec.key))
                            .copied()
                            .ok_or_else(|| {
                                invalid_data(format!(
                                    "Unexpected SOA section '{}' in molecule {}",
                                    sec.key, i
                                ))
                            })?;
                        let schema_entry = &schema[section_idx];

                        if sec.type_tag != schema_entry.type_tag {
                            return Err(invalid_data(format!(
                                "SOA schema mismatch at molecule {} for section '{}'",
                                i, sec.key
                            )));
                        }

                        if is_tensor_type_tag(schema_entry.type_tag) {
                            let _ = crate::soa::tensor_shape_from_payload(
                                schema_entry.type_tag,
                                sec.payload,
                            )?;
                        } else if schema_entry.per_atom {
                            let expected =
                                n.checked_mul(schema_entry.elem_bytes).ok_or_else(|| {
                                    invalid_data(format!(
                                        "Section '{}' payload length overflow",
                                        sec.key
                                    ))
                                })?;
                            if sec.payload.len() != expected {
                                return Err(invalid_data(format!(
                                    "Section '{}' has invalid payload length {} (expected {})",
                                    sec.key,
                                    sec.payload.len(),
                                    expected
                                )));
                            }
                        } else if schema_entry.slot_bytes != 0
                            && sec.payload.len() != schema_entry.slot_bytes
                        {
                            return Err(invalid_data(format!(
                                "Section '{}' has invalid payload length {} (expected {})",
                                sec.key,
                                sec.payload.len(),
                                schema_entry.slot_bytes
                            )));
                        }

                        if is_tensor_type_tag(schema_entry.type_tag) {
                            if let Some(ref mtx) = tensor_mutexes[section_idx] {
                                let mut guard = mtx
                                    .lock()
                                    .map_err(|_| invalid_data("tensor section mutex poisoned"))?;
                                guard[i] = Some(sec.payload.to_vec());
                            }
                        } else if schema_entry.slot_bytes == 0 {
                            if schema_entry.type_tag == TYPE_NONE {
                                continue;
                            }
                            if let Some(ref mtx) = string_mutexes[section_idx] {
                                let val = Some(
                                    std::str::from_utf8(sec.payload)
                                        .map_err(|_| {
                                            invalid_data(format!(
                                                "Invalid UTF-8 in section '{}'",
                                                sec.key
                                            ))
                                        })?
                                        .to_string(),
                                );
                                let mut guard = mtx
                                    .lock()
                                    .map_err(|_| invalid_data("string section mutex poisoned"))?;
                                guard[i] = val;
                            }
                        } else {
                            let buf = &sec_bufs[section_idx];
                            let offset = if schema_entry.per_atom {
                                atom_off * schema_entry.elem_bytes
                            } else {
                                i * schema_entry.slot_bytes
                            };
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    sec.payload.as_ptr(),
                                    buf.at(offset),
                                    sec.payload.len(),
                                );
                            }
                        }
                    }
                }
                Ok(())
            };

            let results: Vec<atompack::Result<()>> = if use_mmap {
                (0..n_mols)
                    .into_par_iter()
                    .map(|i| {
                        let idx = indices[i];
                        if compression == CompressionType::None {
                            let shared = inner.get_shared_mmap_bytes(idx).ok_or_else(|| {
                                invalid_data(format!("Missing mmap bytes for molecule {}", idx))
                            })?;
                            process_mol(i, shared.as_slice())
                        } else {
                            let compressed = inner.get_compressed_slice(idx).ok_or_else(|| {
                                invalid_data(format!(
                                    "Missing compressed bytes for molecule {}",
                                    idx
                                ))
                            })?;
                            let uncompressed_size =
                                inner.uncompressed_size(idx).ok_or_else(|| {
                                    invalid_data(format!(
                                        "Missing uncompressed size for molecule {}",
                                        idx
                                    ))
                                })? as usize;
                            let decompressed = atompack::decompress_bytes(
                                compressed,
                                compression,
                                Some(uncompressed_size),
                            )?;
                            process_mol(i, &decompressed)
                        }
                    })
                    .collect()
            } else {
                let raw_bytes = raw_bytes_owned.expect("raw bytes must exist without mmap");
                raw_bytes
                    .par_iter()
                    .enumerate()
                    .map(|(i, bytes)| process_mol(i, bytes))
                    .collect()
            };
            results.into_iter().collect::<atompack::Result<Vec<_>>>()?;

            Ok(Some((
                n_atoms_vec,
                positions,
                atomic_numbers,
                schema,
                section_buffers,
                string_sections,
                tensor_sections,
                total_atoms,
            )))
        })
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let (
        n_atoms_vec,
        positions,
        atomic_numbers,
        schema,
        section_buffers,
        string_results,
        tensor_results,
        total_atoms,
    ) = match result {
        None => {
            let dict = PyDict::new(py);
            dict.set_item("n_atoms", PyArray1::from_vec(py, Vec::<u32>::new()))?;
            dict.set_item(
                "positions",
                PyArray1::from_vec(py, Vec::<f32>::new())
                    .reshape([0, 3])
                    .map_err(|e| PyValueError::new_err(format!("{}", e)))?,
            )?;
            dict.set_item("atomic_numbers", PyArray1::from_vec(py, Vec::<u8>::new()))?;
            return Ok(dict);
        }
        Some(v) => v,
    };

    let dict = PyDict::new(py);
    dict.set_item("n_atoms", PyArray1::from_slice(py, &n_atoms_vec))?;
    match positions {
        FlatPositions::F32(values) => {
            dict.set_item(
                "positions",
                PyArray1::from_vec(py, values)
                    .reshape([total_atoms, 3])
                    .map_err(|e| PyValueError::new_err(format!("{}", e)))?,
            )?;
        }
        FlatPositions::F64(bytes) => {
            let arr = cast_or_decode_f64(&bytes)?;
            dict.set_item("positions", pyarray2_from_cow(py, arr, total_atoms, 3)?)?;
        }
    }
    dict.set_item("atomic_numbers", PyArray1::from_vec(py, atomic_numbers))?;

    let atom_props_dict = PyDict::new(py);
    let mol_props_dict = PyDict::new(py);

    for (((s, buf), str_result), tensor_result) in schema
        .iter()
        .zip(section_buffers.into_iter())
        .zip(string_results.iter())
        .zip(tensor_results.iter())
    {
        let target = match s.kind {
            KIND_ATOM_PROP => &atom_props_dict,
            KIND_MOL_PROP => &mol_props_dict,
            _ => &dict,
        };

        if is_tensor_type_tag(s.type_tag) {
            if let Some(payloads) = tensor_result {
                target.set_item(
                    &s.key,
                    flat_tensor_array(py, s, payloads, &n_atoms_vec, total_atoms)?,
                )?;
            }
            continue;
        }

        if s.slot_bytes == 0 {
            if let Some(strings) = str_result {
                let py_list: Vec<Py<PyAny>> = strings
                    .iter()
                    .map(|opt| match opt {
                        Some(s) => s.into_bound_py_any(py).map(Bound::unbind),
                        None => Ok(py.None()),
                    })
                    .collect::<PyResult<_>>()?;
                target.set_item(&s.key, py_list)?;
            }
            continue;
        }

        match s.type_tag {
            TYPE_FLOAT => {
                let arr = cast_or_decode_f64(&buf)?;
                target.set_item(&s.key, pyarray1_from_cow(py, arr))?;
            }
            TYPE_FLOAT32 => {
                let arr = cast_or_decode_f32(&buf)?;
                target.set_item(&s.key, pyarray1_from_cow(py, arr))?;
            }
            TYPE_INT => {
                let arr = cast_or_decode_i64(&buf)?;
                target.set_item(&s.key, pyarray1_from_cow(py, arr))?;
            }
            TYPE_F64_ARRAY => {
                let arr = cast_or_decode_f64(&buf)?;
                target.set_item(&s.key, pyarray1_from_cow(py, arr))?;
            }
            TYPE_VEC3_F32 => {
                let arr = cast_or_decode_f32(&buf)?;
                let n = arr.len() / 3;
                target.set_item(&s.key, pyarray2_from_cow(py, arr, n, 3)?)?;
            }
            TYPE_I64_ARRAY => {
                let arr = cast_or_decode_i64(&buf)?;
                target.set_item(&s.key, pyarray1_from_cow(py, arr))?;
            }
            TYPE_F32_ARRAY => {
                let arr = cast_or_decode_f32(&buf)?;
                target.set_item(&s.key, pyarray1_from_cow(py, arr))?;
            }
            TYPE_VEC3_F64 => {
                let arr = cast_or_decode_f64(&buf)?;
                let n = arr.len() / 3;
                target.set_item(&s.key, pyarray2_from_cow(py, arr, n, 3)?)?;
            }
            TYPE_I32_ARRAY => {
                let arr = cast_or_decode_i32(&buf)?;
                target.set_item(&s.key, pyarray1_from_cow(py, arr))?;
            }
            TYPE_BOOL3 => {
                let arr: Vec<bool> = buf.iter().map(|&b| b != 0).collect();
                let n = arr.len() / 3;
                target.set_item(
                    &s.key,
                    PyArray1::from_vec(py, arr)
                        .reshape([n, 3])
                        .map_err(|e| PyValueError::new_err(format!("{}", e)))?,
                )?;
            }
            TYPE_MAT3X3_F64 => {
                let arr = cast_or_decode_f64(&buf)?;
                let n = arr.len() / 9;
                target.set_item(
                    &s.key,
                    pyarray1_from_cow(py, arr)
                        .reshape([n, 3, 3])
                        .map_err(|e| PyValueError::new_err(format!("{}", e)))?,
                )?;
            }
            TYPE_MAT3X3_F32 => {
                let arr = cast_or_decode_f32(&buf)?;
                let n = arr.len() / 9;
                target.set_item(
                    &s.key,
                    pyarray1_from_cow(py, arr)
                        .reshape([n, 3, 3])
                        .map_err(|e| PyValueError::new_err(format!("{}", e)))?,
                )?;
            }
            _ => {}
        }
    }

    if atom_props_dict.len() > 0 {
        dict.set_item("atom_properties", &atom_props_dict)?;
    }
    if mol_props_dict.len() > 0 {
        dict.set_item("properties", &mol_props_dict)?;
    }

    Ok(dict)
}
