use super::*;
use crate::molecule::{
    cast_or_decode_f32, cast_or_decode_f64, cast_or_decode_i32, cast_or_decode_i64,
    pyarray1_from_cow, pyarray2_from_cow,
};

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

            let record_format = inner.record_format();
            let positions_type = inner
                .positions_type()
                .ok_or_else(|| invalid_data("Missing position dtype for batch"))?;
            let (raw_bytes, _) = inner.read_decompress_parallel(&indices)?;

            let mut schema: Vec<SectionSchema> = Vec::new();
            for bytes in &raw_bytes {
                let md = parse_mol_fast_soa(bytes, record_format, Some(positions_type))?;

                for section in &md.sections {
                    let incoming = section_schema_from_ref(section, md.n_atoms)?;
                    if let Some(existing) = schema.iter().find(|candidate| {
                        candidate.kind == incoming.kind && candidate.key == incoming.key
                    }) {
                        if existing.type_tag != incoming.type_tag
                            || existing.per_atom != incoming.per_atom
                            || existing.elem_bytes != incoming.elem_bytes
                            || existing.slot_bytes != incoming.slot_bytes
                        {
                            return Err(invalid_data(format!(
                                "SOA schema mismatch for section '{}'",
                                incoming.key
                            )));
                        }
                    } else {
                        schema.push(incoming);
                    }
                }
            }
            let positions_stride = match positions_type {
                TYPE_VEC3_F32 => 12usize,
                TYPE_VEC3_F64 => 24usize,
                _ => {
                    return Err(invalid_data(format!(
                        "Unsupported positions type tag {}",
                        positions_type
                    )));
                }
            };
            let mut positions = vec![0u8; total_atoms * positions_stride];
            let mut atomic_numbers = vec![0u8; total_atoms];

            let mut section_buffers: Vec<Vec<u8>> = schema
                .iter()
                .map(|s| {
                    if s.slot_bytes == 0 {
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
                    if s.slot_bytes == 0 {
                        Some(vec![None; n_mols])
                    } else {
                        None
                    }
                })
                .collect();

            let pos_buf = RawBuf::new(&mut positions);
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

            let process_mol = |i: usize, mol_bytes: &[u8]| -> atompack::Result<()> {
                let md = parse_mol_fast_soa(mol_bytes, record_format, Some(positions_type))?;
                let atom_off = offsets[i];
                let n = md.n_atoms;

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        md.positions_bytes.as_ptr(),
                        pos_buf.at(atom_off * positions_stride),
                        n * positions_stride,
                    );
                    std::ptr::copy_nonoverlapping(
                        md.atomic_numbers_bytes.as_ptr(),
                        z_buf.at(atom_off),
                        n,
                    );
                }

                for (section_idx, schema_entry) in schema.iter().enumerate() {
                    let sec = md
                        .sections
                        .iter()
                        .find(|sec| sec.kind == schema_entry.kind && sec.key == schema_entry.key);
                    let Some(sec) = sec else {
                        continue;
                    };

                    if sec.type_tag != schema_entry.type_tag {
                        return Err(invalid_data(format!(
                            "SOA schema mismatch at molecule {} for section '{}'",
                            i, sec.key
                        )));
                    }

                    if schema_entry.per_atom {
                        let expected = n.checked_mul(schema_entry.elem_bytes).ok_or_else(|| {
                            invalid_data(format!("Section '{}' payload length overflow", sec.key))
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

                    if schema_entry.slot_bytes == 0 {
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
                Ok(())
            };

            let results: Vec<atompack::Result<()>> = raw_bytes
                .par_iter()
                .enumerate()
                .map(|(i, bytes)| process_mol(i, bytes))
                .collect();
            results.into_iter().collect::<atompack::Result<Vec<_>>>()?;

            Ok(Some((
                n_atoms_vec,
                positions_type,
                positions,
                atomic_numbers,
                schema,
                section_buffers,
                string_sections,
                n_mols,
                total_atoms,
            )))
        })
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let (
        n_atoms_vec,
        positions_type,
        positions,
        atomic_numbers,
        schema,
        section_buffers,
        string_results,
        _n_mols,
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
    dict.set_item("n_atoms", PyArray1::from_vec(py, n_atoms_vec))?;
    match positions_type {
        TYPE_VEC3_F32 => {
            let arr = cast_or_decode_f32(&positions)?;
            dict.set_item("positions", pyarray2_from_cow(py, arr, total_atoms, 3)?)?;
        }
        TYPE_VEC3_F64 => {
            let arr = cast_or_decode_f64(&positions)?;
            dict.set_item("positions", pyarray2_from_cow(py, arr, total_atoms, 3)?)?;
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "Unsupported positions type tag {}",
                other
            )));
        }
    }
    dict.set_item("atomic_numbers", PyArray1::from_vec(py, atomic_numbers))?;

    let atom_props_dict = PyDict::new(py);
    let mol_props_dict = PyDict::new(py);

    for ((s, buf), str_result) in schema
        .iter()
        .zip(section_buffers.into_iter())
        .zip(string_results.iter())
    {
        let target = match s.kind {
            KIND_ATOM_PROP => &atom_props_dict,
            KIND_MOL_PROP => &mol_props_dict,
            _ => &dict,
        };

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
