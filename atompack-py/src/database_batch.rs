use super::*;
use crate::molecule::{SoaBuiltinPayloads, SoaCustomSection, build_soa_record_with_custom};

struct BatchCustomColumn {
    key: String,
    kind: u8,
    type_tag: u8,
    slot_bytes: usize,
    payload: Vec<u8>,
    strings: Option<Vec<String>>,
}

impl BatchCustomColumn {
    fn section_for<'a>(&'a self, index: usize) -> SoaCustomSection<'a> {
        if let Some(strings) = &self.strings {
            return SoaCustomSection {
                kind: self.kind,
                key: self.key.as_str(),
                type_tag: self.type_tag,
                payload: strings[index].as_bytes(),
            };
        }
        let start = index * self.slot_bytes;
        let end = start + self.slot_bytes;
        SoaCustomSection {
            kind: self.kind,
            key: self.key.as_str(),
            type_tag: self.type_tag,
            payload: &self.payload[start..end],
        }
    }
}

fn reject_reserved_key(key: &str) -> PyResult<()> {
    if key == "stress" {
        return Err(PyValueError::new_err(
            "'stress' is a reserved field; use the builtin stress argument instead",
        ));
    }
    Ok(())
}

fn push_unique_key(
    seen_keys: &mut std::collections::HashSet<String>,
    key: &str,
    label: &str,
) -> PyResult<()> {
    if !seen_keys.insert(key.to_string()) {
        return Err(PyValueError::new_err(format!(
            "Duplicate custom key '{}' across batched {} inputs",
            key, label
        )));
    }
    Ok(())
}

fn scalar_f64_payload(values: &[f64]) -> Vec<u8> {
    bytemuck::cast_slice::<f64, u8>(values).to_vec()
}

fn scalar_i64_payload(values: &[i64]) -> Vec<u8> {
    bytemuck::cast_slice::<i64, u8>(values).to_vec()
}

fn extract_string_column(
    value: &Bound<'_, PyAny>,
    batch: usize,
    key: &str,
    kind: u8,
) -> PyResult<Option<BatchCustomColumn>> {
    let Ok(strings) = value.extract::<Vec<String>>() else {
        return Ok(None);
    };
    if strings.len() != batch {
        return Err(PyValueError::new_err(format!(
            "custom property '{}' length ({}) doesn't match batch size ({})",
            key,
            strings.len(),
            batch
        )));
    }
    Ok(Some(BatchCustomColumn {
        key: key.to_string(),
        kind,
        type_tag: TYPE_STRING,
        slot_bytes: 0,
        payload: Vec::new(),
        strings: Some(strings),
    }))
}

fn extract_scalar_column_f64<T: Element + Copy + Into<f64>>(
    arr: &Bound<'_, PyArray1<T>>,
    batch: usize,
    key: &str,
    kind: u8,
) -> PyResult<BatchCustomColumn> {
    let readonly = arr.readonly();
    let view = readonly.as_array();
    if view.len() != batch {
        return Err(PyValueError::new_err(format!(
            "custom property '{}' length ({}) doesn't match batch size ({})",
            key,
            view.len(),
            batch
        )));
    }
    let values: Vec<f64> = view.iter().copied().map(Into::into).collect();
    Ok(BatchCustomColumn {
        key: key.to_string(),
        kind,
        type_tag: TYPE_FLOAT,
        slot_bytes: std::mem::size_of::<f64>(),
        payload: scalar_f64_payload(&values),
        strings: None,
    })
}

fn extract_scalar_column_i64<T: Element + Copy + Into<i64>>(
    arr: &Bound<'_, PyArray1<T>>,
    batch: usize,
    key: &str,
    kind: u8,
) -> PyResult<BatchCustomColumn> {
    let readonly = arr.readonly();
    let view = readonly.as_array();
    if view.len() != batch {
        return Err(PyValueError::new_err(format!(
            "custom property '{}' length ({}) doesn't match batch size ({})",
            key,
            view.len(),
            batch
        )));
    }
    let values: Vec<i64> = view.iter().copied().map(Into::into).collect();
    Ok(BatchCustomColumn {
        key: key.to_string(),
        kind,
        type_tag: TYPE_INT,
        slot_bytes: std::mem::size_of::<i64>(),
        payload: scalar_i64_payload(&values),
        strings: None,
    })
}

fn extract_matrix_column<T: Element + Copy + bytemuck::NoUninit>(
    arr: &Bound<'_, PyArray2<T>>,
    batch: usize,
    key: &str,
    kind: u8,
    type_tag: u8,
) -> PyResult<BatchCustomColumn> {
    let readonly = arr.readonly();
    let view = readonly.as_array();
    let shape = view.shape();
    if shape.len() != 2 || shape[0] != batch {
        return Err(PyValueError::new_err(format!(
            "custom property '{}' must have shape ({}, k)",
            key, batch
        )));
    }
    let slice = readonly.as_slice().map_err(|_| {
        PyValueError::new_err(format!("custom property '{}' must be C-contiguous", key))
    })?;
    Ok(BatchCustomColumn {
        key: key.to_string(),
        kind,
        type_tag,
        slot_bytes: shape[1] * std::mem::size_of::<T>(),
        payload: bytemuck::cast_slice::<T, u8>(slice).to_vec(),
        strings: None,
    })
}

fn extract_vec3_column<T: Element + Copy + bytemuck::NoUninit>(
    arr: &Bound<'_, PyArray3<T>>,
    batch: usize,
    expected_rows: usize,
    key: &str,
    kind: u8,
    type_tag: u8,
    shape_label: &str,
) -> PyResult<BatchCustomColumn> {
    let readonly = arr.readonly();
    let view = readonly.as_array();
    let shape = view.shape();
    if shape.len() != 3 || shape[0] != batch || shape[1] != expected_rows || shape[2] != 3 {
        return Err(PyValueError::new_err(format!(
            "custom property '{}' must have shape ({}, {}, 3) for {}",
            key, batch, expected_rows, shape_label
        )));
    }
    let slice = readonly.as_slice().map_err(|_| {
        PyValueError::new_err(format!("custom property '{}' must be C-contiguous", key))
    })?;
    Ok(BatchCustomColumn {
        key: key.to_string(),
        kind,
        type_tag,
        slot_bytes: expected_rows * 3 * std::mem::size_of::<T>(),
        payload: bytemuck::cast_slice::<T, u8>(slice).to_vec(),
        strings: None,
    })
}

fn extract_property_column(
    value: &Bound<'_, PyAny>,
    batch: usize,
    key: &str,
    kind: u8,
) -> PyResult<Option<BatchCustomColumn>> {
    if let Some(column) = extract_string_column(value, batch, key, kind)? {
        return Ok(Some(column));
    }
    if let Ok(arr) = value.cast::<PyArray1<f64>>() {
        return Ok(Some(extract_scalar_column_f64(arr, batch, key, kind)?));
    }
    if let Ok(arr) = value.cast::<PyArray1<f32>>() {
        return Ok(Some(extract_scalar_column_f64(arr, batch, key, kind)?));
    }
    if let Ok(arr) = value.cast::<PyArray1<i64>>() {
        return Ok(Some(extract_scalar_column_i64(arr, batch, key, kind)?));
    }
    if let Ok(arr) = value.cast::<PyArray1<i32>>() {
        return Ok(Some(extract_scalar_column_i64(arr, batch, key, kind)?));
    }
    if let Ok(arr) = value.cast::<PyArray2<f64>>() {
        return Ok(Some(extract_matrix_column(
            arr,
            batch,
            key,
            kind,
            TYPE_F64_ARRAY,
        )?));
    }
    if let Ok(arr) = value.cast::<PyArray2<f32>>() {
        return Ok(Some(extract_matrix_column(
            arr,
            batch,
            key,
            kind,
            TYPE_F32_ARRAY,
        )?));
    }
    if let Ok(arr) = value.cast::<PyArray2<i64>>() {
        return Ok(Some(extract_matrix_column(
            arr,
            batch,
            key,
            kind,
            TYPE_I64_ARRAY,
        )?));
    }
    if let Ok(arr) = value.cast::<PyArray2<i32>>() {
        return Ok(Some(extract_matrix_column(
            arr,
            batch,
            key,
            kind,
            TYPE_I32_ARRAY,
        )?));
    }
    if let Ok(arr) = value.cast::<PyArray3<f64>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        let shape = view.shape();
        if shape.len() == 3 && shape[0] == batch && shape[2] == 3 {
            return Ok(Some(extract_vec3_column(
                arr,
                batch,
                shape[1],
                key,
                kind,
                TYPE_VEC3_F64,
                "molecule properties",
            )?));
        }
    }
    if let Ok(arr) = value.cast::<PyArray3<f32>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        let shape = view.shape();
        if shape.len() == 3 && shape[0] == batch && shape[2] == 3 {
            return Ok(Some(extract_vec3_column(
                arr,
                batch,
                shape[1],
                key,
                kind,
                TYPE_VEC3_F32,
                "molecule properties",
            )?));
        }
    }
    Ok(None)
}

fn extract_atom_property_column(
    value: &Bound<'_, PyAny>,
    batch: usize,
    n_atoms: usize,
    key: &str,
) -> PyResult<Option<BatchCustomColumn>> {
    if let Ok(arr) = value.cast::<PyArray2<f64>>() {
        let column = extract_matrix_column(arr, batch, key, KIND_ATOM_PROP, TYPE_F64_ARRAY)?;
        if column.slot_bytes != n_atoms * std::mem::size_of::<f64>() {
            return Err(PyValueError::new_err(format!(
                "atom property '{}' must have shape ({}, {})",
                key, batch, n_atoms
            )));
        }
        return Ok(Some(column));
    }
    if let Ok(arr) = value.cast::<PyArray2<f32>>() {
        let column = extract_matrix_column(arr, batch, key, KIND_ATOM_PROP, TYPE_F32_ARRAY)?;
        if column.slot_bytes != n_atoms * std::mem::size_of::<f32>() {
            return Err(PyValueError::new_err(format!(
                "atom property '{}' must have shape ({}, {})",
                key, batch, n_atoms
            )));
        }
        return Ok(Some(column));
    }
    if let Ok(arr) = value.cast::<PyArray2<i64>>() {
        let column = extract_matrix_column(arr, batch, key, KIND_ATOM_PROP, TYPE_I64_ARRAY)?;
        if column.slot_bytes != n_atoms * std::mem::size_of::<i64>() {
            return Err(PyValueError::new_err(format!(
                "atom property '{}' must have shape ({}, {})",
                key, batch, n_atoms
            )));
        }
        return Ok(Some(column));
    }
    if let Ok(arr) = value.cast::<PyArray2<i32>>() {
        let column = extract_matrix_column(arr, batch, key, KIND_ATOM_PROP, TYPE_I32_ARRAY)?;
        if column.slot_bytes != n_atoms * std::mem::size_of::<i32>() {
            return Err(PyValueError::new_err(format!(
                "atom property '{}' must have shape ({}, {})",
                key, batch, n_atoms
            )));
        }
        return Ok(Some(column));
    }
    if let Ok(arr) = value.cast::<PyArray3<f64>>() {
        return Ok(Some(extract_vec3_column(
            arr,
            batch,
            n_atoms,
            key,
            KIND_ATOM_PROP,
            TYPE_VEC3_F64,
            "atom properties",
        )?));
    }
    if let Ok(arr) = value.cast::<PyArray3<f32>>() {
        return Ok(Some(extract_vec3_column(
            arr,
            batch,
            n_atoms,
            key,
            KIND_ATOM_PROP,
            TYPE_VEC3_F32,
            "atom properties",
        )?));
    }
    Ok(None)
}

fn extract_custom_columns(
    properties: Option<&Bound<'_, PyDict>>,
    atom_properties: Option<&Bound<'_, PyDict>>,
    batch: usize,
    n_atoms: usize,
) -> PyResult<Vec<BatchCustomColumn>> {
    let mut columns = Vec::new();
    let mut seen_keys = std::collections::HashSet::new();

    if let Some(props) = properties {
        for (key, value) in props.iter() {
            let key = key.extract::<String>()?;
            reject_reserved_key(&key)?;
            push_unique_key(&mut seen_keys, &key, "property")?;
            let Some(column) = extract_property_column(&value, batch, &key, KIND_MOL_PROP)? else {
                return Err(PyValueError::new_err(format!(
                    "Unsupported batched property '{}' type. Supported: list[str], 1D numeric arrays, 2D numeric arrays, or 3D float arrays with trailing dimension 3",
                    key
                )));
            };
            columns.push(column);
        }
    }

    if let Some(props) = atom_properties {
        for (key, value) in props.iter() {
            let key = key.extract::<String>()?;
            reject_reserved_key(&key)?;
            push_unique_key(&mut seen_keys, &key, "atom property")?;
            let Some(column) = extract_atom_property_column(&value, batch, n_atoms, &key)? else {
                return Err(PyValueError::new_err(format!(
                    "Unsupported batched atom property '{}' type. Supported: 2D numeric arrays with shape (batch, n_atoms) or 3D float arrays with shape (batch, n_atoms, 3)",
                    key
                )));
            };
            columns.push(column);
        }
    }

    Ok(columns)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn add_arrays_batch_impl(
    inner: &mut AtomDatabase,
    py: Python<'_>,
    positions: &Bound<'_, PyArray3<f32>>,
    atomic_numbers: &Bound<'_, PyArray2<u8>>,
    energy: Option<&Bound<'_, PyArray1<f64>>>,
    forces: Option<&Bound<'_, PyArray3<f32>>>,
    charges: Option<&Bound<'_, PyArray2<f64>>>,
    velocities: Option<&Bound<'_, PyArray3<f32>>>,
    cell: Option<&Bound<'_, PyArray3<f64>>>,
    stress: Option<&Bound<'_, PyArray3<f64>>>,
    pbc: Option<&Bound<'_, PyArray2<bool>>>,
    name: Option<Vec<String>>,
    properties: Option<&Bound<'_, PyDict>>,
    atom_properties: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let pos = positions.readonly();
    let pos_arr = pos.as_array();
    let pos_shape = pos_arr.shape();
    if pos_shape.len() != 3 || pos_shape[2] != 3 {
        return Err(PyValueError::new_err(
            "positions must have shape (batch, n_atoms, 3)",
        ));
    }
    let batch = pos_shape[0];
    let n_atoms = pos_shape[1];
    let pos_slice = pos_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("positions must be C-contiguous"))?;

    let z = atomic_numbers.readonly();
    let z_arr = z.as_array();
    if z_arr.shape() != [batch, n_atoms] {
        return Err(PyValueError::new_err(format!(
            "atomic_numbers must have shape ({}, {})",
            batch, n_atoms
        )));
    }
    let z_slice = z_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("atomic_numbers must be C-contiguous"))?;

    let energy_ro = energy.map(|arr| arr.readonly());
    let energy_slice = if let Some(ro) = energy_ro.as_ref() {
        let view = ro.as_array();
        if view.len() != batch {
            return Err(PyValueError::new_err(format!(
                "energy length ({}) doesn't match batch size ({})",
                view.len(),
                batch
            )));
        }
        Some(
            ro.as_slice()
                .map_err(|_| PyValueError::new_err("energy must be C-contiguous"))?,
        )
    } else {
        None
    };

    let forces_ro = forces.map(|arr| arr.readonly());
    let forces_slice = if let Some(ro) = forces_ro.as_ref() {
        let view = ro.as_array();
        if view.shape() != [batch, n_atoms, 3] {
            return Err(PyValueError::new_err(format!(
                "forces must have shape ({}, {}, 3)",
                batch, n_atoms
            )));
        }
        Some(
            ro.as_slice()
                .map_err(|_| PyValueError::new_err("forces must be C-contiguous"))?,
        )
    } else {
        None
    };

    let charges_ro = charges.map(|arr| arr.readonly());
    let charges_slice = if let Some(ro) = charges_ro.as_ref() {
        let view = ro.as_array();
        if view.shape() != [batch, n_atoms] {
            return Err(PyValueError::new_err(format!(
                "charges must have shape ({}, {})",
                batch, n_atoms
            )));
        }
        Some(
            ro.as_slice()
                .map_err(|_| PyValueError::new_err("charges must be C-contiguous"))?,
        )
    } else {
        None
    };

    let velocities_ro = velocities.map(|arr| arr.readonly());
    let velocities_slice = if let Some(ro) = velocities_ro.as_ref() {
        let view = ro.as_array();
        if view.shape() != [batch, n_atoms, 3] {
            return Err(PyValueError::new_err(format!(
                "velocities must have shape ({}, {}, 3)",
                batch, n_atoms
            )));
        }
        Some(
            ro.as_slice()
                .map_err(|_| PyValueError::new_err("velocities must be C-contiguous"))?,
        )
    } else {
        None
    };

    let cell_ro = cell.map(|arr| arr.readonly());
    let cell_slice = if let Some(ro) = cell_ro.as_ref() {
        let view = ro.as_array();
        if view.shape() != [batch, 3, 3] {
            return Err(PyValueError::new_err(format!(
                "cell must have shape ({}, 3, 3)",
                batch
            )));
        }
        Some(
            ro.as_slice()
                .map_err(|_| PyValueError::new_err("cell must be C-contiguous"))?,
        )
    } else {
        None
    };

    let stress_ro = stress.map(|arr| arr.readonly());
    let stress_slice = if let Some(ro) = stress_ro.as_ref() {
        let view = ro.as_array();
        if view.shape() != [batch, 3, 3] {
            return Err(PyValueError::new_err(format!(
                "stress must have shape ({}, 3, 3)",
                batch
            )));
        }
        Some(
            ro.as_slice()
                .map_err(|_| PyValueError::new_err("stress must be C-contiguous"))?,
        )
    } else {
        None
    };

    let pbc_ro = pbc.map(|arr| arr.readonly());
    let pbc_slice = if let Some(ro) = pbc_ro.as_ref() {
        let view = ro.as_array();
        if view.shape() != [batch, 3] {
            return Err(PyValueError::new_err(format!(
                "pbc must have shape ({}, 3)",
                batch
            )));
        }
        Some(
            ro.as_slice()
                .map_err(|_| PyValueError::new_err("pbc must be C-contiguous"))?,
        )
    } else {
        None
    };

    if let Some(names) = &name
        && names.len() != batch
    {
        return Err(PyValueError::new_err(format!(
            "name length ({}) doesn't match batch size ({})",
            names.len(),
            batch
        )));
    }

    let custom_columns = extract_custom_columns(properties, atom_properties, batch, n_atoms)?;

    let build_record = |i: usize| {
        let pos_start = i * n_atoms * 3;
        let pos_end = pos_start + n_atoms * 3;
        let z_start = i * n_atoms;
        let z_end = z_start + n_atoms;
        let forces_payload = forces_slice
            .as_ref()
            .map(|slice| bytemuck::cast_slice::<f32, u8>(&slice[pos_start..pos_end]));
        let charges_payload = charges_slice
            .as_ref()
            .map(|slice| bytemuck::cast_slice::<f64, u8>(&slice[z_start..z_end]));
        let velocities_payload = velocities_slice
            .as_ref()
            .map(|slice| bytemuck::cast_slice::<f32, u8>(&slice[pos_start..pos_end]));
        let mat_start = i * 9;
        let mat_end = mat_start + 9;
        let cell_payload = cell_slice
            .as_ref()
            .map(|slice| bytemuck::cast_slice::<f64, u8>(&slice[mat_start..mat_end]));
        let stress_payload = stress_slice
            .as_ref()
            .map(|slice| bytemuck::cast_slice::<f64, u8>(&slice[mat_start..mat_end]));
        let pbc_value = pbc_slice
            .as_ref()
            .map(|slice| [slice[i * 3], slice[i * 3 + 1], slice[i * 3 + 2]]);
        let name_value = name.as_ref().map(|names| names[i].as_str());
        let custom_sections: Vec<SoaCustomSection<'_>> = custom_columns
            .iter()
            .map(|column| column.section_for(i))
            .collect();

        build_soa_record_with_custom(
            &pos_slice[pos_start..pos_end],
            &z_slice[z_start..z_end],
            SoaBuiltinPayloads {
                energy: energy_slice.as_ref().map(|slice| slice[i]),
                forces: forces_payload,
                charges: charges_payload,
                velocities: velocities_payload,
                cell: cell_payload,
                stress: stress_payload,
                pbc: pbc_value,
                name: name_value,
            },
            &custom_sections,
        )
        .map(|record| record.into_parts())
    };

    let serialized: Vec<(Vec<u8>, u32)> = if batch >= 1024 {
        use rayon::prelude::*;
        (0..batch)
            .into_par_iter()
            .map(build_record)
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?
    } else {
        (0..batch)
            .map(build_record)
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?
    };

    py.detach(|| inner.add_owned_soa_records(serialized))
        .map_err(|e| PyValueError::new_err(format!("{}", e)))
}
