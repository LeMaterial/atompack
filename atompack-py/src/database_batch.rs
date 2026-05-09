use super::*;
use crate::molecule::{SoaRecord, SoaSection, build_soa_record};

struct BatchSectionColumn {
    key: String,
    kind: u8,
    type_tag: u8,
    slot_bytes: usize,
    payload: Vec<u8>,
    strings: Option<Vec<String>>,
}

impl BatchSectionColumn {
    fn section_for<'a>(&'a self, index: usize) -> SoaSection<'a> {
        if let Some(strings) = &self.strings {
            return SoaSection {
                kind: self.kind,
                key: self.key.as_str(),
                type_tag: self.type_tag,
                payload: strings[index].as_bytes(),
            };
        }
        let start = index * self.slot_bytes;
        let end = start + self.slot_bytes;
        SoaSection {
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
) -> PyResult<Option<BatchSectionColumn>> {
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
    Ok(Some(BatchSectionColumn {
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
) -> PyResult<BatchSectionColumn> {
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
    Ok(BatchSectionColumn {
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
) -> PyResult<BatchSectionColumn> {
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
    Ok(BatchSectionColumn {
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
) -> PyResult<BatchSectionColumn> {
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
    Ok(BatchSectionColumn {
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
) -> PyResult<BatchSectionColumn> {
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
    Ok(BatchSectionColumn {
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
) -> PyResult<Option<BatchSectionColumn>> {
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
) -> PyResult<Option<BatchSectionColumn>> {
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
) -> PyResult<Vec<BatchSectionColumn>> {
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

fn extract_positions_payload(value: &Bound<'_, PyAny>) -> PyResult<(usize, usize, u8, Vec<u8>)> {
    if let Ok(arr) = value.cast::<PyArray3<f32>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        if view.shape().len() != 3 || view.shape()[2] != 3 {
            return Err(PyValueError::new_err(
                "positions must have shape (batch, n_atoms, 3)",
            ));
        }
        let slice = readonly
            .as_slice()
            .map_err(|_| PyValueError::new_err("positions must be C-contiguous"))?;
        return Ok((
            view.shape()[0],
            view.shape()[1],
            TYPE_VEC3_F32,
            bytemuck::cast_slice::<f32, u8>(slice).to_vec(),
        ));
    }
    if let Ok(arr) = value.cast::<PyArray3<f64>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        if view.shape().len() != 3 || view.shape()[2] != 3 {
            return Err(PyValueError::new_err(
                "positions must have shape (batch, n_atoms, 3)",
            ));
        }
        let slice = readonly
            .as_slice()
            .map_err(|_| PyValueError::new_err("positions must be C-contiguous"))?;
        return Ok((
            view.shape()[0],
            view.shape()[1],
            TYPE_VEC3_F64,
            bytemuck::cast_slice::<f64, u8>(slice).to_vec(),
        ));
    }
    Err(PyValueError::new_err(
        "positions must be a float32 or float64 ndarray with shape (batch, n_atoms, 3)",
    ))
}

fn extract_atomic_numbers_payload(
    atomic_numbers: &Bound<'_, PyArray2<u8>>,
    batch: usize,
    n_atoms: usize,
) -> PyResult<Vec<u8>> {
    let readonly = atomic_numbers.readonly();
    let view = readonly.as_array();
    if view.shape() != [batch, n_atoms] {
        return Err(PyValueError::new_err(format!(
            "atomic_numbers must have shape ({}, {})",
            batch, n_atoms
        )));
    }
    Ok(readonly
        .as_slice()
        .map_err(|_| PyValueError::new_err("atomic_numbers must be C-contiguous"))?
        .to_vec())
}

fn extract_builtin_scalar_column<T: Element + Copy + bytemuck::NoUninit>(
    arr: &Bound<'_, PyArray1<T>>,
    batch: usize,
    key: &str,
    type_tag: u8,
) -> PyResult<BatchSectionColumn> {
    let readonly = arr.readonly();
    let view = readonly.as_array();
    if view.len() != batch {
        return Err(PyValueError::new_err(format!(
            "{} length ({}) doesn't match batch size ({})",
            key,
            view.len(),
            batch
        )));
    }
    let slice = readonly
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{} must be C-contiguous", key)))?;
    Ok(BatchSectionColumn {
        key: key.to_string(),
        kind: KIND_BUILTIN,
        type_tag,
        slot_bytes: std::mem::size_of::<T>(),
        payload: bytemuck::cast_slice::<T, u8>(slice).to_vec(),
        strings: None,
    })
}

fn extract_builtin_float_array_column<T: Element + Copy + bytemuck::NoUninit>(
    arr: &Bound<'_, PyArray2<T>>,
    batch: usize,
    n_atoms: usize,
    key: &str,
    type_tag: u8,
) -> PyResult<BatchSectionColumn> {
    let readonly = arr.readonly();
    let view = readonly.as_array();
    if view.shape() != [batch, n_atoms] {
        return Err(PyValueError::new_err(format!(
            "{} must have shape ({}, {})",
            key, batch, n_atoms
        )));
    }
    let slice = readonly
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{} must be C-contiguous", key)))?;
    Ok(BatchSectionColumn {
        key: key.to_string(),
        kind: KIND_BUILTIN,
        type_tag,
        slot_bytes: n_atoms * std::mem::size_of::<T>(),
        payload: bytemuck::cast_slice::<T, u8>(slice).to_vec(),
        strings: None,
    })
}

fn extract_builtin_vec3_column<T: Element + Copy + bytemuck::NoUninit>(
    arr: &Bound<'_, PyArray3<T>>,
    batch: usize,
    n_atoms: usize,
    key: &str,
    type_tag: u8,
) -> PyResult<BatchSectionColumn> {
    let readonly = arr.readonly();
    let view = readonly.as_array();
    if view.shape() != [batch, n_atoms, 3] {
        return Err(PyValueError::new_err(format!(
            "{} must have shape ({}, {}, 3)",
            key, batch, n_atoms
        )));
    }
    let slice = readonly
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{} must be C-contiguous", key)))?;
    Ok(BatchSectionColumn {
        key: key.to_string(),
        kind: KIND_BUILTIN,
        type_tag,
        slot_bytes: n_atoms * 3 * std::mem::size_of::<T>(),
        payload: bytemuck::cast_slice::<T, u8>(slice).to_vec(),
        strings: None,
    })
}

fn extract_builtin_mat3_column<T: Element + Copy + bytemuck::NoUninit>(
    arr: &Bound<'_, PyArray3<T>>,
    batch: usize,
    key: &str,
    type_tag: u8,
) -> PyResult<BatchSectionColumn> {
    let readonly = arr.readonly();
    let view = readonly.as_array();
    if view.shape() != [batch, 3, 3] {
        return Err(PyValueError::new_err(format!(
            "{} must have shape ({}, 3, 3)",
            key, batch
        )));
    }
    let slice = readonly
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{} must be C-contiguous", key)))?;
    Ok(BatchSectionColumn {
        key: key.to_string(),
        kind: KIND_BUILTIN,
        type_tag,
        slot_bytes: 9 * std::mem::size_of::<T>(),
        payload: bytemuck::cast_slice::<T, u8>(slice).to_vec(),
        strings: None,
    })
}

fn extract_builtin_pbc_column(
    pbc: &Bound<'_, PyArray2<bool>>,
    batch: usize,
) -> PyResult<BatchSectionColumn> {
    let readonly = pbc.readonly();
    let view = readonly.as_array();
    if view.shape() != [batch, 3] {
        return Err(PyValueError::new_err(format!(
            "pbc must have shape ({}, 3)",
            batch
        )));
    }
    Ok(BatchSectionColumn {
        key: "pbc".to_string(),
        kind: KIND_BUILTIN,
        type_tag: TYPE_BOOL3,
        slot_bytes: 3,
        payload: view.iter().map(|value| u8::from(*value)).collect(),
        strings: None,
    })
}

fn extract_builtin_name_column(
    name: Option<Vec<String>>,
    batch: usize,
) -> PyResult<Option<BatchSectionColumn>> {
    let Some(names) = name else {
        return Ok(None);
    };
    if names.len() != batch {
        return Err(PyValueError::new_err(format!(
            "name length ({}) doesn't match batch size ({})",
            names.len(),
            batch
        )));
    }
    Ok(Some(BatchSectionColumn {
        key: "name".to_string(),
        kind: KIND_BUILTIN,
        type_tag: TYPE_STRING,
        slot_bytes: 0,
        payload: Vec::new(),
        strings: Some(names),
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn add_arrays_batch_impl(
    inner: &mut AtomDatabase,
    py: Python<'_>,
    positions: &Bound<'_, PyAny>,
    atomic_numbers: &Bound<'_, PyArray2<u8>>,
    energy: Option<&Bound<'_, PyAny>>,
    forces: Option<&Bound<'_, PyAny>>,
    charges: Option<&Bound<'_, PyAny>>,
    velocities: Option<&Bound<'_, PyAny>>,
    cell: Option<&Bound<'_, PyAny>>,
    stress: Option<&Bound<'_, PyAny>>,
    pbc: Option<&Bound<'_, PyArray2<bool>>>,
    name: Option<Vec<String>>,
    properties: Option<&Bound<'_, PyDict>>,
    atom_properties: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let (batch, n_atoms, positions_type, positions_payload) = extract_positions_payload(positions)?;
    let atomic_numbers_payload = extract_atomic_numbers_payload(atomic_numbers, batch, n_atoms)?;

    let mut builtin_columns = Vec::new();
    if let Some(energy) = energy {
        if let Ok(arr) = energy.cast::<PyArray1<f32>>() {
            builtin_columns.push(extract_builtin_scalar_column(
                arr,
                batch,
                "energy",
                TYPE_FLOAT32,
            )?);
        } else if let Ok(arr) = energy.cast::<PyArray1<f64>>() {
            builtin_columns.push(extract_builtin_scalar_column(
                arr, batch, "energy", TYPE_FLOAT,
            )?);
        } else {
            return Err(PyValueError::new_err(
                "energy must be a float32 or float64 ndarray with shape (batch,)",
            ));
        }
    }
    if let Some(forces) = forces {
        if let Ok(arr) = forces.cast::<PyArray3<f32>>() {
            builtin_columns.push(extract_builtin_vec3_column(
                arr,
                batch,
                n_atoms,
                "forces",
                TYPE_VEC3_F32,
            )?);
        } else if let Ok(arr) = forces.cast::<PyArray3<f64>>() {
            builtin_columns.push(extract_builtin_vec3_column(
                arr,
                batch,
                n_atoms,
                "forces",
                TYPE_VEC3_F64,
            )?);
        } else {
            return Err(PyValueError::new_err(
                "forces must be a float32 or float64 ndarray with shape (batch, n_atoms, 3)",
            ));
        }
    }
    if let Some(charges) = charges {
        if let Ok(arr) = charges.cast::<PyArray2<f32>>() {
            builtin_columns.push(extract_builtin_float_array_column(
                arr,
                batch,
                n_atoms,
                "charges",
                TYPE_F32_ARRAY,
            )?);
        } else if let Ok(arr) = charges.cast::<PyArray2<f64>>() {
            builtin_columns.push(extract_builtin_float_array_column(
                arr,
                batch,
                n_atoms,
                "charges",
                TYPE_F64_ARRAY,
            )?);
        } else {
            return Err(PyValueError::new_err(
                "charges must be a float32 or float64 ndarray with shape (batch, n_atoms)",
            ));
        }
    }
    if let Some(velocities) = velocities {
        if let Ok(arr) = velocities.cast::<PyArray3<f32>>() {
            builtin_columns.push(extract_builtin_vec3_column(
                arr,
                batch,
                n_atoms,
                "velocities",
                TYPE_VEC3_F32,
            )?);
        } else if let Ok(arr) = velocities.cast::<PyArray3<f64>>() {
            builtin_columns.push(extract_builtin_vec3_column(
                arr,
                batch,
                n_atoms,
                "velocities",
                TYPE_VEC3_F64,
            )?);
        } else {
            return Err(PyValueError::new_err(
                "velocities must be a float32 or float64 ndarray with shape (batch, n_atoms, 3)",
            ));
        }
    }
    if let Some(cell) = cell {
        if let Ok(arr) = cell.cast::<PyArray3<f32>>() {
            builtin_columns.push(extract_builtin_mat3_column(
                arr,
                batch,
                "cell",
                TYPE_MAT3X3_F32,
            )?);
        } else if let Ok(arr) = cell.cast::<PyArray3<f64>>() {
            builtin_columns.push(extract_builtin_mat3_column(
                arr,
                batch,
                "cell",
                TYPE_MAT3X3_F64,
            )?);
        } else {
            return Err(PyValueError::new_err(
                "cell must be a float32 or float64 ndarray with shape (batch, 3, 3)",
            ));
        }
    }
    if let Some(stress) = stress {
        if let Ok(arr) = stress.cast::<PyArray3<f32>>() {
            builtin_columns.push(extract_builtin_mat3_column(
                arr,
                batch,
                "stress",
                TYPE_MAT3X3_F32,
            )?);
        } else if let Ok(arr) = stress.cast::<PyArray3<f64>>() {
            builtin_columns.push(extract_builtin_mat3_column(
                arr,
                batch,
                "stress",
                TYPE_MAT3X3_F64,
            )?);
        } else {
            return Err(PyValueError::new_err(
                "stress must be a float32 or float64 ndarray with shape (batch, 3, 3)",
            ));
        }
    }
    if let Some(pbc) = pbc {
        builtin_columns.push(extract_builtin_pbc_column(pbc, batch)?);
    }
    if let Some(name) = extract_builtin_name_column(name, batch)? {
        builtin_columns.push(name);
    }

    let custom_columns = extract_custom_columns(properties, atom_properties, batch, n_atoms)?;
    let positions_slot_bytes = n_atoms
        .checked_mul(type_tag_elem_bytes(positions_type))
        .ok_or_else(|| PyValueError::new_err("positions byte length overflow"))?;
    let record_format = inner.record_format();

    let mut records = Vec::with_capacity(batch);
    for index in 0..batch {
        let pos_start = index * positions_slot_bytes;
        let pos_end = pos_start + positions_slot_bytes;
        let z_start = index * n_atoms;
        let z_end = z_start + n_atoms;

        let mut sections = Vec::with_capacity(builtin_columns.len() + custom_columns.len());
        sections.extend(
            builtin_columns
                .iter()
                .map(|column| column.section_for(index)),
        );
        sections.extend(
            custom_columns
                .iter()
                .map(|column| column.section_for(index)),
        );

        let record = build_soa_record(SoaRecord {
            record_format,
            positions_type,
            positions: &positions_payload[pos_start..pos_end],
            atomic_numbers: &atomic_numbers_payload[z_start..z_end],
            sections: &sections,
        })
        .map_err(PyValueError::new_err)?;
        records.push((record, n_atoms as u32));
    }

    py.detach(move || inner.add_owned_soa_records(records))
        .map_err(|e| PyValueError::new_err(format!("{}", e)))
}
