use super::*;
use crate::molecule::{SoaRecord, SoaSection, build_soa_record};
use atompack::storage::{DatabaseSchema, DatabaseSchemaSection};

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

fn batch_section_is_per_atom(kind: u8, key: &str) -> bool {
    match kind {
        KIND_ATOM_PROP => true,
        KIND_MOL_PROP => false,
        KIND_BUILTIN => matches!(key, "forces" | "charges" | "velocities"),
        _ => false,
    }
}

fn database_schema_section(column: &BatchSectionColumn) -> PyResult<DatabaseSchemaSection> {
    let per_atom = batch_section_is_per_atom(column.kind, &column.key);
    let elem_bytes = if column.type_tag == TYPE_STRING {
        0
    } else {
        let elem_bytes = type_tag_elem_bytes(column.type_tag);
        if elem_bytes == 0 {
            return Err(PyValueError::new_err(format!(
                "Unsupported section type tag {} for '{}'",
                column.type_tag, column.key
            )));
        }
        elem_bytes
    };
    let slot_bytes = if column.type_tag == TYPE_STRING {
        0
    } else if per_atom {
        elem_bytes
    } else {
        column.slot_bytes
    };

    Ok(DatabaseSchemaSection {
        kind: column.kind,
        key: column.key.clone(),
        type_tag: column.type_tag,
        per_atom,
        elem_bytes,
        slot_bytes,
    })
}

fn build_batch_schema<'a, I>(positions_type: u8, columns: I) -> PyResult<DatabaseSchema>
where
    I: IntoIterator<Item = &'a BatchSectionColumn>,
{
    let sections = columns
        .into_iter()
        .map(database_schema_section)
        .collect::<PyResult<Vec<_>>>()?;
    Ok(DatabaseSchema {
        positions_type: Some(positions_type),
        sections,
    })
}

fn push_builtin_section<'a>(
    sections: &mut Vec<SoaSection<'a>>,
    key: &'a str,
    type_tag: u8,
    payload: &'a [u8],
) {
    sections.push(SoaSection {
        kind: KIND_BUILTIN,
        key,
        type_tag,
        payload,
    });
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
    if let Some(arr) = PyFloatArray1::from_any(value) {
        return Ok(Some(match arr {
            PyFloatArray1::F32(arr) => extract_scalar_column_f64(&arr, batch, key, kind)?,
            PyFloatArray1::F64(arr) => extract_scalar_column_f64(&arr, batch, key, kind)?,
        }));
    }
    if let Some(arr) = PyIntArray1::from_any(value) {
        return Ok(Some(match arr {
            PyIntArray1::I32(arr) => extract_scalar_column_i64(&arr, batch, key, kind)?,
            PyIntArray1::I64(arr) => extract_scalar_column_i64(&arr, batch, key, kind)?,
        }));
    }
    if let Some(arr) = PyFloatArray2::from_any(value) {
        return Ok(Some(match arr {
            PyFloatArray2::F32(arr) => {
                extract_matrix_column(&arr, batch, key, kind, TYPE_F32_ARRAY)?
            }
            PyFloatArray2::F64(arr) => {
                extract_matrix_column(&arr, batch, key, kind, TYPE_F64_ARRAY)?
            }
        }));
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
    if let Some(arr) = PyFloatArray3::from_any(value) {
        match arr {
            PyFloatArray3::F32(arr) => {
                let readonly = arr.readonly();
                let view = readonly.as_array();
                let shape = view.shape();
                if shape.len() == 3 && shape[0] == batch && shape[2] == 3 {
                    return Ok(Some(extract_vec3_column(
                        &arr,
                        batch,
                        shape[1],
                        key,
                        kind,
                        TYPE_VEC3_F32,
                        "molecule properties",
                    )?));
                }
            }
            PyFloatArray3::F64(arr) => {
                let readonly = arr.readonly();
                let view = readonly.as_array();
                let shape = view.shape();
                if shape.len() == 3 && shape[0] == batch && shape[2] == 3 {
                    return Ok(Some(extract_vec3_column(
                        &arr,
                        batch,
                        shape[1],
                        key,
                        kind,
                        TYPE_VEC3_F64,
                        "molecule properties",
                    )?));
                }
            }
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
    if let Some(arr) = PyFloatArray2::from_any(value) {
        let (column, expected) = match arr {
            PyFloatArray2::F32(arr) => (
                extract_matrix_column(&arr, batch, key, KIND_ATOM_PROP, TYPE_F32_ARRAY)?,
                n_atoms * std::mem::size_of::<f32>(),
            ),
            PyFloatArray2::F64(arr) => (
                extract_matrix_column(&arr, batch, key, KIND_ATOM_PROP, TYPE_F64_ARRAY)?,
                n_atoms * std::mem::size_of::<f64>(),
            ),
        };
        if column.slot_bytes != expected {
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
    if let Some(arr) = PyFloatArray3::from_any(value) {
        return Ok(Some(match arr {
            PyFloatArray3::F32(arr) => extract_vec3_column(
                &arr,
                batch,
                n_atoms,
                key,
                KIND_ATOM_PROP,
                TYPE_VEC3_F32,
                "atom properties",
            )?,
            PyFloatArray3::F64(arr) => extract_vec3_column(
                &arr,
                batch,
                n_atoms,
                key,
                KIND_ATOM_PROP,
                TYPE_VEC3_F64,
                "atom properties",
            )?,
        }));
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

struct FastMat3Column {
    type_tag: u8,
    slot_bytes: usize,
    payload: Vec<u8>,
}

impl FastMat3Column {
    fn from_optional(
        value: Option<&Bound<'_, PyAny>>,
        batch: usize,
        label: &str,
    ) -> PyResult<Option<Self>> {
        let Some(value) = value else {
            return Ok(None);
        };
        if let Some(arr) = PyFloatArray3::from_any(value) {
            match arr {
                PyFloatArray3::F32(arr) => {
                    let ro = arr.readonly();
                    if ro.as_array().shape() != [batch, 3, 3] {
                        return Err(PyValueError::new_err(format!(
                            "{label} must have shape ({}, 3, 3)",
                            batch
                        )));
                    }
                    let slice = ro.as_slice().map_err(|_| {
                        PyValueError::new_err(format!("{label} must be C-contiguous"))
                    })?;
                    return Ok(Some(Self {
                        type_tag: TYPE_MAT3X3_F32,
                        slot_bytes: 36,
                        payload: bytemuck::cast_slice::<f32, u8>(slice).to_vec(),
                    }));
                }
                PyFloatArray3::F64(arr) => {
                    let ro = arr.readonly();
                    if ro.as_array().shape() != [batch, 3, 3] {
                        return Err(PyValueError::new_err(format!(
                            "{label} must have shape ({}, 3, 3)",
                            batch
                        )));
                    }
                    let slice = ro.as_slice().map_err(|_| {
                        PyValueError::new_err(format!("{label} must be C-contiguous"))
                    })?;
                    return Ok(Some(Self {
                        type_tag: TYPE_MAT3X3_F64,
                        slot_bytes: 72,
                        payload: bytemuck::cast_slice::<f64, u8>(slice).to_vec(),
                    }));
                }
            }
        }
        Ok(None)
    }

    fn type_tag(&self) -> u8 {
        self.type_tag
    }

    fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    fn payload_bytes(&self, index: usize) -> &[u8] {
        let start = index * self.slot_bytes;
        let end = start + self.slot_bytes;
        &self.payload[start..end]
    }
}

#[allow(clippy::too_many_arguments)]
fn try_add_arrays_batch_fast_canonical(
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
) -> PyResult<bool> {
    let Some(PyFloatArray3::F32(positions)) = PyFloatArray3::from_any(positions) else {
        return Ok(false);
    };
    let energy = match energy {
        Some(value) => {
            let Some(PyFloatArray1::F64(arr)) = PyFloatArray1::from_any(value) else {
                return Ok(false);
            };
            Some(arr)
        }
        None => None,
    };
    let forces = match forces {
        Some(value) => {
            let Some(PyFloatArray3::F32(arr)) = PyFloatArray3::from_any(value) else {
                return Ok(false);
            };
            Some(arr)
        }
        None => None,
    };
    let charges = match charges {
        Some(value) => {
            let Some(PyFloatArray2::F64(arr)) = PyFloatArray2::from_any(value) else {
                return Ok(false);
            };
            Some(arr)
        }
        None => None,
    };
    let velocities = match velocities {
        Some(value) => {
            let Some(PyFloatArray3::F32(arr)) = PyFloatArray3::from_any(value) else {
                return Ok(false);
            };
            Some(arr)
        }
        None => None,
    };

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

    let cell_slice = match FastMat3Column::from_optional(cell, batch, "cell")? {
        Some(column) => Some(column),
        None if cell.is_some() => return Ok(false),
        None => None,
    };

    let stress_slice = match FastMat3Column::from_optional(stress, batch, "stress")? {
        Some(column) => Some(column),
        None if stress.is_some() => return Ok(false),
        None => None,
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
    let mut builtin_columns = Vec::new();
    if energy_slice.is_some() {
        builtin_columns.push(BatchSectionColumn {
            key: "energy".to_string(),
            kind: KIND_BUILTIN,
            type_tag: TYPE_FLOAT,
            slot_bytes: 8,
            payload: Vec::new(),
            strings: None,
        });
    }
    if forces_slice.is_some() {
        builtin_columns.push(BatchSectionColumn {
            key: "forces".to_string(),
            kind: KIND_BUILTIN,
            type_tag: TYPE_VEC3_F32,
            slot_bytes: n_atoms * 12,
            payload: Vec::new(),
            strings: None,
        });
    }
    if charges_slice.is_some() {
        builtin_columns.push(BatchSectionColumn {
            key: "charges".to_string(),
            kind: KIND_BUILTIN,
            type_tag: TYPE_F64_ARRAY,
            slot_bytes: n_atoms * 8,
            payload: Vec::new(),
            strings: None,
        });
    }
    if velocities_slice.is_some() {
        builtin_columns.push(BatchSectionColumn {
            key: "velocities".to_string(),
            kind: KIND_BUILTIN,
            type_tag: TYPE_VEC3_F32,
            slot_bytes: n_atoms * 12,
            payload: Vec::new(),
            strings: None,
        });
    }
    if let Some(column) = cell_slice.as_ref() {
        builtin_columns.push(BatchSectionColumn {
            key: "cell".to_string(),
            kind: KIND_BUILTIN,
            type_tag: column.type_tag(),
            slot_bytes: column.slot_bytes(),
            payload: Vec::new(),
            strings: None,
        });
    }
    if let Some(column) = stress_slice.as_ref() {
        builtin_columns.push(BatchSectionColumn {
            key: "stress".to_string(),
            kind: KIND_BUILTIN,
            type_tag: column.type_tag(),
            slot_bytes: column.slot_bytes(),
            payload: Vec::new(),
            strings: None,
        });
    }
    if pbc_slice.is_some() {
        builtin_columns.push(BatchSectionColumn {
            key: "pbc".to_string(),
            kind: KIND_BUILTIN,
            type_tag: TYPE_BOOL3,
            slot_bytes: 3,
            payload: Vec::new(),
            strings: None,
        });
    }
    if name.is_some() {
        builtin_columns.push(BatchSectionColumn {
            key: "name".to_string(),
            kind: KIND_BUILTIN,
            type_tag: TYPE_STRING,
            slot_bytes: 0,
            payload: Vec::new(),
            strings: None,
        });
    }

    let batch_schema = build_batch_schema(
        TYPE_VEC3_F32,
        builtin_columns.iter().chain(custom_columns.iter()),
    )?;
    let record_format = inner
        .record_format_for_schema(batch_schema.clone())
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let builtin_section_count = usize::from(energy_slice.is_some())
        + usize::from(forces_slice.is_some())
        + usize::from(charges_slice.is_some())
        + usize::from(velocities_slice.is_some())
        + usize::from(cell_slice.is_some())
        + usize::from(stress_slice.is_some())
        + usize::from(pbc_slice.is_some())
        + usize::from(name.is_some());

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
        let cell_payload = cell_slice.as_ref().map(|column| column.payload_bytes(i));
        let stress_payload = stress_slice.as_ref().map(|column| column.payload_bytes(i));
        let energy_bytes = energy_slice.as_ref().map(|slice| slice[i].to_le_bytes());
        let pbc_payload = pbc_slice.as_ref().map(|slice| {
            [
                slice[i * 3] as u8,
                slice[i * 3 + 1] as u8,
                slice[i * 3 + 2] as u8,
            ]
        });

        let mut sections = Vec::with_capacity(builtin_section_count + custom_columns.len());
        if let Some(payload) = charges_payload {
            push_builtin_section(&mut sections, "charges", TYPE_F64_ARRAY, payload);
        }
        if let Some(payload) = cell_payload {
            push_builtin_section(
                &mut sections,
                "cell",
                cell_slice
                    .as_ref()
                    .map(FastMat3Column::type_tag)
                    .expect("cell type tag must exist when payload exists"),
                payload,
            );
        }
        if let Some(bytes) = energy_bytes.as_ref() {
            push_builtin_section(&mut sections, "energy", TYPE_FLOAT, bytes);
        }
        if let Some(payload) = forces_payload {
            push_builtin_section(&mut sections, "forces", TYPE_VEC3_F32, payload);
        }
        if let Some(names) = name.as_ref() {
            push_builtin_section(&mut sections, "name", TYPE_STRING, names[i].as_bytes());
        }
        if let Some(payload) = pbc_payload.as_ref() {
            push_builtin_section(&mut sections, "pbc", TYPE_BOOL3, payload);
        }
        if let Some(payload) = stress_payload {
            push_builtin_section(
                &mut sections,
                "stress",
                stress_slice
                    .as_ref()
                    .map(FastMat3Column::type_tag)
                    .expect("stress type tag must exist when payload exists"),
                payload,
            );
        }
        if let Some(payload) = velocities_payload {
            push_builtin_section(&mut sections, "velocities", TYPE_VEC3_F32, payload);
        }
        sections.extend(custom_columns.iter().map(|column| column.section_for(i)));

        build_soa_record(SoaRecord {
            record_format,
            positions_type: TYPE_VEC3_F32,
            positions: bytemuck::cast_slice::<f32, u8>(&pos_slice[pos_start..pos_end]),
            atomic_numbers: &z_slice[z_start..z_end],
            sections: &sections,
        })
        .map(|record| (record, n_atoms as u32))
    };

    let records: Vec<(Vec<u8>, u32)> = if batch >= 1024 {
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

    py.detach(move || inner.add_owned_soa_records_with_schema(records, batch_schema))
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    Ok(true)
}

fn extract_positions_payload(value: &Bound<'_, PyAny>) -> PyResult<(usize, usize, u8, Vec<u8>)> {
    if let Some(arr) = PyFloatArray3::from_any(value) {
        match arr {
            PyFloatArray3::F32(arr) => {
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
            PyFloatArray3::F64(arr) => {
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
        }
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
    if try_add_arrays_batch_fast_canonical(
        inner,
        py,
        positions,
        atomic_numbers,
        energy,
        forces,
        charges,
        velocities,
        cell,
        stress,
        pbc,
        name.clone(),
        properties,
        atom_properties,
    )? {
        return Ok(());
    }

    let (batch, n_atoms, positions_type, positions_payload) = extract_positions_payload(positions)?;
    let atomic_numbers_payload = extract_atomic_numbers_payload(atomic_numbers, batch, n_atoms)?;

    let mut builtin_columns = Vec::new();
    if let Some(energy) = energy {
        let Some(array) = PyFloatArray1::from_any(energy) else {
            return Err(PyValueError::new_err(
                "energy must be a float32 or float64 ndarray with shape (batch,)",
            ));
        };
        builtin_columns.push(match array {
            PyFloatArray1::F32(arr) => {
                extract_builtin_scalar_column(&arr, batch, "energy", TYPE_FLOAT32)?
            }
            PyFloatArray1::F64(arr) => {
                extract_builtin_scalar_column(&arr, batch, "energy", TYPE_FLOAT)?
            }
        });
    }
    if let Some(forces) = forces {
        let Some(array) = PyFloatArray3::from_any(forces) else {
            return Err(PyValueError::new_err(
                "forces must be a float32 or float64 ndarray with shape (batch, n_atoms, 3)",
            ));
        };
        builtin_columns.push(match array {
            PyFloatArray3::F32(arr) => {
                extract_builtin_vec3_column(&arr, batch, n_atoms, "forces", TYPE_VEC3_F32)?
            }
            PyFloatArray3::F64(arr) => {
                extract_builtin_vec3_column(&arr, batch, n_atoms, "forces", TYPE_VEC3_F64)?
            }
        });
    }
    if let Some(charges) = charges {
        let Some(array) = PyFloatArray2::from_any(charges) else {
            return Err(PyValueError::new_err(
                "charges must be a float32 or float64 ndarray with shape (batch, n_atoms)",
            ));
        };
        builtin_columns.push(match array {
            PyFloatArray2::F32(arr) => {
                extract_builtin_float_array_column(&arr, batch, n_atoms, "charges", TYPE_F32_ARRAY)?
            }
            PyFloatArray2::F64(arr) => {
                extract_builtin_float_array_column(&arr, batch, n_atoms, "charges", TYPE_F64_ARRAY)?
            }
        });
    }
    if let Some(velocities) = velocities {
        let Some(array) = PyFloatArray3::from_any(velocities) else {
            return Err(PyValueError::new_err(
                "velocities must be a float32 or float64 ndarray with shape (batch, n_atoms, 3)",
            ));
        };
        builtin_columns.push(match array {
            PyFloatArray3::F32(arr) => {
                extract_builtin_vec3_column(&arr, batch, n_atoms, "velocities", TYPE_VEC3_F32)?
            }
            PyFloatArray3::F64(arr) => {
                extract_builtin_vec3_column(&arr, batch, n_atoms, "velocities", TYPE_VEC3_F64)?
            }
        });
    }
    if let Some(cell) = cell {
        let Some(array) = PyFloatArray3::from_any(cell) else {
            return Err(PyValueError::new_err(
                "cell must be a float32 or float64 ndarray with shape (batch, 3, 3)",
            ));
        };
        builtin_columns.push(match array {
            PyFloatArray3::F32(arr) => {
                extract_builtin_mat3_column(&arr, batch, "cell", TYPE_MAT3X3_F32)?
            }
            PyFloatArray3::F64(arr) => {
                extract_builtin_mat3_column(&arr, batch, "cell", TYPE_MAT3X3_F64)?
            }
        });
    }
    if let Some(stress) = stress {
        let Some(array) = PyFloatArray3::from_any(stress) else {
            return Err(PyValueError::new_err(
                "stress must be a float32 or float64 ndarray with shape (batch, 3, 3)",
            ));
        };
        builtin_columns.push(match array {
            PyFloatArray3::F32(arr) => {
                extract_builtin_mat3_column(&arr, batch, "stress", TYPE_MAT3X3_F32)?
            }
            PyFloatArray3::F64(arr) => {
                extract_builtin_mat3_column(&arr, batch, "stress", TYPE_MAT3X3_F64)?
            }
        });
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

    let batch_schema = build_batch_schema(
        positions_type,
        builtin_columns.iter().chain(custom_columns.iter()),
    )?;
    let record_format = inner
        .record_format_for_schema(batch_schema.clone())
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let build_record = |index: usize| -> Result<(Vec<u8>, u32), String> {
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
        })?;
        Ok((record, n_atoms as u32))
    };

    let records: Vec<(Vec<u8>, u32)> = if batch >= 1024 {
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

    py.detach(move || inner.add_owned_soa_records_with_schema(records, batch_schema))
        .map_err(|e| PyValueError::new_err(format!("{}", e)))
}
