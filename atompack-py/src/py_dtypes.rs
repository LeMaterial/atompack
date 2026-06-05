use super::*;

pub(crate) enum PyFloatArray1<'py> {
    F32(Bound<'py, PyArray1<f32>>),
    F64(Bound<'py, PyArray1<f64>>),
}

impl<'py> PyFloatArray1<'py> {
    pub(crate) fn from_any(value: &Bound<'py, PyAny>) -> Option<Self> {
        if let Ok(arr) = value.cast::<PyArray1<f32>>() {
            return Some(Self::F32(arr.clone()));
        }
        if let Ok(arr) = value.cast::<PyArray1<f64>>() {
            return Some(Self::F64(arr.clone()));
        }
        None
    }

    pub(crate) fn to_float_array_data(
        &self,
        label: &str,
        expected_len: usize,
    ) -> PyResult<FloatArrayData> {
        match self {
            Self::F32(arr) => {
                let values = arr.readonly().as_array().to_vec();
                if values.len() != expected_len {
                    return Err(PyValueError::new_err(format!(
                        "{} length ({}) doesn't match atom count ({})",
                        label,
                        values.len(),
                        expected_len
                    )));
                }
                Ok(FloatArrayData::F32(values))
            }
            Self::F64(arr) => {
                let values = arr.readonly().as_array().to_vec();
                if values.len() != expected_len {
                    return Err(PyValueError::new_err(format!(
                        "{} length ({}) doesn't match atom count ({})",
                        label,
                        values.len(),
                        expected_len
                    )));
                }
                Ok(FloatArrayData::F64(values))
            }
        }
    }
}

pub(crate) enum PyIntArray1<'py> {
    I32(Bound<'py, PyArray1<i32>>),
    I64(Bound<'py, PyArray1<i64>>),
}

impl<'py> PyIntArray1<'py> {
    pub(crate) fn from_any(value: &Bound<'py, PyAny>) -> Option<Self> {
        if let Ok(arr) = value.cast::<PyArray1<i32>>() {
            return Some(Self::I32(arr.clone()));
        }
        if let Ok(arr) = value.cast::<PyArray1<i64>>() {
            return Some(Self::I64(arr.clone()));
        }
        None
    }
}

pub(crate) enum PyFloatArray2<'py> {
    F32(Bound<'py, PyArray2<f32>>),
    F64(Bound<'py, PyArray2<f64>>),
}

impl<'py> PyFloatArray2<'py> {
    pub(crate) fn from_any(value: &Bound<'py, PyAny>) -> Option<Self> {
        if let Ok(arr) = value.cast::<PyArray2<f32>>() {
            return Some(Self::F32(arr.clone()));
        }
        if let Ok(arr) = value.cast::<PyArray2<f64>>() {
            return Some(Self::F64(arr.clone()));
        }
        None
    }

    pub(crate) fn parse_vec3_data(
        &self,
        label: &str,
        expected_rows: Option<usize>,
    ) -> PyResult<Vec3Data> {
        match self {
            Self::F32(arr) => {
                let readonly = arr.readonly();
                let view = readonly.as_array();
                let shape = view.shape();
                if shape.len() != 2 || shape[1] != 3 {
                    return Err(PyValueError::new_err(format!(
                        "{} must have shape ({}, 3)",
                        label,
                        expected_rows.map_or("n".to_string(), |rows| rows.to_string())
                    )));
                }
                if let Some(rows) = expected_rows
                    && shape[0] != rows
                {
                    return Err(PyValueError::new_err(format!(
                        "{} must have shape ({}, 3)",
                        label, rows
                    )));
                }
                Ok(Vec3Data::F32(
                    view.outer_iter()
                        .map(|row| [row[0], row[1], row[2]])
                        .collect(),
                ))
            }
            Self::F64(arr) => {
                let readonly = arr.readonly();
                let view = readonly.as_array();
                let shape = view.shape();
                if shape.len() != 2 || shape[1] != 3 {
                    return Err(PyValueError::new_err(format!(
                        "{} must have shape ({}, 3)",
                        label,
                        expected_rows.map_or("n".to_string(), |rows| rows.to_string())
                    )));
                }
                if let Some(rows) = expected_rows
                    && shape[0] != rows
                {
                    return Err(PyValueError::new_err(format!(
                        "{} must have shape ({}, 3)",
                        label, rows
                    )));
                }
                Ok(Vec3Data::F64(
                    view.outer_iter()
                        .map(|row| [row[0], row[1], row[2]])
                        .collect(),
                ))
            }
        }
    }

    pub(crate) fn parse_mat3_data(&self, label: &str) -> PyResult<Mat3Data> {
        match self {
            Self::F32(arr) => {
                let readonly = arr.readonly();
                let view = readonly.as_array();
                if view.shape() != [3, 3] {
                    return Err(PyValueError::new_err(format!(
                        "{} must have shape (3, 3)",
                        label
                    )));
                }
                Ok(Mat3Data::F32([
                    [view[[0, 0]], view[[0, 1]], view[[0, 2]]],
                    [view[[1, 0]], view[[1, 1]], view[[1, 2]]],
                    [view[[2, 0]], view[[2, 1]], view[[2, 2]]],
                ]))
            }
            Self::F64(arr) => {
                let readonly = arr.readonly();
                let view = readonly.as_array();
                if view.shape() != [3, 3] {
                    return Err(PyValueError::new_err(format!(
                        "{} must have shape (3, 3)",
                        label
                    )));
                }
                Ok(Mat3Data::F64([
                    [view[[0, 0]], view[[0, 1]], view[[0, 2]]],
                    [view[[1, 0]], view[[1, 1]], view[[1, 2]]],
                    [view[[2, 0]], view[[2, 1]], view[[2, 2]]],
                ]))
            }
        }
    }
}

pub(crate) enum PyFloatArray3<'py> {
    F32(Bound<'py, PyArray3<f32>>),
    F64(Bound<'py, PyArray3<f64>>),
}

impl<'py> PyFloatArray3<'py> {
    pub(crate) fn from_any(value: &Bound<'py, PyAny>) -> Option<Self> {
        if let Ok(arr) = value.cast::<PyArray3<f32>>() {
            return Some(Self::F32(arr.clone()));
        }
        if let Ok(arr) = value.cast::<PyArray3<f64>>() {
            return Some(Self::F64(arr.clone()));
        }
        None
    }
}

pub(crate) fn parse_vec3_field(
    value: &Bound<'_, PyAny>,
    label: &str,
    expected_rows: usize,
) -> PyResult<Vec3Data> {
    let Some(array) = PyFloatArray2::from_any(value) else {
        return Err(PyValueError::new_err(format!(
            "{} must be a float32 or float64 ndarray with shape ({}, 3)",
            label, expected_rows
        )));
    };
    array.parse_vec3_data(label, Some(expected_rows))
}

pub(crate) fn parse_positions_field(value: &Bound<'_, PyAny>) -> PyResult<Vec3Data> {
    let Some(array) = PyFloatArray2::from_any(value) else {
        return Err(PyValueError::new_err(
            "positions must be a float32 or float64 ndarray with shape (n_atoms, 3)",
        ));
    };
    array.parse_vec3_data("positions", None)
}

pub(crate) fn parse_float_array_field(
    value: &Bound<'_, PyAny>,
    label: &str,
    expected_len: usize,
) -> PyResult<FloatArrayData> {
    let Some(array) = PyFloatArray1::from_any(value) else {
        return Err(PyValueError::new_err(format!(
            "{} must be a float32 or float64 ndarray with shape ({},)",
            label, expected_len
        )));
    };
    array.to_float_array_data(label, expected_len)
}

pub(crate) fn parse_mat3_field(value: &Bound<'_, PyAny>, label: &str) -> PyResult<Mat3Data> {
    let Some(array) = PyFloatArray2::from_any(value) else {
        return Err(PyValueError::new_err(format!(
            "{} must be a float32 or float64 ndarray with shape (3, 3)",
            label
        )));
    };
    array.parse_mat3_data(label)
}

fn tensor_shape(shape: &[usize], values_len: usize) -> PyResult<Vec<usize>> {
    let expected = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| PyValueError::new_err("Tensor shape overflows usize"))
    })?;
    if expected != values_len {
        return Err(PyValueError::new_err(format!(
            "Tensor shape {:?} expects {} values, got {}",
            shape, expected, values_len
        )));
    }
    if shape.len() > u8::MAX as usize {
        return Err(PyValueError::new_err(format!(
            "Tensor rank {} exceeds maximum {}",
            shape.len(),
            u8::MAX
        )));
    }
    if shape.iter().any(|dim| *dim > u32::MAX as usize) {
        return Err(PyValueError::new_err(
            "Tensor dimensions must fit in u32 for storage",
        ));
    }
    Ok(shape.to_vec())
}

fn parse_tensor_property_value(value: &Bound<'_, PyAny>) -> PyResult<Option<PropertyValue>> {
    if let Ok(arr) = value.cast::<PyArrayDyn<f32>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        let values: Vec<f32> = view.iter().copied().collect();
        let shape = tensor_shape(view.shape(), values.len())?;
        return Ok(Some(PropertyValue::Tensor(TensorData::F32 {
            shape,
            values,
        })));
    }
    if let Ok(arr) = value.cast::<PyArrayDyn<f64>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        let values: Vec<f64> = view.iter().copied().collect();
        let shape = tensor_shape(view.shape(), values.len())?;
        return Ok(Some(PropertyValue::Tensor(TensorData::F64 {
            shape,
            values,
        })));
    }
    if let Ok(arr) = value.cast::<PyArrayDyn<i32>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        let values: Vec<i32> = view.iter().copied().collect();
        let shape = tensor_shape(view.shape(), values.len())?;
        return Ok(Some(PropertyValue::Tensor(TensorData::I32 {
            shape,
            values,
        })));
    }
    if let Ok(arr) = value.cast::<PyArrayDyn<i64>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        let values: Vec<i64> = view.iter().copied().collect();
        let shape = tensor_shape(view.shape(), values.len())?;
        return Ok(Some(PropertyValue::Tensor(TensorData::I64 {
            shape,
            values,
        })));
    }
    Ok(None)
}

pub(crate) fn parse_property_value(value: &Bound<'_, PyAny>) -> PyResult<PropertyValue> {
    if value.is_none() {
        return Ok(PropertyValue::None);
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(PropertyValue::Int(v));
    }
    if let Ok(v) = value.extract::<f64>() {
        return Ok(PropertyValue::Float(v));
    }
    if let Ok(v) = value.extract::<String>() {
        return Ok(PropertyValue::String(v));
    }
    if let Some(array) = PyFloatArray1::from_any(value) {
        return Ok(match array {
            PyFloatArray1::F32(arr) => {
                PropertyValue::Float32Array(arr.readonly().as_array().to_vec())
            }
            PyFloatArray1::F64(arr) => {
                PropertyValue::FloatArray(arr.readonly().as_array().to_vec())
            }
        });
    }
    if let Some(array) = PyIntArray1::from_any(value) {
        return Ok(match array {
            PyIntArray1::I32(arr) => PropertyValue::Int32Array(arr.readonly().as_array().to_vec()),
            PyIntArray1::I64(arr) => PropertyValue::IntArray(arr.readonly().as_array().to_vec()),
        });
    }
    if let Some(array) = PyFloatArray2::from_any(value) {
        match array {
            PyFloatArray2::F32(arr) => {
                let readonly = arr.readonly();
                let arr_view = readonly.as_array();
                let shape = arr_view.shape();
                if shape[1] == 3 {
                    return Ok(PropertyValue::Vec3Array(
                        arr_view
                            .outer_iter()
                            .map(|row| [row[0], row[1], row[2]])
                            .collect(),
                    ));
                }
            }
            PyFloatArray2::F64(arr) => {
                let readonly = arr.readonly();
                let arr_view = readonly.as_array();
                let shape = arr_view.shape();
                if shape[1] == 3 {
                    return Ok(PropertyValue::Vec3ArrayF64(
                        arr_view
                            .outer_iter()
                            .map(|row| [row[0], row[1], row[2]])
                            .collect(),
                    ));
                }
            }
        }
    }
    if let Some(value) = parse_tensor_property_value(value)? {
        return Ok(value);
    }
    Err(PyValueError::new_err(
        "Unsupported property type. Supported: None, float, int, str, and numeric ndarray with dtype float32, float64, int32, or int64",
    ))
}
