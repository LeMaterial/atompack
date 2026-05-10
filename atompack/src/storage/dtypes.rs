use super::*;
use crate::atom::{FloatArrayData, FloatScalarData, Mat3Data, PropertyValue, Vec3Data};

pub(super) fn arr<const N: usize>(bytes: &[u8]) -> Result<[u8; N]> {
    bytes
        .try_into()
        .map_err(|_| Error::InvalidData("byte slice truncated".into()))
}

pub(super) fn positions_type_from_molecule(molecule: &Molecule) -> u8 {
    vec3_data_type_tag(&molecule.positions)
}

pub(super) fn vec3_data_type_tag(values: &Vec3Data) -> u8 {
    match values {
        Vec3Data::F32(_) => TYPE_VEC3_F32,
        Vec3Data::F64(_) => TYPE_VEC3_F64,
    }
}

pub(super) fn vec3_payload_len(values: &Vec3Data) -> usize {
    match values {
        Vec3Data::F32(values) => values.len() * 12,
        Vec3Data::F64(values) => values.len() * 24,
    }
}

pub(super) fn float_array_data_type_tag(values: &FloatArrayData) -> u8 {
    match values {
        FloatArrayData::F32(_) => TYPE_F32_ARRAY,
        FloatArrayData::F64(_) => TYPE_F64_ARRAY,
    }
}

pub(super) fn float_array_payload_len(values: &FloatArrayData) -> usize {
    match values {
        FloatArrayData::F32(values) => values.len() * 4,
        FloatArrayData::F64(values) => values.len() * 8,
    }
}

pub(super) fn mat3_data_type_tag(values: &Mat3Data) -> u8 {
    match values {
        Mat3Data::F32(_) => TYPE_MAT3X3_F32,
        Mat3Data::F64(_) => TYPE_MAT3X3_F64,
    }
}

pub(super) fn mat3_payload_len(values: &Mat3Data) -> usize {
    match values {
        Mat3Data::F32(_) => 36,
        Mat3Data::F64(_) => 72,
    }
}

pub(super) fn float_scalar_data_type_tag(value: &FloatScalarData) -> u8 {
    match value {
        FloatScalarData::F32(_) => TYPE_FLOAT32,
        FloatScalarData::F64(_) => TYPE_FLOAT,
    }
}

pub(super) fn float_scalar_payload_len(value: &FloatScalarData) -> usize {
    match value {
        FloatScalarData::F32(_) => 4,
        FloatScalarData::F64(_) => 8,
    }
}

pub(super) fn property_value_type_tag(value: &PropertyValue) -> u8 {
    match value {
        PropertyValue::Float(_) => TYPE_FLOAT,
        PropertyValue::Int(_) => TYPE_INT,
        PropertyValue::String(_) => TYPE_STRING,
        PropertyValue::FloatArray(_) => TYPE_F64_ARRAY,
        PropertyValue::Vec3Array(_) => TYPE_VEC3_F32,
        PropertyValue::IntArray(_) => TYPE_I64_ARRAY,
        PropertyValue::Float32Array(_) => TYPE_F32_ARRAY,
        PropertyValue::Vec3ArrayF64(_) => TYPE_VEC3_F64,
        PropertyValue::Int32Array(_) => TYPE_I32_ARRAY,
    }
}

pub(super) fn property_value_payload_len(value: &PropertyValue) -> usize {
    match value {
        PropertyValue::Float(_) | PropertyValue::Int(_) => 8,
        PropertyValue::String(value) => value.len(),
        PropertyValue::FloatArray(values) => values.len() * 8,
        PropertyValue::Vec3Array(values) => values.len() * 12,
        PropertyValue::IntArray(values) => values.len() * 8,
        PropertyValue::Float32Array(values) => values.len() * 4,
        PropertyValue::Vec3ArrayF64(values) => values.len() * 24,
        PropertyValue::Int32Array(values) => values.len() * 4,
    }
}

fn extend_f64(buf: &mut Vec<u8>, values: &[f64]) {
    for value in values {
        buf.extend_from_slice(&f64::to_le_bytes(*value));
    }
}

fn extend_f32(buf: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        buf.extend_from_slice(&f32::to_le_bytes(*value));
    }
}

fn extend_i64(buf: &mut Vec<u8>, values: &[i64]) {
    for value in values {
        buf.extend_from_slice(&i64::to_le_bytes(*value));
    }
}

fn extend_i32(buf: &mut Vec<u8>, values: &[i32]) {
    for value in values {
        buf.extend_from_slice(&i32::to_le_bytes(*value));
    }
}

pub(super) fn property_value_to_bytes(value: &PropertyValue) -> Vec<u8> {
    match value {
        PropertyValue::Float(value) => value.to_le_bytes().to_vec(),
        PropertyValue::Int(value) => value.to_le_bytes().to_vec(),
        PropertyValue::String(value) => value.as_bytes().to_vec(),
        PropertyValue::FloatArray(values) => {
            let mut payload = Vec::with_capacity(values.len() * 8);
            extend_f64(&mut payload, values);
            payload
        }
        PropertyValue::Vec3Array(values) => {
            let mut payload = Vec::with_capacity(values.len() * 12);
            for value in values {
                extend_f32(&mut payload, value);
            }
            payload
        }
        PropertyValue::IntArray(values) => {
            let mut payload = Vec::with_capacity(values.len() * 8);
            extend_i64(&mut payload, values);
            payload
        }
        PropertyValue::Float32Array(values) => {
            let mut payload = Vec::with_capacity(values.len() * 4);
            extend_f32(&mut payload, values);
            payload
        }
        PropertyValue::Vec3ArrayF64(values) => {
            let mut payload = Vec::with_capacity(values.len() * 24);
            for value in values {
                extend_f64(&mut payload, value);
            }
            payload
        }
        PropertyValue::Int32Array(values) => {
            let mut payload = Vec::with_capacity(values.len() * 4);
            extend_i32(&mut payload, values);
            payload
        }
    }
}

pub(super) fn validate_builtin_type_tag_for_record_format(
    record_format: u32,
    key: &str,
    type_tag: u8,
) -> Result<()> {
    match record_format {
        RECORD_FORMAT_SOA_V3 => Ok(()),
        RECORD_FORMAT_SOA_V2 => match key {
            "charges" if type_tag != TYPE_F64_ARRAY => Err(Error::InvalidData(
                "record format 2 does not support float32 charges".into(),
            )),
            "cell" if type_tag != TYPE_MAT3X3_F64 => Err(Error::InvalidData(
                "record format 2 does not support float32 cell".into(),
            )),
            "energy" if type_tag != TYPE_FLOAT => Err(Error::InvalidData(
                "record format 2 does not support float32 energy".into(),
            )),
            "forces" if type_tag != TYPE_VEC3_F32 => Err(Error::InvalidData(
                "record format 2 does not support float64 forces".into(),
            )),
            "stress" if type_tag != TYPE_MAT3X3_F64 => Err(Error::InvalidData(
                "record format 2 does not support float32 stress".into(),
            )),
            "velocities" if type_tag != TYPE_VEC3_F32 => Err(Error::InvalidData(
                "record format 2 does not support float64 velocities".into(),
            )),
            _ => Ok(()),
        },
        _ => Err(Error::InvalidData(format!(
            "Unsupported record format {}",
            record_format
        ))),
    }
}

pub(super) fn decode_vec3_f32(payload: &[u8]) -> Result<Vec<[f32; 3]>> {
    if !payload.len().is_multiple_of(12) {
        return Err(Error::InvalidData(
            "vec3<f32> payload length not divisible by 12".into(),
        ));
    }
    payload
        .chunks_exact(12)
        .map(|chunk| {
            Ok([
                f32::from_le_bytes(arr(&chunk[0..4])?),
                f32::from_le_bytes(arr(&chunk[4..8])?),
                f32::from_le_bytes(arr(&chunk[8..12])?),
            ])
        })
        .collect()
}

pub(super) fn decode_vec3_f64(payload: &[u8]) -> Result<Vec<[f64; 3]>> {
    if !payload.len().is_multiple_of(24) {
        return Err(Error::InvalidData(
            "vec3<f64> payload length not divisible by 24".into(),
        ));
    }
    payload
        .chunks_exact(24)
        .map(|chunk| {
            Ok([
                f64::from_le_bytes(arr(&chunk[0..8])?),
                f64::from_le_bytes(arr(&chunk[8..16])?),
                f64::from_le_bytes(arr(&chunk[16..24])?),
            ])
        })
        .collect()
}

pub(super) fn decode_f32_array(payload: &[u8]) -> Result<Vec<f32>> {
    if !payload.len().is_multiple_of(4) {
        return Err(Error::InvalidData(
            "f32 array payload length not divisible by 4".into(),
        ));
    }
    payload
        .chunks_exact(4)
        .map(|chunk| Ok(f32::from_le_bytes(arr(chunk)?)))
        .collect()
}

pub(super) fn decode_f64_array(payload: &[u8]) -> Result<Vec<f64>> {
    if !payload.len().is_multiple_of(8) {
        return Err(Error::InvalidData(
            "f64 array payload length not divisible by 8".into(),
        ));
    }
    payload
        .chunks_exact(8)
        .map(|chunk| Ok(f64::from_le_bytes(arr(chunk)?)))
        .collect()
}

pub(super) fn decode_mat3x3_f32(payload: &[u8]) -> Result<[[f32; 3]; 3]> {
    if payload.len() != 36 {
        return Err(Error::InvalidData(format!(
            "mat3x3<f32> payload length {} (expected 36)",
            payload.len()
        )));
    }
    let mut mat = [[0.0f32; 3]; 3];
    for (row_idx, row) in mat.iter_mut().enumerate() {
        for (col_idx, cell) in row.iter_mut().enumerate() {
            let offset = (row_idx * 3 + col_idx) * 4;
            *cell = f32::from_le_bytes(arr(&payload[offset..offset + 4])?);
        }
    }
    Ok(mat)
}

pub(super) fn decode_mat3x3_f64(payload: &[u8]) -> Result<[[f64; 3]; 3]> {
    if payload.len() != 72 {
        return Err(Error::InvalidData(format!(
            "mat3x3<f64> payload length {} (expected 72)",
            payload.len()
        )));
    }
    let mut mat = [[0.0f64; 3]; 3];
    for (row_idx, row) in mat.iter_mut().enumerate() {
        for (col_idx, cell) in row.iter_mut().enumerate() {
            let offset = (row_idx * 3 + col_idx) * 8;
            *cell = f64::from_le_bytes(arr(&payload[offset..offset + 8])?);
        }
    }
    Ok(mat)
}

pub(super) fn decode_property_value(type_tag: u8, payload: &[u8]) -> Result<PropertyValue> {
    Ok(match type_tag {
        TYPE_FLOAT => {
            if payload.len() < 8 {
                return Err(Error::InvalidData("f64 property truncated".into()));
            }
            PropertyValue::Float(f64::from_le_bytes(arr(&payload[..8])?))
        }
        TYPE_INT => {
            if payload.len() < 8 {
                return Err(Error::InvalidData("i64 property truncated".into()));
            }
            PropertyValue::Int(i64::from_le_bytes(arr(&payload[..8])?))
        }
        TYPE_STRING => PropertyValue::String(
            std::str::from_utf8(payload)
                .map_err(|_| Error::InvalidData("Invalid UTF-8 in property".into()))?
                .to_string(),
        ),
        TYPE_F64_ARRAY => PropertyValue::FloatArray(decode_f64_array(payload)?),
        TYPE_VEC3_F32 => PropertyValue::Vec3Array(decode_vec3_f32(payload)?),
        TYPE_I64_ARRAY => {
            if !payload.len().is_multiple_of(8) {
                return Err(Error::InvalidData(
                    "i64 array payload length not divisible by 8".into(),
                ));
            }
            PropertyValue::IntArray(
                payload
                    .chunks_exact(8)
                    .map(|chunk| Ok(i64::from_le_bytes(arr(chunk)?)))
                    .collect::<Result<_>>()?,
            )
        }
        TYPE_F32_ARRAY => PropertyValue::Float32Array(decode_f32_array(payload)?),
        TYPE_VEC3_F64 => PropertyValue::Vec3ArrayF64(decode_vec3_f64(payload)?),
        TYPE_I32_ARRAY => {
            if !payload.len().is_multiple_of(4) {
                return Err(Error::InvalidData(
                    "i32 array payload length not divisible by 4".into(),
                ));
            }
            PropertyValue::Int32Array(
                payload
                    .chunks_exact(4)
                    .map(|chunk| Ok(i32::from_le_bytes(arr(chunk)?)))
                    .collect::<Result<_>>()?,
            )
        }
        _ => return Err(Error::InvalidData(format!("Unknown type tag {}", type_tag))),
    })
}

pub(super) fn decode_float_scalar_data(
    payload: &[u8],
    type_tag: u8,
    field_name: &str,
) -> Result<FloatScalarData> {
    match type_tag {
        TYPE_FLOAT => {
            if payload.len() != 8 {
                return Err(Error::InvalidData(format!(
                    "{field_name} f64 payload truncated"
                )));
            }
            Ok(FloatScalarData::F64(f64::from_le_bytes(arr(payload)?)))
        }
        TYPE_FLOAT32 => {
            if payload.len() != 4 {
                return Err(Error::InvalidData(format!(
                    "{field_name} f32 payload truncated"
                )));
            }
            Ok(FloatScalarData::F32(f32::from_le_bytes(arr(payload)?)))
        }
        _ => Err(Error::InvalidData(format!(
            "Unsupported {field_name} type tag {}",
            type_tag
        ))),
    }
}

pub(super) fn decode_vec3_data(payload: &[u8], type_tag: u8, field_name: &str) -> Result<Vec3Data> {
    match type_tag {
        TYPE_VEC3_F32 => Ok(Vec3Data::F32(decode_vec3_f32(payload)?)),
        TYPE_VEC3_F64 => Ok(Vec3Data::F64(decode_vec3_f64(payload)?)),
        _ => Err(Error::InvalidData(format!(
            "Unsupported {field_name} type tag {}",
            type_tag
        ))),
    }
}

pub(super) fn decode_float_array_data(
    payload: &[u8],
    type_tag: u8,
    field_name: &str,
) -> Result<FloatArrayData> {
    match type_tag {
        TYPE_F32_ARRAY => Ok(FloatArrayData::F32(decode_f32_array(payload)?)),
        TYPE_F64_ARRAY => Ok(FloatArrayData::F64(decode_f64_array(payload)?)),
        _ => Err(Error::InvalidData(format!(
            "Unsupported {field_name} type tag {}",
            type_tag
        ))),
    }
}

pub(super) fn decode_mat3_data(payload: &[u8], type_tag: u8, field_name: &str) -> Result<Mat3Data> {
    match type_tag {
        TYPE_MAT3X3_F32 => Ok(Mat3Data::F32(decode_mat3x3_f32(payload)?)),
        TYPE_MAT3X3_F64 => Ok(Mat3Data::F64(decode_mat3x3_f64(payload)?)),
        _ => Err(Error::InvalidData(format!(
            "Unsupported {field_name} type tag {}",
            type_tag
        ))),
    }
}
