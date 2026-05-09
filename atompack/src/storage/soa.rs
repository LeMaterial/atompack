use super::*;
use crate::atom::{FloatArrayData, FloatScalarData, Mat3Data, Vec3Data};

/// Write a single tagged section: [kind:u8][key_len:u8][key][type_tag:u8][payload_len:u32][payload]
fn write_section(buf: &mut Vec<u8>, kind: u8, key: &str, type_tag: u8, payload: &[u8]) {
    buf.push(kind);
    buf.push(key.len() as u8);
    buf.extend_from_slice(key.as_bytes());
    buf.push(type_tag);
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(payload);
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

fn extend_f64(b: &mut Vec<u8>, v: &[f64]) {
    for x in v {
        b.extend_from_slice(&f64::to_le_bytes(*x));
    }
}
fn extend_f32(b: &mut Vec<u8>, v: &[f32]) {
    for x in v {
        b.extend_from_slice(&f32::to_le_bytes(*x));
    }
}
fn extend_i64(b: &mut Vec<u8>, v: &[i64]) {
    for x in v {
        b.extend_from_slice(&i64::to_le_bytes(*x));
    }
}
fn extend_i32(b: &mut Vec<u8>, v: &[i32]) {
    for x in v {
        b.extend_from_slice(&i32::to_le_bytes(*x));
    }
}

pub(super) fn property_value_to_bytes(value: &PropertyValue) -> Vec<u8> {
    match value {
        PropertyValue::Float(v) => v.to_le_bytes().to_vec(),
        PropertyValue::Int(v) => v.to_le_bytes().to_vec(),
        PropertyValue::String(s) => s.as_bytes().to_vec(),
        PropertyValue::FloatArray(v) => {
            let mut b = Vec::with_capacity(v.len() * 8);
            extend_f64(&mut b, v);
            b
        }
        PropertyValue::Vec3Array(v) => {
            let mut b = Vec::with_capacity(v.len() * 12);
            for a in v {
                extend_f32(&mut b, a);
            }
            b
        }
        PropertyValue::IntArray(v) => {
            let mut b = Vec::with_capacity(v.len() * 8);
            extend_i64(&mut b, v);
            b
        }
        PropertyValue::Float32Array(v) => {
            let mut b = Vec::with_capacity(v.len() * 4);
            extend_f32(&mut b, v);
            b
        }
        PropertyValue::Vec3ArrayF64(v) => {
            let mut b = Vec::with_capacity(v.len() * 24);
            for a in v {
                extend_f64(&mut b, a);
            }
            b
        }
        PropertyValue::Int32Array(v) => {
            let mut b = Vec::with_capacity(v.len() * 4);
            extend_i32(&mut b, v);
            b
        }
    }
}

/// Try to read a fixed-size array from a byte slice, returning an error on truncation.
pub(super) fn arr<const N: usize>(bytes: &[u8]) -> Result<[u8; N]> {
    bytes
        .try_into()
        .map_err(|_| Error::InvalidData("byte slice truncated".into()))
}

pub(super) fn decode_vec3_f32(payload: &[u8]) -> Result<Vec<[f32; 3]>> {
    if !payload.len().is_multiple_of(12) {
        return Err(Error::InvalidData(
            "vec3<f32> payload length not divisible by 12".into(),
        ));
    }
    payload
        .chunks_exact(12)
        .map(|c| {
            Ok([
                f32::from_le_bytes(arr(&c[0..4])?),
                f32::from_le_bytes(arr(&c[4..8])?),
                f32::from_le_bytes(arr(&c[8..12])?),
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
        .map(|c| {
            Ok([
                f64::from_le_bytes(arr(&c[0..8])?),
                f64::from_le_bytes(arr(&c[8..16])?),
                f64::from_le_bytes(arr(&c[16..24])?),
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
        .map(|c| Ok(f32::from_le_bytes(arr(c)?)))
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
        .map(|c| Ok(f64::from_le_bytes(arr(c)?)))
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
    for (r, row) in mat.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            let o = (r * 3 + c) * 4;
            *cell = f32::from_le_bytes(arr(&payload[o..o + 4])?);
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
    for (r, row) in mat.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            let o = (r * 3 + c) * 8;
            *cell = f64::from_le_bytes(arr(&payload[o..o + 8])?);
        }
    }
    Ok(mat)
}

fn write_vec3_section(buf: &mut Vec<u8>, key: &str, values: &Vec3Data) {
    match values {
        Vec3Data::F32(values) => {
            let mut payload = Vec::with_capacity(values.len() * 12);
            for value in values {
                extend_f32(&mut payload, value);
            }
            write_section(buf, KIND_BUILTIN, key, TYPE_VEC3_F32, &payload);
        }
        Vec3Data::F64(values) => {
            let mut payload = Vec::with_capacity(values.len() * 24);
            for value in values {
                extend_f64(&mut payload, value);
            }
            write_section(buf, KIND_BUILTIN, key, TYPE_VEC3_F64, &payload);
        }
    }
}

fn write_float_array_section(buf: &mut Vec<u8>, key: &str, values: &FloatArrayData) {
    match values {
        FloatArrayData::F32(values) => {
            let mut payload = Vec::with_capacity(values.len() * 4);
            extend_f32(&mut payload, values);
            write_section(buf, KIND_BUILTIN, key, TYPE_F32_ARRAY, &payload);
        }
        FloatArrayData::F64(values) => {
            let mut payload = Vec::with_capacity(values.len() * 8);
            extend_f64(&mut payload, values);
            write_section(buf, KIND_BUILTIN, key, TYPE_F64_ARRAY, &payload);
        }
    }
}

fn write_mat3_section(buf: &mut Vec<u8>, key: &str, values: &Mat3Data) {
    match values {
        Mat3Data::F32(values) => {
            let mut payload = Vec::with_capacity(36);
            for row in values {
                extend_f32(&mut payload, row);
            }
            write_section(buf, KIND_BUILTIN, key, TYPE_MAT3X3_F32, &payload);
        }
        Mat3Data::F64(values) => {
            let mut payload = Vec::with_capacity(72);
            for row in values {
                extend_f64(&mut payload, row);
            }
            write_section(buf, KIND_BUILTIN, key, TYPE_MAT3X3_F64, &payload);
        }
    }
}

fn write_energy_section(buf: &mut Vec<u8>, value: &FloatScalarData) {
    match value {
        FloatScalarData::F32(value) => {
            write_section(
                buf,
                KIND_BUILTIN,
                "energy",
                TYPE_FLOAT32,
                &value.to_le_bytes(),
            );
        }
        FloatScalarData::F64(value) => {
            write_section(
                buf,
                KIND_BUILTIN,
                "energy",
                TYPE_FLOAT,
                &value.to_le_bytes(),
            );
        }
    }
}

pub(super) fn resolve_positions_type(
    record_format: u32,
    positions_type_hint: Option<u8>,
) -> Result<u8> {
    match record_format {
        RECORD_FORMAT_SOA_V2 => match positions_type_hint {
            Some(TYPE_VEC3_F64) => Err(Error::InvalidData(
                "record format 2 does not support float64 positions".into(),
            )),
            _ => Ok(TYPE_VEC3_F32),
        },
        RECORD_FORMAT_SOA_V3 => positions_type_hint.ok_or_else(|| {
            Error::InvalidData("Missing positions dtype for record format 3".into())
        }),
        _ => Err(Error::InvalidData(format!(
            "Unsupported record format {}",
            record_format
        ))),
    }
}

pub(super) fn positions_stride(positions_type: u8) -> Result<usize> {
    match positions_type {
        TYPE_VEC3_F32 => Ok(12),
        TYPE_VEC3_F64 => Ok(24),
        _ => Err(Error::InvalidData(format!(
            "Unsupported positions type tag {}",
            positions_type
        ))),
    }
}

fn validate_record_format_compat(molecule: &Molecule, record_format: u32) -> Result<()> {
    match record_format {
        RECORD_FORMAT_SOA_V3 => Ok(()),
        RECORD_FORMAT_SOA_V2 => {
            if matches!(molecule.positions, Vec3Data::F64(_)) {
                return Err(Error::InvalidData(
                    "record format 2 does not support float64 positions".into(),
                ));
            }
            if matches!(molecule.charges, Some(FloatArrayData::F32(_))) {
                return Err(Error::InvalidData(
                    "record format 2 does not support float32 charges".into(),
                ));
            }
            if matches!(molecule.cell, Some(Mat3Data::F32(_))) {
                return Err(Error::InvalidData(
                    "record format 2 does not support float32 cell".into(),
                ));
            }
            if matches!(molecule.energy, Some(FloatScalarData::F32(_))) {
                return Err(Error::InvalidData(
                    "record format 2 does not support float32 energy".into(),
                ));
            }
            if matches!(molecule.forces, Some(Vec3Data::F64(_))) {
                return Err(Error::InvalidData(
                    "record format 2 does not support float64 forces".into(),
                ));
            }
            if matches!(molecule.stress, Some(Mat3Data::F32(_))) {
                return Err(Error::InvalidData(
                    "record format 2 does not support float32 stress".into(),
                ));
            }
            if matches!(molecule.velocities, Some(Vec3Data::F64(_))) {
                return Err(Error::InvalidData(
                    "record format 2 does not support float64 velocities".into(),
                ));
            }
            Ok(())
        }
        _ => Err(Error::InvalidData(format!(
            "Unsupported record format {}",
            record_format
        ))),
    }
}

fn write_positions(buf: &mut Vec<u8>, positions: &Vec3Data) {
    match positions {
        Vec3Data::F32(values) => {
            for value in values {
                extend_f32(buf, value);
            }
        }
        Vec3Data::F64(values) => {
            for value in values {
                extend_f64(buf, value);
            }
        }
    }
}

fn count_sections(molecule: &Molecule) -> u16 {
    let mut n_sections = 0;
    if molecule.charges.is_some() {
        n_sections += 1;
    }
    if molecule.cell.is_some() {
        n_sections += 1;
    }
    if molecule.energy.is_some() {
        n_sections += 1;
    }
    if molecule.forces.is_some() {
        n_sections += 1;
    }
    if molecule.name.is_some() {
        n_sections += 1;
    }
    if molecule.pbc.is_some() {
        n_sections += 1;
    }
    if molecule.stress.is_some() {
        n_sections += 1;
    }
    if molecule.velocities.is_some() {
        n_sections += 1;
    }
    n_sections += molecule.atom_properties.len() as u16;
    n_sections += molecule.properties.len() as u16;
    n_sections
}

fn write_sections(buf: &mut Vec<u8>, molecule: &Molecule) {
    if let Some(ref charges) = molecule.charges {
        write_float_array_section(buf, "charges", charges);
    }
    if let Some(ref cell) = molecule.cell {
        write_mat3_section(buf, "cell", cell);
    }
    if let Some(ref energy) = molecule.energy {
        write_energy_section(buf, energy);
    }
    if let Some(ref forces) = molecule.forces {
        write_vec3_section(buf, "forces", forces);
    }
    if let Some(ref name) = molecule.name {
        write_section(buf, KIND_BUILTIN, "name", TYPE_STRING, name.as_bytes());
    }
    if let Some(ref pbc) = molecule.pbc {
        let payload = [pbc[0] as u8, pbc[1] as u8, pbc[2] as u8];
        write_section(buf, KIND_BUILTIN, "pbc", TYPE_BOOL3, &payload);
    }
    if let Some(ref stress) = molecule.stress {
        write_mat3_section(buf, "stress", stress);
    }
    if let Some(ref velocities) = molecule.velocities {
        write_vec3_section(buf, "velocities", velocities);
    }

    let mut atom_keys: Vec<&String> = molecule.atom_properties.keys().collect();
    atom_keys.sort();
    for key in atom_keys {
        let value = &molecule.atom_properties[key];
        let payload = property_value_to_bytes(value);
        write_section(
            buf,
            KIND_ATOM_PROP,
            key,
            property_value_type_tag(value),
            &payload,
        );
    }

    let mut prop_keys: Vec<&String> = molecule.properties.keys().collect();
    prop_keys.sort();
    for key in prop_keys {
        let value = &molecule.properties[key];
        let payload = property_value_to_bytes(value);
        write_section(
            buf,
            KIND_MOL_PROP,
            key,
            property_value_type_tag(value),
            &payload,
        );
    }
}

fn decode_property_value(type_tag: u8, payload: &[u8]) -> Result<PropertyValue> {
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
                    .map(|c| Ok(i64::from_le_bytes(arr(c)?)))
                    .collect::<Result<_>>()?,
            )
        }
        TYPE_F32_ARRAY => {
            if !payload.len().is_multiple_of(4) {
                return Err(Error::InvalidData(
                    "f32 array payload length not divisible by 4".into(),
                ));
            }
            PropertyValue::Float32Array(
                payload
                    .chunks_exact(4)
                    .map(|c| Ok(f32::from_le_bytes(arr(c)?)))
                    .collect::<Result<_>>()?,
            )
        }
        TYPE_VEC3_F64 => {
            if !payload.len().is_multiple_of(24) {
                return Err(Error::InvalidData(
                    "vec3<f64> payload length not divisible by 24".into(),
                ));
            }
            PropertyValue::Vec3ArrayF64(
                payload
                    .chunks_exact(24)
                    .map(|c| {
                        Ok([
                            f64::from_le_bytes(arr(&c[0..8])?),
                            f64::from_le_bytes(arr(&c[8..16])?),
                            f64::from_le_bytes(arr(&c[16..24])?),
                        ])
                    })
                    .collect::<Result<_>>()?,
            )
        }
        TYPE_I32_ARRAY => {
            if !payload.len().is_multiple_of(4) {
                return Err(Error::InvalidData(
                    "i32 array payload length not divisible by 4".into(),
                ));
            }
            PropertyValue::Int32Array(
                payload
                    .chunks_exact(4)
                    .map(|c| Ok(i32::from_le_bytes(arr(c)?)))
                    .collect::<Result<_>>()?,
            )
        }
        _ => return Err(Error::InvalidData(format!("Unknown type tag {}", type_tag))),
    })
}

pub(super) fn serialize_molecule_soa(molecule: &Molecule, record_format: u32) -> Result<Vec<u8>> {
    validate_record_format_compat(molecule, record_format)?;

    let n = molecule.len();
    let mut buf = Vec::new();

    buf.extend_from_slice(&(n as u32).to_le_bytes());
    write_positions(&mut buf, &molecule.positions);
    buf.extend_from_slice(&molecule.atomic_numbers);
    buf.extend_from_slice(&count_sections(molecule).to_le_bytes());
    write_sections(&mut buf, molecule);

    Ok(buf)
}

fn decode_positions(
    bytes: &[u8],
    pos: &mut usize,
    n_atoms: usize,
    positions_type: u8,
) -> Result<Vec3Data> {
    let positions_len = n_atoms
        .checked_mul(positions_stride(positions_type)?)
        .ok_or_else(|| Error::InvalidData("SOA positions overflow".into()))?;
    let positions_end = pos
        .checked_add(positions_len)
        .ok_or_else(|| Error::InvalidData("SOA positions overflow".into()))?;
    if positions_end > bytes.len() {
        return Err(Error::InvalidData(
            "SOA record truncated at positions".into(),
        ));
    }
    let positions = match positions_type {
        TYPE_VEC3_F32 => Vec3Data::F32(decode_vec3_f32(&bytes[*pos..positions_end])?),
        TYPE_VEC3_F64 => Vec3Data::F64(decode_vec3_f64(&bytes[*pos..positions_end])?),
        _ => {
            return Err(Error::InvalidData(format!(
                "Unsupported positions type tag {}",
                positions_type
            )));
        }
    };
    *pos = positions_end;
    Ok(positions)
}

fn decode_float_scalar_data(
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

fn decode_vec3_data(payload: &[u8], type_tag: u8, field_name: &str) -> Result<Vec3Data> {
    match type_tag {
        TYPE_VEC3_F32 => Ok(Vec3Data::F32(decode_vec3_f32(payload)?)),
        TYPE_VEC3_F64 => Ok(Vec3Data::F64(decode_vec3_f64(payload)?)),
        _ => Err(Error::InvalidData(format!(
            "Unsupported {field_name} type tag {}",
            type_tag
        ))),
    }
}

fn decode_float_array_data(
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

fn decode_mat3_data(payload: &[u8], type_tag: u8, field_name: &str) -> Result<Mat3Data> {
    match type_tag {
        TYPE_MAT3X3_F32 => Ok(Mat3Data::F32(decode_mat3x3_f32(payload)?)),
        TYPE_MAT3X3_F64 => Ok(Mat3Data::F64(decode_mat3x3_f64(payload)?)),
        _ => Err(Error::InvalidData(format!(
            "Unsupported {field_name} type tag {}",
            type_tag
        ))),
    }
}

fn decode_builtin_section(
    mol: &mut Molecule,
    key: &str,
    type_tag: u8,
    payload: &[u8],
) -> Result<()> {
    match key {
        "energy" => mol.energy = Some(decode_float_scalar_data(payload, type_tag, "energy")?),
        "forces" => mol.forces = Some(decode_vec3_data(payload, type_tag, "forces")?),
        "charges" => mol.charges = Some(decode_float_array_data(payload, type_tag, "charges")?),
        "velocities" => mol.velocities = Some(decode_vec3_data(payload, type_tag, "velocities")?),
        "cell" => mol.cell = Some(decode_mat3_data(payload, type_tag, "cell")?),
        "pbc" => {
            if payload.len() < 3 {
                return Err(Error::InvalidData("pbc payload truncated".into()));
            }
            mol.pbc = Some([payload[0] != 0, payload[1] != 0, payload[2] != 0]);
        }
        "stress" => mol.stress = Some(decode_mat3_data(payload, type_tag, "stress")?),
        "name" => {
            mol.name = Some(
                std::str::from_utf8(payload)
                    .map_err(|_| Error::InvalidData("Invalid UTF-8 in name".into()))?
                    .to_string(),
            );
        }
        _ => {}
    }
    Ok(())
}

fn deserialize_molecule_soa_with_positions(bytes: &[u8], positions_type: u8) -> Result<Molecule> {
    if bytes.len() < 6 {
        return Err(Error::InvalidData("SOA record too small".into()));
    }

    let mut pos = 0;
    let n = u32::from_le_bytes(arr(&bytes[pos..pos + 4])?) as usize;
    pos += 4;

    let positions = decode_positions(bytes, &mut pos, n, positions_type)?;
    let z_end = pos
        .checked_add(n)
        .ok_or_else(|| Error::InvalidData("SOA atomic_numbers overflow".into()))?;
    if z_end > bytes.len() {
        return Err(Error::InvalidData(
            "SOA record truncated at atomic_numbers".into(),
        ));
    }
    let atomic_numbers = bytes[pos..z_end].to_vec();
    pos = z_end;

    if pos + 2 > bytes.len() {
        return Err(Error::InvalidData(
            "SOA record truncated at n_sections".into(),
        ));
    }
    let n_sections = u16::from_le_bytes(arr(&bytes[pos..pos + 2])?) as usize;
    pos += 2;

    let mut mol = match positions {
        Vec3Data::F32(values) => {
            Molecule::new(values, atomic_numbers).map_err(Error::InvalidData)?
        }
        Vec3Data::F64(values) => {
            Molecule::new_f64(values, atomic_numbers).map_err(Error::InvalidData)?
        }
    };

    for _ in 0..n_sections {
        if pos + 7 > bytes.len() {
            return Err(Error::InvalidData("SOA section header truncated".into()));
        }
        let kind = bytes[pos];
        pos += 1;
        let key_len = bytes[pos] as usize;
        pos += 1;
        if pos + key_len > bytes.len() {
            return Err(Error::InvalidData("SOA section key truncated".into()));
        }
        let key = std::str::from_utf8(&bytes[pos..pos + key_len])
            .map_err(|_| Error::InvalidData("Invalid UTF-8 in section key".into()))?;
        pos += key_len;
        let type_tag = bytes[pos];
        pos += 1;
        let payload_len = u32::from_le_bytes(arr(&bytes[pos..pos + 4])?) as usize;
        pos += 4;
        if pos + payload_len > bytes.len() {
            return Err(Error::InvalidData("SOA section payload truncated".into()));
        }
        let payload = &bytes[pos..pos + payload_len];
        pos += payload_len;

        match kind {
            KIND_BUILTIN => decode_builtin_section(&mut mol, key, type_tag, payload)?,
            KIND_ATOM_PROP => {
                mol.atom_properties
                    .insert(key.to_string(), decode_property_value(type_tag, payload)?);
            }
            KIND_MOL_PROP => {
                mol.properties
                    .insert(key.to_string(), decode_property_value(type_tag, payload)?);
            }
            _ => {}
        }
    }

    Ok(mol)
}

pub(super) fn deserialize_molecule_soa(
    bytes: &[u8],
    record_format: u32,
    positions_type_hint: Option<u8>,
) -> Result<Molecule> {
    let positions_type = resolve_positions_type(record_format, positions_type_hint)?;
    deserialize_molecule_soa_with_positions(bytes, positions_type)
}
