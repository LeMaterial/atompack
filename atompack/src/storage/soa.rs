use super::*;
use crate::atom::{FloatArrayData, FloatScalarData, Mat3Data, Vec3Data};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct SchemaLock {
    pub(super) positions_type: Option<u8>,
    pub(super) sections: BTreeMap<(u8, String), SchemaEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SchemaEntry {
    pub(super) type_tag: u8,
    pub(super) per_atom: bool,
    pub(super) elem_bytes: usize,
    pub(super) slot_bytes: usize,
}

/// Write a single tagged section: [kind:u8][key_len:u8][key][type_tag:u8][payload_len:u32][payload]
fn write_section(buf: &mut Vec<u8>, kind: u8, key: &str, type_tag: u8, payload: &[u8]) {
    buf.push(kind);
    buf.push(key.len() as u8);
    buf.extend_from_slice(key.as_bytes());
    buf.push(type_tag);
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(payload);
}

fn property_value_type_tag(value: &PropertyValue) -> u8 {
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

fn property_value_to_bytes(value: &PropertyValue) -> Vec<u8> {
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

fn serialize_molecule_soa_v2(molecule: &Molecule) -> Result<Vec<u8>> {
    let n = molecule.len();
    let mut buf = Vec::new();

    buf.extend_from_slice(&(n as u32).to_le_bytes());
    let positions = match &molecule.positions {
        Vec3Data::F32(values) => values,
        Vec3Data::F64(_) => {
            return Err(Error::InvalidData(
                "record format 2 does not support float64 positions".into(),
            ));
        }
    };
    for position in positions {
        buf.extend_from_slice(&position[0].to_le_bytes());
        buf.extend_from_slice(&position[1].to_le_bytes());
        buf.extend_from_slice(&position[2].to_le_bytes());
    }
    buf.extend_from_slice(&molecule.atomic_numbers);

    let mut n_sections: u16 = 0;
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
    buf.extend_from_slice(&n_sections.to_le_bytes());

    if let Some(ref charges) = molecule.charges {
        let charges = match charges {
            FloatArrayData::F64(values) => values,
            FloatArrayData::F32(_) => {
                return Err(Error::InvalidData(
                    "record format 2 does not support float32 charges".into(),
                ));
            }
        };
        let mut payload = Vec::with_capacity(charges.len() * 8);
        extend_f64(&mut payload, charges);
        write_section(&mut buf, KIND_BUILTIN, "charges", TYPE_F64_ARRAY, &payload);
    }
    if let Some(ref cell) = molecule.cell {
        let cell = match cell {
            Mat3Data::F64(values) => values,
            Mat3Data::F32(_) => {
                return Err(Error::InvalidData(
                    "record format 2 does not support float32 cell".into(),
                ));
            }
        };
        let mut payload = Vec::with_capacity(72);
        for row in cell {
            extend_f64(&mut payload, row);
        }
        write_section(&mut buf, KIND_BUILTIN, "cell", TYPE_MAT3X3_F64, &payload);
    }
    if let Some(ref energy) = molecule.energy {
        let energy = match energy {
            FloatScalarData::F64(value) => value,
            FloatScalarData::F32(_) => {
                return Err(Error::InvalidData(
                    "record format 2 does not support float32 energy".into(),
                ));
            }
        };
        write_section(
            &mut buf,
            KIND_BUILTIN,
            "energy",
            TYPE_FLOAT,
            &energy.to_le_bytes(),
        );
    }
    if let Some(ref forces) = molecule.forces {
        let forces = match forces {
            Vec3Data::F32(values) => values,
            Vec3Data::F64(_) => {
                return Err(Error::InvalidData(
                    "record format 2 does not support float64 forces".into(),
                ));
            }
        };
        let mut payload = Vec::with_capacity(forces.len() * 12);
        for f in forces {
            extend_f32(&mut payload, f);
        }
        write_section(&mut buf, KIND_BUILTIN, "forces", TYPE_VEC3_F32, &payload);
    }
    if let Some(ref name) = molecule.name {
        write_section(&mut buf, KIND_BUILTIN, "name", TYPE_STRING, name.as_bytes());
    }
    if let Some(ref pbc) = molecule.pbc {
        let payload = [pbc[0] as u8, pbc[1] as u8, pbc[2] as u8];
        write_section(&mut buf, KIND_BUILTIN, "pbc", TYPE_BOOL3, &payload);
    }
    if let Some(ref stress) = molecule.stress {
        let stress = match stress {
            Mat3Data::F64(values) => values,
            Mat3Data::F32(_) => {
                return Err(Error::InvalidData(
                    "record format 2 does not support float32 stress".into(),
                ));
            }
        };
        let mut payload = Vec::with_capacity(72);
        for row in stress {
            extend_f64(&mut payload, row);
        }
        write_section(&mut buf, KIND_BUILTIN, "stress", TYPE_MAT3X3_F64, &payload);
    }
    if let Some(ref velocities) = molecule.velocities {
        let velocities = match velocities {
            Vec3Data::F32(values) => values,
            Vec3Data::F64(_) => {
                return Err(Error::InvalidData(
                    "record format 2 does not support float64 velocities".into(),
                ));
            }
        };
        let mut payload = Vec::with_capacity(velocities.len() * 12);
        for v in velocities {
            extend_f32(&mut payload, v);
        }
        write_section(
            &mut buf,
            KIND_BUILTIN,
            "velocities",
            TYPE_VEC3_F32,
            &payload,
        );
    }

    let mut atom_keys: Vec<&String> = molecule.atom_properties.keys().collect();
    atom_keys.sort();
    for key in atom_keys {
        let value = &molecule.atom_properties[key];
        let payload = property_value_to_bytes(value);
        write_section(
            &mut buf,
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
            &mut buf,
            KIND_MOL_PROP,
            key,
            property_value_type_tag(value),
            &payload,
        );
    }

    Ok(buf)
}

fn serialize_molecule_soa_v3(molecule: &Molecule) -> Result<Vec<u8>> {
    let n = molecule.len();
    let mut buf = Vec::new();

    buf.extend_from_slice(&(n as u32).to_le_bytes());
    match &molecule.positions {
        Vec3Data::F32(values) => {
            buf.push(TYPE_VEC3_F32);
            for value in values {
                buf.extend_from_slice(&value[0].to_le_bytes());
                buf.extend_from_slice(&value[1].to_le_bytes());
                buf.extend_from_slice(&value[2].to_le_bytes());
            }
        }
        Vec3Data::F64(values) => {
            buf.push(TYPE_VEC3_F64);
            for value in values {
                buf.extend_from_slice(&value[0].to_le_bytes());
                buf.extend_from_slice(&value[1].to_le_bytes());
                buf.extend_from_slice(&value[2].to_le_bytes());
            }
        }
    }
    buf.extend_from_slice(&molecule.atomic_numbers);

    let mut n_sections: u16 = 0;
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
    buf.extend_from_slice(&n_sections.to_le_bytes());

    if let Some(ref charges) = molecule.charges {
        write_float_array_section(&mut buf, "charges", charges);
    }
    if let Some(ref cell) = molecule.cell {
        write_mat3_section(&mut buf, "cell", cell);
    }
    if let Some(ref energy) = molecule.energy {
        write_energy_section(&mut buf, energy);
    }
    if let Some(ref forces) = molecule.forces {
        write_vec3_section(&mut buf, "forces", forces);
    }
    if let Some(ref name) = molecule.name {
        write_section(&mut buf, KIND_BUILTIN, "name", TYPE_STRING, name.as_bytes());
    }
    if let Some(ref pbc) = molecule.pbc {
        let payload = [pbc[0] as u8, pbc[1] as u8, pbc[2] as u8];
        write_section(&mut buf, KIND_BUILTIN, "pbc", TYPE_BOOL3, &payload);
    }
    if let Some(ref stress) = molecule.stress {
        write_mat3_section(&mut buf, "stress", stress);
    }
    if let Some(ref velocities) = molecule.velocities {
        write_vec3_section(&mut buf, "velocities", velocities);
    }

    let mut atom_keys: Vec<&String> = molecule.atom_properties.keys().collect();
    atom_keys.sort();
    for key in atom_keys {
        let value = &molecule.atom_properties[key];
        let payload = property_value_to_bytes(value);
        write_section(
            &mut buf,
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
            &mut buf,
            KIND_MOL_PROP,
            key,
            property_value_type_tag(value),
            &payload,
        );
    }

    Ok(buf)
}

pub(super) fn serialize_molecule_soa(molecule: &Molecule, record_format: u32) -> Result<Vec<u8>> {
    match record_format {
        RECORD_FORMAT_SOA_V2 => serialize_molecule_soa_v2(molecule),
        RECORD_FORMAT_SOA_V3 => serialize_molecule_soa_v3(molecule),
        _ => Err(Error::InvalidData(format!(
            "Unsupported record format {}",
            record_format
        ))),
    }
}

fn deserialize_molecule_soa_v2(bytes: &[u8]) -> Result<Molecule> {
    if bytes.len() < 6 {
        return Err(Error::InvalidData("SOA record too small".into()));
    }

    let mut pos = 0;
    let n = u32::from_le_bytes(arr(&bytes[pos..pos + 4])?) as usize;
    pos += 4;

    let positions_end = pos + n * 12;
    if positions_end > bytes.len() {
        return Err(Error::InvalidData(
            "SOA record truncated at positions".into(),
        ));
    }
    let mut positions = Vec::with_capacity(n);
    for i in 0..n {
        let o = pos + i * 12;
        let x = f32::from_le_bytes(arr(&bytes[o..o + 4])?);
        let y = f32::from_le_bytes(arr(&bytes[o + 4..o + 8])?);
        let z = f32::from_le_bytes(arr(&bytes[o + 8..o + 12])?);
        positions.push([x, y, z]);
    }
    pos = positions_end;

    let z_end = pos + n;
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

    let mut mol = Molecule::new(positions, atomic_numbers).map_err(Error::InvalidData)?;

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
            KIND_BUILTIN => match key {
                "energy" => {
                    if payload.len() < 8 {
                        return Err(Error::InvalidData("energy payload truncated".into()));
                    }
                    mol.energy = Some(FloatScalarData::F64(f64::from_le_bytes(
                        arr(&payload[..8])?,
                    )));
                }
                "forces" => mol.forces = Some(Vec3Data::F32(decode_vec3_f32(payload)?)),
                "charges" => mol.charges = Some(FloatArrayData::F64(decode_f64_array(payload)?)),
                "velocities" => mol.velocities = Some(Vec3Data::F32(decode_vec3_f32(payload)?)),
                "cell" => mol.cell = Some(Mat3Data::F64(decode_mat3x3_f64(payload)?)),
                "pbc" => {
                    if payload.len() < 3 {
                        return Err(Error::InvalidData("pbc payload truncated".into()));
                    }
                    mol.pbc = Some([payload[0] != 0, payload[1] != 0, payload[2] != 0]);
                }
                "stress" => mol.stress = Some(Mat3Data::F64(decode_mat3x3_f64(payload)?)),
                "name" => {
                    mol.name = Some(
                        std::str::from_utf8(payload)
                            .map_err(|_| Error::InvalidData("Invalid UTF-8 in name".into()))?
                            .to_string(),
                    )
                }
                _ => {}
            },
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

fn deserialize_molecule_soa_v3(bytes: &[u8]) -> Result<Molecule> {
    if bytes.len() < 7 {
        return Err(Error::InvalidData("SOA v3 record too small".into()));
    }

    let mut pos = 0;
    let n = u32::from_le_bytes(arr(&bytes[pos..pos + 4])?) as usize;
    pos += 4;

    let positions_type = bytes[pos];
    pos += 1;
    let positions = match positions_type {
        TYPE_VEC3_F32 => {
            let positions_end = pos + n * 12;
            if positions_end > bytes.len() {
                return Err(Error::InvalidData(
                    "SOA v3 record truncated at positions".into(),
                ));
            }
            let values = decode_vec3_f32(&bytes[pos..positions_end])?;
            pos = positions_end;
            Vec3Data::F32(values)
        }
        TYPE_VEC3_F64 => {
            let positions_end = pos + n * 24;
            if positions_end > bytes.len() {
                return Err(Error::InvalidData(
                    "SOA v3 record truncated at positions".into(),
                ));
            }
            let values = decode_vec3_f64(&bytes[pos..positions_end])?;
            pos = positions_end;
            Vec3Data::F64(values)
        }
        _ => {
            return Err(Error::InvalidData(format!(
                "Unsupported positions type tag {} in record format 3",
                positions_type
            )));
        }
    };

    let z_end = pos + n;
    if z_end > bytes.len() {
        return Err(Error::InvalidData(
            "SOA v3 record truncated at atomic_numbers".into(),
        ));
    }
    let atomic_numbers = bytes[pos..z_end].to_vec();
    pos = z_end;

    if pos + 2 > bytes.len() {
        return Err(Error::InvalidData(
            "SOA v3 record truncated at n_sections".into(),
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
            KIND_BUILTIN => match key {
                "energy" => match type_tag {
                    TYPE_FLOAT => {
                        if payload.len() != 8 {
                            return Err(Error::InvalidData("energy f64 payload truncated".into()));
                        }
                        mol.energy = Some(FloatScalarData::F64(f64::from_le_bytes(arr(payload)?)));
                    }
                    TYPE_FLOAT32 => {
                        if payload.len() != 4 {
                            return Err(Error::InvalidData("energy f32 payload truncated".into()));
                        }
                        mol.energy = Some(FloatScalarData::F32(f32::from_le_bytes(arr(payload)?)));
                    }
                    _ => {
                        return Err(Error::InvalidData(format!(
                            "Unsupported energy type tag {}",
                            type_tag
                        )));
                    }
                },
                "forces" => match type_tag {
                    TYPE_VEC3_F32 => mol.forces = Some(Vec3Data::F32(decode_vec3_f32(payload)?)),
                    TYPE_VEC3_F64 => mol.forces = Some(Vec3Data::F64(decode_vec3_f64(payload)?)),
                    _ => {
                        return Err(Error::InvalidData(format!(
                            "Unsupported forces type tag {}",
                            type_tag
                        )));
                    }
                },
                "charges" => match type_tag {
                    TYPE_F32_ARRAY => {
                        mol.charges = Some(FloatArrayData::F32(decode_f32_array(payload)?))
                    }
                    TYPE_F64_ARRAY => {
                        mol.charges = Some(FloatArrayData::F64(decode_f64_array(payload)?))
                    }
                    _ => {
                        return Err(Error::InvalidData(format!(
                            "Unsupported charges type tag {}",
                            type_tag
                        )));
                    }
                },
                "velocities" => match type_tag {
                    TYPE_VEC3_F32 => {
                        mol.velocities = Some(Vec3Data::F32(decode_vec3_f32(payload)?))
                    }
                    TYPE_VEC3_F64 => {
                        mol.velocities = Some(Vec3Data::F64(decode_vec3_f64(payload)?))
                    }
                    _ => {
                        return Err(Error::InvalidData(format!(
                            "Unsupported velocities type tag {}",
                            type_tag
                        )));
                    }
                },
                "cell" => match type_tag {
                    TYPE_MAT3X3_F32 => mol.cell = Some(Mat3Data::F32(decode_mat3x3_f32(payload)?)),
                    TYPE_MAT3X3_F64 => mol.cell = Some(Mat3Data::F64(decode_mat3x3_f64(payload)?)),
                    _ => {
                        return Err(Error::InvalidData(format!(
                            "Unsupported cell type tag {}",
                            type_tag
                        )));
                    }
                },
                "pbc" => {
                    if payload.len() < 3 {
                        return Err(Error::InvalidData("pbc payload truncated".into()));
                    }
                    mol.pbc = Some([payload[0] != 0, payload[1] != 0, payload[2] != 0]);
                }
                "stress" => match type_tag {
                    TYPE_MAT3X3_F32 => {
                        mol.stress = Some(Mat3Data::F32(decode_mat3x3_f32(payload)?))
                    }
                    TYPE_MAT3X3_F64 => {
                        mol.stress = Some(Mat3Data::F64(decode_mat3x3_f64(payload)?))
                    }
                    _ => {
                        return Err(Error::InvalidData(format!(
                            "Unsupported stress type tag {}",
                            type_tag
                        )));
                    }
                },
                "name" => {
                    mol.name = Some(
                        std::str::from_utf8(payload)
                            .map_err(|_| Error::InvalidData("Invalid UTF-8 in name".into()))?
                            .to_string(),
                    )
                }
                _ => {}
            },
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

pub(super) fn deserialize_molecule_soa(bytes: &[u8], record_format: u32) -> Result<Molecule> {
    match record_format {
        RECORD_FORMAT_SOA_V2 => deserialize_molecule_soa_v2(bytes),
        RECORD_FORMAT_SOA_V3 => deserialize_molecule_soa_v3(bytes),
        _ => Err(Error::InvalidData(format!(
            "Unsupported record format {}",
            record_format
        ))),
    }
}

fn schema_type_tag_elem_bytes(tag: u8) -> Result<usize> {
    match tag {
        TYPE_FLOAT => Ok(8),
        TYPE_INT => Ok(8),
        TYPE_STRING => Ok(0),
        TYPE_F64_ARRAY => Ok(8),
        TYPE_VEC3_F32 => Ok(12),
        TYPE_I64_ARRAY => Ok(8),
        TYPE_F32_ARRAY => Ok(4),
        TYPE_VEC3_F64 => Ok(24),
        TYPE_I32_ARRAY => Ok(4),
        TYPE_BOOL3 => Ok(3),
        TYPE_MAT3X3_F64 => Ok(72),
        TYPE_FLOAT32 => Ok(4),
        TYPE_MAT3X3_F32 => Ok(36),
        _ => Err(Error::InvalidData(format!(
            "Unsupported section type tag {}",
            tag
        ))),
    }
}

fn schema_is_per_atom(kind: u8, key: &str) -> bool {
    match kind {
        KIND_ATOM_PROP => true,
        KIND_MOL_PROP => false,
        KIND_BUILTIN => matches!(key, "forces" | "charges" | "velocities"),
        _ => false,
    }
}

fn schema_entry(
    kind: u8,
    key: &str,
    type_tag: u8,
    payload_len: usize,
    n_atoms: usize,
) -> Result<SchemaEntry> {
    let per_atom = schema_is_per_atom(kind, key);
    let elem_bytes = schema_type_tag_elem_bytes(type_tag)?;
    let slot_bytes = if type_tag == TYPE_STRING {
        0
    } else if per_atom {
        match type_tag {
            TYPE_F64_ARRAY | TYPE_I64_ARRAY | TYPE_F32_ARRAY | TYPE_I32_ARRAY => elem_bytes,
            TYPE_VEC3_F32 | TYPE_VEC3_F64 => elem_bytes,
            _ => payload_len
                .checked_div(n_atoms.max(1))
                .unwrap_or(elem_bytes),
        }
    } else {
        payload_len
    };

    if per_atom {
        let expected = elem_bytes
            .checked_mul(n_atoms)
            .ok_or_else(|| Error::InvalidData(format!("Schema overflow for section '{}'", key)))?;
        if payload_len != expected {
            return Err(Error::InvalidData(format!(
                "Section '{}' payload length {} does not match expected {}",
                key, payload_len, expected
            )));
        }
    }

    Ok(SchemaEntry {
        type_tag,
        per_atom,
        elem_bytes,
        slot_bytes,
    })
}

fn parse_record_schema_v2(bytes: &[u8]) -> Result<SchemaLock> {
    if bytes.len() < 6 {
        return Err(Error::InvalidData("SOA record too small".into()));
    }

    let mut pos = 0usize;
    let n_atoms = u32::from_le_bytes(arr(&bytes[pos..pos + 4])?) as usize;
    pos += 4;

    let positions_end = pos
        .checked_add(
            n_atoms
                .checked_mul(12)
                .ok_or_else(|| Error::InvalidData("SOA positions overflow".into()))?,
        )
        .ok_or_else(|| Error::InvalidData("SOA positions overflow".into()))?;
    if positions_end > bytes.len() {
        return Err(Error::InvalidData(
            "SOA record truncated at positions".into(),
        ));
    }
    pos = positions_end;

    let z_end = pos
        .checked_add(n_atoms)
        .ok_or_else(|| Error::InvalidData("SOA atomic_numbers overflow".into()))?;
    if z_end > bytes.len() {
        return Err(Error::InvalidData(
            "SOA record truncated at atomic_numbers".into(),
        ));
    }
    pos = z_end;

    if pos + 2 > bytes.len() {
        return Err(Error::InvalidData(
            "SOA record truncated at n_sections".into(),
        ));
    }
    let n_sections = u16::from_le_bytes(arr(&bytes[pos..pos + 2])?) as usize;
    pos += 2;

    let mut schema = SchemaLock {
        positions_type: Some(TYPE_VEC3_F32),
        sections: BTreeMap::new(),
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
            .map_err(|_| Error::InvalidData("Invalid UTF-8 in section key".into()))?
            .to_string();
        pos += key_len;
        let type_tag = bytes[pos];
        pos += 1;
        let payload_len = u32::from_le_bytes(arr(&bytes[pos..pos + 4])?) as usize;
        pos += 4;
        let payload_end = pos
            .checked_add(payload_len)
            .ok_or_else(|| Error::InvalidData("SOA section payload overflow".into()))?;
        if payload_end > bytes.len() {
            return Err(Error::InvalidData("SOA section payload truncated".into()));
        }
        pos = payload_end;
        let entry = schema_entry(kind, &key, type_tag, payload_len, n_atoms)?;
        schema.sections.insert((kind, key), entry);
    }

    Ok(schema)
}

fn parse_record_schema_v3(bytes: &[u8]) -> Result<SchemaLock> {
    if bytes.len() < 7 {
        return Err(Error::InvalidData("SOA record too small".into()));
    }

    let mut pos = 0usize;
    let n_atoms = u32::from_le_bytes(arr(&bytes[pos..pos + 4])?) as usize;
    pos += 4;

    let positions_type = bytes[pos];
    pos += 1;
    let positions_elem_bytes = match positions_type {
        TYPE_VEC3_F32 => 12usize,
        TYPE_VEC3_F64 => 24usize,
        _ => {
            return Err(Error::InvalidData(format!(
                "Unsupported positions type tag {}",
                positions_type
            )));
        }
    };

    let positions_end = pos
        .checked_add(
            n_atoms
                .checked_mul(positions_elem_bytes)
                .ok_or_else(|| Error::InvalidData("SOA positions overflow".into()))?,
        )
        .ok_or_else(|| Error::InvalidData("SOA positions overflow".into()))?;
    if positions_end > bytes.len() {
        return Err(Error::InvalidData(
            "SOA record truncated at positions".into(),
        ));
    }
    pos = positions_end;

    let z_end = pos
        .checked_add(n_atoms)
        .ok_or_else(|| Error::InvalidData("SOA atomic_numbers overflow".into()))?;
    if z_end > bytes.len() {
        return Err(Error::InvalidData(
            "SOA record truncated at atomic_numbers".into(),
        ));
    }
    pos = z_end;

    if pos + 2 > bytes.len() {
        return Err(Error::InvalidData(
            "SOA record truncated at n_sections".into(),
        ));
    }
    let n_sections = u16::from_le_bytes(arr(&bytes[pos..pos + 2])?) as usize;
    pos += 2;

    let mut schema = SchemaLock {
        positions_type: Some(positions_type),
        sections: BTreeMap::new(),
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
            .map_err(|_| Error::InvalidData("Invalid UTF-8 in section key".into()))?
            .to_string();
        pos += key_len;
        let type_tag = bytes[pos];
        pos += 1;
        let payload_len = u32::from_le_bytes(arr(&bytes[pos..pos + 4])?) as usize;
        pos += 4;
        let payload_end = pos
            .checked_add(payload_len)
            .ok_or_else(|| Error::InvalidData("SOA section payload overflow".into()))?;
        if payload_end > bytes.len() {
            return Err(Error::InvalidData("SOA section payload truncated".into()));
        }
        pos = payload_end;
        let entry = schema_entry(kind, &key, type_tag, payload_len, n_atoms)?;
        schema.sections.insert((kind, key), entry);
    }

    Ok(schema)
}

pub(super) fn record_schema(bytes: &[u8], record_format: u32) -> Result<SchemaLock> {
    match record_format {
        RECORD_FORMAT_SOA_V2 => parse_record_schema_v2(bytes),
        RECORD_FORMAT_SOA_V3 => parse_record_schema_v3(bytes),
        _ => Err(Error::InvalidData(format!(
            "Unsupported record format {}",
            record_format
        ))),
    }
}

pub(super) fn merge_schema_lock(lock: &mut SchemaLock, record: &SchemaLock) -> Result<()> {
    match (lock.positions_type, record.positions_type) {
        (None, Some(tag)) => lock.positions_type = Some(tag),
        (Some(expected), Some(actual)) if expected != actual => {
            return Err(Error::InvalidData(format!(
                "Position dtype mismatch: expected type tag {}, got {}",
                expected, actual
            )));
        }
        _ => {}
    }

    for ((kind, key), entry) in &record.sections {
        match lock.sections.get(&(*kind, key.clone())) {
            Some(expected) if expected != entry => {
                return Err(Error::InvalidData(format!(
                    "Schema mismatch for section '{}': expected {:?}, got {:?}",
                    key, expected, entry
                )));
            }
            Some(_) => {}
            None => {
                lock.sections.insert((*kind, key.clone()), entry.clone());
            }
        }
    }

    Ok(())
}
