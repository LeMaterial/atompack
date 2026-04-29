use super::*;

/// Write a single tagged section: [kind:u8][key_len:u8][key][type_tag:u8][payload_len:u32][payload]
///
/// Errors instead of silently truncating when the key exceeds the 255-byte
/// `key_len: u8` field or the payload exceeds the `u32` length field.
fn write_section(
    buf: &mut Vec<u8>,
    kind: u8,
    key: &str,
    type_tag: u8,
    payload: &[u8],
) -> Result<()> {
    let key_len: u8 = key.len().try_into().map_err(|_| {
        Error::InvalidData(format!(
            "Section key '{}...' is {} bytes; max is 255",
            &key[..32.min(key.len())],
            key.len()
        ))
    })?;
    let payload_len: u32 = payload.len().try_into().map_err(|_| {
        Error::InvalidData(format!(
            "Section '{}' payload is {} bytes; max is u32::MAX",
            key,
            payload.len()
        ))
    })?;
    buf.push(kind);
    buf.push(key_len);
    buf.extend_from_slice(key.as_bytes());
    buf.push(type_tag);
    buf.extend_from_slice(&payload_len.to_le_bytes());
    buf.extend_from_slice(payload);
    Ok(())
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

pub(super) fn serialize_molecule_soa(molecule: &Molecule) -> Result<Vec<u8>> {
    let n = molecule.len();
    let mut buf = Vec::new();

    buf.extend_from_slice(&(n as u32).to_le_bytes());
    for position in &molecule.positions {
        buf.extend_from_slice(&position[0].to_le_bytes());
        buf.extend_from_slice(&position[1].to_le_bytes());
        buf.extend_from_slice(&position[2].to_le_bytes());
    }
    buf.extend_from_slice(&molecule.atomic_numbers);

    let mut n_sections: usize = 0;
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
    n_sections += molecule.atom_properties.len();
    n_sections += molecule.properties.len();
    let n_sections_u16: u16 = n_sections.try_into().map_err(|_| {
        Error::InvalidData(format!(
            "Molecule has {} sections; on-disk format limit is {}",
            n_sections,
            u16::MAX
        ))
    })?;
    buf.extend_from_slice(&n_sections_u16.to_le_bytes());

    if let Some(ref charges) = molecule.charges {
        let mut payload = Vec::with_capacity(charges.len() * 8);
        extend_f64(&mut payload, charges);
        write_section(&mut buf, KIND_BUILTIN, "charges", TYPE_F64_ARRAY, &payload)?;
    }
    if let Some(ref cell) = molecule.cell {
        let mut payload = Vec::with_capacity(72);
        for row in cell {
            extend_f64(&mut payload, row);
        }
        write_section(&mut buf, KIND_BUILTIN, "cell", TYPE_MAT3X3_F64, &payload)?;
    }
    if let Some(energy) = molecule.energy {
        write_section(
            &mut buf,
            KIND_BUILTIN,
            "energy",
            TYPE_FLOAT,
            &energy.to_le_bytes(),
        )?;
    }
    if let Some(ref forces) = molecule.forces {
        let mut payload = Vec::with_capacity(forces.len() * 12);
        for f in forces {
            extend_f32(&mut payload, f);
        }
        write_section(&mut buf, KIND_BUILTIN, "forces", TYPE_VEC3_F32, &payload)?;
    }
    if let Some(ref name) = molecule.name {
        write_section(&mut buf, KIND_BUILTIN, "name", TYPE_STRING, name.as_bytes())?;
    }
    if let Some(ref pbc) = molecule.pbc {
        let payload = [pbc[0] as u8, pbc[1] as u8, pbc[2] as u8];
        write_section(&mut buf, KIND_BUILTIN, "pbc", TYPE_BOOL3, &payload)?;
    }
    if let Some(ref stress) = molecule.stress {
        let mut payload = Vec::with_capacity(72);
        for row in stress {
            extend_f64(&mut payload, row);
        }
        write_section(&mut buf, KIND_BUILTIN, "stress", TYPE_MAT3X3_F64, &payload)?;
    }
    if let Some(ref velocities) = molecule.velocities {
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
        )?;
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
        )?;
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
        )?;
    }

    Ok(buf)
}

pub(super) fn deserialize_molecule_soa(bytes: &[u8]) -> Result<Molecule> {
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
                    mol.energy = Some(f64::from_le_bytes(arr(&payload[..8])?));
                }
                "forces" => mol.forces = Some(decode_vec3_f32(payload)?),
                "charges" => mol.charges = Some(decode_f64_array(payload)?),
                "velocities" => mol.velocities = Some(decode_vec3_f32(payload)?),
                "cell" => mol.cell = Some(decode_mat3x3_f64(payload)?),
                "pbc" => {
                    if payload.len() < 3 {
                        return Err(Error::InvalidData("pbc payload truncated".into()));
                    }
                    mol.pbc = Some([payload[0] != 0, payload[1] != 0, payload[2] != 0]);
                }
                "stress" => mol.stress = Some(decode_mat3x3_f64(payload)?),
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
