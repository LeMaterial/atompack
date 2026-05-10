use super::dtypes::{
    arr, decode_float_array_data, decode_float_scalar_data, decode_mat3_data,
    decode_property_value, decode_vec3_data, float_array_data_type_tag, float_array_payload_len,
    float_scalar_data_type_tag, float_scalar_payload_len, mat3_data_type_tag, mat3_payload_len,
    positions_type_from_molecule, property_value_payload_len, property_value_to_bytes,
    property_value_type_tag, validate_builtin_type_tag_for_record_format, vec3_data_type_tag,
    vec3_payload_len,
};
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

#[cfg(not(target_endian = "little"))]
fn extend_f64(buf: &mut Vec<u8>, values: &[f64]) {
    for value in values {
        buf.extend_from_slice(&f64::to_le_bytes(*value));
    }
}

#[cfg(not(target_endian = "little"))]
fn extend_f32(buf: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        buf.extend_from_slice(&f32::to_le_bytes(*value));
    }
}

fn write_vec3_section(buf: &mut Vec<u8>, key: &str, values: &Vec3Data) {
    match values {
        Vec3Data::F32(values) => {
            #[cfg(target_endian = "little")]
            {
                write_section(
                    buf,
                    KIND_BUILTIN,
                    key,
                    TYPE_VEC3_F32,
                    bytemuck::cast_slice::<[f32; 3], u8>(values),
                );
            }
            #[cfg(not(target_endian = "little"))]
            {
                let mut payload = Vec::with_capacity(values.len() * 12);
                for value in values {
                    extend_f32(&mut payload, value);
                }
                write_section(buf, KIND_BUILTIN, key, TYPE_VEC3_F32, &payload);
            }
        }
        Vec3Data::F64(values) => {
            #[cfg(target_endian = "little")]
            {
                write_section(
                    buf,
                    KIND_BUILTIN,
                    key,
                    TYPE_VEC3_F64,
                    bytemuck::cast_slice::<[f64; 3], u8>(values),
                );
            }
            #[cfg(not(target_endian = "little"))]
            {
                let mut payload = Vec::with_capacity(values.len() * 24);
                for value in values {
                    extend_f64(&mut payload, value);
                }
                write_section(buf, KIND_BUILTIN, key, TYPE_VEC3_F64, &payload);
            }
        }
    }
}

fn write_float_array_section(buf: &mut Vec<u8>, key: &str, values: &FloatArrayData) {
    match values {
        FloatArrayData::F32(values) => {
            #[cfg(target_endian = "little")]
            {
                write_section(
                    buf,
                    KIND_BUILTIN,
                    key,
                    TYPE_F32_ARRAY,
                    bytemuck::cast_slice::<f32, u8>(values),
                );
            }
            #[cfg(not(target_endian = "little"))]
            {
                let mut payload = Vec::with_capacity(values.len() * 4);
                extend_f32(&mut payload, values);
                write_section(buf, KIND_BUILTIN, key, TYPE_F32_ARRAY, &payload);
            }
        }
        FloatArrayData::F64(values) => {
            #[cfg(target_endian = "little")]
            {
                write_section(
                    buf,
                    KIND_BUILTIN,
                    key,
                    TYPE_F64_ARRAY,
                    bytemuck::cast_slice::<f64, u8>(values),
                );
            }
            #[cfg(not(target_endian = "little"))]
            {
                let mut payload = Vec::with_capacity(values.len() * 8);
                extend_f64(&mut payload, values);
                write_section(buf, KIND_BUILTIN, key, TYPE_F64_ARRAY, &payload);
            }
        }
    }
}

fn write_mat3_section(buf: &mut Vec<u8>, key: &str, values: &Mat3Data) {
    match values {
        Mat3Data::F32(values) => {
            #[cfg(target_endian = "little")]
            {
                write_section(
                    buf,
                    KIND_BUILTIN,
                    key,
                    TYPE_MAT3X3_F32,
                    bytemuck::cast_slice::<[f32; 3], u8>(values.as_slice()),
                );
            }
            #[cfg(not(target_endian = "little"))]
            {
                let mut payload = Vec::with_capacity(36);
                for row in values {
                    extend_f32(&mut payload, row);
                }
                write_section(buf, KIND_BUILTIN, key, TYPE_MAT3X3_F32, &payload);
            }
        }
        Mat3Data::F64(values) => {
            #[cfg(target_endian = "little")]
            {
                write_section(
                    buf,
                    KIND_BUILTIN,
                    key,
                    TYPE_MAT3X3_F64,
                    bytemuck::cast_slice::<[f64; 3], u8>(values.as_slice()),
                );
            }
            #[cfg(not(target_endian = "little"))]
            {
                let mut payload = Vec::with_capacity(72);
                for row in values {
                    extend_f64(&mut payload, row);
                }
                write_section(buf, KIND_BUILTIN, key, TYPE_MAT3X3_F64, &payload);
            }
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

#[derive(Clone, Copy)]
pub(super) struct SoaLayout {
    pub(super) positions_type: u8,
    pub(super) positions_stride: usize,
}

pub(super) fn resolve_layout(
    record_format: u32,
    positions_type_hint: Option<u8>,
) -> Result<SoaLayout> {
    let positions_type = resolve_positions_type(record_format, positions_type_hint)?;
    let positions_stride = positions_stride(positions_type)?;
    Ok(SoaLayout {
        positions_type,
        positions_stride,
    })
}

fn validate_record_format_compat(molecule: &Molecule, record_format: u32) -> Result<()> {
    resolve_positions_type(record_format, Some(positions_type_from_molecule(molecule)))?;
    if record_format == RECORD_FORMAT_SOA_V3 {
        return Ok(());
    }

    if let Some(charges) = &molecule.charges {
        validate_builtin_type_tag_for_record_format(
            record_format,
            "charges",
            float_array_data_type_tag(charges),
        )?;
    }
    if let Some(cell) = &molecule.cell {
        validate_builtin_type_tag_for_record_format(
            record_format,
            "cell",
            mat3_data_type_tag(cell),
        )?;
    }
    if let Some(energy) = &molecule.energy {
        validate_builtin_type_tag_for_record_format(
            record_format,
            "energy",
            float_scalar_data_type_tag(energy),
        )?;
    }
    if let Some(forces) = &molecule.forces {
        validate_builtin_type_tag_for_record_format(
            record_format,
            "forces",
            vec3_data_type_tag(forces),
        )?;
    }
    if let Some(stress) = &molecule.stress {
        validate_builtin_type_tag_for_record_format(
            record_format,
            "stress",
            mat3_data_type_tag(stress),
        )?;
    }
    if let Some(velocities) = &molecule.velocities {
        validate_builtin_type_tag_for_record_format(
            record_format,
            "velocities",
            vec3_data_type_tag(velocities),
        )?;
    }
    Ok(())
}

pub(super) fn minimum_record_format_for_molecule(molecule: &Molecule) -> u32 {
    if validate_record_format_compat(molecule, RECORD_FORMAT_SOA_V2).is_ok() {
        RECORD_FORMAT_SOA_V2
    } else {
        RECORD_FORMAT_SOA_V3
    }
}

fn write_positions(buf: &mut Vec<u8>, positions: &Vec3Data) {
    match positions {
        Vec3Data::F32(values) => {
            #[cfg(target_endian = "little")]
            {
                buf.extend_from_slice(bytemuck::cast_slice::<[f32; 3], u8>(values));
            }
            #[cfg(not(target_endian = "little"))]
            {
                for value in values {
                    extend_f32(buf, value);
                }
            }
        }
        Vec3Data::F64(values) => {
            #[cfg(target_endian = "little")]
            {
                buf.extend_from_slice(bytemuck::cast_slice::<[f64; 3], u8>(values));
            }
            #[cfg(not(target_endian = "little"))]
            {
                for value in values {
                    extend_f64(buf, value);
                }
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

fn section_size(key_len: usize, payload_len: usize) -> usize {
    1 + 1 + key_len + 1 + 4 + payload_len
}

fn estimate_serialized_len(molecule: &Molecule) -> usize {
    let positions_bytes = vec3_payload_len(&molecule.positions);
    let mut total = 4 + positions_bytes + molecule.atomic_numbers.len() + 2;

    if let Some(charges) = &molecule.charges {
        total += section_size("charges".len(), float_array_payload_len(charges));
    }
    if let Some(cell) = &molecule.cell {
        total += section_size("cell".len(), mat3_payload_len(cell));
    }
    if let Some(energy) = &molecule.energy {
        total += section_size("energy".len(), float_scalar_payload_len(energy));
    }
    if let Some(forces) = &molecule.forces {
        total += section_size("forces".len(), vec3_payload_len(forces));
    }
    if let Some(name) = &molecule.name {
        total += section_size("name".len(), name.len());
    }
    if molecule.pbc.is_some() {
        total += section_size("pbc".len(), 3);
    }
    if let Some(stress) = &molecule.stress {
        total += section_size("stress".len(), mat3_payload_len(stress));
    }
    if let Some(velocities) = &molecule.velocities {
        total += section_size("velocities".len(), vec3_payload_len(velocities));
    }

    for (key, value) in &molecule.atom_properties {
        total += section_size(key.len(), property_value_payload_len(value));
    }
    for (key, value) in &molecule.properties {
        total += section_size(key.len(), property_value_payload_len(value));
    }

    total
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

pub(super) fn serialize_molecule_soa(molecule: &Molecule, record_format: u32) -> Result<Vec<u8>> {
    validate_record_format_compat(molecule, record_format)?;

    let n = molecule.len();
    let mut buf = Vec::with_capacity(estimate_serialized_len(molecule));

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
    layout: SoaLayout,
) -> Result<Vec3Data> {
    let positions_len = n_atoms
        .checked_mul(layout.positions_stride)
        .ok_or_else(|| Error::InvalidData("SOA positions overflow".into()))?;
    let positions_end = pos
        .checked_add(positions_len)
        .ok_or_else(|| Error::InvalidData("SOA positions overflow".into()))?;
    if positions_end > bytes.len() {
        return Err(Error::InvalidData(
            "SOA record truncated at positions".into(),
        ));
    }
    let payload = &bytes[*pos..positions_end];
    let positions = decode_vec3_data(payload, layout.positions_type, "positions")?;
    *pos = positions_end;
    Ok(positions)
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

fn deserialize_molecule_soa_with_layout(bytes: &[u8], layout: SoaLayout) -> Result<Molecule> {
    if bytes.len() < 6 {
        return Err(Error::InvalidData("SOA record too small".into()));
    }

    let mut pos = 0;
    let n = u32::from_le_bytes(arr(&bytes[pos..pos + 4])?) as usize;
    pos += 4;

    let positions = decode_positions(bytes, &mut pos, n, layout)?;
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
    deserialize_molecule_soa_with_layout(bytes, resolve_layout(record_format, positions_type_hint)?)
}
