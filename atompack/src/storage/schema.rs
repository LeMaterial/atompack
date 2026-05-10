use super::dtypes::{
    arr, float_array_data_type_tag, float_array_payload_len, float_scalar_data_type_tag,
    float_scalar_payload_len, mat3_data_type_tag, mat3_payload_len, positions_type_from_molecule,
    property_value_payload_len, property_value_type_tag,
    validate_builtin_type_tag_for_record_format, vec3_data_type_tag, vec3_payload_len,
};
use super::soa::{SoaLayout, resolve_layout};
use super::*;
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

const SCHEMA_BLOB_VERSION: u32 = 1;

pub(super) fn encode_schema_lock(lock: &SchemaLock) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&SCHEMA_BLOB_VERSION.to_le_bytes());
    buf.push(lock.positions_type.unwrap_or(255));
    buf.extend_from_slice(&(lock.sections.len() as u32).to_le_bytes());
    for ((kind, key), entry) in &lock.sections {
        let key_len: u16 = key
            .len()
            .try_into()
            .map_err(|_| Error::InvalidData(format!("Schema key '{}' is too long", key)))?;
        let elem_bytes: u32 = entry
            .elem_bytes
            .try_into()
            .map_err(|_| Error::InvalidData(format!("Schema elem_bytes overflow for '{}'", key)))?;
        let slot_bytes: u32 = entry
            .slot_bytes
            .try_into()
            .map_err(|_| Error::InvalidData(format!("Schema slot_bytes overflow for '{}'", key)))?;
        buf.push(*kind);
        buf.push(entry.type_tag);
        buf.push(u8::from(entry.per_atom));
        buf.extend_from_slice(&key_len.to_le_bytes());
        buf.extend_from_slice(&elem_bytes.to_le_bytes());
        buf.extend_from_slice(&slot_bytes.to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
    }
    Ok(buf)
}

pub(super) fn decode_schema_lock(bytes: &[u8]) -> Result<SchemaLock> {
    if bytes.len() < 9 {
        return Err(Error::InvalidData("Schema blob too small".into()));
    }
    let version = u32::from_le_bytes(arr(&bytes[0..4])?);
    if version != SCHEMA_BLOB_VERSION {
        return Err(Error::InvalidData(format!(
            "Unsupported schema blob version {}",
            version
        )));
    }
    let positions_type = match bytes[4] {
        255 => None,
        TYPE_VEC3_F32 | TYPE_VEC3_F64 => Some(bytes[4]),
        other => {
            return Err(Error::InvalidData(format!(
                "Unsupported schema positions type tag {}",
                other
            )));
        }
    };
    let count = u32::from_le_bytes(arr(&bytes[5..9])?) as usize;
    let mut pos = 9usize;
    let mut sections = BTreeMap::new();
    for _ in 0..count {
        if pos + 13 > bytes.len() {
            return Err(Error::InvalidData("Schema blob truncated".into()));
        }
        let kind = bytes[pos];
        pos += 1;
        let type_tag = bytes[pos];
        pos += 1;
        let per_atom = match bytes[pos] {
            0 => false,
            1 => true,
            _ => return Err(Error::InvalidData("Invalid schema per_atom flag".into())),
        };
        pos += 1;
        let key_len = u16::from_le_bytes(arr(&bytes[pos..pos + 2])?) as usize;
        pos += 2;
        let elem_bytes = u32::from_le_bytes(arr(&bytes[pos..pos + 4])?) as usize;
        pos += 4;
        let slot_bytes = u32::from_le_bytes(arr(&bytes[pos..pos + 4])?) as usize;
        pos += 4;
        if pos + key_len > bytes.len() {
            return Err(Error::InvalidData("Schema blob truncated at key".into()));
        }
        let key = std::str::from_utf8(&bytes[pos..pos + key_len])
            .map_err(|_| Error::InvalidData("Invalid UTF-8 in schema key".into()))?
            .to_string();
        pos += key_len;
        sections.insert(
            (kind, key),
            SchemaEntry {
                type_tag,
                per_atom,
                elem_bytes,
                slot_bytes,
            },
        );
    }
    if pos != bytes.len() {
        return Err(Error::InvalidData("Schema blob trailing bytes".into()));
    }
    Ok(SchemaLock {
        positions_type,
        sections,
    })
}

fn schema_type_tag_elem_bytes(tag: u8) -> Result<usize> {
    match tag {
        TYPE_NONE => Ok(0),
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
    let slot_bytes = if matches!(type_tag, TYPE_STRING | TYPE_NONE) {
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

pub(super) fn validate_schema_lock_for_record_format(
    record_format: u32,
    schema: &SchemaLock,
) -> Result<()> {
    let _ = resolve_layout(record_format, schema.positions_type)?;
    for ((kind, key), entry) in &schema.sections {
        if *kind == KIND_BUILTIN {
            validate_builtin_type_tag_for_record_format(record_format, key, entry.type_tag)?;
        }
    }
    Ok(())
}

pub(super) fn schema_from_molecule(molecule: &Molecule) -> Result<SchemaLock> {
    let n_atoms = molecule.len();
    let mut schema = SchemaLock {
        positions_type: Some(positions_type_from_molecule(molecule)),
        sections: BTreeMap::new(),
    };

    let mut insert = |kind: u8, key: &str, type_tag: u8, payload_len: usize| -> Result<()> {
        let entry = schema_entry(kind, key, type_tag, payload_len, n_atoms)?;
        schema.sections.insert((kind, key.to_string()), entry);
        Ok(())
    };

    if let Some(charges) = &molecule.charges {
        insert(
            KIND_BUILTIN,
            "charges",
            float_array_data_type_tag(charges),
            float_array_payload_len(charges),
        )?;
    }
    if let Some(cell) = &molecule.cell {
        insert(
            KIND_BUILTIN,
            "cell",
            mat3_data_type_tag(cell),
            mat3_payload_len(cell),
        )?;
    }
    if let Some(energy) = &molecule.energy {
        insert(
            KIND_BUILTIN,
            "energy",
            float_scalar_data_type_tag(energy),
            float_scalar_payload_len(energy),
        )?;
    }
    if let Some(forces) = &molecule.forces {
        insert(
            KIND_BUILTIN,
            "forces",
            vec3_data_type_tag(forces),
            vec3_payload_len(forces),
        )?;
    }
    if let Some(name) = &molecule.name {
        insert(KIND_BUILTIN, "name", TYPE_STRING, name.len())?;
    }
    if molecule.pbc.is_some() {
        insert(KIND_BUILTIN, "pbc", TYPE_BOOL3, 3)?;
    }
    if let Some(stress) = &molecule.stress {
        insert(
            KIND_BUILTIN,
            "stress",
            mat3_data_type_tag(stress),
            mat3_payload_len(stress),
        )?;
    }
    if let Some(velocities) = &molecule.velocities {
        insert(
            KIND_BUILTIN,
            "velocities",
            vec3_data_type_tag(velocities),
            vec3_payload_len(velocities),
        )?;
    }

    for (key, value) in &molecule.atom_properties {
        insert(
            KIND_ATOM_PROP,
            key,
            property_value_type_tag(value),
            property_value_payload_len(value),
        )?;
    }
    for (key, value) in &molecule.properties {
        insert(
            KIND_MOL_PROP,
            key,
            property_value_type_tag(value),
            property_value_payload_len(value),
        )?;
    }

    Ok(schema)
}

fn parse_record_schema_with_layout(
    bytes: &[u8],
    record_format: u32,
    layout: SoaLayout,
) -> Result<SchemaLock> {
    if bytes.len() < 6 {
        return Err(Error::InvalidData("SOA record too small".into()));
    }

    let mut pos = 0usize;
    let n_atoms = u32::from_le_bytes(arr(&bytes[pos..pos + 4])?) as usize;
    pos += 4;

    let positions_end = pos
        .checked_add(
            n_atoms
                .checked_mul(layout.positions_stride)
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
        positions_type: Some(layout.positions_type),
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
        if kind == KIND_BUILTIN {
            validate_builtin_type_tag_for_record_format(record_format, &key, type_tag)?;
        }
        let entry = schema_entry(kind, &key, type_tag, payload_len, n_atoms)?;
        schema.sections.insert((kind, key), entry);
    }

    Ok(schema)
}

pub(super) fn record_schema(
    bytes: &[u8],
    record_format: u32,
    positions_type_hint: Option<u8>,
) -> Result<SchemaLock> {
    parse_record_schema_with_layout(
        bytes,
        record_format,
        resolve_layout(record_format, positions_type_hint)?,
    )
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
