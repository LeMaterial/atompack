use super::*;
use atompack::storage::{DatabaseSchema, DatabaseSchemaSection};

/// A single parsed section reference (zero-copy into decompressed bytes).
#[derive(Clone)]
pub(crate) struct SectionRef<'a> {
    pub(crate) kind: u8,
    pub(crate) key: &'a str,
    pub(crate) type_tag: u8,
    pub(crate) payload: &'a [u8],
}

/// Per-molecule extracted data (references into decompressed bytes).
pub(crate) struct MolData<'a> {
    pub(crate) n_atoms: usize,
    pub(crate) positions_bytes: &'a [u8],
    pub(crate) atomic_numbers_bytes: &'a [u8],
    pub(crate) sections: Vec<SectionRef<'a>>,
}

#[derive(Clone)]
pub(crate) struct SectionSchema {
    pub(crate) kind: u8,
    pub(crate) key: String,
    pub(crate) type_tag: u8,
    pub(crate) per_atom: bool,
    pub(crate) elem_bytes: usize,
    pub(crate) slot_bytes: usize,
}

pub(crate) fn parse_mol_fast_soa_v2(bytes: &[u8]) -> atompack::Result<MolData<'_>> {
    let mut pos = 0usize;
    let n_atoms = read_u32_le_at(bytes, &mut pos, "SOA n_atoms")? as usize;
    let positions_len = n_atoms
        .checked_mul(12)
        .ok_or_else(|| invalid_data("SOA positions byte length overflow"))?;
    let positions_bytes = read_bytes_at(bytes, &mut pos, positions_len, "SOA positions")?;
    let atomic_numbers_bytes = read_bytes_at(bytes, &mut pos, n_atoms, "SOA atomic_numbers")?;
    let n_sections = read_u16_le_at(bytes, &mut pos, "SOA n_sections")? as usize;

    let mut sections = Vec::with_capacity(n_sections);
    for _ in 0..n_sections {
        let kind = read_u8_at(bytes, &mut pos, "SOA section kind")?;
        let key_len = read_u8_at(bytes, &mut pos, "SOA section key length")? as usize;
        let key_bytes = read_bytes_at(bytes, &mut pos, key_len, "SOA section key")?;
        let key = std::str::from_utf8(key_bytes)
            .map_err(|_| invalid_data("Invalid UTF-8 in SOA section key"))?;
        let type_tag = read_u8_at(bytes, &mut pos, "SOA section type tag")?;
        let payload_len = read_u32_le_at(bytes, &mut pos, "SOA section payload length")? as usize;
        let payload = read_bytes_at(bytes, &mut pos, payload_len, "SOA section payload")?;
        sections.push(SectionRef {
            kind,
            key,
            type_tag,
            payload,
        });
    }

    Ok(MolData {
        n_atoms,
        positions_bytes,
        atomic_numbers_bytes,
        sections,
    })
}

/// Parse SOA format bytes into MolData without allocation.
///
/// Layout:
///   [n_atoms:u32][positions:n*(12|24)][atomic_numbers:n]
///   [n_sections:u16]
///   per section: [kind:u8][key_len:u8][key][type_tag:u8][payload_len:u32][payload]
pub(crate) fn parse_mol_fast_soa(
    bytes: &[u8],
    record_format: u32,
    positions_type_hint: Option<u8>,
) -> atompack::Result<MolData<'_>> {
    if record_format == RECORD_FORMAT_SOA_V2 {
        return parse_mol_fast_soa_v2(bytes);
    }

    let mut pos = 0usize;
    let n_atoms = read_u32_le_at(bytes, &mut pos, "SOA n_atoms")? as usize;
    let positions_type = match record_format {
        RECORD_FORMAT_SOA_V3 => positions_type_hint
            .ok_or_else(|| invalid_data("Missing positions dtype for record format 3"))?,
        _ => {
            return Err(invalid_data(format!(
                "Unsupported record format {}",
                record_format
            )));
        }
    };
    let positions_stride = match positions_type {
        TYPE_VEC3_F32 => 12usize,
        TYPE_VEC3_F64 => 24usize,
        _ => {
            return Err(invalid_data(format!(
                "Unsupported positions type tag {}",
                positions_type
            )));
        }
    };
    let positions_len = n_atoms
        .checked_mul(positions_stride)
        .ok_or_else(|| invalid_data("SOA positions byte length overflow"))?;
    let positions_bytes = read_bytes_at(bytes, &mut pos, positions_len, "SOA positions")?;
    let atomic_numbers_bytes = read_bytes_at(bytes, &mut pos, n_atoms, "SOA atomic_numbers")?;
    let n_sections = read_u16_le_at(bytes, &mut pos, "SOA n_sections")? as usize;

    let mut sections = Vec::with_capacity(n_sections);
    for _ in 0..n_sections {
        let kind = read_u8_at(bytes, &mut pos, "SOA section kind")?;
        let key_len = read_u8_at(bytes, &mut pos, "SOA section key length")? as usize;
        let key_bytes = read_bytes_at(bytes, &mut pos, key_len, "SOA section key")?;
        let key = std::str::from_utf8(key_bytes)
            .map_err(|_| invalid_data("Invalid UTF-8 in SOA section key"))?;
        let type_tag = read_u8_at(bytes, &mut pos, "SOA section type tag")?;
        let payload_len = read_u32_le_at(bytes, &mut pos, "SOA section payload length")? as usize;
        let payload = read_bytes_at(bytes, &mut pos, payload_len, "SOA section payload")?;
        sections.push(SectionRef {
            kind,
            key,
            type_tag,
            payload,
        });
    }

    Ok(MolData {
        n_atoms,
        positions_bytes,
        atomic_numbers_bytes,
        sections,
    })
}

pub(crate) fn section_schema_from_ref(
    section: &SectionRef<'_>,
    n_atoms: usize,
) -> atompack::Result<SectionSchema> {
    let per_atom = is_per_atom(section.kind, section.key, section.type_tag);
    let elem_bytes = match section.type_tag {
        TYPE_STRING => 0,
        tag if per_atom => {
            let elem_bytes = type_tag_elem_bytes(tag);
            if elem_bytes == 0 {
                return Err(invalid_data(format!(
                    "Unsupported per-atom section type tag {} for key '{}'",
                    tag, section.key
                )));
            }
            elem_bytes
        }
        TYPE_FLOAT | TYPE_INT => 8,
        TYPE_FLOAT32 => 4,
        TYPE_BOOL3 => 3,
        TYPE_MAT3X3_F32 => 36,
        TYPE_MAT3X3_F64 => 72,
        _ => section.payload.len(),
    };
    let slot_bytes = if section.type_tag == TYPE_STRING {
        0
    } else if per_atom {
        elem_bytes
    } else {
        section.payload.len()
    };

    validate_section_payload(section, per_atom, elem_bytes, slot_bytes, n_atoms)?;

    Ok(SectionSchema {
        kind: section.kind,
        key: section.key.to_string(),
        type_tag: section.type_tag,
        per_atom,
        elem_bytes,
        slot_bytes,
    })
}

pub(crate) fn validate_section_payload(
    section: &SectionRef<'_>,
    per_atom: bool,
    elem_bytes: usize,
    slot_bytes: usize,
    n_atoms: usize,
) -> atompack::Result<()> {
    match section.type_tag {
        TYPE_STRING => {
            std::str::from_utf8(section.payload)
                .map_err(|_| invalid_data(format!("Invalid UTF-8 in section '{}'", section.key)))?;
            if per_atom {
                return Err(invalid_data(format!(
                    "String section '{}' cannot be per-atom in flat extraction",
                    section.key
                )));
            }
        }
        TYPE_FLOAT | TYPE_INT | TYPE_FLOAT32 | TYPE_BOOL3 | TYPE_MAT3X3_F32 | TYPE_MAT3X3_F64 => {
            if section.payload.len() != slot_bytes {
                return Err(invalid_data(format!(
                    "Section '{}' has invalid payload length {} (expected {})",
                    section.key,
                    section.payload.len(),
                    slot_bytes
                )));
            }
        }
        _ if per_atom => {
            let expected = n_atoms.checked_mul(elem_bytes).ok_or_else(|| {
                invalid_data(format!("Section '{}' payload length overflow", section.key))
            })?;
            if section.payload.len() != expected {
                return Err(invalid_data(format!(
                    "Section '{}' has invalid payload length {} (expected {})",
                    section.key,
                    section.payload.len(),
                    expected
                )));
            }
        }
        _ => {
            if slot_bytes != 0 && section.payload.len() != slot_bytes {
                return Err(invalid_data(format!(
                    "Section '{}' has invalid payload length {} (expected {})",
                    section.key,
                    section.payload.len(),
                    slot_bytes
                )));
            }
            if elem_bytes != 0 && !section.payload.len().is_multiple_of(elem_bytes) {
                return Err(invalid_data(format!(
                    "Section '{}' has invalid payload length {} for element size {}",
                    section.key,
                    section.payload.len(),
                    elem_bytes
                )));
            }
        }
    }
    Ok(())
}

/// Element size in bytes for a given type tag. Returns 0 for variable-length types.
pub(crate) fn type_tag_elem_bytes(tag: u8) -> usize {
    match tag {
        TYPE_FLOAT => 8,
        TYPE_INT => 8,
        TYPE_STRING => 0,
        TYPE_F64_ARRAY => 8,
        TYPE_VEC3_F32 => 12,
        TYPE_I64_ARRAY => 8,
        TYPE_F32_ARRAY => 4,
        TYPE_VEC3_F64 => 24,
        TYPE_I32_ARRAY => 4,
        TYPE_BOOL3 => 3,
        TYPE_FLOAT32 => 4,
        TYPE_MAT3X3_F32 => 36,
        TYPE_MAT3X3_F64 => 72,
        _ => 0,
    }
}

/// Whether a section with the given kind/key/type_tag is per-atom (vs per-molecule).
pub(crate) fn is_per_atom(kind: u8, key: &str, _type_tag: u8) -> bool {
    match kind {
        KIND_ATOM_PROP => true,
        KIND_MOL_PROP => false,
        KIND_BUILTIN => matches!(key, "forces" | "charges" | "velocities"),
        _ => false,
    }
}

fn database_schema_section(
    kind: u8,
    key: &str,
    type_tag: u8,
    payload_len: usize,
    n_atoms: usize,
) -> PyResult<DatabaseSchemaSection> {
    let per_atom = is_per_atom(kind, key, type_tag);
    let elem_bytes = if type_tag == TYPE_STRING {
        0
    } else {
        let elem_bytes = type_tag_elem_bytes(type_tag);
        if elem_bytes == 0 {
            return Err(PyValueError::new_err(format!(
                "Unsupported section type tag {} for '{}'",
                type_tag, key
            )));
        }
        elem_bytes
    };
    let slot_bytes = if type_tag == TYPE_STRING {
        0
    } else if per_atom {
        let expected = n_atoms.checked_mul(elem_bytes).ok_or_else(|| {
            PyValueError::new_err(format!("Section '{}' payload length overflow", key))
        })?;
        if payload_len != expected {
            return Err(PyValueError::new_err(format!(
                "Section '{}' has invalid payload length {} (expected {})",
                key, payload_len, expected
            )));
        }
        elem_bytes
    } else {
        payload_len
    };

    Ok(DatabaseSchemaSection {
        kind,
        key: key.to_string(),
        type_tag,
        per_atom,
        elem_bytes,
        slot_bytes,
    })
}

fn py_decode_float_array_data(
    payload: &[u8],
    type_tag: u8,
    field_name: &str,
) -> PyResult<FloatArrayData> {
    match type_tag {
        TYPE_F32_ARRAY => Ok(FloatArrayData::F32(decode_f32_array(payload)?)),
        TYPE_F64_ARRAY => Ok(FloatArrayData::F64(decode_f64_array(payload)?)),
        other => Err(PyValueError::new_err(format!(
            "Unsupported {field_name} type tag {}",
            other
        ))),
    }
}

fn py_decode_mat3_data(payload: &[u8], type_tag: u8, field_name: &str) -> PyResult<Mat3Data> {
    match type_tag {
        TYPE_MAT3X3_F32 => Ok(Mat3Data::F32(decode_mat3x3_f32(payload)?)),
        TYPE_MAT3X3_F64 => Ok(Mat3Data::F64(decode_mat3x3_f64(payload)?)),
        other => Err(PyValueError::new_err(format!(
            "Unsupported {field_name} type tag {}",
            other
        ))),
    }
}

fn py_decode_float_scalar_data(
    payload: &[u8],
    type_tag: u8,
    field_name: &str,
) -> PyResult<FloatScalarData> {
    match type_tag {
        TYPE_FLOAT => Ok(FloatScalarData::F64(read_f64_scalar(payload)?)),
        TYPE_FLOAT32 => Ok(FloatScalarData::F32(read_f32_scalar(payload)?)),
        other => Err(PyValueError::new_err(format!(
            "Unsupported {field_name} type tag {}",
            other
        ))),
    }
}

fn py_decode_vec3_data(payload: &[u8], type_tag: u8, field_name: &str) -> PyResult<Vec3Data> {
    match type_tag {
        TYPE_VEC3_F32 => Ok(Vec3Data::F32(decode_vec3_f32(payload)?)),
        TYPE_VEC3_F64 => Ok(Vec3Data::F64(decode_vec3_f64(payload)?)),
        other => Err(PyValueError::new_err(format!(
            "Unsupported {field_name} type tag {}",
            other
        ))),
    }
}

/// Lightweight section descriptor — stores byte offsets into the parent `bytes` buffer.
/// Key is NOT parsed eagerly; use `key()` to read it lazily from `bytes`.
#[derive(Clone, Copy)]
pub(crate) struct LazySection {
    pub(crate) kind: u8,
    pub(crate) key_start: usize,
    pub(crate) key_len: u8,
    pub(crate) type_tag: u8,
    pub(crate) payload_start: usize,
    pub(crate) payload_len: usize,
}

/// Byte-offset pair for a known builtin section (payload_start, payload_len, type_tag).
pub(crate) type BuiltinSlot = (usize, usize, u8);

enum SoaBytes {
    Owned(Vec<u8>),
    Shared(SharedMmapBytes),
}

impl SoaBytes {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Shared(bytes) => bytes.as_slice(),
        }
    }
}

impl std::ops::Deref for SoaBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

pub(crate) struct SoaMoleculeView {
    bytes: SoaBytes,
    pub(crate) n_atoms: usize,
    pub(crate) positions_type: u8,
    positions_start: usize,
    positions_len: usize,
    atomic_numbers_start: usize,
    pub(crate) forces: Option<BuiltinSlot>,
    pub(crate) energy: Option<BuiltinSlot>,
    pub(crate) cell: Option<BuiltinSlot>,
    pub(crate) stress: Option<BuiltinSlot>,
    pub(crate) charges: Option<BuiltinSlot>,
    pub(crate) velocities: Option<BuiltinSlot>,
    pbc: Option<BuiltinSlot>,
    name: Option<BuiltinSlot>,
    pub(crate) custom_sections: Vec<LazySection>,
}

impl SoaMoleculeView {
    fn from_storage_v2(bytes: SoaBytes) -> atompack::Result<Self> {
        if bytes.len() < 6 {
            return Err(invalid_data("SOA record too small"));
        }

        let n_atoms = u32::from_le_bytes(slice_to_array(&bytes[0..4], "SOA atom count")?) as usize;
        let mut pos = 4usize;
        let positions_start = pos;
        let positions_len = n_atoms
            .checked_mul(12)
            .ok_or_else(|| invalid_data("SOA positions overflow"))?;
        pos = pos
            .checked_add(positions_len)
            .ok_or_else(|| invalid_data("SOA positions overflow"))?;
        if pos > bytes.len() {
            return Err(invalid_data("SOA record truncated at positions"));
        }

        let atomic_numbers_start = pos;
        pos = pos
            .checked_add(n_atoms)
            .ok_or_else(|| invalid_data("SOA atomic_numbers overflow"))?;
        if pos + 2 > bytes.len() {
            return Err(invalid_data("SOA record truncated at atomic_numbers"));
        }

        let n_sections =
            u16::from_le_bytes(slice_to_array(&bytes[pos..pos + 2], "SOA section count")?) as usize;
        pos += 2;

        let mut forces = None;
        let mut energy = None;
        let mut cell = None;
        let mut stress = None;
        let mut charges = None;
        let mut velocities = None;
        let mut pbc = None;
        let mut name = None;
        let mut custom_sections = Vec::new();

        for _ in 0..n_sections {
            if pos + 2 > bytes.len() {
                return Err(invalid_data("SOA section header truncated"));
            }
            let kind = bytes[pos];
            pos += 1;
            let key_len = bytes[pos] as usize;
            pos += 1;
            if pos + key_len > bytes.len() {
                return Err(invalid_data("SOA section key truncated"));
            }
            let key_start = pos;
            pos += key_len;
            if pos + 5 > bytes.len() {
                return Err(invalid_data("SOA section header truncated"));
            }
            let type_tag = bytes[pos];
            pos += 1;
            let payload_len = u32::from_le_bytes(slice_to_array(
                &bytes[pos..pos + 4],
                "SOA section payload length",
            )?) as usize;
            pos += 4;
            let payload_start = pos;
            pos = pos
                .checked_add(payload_len)
                .ok_or_else(|| invalid_data("SOA section payload overflow"))?;
            if pos > bytes.len() {
                return Err(invalid_data("SOA section payload truncated"));
            }

            let key_bytes = &bytes[key_start..key_start + key_len];
            if kind == KIND_BUILTIN {
                let slot = (payload_start, payload_len, type_tag);
                match key_bytes {
                    b"forces" => forces = Some(slot),
                    b"energy" => energy = Some(slot),
                    b"cell" => cell = Some(slot),
                    b"stress" => stress = Some(slot),
                    b"charges" => charges = Some(slot),
                    b"velocities" => velocities = Some(slot),
                    b"pbc" => pbc = Some(slot),
                    b"name" => name = Some(slot),
                    _ => {
                        custom_sections.push(LazySection {
                            kind,
                            key_start,
                            key_len: key_len as u8,
                            type_tag,
                            payload_start,
                            payload_len,
                        });
                    }
                }
            } else {
                custom_sections.push(LazySection {
                    kind,
                    key_start,
                    key_len: key_len as u8,
                    type_tag,
                    payload_start,
                    payload_len,
                });
            }
        }

        Ok(Self {
            bytes,
            n_atoms,
            positions_type: TYPE_VEC3_F32,
            positions_start,
            positions_len,
            atomic_numbers_start,
            forces,
            energy,
            cell,
            stress,
            charges,
            velocities,
            pbc,
            name,
            custom_sections,
        })
    }

    /// Pure-Rust parser — no Python dependency, safe to call from rayon threads.
    fn from_storage_inner(
        bytes: SoaBytes,
        record_format: u32,
        positions_type_hint: Option<u8>,
    ) -> atompack::Result<Self> {
        if record_format == RECORD_FORMAT_SOA_V2 {
            return Self::from_storage_v2(bytes);
        }

        if bytes.len() < 6 {
            return Err(invalid_data("SOA record too small"));
        }

        let n_atoms = u32::from_le_bytes(slice_to_array(&bytes[0..4], "SOA atom count")?) as usize;
        let mut pos = 4usize;
        let positions_type = match record_format {
            RECORD_FORMAT_SOA_V3 => positions_type_hint
                .ok_or_else(|| invalid_data("Missing positions dtype for record format 3"))?,
            _ => {
                return Err(invalid_data(format!(
                    "Unsupported record format {}",
                    record_format
                )));
            }
        };
        let positions_stride = match positions_type {
            TYPE_VEC3_F32 => 12usize,
            TYPE_VEC3_F64 => 24usize,
            _ => {
                return Err(invalid_data(format!(
                    "Unsupported positions type tag {}",
                    positions_type
                )));
            }
        };
        let positions_start = pos;
        let positions_len = n_atoms
            .checked_mul(positions_stride)
            .ok_or_else(|| invalid_data("SOA positions overflow"))?;
        pos = pos
            .checked_add(positions_len)
            .ok_or_else(|| invalid_data("SOA positions overflow"))?;
        if pos > bytes.len() {
            return Err(invalid_data("SOA record truncated at positions"));
        }

        let atomic_numbers_start = pos;
        pos = pos
            .checked_add(n_atoms)
            .ok_or_else(|| invalid_data("SOA atomic_numbers overflow"))?;
        if pos + 2 > bytes.len() {
            return Err(invalid_data("SOA record truncated at atomic_numbers"));
        }

        let n_sections =
            u16::from_le_bytes(slice_to_array(&bytes[pos..pos + 2], "SOA section count")?) as usize;
        pos += 2;

        let mut forces = None;
        let mut energy = None;
        let mut cell = None;
        let mut stress = None;
        let mut charges = None;
        let mut velocities = None;
        let mut pbc = None;
        let mut name = None;
        let mut custom_sections = Vec::new();

        for _ in 0..n_sections {
            if pos + 2 > bytes.len() {
                return Err(invalid_data("SOA section header truncated"));
            }
            let kind = bytes[pos];
            pos += 1;
            let key_len = bytes[pos] as usize;
            pos += 1;
            if pos + key_len > bytes.len() {
                return Err(invalid_data("SOA section key truncated"));
            }
            let key_start = pos;
            pos += key_len;
            if pos + 5 > bytes.len() {
                return Err(invalid_data("SOA section header truncated"));
            }
            let type_tag = bytes[pos];
            pos += 1;
            let payload_len = u32::from_le_bytes(slice_to_array(
                &bytes[pos..pos + 4],
                "SOA section payload length",
            )?) as usize;
            pos += 4;
            let payload_start = pos;
            pos = pos
                .checked_add(payload_len)
                .ok_or_else(|| invalid_data("SOA section payload overflow"))?;
            if pos > bytes.len() {
                return Err(invalid_data("SOA section payload truncated"));
            }

            let key_bytes = &bytes[key_start..key_start + key_len];
            if kind == KIND_BUILTIN {
                let slot = (payload_start, payload_len, type_tag);
                match key_bytes {
                    b"forces" => forces = Some(slot),
                    b"energy" => energy = Some(slot),
                    b"cell" => cell = Some(slot),
                    b"stress" => stress = Some(slot),
                    b"charges" => charges = Some(slot),
                    b"velocities" => velocities = Some(slot),
                    b"pbc" => pbc = Some(slot),
                    b"name" => name = Some(slot),
                    _ => {
                        custom_sections.push(LazySection {
                            kind,
                            key_start,
                            key_len: key_len as u8,
                            type_tag,
                            payload_start,
                            payload_len,
                        });
                    }
                }
            } else {
                custom_sections.push(LazySection {
                    kind,
                    key_start,
                    key_len: key_len as u8,
                    type_tag,
                    payload_start,
                    payload_len,
                });
            }
        }

        Ok(Self {
            bytes,
            n_atoms,
            positions_type,
            positions_start,
            positions_len,
            atomic_numbers_start,
            forces,
            energy,
            cell,
            stress,
            charges,
            velocities,
            pbc,
            name,
            custom_sections,
        })
    }

    pub(crate) fn from_bytes_inner(
        bytes: Vec<u8>,
        record_format: u32,
        positions_type_hint: Option<u8>,
    ) -> atompack::Result<Self> {
        Self::from_storage_inner(SoaBytes::Owned(bytes), record_format, positions_type_hint)
    }

    pub(crate) fn from_shared_bytes_inner(
        bytes: SharedMmapBytes,
        record_format: u32,
        positions_type_hint: Option<u8>,
    ) -> atompack::Result<Self> {
        Self::from_storage_inner(SoaBytes::Shared(bytes), record_format, positions_type_hint)
    }

    pub(crate) fn from_bytes(
        bytes: Vec<u8>,
        record_format: u32,
        positions_type_hint: Option<u8>,
    ) -> PyResult<Self> {
        Self::from_bytes_inner(bytes, record_format, positions_type_hint)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    pub(crate) fn positions_bytes(&self) -> &[u8] {
        &self.bytes[self.positions_start..self.positions_start + self.positions_len]
    }

    #[inline]
    pub(crate) fn raw_bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    pub(crate) fn database_schema(&self) -> PyResult<DatabaseSchema> {
        let mut sections = Vec::with_capacity(8 + self.custom_sections.len());

        if let Some(slot) = self.energy {
            sections.push(database_schema_section(
                KIND_BUILTIN,
                "energy",
                slot.2,
                slot.1,
                self.n_atoms,
            )?);
        }
        if let Some(slot) = self.forces {
            sections.push(database_schema_section(
                KIND_BUILTIN,
                "forces",
                slot.2,
                slot.1,
                self.n_atoms,
            )?);
        }
        if let Some(slot) = self.charges {
            sections.push(database_schema_section(
                KIND_BUILTIN,
                "charges",
                slot.2,
                slot.1,
                self.n_atoms,
            )?);
        }
        if let Some(slot) = self.velocities {
            sections.push(database_schema_section(
                KIND_BUILTIN,
                "velocities",
                slot.2,
                slot.1,
                self.n_atoms,
            )?);
        }
        if let Some(slot) = self.cell {
            sections.push(database_schema_section(
                KIND_BUILTIN,
                "cell",
                slot.2,
                slot.1,
                self.n_atoms,
            )?);
        }
        if let Some(slot) = self.stress {
            sections.push(database_schema_section(
                KIND_BUILTIN,
                "stress",
                slot.2,
                slot.1,
                self.n_atoms,
            )?);
        }
        if let Some(slot) = self.pbc {
            sections.push(database_schema_section(
                KIND_BUILTIN,
                "pbc",
                slot.2,
                slot.1,
                self.n_atoms,
            )?);
        }
        if let Some(slot) = self.name {
            sections.push(database_schema_section(
                KIND_BUILTIN,
                "name",
                slot.2,
                slot.1,
                self.n_atoms,
            )?);
        }
        for section in &self.custom_sections {
            sections.push(database_schema_section(
                section.kind,
                self.lazy_section_key(section)?,
                section.type_tag,
                section.payload_len,
                self.n_atoms,
            )?);
        }

        Ok(DatabaseSchema {
            positions_type: Some(self.positions_type),
            sections,
        })
    }

    pub(crate) fn same_schema_as(&self, other: &Self) -> PyResult<bool> {
        if self.positions_type != other.positions_type
            || self.energy != other.energy
            || self.forces != other.forces
            || self.charges != other.charges
            || self.velocities != other.velocities
            || self.cell != other.cell
            || self.stress != other.stress
            || self.pbc != other.pbc
            || self.name != other.name
            || self.custom_sections.len() != other.custom_sections.len()
        {
            return Ok(false);
        }

        for (left, right) in self
            .custom_sections
            .iter()
            .zip(other.custom_sections.iter())
        {
            if left.kind != right.kind
                || left.type_tag != right.type_tag
                || left.payload_len != right.payload_len
                || self.lazy_section_key(left)? != other.lazy_section_key(right)?
            {
                return Ok(false);
            }
        }

        Ok(true)
    }

    pub(crate) fn atomic_numbers_bytes(&self) -> &[u8] {
        &self.bytes[self.atomic_numbers_start..self.atomic_numbers_start + self.n_atoms]
    }

    #[inline]
    pub(crate) fn builtin_payload(&self, slot: BuiltinSlot) -> &[u8] {
        &self.bytes[slot.0..slot.0 + slot.1]
    }

    pub(crate) fn lazy_section_key(&self, s: &LazySection) -> PyResult<&str> {
        std::str::from_utf8(&self.bytes[s.key_start..s.key_start + s.key_len as usize])
            .map_err(|_| PyValueError::new_err("Invalid UTF-8 in section key"))
    }

    pub(crate) fn lazy_section_payload(&self, s: &LazySection) -> &[u8] {
        &self.bytes[s.payload_start..s.payload_start + s.payload_len]
    }

    pub(crate) fn find_custom_section(
        &self,
        kind: u8,
        key: &str,
    ) -> PyResult<Option<&LazySection>> {
        for section in &self.custom_sections {
            if section.kind == kind && self.lazy_section_key(section)? == key {
                return Ok(Some(section));
            }
        }
        Ok(None)
    }

    pub(crate) fn property_keys(&self) -> PyResult<Vec<String>> {
        self.custom_sections
            .iter()
            .filter(|s| s.kind == KIND_MOL_PROP)
            .map(|s| Ok(self.lazy_section_key(s)?.to_string()))
            .collect()
    }

    pub(crate) fn atom_at(&self, index: usize) -> PyResult<Option<Atom>> {
        if index >= self.n_atoms {
            return Ok(None);
        }
        let atomic_number = self.atomic_numbers_bytes()[index];
        Ok(Some(match self.positions_type {
            TYPE_VEC3_F32 => {
                let pos = &self.positions_bytes()[index * 12..(index + 1) * 12];
                Atom::new(
                    f32::from_le_bytes(py_slice_to_array(&pos[0..4], "atom x")?),
                    f32::from_le_bytes(py_slice_to_array(&pos[4..8], "atom y")?),
                    f32::from_le_bytes(py_slice_to_array(&pos[8..12], "atom z")?),
                    atomic_number,
                )
            }
            TYPE_VEC3_F64 => {
                let pos = &self.positions_bytes()[index * 24..(index + 1) * 24];
                Atom::new(
                    f64::from_le_bytes(py_slice_to_array(&pos[0..8], "atom x")?) as f32,
                    f64::from_le_bytes(py_slice_to_array(&pos[8..16], "atom y")?) as f32,
                    f64::from_le_bytes(py_slice_to_array(&pos[16..24], "atom z")?) as f32,
                    atomic_number,
                )
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported positions type tag {}",
                    other
                )));
            }
        }))
    }

    pub(crate) fn energy(&self) -> PyResult<Option<f64>> {
        match self.energy {
            Some(slot) => match slot.2 {
                TYPE_FLOAT => Ok(Some(read_f64_scalar(self.builtin_payload(slot))?)),
                TYPE_FLOAT32 => Ok(Some(read_f32_scalar(self.builtin_payload(slot))? as f64)),
                other => Err(PyValueError::new_err(format!(
                    "Unsupported energy type tag {}",
                    other
                ))),
            },
            None => Ok(None),
        }
    }

    pub(crate) fn pbc(&self) -> PyResult<Option<(bool, bool, bool)>> {
        match self.pbc {
            Some(slot) => {
                let payload = self.builtin_payload(slot);
                if payload.len() != 3 {
                    return Err(PyValueError::new_err("Invalid pbc payload length"));
                }
                Ok(Some((payload[0] != 0, payload[1] != 0, payload[2] != 0)))
            }
            None => Ok(None),
        }
    }

    pub(crate) fn materialize(&self) -> PyResult<Molecule> {
        let atomic_numbers = self.atomic_numbers_bytes().to_vec();
        let mut molecule = match self.positions_type {
            TYPE_VEC3_F32 => {
                let positions: Vec<[f32; 3]> = (0..self.n_atoms)
                    .map(|i| {
                        let pos = &self.positions_bytes()[i * 12..(i + 1) * 12];
                        Ok([
                            f32::from_le_bytes(py_slice_to_array(&pos[0..4], "position x")?),
                            f32::from_le_bytes(py_slice_to_array(&pos[4..8], "position y")?),
                            f32::from_le_bytes(py_slice_to_array(&pos[8..12], "position z")?),
                        ])
                    })
                    .collect::<PyResult<_>>()?;
                Molecule::new(positions, atomic_numbers).map_err(PyValueError::new_err)?
            }
            TYPE_VEC3_F64 => {
                let positions: Vec<[f64; 3]> = (0..self.n_atoms)
                    .map(|i| {
                        let pos = &self.positions_bytes()[i * 24..(i + 1) * 24];
                        Ok([
                            f64::from_le_bytes(py_slice_to_array(&pos[0..8], "position x")?),
                            f64::from_le_bytes(py_slice_to_array(&pos[8..16], "position y")?),
                            f64::from_le_bytes(py_slice_to_array(&pos[16..24], "position z")?),
                        ])
                    })
                    .collect::<PyResult<_>>()?;
                Molecule::new_f64(positions, atomic_numbers).map_err(PyValueError::new_err)?
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported positions type tag {}",
                    other
                )));
            }
        };

        if let Some(slot) = self.charges {
            molecule.charges = Some(py_decode_float_array_data(
                self.builtin_payload(slot),
                slot.2,
                "charges",
            )?);
        }
        if let Some(slot) = self.cell {
            molecule.cell = Some(py_decode_mat3_data(
                self.builtin_payload(slot),
                slot.2,
                "cell",
            )?);
        }
        if let Some(slot) = self.energy {
            molecule.energy = Some(py_decode_float_scalar_data(
                self.builtin_payload(slot),
                slot.2,
                "energy",
            )?);
        }
        if let Some(slot) = self.forces {
            molecule.forces = Some(py_decode_vec3_data(
                self.builtin_payload(slot),
                slot.2,
                "forces",
            )?);
        }
        if let Some(slot) = self.name {
            let payload = self.builtin_payload(slot);
            molecule.name = Some(
                std::str::from_utf8(payload)
                    .map_err(|_| PyValueError::new_err("Invalid UTF-8 in name"))?
                    .to_string(),
            );
        }
        if let Some(slot) = self.pbc {
            let payload = self.builtin_payload(slot);
            if payload.len() != 3 {
                return Err(PyValueError::new_err("Invalid pbc payload length"));
            }
            molecule.pbc = Some([payload[0] != 0, payload[1] != 0, payload[2] != 0]);
        }
        if let Some(slot) = self.stress {
            molecule.stress = Some(py_decode_mat3_data(
                self.builtin_payload(slot),
                slot.2,
                "stress",
            )?);
        }
        if let Some(slot) = self.velocities {
            molecule.velocities = Some(py_decode_vec3_data(
                self.builtin_payload(slot),
                slot.2,
                "velocities",
            )?);
        }

        for s in &self.custom_sections {
            let key = self.lazy_section_key(s)?.to_string();
            let payload = self.lazy_section_payload(s);
            match s.kind {
                KIND_ATOM_PROP => {
                    molecule
                        .atom_properties
                        .insert(key, decode_property_value(s.type_tag, payload)?);
                }
                KIND_MOL_PROP => {
                    molecule
                        .properties
                        .insert(key, decode_property_value(s.type_tag, payload)?);
                }
                _ => {}
            }
        }

        Ok(molecule)
    }
}

pub(crate) fn read_f64_scalar(payload: &[u8]) -> PyResult<f64> {
    if payload.len() != 8 {
        return Err(PyValueError::new_err("Invalid f64 payload length"));
    }
    Ok(f64::from_le_bytes(py_slice_to_array(
        payload,
        "f64 payload",
    )?))
}

fn read_f32_scalar(payload: &[u8]) -> PyResult<f32> {
    if payload.len() != 4 {
        return Err(PyValueError::new_err("Invalid f32 payload length"));
    }
    Ok(f32::from_le_bytes(py_slice_to_array(
        payload,
        "f32 payload",
    )?))
}

pub(crate) fn read_i64_scalar(payload: &[u8]) -> PyResult<i64> {
    if payload.len() != 8 {
        return Err(PyValueError::new_err("Invalid i64 payload length"));
    }
    Ok(i64::from_le_bytes(py_slice_to_array(
        payload,
        "i64 payload",
    )?))
}

fn decode_f64_array(payload: &[u8]) -> PyResult<Vec<f64>> {
    if !payload.len().is_multiple_of(8) {
        return Err(PyValueError::new_err("Invalid f64 array payload length"));
    }
    payload
        .chunks_exact(8)
        .map(|chunk| {
            Ok(f64::from_le_bytes(py_slice_to_array(
                chunk,
                "f64 array chunk",
            )?))
        })
        .collect()
}

fn decode_i64_array(payload: &[u8]) -> PyResult<Vec<i64>> {
    if !payload.len().is_multiple_of(8) {
        return Err(PyValueError::new_err("Invalid i64 array payload length"));
    }
    payload
        .chunks_exact(8)
        .map(|chunk| {
            Ok(i64::from_le_bytes(py_slice_to_array(
                chunk,
                "i64 array chunk",
            )?))
        })
        .collect()
}

fn decode_i32_array(payload: &[u8]) -> PyResult<Vec<i32>> {
    if !payload.len().is_multiple_of(4) {
        return Err(PyValueError::new_err("Invalid i32 array payload length"));
    }
    payload
        .chunks_exact(4)
        .map(|chunk| {
            Ok(i32::from_le_bytes(py_slice_to_array(
                chunk,
                "i32 array chunk",
            )?))
        })
        .collect()
}

fn decode_f32_array(payload: &[u8]) -> PyResult<Vec<f32>> {
    if !payload.len().is_multiple_of(4) {
        return Err(PyValueError::new_err("Invalid f32 array payload length"));
    }
    payload
        .chunks_exact(4)
        .map(|chunk| {
            Ok(f32::from_le_bytes(py_slice_to_array(
                chunk,
                "f32 array chunk",
            )?))
        })
        .collect()
}

fn decode_vec3_f32(payload: &[u8]) -> PyResult<Vec<[f32; 3]>> {
    if !payload.len().is_multiple_of(12) {
        return Err(PyValueError::new_err("Invalid vec3<f32> payload length"));
    }
    payload
        .chunks_exact(12)
        .map(|chunk| {
            Ok([
                f32::from_le_bytes(py_slice_to_array(&chunk[0..4], "vec3<f32> x")?),
                f32::from_le_bytes(py_slice_to_array(&chunk[4..8], "vec3<f32> y")?),
                f32::from_le_bytes(py_slice_to_array(&chunk[8..12], "vec3<f32> z")?),
            ])
        })
        .collect()
}

fn decode_vec3_f64(payload: &[u8]) -> PyResult<Vec<[f64; 3]>> {
    if !payload.len().is_multiple_of(24) {
        return Err(PyValueError::new_err("Invalid vec3<f64> payload length"));
    }
    payload
        .chunks_exact(24)
        .map(|chunk| {
            Ok([
                f64::from_le_bytes(py_slice_to_array(&chunk[0..8], "vec3<f64> x")?),
                f64::from_le_bytes(py_slice_to_array(&chunk[8..16], "vec3<f64> y")?),
                f64::from_le_bytes(py_slice_to_array(&chunk[16..24], "vec3<f64> z")?),
            ])
        })
        .collect()
}

fn decode_mat3x3_f64(payload: &[u8]) -> PyResult<[[f64; 3]; 3]> {
    if payload.len() != 72 {
        return Err(PyValueError::new_err("Invalid mat3x3<f64> payload length"));
    }
    Ok([
        [
            f64::from_le_bytes(py_slice_to_array(&payload[0..8], "mat3x3<f64> [0][0]")?),
            f64::from_le_bytes(py_slice_to_array(&payload[8..16], "mat3x3<f64> [0][1]")?),
            f64::from_le_bytes(py_slice_to_array(&payload[16..24], "mat3x3<f64> [0][2]")?),
        ],
        [
            f64::from_le_bytes(py_slice_to_array(&payload[24..32], "mat3x3<f64> [1][0]")?),
            f64::from_le_bytes(py_slice_to_array(&payload[32..40], "mat3x3<f64> [1][1]")?),
            f64::from_le_bytes(py_slice_to_array(&payload[40..48], "mat3x3<f64> [1][2]")?),
        ],
        [
            f64::from_le_bytes(py_slice_to_array(&payload[48..56], "mat3x3<f64> [2][0]")?),
            f64::from_le_bytes(py_slice_to_array(&payload[56..64], "mat3x3<f64> [2][1]")?),
            f64::from_le_bytes(py_slice_to_array(&payload[64..72], "mat3x3<f64> [2][2]")?),
        ],
    ])
}

fn decode_mat3x3_f32(payload: &[u8]) -> PyResult<[[f32; 3]; 3]> {
    if payload.len() != 36 {
        return Err(PyValueError::new_err("Invalid mat3x3<f32> payload length"));
    }
    Ok([
        [
            f32::from_le_bytes(py_slice_to_array(&payload[0..4], "mat3x3<f32> [0][0]")?),
            f32::from_le_bytes(py_slice_to_array(&payload[4..8], "mat3x3<f32> [0][1]")?),
            f32::from_le_bytes(py_slice_to_array(&payload[8..12], "mat3x3<f32> [0][2]")?),
        ],
        [
            f32::from_le_bytes(py_slice_to_array(&payload[12..16], "mat3x3<f32> [1][0]")?),
            f32::from_le_bytes(py_slice_to_array(&payload[16..20], "mat3x3<f32> [1][1]")?),
            f32::from_le_bytes(py_slice_to_array(&payload[20..24], "mat3x3<f32> [1][2]")?),
        ],
        [
            f32::from_le_bytes(py_slice_to_array(&payload[24..28], "mat3x3<f32> [2][0]")?),
            f32::from_le_bytes(py_slice_to_array(&payload[28..32], "mat3x3<f32> [2][1]")?),
            f32::from_le_bytes(py_slice_to_array(&payload[32..36], "mat3x3<f32> [2][2]")?),
        ],
    ])
}

fn decode_property_value(type_tag: u8, payload: &[u8]) -> PyResult<PropertyValue> {
    Ok(match type_tag {
        TYPE_FLOAT => PropertyValue::Float(read_f64_scalar(payload)?),
        TYPE_INT => PropertyValue::Int(read_i64_scalar(payload)?),
        TYPE_STRING => PropertyValue::String(
            std::str::from_utf8(payload)
                .map_err(|_| PyValueError::new_err("Invalid UTF-8 in string property"))?
                .to_string(),
        ),
        TYPE_F64_ARRAY => PropertyValue::FloatArray(decode_f64_array(payload)?),
        TYPE_VEC3_F32 => PropertyValue::Vec3Array(decode_vec3_f32(payload)?),
        TYPE_I64_ARRAY => PropertyValue::IntArray(decode_i64_array(payload)?),
        TYPE_F32_ARRAY => PropertyValue::Float32Array(decode_f32_array(payload)?),
        TYPE_VEC3_F64 => PropertyValue::Vec3ArrayF64(decode_vec3_f64(payload)?),
        TYPE_I32_ARRAY => PropertyValue::Int32Array(decode_i32_array(payload)?),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported property type tag {}",
                type_tag
            )));
        }
    })
}
