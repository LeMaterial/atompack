use super::*;

#[cfg(target_endian = "little")]
fn copy_bytes_to_vec<T: Copy>(bytes: &[u8]) -> PyResult<Vec<T>> {
    let elem = std::mem::size_of::<T>();
    if elem == 0 || !bytes.len().is_multiple_of(elem) {
        return Err(PyValueError::new_err(format!(
            "Byte length {} is not a multiple of element size {}",
            bytes.len(),
            elem
        )));
    }
    let len = bytes.len() / elem;
    let mut out = Vec::<T>::with_capacity(len);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
        out.set_len(len);
    }
    Ok(out)
}

/// Try zero-cost cast, fall back to per-element decode if unaligned.
/// Returns a borrowed slice when aligned (no copy), owned Vec only when decode is needed.
pub(crate) fn cast_or_decode_f32(bytes: &[u8]) -> PyResult<Cow<'_, [f32]>> {
    match bytemuck::try_cast_slice::<u8, f32>(bytes) {
        Ok(slice) => Ok(Cow::Borrowed(slice)),
        #[cfg(target_endian = "little")]
        Err(_) => copy_bytes_to_vec(bytes).map(Cow::Owned),
        #[cfg(not(target_endian = "little"))]
        Err(_) => decode_f32_array(bytes).map(Cow::Owned),
    }
}

pub(crate) fn cast_or_decode_f64(bytes: &[u8]) -> PyResult<Cow<'_, [f64]>> {
    match bytemuck::try_cast_slice::<u8, f64>(bytes) {
        Ok(slice) => Ok(Cow::Borrowed(slice)),
        #[cfg(target_endian = "little")]
        Err(_) => copy_bytes_to_vec(bytes).map(Cow::Owned),
        #[cfg(not(target_endian = "little"))]
        Err(_) => decode_f64_array(bytes).map(Cow::Owned),
    }
}

pub(crate) fn cast_or_decode_i64(bytes: &[u8]) -> PyResult<Cow<'_, [i64]>> {
    match bytemuck::try_cast_slice::<u8, i64>(bytes) {
        Ok(slice) => Ok(Cow::Borrowed(slice)),
        #[cfg(target_endian = "little")]
        Err(_) => copy_bytes_to_vec(bytes).map(Cow::Owned),
        #[cfg(not(target_endian = "little"))]
        Err(_) => decode_i64_array(bytes).map(Cow::Owned),
    }
}

pub(crate) fn cast_or_decode_i32(bytes: &[u8]) -> PyResult<Cow<'_, [i32]>> {
    match bytemuck::try_cast_slice::<u8, i32>(bytes) {
        Ok(slice) => Ok(Cow::Borrowed(slice)),
        #[cfg(target_endian = "little")]
        Err(_) => copy_bytes_to_vec(bytes).map(Cow::Owned),
        #[cfg(not(target_endian = "little"))]
        Err(_) => decode_i32_array(bytes).map(Cow::Owned),
    }
}

pub(super) fn pyarray2_from_flat<'py, T: Element>(
    py: Python<'py>,
    data: Vec<T>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    PyArray1::from_vec(py, data)
        .reshape([rows, cols])
        .map_err(|e| PyValueError::new_err(format!("{}", e)))
}

pub(crate) fn pyarray1_from_cow<'py, T: Element + Clone>(
    py: Python<'py>,
    data: Cow<'_, [T]>,
) -> Bound<'py, PyArray1<T>> {
    match data {
        Cow::Borrowed(slice) => PyArray1::from_slice(py, slice),
        Cow::Owned(vec) => PyArray1::from_vec(py, vec),
    }
}

fn pyarray2_from_slice<'py, T: Element>(
    py: Python<'py>,
    data: &[T],
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    PyArray1::from_slice(py, data)
        .reshape([rows, cols])
        .map_err(|e| PyValueError::new_err(format!("{}", e)))
}

pub(crate) fn pyarray2_from_cow<'py, T: Element + Clone>(
    py: Python<'py>,
    data: Cow<'_, [T]>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    match data {
        Cow::Borrowed(slice) => pyarray2_from_slice(py, slice, rows, cols),
        Cow::Owned(vec) => pyarray2_from_flat(py, vec, rows, cols),
    }
}

struct WrittenSoaSection {
    slot: BuiltinSlot,
    section: LazySection,
}

fn write_soa_section_raw(
    buf: &mut Vec<u8>,
    kind: u8,
    key: &str,
    type_tag: u8,
    payload: &[u8],
) -> Result<WrittenSoaSection, String> {
    let key_len: u8 = key
        .len()
        .try_into()
        .map_err(|_| format!("Section key '{}' is too long", key))?;
    let payload_len: u32 = payload
        .len()
        .try_into()
        .map_err(|_| format!("Section '{}' payload is too large", key))?;
    buf.push(kind);
    buf.push(key_len);
    let key_start = buf.len();
    buf.extend_from_slice(key.as_bytes());
    buf.push(type_tag);
    buf.extend_from_slice(&payload_len.to_le_bytes());
    let payload_start = buf.len();
    buf.extend_from_slice(payload);
    Ok(WrittenSoaSection {
        slot: (payload_start, payload.len(), type_tag),
        section: LazySection {
            kind,
            key_start,
            key_len,
            type_tag,
            payload_start,
            payload_len: payload.len(),
        },
    })
}

pub(crate) struct SoaBuiltinPayloads<'a> {
    pub(crate) energy: Option<f64>,
    pub(crate) forces: Option<&'a [u8]>,
    pub(crate) charges: Option<&'a [u8]>,
    pub(crate) velocities: Option<&'a [u8]>,
    pub(crate) cell: Option<&'a [u8]>,
    pub(crate) stress: Option<&'a [u8]>,
    pub(crate) pbc: Option<[bool; 3]>,
    pub(crate) name: Option<&'a str>,
}

pub(crate) struct SoaCustomSection<'a> {
    pub(crate) kind: u8,
    pub(crate) key: &'a str,
    pub(crate) type_tag: u8,
    pub(crate) payload: &'a [u8],
}

pub(crate) struct BuiltSoaRecord {
    bytes: Vec<u8>,
    n_atoms: usize,
    positions_start: usize,
    atomic_numbers_start: usize,
    forces: Option<BuiltinSlot>,
    energy: Option<BuiltinSlot>,
    cell: Option<BuiltinSlot>,
    stress: Option<BuiltinSlot>,
    charges: Option<BuiltinSlot>,
    velocities: Option<BuiltinSlot>,
    pbc: Option<BuiltinSlot>,
    name: Option<BuiltinSlot>,
    custom_sections: Vec<LazySection>,
}

impl BuiltSoaRecord {
    pub(crate) fn into_parts(self) -> (Vec<u8>, u32) {
        (self.bytes, self.n_atoms as u32)
    }

    pub(crate) fn into_view(self) -> SoaMoleculeView {
        SoaMoleculeView {
            bytes: SoaBytes::Owned(self.bytes),
            n_atoms: self.n_atoms,
            positions_start: self.positions_start,
            atomic_numbers_start: self.atomic_numbers_start,
            forces: self.forces,
            energy: self.energy,
            cell: self.cell,
            stress: self.stress,
            charges: self.charges,
            velocities: self.velocities,
            pbc: self.pbc,
            name: self.name,
            custom_sections: self.custom_sections,
        }
    }
}

pub(crate) fn build_soa_record(
    positions: &[f32],
    atomic_numbers: &[u8],
    builtins: SoaBuiltinPayloads<'_>,
) -> Result<BuiltSoaRecord, String> {
    build_soa_record_with_custom(positions, atomic_numbers, builtins, &[])
}

pub(crate) fn build_soa_record_with_custom(
    positions: &[f32],
    atomic_numbers: &[u8],
    builtins: SoaBuiltinPayloads<'_>,
    custom_sections: &[SoaCustomSection<'_>],
) -> Result<BuiltSoaRecord, String> {
    if !positions.len().is_multiple_of(3) {
        return Err("positions length must be divisible by 3".to_string());
    }
    let n_atoms = positions.len() / 3;
    if atomic_numbers.len() != n_atoms {
        return Err(format!(
            "Atomic numbers length ({}) doesn't match atom count ({})",
            atomic_numbers.len(),
            n_atoms
        ));
    }

    let validate_bytes = |payload: &[u8], expected: usize, label: &str| -> Result<(), String> {
        if payload.len() != expected {
            return Err(format!(
                "{} payload length ({}) doesn't match expected byte length ({})",
                label,
                payload.len(),
                expected
            ));
        }
        Ok(())
    };

    if let Some(payload) = builtins.forces {
        validate_bytes(payload, n_atoms * 12, "forces")?;
    }
    if let Some(payload) = builtins.charges {
        validate_bytes(payload, n_atoms * 8, "charges")?;
    }
    if let Some(payload) = builtins.velocities {
        validate_bytes(payload, n_atoms * 12, "velocities")?;
    }
    if let Some(payload) = builtins.cell {
        validate_bytes(payload, 72, "cell")?;
    }
    if let Some(payload) = builtins.stress {
        validate_bytes(payload, 72, "stress")?;
    }

    let mut n_sections = 0u16;
    let mut payload_bytes = 0usize;
    let mut section_overhead = 0usize;
    let mut account_section = |payload_len: usize, key_len: usize| {
        n_sections += 1;
        payload_bytes += payload_len;
        section_overhead += 1 + 1 + key_len + 1 + 4;
    };

    if let Some(payload) = builtins.charges {
        account_section(payload.len(), "charges".len());
    }
    if let Some(payload) = builtins.cell {
        account_section(payload.len(), "cell".len());
    }
    if builtins.energy.is_some() {
        account_section(std::mem::size_of::<f64>(), "energy".len());
    }
    if let Some(payload) = builtins.forces {
        account_section(payload.len(), "forces".len());
    }
    if let Some(value) = builtins.name {
        account_section(value.len(), "name".len());
    }
    if builtins.pbc.is_some() {
        account_section(3, "pbc".len());
    }
    if let Some(payload) = builtins.stress {
        account_section(payload.len(), "stress".len());
    }
    if let Some(payload) = builtins.velocities {
        account_section(payload.len(), "velocities".len());
    }
    for section in custom_sections {
        let parsed = SectionRef {
            kind: section.kind,
            key: section.key,
            type_tag: section.type_tag,
            payload: section.payload,
        };
        let per_atom = is_per_atom(parsed.kind, parsed.key, parsed.type_tag);
        let elem_bytes = match parsed.type_tag {
            TYPE_STRING => 0,
            tag if per_atom => {
                let elem_bytes = type_tag_elem_bytes(tag);
                if elem_bytes == 0 {
                    return Err(format!(
                        "Unsupported per-atom section type tag {} for key '{}'",
                        tag, parsed.key
                    ));
                }
                elem_bytes
            }
            TYPE_FLOAT | TYPE_INT => 8,
            TYPE_BOOL3 => 3,
            TYPE_MAT3X3_F64 => 72,
            _ => parsed.payload.len(),
        };
        let slot_bytes = if parsed.type_tag == TYPE_STRING {
            0
        } else if per_atom {
            elem_bytes
        } else {
            parsed.payload.len()
        };
        validate_section_payload(&parsed, per_atom, elem_bytes, slot_bytes, n_atoms)
            .map_err(|e| format!("{}", e))?;
        account_section(parsed.payload.len(), parsed.key.len());
    }

    let positions_start = 4usize;
    let atomic_numbers_start = positions_start + positions.len() * 4;
    let mut buf = Vec::with_capacity(
        4 + positions.len() * 4 + atomic_numbers.len() + 2 + section_overhead + payload_bytes,
    );
    buf.extend_from_slice(&(n_atoms as u32).to_le_bytes());
    buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(positions));
    buf.extend_from_slice(atomic_numbers);
    buf.extend_from_slice(&n_sections.to_le_bytes());

    let mut charges = None;
    let mut cell = None;
    let mut energy_slot = None;
    let mut forces = None;
    let mut name = None;
    let mut pbc = None;
    let mut stress = None;
    let mut velocities = None;
    let mut custom_slots = Vec::with_capacity(custom_sections.len());

    if let Some(payload) = builtins.charges {
        charges = Some(
            write_soa_section_raw(&mut buf, KIND_BUILTIN, "charges", TYPE_F64_ARRAY, payload)?.slot,
        );
    }
    if let Some(payload) = builtins.cell {
        cell = Some(
            write_soa_section_raw(&mut buf, KIND_BUILTIN, "cell", TYPE_MAT3X3_F64, payload)?.slot,
        );
    }
    if let Some(value) = builtins.energy {
        let payload = value.to_le_bytes();
        energy_slot = Some(
            write_soa_section_raw(&mut buf, KIND_BUILTIN, "energy", TYPE_FLOAT, &payload)?.slot,
        );
    }
    if let Some(payload) = builtins.forces {
        forces = Some(
            write_soa_section_raw(&mut buf, KIND_BUILTIN, "forces", TYPE_VEC3_F32, payload)?.slot,
        );
    }
    if let Some(value) = builtins.name {
        name = Some(
            write_soa_section_raw(
                &mut buf,
                KIND_BUILTIN,
                "name",
                TYPE_STRING,
                value.as_bytes(),
            )?
            .slot,
        );
    }
    if let Some([a, b, c]) = builtins.pbc {
        let payload = [a as u8, b as u8, c as u8];
        pbc =
            Some(write_soa_section_raw(&mut buf, KIND_BUILTIN, "pbc", TYPE_BOOL3, &payload)?.slot);
    }
    if let Some(payload) = builtins.stress {
        stress = Some(
            write_soa_section_raw(&mut buf, KIND_BUILTIN, "stress", TYPE_MAT3X3_F64, payload)?.slot,
        );
    }
    if let Some(payload) = builtins.velocities {
        velocities = Some(
            write_soa_section_raw(&mut buf, KIND_BUILTIN, "velocities", TYPE_VEC3_F32, payload)?
                .slot,
        );
    }
    for section in custom_sections {
        custom_slots.push(
            write_soa_section_raw(
                &mut buf,
                section.kind,
                section.key,
                section.type_tag,
                section.payload,
            )?
            .section,
        );
    }

    Ok(BuiltSoaRecord {
        bytes: buf,
        n_atoms,
        positions_start,
        atomic_numbers_start,
        forces,
        energy: energy_slot,
        cell,
        stress,
        charges,
        velocities,
        pbc,
        name,
        custom_sections: custom_slots,
    })
}

fn vec3_f32_payload<'py>(
    readonly: &'py numpy::PyReadonlyArray2<'py, f32>,
    label: &str,
    expected_rows: usize,
) -> PyResult<&'py [u8]> {
    let arr = readonly.as_array();
    let shape = arr.shape();
    if shape != [expected_rows, 3] {
        return Err(PyValueError::new_err(format!(
            "{} must have shape ({}, 3)",
            label, expected_rows
        )));
    }
    let slice = readonly
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{} must be C-contiguous", label)))?;
    Ok(bytemuck::cast_slice::<f32, u8>(slice))
}

fn vec1_f64_payload<'py>(
    readonly: &'py numpy::PyReadonlyArray1<'py, f64>,
    label: &str,
    expected_len: usize,
) -> PyResult<&'py [u8]> {
    let arr = readonly.as_array();
    if arr.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "{} length ({}) doesn't match atom count ({})",
            label,
            arr.len(),
            expected_len
        )));
    }
    let slice = readonly
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{} must be C-contiguous", label)))?;
    Ok(bytemuck::cast_slice::<f64, u8>(slice))
}

fn mat3x3_f64_payload<'py>(
    readonly: &'py numpy::PyReadonlyArray2<'py, f64>,
    label: &str,
) -> PyResult<&'py [u8]> {
    let arr = readonly.as_array();
    if arr.shape() != [3, 3] {
        return Err(PyValueError::new_err(format!(
            "{} must have shape (3, 3)",
            label
        )));
    }
    let slice = readonly
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{} must be C-contiguous", label)))?;
    Ok(bytemuck::cast_slice::<f64, u8>(slice))
}

fn mat3x3_f64_payload_from_any(py: Python<'_>, value: Py<PyAny>, label: &str) -> PyResult<Vec<u8>> {
    let value = value.bind(py);
    if let Ok(arr) = value.cast::<PyArray2<f64>>() {
        let readonly = arr.readonly();
        return Ok(mat3x3_f64_payload(&readonly, label)?.to_vec());
    }
    if let Ok(arr) = value.cast::<PyArray2<f32>>() {
        let readonly = arr.readonly();
        let arr = readonly.as_array();
        if arr.shape() != [3, 3] {
            return Err(PyValueError::new_err(format!(
                "{} must have shape (3, 3)",
                label
            )));
        }
        let mut payload = Vec::with_capacity(72);
        for row in arr.outer_iter() {
            for value in row {
                payload.extend_from_slice(&(*value as f64).to_le_bytes());
            }
        }
        return Ok(payload);
    }
    Err(PyValueError::new_err(format!(
        "{} must be a float32 or float64 ndarray with shape (3, 3)",
        label
    )))
}

pub(super) fn into_py_any<'py, T>(py: Python<'py>, value: T) -> PyResult<Py<PyAny>>
where
    T: IntoPyObject<'py>,
{
    Ok(value.into_bound_py_any(py)?.unbind())
}

pub(super) fn property_value_to_pyobject(
    py: Python<'_>,
    value: &PropertyValue,
) -> PyResult<Py<PyAny>> {
    Ok(match value {
        PropertyValue::Float(v) => into_py_any(py, *v)?,
        PropertyValue::Int(v) => into_py_any(py, *v)?,
        PropertyValue::String(v) => into_py_any(py, v)?,
        PropertyValue::FloatArray(v) => into_py_any(py, PyArray1::from_slice(py, v))?,
        PropertyValue::Float32Array(v) => into_py_any(py, PyArray1::from_slice(py, v))?,
        PropertyValue::Vec3Array(v) => {
            let n = v.len();
            let flat: Vec<f32> = v.iter().flat_map(|x| [x[0], x[1], x[2]]).collect();
            into_py_any(py, pyarray2_from_flat(py, flat, n, 3)?)?
        }
        PropertyValue::Vec3ArrayF64(v) => {
            let n = v.len();
            let flat: Vec<f64> = v.iter().flat_map(|x| [x[0], x[1], x[2]]).collect();
            into_py_any(py, pyarray2_from_flat(py, flat, n, 3)?)?
        }
        PropertyValue::IntArray(v) => into_py_any(py, PyArray1::from_slice(py, v))?,
        PropertyValue::Int32Array(v) => into_py_any(py, PyArray1::from_slice(py, v))?,
    })
}

pub(super) fn property_section_to_pyobject<'py>(
    py: Python<'py>,
    view: &SoaMoleculeView,
    section: &LazySection,
) -> PyResult<Py<PyAny>> {
    let payload = view.lazy_section_payload(section);
    let expect_vec3_payload = |byte_width: usize, label: &str| -> PyResult<usize> {
        let stride = byte_width * 3;
        if !payload.len().is_multiple_of(stride) {
            return Err(PyValueError::new_err(format!(
                "Invalid {} payload length",
                label
            )));
        }
        Ok(payload.len() / stride)
    };
    Ok(match section.type_tag {
        TYPE_FLOAT => into_py_any(py, read_f64_scalar(payload)?)?,
        TYPE_INT => into_py_any(py, read_i64_scalar(payload)?)?,
        TYPE_STRING => into_py_any(
            py,
            std::str::from_utf8(payload)
                .map_err(|_| PyValueError::new_err("Invalid UTF-8 in string property"))?
                .to_string(),
        )?,
        TYPE_F64_ARRAY => into_py_any(py, pyarray1_from_cow(py, cast_or_decode_f64(payload)?))?,
        TYPE_VEC3_F32 => {
            let rows = expect_vec3_payload(4, "vec3<f32>")?;
            into_py_any(
                py,
                pyarray2_from_cow(py, cast_or_decode_f32(payload)?, rows, 3)?,
            )?
        }
        TYPE_I64_ARRAY => into_py_any(py, pyarray1_from_cow(py, cast_or_decode_i64(payload)?))?,
        TYPE_F32_ARRAY => into_py_any(py, pyarray1_from_cow(py, cast_or_decode_f32(payload)?))?,
        TYPE_VEC3_F64 => {
            let rows = expect_vec3_payload(8, "vec3<f64>")?;
            into_py_any(
                py,
                pyarray2_from_cow(py, cast_or_decode_f64(payload)?, rows, 3)?,
            )?
        }
        TYPE_I32_ARRAY => into_py_any(py, pyarray1_from_cow(py, cast_or_decode_i32(payload)?))?,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported property type tag {}",
                section.type_tag
            )));
        }
    })
}

fn property_value_is_atom_array(value: &PropertyValue, n_atoms: usize) -> bool {
    match value {
        PropertyValue::FloatArray(values) => values.len() == n_atoms,
        PropertyValue::Vec3Array(values) => values.len() == n_atoms,
        PropertyValue::IntArray(values) => values.len() == n_atoms,
        PropertyValue::Float32Array(values) => values.len() == n_atoms,
        PropertyValue::Vec3ArrayF64(values) => values.len() == n_atoms,
        PropertyValue::Int32Array(values) => values.len() == n_atoms,
        PropertyValue::Float(_) | PropertyValue::Int(_) | PropertyValue::String(_) => false,
    }
}

fn section_is_atom_array(view: &SoaMoleculeView, section: &LazySection) -> bool {
    match section.type_tag {
        TYPE_F64_ARRAY => section.payload_len == view.n_atoms * 8,
        TYPE_VEC3_F32 => section.payload_len == view.n_atoms * 12,
        TYPE_I64_ARRAY => section.payload_len == view.n_atoms * 8,
        TYPE_F32_ARRAY => section.payload_len == view.n_atoms * 4,
        TYPE_VEC3_F64 => section.payload_len == view.n_atoms * 24,
        TYPE_I32_ARRAY => section.payload_len == view.n_atoms * 4,
        _ => false,
    }
}

fn is_reserved_ase_array_key(key: &str) -> bool {
    matches!(key, "numbers" | "positions")
}

fn owned_vec3_array<'py>(
    py: Python<'py>,
    values: &[[f32; 3]],
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let n_atoms = values.len();
    let flat: Vec<f32> = values
        .iter()
        .flat_map(|value| [value[0], value[1], value[2]])
        .collect();
    pyarray2_from_flat(py, flat, n_atoms, 3)
}

fn owned_mat3x3_array<'py>(
    py: Python<'py>,
    values: &[[f64; 3]; 3],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat: Vec<f64> = values
        .iter()
        .flat_map(|row| [row[0], row[1], row[2]])
        .collect();
    pyarray2_from_flat(py, flat, 3, 3)
}

impl PyMolecule {
    #[allow(dead_code)]
    pub(crate) fn from_owned(inner: Molecule) -> Self {
        Self {
            backing: MoleculeBacking::Owned(inner),
        }
    }

    pub(crate) fn from_view(view: SoaMoleculeView) -> Self {
        Self {
            backing: MoleculeBacking::View(view),
        }
    }

    pub(super) fn as_owned(&self) -> Option<&Molecule> {
        match &self.backing {
            MoleculeBacking::Owned(inner) => Some(inner),
            MoleculeBacking::View(_) => None,
        }
    }

    pub(super) fn as_view(&self) -> Option<&SoaMoleculeView> {
        match &self.backing {
            MoleculeBacking::Owned(_) => None,
            MoleculeBacking::View(view) => Some(view),
        }
    }

    pub(super) fn len(&self) -> usize {
        match &self.backing {
            MoleculeBacking::Owned(inner) => inner.len(),
            MoleculeBacking::View(view) => view.n_atoms,
        }
    }

    pub(super) fn ensure_owned(&mut self) -> PyResult<&mut Molecule> {
        if let MoleculeBacking::View(view) = &self.backing {
            self.backing = MoleculeBacking::Owned(view.materialize()?);
        }
        match &mut self.backing {
            MoleculeBacking::Owned(inner) => Ok(inner),
            MoleculeBacking::View(_) => Err(PyValueError::new_err(
                "Failed to materialize owned molecule state",
            )),
        }
    }

    pub(crate) fn clone_as_owned(&self) -> PyResult<Molecule> {
        match &self.backing {
            MoleculeBacking::Owned(inner) => Ok(inner.clone()),
            MoleculeBacking::View(view) => view.materialize(),
        }
    }

    pub(crate) fn soa_bytes(&self) -> Option<(&[u8], u32)> {
        match &self.backing {
            MoleculeBacking::View(view) => Some((view.bytes.as_slice(), view.n_atoms as u32)),
            MoleculeBacking::Owned(_) => None,
        }
    }

    pub(super) fn positions_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if let Some(inner) = self.as_owned() {
            return pyarray2_from_flat(py, inner.positions_flat(), inner.len(), 3);
        }
        let view = self.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        let positions = cast_or_decode_f32(view.positions_bytes())?;
        pyarray2_from_cow(py, positions, view.n_atoms, 3)
    }

    pub(super) fn atomic_numbers_py<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        if let Some(inner) = self.as_owned() {
            return Ok(PyArray1::from_slice(py, &inner.atomic_numbers));
        }
        let view = self.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        Ok(PyArray1::from_slice(py, view.atomic_numbers_bytes()))
    }

    pub(super) fn forces_py<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        if let Some(inner) = self.as_owned() {
            return inner
                .forces
                .as_ref()
                .map(|forces| owned_vec3_array(py, forces))
                .transpose();
        }
        let view = self.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        let Some(slot) = view.forces else {
            return Ok(None);
        };
        if slot.2 != TYPE_VEC3_F32 || slot.1 != view.n_atoms * 12 {
            return Err(PyValueError::new_err("Invalid forces section"));
        }
        let data = cast_or_decode_f32(view.builtin_payload(slot))?;
        Ok(Some(pyarray2_from_cow(py, data, view.n_atoms, 3)?))
    }

    pub(super) fn charges_py<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        if let Some(inner) = self.as_owned() {
            return Ok(inner
                .charges
                .as_ref()
                .map(|charges| PyArray1::from_slice(py, charges)));
        }
        let view = self.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        let Some(slot) = view.charges else {
            return Ok(None);
        };
        if slot.2 != TYPE_F64_ARRAY || slot.1 != view.n_atoms * 8 {
            return Err(PyValueError::new_err("Invalid charges section"));
        }
        let data = cast_or_decode_f64(view.builtin_payload(slot))?;
        Ok(Some(pyarray1_from_cow(py, data)))
    }

    pub(super) fn velocities_py<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        if let Some(inner) = self.as_owned() {
            return inner
                .velocities
                .as_ref()
                .map(|velocities| owned_vec3_array(py, velocities))
                .transpose();
        }
        let view = self.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        let Some(slot) = view.velocities else {
            return Ok(None);
        };
        if slot.2 != TYPE_VEC3_F32 || slot.1 != view.n_atoms * 12 {
            return Err(PyValueError::new_err("Invalid velocities section"));
        }
        let data = cast_or_decode_f32(view.builtin_payload(slot))?;
        Ok(Some(pyarray2_from_cow(py, data, view.n_atoms, 3)?))
    }

    pub(super) fn cell_py<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        if let Some(inner) = self.as_owned() {
            return inner
                .cell
                .as_ref()
                .map(|cell| owned_mat3x3_array(py, cell))
                .transpose();
        }
        let view = self.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        let Some(slot) = view.cell else {
            return Ok(None);
        };
        if slot.2 != TYPE_MAT3X3_F64 || slot.1 != 72 {
            return Err(PyValueError::new_err("Invalid cell section"));
        }
        let data = cast_or_decode_f64(view.builtin_payload(slot))?;
        Ok(Some(pyarray2_from_cow(py, data, 3, 3)?))
    }

    pub(super) fn stress_py<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        if let Some(inner) = self.as_owned() {
            return inner
                .stress
                .as_ref()
                .map(|stress| owned_mat3x3_array(py, stress))
                .transpose();
        }
        let view = self.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        let Some(slot) = view.stress else {
            return Ok(None);
        };
        if slot.2 != TYPE_MAT3X3_F64 || slot.1 != 72 {
            return Err(PyValueError::new_err("Invalid stress section"));
        }
        let data = cast_or_decode_f64(view.builtin_payload(slot))?;
        Ok(Some(pyarray2_from_cow(py, data, 3, 3)?))
    }

    pub(super) fn append_owned_ase_properties<'py>(
        &self,
        py: Python<'py>,
        arrays: &Bound<'py, PyDict>,
        info: &Bound<'py, PyDict>,
        copy_info: bool,
        copy_arrays: bool,
    ) -> PyResult<()> {
        let Some(inner) = self.as_owned() else {
            return Ok(());
        };

        if copy_arrays {
            for (key, value) in &inner.atom_properties {
                if is_reserved_ase_array_key(key) {
                    continue;
                }
                arrays.set_item(key, property_value_to_pyobject(py, value)?)?;
            }
        }

        for (key, value) in &inner.properties {
            let is_atom_array = property_value_is_atom_array(value, inner.len());
            if copy_arrays && is_atom_array && !is_reserved_ase_array_key(key) {
                arrays.set_item(key, property_value_to_pyobject(py, value)?)?;
            } else if copy_info {
                info.set_item(key, property_value_to_pyobject(py, value)?)?;
            }
        }

        Ok(())
    }

    pub(super) fn append_view_ase_properties<'py>(
        &self,
        py: Python<'py>,
        arrays: &Bound<'py, PyDict>,
        info: &Bound<'py, PyDict>,
        copy_info: bool,
        copy_arrays: bool,
    ) -> PyResult<()> {
        let Some(view) = self.as_view() else {
            return Ok(());
        };

        for section in &view.custom_sections {
            let key = view.lazy_section_key(section)?;
            match section.kind {
                KIND_ATOM_PROP if copy_arrays && !is_reserved_ase_array_key(key) => {
                    arrays.set_item(key, property_section_to_pyobject(py, view, section)?)?;
                }
                KIND_MOL_PROP => {
                    let is_atom_array = section_is_atom_array(view, section);
                    if copy_arrays && is_atom_array && !is_reserved_ase_array_key(key) {
                        arrays.set_item(key, property_section_to_pyobject(py, view, section)?)?;
                    } else if copy_info {
                        info.set_item(key, property_section_to_pyobject(py, view, section)?)?;
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_arrays_impl(
        py: Python<'_>,
        positions: &Bound<'_, PyArray2<f32>>,
        atomic_numbers: &Bound<'_, PyArray1<u8>>,
        energy: Option<f64>,
        forces: Option<&Bound<'_, PyArray2<f32>>>,
        charges: Option<&Bound<'_, PyArray1<f64>>>,
        velocities: Option<&Bound<'_, PyArray2<f32>>>,
        cell: Option<&Bound<'_, PyArray2<f64>>>,
        stress: Option<Py<PyAny>>,
        pbc: Option<(bool, bool, bool)>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let pos = positions.readonly();
        let pos_arr = pos.as_array();
        let shape = pos_arr.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(PyValueError::new_err(
                "positions must have shape (n_atoms, 3)",
            ));
        }

        let z = atomic_numbers.readonly();
        let z_arr = z.as_array();
        let n_atoms = shape[0];
        if z_arr.len() != n_atoms {
            return Err(PyValueError::new_err(format!(
                "Atomic numbers length ({}) doesn't match atom count ({})",
                z_arr.len(),
                n_atoms
            )));
        }

        let pos_bytes = pos_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("positions must be C-contiguous"))?;
        let z_bytes = z_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("atomic_numbers must be C-contiguous"))?;

        let forces_readonly = forces.map(|value| value.readonly());
        let charges_readonly = charges.map(|value| value.readonly());
        let velocities_readonly = velocities.map(|value| value.readonly());
        let cell_readonly = cell.map(|value| value.readonly());

        let forces_payload = forces_readonly
            .as_ref()
            .map(|readonly| vec3_f32_payload(readonly, "forces", n_atoms))
            .transpose()?;
        let charges_payload = charges_readonly
            .as_ref()
            .map(|readonly| vec1_f64_payload(readonly, "charges", n_atoms))
            .transpose()?;
        let velocities_payload = velocities_readonly
            .as_ref()
            .map(|readonly| vec3_f32_payload(readonly, "velocities", n_atoms))
            .transpose()?;
        let cell_payload = cell_readonly
            .as_ref()
            .map(|readonly| mat3x3_f64_payload(readonly, "cell"))
            .transpose()?;
        let stress_payload = stress
            .map(|value| mat3x3_f64_payload_from_any(py, value, "stress"))
            .transpose()?;
        let record = build_soa_record(
            pos_bytes,
            z_bytes,
            SoaBuiltinPayloads {
                energy,
                forces: forces_payload,
                charges: charges_payload,
                velocities: velocities_payload,
                cell: cell_payload,
                stress: stress_payload.as_deref(),
                pbc: pbc.map(|(a, b, c)| [a, b, c]),
                name: name.as_deref(),
            },
        )
        .map_err(PyValueError::new_err)?;

        Ok(Self::from_view(record.into_view()))
    }
}
