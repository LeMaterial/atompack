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

pub(crate) struct SoaSection<'a> {
    pub(crate) kind: u8,
    pub(crate) key: &'a str,
    pub(crate) type_tag: u8,
    pub(crate) payload: &'a [u8],
}

pub(crate) struct SoaRecord<'a> {
    pub(crate) record_format: u32,
    pub(crate) positions_type: u8,
    pub(crate) positions: &'a [u8],
    pub(crate) atomic_numbers: &'a [u8],
    pub(crate) sections: &'a [SoaSection<'a>],
}

fn write_soa_section_raw(buf: &mut Vec<u8>, section: &SoaSection<'_>) -> Result<(), String> {
    let key_len: u8 = section
        .key
        .len()
        .try_into()
        .map_err(|_| format!("Section key '{}' is too long", section.key))?;
    let payload_len: u32 = section
        .payload
        .len()
        .try_into()
        .map_err(|_| format!("Section '{}' payload is too large", section.key))?;
    buf.push(section.kind);
    buf.push(key_len);
    buf.extend_from_slice(section.key.as_bytes());
    buf.push(section.type_tag);
    buf.extend_from_slice(&payload_len.to_le_bytes());
    buf.extend_from_slice(section.payload);
    Ok(())
}

pub(crate) fn build_soa_record(record: SoaRecord<'_>) -> Result<Vec<u8>, String> {
    if !matches!(
        record.record_format,
        RECORD_FORMAT_SOA_V2 | RECORD_FORMAT_SOA_V3
    ) {
        return Err(format!(
            "Unsupported record format {}",
            record.record_format
        ));
    }
    let positions_elem_bytes = match record.positions_type {
        TYPE_VEC3_F32 => 12usize,
        TYPE_VEC3_F64 => 24usize,
        other => {
            return Err(format!("Unsupported positions type tag {}", other));
        }
    };
    if record.record_format == RECORD_FORMAT_SOA_V2 && record.positions_type != TYPE_VEC3_F32 {
        return Err("record format 2 only supports float32 positions".to_string());
    }
    if !record.positions.len().is_multiple_of(positions_elem_bytes) {
        return Err(format!(
            "positions payload length ({}) is not a multiple of {}",
            record.positions.len(),
            positions_elem_bytes
        ));
    }
    let n_atoms = record.positions.len() / positions_elem_bytes;
    if record.atomic_numbers.len() != n_atoms {
        return Err(format!(
            "Atomic numbers length ({}) doesn't match atom count ({})",
            record.atomic_numbers.len(),
            n_atoms
        ));
    }

    let mut n_sections = 0u16;
    let mut payload_bytes = 0usize;
    let mut section_overhead = 0usize;
    let mut account_section = |payload_len: usize, key_len: usize| {
        n_sections += 1;
        payload_bytes += payload_len;
        section_overhead += 1 + 1 + key_len + 1 + 4;
    };

    for section in record.sections {
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
            TYPE_FLOAT32 => 4,
            TYPE_BOOL3 => 3,
            TYPE_MAT3X3_F32 => 36,
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

    let mut buf = Vec::with_capacity(
        4 + record.positions.len()
            + record.atomic_numbers.len()
            + 2
            + section_overhead
            + payload_bytes,
    );
    buf.extend_from_slice(&(n_atoms as u32).to_le_bytes());
    buf.extend_from_slice(record.positions);
    buf.extend_from_slice(record.atomic_numbers);
    buf.extend_from_slice(&n_sections.to_le_bytes());

    for section in record.sections {
        write_soa_section_raw(&mut buf, section)?;
    }

    Ok(buf)
}

fn molecule_from_positions(
    positions: Vec3Data,
    atomic_numbers: Vec<u8>,
) -> Result<Molecule, String> {
    match positions {
        Vec3Data::F32(values) => Molecule::new(values, atomic_numbers),
        Vec3Data::F64(values) => Molecule::new_f64(values, atomic_numbers),
    }
}

pub(crate) fn parse_vec3_field(
    value: &Bound<'_, PyAny>,
    label: &str,
    expected_rows: usize,
) -> PyResult<Vec3Data> {
    if let Ok(arr) = value.cast::<PyArray2<f32>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        if view.shape() != [expected_rows, 3] {
            return Err(PyValueError::new_err(format!(
                "{} must have shape ({}, 3)",
                label, expected_rows
            )));
        }
        return Ok(Vec3Data::F32(
            view.outer_iter()
                .map(|row| [row[0], row[1], row[2]])
                .collect(),
        ));
    }
    if let Ok(arr) = value.cast::<PyArray2<f64>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        if view.shape() != [expected_rows, 3] {
            return Err(PyValueError::new_err(format!(
                "{} must have shape ({}, 3)",
                label, expected_rows
            )));
        }
        return Ok(Vec3Data::F64(
            view.outer_iter()
                .map(|row| [row[0], row[1], row[2]])
                .collect(),
        ));
    }
    Err(PyValueError::new_err(format!(
        "{} must be a float32 or float64 ndarray with shape ({}, 3)",
        label, expected_rows
    )))
}

fn parse_positions_field(value: &Bound<'_, PyAny>) -> PyResult<Vec3Data> {
    if let Ok(arr) = value.cast::<PyArray2<f32>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        if view.shape().len() != 2 || view.shape()[1] != 3 {
            return Err(PyValueError::new_err(
                "positions must have shape (n_atoms, 3)",
            ));
        }
        return Ok(Vec3Data::F32(
            view.outer_iter()
                .map(|row| [row[0], row[1], row[2]])
                .collect(),
        ));
    }
    if let Ok(arr) = value.cast::<PyArray2<f64>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        if view.shape().len() != 2 || view.shape()[1] != 3 {
            return Err(PyValueError::new_err(
                "positions must have shape (n_atoms, 3)",
            ));
        }
        return Ok(Vec3Data::F64(
            view.outer_iter()
                .map(|row| [row[0], row[1], row[2]])
                .collect(),
        ));
    }
    Err(PyValueError::new_err(
        "positions must be a float32 or float64 ndarray with shape (n_atoms, 3)",
    ))
}

pub(crate) fn parse_float_array_field(
    value: &Bound<'_, PyAny>,
    label: &str,
    expected_len: usize,
) -> PyResult<FloatArrayData> {
    if let Ok(arr) = value.cast::<PyArray1<f32>>() {
        let values = arr.readonly().as_array().to_vec();
        if values.len() != expected_len {
            return Err(PyValueError::new_err(format!(
                "{} length ({}) doesn't match atom count ({})",
                label,
                values.len(),
                expected_len
            )));
        }
        return Ok(FloatArrayData::F32(values));
    }
    if let Ok(arr) = value.cast::<PyArray1<f64>>() {
        let values = arr.readonly().as_array().to_vec();
        if values.len() != expected_len {
            return Err(PyValueError::new_err(format!(
                "{} length ({}) doesn't match atom count ({})",
                label,
                values.len(),
                expected_len
            )));
        }
        return Ok(FloatArrayData::F64(values));
    }
    Err(PyValueError::new_err(format!(
        "{} must be a float32 or float64 ndarray with shape ({},)",
        label, expected_len
    )))
}

pub(crate) fn parse_mat3_field(value: &Bound<'_, PyAny>, label: &str) -> PyResult<Mat3Data> {
    if let Ok(arr) = value.cast::<PyArray2<f32>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        if view.shape() != [3, 3] {
            return Err(PyValueError::new_err(format!(
                "{} must have shape (3, 3)",
                label
            )));
        }
        return Ok(Mat3Data::F32([
            [view[[0, 0]], view[[0, 1]], view[[0, 2]]],
            [view[[1, 0]], view[[1, 1]], view[[1, 2]]],
            [view[[2, 0]], view[[2, 1]], view[[2, 2]]],
        ]));
    }
    if let Ok(arr) = value.cast::<PyArray2<f64>>() {
        let readonly = arr.readonly();
        let view = readonly.as_array();
        if view.shape() != [3, 3] {
            return Err(PyValueError::new_err(format!(
                "{} must have shape (3, 3)",
                label
            )));
        }
        return Ok(Mat3Data::F64([
            [view[[0, 0]], view[[0, 1]], view[[0, 2]]],
            [view[[1, 0]], view[[1, 1]], view[[1, 2]]],
            [view[[2, 0]], view[[2, 1]], view[[2, 2]]],
        ]));
    }
    Err(PyValueError::new_err(format!(
        "{} must be a float32 or float64 ndarray with shape (3, 3)",
        label
    )))
}

pub(crate) fn molecule_from_numpy_arrays(
    positions: &Bound<'_, PyAny>,
    atomic_numbers: &Bound<'_, PyArray1<u8>>,
) -> PyResult<Molecule> {
    let z = atomic_numbers.readonly();
    let z_arr = z.as_array();
    let atomic_numbers_vec = z_arr.to_vec();
    let positions = parse_positions_field(positions)?;
    molecule_from_positions(positions, atomic_numbers_vec).map_err(PyValueError::new_err)
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

fn owned_vec3_array<'py>(py: Python<'py>, values: &Vec3Data) -> PyResult<Py<PyAny>> {
    Ok(match values {
        Vec3Data::F32(values) => {
            let n_atoms = values.len();
            let flat: Vec<f32> = values
                .iter()
                .flat_map(|value| [value[0], value[1], value[2]])
                .collect();
            pyarray2_from_flat(py, flat, n_atoms, 3)?
                .into_any()
                .unbind()
        }
        Vec3Data::F64(values) => {
            let n_atoms = values.len();
            let flat: Vec<f64> = values
                .iter()
                .flat_map(|value| [value[0], value[1], value[2]])
                .collect();
            pyarray2_from_flat(py, flat, n_atoms, 3)?
                .into_any()
                .unbind()
        }
    })
}

fn owned_float_array<'py>(py: Python<'py>, values: &FloatArrayData) -> Py<PyAny> {
    match values {
        FloatArrayData::F32(values) => PyArray1::from_slice(py, values).into_any().unbind(),
        FloatArrayData::F64(values) => PyArray1::from_slice(py, values).into_any().unbind(),
    }
}

fn owned_mat3x3_array<'py>(py: Python<'py>, values: &Mat3Data) -> PyResult<Py<PyAny>> {
    Ok(match values {
        Mat3Data::F32(values) => {
            let flat: Vec<f32> = values
                .iter()
                .flat_map(|row| [row[0], row[1], row[2]])
                .collect();
            pyarray2_from_flat(py, flat, 3, 3)?.into_any().unbind()
        }
        Mat3Data::F64(values) => {
            let flat: Vec<f64> = values
                .iter()
                .flat_map(|row| [row[0], row[1], row[2]])
                .collect();
            pyarray2_from_flat(py, flat, 3, 3)?.into_any().unbind()
        }
    })
}

impl PyMolecule {
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

    pub(crate) fn as_owned(&self) -> Option<&Molecule> {
        match &self.backing {
            MoleculeBacking::Owned(inner) => Some(inner),
            MoleculeBacking::View(_) => None,
        }
    }

    pub(crate) fn as_view(&self) -> Option<&SoaMoleculeView> {
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

    pub(super) fn positions_py<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        if let Some(inner) = self.as_owned() {
            return owned_vec3_array(py, &inner.positions);
        }
        let view = self.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        match view.positions_type {
            TYPE_VEC3_F32 => {
                let positions = cast_or_decode_f32(view.positions_bytes())?;
                Ok(pyarray2_from_cow(py, positions, view.n_atoms, 3)?
                    .into_any()
                    .unbind())
            }
            TYPE_VEC3_F64 => {
                let positions = cast_or_decode_f64(view.positions_bytes())?;
                Ok(pyarray2_from_cow(py, positions, view.n_atoms, 3)?
                    .into_any()
                    .unbind())
            }
            other => Err(PyValueError::new_err(format!(
                "Unsupported positions type tag {}",
                other
            ))),
        }
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

    pub(super) fn forces_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
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
        match slot.2 {
            TYPE_VEC3_F32 => {
                if slot.1 != view.n_atoms * 12 {
                    return Err(PyValueError::new_err("Invalid forces section"));
                }
                let data = cast_or_decode_f32(view.builtin_payload(slot))?;
                Ok(Some(
                    pyarray2_from_cow(py, data, view.n_atoms, 3)?
                        .into_any()
                        .unbind(),
                ))
            }
            TYPE_VEC3_F64 => {
                if slot.1 != view.n_atoms * 24 {
                    return Err(PyValueError::new_err("Invalid forces section"));
                }
                let data = cast_or_decode_f64(view.builtin_payload(slot))?;
                Ok(Some(
                    pyarray2_from_cow(py, data, view.n_atoms, 3)?
                        .into_any()
                        .unbind(),
                ))
            }
            _ => Err(PyValueError::new_err("Invalid forces section")),
        }
    }

    pub(super) fn charges_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        if let Some(inner) = self.as_owned() {
            return Ok(inner
                .charges
                .as_ref()
                .map(|charges| owned_float_array(py, charges)));
        }
        let view = self.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        let Some(slot) = view.charges else {
            return Ok(None);
        };
        match slot.2 {
            TYPE_F32_ARRAY => {
                if slot.1 != view.n_atoms * 4 {
                    return Err(PyValueError::new_err("Invalid charges section"));
                }
                let data = cast_or_decode_f32(view.builtin_payload(slot))?;
                Ok(Some(pyarray1_from_cow(py, data).into_any().unbind()))
            }
            TYPE_F64_ARRAY => {
                if slot.1 != view.n_atoms * 8 {
                    return Err(PyValueError::new_err("Invalid charges section"));
                }
                let data = cast_or_decode_f64(view.builtin_payload(slot))?;
                Ok(Some(pyarray1_from_cow(py, data).into_any().unbind()))
            }
            _ => Err(PyValueError::new_err("Invalid charges section")),
        }
    }

    pub(super) fn velocities_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
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
        match slot.2 {
            TYPE_VEC3_F32 => {
                if slot.1 != view.n_atoms * 12 {
                    return Err(PyValueError::new_err("Invalid velocities section"));
                }
                let data = cast_or_decode_f32(view.builtin_payload(slot))?;
                Ok(Some(
                    pyarray2_from_cow(py, data, view.n_atoms, 3)?
                        .into_any()
                        .unbind(),
                ))
            }
            TYPE_VEC3_F64 => {
                if slot.1 != view.n_atoms * 24 {
                    return Err(PyValueError::new_err("Invalid velocities section"));
                }
                let data = cast_or_decode_f64(view.builtin_payload(slot))?;
                Ok(Some(
                    pyarray2_from_cow(py, data, view.n_atoms, 3)?
                        .into_any()
                        .unbind(),
                ))
            }
            _ => Err(PyValueError::new_err("Invalid velocities section")),
        }
    }

    pub(super) fn cell_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
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
        match slot.2 {
            TYPE_MAT3X3_F32 => {
                if slot.1 != 36 {
                    return Err(PyValueError::new_err("Invalid cell section"));
                }
                let data = cast_or_decode_f32(view.builtin_payload(slot))?;
                Ok(Some(pyarray2_from_cow(py, data, 3, 3)?.into_any().unbind()))
            }
            TYPE_MAT3X3_F64 => {
                if slot.1 != 72 {
                    return Err(PyValueError::new_err("Invalid cell section"));
                }
                let data = cast_or_decode_f64(view.builtin_payload(slot))?;
                Ok(Some(pyarray2_from_cow(py, data, 3, 3)?.into_any().unbind()))
            }
            _ => Err(PyValueError::new_err("Invalid cell section")),
        }
    }

    pub(super) fn stress_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
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
        match slot.2 {
            TYPE_MAT3X3_F32 => {
                if slot.1 != 36 {
                    return Err(PyValueError::new_err("Invalid stress section"));
                }
                let data = cast_or_decode_f32(view.builtin_payload(slot))?;
                Ok(Some(pyarray2_from_cow(py, data, 3, 3)?.into_any().unbind()))
            }
            TYPE_MAT3X3_F64 => {
                if slot.1 != 72 {
                    return Err(PyValueError::new_err("Invalid stress section"));
                }
                let data = cast_or_decode_f64(view.builtin_payload(slot))?;
                Ok(Some(pyarray2_from_cow(py, data, 3, 3)?.into_any().unbind()))
            }
            _ => Err(PyValueError::new_err("Invalid stress section")),
        }
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
        positions: &Bound<'_, PyAny>,
        atomic_numbers: &Bound<'_, PyArray1<u8>>,
        energy: Option<f64>,
        forces: Option<Py<PyAny>>,
        charges: Option<Py<PyAny>>,
        velocities: Option<Py<PyAny>>,
        cell: Option<Py<PyAny>>,
        stress: Option<Py<PyAny>>,
        pbc: Option<(bool, bool, bool)>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let mut molecule = molecule_from_numpy_arrays(positions, atomic_numbers)?;

        if let Some(name) = name {
            molecule.name = Some(name);
        }
        if let Some(energy) = energy {
            molecule.energy = Some(FloatScalarData::F64(energy));
        }
        if let Some((a, b, c)) = pbc {
            molecule.pbc = Some([a, b, c]);
        }

        let n_atoms = molecule.len();

        if let Some(forces) = forces {
            molecule.forces = Some(parse_vec3_field(forces.bind(py), "forces", n_atoms)?);
        }

        if let Some(charges) = charges {
            molecule.charges = Some(parse_float_array_field(
                charges.bind(py),
                "charges",
                n_atoms,
            )?);
        }

        if let Some(velocities) = velocities {
            molecule.velocities = Some(parse_vec3_field(
                velocities.bind(py),
                "velocities",
                n_atoms,
            )?);
        }

        if let Some(cell) = cell {
            molecule.cell = Some(parse_mat3_field(cell.bind(py), "cell")?);
        }

        if let Some(stress) = stress {
            molecule.stress = Some(parse_mat3_field(stress.bind(py), "stress")?);
        }

        Ok(Self::from_owned(molecule))
    }
}
