use super::*;

/// Python wrapper for Atom
///
/// ## Rust + Python Integration:
/// - `#[pyclass]`: Makes this struct available in Python
/// - PyO3 automatically handles conversion between Rust and Python
#[pyclass]
#[derive(Clone)]
pub(crate) struct PyAtom {
    inner: Atom,
}

#[pymethods]
impl PyAtom {
    /// Create a new atom
    #[new]
    fn new(x: f32, y: f32, z: f32, atomic_number: u8) -> Self {
        Self {
            inner: Atom::new(x, y, z, atomic_number),
        }
    }

    /// Get the position as a tuple (x, y, z)
    fn position(&self) -> (f32, f32, f32) {
        (self.inner.x, self.inner.y, self.inner.z)
    }

    /// Get the atomic number
    #[getter]
    fn atomic_number(&self) -> u8 {
        self.inner.atomic_number
    }

    /// Calculate distance to another atom
    fn distance_to(&self, other: &PyAtom) -> f32 {
        self.inner.distance_to(&other.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Atom(x={}, y={}, z={}, atomic_number={})",
            self.inner.x, self.inner.y, self.inner.z, self.inner.atomic_number
        )
    }
}

/// Python wrapper for Molecule
#[pyclass]
pub(crate) struct PyMolecule {
    backing: MoleculeBacking,
}

enum MoleculeBacking {
    Owned(Molecule),
    View(SoaMoleculeView),
}
#[path = "molecule_helpers.rs"]
mod helpers;

pub(crate) use self::helpers::{
    SoaBuiltinPayloads, SoaCustomSection, SoaTypedPayload, build_soa_record_unchecked,
    cast_or_decode_f32, cast_or_decode_f64, cast_or_decode_i32, cast_or_decode_i64,
    pyarray1_from_cow, pyarray2_from_cow,
};
use self::helpers::{into_py_any, property_section_to_pyobject, property_value_to_pyobject};

#[derive(Clone, Copy, PartialEq, Eq)]
enum CustomPropertyScope {
    Molecule,
    Atom,
}

fn parse_custom_property_scope(scope: Option<&str>) -> PyResult<Option<CustomPropertyScope>> {
    match scope {
        None => Ok(None),
        Some("molecule") => Ok(Some(CustomPropertyScope::Molecule)),
        Some("atom") => Ok(Some(CustomPropertyScope::Atom)),
        Some(other) => Err(PyValueError::new_err(format!(
            "scope must be 'molecule' or 'atom', got '{}'",
            other
        ))),
    }
}

fn atom_property_shape_error(key: &str, actual: usize, n_atoms: usize) -> PyErr {
    PyValueError::new_err(format!(
        "Atom property '{}' first dimension ({}) doesn't match atom count ({})",
        key, actual, n_atoms
    ))
}

fn validate_atom_property_value(key: &str, value: &PropertyValue, n_atoms: usize) -> PyResult<()> {
    match value {
        PropertyValue::FloatArray(values) if values.len() == n_atoms => Ok(()),
        PropertyValue::Vec3Array(values) if values.len() == n_atoms => Ok(()),
        PropertyValue::IntArray(values) if values.len() == n_atoms => Ok(()),
        PropertyValue::Float32Array(values) if values.len() == n_atoms => Ok(()),
        PropertyValue::Vec3ArrayF64(values) if values.len() == n_atoms => Ok(()),
        PropertyValue::Int32Array(values) if values.len() == n_atoms => Ok(()),
        PropertyValue::Tensor(values) => match values.shape().first() {
            Some(first_dim) if *first_dim == n_atoms => Ok(()),
            Some(first_dim) => Err(atom_property_shape_error(key, *first_dim, n_atoms)),
            None => Err(PyValueError::new_err(format!(
                "Atom tensor property '{}' must have at least one dimension",
                key
            ))),
        },
        PropertyValue::FloatArray(values) => {
            Err(atom_property_shape_error(key, values.len(), n_atoms))
        }
        PropertyValue::Vec3Array(values) => {
            Err(atom_property_shape_error(key, values.len(), n_atoms))
        }
        PropertyValue::IntArray(values) => {
            Err(atom_property_shape_error(key, values.len(), n_atoms))
        }
        PropertyValue::Float32Array(values) => {
            Err(atom_property_shape_error(key, values.len(), n_atoms))
        }
        PropertyValue::Vec3ArrayF64(values) => {
            Err(atom_property_shape_error(key, values.len(), n_atoms))
        }
        PropertyValue::Int32Array(values) => {
            Err(atom_property_shape_error(key, values.len(), n_atoms))
        }
        PropertyValue::None
        | PropertyValue::Float(_)
        | PropertyValue::Int(_)
        | PropertyValue::String(_) => Err(PyValueError::new_err(format!(
            "Atom property '{}' must be a numeric ndarray with first dimension equal to atom count ({})",
            key, n_atoms
        ))),
    }
}

#[pymethods]
impl PyMolecule {
    /// Create a new molecule from numpy arrays.
    #[new]
    #[pyo3(signature = (
        positions,
        atomic_numbers,
        *,
        energy=None,
        forces=None,
        charges=None,
        velocities=None,
        cell=None,
        stress=None,
        pbc=None,
        name=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
        Self::from_arrays_impl(
            py,
            positions,
            atomic_numbers,
            energy,
            forces,
            charges,
            velocities,
            cell,
            stress,
            pbc,
            name,
        )
    }

    /// Create a molecule from numpy arrays.
    #[staticmethod]
    #[pyo3(signature = (
        positions,
        atomic_numbers,
        *,
        energy=None,
        forces=None,
        charges=None,
        velocities=None,
        cell=None,
        stress=None,
        pbc=None,
        name=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_arrays(
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
        Self::from_arrays_impl(
            py,
            positions,
            atomic_numbers,
            energy,
            forces,
            charges,
            velocities,
            cell,
            stress,
            pbc,
            name,
        )
    }

    /// Get the number of atoms
    fn __len__(&self) -> usize {
        self.len()
    }

    /// Get atoms as a list
    fn atoms(&self) -> PyResult<Vec<PyAtom>> {
        let molecule = self.clone_as_owned()?;
        Ok(molecule
            .to_atoms()
            .into_iter()
            .map(|inner| PyAtom { inner })
            .collect())
    }

    /// Materialize the molecule into an owned, self-contained object.
    fn to_owned(&self) -> PyResult<Self> {
        Ok(Self::from_owned(self.clone_as_owned()?))
    }

    fn __reduce__<'py>(slf: Bound<'py, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let payload = {
            let molecule = slf.borrow().clone_as_owned()?;
            bincode::serialize(&molecule).map_err(|e| {
                PyValueError::new_err(format!("Failed to serialize molecule: {}", e))
            })?
        };
        let constructor = py
            .import("atompack._atompack_rs")?
            .getattr("_molecule_from_pickle_bytes")?;
        let args = PyTuple::new(py, [PyBytes::new(py, &payload).into_any().unbind()])?;
        Ok(
            PyTuple::new(py, [constructor.unbind(), args.into_any().unbind()])?
                .into_any()
                .unbind(),
        )
    }

    #[pyo3(signature = (*, copy_info=true, copy_arrays=true))]
    fn _ase_builtin_tuple_fast<'py>(
        &self,
        py: Python<'py>,
        copy_info: bool,
        copy_arrays: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let numbers = self.atomic_numbers_py(py)?.into_any().unbind();
        let positions = self.positions_py(py)?;
        let cell = match self.cell_py(py)? {
            Some(value) => value,
            None => py.None(),
        };
        let pbc = match self.pbc()? {
            Some(value) => into_py_any(py, value)?,
            None => py.None(),
        };
        let velocities = match self.velocities_py(py)? {
            Some(value) => value,
            None => py.None(),
        };
        let energy = match self.energy()? {
            Some(value) => into_py_any(py, value)?,
            None => py.None(),
        };
        let forces = match self.forces_py(py)? {
            Some(value) => value,
            None => py.None(),
        };
        let stress = match self.stress_py(py)? {
            Some(value) => value,
            None => py.None(),
        };
        let charges = match self.charges_py(py)? {
            Some(value) => value,
            None => py.None(),
        };

        let (arrays_obj, info_obj) = if copy_arrays || copy_info {
            let has_custom = if let Some(inner) = self.as_owned() {
                !inner.atom_properties.is_empty() || !inner.properties.is_empty()
            } else {
                self.as_view()
                    .is_some_and(|view| !view.custom_sections.is_empty())
            };

            if has_custom {
                let arrays = PyDict::new(py);
                let info = PyDict::new(py);
                if self.as_owned().is_some() {
                    self.append_owned_ase_properties(py, &arrays, &info, copy_info, copy_arrays)?;
                } else {
                    self.append_view_ase_properties(py, &arrays, &info, copy_info, copy_arrays)?;
                }
                let arrays_obj = if arrays.len() > 0 {
                    arrays.into_any().unbind()
                } else {
                    py.None()
                };
                let info_obj = if info.len() > 0 {
                    info.into_any().unbind()
                } else {
                    py.None()
                };
                (arrays_obj, info_obj)
            } else {
                (py.None(), py.None())
            }
        } else {
            (py.None(), py.None())
        };

        PyTuple::new(
            py,
            [
                numbers, positions, cell, pbc, velocities, energy, forces, stress, charges,
                arrays_obj, info_obj,
            ],
        )
    }

    #[pyo3(signature = (*, copy_info=true, copy_arrays=true))]
    fn _ase_payload<'py>(
        &self,
        py: Python<'py>,
        copy_info: bool,
        copy_arrays: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        self._ase_builtin_tuple_fast(py, copy_info, copy_arrays)
    }

    /// forces property (getter)
    #[getter]
    fn forces<'py>(slf: Bound<'py, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        slf.borrow().forces_py(py)
    }

    /// forces property (setter)
    #[setter]
    fn set_forces(&mut self, py: Python<'_>, forces: Py<PyAny>) -> PyResult<()> {
        let n_atoms = self.len();
        self.ensure_owned()?.forces = Some(parse_vec3_field(forces.bind(py), "Forces", n_atoms)?);
        Ok(())
    }

    /// Get energy
    #[getter]
    fn energy(&self) -> PyResult<Option<f64>> {
        if let Some(inner) = self.as_owned() {
            Ok(inner.energy.as_ref().map(FloatScalarData::as_f64))
        } else if let Some(view) = self.as_view() {
            view.energy()
        } else {
            Ok(None)
        }
    }

    /// Set energy
    #[setter]
    fn set_energy(&mut self, energy: Option<f64>) -> PyResult<()> {
        self.ensure_owned()?.energy = energy.map(FloatScalarData::F64);
        Ok(())
    }

    /// charges property (getter)
    #[getter]
    fn charges<'py>(slf: Bound<'py, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        slf.borrow().charges_py(py)
    }

    /// charges property (setter)
    #[setter]
    fn set_charges(&mut self, py: Python<'_>, charges: Py<PyAny>) -> PyResult<()> {
        let n_atoms = self.len();
        self.ensure_owned()?.charges = Some(parse_float_array_field(
            charges.bind(py),
            "Charges",
            n_atoms,
        )?);
        Ok(())
    }

    /// velocities property (getter)
    #[getter]
    fn velocities<'py>(slf: Bound<'py, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        slf.borrow().velocities_py(py)
    }

    /// velocities property (setter)
    #[setter]
    fn set_velocities(&mut self, py: Python<'_>, velocities: Py<PyAny>) -> PyResult<()> {
        let n_atoms = self.len();
        self.ensure_owned()?.velocities = Some(parse_vec3_field(
            velocities.bind(py),
            "Velocities",
            n_atoms,
        )?);
        Ok(())
    }

    /// cell property (getter)
    #[getter]
    fn cell<'py>(slf: Bound<'py, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        slf.borrow().cell_py(py)
    }

    /// cell property (setter)
    #[setter]
    fn set_cell(&mut self, py: Python<'_>, cell: Py<PyAny>) -> PyResult<()> {
        self.ensure_owned()?.cell = Some(parse_mat3_field(cell.bind(py), "Cell")?);
        Ok(())
    }

    /// stress property (getter)
    #[getter]
    fn stress<'py>(slf: Bound<'py, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        slf.borrow().stress_py(py)
    }

    /// stress property (setter)
    #[setter]
    fn set_stress(&mut self, py: Python<'_>, stress: Py<PyAny>) -> PyResult<()> {
        let inner = self.ensure_owned()?;
        inner.stress = Some(parse_mat3_field(stress.bind(py), "Stress")?);
        inner.properties.remove("stress");
        Ok(())
    }

    /// pbc property (getter)
    #[getter]
    fn pbc(&self) -> PyResult<Option<(bool, bool, bool)>> {
        if let Some(inner) = self.as_owned() {
            Ok(inner.pbc.map(|p| (p[0], p[1], p[2])))
        } else if let Some(view) = self.as_view() {
            view.pbc()
        } else {
            Ok(None)
        }
    }

    /// pbc property (setter)
    #[setter]
    fn set_pbc(&mut self, pbc: Option<(bool, bool, bool)>) -> PyResult<()> {
        self.ensure_owned()?.pbc = pbc.map(|(a, b, c)| [a, b, c]);
        Ok(())
    }

    /// positions property (read-only)
    #[getter]
    fn positions<'py>(slf: Bound<'py, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        slf.borrow().positions_py(py)
    }

    /// atomic_numbers property (read-only)
    #[getter]
    fn atomic_numbers<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let py = slf.py();
        let molecule = slf.borrow();
        if let Some(inner) = molecule.as_owned() {
            return Ok(PyArray1::from_slice(py, &inner.atomic_numbers));
        }
        let view = molecule.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        Ok(PyArray1::from_slice(py, view.atomic_numbers_bytes()))
    }

    // --- Custom Properties ---

    /// Get a custom property by key
    fn get_property<'py>(slf: Bound<'py, Self>, key: &str) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let molecule = slf.borrow();
        if let Some(inner) = molecule.as_owned() {
            let molecule_value = inner.properties.get(key);
            let atom_value = inner.atom_properties.get(key);
            return match (molecule_value, atom_value) {
                (Some(_), Some(_)) => Err(PyValueError::new_err(format!(
                    "Property '{}' exists in both molecule and atom scopes",
                    key
                ))),
                (Some(v), None) | (None, Some(v)) => property_value_to_pyobject(py, v),
                (None, None) => Err(PyKeyError::new_err(format!("Property '{}' not found", key))),
            };
        }
        let view = molecule.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        let molecule_section = view.find_custom_section(KIND_MOL_PROP, key)?;
        let atom_section = view.find_custom_section(KIND_ATOM_PROP, key)?;
        match (molecule_section, atom_section) {
            (Some(_), Some(_)) => Err(PyValueError::new_err(format!(
                "Property '{}' exists in both molecule and atom scopes",
                key
            ))),
            (Some(section), None) | (None, Some(section)) => {
                property_section_to_pyobject(py, view, section)
            }
            (None, None) => Err(PyKeyError::new_err(format!("Property '{}' not found", key))),
        }
    }

    /// Set a custom property
    #[pyo3(signature = (key, value, *, scope=None))]
    fn set_property(
        &mut self,
        py: Python<'_>,
        key: String,
        value: Py<PyAny>,
        scope: Option<&str>,
    ) -> PyResult<()> {
        let requested_scope = parse_custom_property_scope(scope)?;
        let n_atoms = self.len();
        if key == "stress" {
            return Err(PyValueError::new_err(
                "'stress' is a reserved field; use molecule.stress instead",
            ));
        }
        let parsed = parse_property_value(value.bind(py))?;
        let inner = self.ensure_owned()?;
        let has_molecule = inner.properties.contains_key(&key);
        let has_atom = inner.atom_properties.contains_key(&key);
        if has_molecule && has_atom {
            return Err(PyValueError::new_err(format!(
                "Property '{}' exists in both molecule and atom scopes",
                key
            )));
        }
        let target_scope = match (has_molecule, has_atom, requested_scope) {
            (true, false, None | Some(CustomPropertyScope::Molecule)) => {
                CustomPropertyScope::Molecule
            }
            (true, false, Some(CustomPropertyScope::Atom)) => {
                return Err(PyValueError::new_err(format!(
                    "Property '{}' already exists as a molecule property; delete it before setting atom scope",
                    key
                )));
            }
            (false, true, None | Some(CustomPropertyScope::Atom)) => CustomPropertyScope::Atom,
            (false, true, Some(CustomPropertyScope::Molecule)) => {
                return Err(PyValueError::new_err(format!(
                    "Property '{}' already exists as an atom property; delete it before setting molecule scope",
                    key
                )));
            }
            (false, false, Some(CustomPropertyScope::Atom)) => CustomPropertyScope::Atom,
            (false, false, None | Some(CustomPropertyScope::Molecule)) => {
                CustomPropertyScope::Molecule
            }
            (true, true, _) => unreachable!(),
        };
        match target_scope {
            CustomPropertyScope::Molecule => {
                inner.properties.insert(key, parsed);
            }
            CustomPropertyScope::Atom => {
                validate_atom_property_value(&key, &parsed, n_atoms)?;
                inner.atom_properties.insert(key, parsed);
            }
        }
        Ok(())
    }

    /// Get all property keys
    #[pyo3(signature = (*, scope=None))]
    fn property_keys(&self, scope: Option<&str>) -> PyResult<Vec<String>> {
        let requested_scope = parse_custom_property_scope(scope)?;
        if let Some(inner) = self.as_owned() {
            let mut keys = Vec::new();
            if requested_scope != Some(CustomPropertyScope::Atom) {
                keys.extend(inner.properties.keys().cloned());
            }
            if requested_scope != Some(CustomPropertyScope::Molecule) {
                keys.extend(inner.atom_properties.keys().cloned());
            }
            keys.sort();
            Ok(keys)
        } else if let Some(view) = self.as_view() {
            let mut keys = Vec::new();
            for section in &view.custom_sections {
                let include = match requested_scope {
                    Some(CustomPropertyScope::Molecule) => section.kind == KIND_MOL_PROP,
                    Some(CustomPropertyScope::Atom) => section.kind == KIND_ATOM_PROP,
                    None => matches!(section.kind, KIND_MOL_PROP | KIND_ATOM_PROP),
                };
                if include {
                    keys.push(view.lazy_section_key(section)?.to_string());
                }
            }
            keys.sort();
            Ok(keys)
        } else {
            Ok(Vec::new())
        }
    }

    /// Check if a property exists
    #[pyo3(signature = (key, *, scope=None))]
    fn has_property(&self, key: &str, scope: Option<&str>) -> PyResult<bool> {
        let requested_scope = parse_custom_property_scope(scope)?;
        if let Some(inner) = self.as_owned() {
            Ok(match requested_scope {
                Some(CustomPropertyScope::Molecule) => inner.properties.contains_key(key),
                Some(CustomPropertyScope::Atom) => inner.atom_properties.contains_key(key),
                None => {
                    inner.properties.contains_key(key) || inner.atom_properties.contains_key(key)
                }
            })
        } else if let Some(view) = self.as_view() {
            Ok(match requested_scope {
                Some(CustomPropertyScope::Molecule) => {
                    view.find_custom_section(KIND_MOL_PROP, key)?.is_some()
                }
                Some(CustomPropertyScope::Atom) => {
                    view.find_custom_section(KIND_ATOM_PROP, key)?.is_some()
                }
                None => {
                    view.find_custom_section(KIND_MOL_PROP, key)?.is_some()
                        || view.find_custom_section(KIND_ATOM_PROP, key)?.is_some()
                }
            })
        } else {
            Ok(false)
        }
    }

    /// Delete a custom property by key.
    fn delete_property(&mut self, key: &str) -> PyResult<()> {
        let inner = self.ensure_owned()?;
        let removed_molecule = inner.properties.remove(key).is_some();
        let removed_atom = inner.atom_properties.remove(key).is_some();
        if removed_molecule || removed_atom {
            Ok(())
        } else {
            Err(PyKeyError::new_err(format!("Property '{}' not found", key)))
        }
    }

    /// Index molecule atoms by integer, or custom properties by string.
    fn __getitem__<'py>(slf: Bound<'py, Self>, index: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let molecule = slf.borrow();

        if let Ok(key) = index.extract::<String>() {
            if let Some(inner) = molecule.as_owned() {
                let molecule_value = inner.properties.get(&key);
                let atom_value = inner.atom_properties.get(&key);
                return match (molecule_value, atom_value) {
                    (Some(_), Some(_)) => Err(PyValueError::new_err(format!(
                        "Property '{}' exists in both molecule and atom scopes",
                        key
                    ))),
                    (Some(v), None) | (None, Some(v)) => property_value_to_pyobject(py, v),
                    (None, None) => {
                        Err(PyKeyError::new_err(format!("Property '{}' not found", key)))
                    }
                };
            }
            let view = molecule.as_view().ok_or_else(|| {
                PyValueError::new_err("Molecule is missing both owned and view state")
            })?;
            let molecule_section = view.find_custom_section(KIND_MOL_PROP, &key)?;
            let atom_section = view.find_custom_section(KIND_ATOM_PROP, &key)?;
            return match (molecule_section, atom_section) {
                (Some(_), Some(_)) => Err(PyValueError::new_err(format!(
                    "Property '{}' exists in both molecule and atom scopes",
                    key
                ))),
                (Some(section), None) | (None, Some(section)) => {
                    property_section_to_pyobject(py, view, section)
                }
                (None, None) => Err(PyKeyError::new_err(format!("Property '{}' not found", key))),
            };
        }

        if let Ok(i) = index.extract::<isize>() {
            let n = molecule.len() as isize;
            let normalized = if i < 0 { i + n } else { i };
            if normalized < 0 || normalized >= n {
                return Err(PyIndexError::new_err(format!(
                    "Atom index {} out of bounds for molecule of length {}",
                    i, n
                )));
            }
            let atom = if let Some(inner) = molecule.as_owned() {
                inner.atom(normalized as usize).ok_or_else(|| {
                    PyValueError::new_err("Failed to read atom from owned molecule")
                })?
            } else {
                molecule
                    .as_view()
                    .ok_or_else(|| {
                        PyValueError::new_err("Molecule is missing both owned and view state")
                    })?
                    .atom_at(normalized as usize)?
                    .ok_or_else(|| PyValueError::new_err("Failed to read atom from view"))?
            };
            return Ok(Bound::new(py, PyAtom { inner: atom })?.into_any().unbind());
        }

        Err(PyTypeError::new_err(
            "Molecule indices must be integers or strings",
        ))
    }

    /// String representation
    fn __repr__(&self) -> String {
        let has_forces = if let Some(inner) = self.as_owned() {
            inner.forces.is_some()
        } else {
            self.as_view().is_some_and(|view| view.forces.is_some())
        };
        let has_energy = if let Some(inner) = self.as_owned() {
            inner.energy.is_some()
        } else {
            self.as_view().is_some_and(|view| view.energy.is_some())
        };
        format!(
            "Molecule(n_atoms={}, has_forces={}, has_energy={})",
            self.len(),
            has_forces,
            has_energy
        )
    }
}
