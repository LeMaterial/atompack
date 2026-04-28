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
    SoaBuiltinPayloads, SoaCustomSection, build_soa_record_with_custom, cast_or_decode_f32,
    cast_or_decode_f64, cast_or_decode_i32, cast_or_decode_i64, pyarray1_from_cow,
    pyarray2_from_cow,
};
use self::helpers::{
    into_py_any, property_section_to_pyobject, property_value_to_pyobject, pyarray2_from_flat,
};

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

    /// Create a molecule from numpy arrays (fast path).
    ///
    /// Parameters:
    /// - positions: float32 array of shape (n_atoms, 3)
    /// - atomic_numbers: uint8 array of shape (n_atoms,)
    /// - builtins: optional keyword arguments such as energy, forces, charges,
    ///   velocities, cell, stress, pbc, and name
    ///
    /// Builds an SOA view directly from the numpy buffers — no intermediate
    /// Atom structs are created. If you later mutate the molecule (e.g. set
    /// energy), it will be materialized on demand.
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
        let positions = self.positions_py(py)?.into_any().unbind();
        let cell = match self.cell_py(py)? {
            Some(value) => value.into_any().unbind(),
            None => py.None(),
        };
        let pbc = match self.pbc()? {
            Some(value) => into_py_any(py, value)?,
            None => py.None(),
        };
        let velocities = match self.velocities_py(py)? {
            Some(value) => value.into_any().unbind(),
            None => py.None(),
        };
        let energy = match self.energy()? {
            Some(value) => into_py_any(py, value)?,
            None => py.None(),
        };
        let forces = match self.forces_py(py)? {
            Some(value) => value.into_any().unbind(),
            None => py.None(),
        };
        let stress = match self.stress_py(py)? {
            Some(value) => value.into_any().unbind(),
            None => py.None(),
        };
        let charges = match self.charges_py(py)? {
            Some(value) => value.into_any().unbind(),
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
    fn forces<'py>(slf: Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        let py = slf.py();
        let molecule = slf.borrow();
        if let Some(inner) = molecule.as_owned() {
            return inner
                .forces
                .as_ref()
                .map(|forces| {
                    let n_atoms = forces.len();
                    let flat: Vec<f32> = forces.iter().flat_map(|f| [f[0], f[1], f[2]]).collect();
                    pyarray2_from_flat(py, flat, n_atoms, 3)
                })
                .transpose();
        }
        let Some(view) = molecule.as_view() else {
            return Ok(None);
        };
        let Some(slot) = view.forces else {
            return Ok(None);
        };
        if slot.2 != TYPE_VEC3_F32 || slot.1 != view.n_atoms * 12 {
            return Err(PyValueError::new_err("Invalid forces section"));
        }
        let payload = view.builtin_payload(slot);
        let data = cast_or_decode_f32(payload)?;
        Ok(Some(pyarray2_from_cow(py, data, view.n_atoms, 3)?))
    }

    /// forces property (setter)
    #[setter]
    fn set_forces(&mut self, forces: &Bound<'_, PyArray2<f32>>) -> PyResult<()> {
        let readonly = forces.readonly();
        let arr = readonly.as_array();
        let shape = arr.shape();

        if shape[1] != 3 {
            return Err(PyValueError::new_err("Forces must have shape (n_atoms, 3)"));
        }

        let forces_vec: Vec<[f32; 3]> = arr
            .outer_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect();

        if forces_vec.len() != self.len() {
            return Err(PyValueError::new_err(format!(
                "Forces length ({}) doesn't match atom count ({})",
                forces_vec.len(),
                self.len()
            )));
        }

        self.ensure_owned()?.forces = Some(forces_vec);
        Ok(())
    }

    /// Get energy
    #[getter]
    fn energy(&self) -> PyResult<Option<f64>> {
        if let Some(inner) = self.as_owned() {
            Ok(inner.energy)
        } else if let Some(view) = self.as_view() {
            view.energy()
        } else {
            Ok(None)
        }
    }

    /// Set energy
    #[setter]
    fn set_energy(&mut self, energy: Option<f64>) -> PyResult<()> {
        self.ensure_owned()?.energy = energy;
        Ok(())
    }

    /// charges property (getter)
    #[getter]
    fn charges<'py>(slf: Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let py = slf.py();
        let molecule = slf.borrow();
        if let Some(inner) = molecule.as_owned() {
            return Ok(inner
                .charges
                .as_ref()
                .map(|charges| PyArray1::from_slice(py, charges)));
        }
        let Some(view) = molecule.as_view() else {
            return Ok(None);
        };
        let Some(slot) = view.charges else {
            return Ok(None);
        };
        if slot.2 != TYPE_F64_ARRAY || slot.1 != view.n_atoms * 8 {
            return Err(PyValueError::new_err("Invalid charges section"));
        }
        let payload = view.builtin_payload(slot);
        let data = cast_or_decode_f64(payload)?;
        Ok(Some(pyarray1_from_cow(py, data)))
    }

    /// charges property (setter)
    #[setter]
    fn set_charges(&mut self, charges: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        let charges_vec: Vec<f64> = charges.readonly().as_array().to_vec();

        if charges_vec.len() != self.len() {
            return Err(PyValueError::new_err(format!(
                "Charges length ({}) doesn't match atom count ({})",
                charges_vec.len(),
                self.len()
            )));
        }

        self.ensure_owned()?.charges = Some(charges_vec);
        Ok(())
    }

    /// velocities property (getter)
    #[getter]
    fn velocities<'py>(slf: Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        let py = slf.py();
        let molecule = slf.borrow();
        if let Some(inner) = molecule.as_owned() {
            return inner
                .velocities
                .as_ref()
                .map(|vels| {
                    let n_atoms = vels.len();
                    let flat: Vec<f32> = vels.iter().flat_map(|v| [v[0], v[1], v[2]]).collect();
                    pyarray2_from_flat(py, flat, n_atoms, 3)
                })
                .transpose();
        }
        let Some(view) = molecule.as_view() else {
            return Ok(None);
        };
        let Some(slot) = view.velocities else {
            return Ok(None);
        };
        if slot.2 != TYPE_VEC3_F32 || slot.1 != view.n_atoms * 12 {
            return Err(PyValueError::new_err("Invalid velocities section"));
        }
        let payload = view.builtin_payload(slot);
        let data = cast_or_decode_f32(payload)?;
        Ok(Some(pyarray2_from_cow(py, data, view.n_atoms, 3)?))
    }

    /// velocities property (setter)
    #[setter]
    fn set_velocities(&mut self, velocities: &Bound<'_, PyArray2<f32>>) -> PyResult<()> {
        let readonly = velocities.readonly();
        let arr = readonly.as_array();
        let shape = arr.shape();

        if shape[1] != 3 {
            return Err(PyValueError::new_err(
                "Velocities must have shape (n_atoms, 3)",
            ));
        }

        let vel_vec: Vec<[f32; 3]> = arr
            .outer_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect();

        if vel_vec.len() != self.len() {
            return Err(PyValueError::new_err(format!(
                "Velocities length ({}) doesn't match atom count ({})",
                vel_vec.len(),
                self.len()
            )));
        }

        self.ensure_owned()?.velocities = Some(vel_vec);
        Ok(())
    }

    /// cell property (getter)
    #[getter]
    fn cell<'py>(slf: Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let py = slf.py();
        let molecule = slf.borrow();
        if let Some(inner) = molecule.as_owned() {
            return inner
                .cell
                .as_ref()
                .map(|cell| {
                    let flat: Vec<f64> = cell
                        .iter()
                        .flat_map(|row| [row[0], row[1], row[2]])
                        .collect();
                    pyarray2_from_flat(py, flat, 3, 3)
                })
                .transpose();
        }
        let Some(view) = molecule.as_view() else {
            return Ok(None);
        };
        let Some(slot) = view.cell else {
            return Ok(None);
        };
        if slot.2 != TYPE_MAT3X3_F64 || slot.1 != 72 {
            return Err(PyValueError::new_err("Invalid cell section"));
        }
        let payload = view.builtin_payload(slot);
        let data = cast_or_decode_f64(payload)?;
        Ok(Some(pyarray2_from_cow(py, data, 3, 3)?))
    }

    /// cell property (setter)
    #[setter]
    fn set_cell(&mut self, cell: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
        let readonly = cell.readonly();
        let arr = readonly.as_array();
        let shape = arr.shape();

        if shape != [3, 3] {
            return Err(PyValueError::new_err("Cell must have shape (3, 3)"));
        }

        let cell_array: [[f64; 3]; 3] = [
            [arr[[0, 0]], arr[[0, 1]], arr[[0, 2]]],
            [arr[[1, 0]], arr[[1, 1]], arr[[1, 2]]],
            [arr[[2, 0]], arr[[2, 1]], arr[[2, 2]]],
        ];

        self.ensure_owned()?.cell = Some(cell_array);
        Ok(())
    }

    /// stress property (getter)
    #[getter]
    fn stress<'py>(slf: Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let py = slf.py();
        let molecule = slf.borrow();
        if let Some(inner) = molecule.as_owned() {
            return inner
                .stress
                .as_ref()
                .map(|stress| {
                    let flat: Vec<f64> = stress
                        .iter()
                        .flat_map(|row| [row[0], row[1], row[2]])
                        .collect();
                    pyarray2_from_flat(py, flat, 3, 3)
                })
                .transpose();
        }
        let Some(view) = molecule.as_view() else {
            return Ok(None);
        };
        let Some(slot) = view.stress else {
            return Ok(None);
        };
        if slot.2 != TYPE_MAT3X3_F64 || slot.1 != 72 {
            return Err(PyValueError::new_err("Invalid stress section"));
        }
        let payload = view.builtin_payload(slot);
        let data = cast_or_decode_f64(payload)?;
        Ok(Some(pyarray2_from_cow(py, data, 3, 3)?))
    }

    /// stress property (setter)
    #[setter]
    fn set_stress(&mut self, py: Python<'_>, stress: Py<PyAny>) -> PyResult<()> {
        let stress = stress.bind(py);
        if let Ok(arr) = stress.cast::<PyArray2<f64>>() {
            let readonly = arr.readonly();
            let arr = readonly.as_array();
            let shape = arr.shape();

            if shape != [3, 3] {
                return Err(PyValueError::new_err("Stress must have shape (3, 3)"));
            }

            let inner = self.ensure_owned()?;
            inner.stress = Some([
                [arr[[0, 0]], arr[[0, 1]], arr[[0, 2]]],
                [arr[[1, 0]], arr[[1, 1]], arr[[1, 2]]],
                [arr[[2, 0]], arr[[2, 1]], arr[[2, 2]]],
            ]);
            inner.properties.remove("stress");
            return Ok(());
        }

        if let Ok(arr) = stress.cast::<PyArray2<f32>>() {
            let readonly = arr.readonly();
            let arr = readonly.as_array();
            let shape = arr.shape();

            if shape != [3, 3] {
                return Err(PyValueError::new_err("Stress must have shape (3, 3)"));
            }

            let inner = self.ensure_owned()?;
            inner.stress = Some([
                [arr[[0, 0]] as f64, arr[[0, 1]] as f64, arr[[0, 2]] as f64],
                [arr[[1, 0]] as f64, arr[[1, 1]] as f64, arr[[1, 2]] as f64],
                [arr[[2, 0]] as f64, arr[[2, 1]] as f64, arr[[2, 2]] as f64],
            ]);
            inner.properties.remove("stress");
            return Ok(());
        }

        Err(PyValueError::new_err(
            "Stress must be a float32 or float64 ndarray with shape (3, 3)",
        ))
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
    fn positions<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let py = slf.py();
        let molecule = slf.borrow();
        if let Some(inner) = molecule.as_owned() {
            let flat = inner.positions_flat();
            let n_atoms = inner.len();
            return pyarray2_from_flat(py, flat, n_atoms, 3);
        }
        let view = molecule.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        let pos_f32 = cast_or_decode_f32(view.positions_bytes())?;
        pyarray2_from_cow(py, pos_f32, view.n_atoms, 3)
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
            return match inner.properties.get(key) {
                Some(v) => property_value_to_pyobject(py, v),
                None => Err(PyKeyError::new_err(format!(
                    "Property '{}' not found",
                    key
                ))),
            };
        }
        let view = molecule.as_view().ok_or_else(|| {
            PyValueError::new_err("Molecule is missing both owned and view state")
        })?;
        match view.find_custom_section(KIND_MOL_PROP, key)? {
            Some(section) => property_section_to_pyobject(py, view, section),
            None => Err(PyKeyError::new_err(format!(
                "Property '{}' not found",
                key
            ))),
        }
    }

    /// Set a custom property
    fn set_property(&mut self, py: Python<'_>, key: String, value: Py<PyAny>) -> PyResult<()> {
        let inner = self.ensure_owned()?;
        let value = value.bind(py);
        if key == "stress" {
            return Err(PyValueError::new_err(
                "'stress' is a reserved field; use molecule.stress instead",
            ));
        }

        // Try to extract different Python types
        if let Ok(v) = value.extract::<i64>() {
            inner.properties.insert(key, PropertyValue::Int(v));
        } else if let Ok(v) = value.extract::<f64>() {
            inner.properties.insert(key, PropertyValue::Float(v));
        } else if let Ok(v) = value.extract::<String>() {
            inner.properties.insert(key, PropertyValue::String(v));
        } else if let Ok(arr) = value.cast::<PyArray1<f32>>() {
            let vec = arr.readonly().as_array().to_vec();
            inner
                .properties
                .insert(key, PropertyValue::Float32Array(vec));
        } else if let Ok(arr) = value.cast::<PyArray1<f64>>() {
            let vec = arr.readonly().as_array().to_vec();
            inner.properties.insert(key, PropertyValue::FloatArray(vec));
        } else if let Ok(arr) = value.cast::<PyArray1<i32>>() {
            let vec = arr.readonly().as_array().to_vec();
            inner.properties.insert(key, PropertyValue::Int32Array(vec));
        } else if let Ok(arr) = value.cast::<PyArray1<i64>>() {
            let vec = arr.readonly().as_array().to_vec();
            inner.properties.insert(key, PropertyValue::IntArray(vec));
        } else if let Ok(arr) = value.cast::<PyArray2<f32>>() {
            let readonly = arr.readonly();
            let arr_view = readonly.as_array();
            let shape = arr_view.shape();
            if shape[1] != 3 {
                return Err(PyValueError::new_err(
                    "Vec3Array properties must have shape (n, 3)",
                ));
            }
            let vec: Vec<[f32; 3]> = arr_view
                .outer_iter()
                .map(|row| [row[0], row[1], row[2]])
                .collect();
            inner.properties.insert(key, PropertyValue::Vec3Array(vec));
        } else if let Ok(arr) = value.cast::<PyArray2<f64>>() {
            let readonly = arr.readonly();
            let arr_view = readonly.as_array();
            let shape = arr_view.shape();
            if shape[1] != 3 {
                return Err(PyValueError::new_err(
                    "Vec3Array properties must have shape (n, 3)",
                ));
            }
            let vec: Vec<[f64; 3]> = arr_view
                .outer_iter()
                .map(|row| [row[0], row[1], row[2]])
                .collect();
            inner
                .properties
                .insert(key, PropertyValue::Vec3ArrayF64(vec));
        } else {
            return Err(PyValueError::new_err(
                "Unsupported property type. Supported: float, int, str, ndarray",
            ));
        }
        Ok(())
    }

    /// Get all property keys
    fn property_keys(&self) -> PyResult<Vec<String>> {
        if let Some(inner) = self.as_owned() {
            Ok(inner.properties.keys().cloned().collect())
        } else if let Some(view) = self.as_view() {
            view.property_keys()
        } else {
            Ok(Vec::new())
        }
    }

    /// Check if a property exists
    fn has_property(&self, key: &str) -> PyResult<bool> {
        if let Some(inner) = self.as_owned() {
            Ok(inner.properties.contains_key(key))
        } else if let Some(view) = self.as_view() {
            Ok(view.find_custom_section(KIND_MOL_PROP, key)?.is_some())
        } else {
            Ok(false)
        }
    }

    /// Index molecule atoms by integer, or custom properties by string.
    fn __getitem__<'py>(slf: Bound<'py, Self>, index: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let molecule = slf.borrow();

        if let Ok(key) = index.extract::<String>() {
            if let Some(inner) = molecule.as_owned() {
                return match inner.properties.get(&key) {
                    Some(v) => property_value_to_pyobject(py, v),
                    None => Err(PyKeyError::new_err(format!("Property '{}' not found", key))),
                };
            }
            let view = molecule.as_view().ok_or_else(|| {
                PyValueError::new_err("Molecule is missing both owned and view state")
            })?;
            return match view.find_custom_section(KIND_MOL_PROP, &key)? {
                Some(section) => property_section_to_pyobject(py, view, section),
                None => Err(PyKeyError::new_err(format!("Property '{}' not found", key))),
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
