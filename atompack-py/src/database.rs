use super::*;
#[path = "database_batch.rs"]
mod batch;
#[path = "database_flat.rs"]
mod flat;

/// Python wrapper for AtomDatabase
#[pyclass]
pub(crate) struct PyAtomDatabase {
    inner: AtomDatabase,
}

impl PyAtomDatabase {
    fn soa_context(&self) -> PyResult<SoaContext> {
        SoaContext::from_database(&self.inner).map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    fn load_mmap_view(
        &self,
        index: usize,
        compression: CompressionType,
        ctx: SoaContext,
    ) -> atompack::Result<SoaMoleculeView> {
        if compression == CompressionType::None {
            let bytes = self.inner.get_shared_mmap_bytes(index).ok_or_else(|| {
                invalid_data(format!("Missing mmap bytes for molecule {}", index))
            })?;
            return SoaMoleculeView::from_shared_bytes(bytes, ctx);
        }

        let compressed = self.inner.get_compressed_slice(index).ok_or_else(|| {
            invalid_data(format!("Missing compressed bytes for molecule {}", index))
        })?;
        let uncompressed_size = self.inner.uncompressed_size(index).ok_or_else(|| {
            invalid_data(format!("Missing uncompressed size for molecule {}", index))
        })? as usize;
        let decompressed =
            atompack::decompress_bytes(compressed, compression, Some(uncompressed_size))?;
        SoaMoleculeView::from_owned_bytes(decompressed, ctx)
    }

    fn single_molecule_view(&self, py: Python<'_>, index: usize) -> PyResult<SoaMoleculeView> {
        let compression = self.inner.compression();
        let ctx = self.soa_context()?;
        let use_mmap = self.inner.get_compressed_slice(0).is_some();

        if use_mmap {
            return py
                .detach(|| self.load_mmap_view(index, compression, ctx))
                .map_err(|e| PyValueError::new_err(format!("{}", e)));
        }

        let mut raw_bytes = py
            .detach(|| self.inner.get_raw_bytes(&[index]))
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let raw = raw_bytes.pop().ok_or_else(|| {
            PyValueError::new_err(format!("Missing raw bytes for molecule {}", index))
        })?;
        SoaMoleculeView::from_bytes(raw, ctx)
    }

    fn molecule_views(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
    ) -> PyResult<Vec<SoaMoleculeView>> {
        let compression = self.inner.compression();
        let ctx = self.soa_context()?;
        let use_mmap = self.inner.get_compressed_slice(0).is_some();

        let views = if use_mmap {
            let results: Vec<atompack::Result<SoaMoleculeView>> = py.detach(|| {
                use rayon::prelude::*;
                indices
                    .par_iter()
                    .map(|&idx| self.load_mmap_view(idx, compression, ctx))
                    .collect()
            });
            results.into_iter().collect::<atompack::Result<Vec<_>>>()
        } else {
            let raw_bytes = py
                .detach(|| self.inner.get_raw_bytes(&indices))
                .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
            raw_bytes
                .into_iter()
                .map(|bytes| SoaMoleculeView::from_bytes(bytes, ctx))
                .collect::<PyResult<Vec<_>>>()
                .map_err(|e| invalid_data(format!("{}", e)))
        }
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(views)
    }
}

#[pymethods]
impl PyAtomDatabase {
    /// Create a new database
    ///
    /// Parameters:
    /// - path: Path to the database file
    /// - compression: "none", "lz4", or "zstd" (default: "none")
    /// - level: Compression level for zstd (1-22, default: 3)
    #[new]
    #[pyo3(signature = (path, compression="none", level=3, overwrite=false))]
    fn new(path: String, compression: &str, level: i32, overwrite: bool) -> PyResult<Self> {
        let compression_type = match compression.to_lowercase().as_str() {
            "none" => CompressionType::None,
            "lz4" => CompressionType::Lz4,
            "zstd" => CompressionType::Zstd(level),
            _ => return Err(PyValueError::new_err("Invalid compression type")),
        };

        let path_buf = PathBuf::from(path);
        if path_buf.exists() && !overwrite {
            return Err(PyFileExistsError::new_err(format!(
                "Database file already exists at '{}'. Use Database.open(path) for read-only access, Database.open(path, mmap=False) to append, or pass overwrite=True to recreate it.",
                path_buf.display()
            )));
        }

        let db = AtomDatabase::create(&path_buf, compression_type)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(Self { inner: db })
    }

    /// Open an existing database.
    ///
    /// By default this uses a memory-mapped index and is read-only. Reopen with
    /// `mmap=False` if you need to append molecules.
    #[staticmethod]
    #[pyo3(signature = (path, mmap=true, populate=false))]
    fn open(path: String, mmap: bool, populate: bool) -> PyResult<Self> {
        if populate && !mmap {
            return Err(PyValueError::new_err("populate=True requires mmap=True"));
        }

        let db = if mmap {
            if populate {
                AtomDatabase::open_mmap_populate(PathBuf::from(path))
            } else {
                AtomDatabase::open_mmap(PathBuf::from(path))
            }
        } else {
            AtomDatabase::open(PathBuf::from(path))
        }
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(Self { inner: db })
    }

    /// Add a molecule to the database
    fn add_molecule(&mut self, molecule: &PyMolecule) -> PyResult<()> {
        if let Some(view) = molecule.as_view() {
            return self
                .inner
                .add_raw_soa_records_with_schema(
                    &[(view.raw_bytes(), view.n_atoms as u32)],
                    view.database_schema()?,
                )
                .map_err(|e| PyValueError::new_err(format!("{}", e)));
        }
        let owned = molecule.clone_as_owned()?;
        self.inner
            .add_molecule(&owned)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    /// Add multiple molecules (processed in parallel)
    fn add_molecules(&mut self, molecules: Vec<PyRef<PyMolecule>>) -> PyResult<()> {
        let mut raw_records: Vec<(&[u8], u32)> = Vec::new();
        let mut raw_views: Vec<&SoaMoleculeView> = Vec::new();
        let mut owned_molecules: Vec<Molecule> = Vec::new();

        for m in &molecules {
            if let Some(view) = m.as_view() {
                raw_records.push((view.raw_bytes(), view.n_atoms as u32));
                raw_views.push(view);
            } else {
                owned_molecules.push(m.clone_as_owned()?);
            }
        }

        if !raw_records.is_empty() {
            let mut fast_schema = None;
            if let Some((first, rest)) = raw_views.split_first() {
                let mut all_match = true;
                for view in rest {
                    if !view.same_schema_as(first)? {
                        all_match = false;
                        break;
                    }
                }
                if all_match {
                    fast_schema = Some(first.database_schema()?);
                }
            }

            if let Some(schema) = fast_schema {
                self.inner
                    .add_raw_soa_records_with_schema(&raw_records, schema)
                    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
            } else {
                let raw_records_with_positions = raw_views
                    .iter()
                    .map(|view| (view.raw_bytes(), view.n_atoms as u32, view.positions_type))
                    .collect::<Vec<_>>();
                self.inner
                    .add_raw_soa_records_with_positions_type(&raw_records_with_positions)
                    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
            }
        }
        if !owned_molecules.is_empty() {
            let mol_refs: Vec<&Molecule> = owned_molecules.iter().collect();
            self.inner
                .add_molecules(&mol_refs)
                .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        }
        Ok(())
    }

    /// Add a batch of molecules from stacked numpy arrays.
    ///
    /// This avoids creating one Python Molecule object per record.
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
        name=None,
        properties=None,
        atom_properties=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_arrays_batch(
        &mut self,
        py: Python<'_>,
        positions: &Bound<'_, PyAny>,
        atomic_numbers: &Bound<'_, PyArray2<u8>>,
        energy: Option<&Bound<'_, PyAny>>,
        forces: Option<&Bound<'_, PyAny>>,
        charges: Option<&Bound<'_, PyAny>>,
        velocities: Option<&Bound<'_, PyAny>>,
        cell: Option<&Bound<'_, PyAny>>,
        stress: Option<&Bound<'_, PyAny>>,
        pbc: Option<&Bound<'_, PyArray2<bool>>>,
        name: Option<Vec<String>>,
        properties: Option<&Bound<'_, PyDict>>,
        atom_properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        batch::add_arrays_batch_impl(
            &mut self.inner,
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
            properties,
            atom_properties,
        )
    }

    /// Get a molecule by index as a lazy view-backed molecule.
    fn get_molecule(&self, py: Python<'_>, index: usize) -> PyResult<PyMolecule> {
        let len = self.inner.len();
        if index >= len {
            return Err(PyIndexError::new_err(format!(
                "Index {} out of bounds for database of length {}",
                index, len
            )));
        }
        Ok(PyMolecule::from_view(self.single_molecule_view(py, index)?))
    }

    /// Get multiple molecules by indices (parallel batch reading)
    ///
    /// This is MUCH faster than calling get_molecule() in a loop!
    /// Uses parallel I/O to read multiple molecules at once.
    ///
    /// Parameters:
    /// - indices: List of molecule indices to fetch
    ///
    /// Returns:
    /// - List of molecules
    fn get_molecules(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<Vec<PyMolecule>> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        let views = self.molecule_views(py, indices)?;
        Ok(views.into_iter().map(PyMolecule::from_view).collect())
    }

    /// Get multiple molecules as batched numpy arrays (zero-copy fast path).
    ///
    /// Returns a dict with contiguous arrays instead of per-molecule Python objects.
    /// This is the fastest way to load training batches.
    ///
    /// The dict always contains:
    ///   "n_atoms":         uint32  (n_molecules,)
    ///   "positions":       float32 (total_atoms, 3)
    ///   "atomic_numbers":  uint8   (total_atoms,)
    ///
    /// Optional builtin fields (present if the first molecule has them):
    ///   "energy":     float64 (n_molecules,)
    ///   "forces":     float32 (total_atoms, 3)
    ///   "charges":    float64 (total_atoms,)
    ///   "velocities": float32 (total_atoms, 3)
    ///   "cell":       float64 (n_molecules, 3, 3)
    ///   "stress":     float64 (n_molecules, 3, 3)
    ///   "pbc":        bool    (n_molecules, 3)
    ///
    /// Custom properties are in nested dicts:
    ///   "atom_properties": {key: array, ...}
    ///   "properties":      {key: array, ...}
    fn get_molecules_flat<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        flat::get_molecules_flat_soa_impl(&self.inner, py, indices)
    }

    /// Get the atom count for one molecule without materializing it.
    fn num_atoms(&self, index: usize) -> PyResult<u32> {
        self.inner.num_atoms(index).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "Index {} out of bounds for database of length {}",
                index,
                self.inner.len()
            ))
        })
    }

    /// Get atom counts for a selection of molecules.
    #[pyo3(signature = (indices=None))]
    fn atom_counts<'py>(
        &self,
        py: Python<'py>,
        indices: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let selected = indices.unwrap_or_else(|| (0..self.inner.len()).collect());
        let counts = selected
            .into_iter()
            .map(|index| self.num_atoms(index))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyArray1::from_slice(py, &counts))
    }

    /// Get the number of molecules in the database
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Enable indexing: db[i]
    fn __getitem__(&self, py: Python<'_>, index: usize) -> PyResult<PyMolecule> {
        self.get_molecule(py, index)
    }

    /// Flush and save the database
    fn flush(&mut self) -> PyResult<()> {
        self.inner
            .flush()
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }
}
