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
    fn single_molecule_view(&self, py: Python<'_>, index: usize) -> PyResult<SoaMoleculeView> {
        let compression = self.inner.compression();
        let use_mmap = self.inner.get_compressed_slice(0).is_some();

        if use_mmap {
            return py
                .detach(|| -> atompack::Result<SoaMoleculeView> {
                    if compression == CompressionType::None {
                        let bytes = self.inner.get_shared_mmap_bytes(index).ok_or_else(|| {
                            invalid_data(format!("Missing mmap bytes for molecule {}", index))
                        })?;
                        SoaMoleculeView::from_shared_bytes_inner(bytes)
                    } else {
                        let compressed =
                            self.inner.get_compressed_slice(index).ok_or_else(|| {
                                invalid_data(format!(
                                    "Missing compressed bytes for molecule {}",
                                    index
                                ))
                            })?;
                        let uncompressed_size =
                            self.inner.uncompressed_size(index).ok_or_else(|| {
                                invalid_data(format!(
                                    "Missing uncompressed size for molecule {}",
                                    index
                                ))
                            })? as usize;
                        let decompressed = atompack::decompress_bytes(
                            compressed,
                            compression,
                            Some(uncompressed_size),
                        )?;
                        SoaMoleculeView::from_bytes_inner(decompressed)
                    }
                })
                .map_err(|e| PyValueError::new_err(format!("{}", e)));
        }

        let mut raw_bytes = py
            .detach(|| self.inner.get_raw_bytes(&[index]))
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let raw = raw_bytes.pop().ok_or_else(|| {
            PyValueError::new_err(format!("Missing raw bytes for molecule {}", index))
        })?;
        SoaMoleculeView::from_bytes(raw)
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
        if let Some((soa_bytes, n_atoms)) = molecule.soa_bytes() {
            return self
                .inner
                .add_raw_soa_records(&[(soa_bytes, n_atoms)])
                .map_err(|e| PyValueError::new_err(format!("{}", e)));
        }
        let owned = molecule.clone_as_owned()?;
        self.inner
            .add_molecule(&owned)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    /// Add multiple molecules (processed in parallel)
    fn add_molecules(&mut self, molecules: Vec<PyRef<PyMolecule>>) -> PyResult<()> {
        // Split into view-backed (fast path) and owned molecules
        let mut raw_records: Vec<(&[u8], u32)> = Vec::new();
        let mut owned_molecules: Vec<Molecule> = Vec::new();

        for m in &molecules {
            if let Some((soa_bytes, n_atoms)) = m.soa_bytes() {
                raw_records.push((soa_bytes, n_atoms));
            } else {
                owned_molecules.push(m.clone_as_owned()?);
            }
        }

        if !raw_records.is_empty() {
            self.inner
                .add_raw_soa_records(&raw_records)
                .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
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
        positions: &Bound<'_, PyArray3<f32>>,
        atomic_numbers: &Bound<'_, PyArray2<u8>>,
        energy: Option<&Bound<'_, PyArray1<f64>>>,
        forces: Option<&Bound<'_, PyArray3<f32>>>,
        charges: Option<&Bound<'_, PyArray2<f64>>>,
        velocities: Option<&Bound<'_, PyArray3<f32>>>,
        cell: Option<&Bound<'_, PyArray3<f64>>>,
        stress: Option<&Bound<'_, PyArray3<f64>>>,
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

        let compression = self.inner.compression();
        let use_mmap = self.inner.get_compressed_slice(0).is_some();

        let views: Vec<SoaMoleculeView> = if use_mmap {
            let results: Vec<atompack::Result<SoaMoleculeView>> = py.detach(|| {
                use rayon::prelude::*;
                indices
                    .par_iter()
                    .map(|&idx| -> atompack::Result<SoaMoleculeView> {
                        if compression == CompressionType::None {
                            let bytes = self.inner.get_shared_mmap_bytes(idx).ok_or_else(|| {
                                invalid_data(format!("Missing mmap bytes for molecule {}", idx))
                            })?;
                            SoaMoleculeView::from_shared_bytes_inner(bytes)
                        } else {
                            let compressed =
                                self.inner.get_compressed_slice(idx).ok_or_else(|| {
                                    invalid_data(format!(
                                        "Missing compressed bytes for molecule {}",
                                        idx
                                    ))
                                })?;
                            let uncompressed_size =
                                self.inner.uncompressed_size(idx).ok_or_else(|| {
                                    invalid_data(format!(
                                        "Missing uncompressed size for molecule {}",
                                        idx
                                    ))
                                })? as usize;
                            let decompressed = atompack::decompress_bytes(
                                compressed,
                                compression,
                                Some(uncompressed_size),
                            )?;
                            SoaMoleculeView::from_bytes_inner(decompressed)
                        }
                    })
                    .collect()
            });
            results.into_iter().collect::<atompack::Result<Vec<_>>>()
        } else {
            let raw_bytes = py
                .detach(|| self.inner.get_raw_bytes(&indices))
                .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
            raw_bytes
                .into_iter()
                .map(SoaMoleculeView::from_bytes)
                .collect::<PyResult<Vec<_>>>()
                .map_err(|e| invalid_data(format!("{}", e)))
        }
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

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
