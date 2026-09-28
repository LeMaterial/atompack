use super::*;
use atompack::{GroupColumn, Grouping};
use numpy::PyReadonlyArray2;
use pyo3::types::{PyIterator, PySlice};
use std::collections::HashMap;
use std::ops::Range;

fn db_err(e: atompack::Error) -> PyErr {
    PyValueError::new_err(format!("{}", e))
}

fn grouping<'a>(db: &'a AtomDatabase, name: &str) -> PyResult<&'a Grouping> {
    db.groups()
        .map_err(db_err)?
        .get(name)
        .ok_or_else(|| PyKeyError::new_err(format!("No grouping named '{}'", name)))
}

fn member_range(g: &Grouping, index: usize) -> PyResult<Range<usize>> {
    g.member_range(index).ok_or_else(|| {
        PyIndexError::new_err(format!(
            "Group index {} out of bounds for grouping of length {}",
            index,
            g.len()
        ))
    })
}

fn record_index(value: i64) -> PyResult<u64> {
    u64::try_from(value).map_err(|_| {
        PyValueError::new_err(format!(
            "Group member index must be non-negative, got {}",
            value
        ))
    })
}

fn role_id(g: &mut Grouping, role: String) -> u16 {
    match g.roles.iter().position(|r| *r == role) {
        Some(id) => id as u16,
        None => {
            g.roles.push(role);
            (g.roles.len() - 1) as u16
        }
    }
}

/// Build a CSR grouping from `members` (dicts of role -> index, or lists of
/// indices) or, with `roles`, from a (n_groups, n_roles) array where -1 marks
/// an absent member. Invariants are checked by `AtomDatabase::add_groups`.
pub(super) fn parse_grouping(
    members: &Bound<'_, PyAny>,
    properties: Option<&Bound<'_, PyDict>>,
    roles: Option<Vec<String>>,
) -> PyResult<Grouping> {
    let mut g = Grouping::default();
    if let Some(roles) = roles {
        g.roles = roles;
        let push_row = |g: &mut Grouping, row: &mut dyn Iterator<Item = i64>| -> PyResult<()> {
            let mut n = 0;
            for (role, idx) in row.enumerate() {
                n += 1;
                if idx != -1 {
                    g.records.push(record_index(idx)?);
                    g.member_roles.push(role as u16);
                }
            }
            if n != g.roles.len() {
                return Err(PyValueError::new_err(format!(
                    "Each members row needs {} entries (one per role), got {}",
                    g.roles.len(),
                    n
                )));
            }
            g.offsets.push(g.records.len() as u64);
            Ok(())
        };
        if let Ok(array) = members.extract::<PyReadonlyArray2<i64>>() {
            for row in array.as_array().rows() {
                push_row(&mut g, &mut row.iter().copied())?;
            }
        } else {
            for row in members.extract::<Vec<Vec<i64>>>()? {
                push_row(&mut g, &mut row.into_iter())?;
            }
        }
    } else {
        let mut named = None;
        for item in members.try_iter()? {
            let item = item?;
            let dict = item.downcast::<PyDict>().ok();
            if *named.get_or_insert(dict.is_some()) != dict.is_some() {
                return Err(PyTypeError::new_err(
                    "members must be all dicts (named roles) or all lists (ordered)",
                ));
            }
            if let Some(dict) = dict {
                for (role, idx) in dict.iter() {
                    let id = role_id(&mut g, role.extract()?);
                    g.records.push(record_index(idx.extract()?)?);
                    g.member_roles.push(id);
                }
            } else {
                for idx in item.extract::<Vec<i64>>()? {
                    g.records.push(record_index(idx)?);
                }
            }
            g.offsets.push(g.records.len() as u64);
        }
    }

    for (key, values) in properties.into_iter().flat_map(|p| p.iter()) {
        let key: String = key.extract()?;
        let column = if let Ok(v) = values.extract::<Vec<i64>>() {
            GroupColumn::Int(v)
        } else if let Ok(v) = values.extract::<Vec<f64>>() {
            GroupColumn::Float(v)
        } else if let Ok(v) = values.extract::<Vec<String>>() {
            GroupColumn::String(v)
        } else {
            return Err(PyTypeError::new_err(format!(
                "Group property '{}' must be a sequence of ints, floats, or strings",
                key
            )));
        };
        g.properties.push((key, column));
    }
    Ok(g)
}

/// Record indices of one group: {role: index} for named roles, else [index].
fn members_object<'py, T>(
    py: Python<'py>,
    g: &Grouping,
    range: Range<usize>,
    mut value: impl FnMut(u64) -> PyResult<T>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: IntoPyObject<'py>,
{
    if g.roles.is_empty() {
        let items = g.records[range]
            .iter()
            .map(|&r| value(r))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(PyList::new(py, items)?.into_any());
    }
    let dict = PyDict::new(py);
    for i in range {
        let role = &g.roles[g.member_roles[i] as usize];
        dict.set_item(role, value(g.records[i])?)?;
    }
    Ok(dict.into_any())
}

fn property_columns<'py>(py: Python<'py>, g: &Grouping) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, column) in &g.properties {
        match column {
            GroupColumn::Float(v) => dict.set_item(key, PyArray1::from_slice(py, v))?,
            GroupColumn::Int(v) => dict.set_item(key, PyArray1::from_slice(py, v))?,
            GroupColumn::String(v) => dict.set_item(key, v)?,
        }
    }
    Ok(dict)
}

fn property_values<'py>(
    py: Python<'py>,
    g: &Grouping,
    index: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, column) in &g.properties {
        match column {
            GroupColumn::Float(v) => dict.set_item(key, v[index])?,
            GroupColumn::Int(v) => dict.set_item(key, v[index])?,
            GroupColumn::String(v) => dict.set_item(key, &v[index])?,
        }
    }
    Ok(dict)
}

/// Load groups; records shared between the requested groups are read and
/// decompressed once.
fn load_groups(
    db: &PyAtomDatabase,
    py: Python<'_>,
    name: &str,
    indices: Vec<usize>,
) -> PyResult<Vec<PyGroup>> {
    let g = grouping(&db.inner, name)?;
    let ranges = indices
        .iter()
        .map(|&i| member_range(g, i))
        .collect::<PyResult<Vec<_>>>()?;

    let mut slot: HashMap<u64, usize> = HashMap::new();
    let mut unique = Vec::new();
    for &r in ranges.iter().flat_map(|range| &g.records[range.clone()]) {
        slot.entry(r).or_insert_with(|| {
            unique.push(r as usize);
            unique.len() - 1
        });
    }
    let views = if unique.is_empty() {
        Vec::new()
    } else {
        db.molecule_views(py, unique)?
    };

    ranges
        .into_iter()
        .zip(indices)
        .map(|(range, index)| {
            let members = members_object(py, g, range.clone(), |r| {
                Ok(PyMolecule::from_view(views[slot[&r]].clone()))
            })?;
            Ok(PyGroup {
                members: members.unbind(),
                indices: members_object(py, g, range, Ok)?.unbind(),
                properties: property_values(py, g, index)?.unbind(),
            })
        })
        .collect()
}

fn normalize_index(index: isize, len: usize) -> PyResult<usize> {
    let resolved = if index < 0 {
        index + len as isize
    } else {
        index
    };
    if resolved < 0 || resolved as usize >= len {
        return Err(PyIndexError::new_err(format!(
            "Group index {} out of bounds for grouping of length {}",
            index, len
        )));
    }
    Ok(resolved as usize)
}

/// Mapping of grouping name -> Grouping, available as `Database.groups`.
#[pyclass(name = "Groups", module = "atompack")]
pub(crate) struct PyGroups {
    pub(super) db: Py<PyAtomDatabase>,
}

impl PyGroups {
    fn names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let db = self.db.borrow(py);
        let groups = db.inner.groups().map_err(db_err)?;
        Ok(groups.keys().cloned().collect())
    }
}

#[pymethods]
impl PyGroups {
    fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<PyGrouping> {
        grouping(&self.db.borrow(py).inner, name)?;
        Ok(PyGrouping {
            db: self.db.clone_ref(py),
            name: name.to_string(),
        })
    }

    fn __contains__(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        Ok(self.names(py)?.iter().any(|n| n == name))
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.names(py)?.len())
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        PyList::new(py, self.names(py)?)?.try_iter()
    }

    /// Names of the groupings.
    fn keys(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.names(py)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("Groups({:?})", self.names(py)?))
    }
}

/// A named grouping: a sequence of groups sharing roles and property keys.
#[pyclass(name = "Grouping", module = "atompack")]
pub(crate) struct PyGrouping {
    db: Py<PyAtomDatabase>,
    name: String,
}

impl PyGrouping {
    fn with<T>(&self, py: Python<'_>, f: impl FnOnce(&Grouping) -> PyResult<T>) -> PyResult<T> {
        f(grouping(&self.db.borrow(py).inner, &self.name)?)
    }

    fn load(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<Vec<PyGroup>> {
        load_groups(&self.db.borrow(py), py, &self.name, indices)
    }
}

#[pymethods]
impl PyGrouping {
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// Role names (empty for ordered groups).
    #[getter]
    fn roles(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.with(py, |g| Ok(g.roles.clone()))
    }

    /// Group properties as columns: numpy arrays for numbers, lists for str.
    #[getter]
    fn properties<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.with(py, |g| property_columns(py, g))
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |g| Ok(g.len()))
    }

    /// `grouping[i]` -> Group; `grouping[a:b]` or `grouping[[i, j]]` -> list of Groups.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let len = self.__len__(py)?;
        if let Ok(i) = index.extract::<isize>() {
            let group = self.load(py, vec![normalize_index(i, len)?])?.remove(0);
            return Ok(Bound::new(py, group)?.into_any());
        }
        let indices: Vec<usize> = if let Ok(slice) = index.downcast::<PySlice>() {
            let s = slice.indices(len as isize)?;
            (0..s.slicelength)
                .map(|k| (s.start + k as isize * s.step) as usize)
                .collect()
        } else {
            index
                .extract::<Vec<isize>>()?
                .into_iter()
                .map(|i| normalize_index(i, len))
                .collect::<PyResult<_>>()?
        };
        Ok(PyList::new(py, self.load(py, indices)?)?.into_any())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyGroupingIter {
        PyGroupingIter {
            grouping: slf.into(),
            next: 0,
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.with(py, |g| {
            let keys: Vec<_> = g.properties.iter().map(|(k, _)| k).collect();
            Ok(format!(
                "Grouping({:?}, len={}, roles={:?}, properties={:?})",
                self.name,
                g.len(),
                g.roles,
                keys
            ))
        })
    }
}

#[pyclass(module = "atompack")]
pub(crate) struct PyGroupingIter {
    grouping: Py<PyGrouping>,
    next: usize,
}

#[pymethods]
impl PyGroupingIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyGroup>> {
        let grouping = self.grouping.borrow(py);
        if self.next >= grouping.__len__(py)? {
            return Ok(None);
        }
        let group = grouping.load(py, vec![self.next])?.remove(0);
        self.next += 1;
        Ok(Some(group))
    }
}

/// One group: `group[role]` (or `group[i]` for ordered groups) is a Molecule.
#[pyclass(name = "Group", module = "atompack")]
pub(crate) struct PyGroup {
    members: Py<PyAny>,
    indices: Py<PyAny>,
    properties: Py<PyDict>,
}

#[pymethods]
impl PyGroup {
    /// {role: Molecule} for named roles, [Molecule, ...] for ordered groups.
    #[getter]
    fn members(&self, py: Python<'_>) -> Py<PyAny> {
        self.members.clone_ref(py)
    }

    /// Record indices, shaped like `members`.
    #[getter]
    fn indices(&self, py: Python<'_>) -> Py<PyAny> {
        self.indices.clone_ref(py)
    }

    #[getter]
    fn properties(&self, py: Python<'_>) -> Py<PyDict> {
        self.properties.clone_ref(py)
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.members.bind(py).get_item(key)
    }

    fn __contains__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.members.bind(py).contains(key)
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.members.bind(py).len()
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.members.bind(py).try_iter()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Group(indices={}, properties={})",
            self.indices.bind(py).repr()?,
            self.properties.bind(py).repr()?
        ))
    }
}
