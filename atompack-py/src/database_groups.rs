use super::*;
use atompack::{GroupColumn, Grouping};
use numpy::PyReadonlyArray2;
use std::collections::HashMap;
use std::ops::Range;

fn db_err(e: atompack::Error) -> PyErr {
    PyValueError::new_err(format!("{}", e))
}

pub(super) fn grouping<'a>(db: &'a AtomDatabase, name: &str) -> PyResult<&'a Grouping> {
    db.groups()
        .map_err(db_err)?
        .get(name)
        .ok_or_else(|| PyKeyError::new_err(format!("No grouping named '{}'", name)))
}

pub(super) fn member_range(g: &Grouping, index: usize) -> PyResult<Range<usize>> {
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
pub(super) fn members_object<'py, T>(
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

pub(super) fn property_columns<'py>(py: Python<'py>, g: &Grouping) -> PyResult<Bound<'py, PyDict>> {
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

/// Load groups as {"members": ..., "properties": ...} dicts. Records shared
/// between the requested groups are read and decompressed once.
pub(super) fn get_groups_impl<'py>(
    db: &PyAtomDatabase,
    py: Python<'py>,
    name: &str,
    indices: Vec<usize>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
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
            let members = members_object(py, g, range, |r| {
                Ok(PyMolecule::from_view(views[slot[&r]].clone()))
            })?;
            let dict = PyDict::new(py);
            dict.set_item("members", members)?;
            dict.set_item("properties", property_values(py, g, index)?)?;
            Ok(dict)
        })
        .collect()
}
