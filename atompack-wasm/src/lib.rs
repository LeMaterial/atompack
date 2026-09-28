//! WebAssembly reader for atompack files, driven by a JS host.
//!
//! The host imports `atompack.read_at(source, offset, len, dst) -> u32` (0 on
//! success), which copies a byte range of the file it registered as `source`
//! into wasm memory. Every exported call leaves a JSON reply, `{"ok": ...}` or
//! `{"error": "..."}`, readable through `reply_ptr()` / `reply_len()`.
//! Groupings are addressed by their position in `overview().groupings`.

use atompack::storage::type_tag_name;
use atompack::types::TensorData;
use atompack::{AtomReader, GroupColumn, Molecule, PropertyValue, ReadAt};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

pub fn overview<R: ReadAt>(reader: &AtomReader<R>) -> atompack::Result<Value> {
    let compression = match reader.compression() {
        atompack::compression::CompressionType::None => json!({"kind": "none"}),
        atompack::compression::CompressionType::Lz4 => json!({"kind": "lz4"}),
        atompack::compression::CompressionType::Zstd(level) => {
            json!({"kind": "zstd", "level": level})
        }
    };
    let schema = reader.schema_info().map(|schema| {
        let sections: Vec<Value> = schema
            .sections
            .iter()
            .map(|s| {
                json!({
                    "kind": match s.kind { 0 => "builtin", 1 => "atom", _ => "molecule" },
                    "key": s.key,
                    "dtype": type_tag_name(s.type_tag),
                    "per_atom": s.per_atom,
                })
            })
            .collect();
        json!({
            "positions_dtype": schema.positions_type.map(type_tag_name),
            "sections": sections,
        })
    });
    let groupings: Vec<Value> = reader
        .groups()?
        .iter()
        .map(|(name, g)| {
            let properties: Vec<Value> = g
                .properties
                .iter()
                .map(|(key, column)| {
                    let dtype = match column {
                        GroupColumn::Float(_) => "float64",
                        GroupColumn::Int(_) => "int64",
                        GroupColumn::String(_) => "string",
                    };
                    json!({"key": key, "dtype": dtype})
                })
                .collect();
            json!({
                "name": name,
                "count": g.len(),
                "n_members": g.records.len(),
                "roles": g.roles,
                "properties": properties,
            })
        })
        .collect();
    Ok(json!({
        "num_records": reader.len(),
        "compression": compression,
        "record_format": reader.record_format(),
        "schema": schema,
        "groupings": groupings,
    }))
}

/// One table row per record: counts, composition and scalar values.
pub fn records<R: ReadAt>(
    reader: &AtomReader<R>,
    start: usize,
    count: usize,
) -> atompack::Result<Value> {
    let end = start.saturating_add(count).min(reader.len());
    let rows: Vec<Value> = reader
        .get_molecules(start.min(end)..end)?
        .iter()
        .zip(start..)
        .map(|(mol, index)| {
            let mut composition = BTreeMap::<u8, usize>::new();
            for &z in &mol.atomic_numbers {
                *composition.entry(z).or_default() += 1;
            }
            let properties: Map<String, Value> = sorted(&mol.properties)
                .map(|(key, value)| (key.clone(), summary(value)))
                .collect();
            json!({
                "index": index,
                "n_atoms": mol.len(),
                "name": mol.name,
                "composition": composition.into_iter().collect::<Vec<_>>(),
                "energy": mol.energy.as_ref().map(|e| e.as_f64()),
                "periodic": mol.cell.is_some(),
                "properties": properties,
            })
        })
        .collect();
    Ok(Value::Array(rows))
}

/// Full record for rendering and inspection.
pub fn molecule<R: ReadAt>(reader: &AtomReader<R>, index: usize) -> atompack::Result<Value> {
    Ok(molecule_json(index, &reader.get_molecule(index)?))
}

fn molecule_json(index: usize, mol: &Molecule) -> Value {
    let props = |map| {
        sorted(map)
            .map(|(k, v)| (k.clone(), property(v)))
            .collect::<Map<_, _>>()
    };
    json!({
        "index": index,
        "name": mol.name,
        "numbers": mol.atomic_numbers,
        "positions": mol.positions.flatten_f64(),
        "cell": mol.cell.as_ref().map(|c| c.flatten_f64()),
        "pbc": mol.pbc,
        "energy": mol.energy.as_ref().map(|e| e.as_f64()),
        "forces": mol.forces.as_ref().map(|f| f.flatten_f64()),
        "charges": mol.charges.as_ref().map(|c| c.to_f64_vec()),
        "velocities": mol.velocities.as_ref().map(|v| v.flatten_f64()),
        "stress": mol.stress.as_ref().map(|s| s.flatten_f64()),
        "properties": props(&mol.properties),
        "atom_properties": props(&mol.atom_properties),
    })
}

/// A page of groups: members with roles, plus the group's properties.
pub fn groups<R: ReadAt>(
    reader: &AtomReader<R>,
    grouping: usize,
    start: usize,
    count: usize,
) -> atompack::Result<Value> {
    let g = grouping_at(reader, grouping)?;
    let end = start.saturating_add(count).min(g.len());
    let rows: Vec<Value> = (start.min(end)..end)
        .map(|i| {
            let range = g.member_range(i).expect("index within grouping");
            let members: Vec<Value> = range
                .map(|m| {
                    let role = g.member_roles.get(m).map(|&r| g.roles[r as usize].as_str());
                    json!({"record": g.records[m], "role": role})
                })
                .collect();
            let properties: Map<String, Value> = g
                .properties
                .iter()
                .map(|(key, column)| {
                    let value = match column {
                        GroupColumn::Float(v) => json!(v[i]),
                        GroupColumn::Int(v) => json!(v[i]),
                        GroupColumn::String(v) => json!(v[i]),
                    };
                    (key.clone(), value)
                })
                .collect();
            json!({"index": i, "members": members, "properties": properties})
        })
        .collect();
    Ok(Value::Array(rows))
}

/// All values of every property of a grouping (stored columnar, so cheap).
pub fn group_columns<R: ReadAt>(
    reader: &AtomReader<R>,
    grouping: usize,
) -> atompack::Result<Value> {
    let columns: Map<String, Value> = grouping_at(reader, grouping)?
        .properties
        .iter()
        .map(|(key, column)| {
            let values = match column {
                GroupColumn::Float(v) => json!(v),
                GroupColumn::Int(v) => json!(v),
                GroupColumn::String(v) => json!(v),
            };
            (key.clone(), values)
        })
        .collect();
    Ok(Value::Object(columns))
}

fn grouping_at<R: ReadAt>(
    reader: &AtomReader<R>,
    index: usize,
) -> atompack::Result<&atompack::Grouping> {
    reader
        .groups()?
        .values()
        .nth(index)
        .ok_or_else(|| atompack::Error::InvalidData(format!("Grouping {} out of bounds", index)))
}

fn sorted(
    map: &std::collections::HashMap<String, PropertyValue>,
) -> impl Iterator<Item = (&String, &PropertyValue)> {
    map.iter().collect::<BTreeMap<_, _>>().into_iter()
}

fn property(value: &PropertyValue) -> Value {
    match value {
        PropertyValue::None => Value::Null,
        PropertyValue::Float(v) => json!(v),
        PropertyValue::Int(v) => json!(v),
        PropertyValue::String(v) => json!(v),
        PropertyValue::FloatArray(v) => json!(v),
        PropertyValue::Vec3Array(v) => json!(v),
        PropertyValue::IntArray(v) => json!(v),
        PropertyValue::Float32Array(v) => json!(v),
        PropertyValue::Vec3ArrayF64(v) => json!(v),
        PropertyValue::Int32Array(v) => json!(v),
        PropertyValue::Tensor(t) => match t {
            TensorData::F32 { shape, values } => json!({"shape": shape, "values": values}),
            TensorData::F64 { shape, values } => json!({"shape": shape, "values": values}),
            TensorData::I32 { shape, values } => json!({"shape": shape, "values": values}),
            TensorData::I64 { shape, values } => json!({"shape": shape, "values": values}),
        },
    }
}

/// Scalars as-is; arrays as their shape, e.g. "[12×3]".
fn summary(value: &PropertyValue) -> Value {
    let shape = match value {
        PropertyValue::Vec3Array(v) => vec![v.len(), 3],
        PropertyValue::Vec3ArrayF64(v) => vec![v.len(), 3],
        PropertyValue::Tensor(t) => t.shape().to_vec(),
        other => match other.len() {
            Some(n) => vec![n],
            None => return property(other),
        },
    };
    let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
    json!(format!("[{}]", dims.join("×")))
}

#[cfg(target_arch = "wasm32")]
mod host {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    #[link(wasm_import_module = "atompack")]
    unsafe extern "C" {
        fn read_at(source: u32, offset: f64, len: u32, dst: *mut u8) -> u32;
    }

    struct HostSource {
        id: u32,
        size: u64,
    }

    impl ReadAt for HostSource {
        fn size(&self) -> atompack::Result<u64> {
            Ok(self.size)
        }

        fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> atompack::Result<()> {
            let status =
                unsafe { read_at(self.id, offset as f64, buf.len() as u32, buf.as_mut_ptr()) };
            if status == 0 {
                Ok(())
            } else {
                Err(std::io::Error::other(format!("host read failed at offset {offset}")).into())
            }
        }
    }

    thread_local! {
        static READERS: RefCell<HashMap<u32, AtomReader<HostSource>>> = RefCell::default();
        static REPLY: RefCell<Vec<u8>> = RefCell::default();
    }

    /// Store the reply; returns 0 on success, 1 on error.
    fn reply(result: atompack::Result<Value>) -> u32 {
        let (status, value) = match result {
            Ok(value) => (0, json!({"ok": value})),
            Err(err) => (1, json!({"error": err.to_string()})),
        };
        REPLY.with(|r| *r.borrow_mut() = serde_json::to_vec(&value).expect("JSON value"));
        status
    }

    fn with_reader(
        source: u32,
        f: impl FnOnce(&AtomReader<HostSource>) -> atompack::Result<Value>,
    ) -> u32 {
        reply(READERS.with(|readers| match readers.borrow().get(&source) {
            Some(reader) => f(reader),
            None => Err(atompack::Error::InvalidData(format!(
                "Source {source} is not open"
            ))),
        }))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn reply_ptr() -> *const u8 {
        REPLY.with(|r| r.borrow().as_ptr())
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn reply_len() -> u32 {
        REPLY.with(|r| r.borrow().len() as u32)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn open(source: u32, size: f64) -> u32 {
        let opened = AtomReader::open(HostSource {
            id: source,
            size: size as u64,
        });
        reply(opened.map(|reader| {
            READERS.with(|readers| readers.borrow_mut().insert(source, reader));
            Value::Null
        }))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn close(source: u32) {
        READERS.with(|readers| readers.borrow_mut().remove(&source));
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn overview(source: u32) -> u32 {
        with_reader(source, super::overview)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn records(source: u32, start: u32, count: u32) -> u32 {
        with_reader(source, |r| {
            super::records(r, start as usize, count as usize)
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn molecule(source: u32, index: u32) -> u32 {
        with_reader(source, |r| super::molecule(r, index as usize))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn groups(source: u32, grouping: u32, start: u32, count: u32) -> u32 {
        with_reader(source, |r| {
            super::groups(r, grouping as usize, start as usize, count as usize)
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn group_columns(source: u32, grouping: u32) -> u32 {
        with_reader(source, |r| super::group_columns(r, grouping as usize))
    }
}
