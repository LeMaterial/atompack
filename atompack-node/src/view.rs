//! Viewer projections over the existing native database API.

use atompack::types::TensorData;
use atompack::{AtomDatabase, GroupColumn, Molecule, PropertyValue};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

pub fn overview(reader: &AtomDatabase) -> atompack::Result<Value> {
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
pub fn records(reader: &AtomDatabase, start: usize, count: usize) -> atompack::Result<Value> {
    let end = start.saturating_add(count).min(reader.len());
    let rows: Vec<Value> = reader
        .get_molecules(&(start.min(end)..end).collect::<Vec<_>>())?
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

/// Numeric per-record values for plotting, columnar: `{key: [value or null, ...]}`.
/// Built-ins (`n_atoms`, `energy`, `energy_per_atom`, `fmax`) take precedence over same-named
/// properties. `composition` holds `"Z:count"` pairs by atomic number, e.g. `"1:2 8:1"`.
pub fn record_columns(
    reader: &AtomDatabase,
    start: usize,
    count: usize,
) -> atompack::Result<Value> {
    let end = start.saturating_add(count).min(reader.len());
    let start = start.min(end);
    let mut columns = BTreeMap::<String, Vec<Value>>::new();
    // Decode a few records at a time: only the values are kept, and holding a whole chunk of
    // large structures could otherwise exhaust memory.
    for batch in (start..end).step_by(64) {
        let mols = reader.get_molecules(&(batch..(batch + 64).min(end)).collect::<Vec<_>>())?;
        for (j, mol) in mols.iter().enumerate() {
            let mut set = |key: &str, value: Value| {
                columns
                    .entry(key.to_owned())
                    .or_insert_with(|| vec![Value::Null; end - start])[batch - start + j] = value
            };
            for (key, value) in &mol.properties {
                match value {
                    PropertyValue::Float(v) => set(key, json!(v)),
                    PropertyValue::Int(v) => set(key, json!(v)),
                    _ => {}
                }
            }
            set("n_atoms", json!(mol.len()));
            if let Some(energy) = &mol.energy {
                set("energy", json!(energy.as_f64()));
                if !mol.is_empty() {
                    set("energy_per_atom", json!(energy.as_f64() / mol.len() as f64));
                }
            }
            let mut composition = BTreeMap::<u8, usize>::new();
            for &z in &mol.atomic_numbers {
                *composition.entry(z).or_default() += 1;
            }
            let key: Vec<String> = composition
                .iter()
                .map(|(z, n)| format!("{z}:{n}"))
                .collect();
            set("composition", json!(key.join(" ")));
            if let Some(forces) = &mol.forces {
                let flat = forces.flatten_f64();
                let fmax = flat
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|[x, y, z]| (x * x + y * y + z * z).sqrt())
                    .fold(0.0, f64::max);
                set("fmax", json!(fmax));
            }
        }
    }
    Ok(json!(columns))
}

/// Full record for rendering and inspection.
pub fn molecule(reader: &AtomDatabase, index: usize) -> atompack::Result<Value> {
    Ok(molecule_json(index, &reader.get_molecules(&[index])?[0]))
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
pub fn groups(
    reader: &AtomDatabase,
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
pub fn group_columns(reader: &AtomDatabase, grouping: usize) -> atompack::Result<Value> {
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

fn grouping_at(reader: &AtomDatabase, index: usize) -> atompack::Result<&atompack::Grouping> {
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

fn type_tag_name(type_tag: u8) -> &'static str {
    match type_tag {
        0 => "float64",
        1 => "int64",
        2 => "string",
        3 => "float64[]",
        4 => "vec3<float32>",
        5 => "int64[]",
        6 => "float32[]",
        7 => "vec3<float64>",
        8 => "int32[]",
        9 => "bool[3]",
        10 => "mat3x3<float64>",
        11 => "float32",
        12 => "mat3x3<float32>",
        13 => "none",
        14 => "tensor<float32>",
        15 => "tensor<float64>",
        16 => "tensor<int32>",
        17 => "tensor<int64>",
        _ => "unknown",
    }
}
