// Copyright 2026 Entalpic
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyValue {
    None,
    Float(f64),
    Int(i64),
    String(String),
    FloatArray(Vec<f64>),
    Vec3Array(Vec<[f32; 3]>),
    IntArray(Vec<i64>),
    Float32Array(Vec<f32>),
    Vec3ArrayF64(Vec<[f64; 3]>),
    Int32Array(Vec<i32>),
}

impl PropertyValue {
    pub fn len(&self) -> Option<usize> {
        match self {
            PropertyValue::FloatArray(values) => Some(values.len()),
            PropertyValue::Vec3Array(values) => Some(values.len()),
            PropertyValue::IntArray(values) => Some(values.len()),
            PropertyValue::Float32Array(values) => Some(values.len()),
            PropertyValue::Vec3ArrayF64(values) => Some(values.len()),
            PropertyValue::Int32Array(values) => Some(values.len()),
            _ => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Vec3Data {
    F32(Vec<[f32; 3]>),
    F64(Vec<[f64; 3]>),
}

impl Vec3Data {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn atom_position(&self, index: usize) -> Option<[f32; 3]> {
        match self {
            Self::F32(values) => values.get(index).copied(),
            Self::F64(values) => values
                .get(index)
                .map(|value| [value[0] as f32, value[1] as f32, value[2] as f32]),
        }
    }

    pub fn flatten_f32_lossy(&self) -> Vec<f32> {
        match self {
            Self::F32(values) => values
                .iter()
                .flat_map(|value| [value[0], value[1], value[2]])
                .collect(),
            Self::F64(values) => values
                .iter()
                .flat_map(|value| [value[0] as f32, value[1] as f32, value[2] as f32])
                .collect(),
        }
    }

    pub fn flatten_f64(&self) -> Vec<f64> {
        match self {
            Self::F32(values) => values
                .iter()
                .flat_map(|value| [value[0] as f64, value[1] as f64, value[2] as f64])
                .collect(),
            Self::F64(values) => values
                .iter()
                .flat_map(|value| [value[0], value[1], value[2]])
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FloatScalarData {
    F32(f32),
    F64(f64),
}

impl FloatScalarData {
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::F32(value) => *value as f64,
            Self::F64(value) => *value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FloatArrayData {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl FloatArrayData {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Self::F32(values) => values.iter().map(|value| *value as f64).collect(),
            Self::F64(values) => values.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Mat3Data {
    F32([[f32; 3]; 3]),
    F64([[f64; 3]; 3]),
}

impl Mat3Data {
    pub fn flatten_f32_lossy(&self) -> Vec<f32> {
        match self {
            Self::F32(values) => values
                .iter()
                .flat_map(|row| [row[0], row[1], row[2]])
                .collect(),
            Self::F64(values) => values
                .iter()
                .flat_map(|row| [row[0] as f32, row[1] as f32, row[2] as f32])
                .collect(),
        }
    }

    pub fn flatten_f64(&self) -> Vec<f64> {
        match self {
            Self::F32(values) => values
                .iter()
                .flat_map(|row| [row[0] as f64, row[1] as f64, row[2] as f64])
                .collect(),
            Self::F64(values) => values
                .iter()
                .flat_map(|row| [row[0], row[1], row[2]])
                .collect(),
        }
    }
}
