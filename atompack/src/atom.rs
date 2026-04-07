// Copyright 2026 Entalpic
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct Atom {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Atomic number (1 = H, 6 = C, 8 = O, etc.)
    pub atomic_number: u8,
    /// Padding to ensure 16-byte alignment (required for bytemuck::Pod)
    _padding: [u8; 3],
}

impl Atom {
    pub fn new(x: f32, y: f32, z: f32, atomic_number: u8) -> Self {
        Self {
            x,
            y,
            z,
            atomic_number,
            _padding: [0; 3],
        }
    }

    pub fn position(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn distance_to(&self, other: &Atom) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyValue {
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
            PropertyValue::FloatArray(v) => Some(v.len()),
            PropertyValue::Vec3Array(v) => Some(v.len()),
            PropertyValue::IntArray(v) => Some(v.len()),
            PropertyValue::Float32Array(v) => Some(v.len()),
            PropertyValue::Vec3ArrayF64(v) => Some(v.len()),
            PropertyValue::Int32Array(v) => Some(v.len()),
            _ => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Molecule {
    pub name: Option<String>,

    pub positions: Vec<[f32; 3]>,

    pub atomic_numbers: Vec<u8>,

    pub forces: Option<Vec<[f32; 3]>>,

    pub energy: Option<f64>,

    pub charges: Option<Vec<f64>>,

    pub velocities: Option<Vec<[f32; 3]>>,

    pub cell: Option<[[f64; 3]; 3]>,

    pub pbc: Option<[bool; 3]>,

    pub stress: Option<[[f64; 3]; 3]>,

    /// Per-atom properties (features or targets)
    pub atom_properties: HashMap<String, PropertyValue>,

    /// Per-molecule properties (targets or metadata)
    pub properties: HashMap<String, PropertyValue>,
}

impl Molecule {
    pub fn new(positions: Vec<[f32; 3]>, atomic_numbers: Vec<u8>) -> Result<Self, String> {
        Self::with_name_internal(None, positions, atomic_numbers)
    }

    pub fn with_name(
        name: String,
        positions: Vec<[f32; 3]>,
        atomic_numbers: Vec<u8>,
    ) -> Result<Self, String> {
        Self::with_name_internal(Some(name), positions, atomic_numbers)
    }

    fn with_name_internal(
        name: Option<String>,
        positions: Vec<[f32; 3]>,
        atomic_numbers: Vec<u8>,
    ) -> Result<Self, String> {
        if positions.len() != atomic_numbers.len() {
            return Err(format!(
                "Atomic numbers length ({}) doesn't match atom count ({})",
                atomic_numbers.len(),
                positions.len()
            ));
        }
        Ok(Self {
            name,
            positions,
            atomic_numbers,
            forces: None,
            energy: None,
            charges: None,
            velocities: None,
            cell: None,
            pbc: None,
            stress: None,
            atom_properties: HashMap::new(),
            properties: HashMap::new(),
        })
    }

    pub fn from_atoms(atoms: Vec<Atom>) -> Self {
        let positions = atoms.iter().map(|atom| atom.position()).collect();
        let atomic_numbers = atoms.iter().map(|atom| atom.atomic_number).collect();
        Self {
            name: None,
            positions,
            atomic_numbers,
            forces: None,
            energy: None,
            charges: None,
            velocities: None,
            cell: None,
            pbc: None,
            stress: None,
            atom_properties: HashMap::new(),
            properties: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn atom(&self, index: usize) -> Option<Atom> {
        Some(Atom::new(
            self.positions.get(index)?[0],
            self.positions.get(index)?[1],
            self.positions.get(index)?[2],
            *self.atomic_numbers.get(index)?,
        ))
    }

    pub fn to_atoms(&self) -> Vec<Atom> {
        self.positions
            .iter()
            .zip(&self.atomic_numbers)
            .map(|(position, &atomic_number)| {
                Atom::new(position[0], position[1], position[2], atomic_number)
            })
            .collect()
    }

    pub fn forces_mut(&mut self) -> &mut Vec<[f32; 3]> {
        let n_atoms = self.len();
        self.forces.get_or_insert_with(|| vec![[0.0; 3]; n_atoms])
    }

    pub fn velocities_mut(&mut self) -> &mut Vec<[f32; 3]> {
        let n_atoms = self.len();
        self.velocities
            .get_or_insert_with(|| vec![[0.0; 3]; n_atoms])
    }

    pub fn charges_mut(&mut self) -> &mut Vec<f64> {
        let n_atoms = self.len();
        self.charges.get_or_insert_with(|| vec![0.0; n_atoms])
    }

    /// Add a per-atom floating-point property
    ///
    /// # Errors
    ///
    /// Returns an error if the property length doesn't match the number of atoms
    pub fn add_atom_property(&mut self, name: String, values: Vec<f64>) -> Result<(), String> {
        if values.len() != self.len() {
            return Err(format!(
                "Property '{}' length ({}) doesn't match atom count ({})",
                name,
                values.len(),
                self.len()
            ));
        }
        self.atom_properties
            .insert(name, PropertyValue::FloatArray(values));
        Ok(())
    }

    /// Add a per-atom vector property
    ///
    /// # Errors
    ///
    /// Returns an error if the property length doesn't match the number of atoms
    pub fn add_atom_vec3_property(
        &mut self,
        name: String,
        values: Vec<[f32; 3]>,
    ) -> Result<(), String> {
        if values.len() != self.len() {
            return Err(format!(
                "Property '{}' length ({}) doesn't match atom count ({})",
                name,
                values.len(),
                self.len()
            ));
        }
        self.atom_properties
            .insert(name, PropertyValue::Vec3Array(values));
        Ok(())
    }

    /// Add a per-atom integer property
    ///
    /// # Errors
    ///
    /// Returns an error if the property length doesn't match the number of atoms
    pub fn add_atom_int_property(&mut self, name: String, values: Vec<i64>) -> Result<(), String> {
        if values.len() != self.len() {
            return Err(format!(
                "Property '{}' length ({}) doesn't match atom count ({})",
                name,
                values.len(),
                self.len()
            ));
        }
        self.atom_properties
            .insert(name, PropertyValue::IntArray(values));
        Ok(())
    }

    /// Set a molecular property (scalar, string, or other)
    pub fn set_property(&mut self, name: String, value: PropertyValue) {
        self.properties.insert(name, value);
    }

    /// Get a per-atom float property by name
    pub fn get_atom_property(&self, name: &str) -> Option<&[f64]> {
        match self.atom_properties.get(name)? {
            PropertyValue::FloatArray(v) => Some(v),
            _ => None,
        }
    }

    /// Get a per-atom vector property by name
    pub fn get_atom_vec3_property(&self, name: &str) -> Option<&[[f32; 3]]> {
        match self.atom_properties.get(name)? {
            PropertyValue::Vec3Array(v) => Some(v),
            _ => None,
        }
    }

    /// Get a per-atom integer property by name
    pub fn get_atom_int_property(&self, name: &str) -> Option<&[i64]> {
        match self.atom_properties.get(name)? {
            PropertyValue::IntArray(v) => Some(v),
            _ => None,
        }
    }

    /// Get a molecular property by name
    pub fn get_property(&self, name: &str) -> Option<&PropertyValue> {
        self.properties.get(name)
    }

    /// Get a molecular scalar property as f64
    pub fn get_scalar(&self, name: &str) -> Option<f64> {
        match self.properties.get(name)? {
            PropertyValue::Float(v) => Some(*v),
            PropertyValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Get a molecular string property
    pub fn get_string(&self, name: &str) -> Option<&str> {
        match self.properties.get(name)? {
            PropertyValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get positions as a flat array
    ///
    /// Returns: [x1, y1, z1, x2, y2, z2, ...]
    pub fn positions_flat(&self) -> Vec<f32> {
        self.positions
            .iter()
            .flat_map(|position| [position[0], position[1], position[2]])
            .collect()
    }

    /// Get atomic numbers as an array
    pub fn atomic_numbers(&self) -> Vec<u8> {
        self.atomic_numbers.clone()
    }

    /// Get forces as a flat array (if present)
    ///
    /// Returns: Some([fx1, fy1, fz1, fx2, fy2, fz2, ...]) or None
    pub fn forces_flat(&self) -> Option<Vec<f32>> {
        self.forces
            .as_ref()
            .map(|f| f.iter().flat_map(|v| [v[0], v[1], v[2]]).collect())
    }

    /// Get velocities as a flat array (if present)
    pub fn velocities_flat(&self) -> Option<Vec<f32>> {
        self.velocities
            .as_ref()
            .map(|v| v.iter().flat_map(|vec| [vec[0], vec[1], vec[2]]).collect())
    }

    /// Get cell as a flat array (if present)
    ///
    /// Returns: Some([a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z]) or None
    pub fn cell_flat(&self) -> Option<Vec<f64>> {
        self.cell
            .as_ref()
            .map(|c| c.iter().flat_map(|row| [row[0], row[1], row[2]]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_creation() {
        let atom = Atom::new(1.0, 2.0, 3.0, 6); // Carbon at (1, 2, 3)
        assert_eq!(atom.position(), [1.0, 2.0, 3.0]);
        assert_eq!(atom.atomic_number, 6);
    }

    #[test]
    fn test_distance() {
        let atom1 = Atom::new(0.0, 0.0, 0.0, 1);
        let atom2 = Atom::new(3.0, 4.0, 0.0, 1);
        assert_eq!(atom1.distance_to(&atom2), 5.0);
    }

    #[test]
    fn test_molecule() {
        let atoms = [
            Atom::new(0.0, 0.0, 0.0, 8),  // Oxygen
            Atom::new(1.0, 0.0, 0.0, 1),  // Hydrogen
            Atom::new(-1.0, 0.0, 0.0, 1), // Hydrogen
        ];
        let mol = Molecule::with_name(
            "Water".to_string(),
            atoms.iter().map(|atom| atom.position()).collect(),
            atoms.iter().map(|atom| atom.atomic_number).collect(),
        )
        .unwrap();
        assert_eq!(mol.len(), 3);
    }

    #[test]
    fn test_forces() {
        let mut mol = Molecule::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], vec![6, 8]).unwrap();

        // Add forces
        mol.forces = Some(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);

        // Test flat forces
        let forces_flat = mol.forces_flat().unwrap();
        assert_eq!(forces_flat, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    }

    #[test]
    fn test_atom_properties() {
        let mut mol = Molecule::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], vec![6, 8]).unwrap();

        // Add custom per-atom property
        mol.add_atom_property("custom_feature".to_string(), vec![1.0, 2.0])
            .unwrap();

        // Retrieve it
        let feature = mol.get_atom_property("custom_feature").unwrap();
        assert_eq!(feature, &[1.0, 2.0]);

        // Try to add property with wrong length - should fail
        let result = mol.add_atom_property("bad_feature".to_string(), vec![1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_molecule_properties() {
        let mut mol = Molecule::new(vec![[0.0, 0.0, 0.0]], vec![6]).unwrap();

        // Set energy
        mol.energy = Some(-100.5);

        // Set custom molecular property
        mol.set_property("homo_lumo_gap".to_string(), PropertyValue::Float(5.2));

        // Retrieve
        assert_eq!(mol.energy.unwrap(), -100.5);
        assert_eq!(mol.get_scalar("homo_lumo_gap").unwrap(), 5.2);
    }

    #[test]
    fn test_ml_array_conversion() {
        let mol = Molecule::new(vec![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], vec![1, 6]).unwrap();

        // Test positions_flat
        assert_eq!(mol.positions_flat(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        // Test atomic_numbers
        assert_eq!(mol.atomic_numbers(), vec![1, 6]);
    }

    #[test]
    fn test_periodic_cell() {
        let mut mol = Molecule::new(vec![[0.0, 0.0, 0.0]], vec![6]).unwrap();

        // Set unit cell (cubic 10x10x10)
        mol.cell = Some([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]);
        mol.pbc = Some([true, true, true]);

        // Test cell_flat
        let cell_flat = mol.cell_flat().unwrap();
        assert_eq!(
            cell_flat,
            vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]
        );
    }

    #[test]
    fn test_from_atoms_roundtrip() {
        let atoms = vec![Atom::new(1.0, 2.0, 3.0, 6), Atom::new(4.0, 5.0, 6.0, 8)];
        let mol = Molecule::from_atoms(atoms.clone());
        assert_eq!(mol.to_atoms(), atoms);
        assert_eq!(mol.atom(1).unwrap().atomic_number, 8);
    }

    #[test]
    fn test_constructor_validation() {
        let err = Molecule::new(vec![[0.0, 0.0, 0.0]], vec![1, 6]).unwrap_err();
        assert!(err.contains("Atomic numbers length"));
    }
}
