// Copyright 2026 Entalpic
//! Minimal read/write roundtrip for atompack.
//!
//! Run with: cargo run -p atompack --example basic_usage

use atompack::{Atom, AtomDatabase, Molecule, compression::CompressionType};

fn main() -> atompack::Result<()> {
    let db_path = std::env::temp_dir().join(format!("atompack-basic-{}.atp", std::process::id()));

    let mut water = Molecule::from_atoms(vec![
        Atom::new(0.0, 0.0, 0.0, 8),
        Atom::new(0.96, 0.0, 0.0, 1),
        Atom::new(-0.24, 0.93, 0.0, 1),
    ]);
    water.name = Some("water".to_string());
    water.energy = Some(-76.4);

    let methane = Molecule::from_atoms(vec![
        Atom::new(0.0, 0.0, 0.0, 6),
        Atom::new(1.09, 0.0, 0.0, 1),
        Atom::new(-0.36, 1.03, 0.0, 1),
        Atom::new(-0.36, -0.51, 0.89, 1),
        Atom::new(-0.36, -0.51, -0.89, 1),
    ]);

    let mut db = AtomDatabase::create(&db_path, CompressionType::Zstd(3))?;
    db.add_molecules(&[&water, &methane])?;
    db.flush()?;

    let mut db = AtomDatabase::open(&db_path)?;
    assert_eq!(db.len(), 2);

    let roundtrip = db.get_molecule(0)?;
    assert_eq!(roundtrip.name.as_deref(), Some("water"));
    assert_eq!(roundtrip.energy, Some(-76.4));
    assert_eq!(roundtrip.len(), 3);

    println!("wrote {}", db_path.display());
    println!("loaded {} molecules", db.len());
    println!(
        "first molecule: name={:?} atoms={} energy={:?}",
        roundtrip.name,
        roundtrip.len(),
        roundtrip.energy
    );

    std::fs::remove_file(&db_path)?;
    Ok(())
}
