// Copyright 2026 Entalpic
use atompack::{Atom, AtomDatabase, Molecule, compression::CompressionType};
use std::hint::black_box;
use std::time::{Duration, Instant};

const N_MOLECULES: usize = 10_000;
const ATOMS_PER_MOLECULE: usize = 64;
const WRITE_BATCH: usize = 2_048;
const RANDOM_READS: usize = 2_000;
const SHUFFLE_READS: usize = 5_000;
const READ_BATCH: usize = 1_024;

#[derive(Clone, Copy)]
struct Metrics {
    write_mol_s: f64,
    seq_read_mol_s: f64,
    rand_read_mol_s: f64,
    shuffle_read_mol_s: f64,
    file_size_bytes: u64,
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }
}

fn env_threshold(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn rate(units: usize, elapsed: Duration) -> f64 {
    units as f64 / elapsed.as_secs_f64().max(f64::EPSILON)
}

fn synthetic_molecule(id: usize) -> Molecule {
    let atoms = (0..ATOMS_PER_MOLECULE)
        .map(|i| {
            let t = id as f32 * 0.001 + i as f32 * 0.01;
            let atomic_number = match (id + i) % 5 {
                0 => 1,
                1 => 6,
                2 => 7,
                3 => 8,
                _ => 16,
            };
            Atom::new(
                (t * 1.3).sin() * 10.0,
                (t * 1.7).cos() * 10.0,
                (t * 2.1).sin() * 10.0,
                atomic_number,
            )
        })
        .collect();
    let mut molecule = Molecule::from_atoms(atoms);
    molecule.energy = Some(-1_000.0 - id as f64 * 1e-3);
    molecule.forces = Some(
        (0..ATOMS_PER_MOLECULE)
            .map(|i| {
                let t = id as f32 * 0.002 + i as f32 * 0.02;
                [(t * 0.7).sin(), (t * 0.9).cos(), (t * 1.1).sin()]
            })
            .collect(),
    );
    molecule
}

fn run_smoke() -> atompack::Result<Metrics> {
    let file = tempfile::Builder::new()
        .prefix("atompack-throughput-smoke-")
        .suffix(".atp")
        .tempfile()?;
    let path = file.path().to_path_buf();

    let mut db = AtomDatabase::create(&path, CompressionType::None)?;
    let start_write = Instant::now();
    let mut written = 0usize;
    while written < N_MOLECULES {
        let batch_len = (N_MOLECULES - written).min(WRITE_BATCH);
        let molecules = (0..batch_len)
            .map(|offset| synthetic_molecule(written + offset))
            .collect::<Vec<_>>();
        let refs = molecules.iter().collect::<Vec<_>>();
        db.add_molecules(&refs)?;
        written += batch_len;
    }
    db.flush()?;
    let write_elapsed = start_write.elapsed();
    let file_size_bytes = std::fs::metadata(&path)?.len();

    let mut db = AtomDatabase::open(&path)?;
    let start_seq = Instant::now();
    let mut seq_atoms = 0usize;
    for index in 0..db.len() {
        let molecule = db.get_molecule(index)?;
        seq_atoms += molecule.len();
        black_box(molecule.energy);
    }
    let seq_elapsed = start_seq.elapsed();
    assert_eq!(seq_atoms, N_MOLECULES * ATOMS_PER_MOLECULE);

    let mut rng = XorShift64::new(12_345);
    let start_rand = Instant::now();
    let mut rand_atoms = 0usize;
    for _ in 0..RANDOM_READS {
        let molecule = db.get_molecule(rng.index(db.len()))?;
        rand_atoms += molecule.len();
        black_box(molecule.forces.as_ref().map(Vec::len));
    }
    let rand_elapsed = start_rand.elapsed();
    assert_eq!(rand_atoms, RANDOM_READS * ATOMS_PER_MOLECULE);

    let db = AtomDatabase::open_mmap(&path)?;
    let mut rng = XorShift64::new(67_890);
    let mut remaining = SHUFFLE_READS;
    let mut indices = vec![0usize; READ_BATCH];
    let start_shuffle = Instant::now();
    let mut shuffle_atoms = 0usize;
    while remaining > 0 {
        let batch_len = remaining.min(READ_BATCH);
        for index in &mut indices[..batch_len] {
            *index = rng.index(db.len());
        }
        let molecules = db.get_molecules(&indices[..batch_len])?;
        shuffle_atoms += molecules.iter().map(Molecule::len).sum::<usize>();
        black_box(molecules.len());
        remaining -= batch_len;
    }
    let shuffle_elapsed = start_shuffle.elapsed();
    assert_eq!(shuffle_atoms, SHUFFLE_READS * ATOMS_PER_MOLECULE);

    Ok(Metrics {
        write_mol_s: rate(N_MOLECULES, write_elapsed),
        seq_read_mol_s: rate(N_MOLECULES, seq_elapsed),
        rand_read_mol_s: rate(RANDOM_READS, rand_elapsed),
        shuffle_read_mol_s: rate(SHUFFLE_READS, shuffle_elapsed),
        file_size_bytes,
    })
}

#[test]
#[ignore = "run with `make perf-smoke-rust` so throughput is measured in release mode"]
fn atompack_rust_throughput_smoke() -> atompack::Result<()> {
    let metrics = run_smoke()?;

    println!(
        "atompack_rust_perf_smoke n_molecules={} atoms_per_molecule={} \
         write_mol_s={:.0} seq_read_mol_s={:.0} rand_read_mol_s={:.0} \
         shuffle_read_mol_s={:.0} file_size_bytes={}",
        N_MOLECULES,
        ATOMS_PER_MOLECULE,
        metrics.write_mol_s,
        metrics.seq_read_mol_s,
        metrics.rand_read_mol_s,
        metrics.shuffle_read_mol_s,
        metrics.file_size_bytes,
    );

    assert!(
        metrics.write_mol_s >= env_threshold("ATOMPACK_RUST_MIN_WRITE_MOL_S", 25_000.0),
        "Rust write throughput regressed: {:.0} mol/s",
        metrics.write_mol_s
    );
    assert!(
        metrics.seq_read_mol_s >= env_threshold("ATOMPACK_RUST_MIN_SEQ_READ_MOL_S", 75_000.0),
        "Rust sequential read throughput regressed: {:.0} mol/s",
        metrics.seq_read_mol_s
    );
    assert!(
        metrics.rand_read_mol_s >= env_threshold("ATOMPACK_RUST_MIN_RAND_READ_MOL_S", 50_000.0),
        "Rust random read throughput regressed: {:.0} mol/s",
        metrics.rand_read_mol_s
    );
    assert!(
        metrics.shuffle_read_mol_s
            >= env_threshold("ATOMPACK_RUST_MIN_SHUFFLE_READ_MOL_S", 50_000.0),
        "Rust shuffled batch read throughput regressed: {:.0} mol/s",
        metrics.shuffle_read_mol_s
    );

    Ok(())
}
