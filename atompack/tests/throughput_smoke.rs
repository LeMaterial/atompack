// Copyright 2026 Entalpic
use atompack::{
    Atom, AtomDatabase, FloatScalarData, Molecule, Vec3Data, compression::CompressionType,
};
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

#[derive(Clone, Copy)]
struct Threshold {
    label: &'static str,
    env: &'static str,
    default: f64,
}

impl Threshold {
    fn value(self) -> f64 {
        std::env::var(self.env)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(self.default)
    }
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

fn rate(units: usize, elapsed: Duration) -> f64 {
    units as f64 / elapsed.as_secs_f64().max(f64::EPSILON)
}

fn color(code: &str, text: impl AsRef<str>) -> String {
    let enabled = match std::env::var("ATOMPACK_PERF_COLOR").as_deref() {
        Ok("always") => true,
        Ok("never") => false,
        _ => std::env::var_os("NO_COLOR").is_none(),
    };
    if enabled {
        format!("\x1b[{code}m{}\x1b[0m", text.as_ref())
    } else {
        text.as_ref().to_string()
    }
}

fn cyan(text: impl AsRef<str>) -> String {
    color("36;1", text)
}

fn green(text: impl AsRef<str>) -> String {
    color("32;1", text)
}

fn red(text: impl AsRef<str>) -> String {
    color("31;1", text)
}

fn yellow(text: impl AsRef<str>) -> String {
    color("33;1", text)
}

fn format_rate(value: f64) -> String {
    let mut digits = format!("{:.0}", value);
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    while digits.len() > 3 {
        let tail = digits.split_off(digits.len() - 3);
        if grouped.is_empty() {
            grouped = tail;
        } else {
            grouped = format!("{tail},{grouped}");
        }
    }
    if grouped.is_empty() {
        grouped = digits;
    } else {
        grouped = format!("{digits},{grouped}");
    }
    format!("{grouped:>12}")
}

fn print_metric(label: &str, value: f64, threshold: Threshold) {
    let floor = threshold.value();
    let passed = value >= floor;
    let status = if passed { green("PASS") } else { red("FAIL") };
    println!(
        "  {label:<22} {mol_s} mol/s  {atoms_s} atoms/s  min {floor_s}  {status:<13} {env}",
        mol_s = format_rate(value),
        atoms_s = format_rate(value * ATOMS_PER_MOLECULE as f64),
        floor_s = format_rate(floor),
        env = threshold.env,
    );
}

fn print_report(metrics: Metrics, thresholds: &[Threshold]) {
    println!();
    println!("{}", cyan("Atompack Rust Throughput Smoke"));
    println!(
        "  dataset: {} molecules x {} atoms, compression=none, props=energy+forces",
        N_MOLECULES, ATOMS_PER_MOLECULE
    );
    println!(
        "  reads: random_reads={}, shuffled_batch_reads={}, read_batch={}, file_size={} bytes",
        RANDOM_READS, SHUFFLE_READS, READ_BATCH, metrics.file_size_bytes
    );
    println!(
        "  {}",
        yellow("small warm-cache smoke test; not a publication benchmark")
    );
    println!();
    println!(
        "  {metric:<22} {mol_s:>18}  {atoms_s:>18}  {floor:>16}  {status:<13} env override",
        metric = "metric",
        mol_s = "throughput",
        atoms_s = "atom throughput",
        floor = "floor",
        status = "status",
    );
    println!("  {}", "-".repeat(103));
    print_metric("write", metrics.write_mol_s, thresholds[0]);
    print_metric("sequential read", metrics.seq_read_mol_s, thresholds[1]);
    print_metric("random read", metrics.rand_read_mol_s, thresholds[2]);
    print_metric(
        "shuffled batch read",
        metrics.shuffle_read_mol_s,
        thresholds[3],
    );
    println!();
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
    molecule.energy = Some(FloatScalarData::F64(-1_000.0 - id as f64 * 1e-3));
    molecule.forces = Some(Vec3Data::F32(
        (0..ATOMS_PER_MOLECULE)
            .map(|i| {
                let t = id as f32 * 0.002 + i as f32 * 0.02;
                [(t * 0.7).sin(), (t * 0.9).cos(), (t * 1.1).sin()]
            })
            .collect(),
    ));
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
        black_box(molecule.forces.as_ref().map(Vec3Data::len));
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
    let thresholds = [
        Threshold {
            label: "Rust write",
            env: "ATOMPACK_RUST_MIN_WRITE_MOL_S",
            default: 25_000.0,
        },
        Threshold {
            label: "Rust sequential read",
            env: "ATOMPACK_RUST_MIN_SEQ_READ_MOL_S",
            default: 75_000.0,
        },
        Threshold {
            label: "Rust random read",
            env: "ATOMPACK_RUST_MIN_RAND_READ_MOL_S",
            default: 50_000.0,
        },
        Threshold {
            label: "Rust shuffled batch read",
            env: "ATOMPACK_RUST_MIN_SHUFFLE_READ_MOL_S",
            default: 50_000.0,
        },
    ];

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
    print_report(metrics, &thresholds);

    assert!(
        metrics.write_mol_s >= thresholds[0].value(),
        "{} throughput regressed: {:.0} mol/s",
        thresholds[0].label,
        metrics.write_mol_s
    );
    assert!(
        metrics.seq_read_mol_s >= thresholds[1].value(),
        "{} throughput regressed: {:.0} mol/s",
        thresholds[1].label,
        metrics.seq_read_mol_s
    );
    assert!(
        metrics.rand_read_mol_s >= thresholds[2].value(),
        "{} throughput regressed: {:.0} mol/s",
        thresholds[2].label,
        metrics.rand_read_mol_s
    );
    assert!(
        metrics.shuffle_read_mol_s >= thresholds[3].value(),
        "{} throughput regressed: {:.0} mol/s",
        thresholds[3].label,
        metrics.shuffle_read_mol_s
    );

    Ok(())
}
