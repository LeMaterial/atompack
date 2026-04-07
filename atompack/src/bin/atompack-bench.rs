// Copyright 2026 Entalpic
//! Atompack benchmark runner (size, creation speed, read throughput, shuffling).
//!
//! Run with:
//!   cargo run -p atompack --release --bin atompack-bench -- --help

use atompack::{Atom, AtomDatabase, Molecule, compression::CompressionType};
use std::env;
use std::time::{Duration, Instant};

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn gen_index(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() % upper as u64) as usize
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.gen_index(i + 1);
            values.swap(i, j);
        }
    }
}

fn parse_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn parse_arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<T>().ok())
        .unwrap_or(default)
}

fn parse_compression(args: &[String]) -> CompressionType {
    let value = args
        .iter()
        .position(|a| a == "--compression")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("zstd:3");

    if value == "none" {
        return CompressionType::None;
    }
    if value == "lz4" {
        return CompressionType::Lz4;
    }
    if let Some(level_str) = value.strip_prefix("zstd:")
        && let Ok(level) = level_str.parse::<i32>()
    {
        return CompressionType::Zstd(level);
    }
    CompressionType::default()
}

fn create_synthetic_molecule(atoms_per_molecule: usize, id: u64, with_props: bool) -> Molecule {
    let atoms: Vec<Atom> = (0..atoms_per_molecule)
        .map(|i| {
            let t = (id as f32) * 0.001 + (i as f32) * 0.01;
            let x = (t * 1.3).sin() * 10.0;
            let y = (t * 1.7).cos() * 10.0;
            let z = (t * 2.1).sin() * 10.0;
            let znum = match (id as usize + i) % 5 {
                0 => 1,
                1 => 6,
                2 => 7,
                3 => 8,
                _ => 16,
            };
            Atom::new(x, y, z, znum)
        })
        .collect();

    let mut mol = Molecule::from_atoms(atoms);

    if with_props {
        let n = atoms_per_molecule;
        mol.energy = Some(-1000.0 - (id as f64) * 1e-3);
        mol.forces = Some(
            (0..n)
                .map(|i| {
                    let t = (id as f32) * 0.002 + (i as f32) * 0.02;
                    [(t * 0.7).sin(), (t * 0.9).cos(), (t * 1.1).sin()]
                })
                .collect(),
        );
    }

    mol
}

fn format_duration(d: Duration) -> String {
    format!("{:.6}", d.as_secs_f64())
}

fn read_proc_status_kb(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let mut parts = rest.split_whitespace();
            if let Some(num) = parts.next().and_then(|v| v.parse::<u64>().ok()) {
                return Some(num);
            }
        }
    }
    None
}

fn print_help() {
    eprintln!(
        r#"atompack-bench

Usage:
  cargo run -p atompack --release --bin atompack-bench -- [options]

Options:
  --path PATH                 Output .atp file (default: /tmp/atompack_bench.atp)
  --num-molecules N           Number of molecules (default: 100000)
  --atoms-per-molecule N      Atoms per molecule (default: 100)
  --write-batch N             Molecules per write batch (default: 8192)
  --read-batch N              Indices per parallel read batch (default: 4096)
  --random-reads N            Number of random single reads (default: 10000)
  --shuffle-samples N         Number of molecules to read in shuffled batches (default: 0 = all)
  --compression SPEC          none | lz4 | zstd:LEVEL (default: zstd:3)
  --with-props                Include energy + forces fields
  --keep                      Keep the generated file
  --help                      Show this help

Output:
  Prints key=value lines suitable for copy/paste into a spreadsheet or parser.
"#
    );
}

fn main() -> atompack::Result<()> {
    let args: Vec<String> = env::args().collect();
    if parse_flag(&args, "--help") {
        print_help();
        return Ok(());
    }

    let path = parse_arg(&args, "--path", "/tmp/atompack_bench.atp".to_string());
    let num_molecules: usize = parse_arg(&args, "--num-molecules", 100_000usize);
    let atoms_per_molecule: usize = parse_arg(&args, "--atoms-per-molecule", 100usize);
    let write_batch: usize = parse_arg(&args, "--write-batch", 8192usize);
    let read_batch: usize = parse_arg(&args, "--read-batch", 4096usize);
    let random_reads: usize = parse_arg(&args, "--random-reads", 10_000usize);
    let shuffle_samples: usize = parse_arg(&args, "--shuffle-samples", 0usize);
    let with_props = parse_flag(&args, "--with-props");
    let keep = parse_flag(&args, "--keep");
    let compression = parse_compression(&args);

    println!("format=atompack_v2");
    println!("path={}", path);
    println!("num_molecules={}", num_molecules);
    println!("atoms_per_molecule={}", atoms_per_molecule);
    println!("with_props={}", with_props);
    println!("write_batch={}", write_batch);
    println!("read_batch={}", read_batch);
    println!("random_reads={}", random_reads);
    println!("shuffle_samples={}", shuffle_samples);
    println!("compression={:?}", compression);

    let start_create = Instant::now();
    let mut db = AtomDatabase::create(&path, compression)?;
    let create_time = start_create.elapsed();

    let start_write = Instant::now();
    let mut written = 0usize;
    while written < num_molecules {
        let batch_n = (num_molecules - written).min(write_batch);
        let molecules: Vec<Molecule> = (0..batch_n)
            .map(|i| {
                create_synthetic_molecule(atoms_per_molecule, (written + i) as u64, with_props)
            })
            .collect();
        let refs: Vec<&Molecule> = molecules.iter().collect();
        db.add_molecules(&refs)?;
        written += batch_n;
    }
    let write_time = start_write.elapsed();

    let start_flush = Instant::now();
    db.flush()?;
    let flush_time = start_flush.elapsed();

    let file_size_bytes = std::fs::metadata(&path)?.len();

    println!("create_s={}", format_duration(create_time));
    println!("write_s={}", format_duration(write_time));
    println!("flush_s={}", format_duration(flush_time));
    println!("file_size_bytes={}", file_size_bytes);
    println!(
        "write_molecules_per_s={:.3}",
        (num_molecules as f64) / write_time.as_secs_f64()
    );

    let start_open = Instant::now();
    let mut db = AtomDatabase::open(&path)?;
    let open_time = start_open.elapsed();

    println!("open_s={}", format_duration(open_time));
    println!("len={}", db.len());

    let start_seq = Instant::now();
    let mut seq_atoms = 0usize;
    for i in 0..db.len() {
        let mol = db.get_molecule(i)?;
        seq_atoms += mol.len();
    }
    let seq_time = start_seq.elapsed();
    println!("seq_read_s={}", format_duration(seq_time));
    println!(
        "seq_read_molecules_per_s={:.3}",
        (db.len() as f64) / seq_time.as_secs_f64()
    );
    println!("seq_read_atoms={}", seq_atoms);

    let mut rng = XorShift64::new(12_345);
    let start_rand = Instant::now();
    let mut rand_atoms = 0usize;
    for _ in 0..random_reads.min(db.len().max(1)) {
        let idx = rng.gen_index(db.len());
        let mol = db.get_molecule(idx)?;
        rand_atoms += mol.len();
    }
    let rand_time = start_rand.elapsed();
    println!("rand_read_s={}", format_duration(rand_time));
    println!(
        "rand_read_molecules_per_s={:.3}",
        (random_reads.min(db.len().max(1)) as f64) / rand_time.as_secs_f64()
    );
    println!("rand_read_atoms={}", rand_atoms);

    let shuffle_total = if shuffle_samples == 0 {
        db.len()
    } else {
        shuffle_samples.min(db.len())
    };
    const MAX_PERM_LEN: usize = 5_000_000;
    let use_full_perm = shuffle_total == db.len() && db.len() <= MAX_PERM_LEN;
    println!(
        "shuffle_mode={}",
        if use_full_perm {
            "perm"
        } else {
            "random_with_replacement"
        }
    );
    let start_shuffle = Instant::now();
    let mut shuffle_atoms = 0usize;
    if use_full_perm {
        let mut perm: Vec<usize> = (0..db.len()).collect();
        rng.shuffle(&mut perm);
        for chunk in perm.chunks(read_batch.max(1)) {
            let batch = db.get_molecules(chunk)?;
            shuffle_atoms += batch.iter().map(|m| m.len()).sum::<usize>();
        }
    } else {
        let mut remaining = shuffle_total;
        let batch_size = read_batch.max(1);
        let mut indices = vec![0usize; batch_size];
        while remaining > 0 {
            let n = remaining.min(batch_size);
            for idx in &mut indices[..n] {
                *idx = rng.gen_index(db.len());
            }
            let batch = db.get_molecules(&indices[..n])?;
            shuffle_atoms += batch.iter().map(|m| m.len()).sum::<usize>();
            remaining -= n;
        }
    }
    let shuffle_time = start_shuffle.elapsed();
    println!("shuffle_read_s={}", format_duration(shuffle_time));
    println!(
        "shuffle_read_molecules_per_s={:.3}",
        (shuffle_total as f64) / shuffle_time.as_secs_f64()
    );
    println!("shuffle_read_atoms={}", shuffle_atoms);

    let start_open_mmap = Instant::now();
    let db_mmap = AtomDatabase::open_mmap(&path)?;
    let open_mmap_time = start_open_mmap.elapsed();
    println!("open_mmap_s={}", format_duration(open_mmap_time));

    if let Some(rss_kb) = read_proc_status_kb("VmRSS:") {
        println!("vmrss_kb={}", rss_kb);
    }
    if let Some(hwm_kb) = read_proc_status_kb("VmHWM:") {
        println!("vmhwm_kb={}", hwm_kb);
    }

    if !db_mmap.is_empty() {
        let sample = [0usize, db_mmap.len() / 2, db_mmap.len() - 1];
        let _ = db_mmap.get_molecules(&sample)?;
    }

    if !keep {
        let _ = std::fs::remove_file(&path);
    }

    Ok(())
}
