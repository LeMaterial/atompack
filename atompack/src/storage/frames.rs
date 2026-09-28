//! Framed entries in the data region.
//!
//! Everything written after the header region is a frame:
//!
//! ```text
//! [kind:u32][len:u32][uncompressed_len:u32][n_atoms:u32][crc32:u32][payload: len bytes]
//! ```
//!
//! `crc32` covers the first 16 header bytes and the payload. Index entries
//! point at record payloads, so readers never see frame headers. The index
//! and extensions directory written by `flush` after the last frame are a
//! cache: after an interrupted write, `open` rebuilds them by scanning frames
//! up to the first incomplete or corrupted one.

use super::*;
use std::io::BufReader;

pub(super) const FRAME_HEADER_SIZE: usize = 20;
pub(super) const FRAME_RECORD: u32 = 1;
/// Payload: `[record_format:u32][schema lock blob]`.
pub(super) const FRAME_SCHEMA: u32 = 2;
pub(super) const FRAME_GROUPS: u32 = 3;

/// A frame other than a record: its kind and payload span in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MetaFrame {
    pub(super) kind: u32,
    pub(super) offset: u64,
    pub(super) len: u64,
}

pub(super) fn frame_header(
    kind: u32,
    payload: &[u8],
    uncompressed_len: u32,
    n_atoms: u32,
) -> [u8; FRAME_HEADER_SIZE] {
    let mut header = [0u8; FRAME_HEADER_SIZE];
    header[0..4].copy_from_slice(&kind.to_le_bytes());
    header[4..8].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    header[8..12].copy_from_slice(&uncompressed_len.to_le_bytes());
    header[12..16].copy_from_slice(&n_atoms.to_le_bytes());
    let crc = frame_crc(&header, payload);
    header[16..20].copy_from_slice(&crc.to_le_bytes());
    header
}

fn frame_crc(header: &[u8; FRAME_HEADER_SIZE], payload: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&header[..16]);
    hasher.update(payload);
    hasher.finalize()
}

pub(super) fn encode_schema_payload(record_format: u32, lock: &SchemaLock) -> Result<Vec<u8>> {
    let mut payload = record_format.to_le_bytes().to_vec();
    payload.extend(encode_schema_lock(lock)?);
    Ok(payload)
}

/// Metadata recovered by scanning frames.
#[derive(Debug, Default)]
pub(super) struct Scan {
    pub(super) index: Vec<MoleculeIndex>,
    /// The latest schema frame.
    pub(super) schema: Option<MetaFrame>,
    /// Other non-record frames, in file order.
    pub(super) metadata: Vec<MetaFrame>,
    /// End of the last valid frame.
    pub(super) end: u64,
}

/// Read frames from `start` until the first one that is incomplete or fails
/// its checksum; everything after it is an interrupted write.
pub(super) fn scan_frames(path: &Path, start: u64) -> Result<Scan> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(1 << 22, file);
    reader.seek(SeekFrom::Start(start))?;

    let mut scan = Scan {
        end: start,
        ..Scan::default()
    };
    let mut header = [0u8; FRAME_HEADER_SIZE];
    let mut payload = Vec::new();
    while scan.end + FRAME_HEADER_SIZE as u64 <= file_len {
        reader.read_exact(&mut header)?;
        let field = |i: usize| u32::from_le_bytes(header[i..i + 4].try_into().unwrap());
        let (kind, len, uncompressed_len, n_atoms) = (field(0), field(4), field(8), field(12));
        let offset = scan.end + FRAME_HEADER_SIZE as u64;
        if offset + len as u64 > file_len {
            break;
        }
        payload.resize(len as usize, 0);
        reader.read_exact(&mut payload)?;
        if frame_crc(&header, &payload) != field(16) {
            break;
        }
        match kind {
            FRAME_RECORD => scan.index.push(MoleculeIndex {
                offset,
                compressed_size: len,
                uncompressed_size: uncompressed_len,
                num_atoms: n_atoms,
            }),
            _ => {
                let frame = MetaFrame {
                    kind,
                    offset,
                    len: len as u64,
                };
                if kind == FRAME_SCHEMA {
                    scan.schema = Some(frame);
                } else {
                    scan.metadata.push(frame);
                }
            }
        }
        scan.end = offset + len as u64;
    }
    Ok(scan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec3Data;
    use tempfile::NamedTempFile;

    fn molecule(x: f32) -> Molecule {
        Molecule::new(vec![[x, 0.0, 0.0]], vec![6]).unwrap()
    }

    fn add(db: &mut AtomDatabase, xs: std::ops::Range<usize>) {
        let molecules: Vec<Molecule> = xs.map(|x| molecule(x as f32)).collect();
        db.add_molecules(&molecules.iter().collect::<Vec<_>>())
            .unwrap();
    }

    fn xs(db: &AtomDatabase) -> Vec<f32> {
        let all: Vec<usize> = (0..db.len()).collect();
        db.get_molecules(&all)
            .unwrap()
            .iter()
            .map(|m| m.atom(0).unwrap().position()[0])
            .collect()
    }

    /// (index_offset, num_molecules, flags) of both header slots.
    fn slots(path: &Path) -> [(u64, u64, u32); 2] {
        let bytes = std::fs::read(path).unwrap();
        let slot = |at: usize| {
            let u64_at =
                |i: usize| u64::from_le_bytes(bytes[at + i..at + i + 8].try_into().unwrap());
            let flags = u32::from_le_bytes(bytes[at + 92..at + 96].try_into().unwrap());
            (u64_at(24), u64_at(40), flags)
        };
        [slot(0), slot(HEADER_SLOT_SIZE)]
    }

    fn pairs() -> Grouping {
        Grouping {
            offsets: vec![0, 2],
            records: vec![0, 4],
            ..Grouping::default()
        }
    }

    #[test]
    fn unflushed_writes_survive_a_crash() {
        let temp = NamedTempFile::new().unwrap();
        let mut db = AtomDatabase::create(temp.path(), CompressionType::Zstd(3)).unwrap();
        add(&mut db, 0..3);
        db.flush().unwrap();
        add(&mut db, 3..5);
        db.add_groups("pairs", pairs()).unwrap();
        drop(db); // crash: no flush

        // Both slots are "writing" and invalid for older versions, which
        // require an index whenever the molecule count is non-zero.
        for (index_offset, count, flags) in slots(temp.path()) {
            assert_eq!((index_offset, count), (0, u64::MAX));
            assert_eq!(flags, HEADER_FRAMED | HEADER_WRITING);
        }

        let db = AtomDatabase::open_mmap(temp.path()).unwrap();
        assert_eq!(xs(&db), [0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(db.groups().unwrap()["pairs"], pairs());

        let mut db = AtomDatabase::open(temp.path()).unwrap();
        add(&mut db, 5..6);
        db.flush().unwrap();
        let db = AtomDatabase::open(temp.path()).unwrap();
        assert_eq!(xs(&db), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(db.groups().unwrap()["pairs"], pairs());
        assert_eq!(
            slots(temp.path())
                .iter()
                .filter(|s| s.2 == HEADER_FRAMED)
                .count(),
            1
        );
    }

    #[test]
    fn torn_and_corrupted_frames_end_the_recovery() {
        let temp = NamedTempFile::new().unwrap();
        let mut db = AtomDatabase::create(temp.path(), CompressionType::None).unwrap();
        add(&mut db, 0..3);
        db.flush().unwrap();
        add(&mut db, 3..5);
        let fourth_payload = match &db.index {
            IndexStorage::InMemory(entries) => entries[3].offset,
            IndexStorage::MemoryMapped { .. } => unreachable!(),
        };
        drop(db);

        let file = OpenOptions::new().write(true).open(temp.path()).unwrap();
        let full = file.metadata().unwrap().len();
        file.set_len(full - 3).unwrap(); // the last frame is cut short
        assert_eq!(
            xs(&AtomDatabase::open(temp.path()).unwrap()),
            [0.0, 1.0, 2.0, 3.0]
        );

        // Flip one payload byte of the 4th record: its checksum fails.
        let mut bytes = std::fs::read(temp.path()).unwrap();
        bytes[fourth_payload as usize] ^= 0xFF;
        std::fs::write(temp.path(), &bytes).unwrap();
        let mut db = AtomDatabase::open(temp.path()).unwrap();
        assert_eq!(xs(&db), [0.0, 1.0, 2.0]);

        // Writing continues after the last valid frame.
        add(&mut db, 7..8);
        db.flush().unwrap();
        assert_eq!(
            xs(&AtomDatabase::open(temp.path()).unwrap()),
            [0.0, 1.0, 2.0, 7.0]
        );
    }

    #[test]
    fn frequent_flushes_leave_no_dead_bytes() {
        let once = NamedTempFile::new().unwrap();
        let mut db = AtomDatabase::create(once.path(), CompressionType::Zstd(3)).unwrap();
        add(&mut db, 0..50);
        db.flush().unwrap();

        let often = NamedTempFile::new().unwrap();
        let mut db = AtomDatabase::create(often.path(), CompressionType::Zstd(3)).unwrap();
        for x in 0..50 {
            add(&mut db, x..x + 1);
            db.flush().unwrap();
        }

        let size = |f: &NamedTempFile| f.as_file().metadata().unwrap().len();
        assert_eq!(size(&often), size(&once));
        assert_eq!(xs(&AtomDatabase::open(often.path()).unwrap()), xs(&db));
    }

    #[test]
    fn flush_without_changes_writes_nothing() {
        let temp = NamedTempFile::new().unwrap();
        let mut db = AtomDatabase::create(temp.path(), CompressionType::None).unwrap();
        add(&mut db, 0..2);
        db.flush().unwrap();
        let before = std::fs::read(temp.path()).unwrap();
        AtomDatabase::open(temp.path()).unwrap().flush().unwrap();
        assert_eq!(std::fs::read(temp.path()).unwrap(), before);
    }

    #[test]
    fn recovery_restores_the_schema() {
        let temp = NamedTempFile::new().unwrap();
        let mut db = AtomDatabase::create(temp.path(), CompressionType::None).unwrap();
        let mut mol = Molecule::new_f64(vec![[0.1, 0.2, 0.3]], vec![6]).unwrap();
        mol.forces = Some(Vec3Data::F64(vec![[1.0, 2.0, 3.0]]));
        db.add_molecule(&mol).unwrap();
        let (format, positions) = (db.record_format(), db.positions_type());
        drop(db); // crash before the first flush

        let mut db = AtomDatabase::open(temp.path()).unwrap();
        assert_eq!(
            (db.record_format(), db.positions_type()),
            (format, positions)
        );
        let got = db.get_molecule(0).unwrap();
        assert_eq!((got.positions, got.forces), (mol.positions, mol.forces));
    }

    #[test]
    fn unframed_files_keep_appending_after_their_commit() {
        let temp = NamedTempFile::new().unwrap();
        let mut db = AtomDatabase::create(temp.path(), CompressionType::None).unwrap();
        add(&mut db, 0..2);
        db.flush().unwrap();

        // Older versions rewrite the header without the framed flag.
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(temp.path())
            .unwrap();
        let mut header = read_best_header(&mut file).unwrap();
        header.flags = 0;
        header.data_end = 0;
        for generation in [header.generation + 1, header.generation + 2] {
            AtomDatabase::write_header(
                &mut file,
                Header {
                    generation,
                    ..header
                },
            )
            .unwrap();
        }
        let committed = file.metadata().unwrap().len();

        let mut db = AtomDatabase::open(temp.path()).unwrap();
        add(&mut db, 2..3);
        assert!(slots(temp.path()).iter().all(|s| s.2 == 0)); // no writing state
        db.flush().unwrap();

        // The old index stays in place; everything is readable.
        assert!(temp.as_file().metadata().unwrap().len() > committed);
        assert_eq!(
            xs(&AtomDatabase::open(temp.path()).unwrap()),
            [0.0, 1.0, 2.0]
        );
    }
}
