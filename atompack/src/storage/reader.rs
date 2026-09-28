//! Read-only access over byte ranges.
//!
//! [`AtomReader`] reads only what it needs (header, index entries, records,
//! sections) from any [`ReadAt`] source, so it works on multi-GB files behind
//! slow or remote sources and on targets without files or mmap (WASM).

use super::*;
use std::ops::Range;

/// Random-access byte source: a file, an in-memory buffer, or a host callback.
pub trait ReadAt {
    fn size(&self) -> Result<u64>;
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()>;

    fn read_vec(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let len =
            usize::try_from(len).map_err(|_| Error::InvalidData("Read length overflow".into()))?;
        let mut buf = vec![0u8; len];
        self.read_exact_at(offset, &mut buf)?;
        Ok(buf)
    }
}

impl<T: ReadAt + ?Sized> ReadAt for &T {
    fn size(&self) -> Result<u64> {
        (**self).size()
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        (**self).read_exact_at(offset, buf)
    }
}

impl ReadAt for File {
    fn size(&self) -> Result<u64> {
        Ok(self.metadata()?.len())
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let mut file = self;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf)?;
        Ok(())
    }
}

impl ReadAt for [u8] {
    fn size(&self) -> Result<u64> {
        Ok(self.len() as u64)
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let src = usize::try_from(offset)
            .ok()
            .and_then(|start| self.get(start..start.checked_add(buf.len())?))
            .ok_or_else(|| Error::InvalidData("Read out of bounds".into()))?;
        buf.copy_from_slice(src);
        Ok(())
    }
}

/// Everything a reader needs from the committed part of a file.
pub(super) struct Committed {
    pub(super) header: Header,
    /// End of the committed data; bytes past it are an uncommitted tail.
    pub(super) end: u64,
    pub(super) schema_lock: Option<SchemaLock>,
    pub(super) extensions: Extensions,
}

pub(super) fn read_committed(src: &(impl ReadAt + ?Sized)) -> Result<Committed> {
    // [magic][u32 version LE] first, for a clearer error than a bad header.
    let mut prefix = [0u8; 8];
    src.read_exact_at(0, &mut prefix)?;
    if &prefix[0..4] != MAGIC {
        return Err(Error::InvalidData("Invalid file format".into()));
    }
    let version = u32::from_le_bytes(arr(&prefix[4..8])?);
    if version != FILE_FORMAT_VERSION {
        return Err(Error::InvalidData(format!(
            "Unsupported file format version {} (expected {})",
            version, FILE_FORMAT_VERSION
        )));
    }

    let header = read_best_header(src)?;
    if header.record_format != RECORD_FORMAT_SOA_V2 && header.record_format != RECORD_FORMAT_SOA_V3
    {
        return Err(Error::InvalidData(format!(
            "Unsupported record format {}.",
            header.record_format
        )));
    }

    let end = if header.index_offset == 0 || header.index_len == 0 {
        header.data_start
    } else {
        header
            .index_offset
            .checked_add(header.index_len)
            .ok_or_else(|| Error::InvalidData("Index end overflow".into()))?
    };

    let schema_lock = if header.schema_offset > 0 && header.schema_len > 0 {
        Some(decode_schema_lock(
            &src.read_vec(header.schema_offset, header.schema_len)?,
        )?)
    } else {
        None
    };

    let extensions = Extensions::open(
        src,
        (header.extensions_offset, header.extensions_len),
        header.data_start,
        end,
    )?;

    Ok(Committed {
        header,
        end,
        schema_lock,
        extensions,
    })
}

/// Read-only database over any [`ReadAt`] source.
pub struct AtomReader<R> {
    src: R,
    committed: Committed,
}

impl<R: ReadAt> AtomReader<R> {
    pub fn open(src: R) -> Result<Self> {
        let committed = read_committed(&src)?;
        Ok(Self { src, committed })
    }

    pub fn len(&self) -> usize {
        self.committed.header.num_molecules as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn compression(&self) -> CompressionType {
        self.committed.header.compression
    }

    pub fn record_format(&self) -> u32 {
        self.committed.header.record_format
    }

    pub fn schema_info(&self) -> Option<DatabaseSchema> {
        self.committed
            .schema_lock
            .as_ref()
            .map(database_schema_from_lock)
    }

    /// Atom counts for a range of molecules (index only, no record reads).
    pub fn num_atoms(&self, range: Range<usize>) -> Result<Vec<u32>> {
        Ok(self.entries(range)?.iter().map(|e| e.num_atoms).collect())
    }

    pub fn get_molecule(&self, index: usize) -> Result<Molecule> {
        let entry = self.entries(index..index.saturating_add(1))?[0];
        self.decode(entry)
    }

    pub fn get_molecules(&self, range: Range<usize>) -> Result<Vec<Molecule>> {
        self.entries(range)?
            .into_iter()
            .map(|entry| self.decode(entry))
            .collect()
    }

    /// All groupings by name. Decoded from the source on first access.
    pub fn groups(&self) -> Result<&BTreeMap<String, Grouping>> {
        self.committed
            .extensions
            .groups(self.len(), |offset, len| self.src.read_vec(offset, len))
    }

    fn entries(&self, range: Range<usize>) -> Result<Vec<MoleculeIndex>> {
        if range.start > range.end || range.end > self.len() {
            return Err(Error::InvalidData(format!(
                "Range {:?} out of bounds for database of length {}",
                range,
                self.len()
            )));
        }
        let offset = self.committed.header.index_offset
            + (INDEX_PREFIX_SIZE + range.start * INDEX_ENTRY_SIZE) as u64;
        let bytes = self
            .src
            .read_vec(offset, (range.len() * INDEX_ENTRY_SIZE) as u64)?;
        Ok(decode_index_entries(&bytes))
    }

    fn decode(&self, entry: MoleculeIndex) -> Result<Molecule> {
        let compressed = self
            .src
            .read_vec(entry.offset, entry.compressed_size as u64)?;
        let bytes = decompress(
            &compressed,
            self.compression(),
            Some(entry.uncompressed_size as usize),
        )?;
        let positions_type = self
            .committed
            .schema_lock
            .as_ref()
            .and_then(|lock| lock.positions_type);
        deserialize_molecule_soa(&bytes, self.record_format(), positions_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FloatScalarData, PropertyValue, Vec3Data};
    use tempfile::NamedTempFile;

    fn molecule(i: usize) -> Molecule {
        let n = i % 3 + 1;
        let mut mol = Molecule::new(vec![[i as f32, 0.5, -1.0]; n], vec![6 + i as u8; n]).unwrap();
        mol.energy = Some(FloatScalarData::F64(-(i as f64)));
        mol.forces = Some(Vec3Data::F32(vec![[0.1, 0.2, 0.3]; n]));
        mol.properties
            .insert("tag".into(), PropertyValue::String(format!("m{i}")));
        mol
    }

    // Molecule has no PartialEq; compare a deterministic rendering.
    fn render(mol: &Molecule) -> String {
        let props: BTreeMap<_, _> = mol.properties.iter().collect();
        format!(
            "{:?} {:?} {:?} {:?} {:?}",
            mol.positions, mol.atomic_numbers, mol.energy, mol.forces, props
        )
    }

    fn check<R: ReadAt>(reader: &AtomReader<R>, db: &mut AtomDatabase, groups: &Grouping) {
        assert_eq!(reader.len(), 6);
        assert_eq!(reader.num_atoms(1..4).unwrap(), vec![2, 3, 1]);
        for (i, mol) in reader.get_molecules(0..6).unwrap().iter().enumerate() {
            assert_eq!(render(mol), render(&db.get_molecule(i).unwrap()));
        }
        assert_eq!(
            render(&reader.get_molecule(5).unwrap()),
            render(&molecule(5))
        );
        assert_eq!(&reader.groups().unwrap()["pairs"], groups);
        assert!(reader.get_molecule(6).is_err());
        assert!(reader.num_atoms(5..7).is_err());
    }

    #[test]
    fn reader_matches_database_for_file_and_bytes() {
        for compression in [
            CompressionType::None,
            CompressionType::Lz4,
            CompressionType::Zstd(3),
        ] {
            let temp = NamedTempFile::new().unwrap();
            let mut db = AtomDatabase::create(temp.path(), compression).unwrap();
            let first: Vec<_> = (0..5).map(molecule).collect();
            db.add_molecules(&first.iter().collect::<Vec<_>>()).unwrap();
            db.flush().unwrap();
            // Second session: its record lands after the first index.
            let mut db = AtomDatabase::open(temp.path()).unwrap();
            db.add_molecule(&molecule(5)).unwrap();
            let groups = Grouping {
                roles: vec!["a".into(), "b".into()],
                offsets: vec![0, 2],
                records: vec![5, 0],
                member_roles: vec![0, 1],
                properties: vec![("x".into(), GroupColumn::Float(vec![1.5]))],
            };
            db.add_groups("pairs", groups.clone()).unwrap();
            db.flush().unwrap();
            // Uncommitted tail must be ignored.
            db.add_molecule(&molecule(6)).unwrap();

            let bytes = std::fs::read(temp.path()).unwrap();
            let mut db = AtomDatabase::open(temp.path()).unwrap();
            let file = AtomReader::open(File::open(temp.path()).unwrap()).unwrap();
            check(&file, &mut db, &groups);
            check(
                &AtomReader::open(bytes.as_slice()).unwrap(),
                &mut db,
                &groups,
            );
        }
    }

    #[test]
    fn reader_rejects_non_atompack_bytes() {
        let bytes = vec![0u8; 2 * HEADER_SLOT_SIZE];
        let err = AtomReader::open(bytes.as_slice()).err().unwrap();
        assert!(err.to_string().contains("Invalid file format"));
    }
}
