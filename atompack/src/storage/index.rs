use super::*;

pub(super) fn encode_index(entries: &[MoleculeIndex]) -> Vec<u8> {
    let mut bytes =
        Vec::with_capacity(INDEX_PREFIX_SIZE + entries.len().saturating_mul(INDEX_ENTRY_SIZE));
    bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for entry in entries {
        bytes.extend_from_slice(&entry.offset.to_le_bytes());
        bytes.extend_from_slice(&entry.compressed_size.to_le_bytes());
        bytes.extend_from_slice(&entry.uncompressed_size.to_le_bytes());
        bytes.extend_from_slice(&entry.num_atoms.to_le_bytes());
    }
    bytes
}

pub(super) fn decode_index(bytes: &[u8]) -> Result<Vec<MoleculeIndex>> {
    if bytes.len() < INDEX_PREFIX_SIZE {
        return Err(Error::InvalidData("Index too small".into()));
    }

    let count = u64::from_le_bytes(arr(&bytes[0..8])?) as usize;
    let expected_len = INDEX_PREFIX_SIZE
        .checked_add(
            count
                .checked_mul(INDEX_ENTRY_SIZE)
                .ok_or_else(|| Error::InvalidData("Index length overflow".into()))?,
        )
        .ok_or_else(|| Error::InvalidData("Index length overflow".into()))?;

    if bytes.len() != expected_len {
        return Err(Error::InvalidData(format!(
            "Index length mismatch (expected {}, got {})",
            expected_len,
            bytes.len()
        )));
    }

    Ok(decode_index_entries(&bytes[INDEX_PREFIX_SIZE..]))
}

/// Decode consecutive index entries (no count prefix).
pub(super) fn decode_index_entries(bytes: &[u8]) -> Vec<MoleculeIndex> {
    bytes
        .as_chunks::<INDEX_ENTRY_SIZE>()
        .0
        .iter()
        .map(MoleculeIndex::from_le_bytes)
        .collect()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C, packed)]
pub(super) struct MoleculeIndex {
    pub(super) offset: u64,
    pub(super) compressed_size: u32,
    pub(super) uncompressed_size: u32,
    pub(super) num_atoms: u32,
}

impl MoleculeIndex {
    fn from_le_bytes(b: &[u8; INDEX_ENTRY_SIZE]) -> Self {
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        Self {
            offset: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            compressed_size: u32_at(8),
            uncompressed_size: u32_at(12),
            num_atoms: u32_at(16),
        }
    }
}

#[derive(Debug)]
pub(super) enum IndexStorage {
    InMemory(Vec<MoleculeIndex>),
    MemoryMapped {
        mmap: Mmap,
        index_offset: u64,
        count: usize,
    },
}

impl IndexStorage {
    pub(super) fn len(&self) -> usize {
        match self {
            IndexStorage::InMemory(vec) => vec.len(),
            IndexStorage::MemoryMapped { count, .. } => *count,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn get(&self, index: usize) -> Option<MoleculeIndex> {
        match self {
            IndexStorage::InMemory(vec) => vec.get(index).copied(),
            IndexStorage::MemoryMapped {
                mmap,
                index_offset,
                count,
            } => {
                if index >= *count {
                    return None;
                }

                let index_start = *index_offset as usize;
                let entry_offset = index_start + INDEX_PREFIX_SIZE + (index * INDEX_ENTRY_SIZE);
                if entry_offset + INDEX_ENTRY_SIZE > mmap.len() {
                    return None;
                }

                let entry_bytes = &mmap[entry_offset..entry_offset + INDEX_ENTRY_SIZE];
                Some(MoleculeIndex::from_le_bytes(entry_bytes.try_into().ok()?))
            }
        }
    }

    pub(super) fn extend(&mut self, new_indices: Vec<MoleculeIndex>) -> Result<()> {
        match self {
            IndexStorage::InMemory(vec) => {
                vec.extend(new_indices);
                Ok(())
            }
            IndexStorage::MemoryMapped { .. } => Err(Error::InvalidData(
                "Cannot add molecules to a database opened with a memory-mapped index (read-only); reopen without mmap to write."
                    .into(),
            )),
        }
    }
}
