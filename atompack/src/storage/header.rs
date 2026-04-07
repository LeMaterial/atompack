use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) struct Header {
    pub(super) generation: u64,
    pub(super) data_start: u64,
    pub(super) num_molecules: u64,
    pub(super) compression: CompressionType,
    pub(super) record_format: u32,
    pub(super) index_offset: u64,
    pub(super) index_len: u64,
}

/// Simple corruption detector (not cryptographic). https://en.wikipedia.org/wiki/Adler-32
fn adler32(bytes: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in bytes {
        a = (a + (byte as u32)) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    (b << 16) | a
}

pub(super) fn encode_header_slot(header: Header) -> [u8; HEADER_SLOT_SIZE] {
    let mut slot = [0u8; HEADER_SLOT_SIZE];

    slot[0..4].copy_from_slice(MAGIC);
    slot[4..8].copy_from_slice(&FILE_FORMAT_VERSION.to_le_bytes());
    slot[8..16].copy_from_slice(&header.generation.to_le_bytes());
    slot[16..24].copy_from_slice(&header.data_start.to_le_bytes());
    slot[24..32].copy_from_slice(&header.index_offset.to_le_bytes());
    slot[32..40].copy_from_slice(&header.index_len.to_le_bytes());
    slot[40..48].copy_from_slice(&header.num_molecules.to_le_bytes());

    let (compression_type, compression_level) = match header.compression {
        CompressionType::None => (0u8, 0i32),
        CompressionType::Lz4 => (1u8, 0i32),
        CompressionType::Zstd(level) => (2u8, level),
    };
    slot[48] = compression_type;
    slot[52..56].copy_from_slice(&compression_level.to_le_bytes());
    slot[56..60].copy_from_slice(&header.record_format.to_le_bytes());

    let checksum = adler32(&slot[..HEADER_SLOT_SIZE - 4]);
    slot[HEADER_SLOT_SIZE - 4..HEADER_SLOT_SIZE].copy_from_slice(&checksum.to_le_bytes());

    slot
}

fn decode_header_slot(slot: &[u8; HEADER_SLOT_SIZE], file_size: u64) -> Option<Header> {
    if &slot[0..4] != MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(slot[4..8].try_into().ok()?);
    if version != FILE_FORMAT_VERSION {
        return None;
    }

    let expected_checksum = u32::from_le_bytes(
        slot[HEADER_SLOT_SIZE - 4..HEADER_SLOT_SIZE]
            .try_into()
            .ok()?,
    );
    let actual_checksum = adler32(&slot[..HEADER_SLOT_SIZE - 4]);
    if expected_checksum != actual_checksum {
        return None;
    }

    let generation = u64::from_le_bytes(slot[8..16].try_into().ok()?);
    let data_start = u64::from_le_bytes(slot[16..24].try_into().ok()?);
    let index_offset = u64::from_le_bytes(slot[24..32].try_into().ok()?);
    let index_len = u64::from_le_bytes(slot[32..40].try_into().ok()?);
    let num_molecules = u64::from_le_bytes(slot[40..48].try_into().ok()?);
    let compression_type = slot[48];
    let compression_level = i32::from_le_bytes(slot[52..56].try_into().ok()?);
    let record_format = u32::from_le_bytes(slot[56..60].try_into().ok()?);

    let compression = match compression_type {
        0 => CompressionType::None,
        1 => CompressionType::Lz4,
        2 => CompressionType::Zstd(compression_level),
        _ => return None,
    };

    if data_start < HEADER_REGION_SIZE || data_start > file_size {
        return None;
    }

    if index_offset == 0 || index_len == 0 {
        if num_molecules != 0 || index_offset != 0 || index_len != 0 {
            return None;
        }
    } else {
        let expected_index_len = (num_molecules as usize)
            .checked_mul(INDEX_ENTRY_SIZE)?
            .checked_add(INDEX_PREFIX_SIZE)? as u64;

        if index_len != expected_index_len {
            return None;
        }

        let end = index_offset.checked_add(index_len)?;
        if index_offset < data_start || end > file_size {
            return None;
        }
    }

    Some(Header {
        generation,
        data_start,
        num_molecules,
        compression,
        record_format,
        index_offset,
        index_len,
    })
}

pub(super) fn read_best_header(file: &mut File) -> Result<Header> {
    let file_size = file.metadata()?.len();
    file.seek(SeekFrom::Start(HEADER_SLOT_A_OFFSET))?;
    let mut slot_a = [0u8; HEADER_SLOT_SIZE];
    file.read_exact(&mut slot_a)?;

    file.seek(SeekFrom::Start(HEADER_SLOT_B_OFFSET))?;
    let mut slot_b = [0u8; HEADER_SLOT_SIZE];
    file.read_exact(&mut slot_b)?;

    let a = decode_header_slot(&slot_a, file_size);
    let b = decode_header_slot(&slot_b, file_size);

    match (a, b) {
        (Some(ha), Some(hb)) => Ok(if ha.generation >= hb.generation {
            ha
        } else {
            hb
        }),
        (Some(h), None) | (None, Some(h)) => Ok(h),
        (None, None) => Err(Error::InvalidData(
            "Invalid or corrupted header (no valid header slots)".into(),
        )),
    }
}
