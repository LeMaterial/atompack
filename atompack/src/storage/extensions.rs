//! Metadata frames other than records and schema (grouped records, ...).
//!
//! A group references related records (e.g. adsorbate+slab, slab, gas) by
//! index; records are stored once and may belong to any number of groups.
//! Each `add_groups` call is written immediately as one groups frame (a
//! chunk); a grouping is the concatenation of its chunks in file order.
//! Chunk payload:
//!
//! ```text
//! [version:u32][n_groupings:u32]
//! per grouping:
//!   [name:str][n_roles:u32][roles:str...]
//!   [n_groups:u64][n_members:u64]
//!   [offsets:u64 × (n_groups + 1)]      CSR: group g = members[offsets[g]..offsets[g+1]]
//!   [records:u64 × n_members]
//!   [member_roles:u16 × n_members]      only when n_roles > 0
//!   [n_props:u32] per prop: [key:str][type_tag:u8][values × n_groups]
//! str = [len:u32][utf8 bytes]
//! ```
//!
//! `flush` writes a directory of these frames (`[version:u32][count:u32]`,
//! then `[kind:u32][offset:u64][len:u64]` per frame), referenced from the
//! header. Frames of unknown kinds are listed and kept, so newer sections
//! survive older writers.

use super::frames::{FRAME_GROUPS, MetaFrame};
use super::*;
use std::collections::BTreeMap;
use std::sync::OnceLock;

const DIRECTORY_VERSION: u32 = 1;
const GROUPS_VERSION: u32 = 1;

/// Metadata frames in file order, and the groupings decoded from them on
/// first access.
#[derive(Debug, Default)]
pub(super) struct Extensions {
    frames: Vec<MetaFrame>,
    groups: OnceLock<BTreeMap<String, Grouping>>,
}

impl Extensions {
    pub(super) fn from_frames(frames: Vec<MetaFrame>) -> Self {
        Self {
            frames,
            ..Self::default()
        }
    }

    /// Read the directory at `span`; every listed frame must lie before it.
    pub(super) fn open(file: &mut File, span: (u64, u64), data_start: u64) -> Result<Self> {
        let (offset, len) = span;
        if len == 0 {
            return Ok(Self::default());
        }
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0u8; len as usize];
        file.read_exact(&mut bytes)?;
        let frames = decode_directory(&bytes)?;
        for frame in &frames {
            let end = frame.offset.checked_add(frame.len);
            if frame.offset < data_start || end.is_none_or(|end| end > offset) {
                return Err(Error::InvalidData(format!(
                    "Metadata frame of kind {} is out of bounds",
                    frame.kind
                )));
            }
        }
        Ok(Self::from_frames(frames))
    }

    /// The directory `flush` writes, or `None` when there are no frames.
    pub(super) fn encode_directory(&self) -> Option<Vec<u8>> {
        if self.frames.is_empty() {
            return None;
        }
        let mut buf = Vec::with_capacity(8 + self.frames.len() * 20);
        buf.extend_from_slice(&DIRECTORY_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.frames.len() as u32).to_le_bytes());
        for frame in &self.frames {
            buf.extend_from_slice(&frame.kind.to_le_bytes());
            buf.extend_from_slice(&frame.offset.to_le_bytes());
            buf.extend_from_slice(&frame.len.to_le_bytes());
        }
        Some(buf)
    }

    /// All groupings; `read(offset, len)` loads each chunk on first access.
    pub(super) fn groups(
        &self,
        num_records: usize,
        read: impl Fn(u64, u64) -> Result<Vec<u8>>,
    ) -> Result<&BTreeMap<String, Grouping>> {
        if let Some(groups) = self.groups.get() {
            return Ok(groups);
        }
        let mut groups = BTreeMap::new();
        for frame in self.frames.iter().filter(|f| f.kind == FRAME_GROUPS) {
            for (name, chunk) in decode_groups(&read(frame.offset, frame.len)?, num_records)? {
                merge_grouping(&mut groups, name, chunk)?;
            }
        }
        Ok(self.groups.get_or_init(|| groups))
    }

    /// Check `groups` can be added to the grouping `name` and encode them as
    /// a chunk. Pass the written chunk to `push_groups` afterwards.
    pub(super) fn encode_groups_chunk(
        &self,
        name: &str,
        groups: &Grouping,
        num_records: usize,
        read: impl Fn(u64, u64) -> Result<Vec<u8>>,
    ) -> Result<Vec<u8>> {
        groups.validate(num_records)?;
        if let Some(existing) = self.groups(num_records, read)?.get(name) {
            existing.check_append(groups)?;
        }
        Ok(encode_groups(&[(name, groups)]))
    }

    /// Record a chunk written for `encode_groups_chunk`.
    pub(super) fn push_groups(&mut self, name: &str, groups: Grouping, frame: MetaFrame) {
        self.frames.push(frame);
        let all = self
            .groups
            .get_mut()
            .expect("loaded by encode_groups_chunk");
        merge_grouping(all, name.to_string(), groups).expect("checked by encode_groups_chunk");
    }
}

fn merge_grouping(
    groups: &mut BTreeMap<String, Grouping>,
    name: String,
    chunk: Grouping,
) -> Result<()> {
    match groups.get_mut(&name) {
        Some(existing) => existing.append(chunk),
        None => {
            groups.insert(name, chunk);
            Ok(())
        }
    }
}

/// Per-group property values, one entry per group.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupColumn {
    Float(Vec<f64>),
    Int(Vec<i64>),
    String(Vec<String>),
}

impl GroupColumn {
    pub fn len(&self) -> usize {
        match self {
            Self::Float(v) => v.len(),
            Self::Int(v) => v.len(),
            Self::String(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn type_tag(&self) -> u8 {
        match self {
            Self::Float(_) => TYPE_FLOAT,
            Self::Int(_) => TYPE_INT,
            Self::String(_) => TYPE_STRING,
        }
    }

    fn extend(&mut self, other: GroupColumn) {
        match (self, other) {
            (Self::Float(a), Self::Float(b)) => a.extend(b),
            (Self::Int(a), Self::Int(b)) => a.extend(b),
            (Self::String(a), Self::String(b)) => a.extend(b),
            _ => unreachable!("column types are checked before extending"),
        }
    }
}

/// Groups of record indices in CSR layout.
///
/// Members of group `g` are `records[offsets[g]..offsets[g + 1]]`. With named
/// roles, `member_roles[i]` indexes `roles`; ordered groups leave both empty.
#[derive(Debug, Clone, PartialEq)]
pub struct Grouping {
    pub roles: Vec<String>,
    pub offsets: Vec<u64>,
    pub records: Vec<u64>,
    pub member_roles: Vec<u16>,
    pub properties: Vec<(String, GroupColumn)>,
}

impl Default for Grouping {
    fn default() -> Self {
        Self {
            roles: Vec::new(),
            offsets: vec![0],
            records: Vec::new(),
            member_roles: Vec::new(),
            properties: Vec::new(),
        }
    }
}

impl Grouping {
    /// Number of groups.
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Member range of group `index` in `records` / `member_roles`.
    pub fn member_range(&self, index: usize) -> Option<std::ops::Range<usize>> {
        let start = *self.offsets.get(index)? as usize;
        let end = *self.offsets.get(index + 1)? as usize;
        Some(start..end)
    }

    /// Check all invariants; `num_records` bounds the referenced indices.
    pub(super) fn validate(&self, num_records: usize) -> Result<()> {
        let invalid = |msg: String| Err(Error::InvalidData(msg));
        if self.offsets.first() != Some(&0) {
            return invalid("Group offsets must start at 0".into());
        }
        if self.offsets.windows(2).any(|w| w[0] >= w[1]) {
            return invalid("Every group must have at least one member".into());
        }
        if self.offsets.last() != Some(&(self.records.len() as u64)) {
            return invalid("Group offsets do not match the member count".into());
        }
        if let Some(&record) = self.records.iter().find(|&&r| r >= num_records as u64) {
            return invalid(format!(
                "Group member {} out of bounds for database of length {}",
                record, num_records
            ));
        }
        if self.roles.is_empty() {
            if !self.member_roles.is_empty() {
                return invalid("Ordered groups cannot have member roles".into());
            }
        } else {
            if self.roles.len() > u16::MAX as usize {
                return invalid("Too many group roles".into());
            }
            if self.member_roles.len() != self.records.len() {
                return invalid("Every group member needs a role".into());
            }
            let mut seen = vec![usize::MAX; self.roles.len()];
            for g in 0..self.len() {
                for &role in &self.member_roles[self.member_range(g).unwrap()] {
                    let slot = seen.get_mut(role as usize).ok_or_else(|| {
                        Error::InvalidData(format!("Group role id {} out of bounds", role))
                    })?;
                    if *slot == g {
                        return invalid(format!(
                            "Group {} has role '{}' more than once",
                            g, self.roles[role as usize]
                        ));
                    }
                    *slot = g;
                }
            }
        }
        check_unique(self.roles.iter(), "group role")?;
        check_unique(self.properties.iter().map(|(k, _)| k), "group property")?;
        for (key, column) in &self.properties {
            if column.len() != self.len() {
                return invalid(format!(
                    "Group property '{}' has {} values for {} groups",
                    key,
                    column.len(),
                    self.len()
                ));
            }
        }
        Ok(())
    }

    /// Check `other` can be appended: same kind of groups, same property
    /// keys and types, and room for its new roles.
    pub(super) fn check_append(&self, other: &Grouping) -> Result<()> {
        if self.roles.is_empty() != other.roles.is_empty() {
            return Err(Error::InvalidData(
                "Cannot mix named-role and ordered groups in one grouping".into(),
            ));
        }
        let mut own_keys: Vec<_> = self
            .properties
            .iter()
            .map(|(k, c)| (k, c.type_tag()))
            .collect();
        let mut new_keys: Vec<_> = other
            .properties
            .iter()
            .map(|(k, c)| (k, c.type_tag()))
            .collect();
        own_keys.sort();
        new_keys.sort();
        if own_keys != new_keys {
            return Err(Error::InvalidData(format!(
                "Group properties {:?} do not match the existing grouping's {:?}",
                new_keys, own_keys
            )));
        }
        let new_roles = other
            .roles
            .iter()
            .filter(|r| !self.roles.contains(r))
            .count();
        if self.roles.len() + new_roles > u16::MAX as usize {
            return Err(Error::InvalidData("Too many group roles".into()));
        }
        Ok(())
    }

    /// Append validated groups. Nothing is modified if they are incompatible.
    pub(super) fn append(&mut self, mut other: Grouping) -> Result<()> {
        self.check_append(&other)?;
        let role_map: Vec<u16> = other
            .roles
            .iter()
            .map(|role| match self.roles.iter().position(|r| r == role) {
                Some(id) => id as u16,
                None => {
                    self.roles.push(role.clone());
                    (self.roles.len() - 1) as u16
                }
            })
            .collect();
        self.member_roles
            .extend(other.member_roles.iter().map(|&r| role_map[r as usize]));
        let base = self.records.len() as u64;
        self.offsets
            .extend(other.offsets[1..].iter().map(|o| o + base));
        self.records.append(&mut other.records);
        for (key, column) in other.properties {
            let own = self.properties.iter_mut().find(|(k, _)| *k == key).unwrap();
            own.1.extend(column);
        }
        Ok(())
    }
}

fn check_unique<'a>(names: impl Iterator<Item = &'a String>, what: &str) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(Error::InvalidData(format!("Duplicate {} '{}'", what, name)));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

fn put_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn encode_groups(groups: &[(&str, &Grouping)]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GROUPS_VERSION.to_le_bytes());
    buf.extend_from_slice(&(groups.len() as u32).to_le_bytes());
    for &(name, g) in groups {
        put_str(&mut buf, name);
        buf.extend_from_slice(&(g.roles.len() as u32).to_le_bytes());
        for role in &g.roles {
            put_str(&mut buf, role);
        }
        buf.extend_from_slice(&(g.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(g.records.len() as u64).to_le_bytes());
        g.offsets
            .iter()
            .for_each(|v| buf.extend_from_slice(&v.to_le_bytes()));
        g.records
            .iter()
            .for_each(|v| buf.extend_from_slice(&v.to_le_bytes()));
        g.member_roles
            .iter()
            .for_each(|v| buf.extend_from_slice(&v.to_le_bytes()));
        buf.extend_from_slice(&(g.properties.len() as u32).to_le_bytes());
        for (key, column) in &g.properties {
            put_str(&mut buf, key);
            buf.push(column.type_tag());
            match column {
                GroupColumn::Float(v) => v
                    .iter()
                    .for_each(|x| buf.extend_from_slice(&x.to_le_bytes())),
                GroupColumn::Int(v) => v
                    .iter()
                    .for_each(|x| buf.extend_from_slice(&x.to_le_bytes())),
                GroupColumn::String(v) => v.iter().for_each(|s| put_str(&mut buf, s)),
            }
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Bounds-checked cursor; every count is checked against the remaining bytes
/// before allocating, so corrupted lengths fail instead of exhausting memory.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    what: &'static str,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| Error::InvalidData(format!("{} truncated", self.what)))?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(arr(self.take(4)?)?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(arr(self.take(8)?)?))
    }

    fn count(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?)
            .map_err(|_| Error::InvalidData(format!("{} count overflow", self.what)))
    }

    fn str(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        std::str::from_utf8(self.take(len)?)
            .map(str::to_owned)
            .map_err(|_| Error::InvalidData(format!("Invalid UTF-8 in {}", self.what)))
    }

    fn array<const N: usize, T>(&mut self, n: usize, from: fn([u8; N]) -> T) -> Result<Vec<T>> {
        let len = n
            .checked_mul(N)
            .ok_or_else(|| Error::InvalidData(format!("{} count overflow", self.what)))?;
        Ok(self
            .take(len)?
            .as_chunks::<N>()
            .0
            .iter()
            .map(|&c| from(c))
            .collect())
    }

    fn finish(self) -> Result<()> {
        if self.pos != self.bytes.len() {
            return Err(Error::InvalidData(format!(
                "{} has trailing bytes",
                self.what
            )));
        }
        Ok(())
    }
}

fn decode_directory(bytes: &[u8]) -> Result<Vec<MetaFrame>> {
    let mut r = Reader {
        bytes,
        pos: 0,
        what: "Metadata directory",
    };
    let version = r.u32()?;
    if version != DIRECTORY_VERSION {
        return Err(Error::InvalidData(format!(
            "Unsupported metadata directory version {}",
            version
        )));
    }
    let count = r.u32()?;
    let mut frames = Vec::new();
    for _ in 0..count {
        frames.push(MetaFrame {
            kind: r.u32()?,
            offset: r.u64()?,
            len: r.u64()?,
        });
    }
    r.finish()?;
    Ok(frames)
}

fn decode_groups(bytes: &[u8], num_records: usize) -> Result<BTreeMap<String, Grouping>> {
    let mut r = Reader {
        bytes,
        pos: 0,
        what: "Groups section",
    };
    let version = r.u32()?;
    if version != GROUPS_VERSION {
        return Err(Error::InvalidData(format!(
            "Unsupported groups section version {}",
            version
        )));
    }
    let mut groups = BTreeMap::new();
    for _ in 0..r.u32()? {
        let name = r.str()?;
        let n_roles = r.u32()?;
        let roles = (0..n_roles).map(|_| r.str()).collect::<Result<Vec<_>>>()?;
        let n_groups = r.count()?;
        let n_members = r.count()?;
        let offsets = r.array(n_groups.saturating_add(1), u64::from_le_bytes)?;
        let records = r.array(n_members, u64::from_le_bytes)?;
        let member_roles = if roles.is_empty() {
            Vec::new()
        } else {
            r.array(n_members, u16::from_le_bytes)?
        };
        let mut properties = Vec::new();
        for _ in 0..r.u32()? {
            let key = r.str()?;
            let column = match r.take(1)?[0] {
                TYPE_FLOAT => GroupColumn::Float(r.array(n_groups, f64::from_le_bytes)?),
                TYPE_INT => GroupColumn::Int(r.array(n_groups, i64::from_le_bytes)?),
                TYPE_STRING => {
                    GroupColumn::String((0..n_groups).map(|_| r.str()).collect::<Result<_>>()?)
                }
                tag => {
                    return Err(Error::InvalidData(format!(
                        "Unsupported group property type tag {}",
                        tag
                    )));
                }
            };
            properties.push((key, column));
        }
        let grouping = Grouping {
            roles,
            offsets,
            records,
            member_roles,
            properties,
        };
        grouping.validate(num_records)?;
        if groups.insert(name.clone(), grouping).is_some() {
            return Err(Error::InvalidData(format!("Duplicate grouping '{}'", name)));
        }
    }
    r.finish()?;
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn db_with_records(path: &Path, n: usize) -> AtomDatabase {
        let mut db = AtomDatabase::create(path, CompressionType::Zstd(3)).unwrap();
        let molecules: Vec<Molecule> = (0..n)
            .map(|i| Molecule::new(vec![[i as f32, 0.0, 0.0]], vec![6]).unwrap())
            .collect();
        db.add_molecules(&molecules.iter().collect::<Vec<_>>())
            .unwrap();
        db
    }

    fn adsorption() -> Grouping {
        // g0 = {adslab: 1, slab: 0}, g1 = {adslab: 2, slab: 0, gas: 3}
        Grouping {
            roles: vec!["adslab".into(), "slab".into(), "gas".into()],
            offsets: vec![0, 2, 5],
            records: vec![1, 0, 2, 0, 3],
            member_roles: vec![0, 1, 0, 1, 2],
            properties: vec![
                ("e_ads".into(), GroupColumn::Float(vec![-1.5, -0.25])),
                (
                    "id".into(),
                    GroupColumn::String(vec!["a".into(), "b".into()]),
                ),
            ],
        }
    }

    fn ordered() -> Grouping {
        Grouping {
            offsets: vec![0, 3],
            records: vec![3, 1, 2],
            properties: vec![("n".into(), GroupColumn::Int(vec![3]))],
            ..Grouping::default()
        }
    }

    #[test]
    fn groups_round_trip_through_flush_and_reopen() {
        let temp = NamedTempFile::new().unwrap();
        let mut db = db_with_records(temp.path(), 4);
        db.add_groups("adsorption", adsorption()).unwrap();
        db.add_groups("ordered", ordered()).unwrap();
        db.flush().unwrap();

        for db in [
            AtomDatabase::open(temp.path()).unwrap(),
            AtomDatabase::open_mmap(temp.path()).unwrap(),
        ] {
            let groups = db.groups().unwrap();
            assert_eq!(groups.keys().collect::<Vec<_>>(), ["adsorption", "ordered"]);
            assert_eq!(groups["adsorption"], adsorption());
            assert_eq!(groups["ordered"], ordered());
            assert_eq!(db.len(), 4);
        }
    }

    #[test]
    fn append_remaps_roles_and_survives_record_only_flushes() {
        let temp = NamedTempFile::new().unwrap();
        let mut db = db_with_records(temp.path(), 4);
        db.add_groups("adsorption", adsorption()).unwrap();
        db.flush().unwrap();

        // Record-only session: groups are never decoded but must be kept.
        let mut db = AtomDatabase::open(temp.path()).unwrap();
        let mol = Molecule::new(vec![[9.0, 0.0, 0.0]], vec![8]).unwrap();
        db.add_molecule(&mol).unwrap();
        db.flush().unwrap();

        let mut db = AtomDatabase::open(temp.path()).unwrap();
        let more = Grouping {
            roles: vec!["gas".into(), "adslab".into()],
            offsets: vec![0, 2],
            records: vec![4, 2],
            member_roles: vec![0, 1],
            properties: vec![
                ("id".into(), GroupColumn::String(vec!["c".into()])),
                ("e_ads".into(), GroupColumn::Float(vec![0.5])),
            ],
        };
        db.add_groups("adsorption", more).unwrap();
        db.flush().unwrap();

        let db = AtomDatabase::open_mmap(temp.path()).unwrap();
        let g = &db.groups().unwrap()["adsorption"];
        assert_eq!(g.len(), 3);
        assert_eq!(g.offsets, [0, 2, 5, 7]);
        assert_eq!(g.records[5..], [4, 2]);
        assert_eq!(g.member_roles[5..], [2, 0]); // gas, adslab in the existing role table
        assert_eq!(
            g.properties[0].1,
            GroupColumn::Float(vec![-1.5, -0.25, 0.5])
        );
    }

    #[test]
    fn invalid_groups_are_rejected_without_side_effects() {
        let temp = NamedTempFile::new().unwrap();
        let mut db = db_with_records(temp.path(), 4);
        db.add_groups("adsorption", adsorption()).unwrap();

        let out_of_bounds = Grouping {
            offsets: vec![0, 1],
            records: vec![4],
            ..Grouping::default()
        };
        let empty_group = Grouping {
            offsets: vec![0, 0, 1],
            records: vec![0],
            ..Grouping::default()
        };
        let mut duplicate_role = adsorption();
        duplicate_role.member_roles = vec![0, 0, 0, 1, 2];
        let mut short_column = adsorption();
        short_column.properties[0].1 = GroupColumn::Float(vec![1.0]);
        let mut other_keys = adsorption();
        other_keys.properties.pop();
        let mut other_type = adsorption();
        other_type.properties[0].1 = GroupColumn::Int(vec![1, 2]);

        for bad in [
            out_of_bounds.clone(),
            empty_group,
            duplicate_role,
            short_column,
        ] {
            assert!(db.add_groups("new", bad).is_err());
        }
        for bad in [other_keys, other_type, ordered()] {
            assert!(db.add_groups("adsorption", bad).is_err());
        }
        let groups = db.groups().unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups["adsorption"], adsorption());

        db.flush().unwrap();
        let mut ro = AtomDatabase::open_mmap(temp.path()).unwrap();
        assert!(ro.add_groups("ordered", ordered()).is_err());
    }

    #[test]
    fn unknown_metadata_frames_survive_flush_and_recovery() {
        let temp = NamedTempFile::new().unwrap();
        let mut db = db_with_records(temp.path(), 4);
        let future = db
            .write_metadata_frame(99, b"from a newer version")
            .unwrap();
        db.extensions.frames.push(future);
        db.flush().unwrap();

        let mut db = AtomDatabase::open(temp.path()).unwrap();
        assert!(db.extensions.frames.contains(&future));
        db.add_groups("ordered", ordered()).unwrap(); // session left open (crash)

        let db = AtomDatabase::open(temp.path()).unwrap();
        assert!(db.extensions.frames.contains(&future));
        assert_eq!(db.groups().unwrap()["ordered"], ordered());
    }

    #[test]
    fn corrupted_sections_fail_cleanly() {
        let groups = BTreeMap::from([
            ("a".to_string(), adsorption()),
            ("o".to_string(), ordered()),
        ]);
        let bytes = encode_groups(&[("a", &adsorption()), ("o", &ordered())]);
        assert_eq!(decode_groups(&bytes, 4).unwrap(), groups);
        assert!(decode_groups(&bytes, 3).is_err()); // records must exist
        for cut in 0..bytes.len() {
            assert!(decode_groups(&bytes[..cut], 4).is_err());
        }
        let mut huge = bytes.clone();
        // n_groups of grouping "a": after version, count, name and 3 roles.
        let n_groups_at = 8 + (4 + 1) + 4 + (4 + 6) + (4 + 4) + (4 + 3);
        huge[n_groups_at..][..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(decode_groups(&huge, 4).is_err());

        let frame = MetaFrame {
            kind: FRAME_GROUPS,
            offset: 9,
            len: 3,
        };
        let dir = Extensions::from_frames(vec![frame])
            .encode_directory()
            .unwrap();
        assert_eq!(decode_directory(&dir).unwrap(), [frame]);
        for cut in 0..dir.len() {
            assert!(decode_directory(&dir[..cut]).is_err());
        }
    }
}
