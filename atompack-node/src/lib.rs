//! Read-only Node-API binding for the VS Code viewer.
//! Each instance owns the existing AtomDatabase mmap reader, including in scan workers.

mod view;

use atompack::AtomDatabase;
use napi_derive::napi;
use serde_json::Value;

#[napi]
pub struct NativeReader {
    database: Option<AtomDatabase>,
}

impl NativeReader {
    fn read(
        &self,
        f: impl FnOnce(&AtomDatabase) -> atompack::Result<Value>,
    ) -> napi::Result<Value> {
        let database = self
            .database
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Reader is not open"))?;
        f(database).map_err(|err| napi::Error::from_reason(err.to_string()))
    }
}

#[napi]
impl NativeReader {
    #[napi(constructor)]
    pub fn new(path: String) -> napi::Result<Self> {
        let database = AtomDatabase::open_mmap(path)
            .map_err(|err| napi::Error::from_reason(err.to_string()))?;
        Ok(Self {
            database: Some(database),
        })
    }

    #[napi]
    pub fn overview(&self) -> napi::Result<Value> {
        self.read(view::overview)
    }

    #[napi]
    pub fn records(&self, start: u32, count: u32) -> napi::Result<Value> {
        self.read(|db| view::records(db, start as usize, count as usize))
    }

    #[napi(js_name = "record_columns")]
    pub fn record_columns(&self, start: u32, count: u32) -> napi::Result<Value> {
        self.read(|db| view::record_columns(db, start as usize, count as usize))
    }

    #[napi]
    pub fn molecule(&self, index: u32) -> napi::Result<Value> {
        self.read(|db| view::molecule(db, index as usize))
    }

    #[napi]
    pub fn groups(&self, grouping: u32, start: u32, count: u32) -> napi::Result<Value> {
        self.read(|db| view::groups(db, grouping as usize, start as usize, count as usize))
    }

    #[napi(js_name = "group_columns")]
    pub fn group_columns(&self, grouping: u32) -> napi::Result<Value> {
        self.read(|db| view::group_columns(db, grouping as usize))
    }

    /// Release mappings before the host removes a temporary file (required on Windows).
    #[napi]
    pub fn dispose(&mut self) {
        self.database = None;
    }
}
