// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Common types shared across LanceDB C bindings modules

use std::collections::HashMap;
use std::ffi::CString;
use std::fmt;
use std::os::raw::c_char;
use std::sync::Arc;

use arrow_array::RecordBatchReader;
use lance::io::WrappingObjectStore;
use lancedb::DistanceType;
use object_store::ObjectStore as OSObjectStore;

/// Distance type enum for C API
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum LanceDBDistanceType {
    L2 = 0,
    Cosine = 1,
    Dot = 2,
    Hamming = 3,
}

impl From<LanceDBDistanceType> for DistanceType {
    fn from(dt: LanceDBDistanceType) -> Self {
        match dt {
            LanceDBDistanceType::L2 => Self::L2,
            LanceDBDistanceType::Cosine => Self::Cosine,
            LanceDBDistanceType::Dot => Self::Dot,
            LanceDBDistanceType::Hamming => Self::Hamming,
        }
    }
}

/// Opaque handle to Arrow RecordBatchReader for C interop
#[repr(C)]
pub struct LanceDBRecordBatchReader {
    inner: Box<dyn RecordBatchReader + Send>,
}

impl LanceDBRecordBatchReader {
    /// Create a new reader from a Box<dyn RecordBatchReader + Send>
    pub fn new(reader: Box<dyn RecordBatchReader + Send>) -> Self {
        Self { inner: reader }
    }

    /// Extract the inner reader (consumes self)
    pub fn into_inner(self) -> Box<dyn RecordBatchReader + Send> {
        self.inner
    }
}

/// Merge insert configuration
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LanceDBMergeInsertConfig {
    pub when_matched_update_all: i32, // Update all columns for matched records (1 = true, 0 = false)
    pub when_not_matched_insert_all: i32, // Insert all new records (1 = true, 0 = false)
}

/// Write mode enum for C API
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum LanceDBWriteMode {
    Create = 0,
    Append = 1,
    Overwrite = 2,
}

/// Lance file version enum for C API
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum LanceDBDataStorageVersion {
    None = 0,   // Use default (latest stable)
    Legacy = 1, // The legacy (0.1) format
    V2_0 = 2,
    Stable = 3, // The latest stable release (default for new datasets)
    V2_1 = 4,
    Next = 5, // The latest unstable release
    V2_2 = 6,
}

/// Write options configuration for table creation
///
/// See: https://docs.rs/lance/latest/lance/dataset/write/struct.WriteParams.html
///
/// LanceDBDataStorageVersion uses None for default (latest stable).
///
/// Note: the following WriteParams fields cannot be represented in C and are not included:
/// progress, commit_handler, session, auto_cleanup,
/// transaction_properties, initial_bases, target_bases, target_base_names_or_paths
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LanceDBWriteOptions {
    pub max_rows_per_file: u64,  // Max records per file
    pub max_rows_per_group: u64, // Max rows per row group
    pub max_bytes_per_file: u64, // Max file size in bytes (soft limit)
    pub mode: LanceDBWriteMode,  // Write mode (Create, Append, Overwrite)
    pub data_storage_version: LanceDBDataStorageVersion, // Data file format version (None = default)
    pub enable_stable_row_ids: i32, // Use stable row IDs, experimental (1 = true, 0 = false)
    pub enable_v2_manifest_paths: i32, // Use v2 manifest paths (1 = true, 0 = false)
    pub skip_auto_cleanup: i32,     // Skip auto cleanup during commits (1 = true, 0 = false)
    pub store_params: *const LanceDBObjectStoreParams, // Object store params (NULL = defaults)
}

/// Opaque handle wrapping an `Arc<dyn ObjectStore>` for use in C callbacks
pub struct LanceDBObjectStore {
    pub(crate) inner: Arc<dyn OSObjectStore>,
}

/// C callback type for wrapping an object store
///
/// Called with the original object store handle, key/value storage options arrays,
/// their count, and user data. Returns a new handle to a wrapped object store,
/// or NULL to use the original unchanged.
///
/// The `original` pointer is only valid during the callback invocation.
pub type LanceDBWrapObjectStoreFn = Option<
    unsafe extern "C" fn(
        original: *const LanceDBObjectStore,
        keys: *const *const c_char,
        values: *const *const c_char,
        count: usize,
        user_data: *mut std::ffi::c_void,
    ) -> *mut LanceDBObjectStore,
>;

/// C callback type for freeing user data associated with a wrap callback
pub type LanceDBFreeUserDataFn = Option<unsafe extern "C" fn(*mut std::ffi::c_void)>;

/// Object store parameters for C API
///
/// Controls object store behavior when creating tables.
/// Use `lancedb_object_store_params_defaults()` to initialize with defaults.
#[repr(C)]
pub struct LanceDBObjectStoreParams {
    pub s3_credentials_refresh_offset_secs: u64, // Duration in seconds (default: 60)
    pub use_constant_size_upload_parts: i32,     // 1=true, 0=false (default: 0)
    pub wrap_fn: LanceDBWrapObjectStoreFn, // Object store wrapper callback (NULL = no wrapper)
    pub wrap_user_data: *mut std::ffi::c_void, // User data passed to wrap_fn
    pub free_user_data: LanceDBFreeUserDataFn, // Callback to free wrap_user_data (NULL = no-op)
}

/// Implementation of `WrappingObjectStore` that delegates to a C callback
pub(crate) struct CWrappingObjectStore {
    wrap_fn: unsafe extern "C" fn(
        original: *const LanceDBObjectStore,
        keys: *const *const c_char,
        values: *const *const c_char,
        count: usize,
        user_data: *mut std::ffi::c_void,
    ) -> *mut LanceDBObjectStore,
    user_data: *mut std::ffi::c_void,
    free_fn: LanceDBFreeUserDataFn,
}

// Safety: C callback must be thread-safe by contract
unsafe impl Send for CWrappingObjectStore {}
unsafe impl Sync for CWrappingObjectStore {}

impl fmt::Debug for CWrappingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CWrappingObjectStore")
    }
}

impl Drop for CWrappingObjectStore {
    fn drop(&mut self) {
        if let Some(free_fn) = self.free_fn {
            unsafe {
                free_fn(self.user_data);
            }
        }
    }
}

impl CWrappingObjectStore {
    pub(crate) fn new(
        wrap_fn: unsafe extern "C" fn(
            original: *const LanceDBObjectStore,
            keys: *const *const c_char,
            values: *const *const c_char,
            count: usize,
            user_data: *mut std::ffi::c_void,
        ) -> *mut LanceDBObjectStore,
        user_data: *mut std::ffi::c_void,
        free_fn: LanceDBFreeUserDataFn,
    ) -> Self {
        Self {
            wrap_fn,
            user_data,
            free_fn,
        }
    }
}

impl WrappingObjectStore for CWrappingObjectStore {
    fn wrap(
        &self,
        original: Arc<dyn OSObjectStore>,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Arc<dyn OSObjectStore> {
        // Wrap original in a temporary struct (non-owning, valid only during callback)
        let temp = LanceDBObjectStore {
            inner: original.clone(),
        };

        // Convert storage_options to C key/value arrays
        let (c_keys, c_values, count) = if let Some(opts) = storage_options {
            let keys: Vec<CString> = opts
                .keys()
                .map(|k| CString::new(k.as_str()).unwrap_or_default())
                .collect();
            let values: Vec<CString> = opts
                .values()
                .map(|v| CString::new(v.as_str()).unwrap_or_default())
                .collect();
            let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
            let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();
            let count = keys.len();
            // Keep keys/values alive until after the callback
            (Some((keys, key_ptrs)), Some((values, value_ptrs)), count)
        } else {
            (None, None, 0)
        };

        let key_ptrs_ptr = c_keys
            .as_ref()
            .map(|(_, ptrs)| ptrs.as_ptr())
            .unwrap_or(std::ptr::null());
        let value_ptrs_ptr = c_values
            .as_ref()
            .map(|(_, ptrs)| ptrs.as_ptr())
            .unwrap_or(std::ptr::null());

        let result = unsafe {
            (self.wrap_fn)(
                &temp as *const LanceDBObjectStore,
                key_ptrs_ptr,
                value_ptrs_ptr,
                count,
                self.user_data,
            )
        };

        if result.is_null() {
            // NULL means use original unchanged
            original
        } else {
            // Take ownership of the returned handle
            let wrapped = unsafe { Box::from_raw(result) };
            wrapped.inner
        }
    }
}
