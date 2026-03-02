// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Common types shared across LanceDB C bindings modules

use arrow_array::RecordBatchReader;
use lancedb::DistanceType;

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
/// store_params, progress, commit_handler, session, auto_cleanup,
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
}
