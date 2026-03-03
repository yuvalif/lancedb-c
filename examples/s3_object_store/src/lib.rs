// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! S3 object store helper for lancedb-c examples.
//!
//! Provides FFI functions to create an S3 object store that can be returned
//! from a `LanceDBWrapObjectStoreFn` callback.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

use object_store::aws::AmazonS3Builder;
use object_store::ObjectStore;

/// Opaque handle wrapping an `Arc<dyn ObjectStore>`.
///
/// Must match the memory layout of `LanceDBObjectStore` in the main lancedb-c
/// crate (both contain a single `Arc<dyn object_store::ObjectStore>` field).
pub struct LanceDBObjectStore {
    #[allow(dead_code)]
    inner: Arc<dyn ObjectStore>,
}

/// Create an S3 object store for use in wrap callbacks
///
/// # Safety
/// - `bucket` must be a valid null-terminated C string
/// - `region`, `endpoint`, `access_key_id`, `secret_access_key` can be NULL
/// - The returned pointer should be returned from a wrap callback (ownership
///   transfers to lance) or freed with `lancedb_object_store_free`
///
/// # Returns
/// - Non-null pointer to LanceDBObjectStore on success
/// - Null pointer on failure
#[no_mangle]
pub unsafe extern "C" fn lancedb_object_store_create_s3(
    bucket: *const c_char,
    region: *const c_char,
    endpoint: *const c_char,
    access_key_id: *const c_char,
    secret_access_key: *const c_char,
    allow_http: i32,
) -> *mut LanceDBObjectStore {
    if bucket.is_null() {
        return ptr::null_mut();
    }

    let Ok(bucket_str) = CStr::from_ptr(bucket).to_str() else {
        return ptr::null_mut();
    };

    let mut builder = AmazonS3Builder::new().with_bucket_name(bucket_str);

    if !region.is_null() {
        if let Ok(s) = CStr::from_ptr(region).to_str() {
            builder = builder.with_region(s);
        }
    }

    if !endpoint.is_null() {
        if let Ok(s) = CStr::from_ptr(endpoint).to_str() {
            builder = builder.with_endpoint(s);
        }
    }

    if !access_key_id.is_null() {
        if let Ok(s) = CStr::from_ptr(access_key_id).to_str() {
            builder = builder.with_access_key_id(s);
        }
    }

    if !secret_access_key.is_null() {
        if let Ok(s) = CStr::from_ptr(secret_access_key).to_str() {
            builder = builder.with_secret_access_key(s);
        }
    }

    if allow_http != 0 {
        builder = builder.with_allow_http(true);
    }

    match builder.build() {
        Ok(store) => Box::into_raw(Box::new(LanceDBObjectStore {
            inner: Arc::new(store),
        })),
        Err(_) => ptr::null_mut(),
    }
}

/// Free an object store handle
///
/// Use this only for object stores that were NOT returned from a wrap callback.
/// If returned from a wrap callback, ownership is transferred to lance automatically.
///
/// # Safety
/// - `store` must be a valid pointer returned from `lancedb_object_store_create_s3`
///   or NULL
/// - `store` must not be used after calling this function
#[no_mangle]
pub unsafe extern "C" fn lancedb_object_store_free(store: *mut LanceDBObjectStore) {
    if !store.is_null() {
        let _ = Box::from_raw(store);
    }
}
