/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright The LanceDB Authors
 *
 * Example: Using the ObjectStore wrapper to provide S3 credentials
 *
 * This example demonstrates creating a table on S3 storage using the
 * WrappingObjectStore callback mechanism, instead of passing credentials
 * via the connection builder's storage options.
 *
 * The wrap callback receives user_data containing S3 configuration and
 * creates an S3 object store programmatically using lancedb_object_store_create_s3.
 *
 * The bucket must already exist on the S3 endpoint. For example, create
 * the S3 bucket using aws cli:
 *
 *   export AWS_ACCESS_KEY_ID=apple
 *   export AWS_SECRET_ACCESS_KEY=banana
 *   export AWS_DEFAULT_REGION=orange
 *   aws s3 mb s3://my-bucket --endpoint-url http://localhost:9000
 *
 * Then run the example:
 *   ./s3_wrapper http://localhost:9000 orange apple banana my-bucket
 */

#include <iostream>
#include <memory>
#include <vector>
#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include "lancedb.h"

// These functions are provided by the lancedb library but are not part of
// the public API in lancedb.h. They are used here to create S3 object stores
// from within the wrap callback.
extern "C" {
LanceDBObjectStore* lancedb_object_store_create_s3(
    const char* bucket,
    const char* region,
    const char* endpoint,
    const char* access_key_id,
    const char* secret_access_key,
    int allow_http);

void lancedb_object_store_free(LanceDBObjectStore* store);
}

constexpr size_t DIM = 128;

// S3 configuration passed to the wrap callback via user_data
struct S3Config {
  const char* bucket;
  const char* region;
  const char* endpoint;
  const char* access_key_id;
  const char* secret_access_key;
};

// Wrap callback: creates an S3 object store from the user_data configuration,
// replacing the original object store that lance would use by default.
LanceDBObjectStore* wrap_with_s3(
    const LanceDBObjectStore* /*original*/,
    const char* const* /*keys*/,
    const char* const* /*values*/,
    size_t /*count*/,
    void* user_data) {
  auto* cfg = static_cast<S3Config*>(user_data);
  return lancedb_object_store_create_s3(
      cfg->bucket, cfg->region, cfg->endpoint,
      cfg->access_key_id, cfg->secret_access_key,
      1 /* allow_http */);
}

auto create_schema() {
  auto id_field = arrow::field("id", arrow::int32());
  auto item_field = arrow::field("item", arrow::fixed_size_list(arrow::float32(), DIM));
  return arrow::schema({id_field, item_field});
}

LanceDBTable* create_empty_table(LanceDBConnection* db, S3Config& config) {
  auto schema = create_schema();
  struct ArrowSchema c_schema;
  if (const auto status = arrow::ExportSchema(*schema, &c_schema); !status.ok()) {
    std::cerr << "failed to export schema to C ABI: " << status.ToString() << std::endl;
    return nullptr;
  }

  const std::string table_name = "empty_table";

  // Create table via builder so we can attach write options with the wrapper
  LanceDBCreateTableBuilder* builder = lancedb_connection_create_table_builder(
      db, table_name.c_str(),
      reinterpret_cast<FFI_ArrowSchema*>(&c_schema),
      nullptr);
  if (!builder) {
    std::cerr << "failed to create table builder" << std::endl;
    if (c_schema.release) c_schema.release(&c_schema);
    return nullptr;
  }

  // Set up write options with the S3 object store wrapper
  LanceDBWriteOptions write_opts;
  lancedb_write_options_defaults(&write_opts);

  LanceDBObjectStoreParams store_params;
  lancedb_object_store_params_defaults(&store_params);
  store_params.wrap_fn = wrap_with_s3;
  store_params.wrap_user_data = &config;
  write_opts.store_params = &store_params;

  builder = lancedb_create_table_builder_write_options(builder, &write_opts);
  if (!builder) {
    std::cerr << "failed to set write options" << std::endl;
    if (c_schema.release) c_schema.release(&c_schema);
    return nullptr;
  }

  LanceDBTable* tbl = nullptr;
  char* error_message = nullptr;
  if (const LanceDBError result = lancedb_create_table_builder_execute(
          builder, &tbl, &error_message);
      result != LANCEDB_SUCCESS) {
    std::cerr << "error creating table: " << table_name
              << ", error: " << error_message << std::endl;
    lancedb_free_string(error_message);
  } else {
    std::cout << "created table: " << table_name << " (empty)" << std::endl;
  }

  if (c_schema.release) {
    c_schema.release(&c_schema);
  }
  return tbl;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <s3_endpoint> <aws_region> <aws_access_key_id>"
              << " <aws_secret_access_key> <bucket_name>" << std::endl;
    return 1;
  }

  const std::string s3_endpoint = argv[1];
  const std::string aws_region = argv[2];
  const std::string aws_access_key_id = argv[3];
  const std::string aws_secret_access_key = argv[4];
  const std::string bucket_name = argv[5];

  S3Config config = {
    bucket_name.c_str(),
    aws_region.c_str(),
    s3_endpoint.c_str(),
    aws_access_key_id.c_str(),
    aws_secret_access_key.c_str()
  };

  // Connect to S3.
  // Note: we still need storage options on the connection for operations
  // that don't go through the write path (e.g. listing/dropping tables).
  // The wrapper handles the object store used during table creation.
  const std::string uri = "s3://" + bucket_name + "/sample-lancedb-wrapper";
  LanceDBConnectBuilder* builder = lancedb_connect(uri.c_str());
  if (!builder) {
    std::cerr << "failed to create connection builder" << std::endl;
    return 1;
  }
  builder = lancedb_connect_builder_storage_option(builder, "endpoint", s3_endpoint.c_str());
  builder = lancedb_connect_builder_storage_option(builder, "aws_region", aws_region.c_str());
  builder = lancedb_connect_builder_storage_option(builder, "aws_access_key_id", aws_access_key_id.c_str());
  builder = lancedb_connect_builder_storage_option(builder, "aws_secret_access_key", aws_secret_access_key.c_str());
  builder = lancedb_connect_builder_storage_option(builder, "allow_http", "true");
  builder = lancedb_connect_builder_storage_option(builder, "aws_s3_addressing_style", "path");

  LanceDBConnection* db = lancedb_connect_builder_execute(builder);
  if (!db) {
    std::cerr << "failed to connect to database" << std::endl;
    return 1;
  }

  // Create table using the object store wrapper (not connection storage options)
  auto empty_table = create_empty_table(db, config);
  if (!empty_table) {
    std::cerr << "failed to create empty table" << std::endl;
    lancedb_connection_free(db);
    return 1;
  }
  lancedb_table_free(empty_table);

  // List table names
  char** table_names;
  size_t name_count;
  char* error_message = nullptr;
  if (const LanceDBError result = lancedb_connection_table_names(
          db, &table_names, &name_count, &error_message);
      result != LANCEDB_SUCCESS) {
    std::cerr << "error listing table names, error: " << error_message << std::endl;
    lancedb_free_string(error_message);
  } else {
    std::cout << name_count << " tables found" << std::endl;
    for (size_t i = 0; i < name_count; i++) {
      std::cout << "table: " << table_names[i] << std::endl;
    }
    lancedb_free_table_names(table_names, name_count);
  }

  if (const LanceDBError result = lancedb_connection_drop_table(
          db, "empty_table", nullptr, &error_message);
      result != LANCEDB_SUCCESS) {
    std::cerr << "error dropping table, error: " << error_message << std::endl;
    lancedb_free_string(error_message);
  } else {
    std::cout << "dropped table empty_table" << std::endl;
  }

  lancedb_connection_free(db);
  return 0;
}
