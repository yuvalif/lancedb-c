/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright The LanceDB Authors
 */

#include "test_common.h"

TEST_CASE_METHOD(LanceDBFixture, "LanceDB Table List Versions", "[table]") {
  SECTION("List versions on freshly created table") {
    constexpr auto row_num = 10;
    LanceDBTable* table = create_table_with_data("versions_test", row_num, 0);
    REQUIRE(table != nullptr);

    LanceDBVersion* versions = nullptr;
    size_t count = 0;
    char* error_message = nullptr;

    LanceDBError result = lancedb_table_list_versions(table, &versions, nullptr, &count, &error_message);

    REQUIRE(result == LANCEDB_SUCCESS);
    REQUIRE(error_message == nullptr);
    REQUIRE(count == 1);
    REQUIRE(versions != nullptr);

    // Verify version 1 has a valid timestamp within 5 seconds of now
    auto now = std::chrono::system_clock::now();
    auto now_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    REQUIRE(versions[0].version == 1);
    REQUIRE(versions[0].timestamp_nanos < 1000000000);
    REQUIRE(std::abs(now_seconds - versions[0].timestamp_seconds) <= 5);

    lancedb_free_versions(versions, count);
    lancedb_table_free(table);
  }

  SECTION("List versions after adding data") {
    const std::string table_name = "versions_add_test";
    create_empty_table(table_name);
    LanceDBTable* table = lancedb_connection_open_table(db, table_name.c_str());
    REQUIRE(table != nullptr);

    // Add data 3 times to create versions 2, 3, 4
    char* error_message = nullptr;
    for (int i = 0; i < 3; i++) {
      auto batch = create_test_record_batch(5, i * 5);
      auto reader = create_reader_from_batch(batch);
      REQUIRE(reader != nullptr);

      LanceDBError add_result = lancedb_table_add(table, reader, &error_message);
      REQUIRE(add_result == LANCEDB_SUCCESS);
      REQUIRE(error_message == nullptr);
    }

    LanceDBVersion* versions = nullptr;
    size_t count = 0;

    LanceDBError result = lancedb_table_list_versions(table, &versions, nullptr, &count, &error_message);

    REQUIRE(result == LANCEDB_SUCCESS);
    REQUIRE(error_message == nullptr);
    // 1 (empty table creation) + 3 (adds) = 4 versions
    REQUIRE(count == 4);
    REQUIRE(versions != nullptr);

    lancedb_free_versions(versions, count);
    lancedb_table_free(table);
  }

  SECTION("List versions with metadata") {
    constexpr auto row_num = 10;
    LanceDBTable* table = create_table_with_data("versions_metadata_test", row_num, 0);
    REQUIRE(table != nullptr);

    LanceDBVersion* versions = nullptr;
    LanceDBVersionMetadata* metadata = nullptr;
    size_t count = 0;
    char* error_message = nullptr;

    LanceDBError result = lancedb_table_list_versions(table, &versions, &metadata, &count, &error_message);

    REQUIRE(result == LANCEDB_SUCCESS);
    REQUIRE(error_message == nullptr);
    REQUIRE(count == 1);
    REQUIRE(versions != nullptr);
    REQUIRE(metadata != nullptr);
    REQUIRE(metadata[0].count > 0);
    REQUIRE(metadata[0].keys != nullptr);
    REQUIRE(metadata[0].values != nullptr);

    lancedb_free_version_metadata(metadata, count);
    lancedb_free_versions(versions, count);
    lancedb_table_free(table);
  }

  SECTION("List versions with null table should fail") {
    LanceDBVersion* versions = nullptr;
    size_t count = 0;
    char* error_message = nullptr;

    LanceDBError result = lancedb_table_list_versions(nullptr, &versions, nullptr, &count, &error_message);

    REQUIRE(result == LANCEDB_INVALID_ARGUMENT);
    REQUIRE(error_message != nullptr);
    lancedb_free_string(error_message);
  }

  SECTION("List versions with null output should fail") {
    constexpr auto row_num = 5;
    LanceDBTable* table = create_table_with_data("versions_null_out", row_num, 0);
    REQUIRE(table != nullptr);

    char* error_message = nullptr;
    size_t count = 0;

    LanceDBError result = lancedb_table_list_versions(table, nullptr, nullptr, &count, &error_message);
    REQUIRE(result == LANCEDB_INVALID_ARGUMENT);
    REQUIRE(error_message != nullptr);
    lancedb_free_string(error_message);

    lancedb_table_free(table);
  }

  SECTION("Free null versions is safe") {
    lancedb_free_versions(nullptr, 0);
  }

  SECTION("Free null version metadata is safe") {
    lancedb_free_version_metadata(nullptr, 0);
  }
}
