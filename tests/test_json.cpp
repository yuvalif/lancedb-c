/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright The LanceDB Authors
 */

#include <catch2/catch.hpp>
#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include "lancedb.h"

static const char* JSON_COL = "json";

// Helper: build a single-column RecordBatch of JSON strings and export via Arrow C FFI.
// Caller must release c_array and c_schema when done.
static void export_json_batch(
    std::initializer_list<std::string> jsons,
    const char* col_name,
    ArrowArray* c_array,
    ArrowSchema* c_schema) {
  arrow::StringBuilder builder;
  for (const auto& j : jsons) {
    REQUIRE(builder.Append(j).ok());
  }
  std::shared_ptr<arrow::Array> array;
  REQUIRE(builder.Finish(&array).ok());

  auto schema = arrow::schema({arrow::field(col_name, arrow::utf8())});
  auto batch = arrow::RecordBatch::Make(schema, static_cast<int64_t>(jsons.size()), {array});

  REQUIRE(arrow::ExportRecordBatch(*batch, c_array, c_schema).ok());
}

TEST_CASE("JSON matches - string extraction", "[json]") {
  const char* path[] = {"name"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_str(lancedb_expr_column(JSON_COL), path, 1),
      LANCEDB_BINARY_OP_EQ,
      lancedb_expr_literal_string("alice"));
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({R"({"name":"alice","age":30})", R"({"name":"bob","age":40})"}, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(error_message == nullptr);
  REQUIRE(count == 2);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - integer comparison", "[json]") {
  const char* path[] = {"age"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_int(lancedb_expr_column(JSON_COL), path, 1),
      LANCEDB_BINARY_OP_GT,
      lancedb_expr_literal_i64(25));
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({R"({"age":30})", R"({"age":20})", R"({"age":25})"}, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 3);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  REQUIRE(results[2] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - float comparison", "[json]") {
  const char* path[] = {"score"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_float(lancedb_expr_column(JSON_COL), path, 1),
      LANCEDB_BINARY_OP_GT_EQ,
      lancedb_expr_literal_f64(3.5));
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({R"({"score":4.2})", R"({"score":2.1})", R"({"score":3.5})"}, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 3);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  REQUIRE(results[2] == true);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - boolean", "[json]") {
  const char* path[] = {"active"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_bool(lancedb_expr_column(JSON_COL), path, 1),
      LANCEDB_BINARY_OP_EQ,
      lancedb_expr_literal_bool(true));
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({R"({"active":true})", R"({"active":false})"}, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 2);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - json_contains", "[json]") {
  const char* path[] = {"name"};
  LanceDBExpr* expr = lancedb_expr_json_contains(lancedb_expr_column(JSON_COL), path, 1);
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({R"({"name":"alice"})", R"({"age":30})"}, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 2);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - json_array_has", "[json]") {
  const char* path[] = {"tags"};
  LanceDBExpr* expr = lancedb_expr_json_array_has(
      lancedb_expr_column(JSON_COL), path, 1,
      lancedb_expr_literal_string("red"),
      true);
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"tags":["red","blue"]})",
    R"({"tags":["green"]})",
    R"({"tags":["red","green","blue"]})",
    R"({"tags":["yellow"]})",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(error_message == nullptr);
  REQUIRE(count == 4);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  REQUIRE(results[2] == true);
  REQUIRE(results[3] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - json_array_has no match", "[json]") {
  const char* path[] = {"tags"};
  LanceDBExpr* expr = lancedb_expr_json_array_has(
      lancedb_expr_column(JSON_COL), path, 1,
      lancedb_expr_literal_string("purple"),
      true);
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"tags":["red","blue"]})",
    R"({"tags":["green"]})",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 2);
  REQUIRE(results[0] == false);
  REQUIRE(results[1] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - json_array_has nested path", "[json]") {
  const char* path[] = {"meta", "tags"};
  LanceDBExpr* expr = lancedb_expr_json_array_has(
      lancedb_expr_column(JSON_COL), path, 2,
      lancedb_expr_literal_string("important"),
      true);
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"meta":{"tags":["important","urgent"]}})",
    R"({"meta":{"tags":["low"]}})",
    R"({"meta":{"tags":[]}})",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 3);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  REQUIRE(results[2] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - json_array_has number array", "[json]") {
  const char* path[] = {"scores"};
  LanceDBExpr* expr = lancedb_expr_json_array_has(
      lancedb_expr_column(JSON_COL), path, 1,
      lancedb_expr_literal_f64(3.14),
      false);
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"scores":[1.5, 3.14, 2.7]})",
    R"({"scores":[10, 20]})",
    R"({"scores":[3.14]})",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(error_message == nullptr);
  REQUIRE(count == 3);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  REQUIRE(results[2] == true);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - json_array_has bool array", "[json]") {
  const char* path[] = {"flags"};
  LanceDBExpr* expr = lancedb_expr_json_array_has(
      lancedb_expr_column(JSON_COL), path, 1,
      lancedb_expr_literal_bool(true),
      false);
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"flags":[true, false]})",
    R"({"flags":[false]})",
    R"({"flags":[true]})",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(error_message == nullptr);
  REQUIRE(count == 3);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  REQUIRE(results[2] == true);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - json_array_has non-literal value expr", "[json]") {
  const char* path[] = {"flags"};
  // Non-literal value: false OR true evaluates to true
  LanceDBExpr* value_expr = lancedb_expr_or(
      lancedb_expr_literal_bool(false),
      lancedb_expr_literal_bool(true));
  REQUIRE(value_expr != nullptr);
  LanceDBExpr* expr = lancedb_expr_json_array_has(
      lancedb_expr_column(JSON_COL), path, 1,
      value_expr,
      false);
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"flags":[true, false]})",
    R"({"flags":[false]})",
    R"({"flags":[true]})",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(error_message == nullptr);
  REQUIRE(count == 3);
  // false OR true = true; rows containing true match
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  REQUIRE(results[2] == true);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - json_array_has type mismatch", "[json]") {
  const char* path[] = {"scores"};
  // Boolean value against a number array
  LanceDBExpr* expr = lancedb_expr_json_array_has(
      lancedb_expr_column(JSON_COL), path, 1,
      lancedb_expr_literal_bool(true),
      false);
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"scores":[1, 2, 3]})",
    R"({"scores":[10, 20]})",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  // Cast produces "true" which doesn't match "1", "2", etc. - no error, just no matches
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 2);
  REQUIRE(results[0] == false);
  REQUIRE(results[1] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - json_array_has wrong quote flag", "[json]") {
  SECTION("Quoting number against number array") {
    const char* path[] = {"scores"};
    // quote_value=true wraps value in JSON quotes
    LanceDBExpr* expr = lancedb_expr_json_array_has(
        lancedb_expr_column(JSON_COL), path, 1,
        lancedb_expr_literal_i64(42),
        true);
    REQUIRE(expr != nullptr);

    ArrowArray c_array;
    ArrowSchema c_schema;
    export_json_batch({
      R"({"scores":[42, 10]})",
      R"({"scores":[42]})",
    }, JSON_COL, &c_array, &c_schema);

    auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
    auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
    FFI_ArrowArray* arrays[] = {ffi_array};

    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, ffi_schema, 1, expr, &results, &count, &error_message);
    // Quoting wraps value as '"42"' but array elements are '42' - no match
    REQUIRE(err == LANCEDB_SUCCESS);
    REQUIRE(count == 2);
    REQUIRE(results[0] == false);
    REQUIRE(results[1] == false);
    lancedb_free_json_matches(results);

    if (c_array.release) c_array.release(&c_array);
    if (c_schema.release) c_schema.release(&c_schema);
  }

  SECTION("Not quoting string against string array") {
    const char* path[] = {"tags"};
    // quote_value=false casts to Utf8 - wrong for string arrays
    LanceDBExpr* expr = lancedb_expr_json_array_has(
        lancedb_expr_column(JSON_COL), path, 1,
        lancedb_expr_literal_string("red"),
        false);
    REQUIRE(expr != nullptr);

    ArrowArray c_array;
    ArrowSchema c_schema;
    export_json_batch({
      R"({"tags":["red","blue"]})",
      R"({"tags":["red"]})",
    }, JSON_COL, &c_array, &c_schema);

    auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
    auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
    FFI_ArrowArray* arrays[] = {ffi_array};

    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, ffi_schema, 1, expr, &results, &count, &error_message);
    // Cast keeps 'red' but elements are '"red"' (with JSON quotes) - no match
    REQUIRE(err == LANCEDB_SUCCESS);
    REQUIRE(count == 2);
    REQUIRE(results[0] == false);
    REQUIRE(results[1] == false);
    lancedb_free_json_matches(results);

    if (c_array.release) c_array.release(&c_array);
    if (c_schema.release) c_schema.release(&c_schema);
  }
}

TEST_CASE("JSON matches - json_array_has non-scalar elements", "[json]") {
  // Non-scalar array elements use exact string matching against raw JSON text,
  // not JSON-aware comparison. Whitespace or key ordering differences cause mismatches.

  SECTION("Array of JSON objects - exact match") {
    const char* path[] = {"items"};
    LanceDBExpr* expr = lancedb_expr_json_array_has(
        lancedb_expr_column(JSON_COL), path, 1,
        lancedb_expr_literal_string(R"({"name":"alice"})"),
        false);
    REQUIRE(expr != nullptr);

    ArrowArray c_array;
    ArrowSchema c_schema;
    export_json_batch({
      R"({"items":[{"name":"alice"},{"name":"bob"}]})",
      R"({"items":[{"name":"carol"}]})",
    }, JSON_COL, &c_array, &c_schema);

    auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
    auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
    FFI_ArrowArray* arrays[] = {ffi_array};

    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, ffi_schema, 1, expr, &results, &count, &error_message);
    REQUIRE(err == LANCEDB_SUCCESS);
    REQUIRE(count == 2);
    REQUIRE(results[0] == true);
    REQUIRE(results[1] == false);
    lancedb_free_json_matches(results);

    if (c_array.release) c_array.release(&c_array);
    if (c_schema.release) c_schema.release(&c_schema);
  }

  SECTION("Array of JSON objects - whitespace mismatch") {
    const char* path[] = {"items"};
    // Extra space after colon — semantically identical JSON, but won't match
    LanceDBExpr* expr = lancedb_expr_json_array_has(
        lancedb_expr_column(JSON_COL), path, 1,
        lancedb_expr_literal_string(R"({"name": "alice"})"),
        false);
    REQUIRE(expr != nullptr);

    ArrowArray c_array;
    ArrowSchema c_schema;
    export_json_batch({
      R"({"items":[{"name":"alice"},{"name":"bob"}]})",
      R"({"items":[{"name":"carol"}]})",
    }, JSON_COL, &c_array, &c_schema);

    auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
    auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
    FFI_ArrowArray* arrays[] = {ffi_array};

    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, ffi_schema, 1, expr, &results, &count, &error_message);
    REQUIRE(err == LANCEDB_SUCCESS);
    REQUIRE(count == 2);
    // No match despite semantically equivalent JSON — raw string comparison
    REQUIRE(results[0] == false);
    REQUIRE(results[1] == false);
    lancedb_free_json_matches(results);

    if (c_array.release) c_array.release(&c_array);
    if (c_schema.release) c_schema.release(&c_schema);
  }

  SECTION("Array of arrays - exact match") {
    const char* path[] = {"matrix"};
    LanceDBExpr* expr = lancedb_expr_json_array_has(
        lancedb_expr_column(JSON_COL), path, 1,
        lancedb_expr_literal_string("[1,2]"),
        false);
    REQUIRE(expr != nullptr);

    ArrowArray c_array;
    ArrowSchema c_schema;
    export_json_batch({
      R"({"matrix":[[1,2],[3,4]]})",
      R"({"matrix":[[5,6]]})",
    }, JSON_COL, &c_array, &c_schema);

    auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
    auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
    FFI_ArrowArray* arrays[] = {ffi_array};

    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, ffi_schema, 1, expr, &results, &count, &error_message);
    REQUIRE(err == LANCEDB_SUCCESS);
    REQUIRE(count == 2);
    REQUIRE(results[0] == true);
    REQUIRE(results[1] == false);
    lancedb_free_json_matches(results);

    if (c_array.release) c_array.release(&c_array);
    if (c_schema.release) c_schema.release(&c_schema);
  }

  SECTION("Array of arrays - reorder mismatch") {
    const char* path[] = {"matrix"};
    // Extra space after comma — semantically identical, but won't match
    LanceDBExpr* expr = lancedb_expr_json_array_has(
        lancedb_expr_column(JSON_COL), path, 1,
        lancedb_expr_literal_string("[2,1]"),
        false);
    REQUIRE(expr != nullptr);

    ArrowArray c_array;
    ArrowSchema c_schema;
    export_json_batch({
      R"({"matrix":[[1,2],[3,4]]})",
      R"({"matrix":[[5,6]]})",
    }, JSON_COL, &c_array, &c_schema);

    auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
    auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
    FFI_ArrowArray* arrays[] = {ffi_array};

    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, ffi_schema, 1, expr, &results, &count, &error_message);
    REQUIRE(err == LANCEDB_SUCCESS);
    REQUIRE(count == 2);
    // No match despite semantically equivalent JSON — raw string comparison
    REQUIRE(results[0] == false);
    REQUIRE(results[1] == false);
    lancedb_free_json_matches(results);

    if (c_array.release) c_array.release(&c_array);
    if (c_schema.release) c_schema.release(&c_schema);
  }
}

TEST_CASE("JSON expr builders - json_array_has null arguments", "[json]") {
  const char* path[] = {"tags"};

  SECTION("Null json_expr") {
    REQUIRE(lancedb_expr_json_array_has(nullptr, path, 1,
        lancedb_expr_literal_string("red"), true) == nullptr);
  }

  SECTION("Null path") {
    REQUIRE(lancedb_expr_json_array_has(lancedb_expr_column(JSON_COL), nullptr, 1,
        lancedb_expr_literal_string("red"), true) == nullptr);
  }

  SECTION("Zero path_len") {
    REQUIRE(lancedb_expr_json_array_has(lancedb_expr_column(JSON_COL), path, 0,
        lancedb_expr_literal_string("red"), true) == nullptr);
  }

  SECTION("Null value_expr") {
    REQUIRE(lancedb_expr_json_array_has(lancedb_expr_column(JSON_COL), path, 1,
        nullptr, true) == nullptr);
  }
}

TEST_CASE("JSON matches - compound expression", "[json]") {
  const char* name_path[] = {"name"};
  const char* age_path[] = {"age"};
  LanceDBExpr* name_check = lancedb_expr_binary(
      lancedb_expr_json_get_str(lancedb_expr_column(JSON_COL), name_path, 1),
      LANCEDB_BINARY_OP_EQ,
      lancedb_expr_literal_string("alice"));
  REQUIRE(name_check != nullptr);

  LanceDBExpr* age_check = lancedb_expr_binary(
      lancedb_expr_json_get_int(lancedb_expr_column(JSON_COL), age_path, 1),
      LANCEDB_BINARY_OP_GT,
      lancedb_expr_literal_i64(25));
  REQUIRE(age_check != nullptr);

  LanceDBExpr* name_check_clone = lancedb_expr_clone(name_check);
  LanceDBExpr* age_check_clone = lancedb_expr_clone(age_check);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"name":"alice","age":30})",
    R"({"name":"bob","age":30})",
    R"({"name":"alice","age":20})"
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  SECTION("AND - both conditions met") {
    LanceDBExpr* and_expr = lancedb_expr_binary(name_check, LANCEDB_BINARY_OP_AND, age_check);
    REQUIRE(and_expr != nullptr);

    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, ffi_schema, 1, and_expr, &results, &count, &error_message);
    REQUIRE(err == LANCEDB_SUCCESS);
    REQUIRE(count == 3);
    REQUIRE(results[0] == true);
    REQUIRE(results[1] == false);
    REQUIRE(results[2] == false);
    lancedb_free_json_matches(results);

    lancedb_expr_free(name_check_clone);
    lancedb_expr_free(age_check_clone);
  }

  SECTION("OR - one condition met") {
    LanceDBExpr* or_expr = lancedb_expr_binary(name_check_clone, LANCEDB_BINARY_OP_OR, age_check_clone);
    REQUIRE(or_expr != nullptr);

    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, ffi_schema, 1, or_expr, &results, &count, &error_message);
    REQUIRE(err == LANCEDB_SUCCESS);
    REQUIRE(count == 3);
    REQUIRE(results[0] == true);
    REQUIRE(results[1] == true);
    REQUIRE(results[2] == true);
    lancedb_free_json_matches(results);

    lancedb_expr_free(name_check);
    lancedb_expr_free(age_check);
  }

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - wrong column name in expression", "[json]") {
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_column("wrong_column"),
      LANCEDB_BINARY_OP_EQ,
      lancedb_expr_literal_string("alice"));
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({R"({"name":"alice"})"}, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_OTHER);
  REQUIRE(error_message != nullptr);
  lancedb_free_string(error_message);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - non-boolean expression", "[json]") {
  // Pass a raw json_get_int without a comparison — result is Int64, not Boolean
  const char* path[] = {"age"};
  LanceDBExpr* expr = lancedb_expr_json_get_int(lancedb_expr_column(JSON_COL), path, 1);
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({R"({"age":30})"}, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_INVALID_ARGUMENT);
  REQUIRE(error_message != nullptr);
  lancedb_free_string(error_message);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - json_get on non-existent column", "[json]") {
  const char* path[] = {"name"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_str(lancedb_expr_column("no_such_column"), path, 1),
      LANCEDB_BINARY_OP_EQ,
      lancedb_expr_literal_string("alice"));
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({R"({"name":"alice"})"}, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_OTHER);
  REQUIRE(error_message != nullptr);
  lancedb_free_string(error_message);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - json_get on non-string column", "[json]") {
  const char* path[] = {"key"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_int(lancedb_expr_column("nums"), path, 1),
      LANCEDB_BINARY_OP_GT,
      lancedb_expr_literal_i64(0));
  REQUIRE(expr != nullptr);

  // Build a RecordBatch with an int64 column instead of string
  arrow::Int64Builder builder;
  REQUIRE(builder.Append(42).ok());
  REQUIRE(builder.Append(7).ok());
  std::shared_ptr<arrow::Array> array;
  REQUIRE(builder.Finish(&array).ok());

  auto schema = arrow::schema({arrow::field("nums", arrow::int64())});
  auto batch = arrow::RecordBatch::Make(schema, 2, {array});

  ArrowArray c_array;
  ArrowSchema c_schema;
  REQUIRE(arrow::ExportRecordBatch(*batch, &c_array, &c_schema).ok());

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_OTHER);
  REQUIRE(error_message != nullptr);
  lancedb_free_string(error_message);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - json_get on literal instead of column", "[json]") {
  // Use a string literal where a column expression is expected
  const char* path[] = {"name"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_str(lancedb_expr_literal_string(R"({"name":"alice"})"), path, 1),
      LANCEDB_BINARY_OP_EQ,
      lancedb_expr_literal_string("alice"));
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({R"({"name":"bob"})", R"({"name":"carol"})", R"({"name":"dave"})"}, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  // The UDF evaluates the literal as a constant — it should succeed and
  // return true for every row since the literal contains name=alice,
  // regardless of the actual row data
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 3);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == true);
  REQUIRE(results[2] == true);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - null arguments", "[json]") {
  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({R"({})"}, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);

  SECTION("Null arrays") {
    const char* path[] = {"key"};
    LanceDBExpr* expr = lancedb_expr_json_contains(lancedb_expr_column(JSON_COL), path, 1);
    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        nullptr, ffi_schema, 1, expr, &results, &count, &error_message);
    REQUIRE(err == LANCEDB_INVALID_ARGUMENT);
    REQUIRE(error_message != nullptr);
    lancedb_free_string(error_message);
  }

  SECTION("Null schema") {
    const char* path[] = {"key"};
    LanceDBExpr* expr = lancedb_expr_json_contains(lancedb_expr_column(JSON_COL), path, 1);
    FFI_ArrowArray* arrays[] = {ffi_array};
    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, nullptr, 1, expr, &results, &count, &error_message);
    REQUIRE(err == LANCEDB_INVALID_ARGUMENT);
    REQUIRE(error_message != nullptr);
    lancedb_free_string(error_message);
  }

  SECTION("Null expr") {
    FFI_ArrowArray* arrays[] = {ffi_array};
    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, ffi_schema, 1, nullptr, &results, &count, &error_message);
    REQUIRE(err == LANCEDB_INVALID_ARGUMENT);
    REQUIRE(error_message != nullptr);
    lancedb_free_string(error_message);
  }

  SECTION("Null results_out") {
    const char* path[] = {"key"};
    LanceDBExpr* expr = lancedb_expr_json_contains(lancedb_expr_column(JSON_COL), path, 1);
    FFI_ArrowArray* arrays[] = {ffi_array};
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, ffi_schema, 1, expr, nullptr, &count, &error_message);
    REQUIRE(err == LANCEDB_INVALID_ARGUMENT);
    REQUIRE(error_message != nullptr);
    lancedb_free_string(error_message);
  }

  SECTION("Zero batch_count") {
    const char* path[] = {"key"};
    LanceDBExpr* expr = lancedb_expr_json_contains(lancedb_expr_column(JSON_COL), path, 1);
    FFI_ArrowArray* arrays[] = {ffi_array};
    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, ffi_schema, 0, expr, &results, &count, &error_message);
    REQUIRE(err == LANCEDB_INVALID_ARGUMENT);
    REQUIRE(error_message != nullptr);
    lancedb_free_string(error_message);
  }

  SECTION("Null entry in arrays") {
    const char* path[] = {"key"};
    LanceDBExpr* expr = lancedb_expr_json_contains(lancedb_expr_column(JSON_COL), path, 1);
    FFI_ArrowArray* arrays[] = {ffi_array, nullptr};
    bool* results = nullptr;
    size_t count = 0;
    char* error_message = nullptr;
    LanceDBError err = lancedb_json_matches(
        arrays, ffi_schema, 2, expr, &results, &count, &error_message);
    REQUIRE(err == LANCEDB_INVALID_ARGUMENT);
    REQUIRE(error_message != nullptr);
    lancedb_free_string(error_message);
  }

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON expr builders - null arguments", "[json]") {
  SECTION("Null json_expr") {
    const char* path[] = {"key"};
    REQUIRE(lancedb_expr_json_get_str(nullptr, path, 1) == nullptr);
    REQUIRE(lancedb_expr_json_get_int(nullptr, path, 1) == nullptr);
    REQUIRE(lancedb_expr_json_get_float(nullptr, path, 1) == nullptr);
    REQUIRE(lancedb_expr_json_get_bool(nullptr, path, 1) == nullptr);
    REQUIRE(lancedb_expr_json_contains(nullptr, path, 1) == nullptr);
  }

  SECTION("Null path") {
    REQUIRE(lancedb_expr_json_get_str(lancedb_expr_column(JSON_COL), nullptr, 1) == nullptr);
    REQUIRE(lancedb_expr_json_get_int(lancedb_expr_column(JSON_COL), nullptr, 1) == nullptr);
    REQUIRE(lancedb_expr_json_get_float(lancedb_expr_column(JSON_COL), nullptr, 1) == nullptr);
    REQUIRE(lancedb_expr_json_get_bool(lancedb_expr_column(JSON_COL), nullptr, 1) == nullptr);
    REQUIRE(lancedb_expr_json_contains(lancedb_expr_column(JSON_COL), nullptr, 1) == nullptr);
  }

  SECTION("Zero path_len") {
    const char* path[] = {"key"};
    REQUIRE(lancedb_expr_json_get_str(lancedb_expr_column(JSON_COL), path, 0) == nullptr);
    REQUIRE(lancedb_expr_json_get_int(lancedb_expr_column(JSON_COL), path, 0) == nullptr);
    REQUIRE(lancedb_expr_json_get_float(lancedb_expr_column(JSON_COL), path, 0) == nullptr);
    REQUIRE(lancedb_expr_json_get_bool(lancedb_expr_column(JSON_COL), path, 0) == nullptr);
    REQUIRE(lancedb_expr_json_contains(lancedb_expr_column(JSON_COL), path, 0) == nullptr);
  }

  SECTION("Null element in path array") {
    const char* path[] = {nullptr};
    REQUIRE(lancedb_expr_json_get_str(lancedb_expr_column(JSON_COL), path, 1) == nullptr);
    REQUIRE(lancedb_expr_json_get_int(lancedb_expr_column(JSON_COL), path, 1) == nullptr);
    REQUIRE(lancedb_expr_json_get_float(lancedb_expr_column(JSON_COL), path, 1) == nullptr);
    REQUIRE(lancedb_expr_json_get_bool(lancedb_expr_column(JSON_COL), path, 1) == nullptr);
    REQUIRE(lancedb_expr_json_contains(lancedb_expr_column(JSON_COL), path, 1) == nullptr);
  }
}

TEST_CASE("JSON expr builders - invalid UTF-8 path", "[json]") {
  const char bad_path[] = {'\xff', '\xfe', '\0'};
  const char* path[] = {bad_path};

  REQUIRE(lancedb_expr_json_get_str(lancedb_expr_column(JSON_COL), path, 1) == nullptr);
  REQUIRE(lancedb_expr_json_get_int(lancedb_expr_column(JSON_COL), path, 1) == nullptr);
  REQUIRE(lancedb_expr_json_get_float(lancedb_expr_column(JSON_COL), path, 1) == nullptr);
  REQUIRE(lancedb_expr_json_get_bool(lancedb_expr_column(JSON_COL), path, 1) == nullptr);
  REQUIRE(lancedb_expr_json_contains(lancedb_expr_column(JSON_COL), path, 1) == nullptr);
}

TEST_CASE("JSON matches - multiple batches", "[json]") {
  const char* path[] = {"val"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_int(lancedb_expr_column(JSON_COL), path, 1),
      LANCEDB_BINARY_OP_GT,
      lancedb_expr_literal_i64(5));
  REQUIRE(expr != nullptr);

  // Export two separate batches
  ArrowArray c_array1, c_array2;
  ArrowSchema c_schema1, c_schema2;
  export_json_batch({R"({"val":10})", R"({"val":2})"}, JSON_COL, &c_array1, &c_schema1);
  export_json_batch({R"({"val":3})", R"({"val":8})", R"({"val":6})"}, JSON_COL, &c_array2, &c_schema2);

  auto* ffi_array1 = reinterpret_cast<FFI_ArrowArray*>(&c_array1);
  auto* ffi_array2 = reinterpret_cast<FFI_ArrowArray*>(&c_array2);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema1);
  FFI_ArrowArray* arrays[] = {ffi_array1, ffi_array2};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 2, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 5);
  // batch1: 10>5=true, 2>5=false; batch2: 3>5=false, 8>5=true, 6>5=true
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  REQUIRE(results[2] == false);
  REQUIRE(results[3] == true);
  REQUIRE(results[4] == true);
  lancedb_free_json_matches(results);

  if (c_array1.release) c_array1.release(&c_array1);
  if (c_array2.release) c_array2.release(&c_array2);
  if (c_schema1.release) c_schema1.release(&c_schema1);
  if (c_schema2.release) c_schema2.release(&c_schema2);
}

TEST_CASE("JSON matches - arrays still valid after call", "[json]") {
  const char* path[] = {"key"};
  LanceDBExpr* expr = lancedb_expr_json_contains(lancedb_expr_column(JSON_COL), path, 1);
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({R"({"key":"val"})"}, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(results[0] == true);
  lancedb_free_json_matches(results);

  // Arrays are still valid — can import them in C++
  REQUIRE(c_array.release != nullptr);
  auto imported_schema = arrow::ImportSchema(reinterpret_cast<ArrowSchema*>(ffi_schema));
  REQUIRE(imported_schema.ok());
  auto imported_batch = arrow::ImportRecordBatch(&c_array, imported_schema.ValueUnsafe());
  REQUIRE(imported_batch.ok());
  REQUIRE(imported_batch.ValueUnsafe()->num_rows() == 1);

  // c_array consumed by ImportRecordBatch, c_schema consumed by ImportSchema
}

TEST_CASE("JSON matches - invalid JSON input", "[json]") {
  const char* path[] = {"name"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_str(lancedb_expr_column(JSON_COL), path, 1),
      LANCEDB_BINARY_OP_EQ,
      lancedb_expr_literal_string("alice"));
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"(}{"name":"alice"})",
    R"({"name":"alice"})",
    "not json at all",
    "{broken",
    "",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 5);
  // Only the valid JSON with name=alice should match
  REQUIRE(results[0] == false);
  REQUIRE(results[1] == true);
  REQUIRE(results[2] == false);
  REQUIRE(results[3] == false);
  REQUIRE(results[4] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - invalid nested JSON input", "[json]") {
  // Filter on top-level key "name" — should match even when nested value is broken
  const char* path[] = {"name"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_str(lancedb_expr_column(JSON_COL), path, 1),
      LANCEDB_BINARY_OP_EQ,
      lancedb_expr_literal_string("alice"));
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"name":"alice","nested":{"broken)",
    R"({"name":"alice","nested":{}})",
    R"({"name":"bob","nested":"not an object"})",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 3);
  // Row 0: nested value is broken, but top-level "name" is extracted before the
  // parser reaches the malformed part — match
  REQUIRE(results[0] == true);
  // Row 1: valid JSON, name=alice — match
  REQUIRE(results[1] == true);
  // Row 2: valid JSON, name=bob — no match
  REQUIRE(results[2] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - nested string extraction", "[json]") {
  const char* path[] = {"user", "name"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_str(lancedb_expr_column(JSON_COL), path, 2),
      LANCEDB_BINARY_OP_EQ,
      lancedb_expr_literal_string("alice"));
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"user":{"name":"alice","age":30}})",
    R"({"user":{"name":"bob","age":40}})",
    R"({"user":{"age":25}})",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 3);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  REQUIRE(results[2] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - nested integer extraction", "[json]") {
  const char* path[] = {"stats", "count"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_int(lancedb_expr_column(JSON_COL), path, 2),
      LANCEDB_BINARY_OP_GT,
      lancedb_expr_literal_i64(10));
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"stats":{"count":20}})",
    R"({"stats":{"count":5}})",
    R"({"stats":{"count":15}})",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 3);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  REQUIRE(results[2] == true);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - deeply nested path", "[json]") {
  const char* path[] = {"a", "b", "c"};
  LanceDBExpr* expr = lancedb_expr_binary(
      lancedb_expr_json_get_str(lancedb_expr_column(JSON_COL), path, 3),
      LANCEDB_BINARY_OP_EQ,
      lancedb_expr_literal_string("found"));
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"a":{"b":{"c":"found"}}})",
    R"({"a":{"b":{"c":"other"}}})",
    R"({"a":{"b":{}}})",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 3);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  REQUIRE(results[2] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

TEST_CASE("JSON matches - nested json_contains", "[json]") {
  const char* path[] = {"user", "email"};
  LanceDBExpr* expr = lancedb_expr_json_contains(lancedb_expr_column(JSON_COL), path, 2);
  REQUIRE(expr != nullptr);

  ArrowArray c_array;
  ArrowSchema c_schema;
  export_json_batch({
    R"({"user":{"email":"a@b.com"}})",
    R"({"user":{"name":"bob"}})",
    R"({"other":true})",
  }, JSON_COL, &c_array, &c_schema);

  auto* ffi_array = reinterpret_cast<FFI_ArrowArray*>(&c_array);
  auto* ffi_schema = reinterpret_cast<FFI_ArrowSchema*>(&c_schema);
  FFI_ArrowArray* arrays[] = {ffi_array};

  bool* results = nullptr;
  size_t count = 0;
  char* error_message = nullptr;
  LanceDBError err = lancedb_json_matches(
      arrays, ffi_schema, 1, expr, &results, &count, &error_message);
  REQUIRE(err == LANCEDB_SUCCESS);
  REQUIRE(count == 3);
  REQUIRE(results[0] == true);
  REQUIRE(results[1] == false);
  REQUIRE(results[2] == false);
  lancedb_free_json_matches(results);

  if (c_array.release) c_array.release(&c_array);
  if (c_schema.release) c_schema.release(&c_schema);
}

