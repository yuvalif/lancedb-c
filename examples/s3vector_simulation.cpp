/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright The LanceDB Authors
 *
 * S3 Vector API Simulation using LanceDB C Bindings
 *
 * This application simulates the AWS S3 Vector API using local filesystem
 * and LanceDB for vector storage and search operations.
 *
 * Mapping:
 * - Vector Bucket -> Local directory at /tmp/s3vectors/<bucket-name>
 * - Index -> LanceDB table within the bucket directory
 * - Vectors -> Rows in the LanceDB table with schema:
 *     - key (string): unique vector identifier
 *     - data (fixed_size_list<float32>): vector embedding
 *     - metadata (string): JSON-encoded metadata
 *
 * Limitations:
 * - No actual encryption support (encryption config is stored but not applied)
 * - No IAM/policy enforcement (policies are stored but not enforced)
 * - No concurrent write conflict resolution (relies on LanceDB's built-in handling)
 * - Tags and policies are stored as JSON files, not in a database
 * - No ARN generation (uses simple path-based identifiers)
 */

#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <nlohmann/json.hpp>
#include "lancedb.h"

using json = nlohmann::json;

// ============================================================================
// Constants and Configuration
// ============================================================================

const std::string S3_VECTORS_ROOT = "/tmp/s3vectors";
const std::string METADATA_DIR_NAME = ".s3v_metadata";
const std::string POLICY_FILE_NAME = "policy.json";
const std::string TAGS_FILE_NAME = "tags.json";
const std::string BUCKET_CONFIG_FILE = "bucket_config.json";
const std::string INDEX_CONFIG_SUFFIX = "_index_config.json";

// ============================================================================
// Utility Functions
// ============================================================================

namespace utils {

bool directory_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool create_directory(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0;
}

bool create_directories(const std::string& path) {
    std::string current;
    std::istringstream stream(path);
    std::string segment;

    while (std::getline(stream, segment, '/')) {
        if (segment.empty()) {
            current = "/";
            continue;
        }
        current += segment + "/";
        if (!directory_exists(current)) {
            if (!create_directory(current)) {
                return false;
            }
        }
    }
    return true;
}

bool remove_directory_recursive(const std::string& path) {
    std::string command = "rm -rf \"" + path + "\"";
    return system(command.c_str()) == 0;
}

std::vector<std::string> list_directories(const std::string& path) {
    std::vector<std::string> dirs;
    DIR* dir = opendir(path.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_DIR) {
                std::string name = entry->d_name;
                if (name != "." && name != ".." && name[0] != '.') {
                    dirs.push_back(name);
                }
            }
        }
        closedir(dir);
    }
    return dirs;
}

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool write_file(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }
    file << content;
    return file.good();
}

bool delete_file(const std::string& path) {
    return unlink(path.c_str()) == 0;
}

std::string get_bucket_path(const std::string& bucket_name) {
    return S3_VECTORS_ROOT + "/" + bucket_name;
}

std::string get_metadata_path(const std::string& bucket_name) {
    return get_bucket_path(bucket_name) + "/" + METADATA_DIR_NAME;
}

std::string get_index_db_path(const std::string& bucket_name, const std::string& index_name) {
    return get_bucket_path(bucket_name) + "/" + index_name + "_lancedb";
}

std::string get_current_timestamp() {
    time_t now = time(nullptr);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    return buf;
}

bool validate_name(const std::string& name, std::string& error) {
    if (name.length() < 3 || name.length() > 63) {
        error = "Name must be between 3 and 63 characters";
        return false;
    }
    for (char c : name) {
        if (!std::isalnum(c) && c != '-' && c != '_') {
            error = "Name contains invalid characters. Only alphanumeric, hyphens, and underscores allowed";
            return false;
        }
    }
    return true;
}

} // namespace utils

// ============================================================================
// Response Types
// ============================================================================

struct ApiResponse {
    int status_code = 200;
    json body;
    std::string error_type;
    std::string error_message;

    bool is_success() const { return status_code >= 200 && status_code < 300; }

    std::string to_string() const {
        if (is_success()) {
            return body.dump(2);
        } else {
            json error_response = {
                {"error", {
                    {"type", error_type},
                    {"message", error_message}
                }}
            };
            return error_response.dump(2);
        }
    }
};

ApiResponse make_error(int code, const std::string& type, const std::string& message) {
    ApiResponse resp;
    resp.status_code = code;
    resp.error_type = type;
    resp.error_message = message;
    return resp;
}

ApiResponse make_success(const json& body = json::object()) {
    ApiResponse resp;
    resp.status_code = 200;
    resp.body = body;
    return resp;
}

// ============================================================================
// LanceDB Helper Class
// ============================================================================

class LanceDBHelper {
public:
    static std::shared_ptr<arrow::Schema> create_vector_schema(int dimension) {
        auto key_field = arrow::field("key", arrow::utf8());
        auto data_field = arrow::field("data", arrow::fixed_size_list(arrow::float32(), dimension));
        auto metadata_field = arrow::field("metadata", arrow::utf8());
        // Additional scalar columns for filtering
        auto function_name_field = arrow::field("function_name", arrow::utf8());
        auto class_name_field = arrow::field("class_name", arrow::utf8());
        return arrow::schema({key_field, data_field, metadata_field, function_name_field, class_name_field});
    }

    static LanceDBConnection* connect(const std::string& db_path) {
        LanceDBConnectBuilder* builder = lancedb_connect(db_path.c_str());
        if (!builder) {
            return nullptr;
        }
        LanceDBConnection* conn = lancedb_connect_builder_execute(builder);
        return conn;
    }

    static LanceDBTable* create_table(LanceDBConnection* conn, const std::string& table_name,
                                       int dimension, char** error_msg) {
        auto schema = create_vector_schema(dimension);
        struct ArrowSchema c_schema;
        if (!arrow::ExportSchema(*schema, &c_schema).ok()) {
            return nullptr;
        }

        LanceDBTable* table = nullptr;
        LanceDBError result = lancedb_table_create(
            conn, table_name.c_str(),
            reinterpret_cast<FFI_ArrowSchema*>(&c_schema),
            nullptr, &table, error_msg);

        if (c_schema.release) {
            c_schema.release(&c_schema);
        }

        if (result != LANCEDB_SUCCESS) {
            return nullptr;
        }
        return table;
    }

    static LanceDBTable* open_table(LanceDBConnection* conn, const std::string& table_name) {
        return lancedb_connection_open_table(conn, table_name.c_str());
    }

    static bool add_vectors(LanceDBTable* table,
                           const std::vector<std::string>& keys,
                           const std::vector<std::vector<float>>& vectors,
                           const std::vector<std::string>& metadata_list,
                           const std::vector<std::string>& function_names,
                           const std::vector<std::string>& class_names,
                           int dimension,
                           std::string& error) {
        if (keys.empty()) {
            error = "No vectors to add";
            return false;
        }

        auto schema = create_vector_schema(dimension);

        // Build arrays
        arrow::StringBuilder key_builder;
        arrow::FixedSizeListBuilder data_builder(
            arrow::default_memory_pool(),
            std::make_unique<arrow::FloatBuilder>(),
            dimension);
        arrow::StringBuilder metadata_builder;
        arrow::StringBuilder function_name_builder;
        arrow::StringBuilder class_name_builder;

        for (size_t i = 0; i < keys.size(); i++) {
            if (!key_builder.Append(keys[i]).ok()) {
                error = "Failed to append key";
                return false;
            }

            auto* float_builder = static_cast<arrow::FloatBuilder*>(data_builder.value_builder());
            for (int j = 0; j < dimension; j++) {
                if (!float_builder->Append(vectors[i][j]).ok()) {
                    error = "Failed to append vector value";
                    return false;
                }
            }
            if (!data_builder.Append().ok()) {
                error = "Failed to append vector";
                return false;
            }

            if (!metadata_builder.Append(metadata_list[i]).ok()) {
                error = "Failed to append metadata";
                return false;
            }

            // Append function_name (use empty string if not provided)
            std::string fn = (i < function_names.size()) ? function_names[i] : "";
            if (!function_name_builder.Append(fn).ok()) {
                error = "Failed to append function_name";
                return false;
            }

            // Append class_name (use empty string if not provided)
            std::string cn = (i < class_names.size()) ? class_names[i] : "";
            if (!class_name_builder.Append(cn).ok()) {
                error = "Failed to append class_name";
                return false;
            }
        }

        std::shared_ptr<arrow::Array> key_array, data_array, metadata_array;
        std::shared_ptr<arrow::Array> function_name_array, class_name_array;
        (void)key_builder.Finish(&key_array);
        (void)data_builder.Finish(&data_array);
        (void)metadata_builder.Finish(&metadata_array);
        (void)function_name_builder.Finish(&function_name_array);
        (void)class_name_builder.Finish(&class_name_array);

        auto record_batch = arrow::RecordBatch::Make(
            schema, keys.size(), {key_array, data_array, metadata_array, function_name_array, class_name_array});

        struct ArrowSchema c_schema;
        struct ArrowArray c_array;
        if (!arrow::ExportRecordBatch(*record_batch, &c_array, &c_schema).ok()) {
            error = "Failed to export record batch";
            return false;
        }

        LanceDBRecordBatchReader* reader = lancedb_record_batch_reader_from_arrow(
            reinterpret_cast<FFI_ArrowArray*>(&c_array),
            reinterpret_cast<FFI_ArrowSchema*>(&c_schema));

        if (!reader) {
            if (c_schema.release) c_schema.release(&c_schema);
            if (c_array.release) c_array.release(&c_array);
            error = "Failed to create record batch reader";
            return false;
        }

        char* err_msg = nullptr;
        LanceDBError result = lancedb_table_add(table, reader, &err_msg);

        if (c_schema.release) c_schema.release(&c_schema);

        if (result != LANCEDB_SUCCESS) {
            error = err_msg ? err_msg : lancedb_error_to_message(result);
            if (err_msg) lancedb_free_string(err_msg);
            return false;
        }

        return true;
    }

    struct QueryResult {
        std::string key;
        std::vector<float> data;
        std::string metadata;
        float distance = 0.0f;
    };

    static std::vector<QueryResult> query_vectors(
        LanceDBTable* table,
        const std::vector<float>& query_vector,
        int top_k,
        const std::string& filter,
        bool return_distance,
        bool return_metadata,
        LanceDBDistanceType distance_type,
        std::string& error) {

        std::vector<QueryResult> results;

        LanceDBVectorQuery* query = lancedb_vector_query_new(
            table, query_vector.data(), query_vector.size());

        if (!query) {
            error = "Failed to create vector query";
            return results;
        }

        if (lancedb_vector_query_limit(query, top_k, nullptr) != LANCEDB_SUCCESS) {
            lancedb_vector_query_free(query);
            error = "Failed to set query limit";
            return results;
        }

        if (lancedb_vector_query_column(query, "data", nullptr) != LANCEDB_SUCCESS) {
            lancedb_vector_query_free(query);
            error = "Failed to set query column";
            return results;
        }

        if (lancedb_vector_query_distance_type(query, distance_type, nullptr) != LANCEDB_SUCCESS) {
            lancedb_vector_query_free(query);
            error = "Failed to set distance type";
            return results;
        }

        if (!filter.empty()) {
            if (lancedb_vector_query_where_filter(query, filter.c_str(), nullptr) != LANCEDB_SUCCESS) {
                lancedb_vector_query_free(query);
                error = "Failed to set query filter";
                return results;
            }
        }

        LanceDBQueryResult* query_result = lancedb_vector_query_execute(query);
        if (!query_result) {
            error = "Failed to execute query";
            return results;
        }

        struct ArrowArray** c_arrays = nullptr;
        struct ArrowSchema* c_schema = nullptr;
        size_t count = 0;

        if (lancedb_query_result_to_arrow(
                query_result,
                reinterpret_cast<FFI_ArrowArray***>(&c_arrays),
                reinterpret_cast<FFI_ArrowSchema**>(&c_schema),
                &count, nullptr) != LANCEDB_SUCCESS) {
            lancedb_query_result_free(query_result);
            error = "Failed to convert query result to arrow";
            return results;
        }

        // Parse results
        if (count > 0 && c_arrays && c_schema) {
            auto schema_result = arrow::ImportSchema(c_schema);
            if (schema_result.ok()) {
                auto schema = *schema_result;
                auto batch_result = arrow::ImportRecordBatch(
                    reinterpret_cast<struct ArrowArray*>(*c_arrays), schema);

                if (batch_result.ok()) {
                    auto batch = *batch_result;

                    // Find column indices
                    int key_idx = schema->GetFieldIndex("key");
                    int data_idx = schema->GetFieldIndex("data");
                    int metadata_idx = schema->GetFieldIndex("metadata");
                    int distance_idx = schema->GetFieldIndex("_distance");

                    for (int64_t row = 0; row < batch->num_rows(); row++) {
                        QueryResult result;

                        if (key_idx >= 0) {
                            auto key_array = std::static_pointer_cast<arrow::StringArray>(
                                batch->column(key_idx));
                            if (!key_array->IsNull(row)) {
                                result.key = key_array->GetString(row);
                            }
                        }

                        if (data_idx >= 0) {
                            auto data_array = std::static_pointer_cast<arrow::FixedSizeListArray>(
                                batch->column(data_idx));
                            if (!data_array->IsNull(row)) {
                                auto values = std::static_pointer_cast<arrow::FloatArray>(
                                    data_array->values());
                                int32_t start = data_array->value_offset(row);
                                int32_t length = data_array->value_length();
                                for (int32_t i = 0; i < length; i++) {
                                    result.data.push_back(values->Value(start + i));
                                }
                            }
                        }

                        if (metadata_idx >= 0 && return_metadata) {
                            auto metadata_array = std::static_pointer_cast<arrow::StringArray>(
                                batch->column(metadata_idx));
                            if (!metadata_array->IsNull(row)) {
                                result.metadata = metadata_array->GetString(row);
                            }
                        }

                        if (distance_idx >= 0 && return_distance) {
                            auto distance_array = std::static_pointer_cast<arrow::FloatArray>(
                                batch->column(distance_idx));
                            if (!distance_array->IsNull(row)) {
                                result.distance = distance_array->Value(row);
                            }
                        }

                        results.push_back(result);
                    }
                }
            }
        }

        if (c_arrays) lancedb_free_arrow_arrays(reinterpret_cast<FFI_ArrowArray**>(c_arrays), count);
        if (c_schema) lancedb_free_arrow_schema(reinterpret_cast<FFI_ArrowSchema*>(c_schema));

        return results;
    }

    static std::vector<QueryResult> get_vectors_by_keys(
        LanceDBTable* table,
        const std::vector<std::string>& keys,
        [[maybe_unused]] int dimension,
        std::string& error) {

        std::vector<QueryResult> results;

        // Build filter for keys
        std::string filter;
        for (size_t i = 0; i < keys.size(); i++) {
            if (i > 0) filter += " OR ";
            filter += "key = \"" + keys[i] + "\"";
        }

        LanceDBQuery* query = lancedb_query_new(table);
        if (!query) {
            error = "Failed to create query";
            return results;
        }

        if (lancedb_query_where_filter(query, filter.c_str(), nullptr) != LANCEDB_SUCCESS) {
            lancedb_query_free(query);
            error = "Failed to set query filter";
            return results;
        }

        LanceDBQueryResult* query_result = lancedb_query_execute(query);
        if (!query_result) {
            error = "Failed to execute query";
            return results;
        }

        struct ArrowArray** c_arrays = nullptr;
        struct ArrowSchema* c_schema = nullptr;
        size_t count = 0;

        if (lancedb_query_result_to_arrow(
                query_result,
                reinterpret_cast<FFI_ArrowArray***>(&c_arrays),
                reinterpret_cast<FFI_ArrowSchema**>(&c_schema),
                &count, nullptr) != LANCEDB_SUCCESS) {
            lancedb_query_result_free(query_result);
            error = "Failed to convert query result";
            return results;
        }

        // Parse results (similar to query_vectors)
        if (count > 0 && c_arrays && c_schema) {
            auto schema_result = arrow::ImportSchema(c_schema);
            if (schema_result.ok()) {
                auto schema = *schema_result;
                auto batch_result = arrow::ImportRecordBatch(
                    reinterpret_cast<struct ArrowArray*>(*c_arrays), schema);

                if (batch_result.ok()) {
                    auto batch = *batch_result;

                    int key_idx = schema->GetFieldIndex("key");
                    int data_idx = schema->GetFieldIndex("data");
                    int metadata_idx = schema->GetFieldIndex("metadata");

                    for (int64_t row = 0; row < batch->num_rows(); row++) {
                        QueryResult result;

                        if (key_idx >= 0) {
                            auto key_array = std::static_pointer_cast<arrow::StringArray>(
                                batch->column(key_idx));
                            if (!key_array->IsNull(row)) {
                                result.key = key_array->GetString(row);
                            }
                        }

                        if (data_idx >= 0) {
                            auto data_array = std::static_pointer_cast<arrow::FixedSizeListArray>(
                                batch->column(data_idx));
                            if (!data_array->IsNull(row)) {
                                auto values = std::static_pointer_cast<arrow::FloatArray>(
                                    data_array->values());
                                int32_t start = data_array->value_offset(row);
                                int32_t length = data_array->value_length();
                                for (int32_t i = 0; i < length; i++) {
                                    result.data.push_back(values->Value(start + i));
                                }
                            }
                        }

                        if (metadata_idx >= 0) {
                            auto metadata_array = std::static_pointer_cast<arrow::StringArray>(
                                batch->column(metadata_idx));
                            if (!metadata_array->IsNull(row)) {
                                result.metadata = metadata_array->GetString(row);
                            }
                        }

                        results.push_back(result);
                    }
                }
            }
        }

        if (c_arrays) lancedb_free_arrow_arrays(reinterpret_cast<FFI_ArrowArray**>(c_arrays), count);
        if (c_schema) lancedb_free_arrow_schema(reinterpret_cast<FFI_ArrowSchema*>(c_schema));

        return results;
    }
};

// ============================================================================
// S3 Vector API Implementations
// ============================================================================

// CreateVectorBucket: Creates a vector bucket (local directory)
ApiResponse CreateVectorBucket(const json& request) {
    // Validate request
    if (!request.contains("vectorBucketName")) {
        return make_error(400, "ValidationException", "vectorBucketName is required");
    }

    std::string bucket_name = request["vectorBucketName"].get<std::string>();
    std::string error;
    if (!utils::validate_name(bucket_name, error)) {
        return make_error(400, "ValidationException", error);
    }

    std::string bucket_path = utils::get_bucket_path(bucket_name);

    // Check if bucket already exists
    if (utils::directory_exists(bucket_path)) {
        return make_error(409, "ConflictException",
            "Vector bucket '" + bucket_name + "' already exists");
    }

    // Create bucket directory
    if (!utils::create_directories(bucket_path)) {
        return make_error(500, "InternalServerException",
            "Failed to create vector bucket directory");
    }

    // Create metadata directory
    std::string metadata_path = utils::get_metadata_path(bucket_name);
    if (!utils::create_directory(metadata_path)) {
        utils::remove_directory_recursive(bucket_path);
        return make_error(500, "InternalServerException",
            "Failed to create metadata directory");
    }

    // Store bucket configuration
    json config;
    config["vectorBucketName"] = bucket_name;
    config["creationDate"] = utils::get_current_timestamp();

    if (request.contains("encryptionConfiguration")) {
        config["encryptionConfiguration"] = request["encryptionConfiguration"];
    } else {
        config["encryptionConfiguration"] = {
            {"sseType", "AES256"}
        };
    }

    std::string config_path = metadata_path + "/" + BUCKET_CONFIG_FILE;
    if (!utils::write_file(config_path, config.dump(2))) {
        utils::remove_directory_recursive(bucket_path);
        return make_error(500, "InternalServerException",
            "Failed to save bucket configuration");
    }

    // Handle initial tags if provided
    if (request.contains("tags")) {
        std::string tags_path = metadata_path + "/" + TAGS_FILE_NAME;
        utils::write_file(tags_path, request["tags"].dump(2));
    }

    // Generate ARN (simulated)
    std::string arn = "arn:aws:s3vectors:local:000000000000:vector-bucket/" + bucket_name;

    return make_success({
        {"vectorBucketArn", arn}
    });
}

// DeleteVectorBucket: Deletes a vector bucket
ApiResponse DeleteVectorBucket(const json& request) {
    std::string bucket_name;

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        return make_error(400, "ValidationException",
            "vectorBucketName is required");
    }

    std::string bucket_path = utils::get_bucket_path(bucket_name);

    if (!utils::directory_exists(bucket_path)) {
        return make_error(404, "NotFoundException",
            "Vector bucket '" + bucket_name + "' not found");
    }

    // Check if bucket has any indices
    auto dirs = utils::list_directories(bucket_path);
    for (const auto& dir : dirs) {
        if (dir.find("_lancedb") != std::string::npos) {
            return make_error(409, "ConflictException",
                "Cannot delete bucket with existing indices. Delete indices first.");
        }
    }

    if (!utils::remove_directory_recursive(bucket_path)) {
        return make_error(500, "InternalServerException",
            "Failed to delete vector bucket");
    }

    return make_success();
}

// GetVectorBucket: Gets information about a vector bucket
ApiResponse GetVectorBucket(const json& request) {
    std::string bucket_name;

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        return make_error(400, "ValidationException",
            "vectorBucketName is required");
    }

    std::string bucket_path = utils::get_bucket_path(bucket_name);

    if (!utils::directory_exists(bucket_path)) {
        return make_error(404, "NotFoundException",
            "Vector bucket '" + bucket_name + "' not found");
    }

    std::string config_path = utils::get_metadata_path(bucket_name) + "/" + BUCKET_CONFIG_FILE;
    std::string config_str = utils::read_file(config_path);

    json response;
    if (!config_str.empty()) {
        try {
            json config = json::parse(config_str);
            response["vectorBucketName"] = bucket_name;
            response["vectorBucketArn"] = "arn:aws:s3vectors:local:000000000000:vector-bucket/" + bucket_name;
            response["creationDate"] = config.value("creationDate", "");
            response["encryptionConfiguration"] = config.value("encryptionConfiguration", json::object());
        } catch (...) {
            response["vectorBucketName"] = bucket_name;
        }
    } else {
        response["vectorBucketName"] = bucket_name;
    }

    return make_success(response);
}

// ListVectorBuckets: Lists all vector buckets
ApiResponse ListVectorBuckets([[maybe_unused]] const json& request) {
    if (!utils::directory_exists(S3_VECTORS_ROOT)) {
        return make_success({{"vectorBuckets", json::array()}});
    }

    auto buckets = utils::list_directories(S3_VECTORS_ROOT);

    json bucket_list = json::array();
    for (const auto& bucket_name : buckets) {
        json bucket_info;
        bucket_info["vectorBucketName"] = bucket_name;
        bucket_info["vectorBucketArn"] = "arn:aws:s3vectors:local:000000000000:vector-bucket/" + bucket_name;

        std::string config_path = utils::get_metadata_path(bucket_name) + "/" + BUCKET_CONFIG_FILE;
        std::string config_str = utils::read_file(config_path);
        if (!config_str.empty()) {
            try {
                json config = json::parse(config_str);
                bucket_info["creationDate"] = config.value("creationDate", "");
            } catch (...) {}
        }

        bucket_list.push_back(bucket_info);
    }

    return make_success({{"vectorBuckets", bucket_list}});
}

// CreateIndex: Creates a vector index within a bucket
ApiResponse CreateIndex(const json& request) {
    // Validate required fields
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("dimension")) {
        return make_error(400, "ValidationException", "dimension is required");
    }
    if (!request.contains("distanceMetric")) {
        return make_error(400, "ValidationException", "distanceMetric is required");
    }
    if (!request.contains("vectorBucketName") && !request.contains("vectorBucketArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or vectorBucketArn is required");
    }

    std::string bucket_name;
    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        // Extract bucket name from ARN (simplified parsing)
        std::string arn = request["vectorBucketArn"].get<std::string>();
        size_t pos = arn.rfind('/');
        if (pos != std::string::npos) {
            bucket_name = arn.substr(pos + 1);
        }
    }

    std::string index_name = request["indexName"].get<std::string>();
    int dimension = request["dimension"].get<int>();
    std::string distance_metric = request["distanceMetric"].get<std::string>();
    std::string data_type = request.value("dataType", "float32");

    // Validate names
    std::string error;
    if (!utils::validate_name(index_name, error)) {
        return make_error(400, "ValidationException", "indexName: " + error);
    }

    // Validate dimension
    if (dimension < 1 || dimension > 4096) {
        return make_error(400, "ValidationException",
            "dimension must be between 1 and 4096");
    }

    // Validate distance metric
    if (distance_metric != "euclidean" && distance_metric != "cosine") {
        return make_error(400, "ValidationException",
            "distanceMetric must be 'euclidean' or 'cosine'");
    }

    // Check bucket exists
    std::string bucket_path = utils::get_bucket_path(bucket_name);
    if (!utils::directory_exists(bucket_path)) {
        return make_error(404, "NotFoundException",
            "Vector bucket '" + bucket_name + "' not found");
    }

    // Check if index already exists
    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    if (utils::directory_exists(db_path)) {
        return make_error(409, "ConflictException",
            "Index '" + index_name + "' already exists");
    }

    // Create LanceDB database and table
    LanceDBConnection* conn = LanceDBHelper::connect(db_path);
    if (!conn) {
        return make_error(500, "InternalServerException",
            "Failed to create LanceDB connection");
    }

    char* err_msg = nullptr;
    LanceDBTable* table = LanceDBHelper::create_table(conn, "vectors", dimension, &err_msg);
    if (!table) {
        std::string error_str = err_msg ? err_msg : "Failed to create table";
        if (err_msg) lancedb_free_string(err_msg);
        lancedb_connection_free(conn);
        utils::remove_directory_recursive(db_path);
        return make_error(500, "InternalServerException", error_str);
    }

    // Store index configuration
    json config;
    config["indexName"] = index_name;
    config["dimension"] = dimension;
    config["distanceMetric"] = distance_metric;
    config["dataType"] = data_type;
    config["creationDate"] = utils::get_current_timestamp();
    config["vectorBucketName"] = bucket_name;

    if (request.contains("metadataConfiguration")) {
        config["metadataConfiguration"] = request["metadataConfiguration"];
    }

    std::string config_path = utils::get_metadata_path(bucket_name) + "/" + index_name + INDEX_CONFIG_SUFFIX;
    utils::write_file(config_path, config.dump(2));

    lancedb_table_free(table);
    lancedb_connection_free(conn);

    std::string index_arn = "arn:aws:s3vectors:local:000000000000:vector-bucket/" +
                           bucket_name + "/index/" + index_name;

    return make_success({
        {"indexArn", index_arn}
    });
}

// DeleteIndex: Deletes a vector index
ApiResponse DeleteIndex(const json& request) {
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName") && !request.contains("indexArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or indexArn is required");
    }

    std::string bucket_name;
    std::string index_name = request["indexName"].get<std::string>();

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        // Extract from ARN
        std::string arn = request["indexArn"].get<std::string>();
        // Parse ARN: arn:aws:s3vectors:local:000000000000:vector-bucket/BUCKET/index/INDEX
        size_t bucket_pos = arn.find("vector-bucket/");
        size_t index_pos = arn.find("/index/");
        if (bucket_pos != std::string::npos && index_pos != std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14, index_pos - bucket_pos - 14);
        }
    }

    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    if (!utils::directory_exists(db_path)) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found in bucket '" + bucket_name + "'");
    }

    // Remove the LanceDB database directory
    if (!utils::remove_directory_recursive(db_path)) {
        return make_error(500, "InternalServerException",
            "Failed to delete index");
    }

    // Remove index configuration
    std::string config_path = utils::get_metadata_path(bucket_name) + "/" + index_name + INDEX_CONFIG_SUFFIX;
    utils::delete_file(config_path);

    return make_success();
}

// GetIndex: Gets information about a vector index
ApiResponse GetIndex(const json& request) {
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName") && !request.contains("indexArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or indexArn is required");
    }

    std::string bucket_name;
    std::string index_name = request["indexName"].get<std::string>();

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        std::string arn = request["indexArn"].get<std::string>();
        size_t bucket_pos = arn.find("vector-bucket/");
        size_t index_pos = arn.find("/index/");
        if (bucket_pos != std::string::npos && index_pos != std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14, index_pos - bucket_pos - 14);
        }
    }

    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    if (!utils::directory_exists(db_path)) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found");
    }

    // Read index configuration
    std::string config_path = utils::get_metadata_path(bucket_name) + "/" + index_name + INDEX_CONFIG_SUFFIX;
    std::string config_str = utils::read_file(config_path);

    json response;
    if (!config_str.empty()) {
        try {
            response = json::parse(config_str);
        } catch (...) {
            response["indexName"] = index_name;
        }
    } else {
        response["indexName"] = index_name;
    }

    response["indexArn"] = "arn:aws:s3vectors:local:000000000000:vector-bucket/" +
                          bucket_name + "/index/" + index_name;

    // Get vector count from LanceDB
    LanceDBConnection* conn = LanceDBHelper::connect(db_path);
    if (conn) {
        LanceDBTable* table = LanceDBHelper::open_table(conn, "vectors");
        if (table) {
            response["vectorCount"] = lancedb_table_count_rows(table);
            lancedb_table_free(table);
        }
        lancedb_connection_free(conn);
    }

    return make_success(response);
}

// ListIndexes: Lists all indices in a bucket
ApiResponse ListIndexes(const json& request) {
    if (!request.contains("vectorBucketName") && !request.contains("vectorBucketArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or vectorBucketArn is required");
    }

    std::string bucket_name;
    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        std::string arn = request["vectorBucketArn"].get<std::string>();
        size_t pos = arn.rfind('/');
        if (pos != std::string::npos) {
            bucket_name = arn.substr(pos + 1);
        }
    }

    std::string bucket_path = utils::get_bucket_path(bucket_name);
    if (!utils::directory_exists(bucket_path)) {
        return make_error(404, "NotFoundException",
            "Vector bucket '" + bucket_name + "' not found");
    }

    auto dirs = utils::list_directories(bucket_path);
    json indices = json::array();

    for (const auto& dir : dirs) {
        if (dir.find("_lancedb") != std::string::npos) {
            std::string index_name = dir.substr(0, dir.length() - 8); // Remove "_lancedb"

            json index_info;
            index_info["indexName"] = index_name;
            index_info["indexArn"] = "arn:aws:s3vectors:local:000000000000:vector-bucket/" +
                                    bucket_name + "/index/" + index_name;

            // Read index configuration
            std::string config_path = utils::get_metadata_path(bucket_name) + "/" +
                                     index_name + INDEX_CONFIG_SUFFIX;
            std::string config_str = utils::read_file(config_path);
            if (!config_str.empty()) {
                try {
                    json config = json::parse(config_str);
                    index_info["dimension"] = config.value("dimension", 0);
                    index_info["distanceMetric"] = config.value("distanceMetric", "");
                    index_info["creationDate"] = config.value("creationDate", "");
                } catch (...) {}
            }

            indices.push_back(index_info);
        }
    }

    return make_success({{"indexes", indices}});
}

// PutVectors: Adds vectors to an index
ApiResponse PutVectors(const json& request) {
    if (!request.contains("vectors") || !request["vectors"].is_array()) {
        return make_error(400, "ValidationException", "vectors array is required");
    }
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName") && !request.contains("indexArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or indexArn is required");
    }

    std::string bucket_name;
    std::string index_name = request["indexName"].get<std::string>();

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        std::string arn = request["indexArn"].get<std::string>();
        size_t bucket_pos = arn.find("vector-bucket/");
        size_t index_pos = arn.find("/index/");
        if (bucket_pos != std::string::npos && index_pos != std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14, index_pos - bucket_pos - 14);
        }
    }

    // Read index configuration to get dimension
    std::string config_path = utils::get_metadata_path(bucket_name) + "/" +
                             index_name + INDEX_CONFIG_SUFFIX;
    std::string config_str = utils::read_file(config_path);
    if (config_str.empty()) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found");
    }

    int dimension;
    try {
        json config = json::parse(config_str);
        dimension = config["dimension"].get<int>();
    } catch (...) {
        return make_error(500, "InternalServerException",
            "Failed to read index configuration");
    }

    // Parse vectors
    std::vector<std::string> keys;
    std::vector<std::vector<float>> vectors;
    std::vector<std::string> metadata_list;
    std::vector<std::string> function_names;
    std::vector<std::string> class_names;

    const auto& vectors_array = request["vectors"];
    if (vectors_array.size() > 500) {
        return make_error(400, "ValidationException",
            "Maximum 500 vectors per request");
    }

    for (const auto& vec : vectors_array) {
        if (!vec.contains("key")) {
            return make_error(400, "ValidationException",
                "Each vector must have a 'key' field");
        }
        if (!vec.contains("data")) {
            return make_error(400, "ValidationException",
                "Each vector must have a 'data' field");
        }

        keys.push_back(vec["key"].get<std::string>());

        std::vector<float> data;
        if (vec["data"].contains("float32")) {
            for (const auto& val : vec["data"]["float32"]) {
                data.push_back(val.get<float>());
            }
        } else if (vec["data"].is_array()) {
            for (const auto& val : vec["data"]) {
                data.push_back(val.get<float>());
            }
        }

        if (static_cast<int>(data.size()) != dimension) {
            return make_error(400, "ValidationException",
                "Vector dimension mismatch. Expected " + std::to_string(dimension) +
                ", got " + std::to_string(data.size()));
        }

        vectors.push_back(data);

        std::string metadata = "{}";
        if (vec.contains("metadata")) {
            metadata = vec["metadata"].dump();
        }
        metadata_list.push_back(metadata);

        // Extract function_name and class_name from metadata or top-level
        std::string function_name = "";
        std::string class_name = "";

        // Try top-level first
        if (vec.contains("function_name") && vec["function_name"].is_string()) {
            function_name = vec["function_name"].get<std::string>();
        }
        if (vec.contains("class_name") && vec["class_name"].is_string()) {
            class_name = vec["class_name"].get<std::string>();
        }

        // If not found, try nested metadata.metadata (for code index format)
        if (vec.contains("metadata") && vec["metadata"].is_object()) {
            const auto& meta = vec["metadata"];
            // Check direct metadata fields
            if (function_name.empty() && meta.contains("function_name") && meta["function_name"].is_string()) {
                function_name = meta["function_name"].get<std::string>();
            }
            if (class_name.empty() && meta.contains("class_name") && meta["class_name"].is_string()) {
                class_name = meta["class_name"].get<std::string>();
            }
            // Check nested metadata.metadata fields (code index format)
            if (meta.contains("metadata") && meta["metadata"].is_object()) {
                const auto& nested_meta = meta["metadata"];
                if (function_name.empty() && nested_meta.contains("function_name") && nested_meta["function_name"].is_string()) {
                    function_name = nested_meta["function_name"].get<std::string>();
                }
                if (class_name.empty() && nested_meta.contains("class_name") && nested_meta["class_name"].is_string()) {
                    class_name = nested_meta["class_name"].get<std::string>();
                }
            }
        }

        function_names.push_back(function_name);
        class_names.push_back(class_name);
    }

    // Connect to LanceDB
    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    LanceDBConnection* conn = LanceDBHelper::connect(db_path);
    if (!conn) {
        return make_error(500, "InternalServerException",
            "Failed to connect to index database");
    }

    LanceDBTable* table = LanceDBHelper::open_table(conn, "vectors");
    if (!table) {
        lancedb_connection_free(conn);
        return make_error(500, "InternalServerException",
            "Failed to open vectors table");
    }

    std::string error;
    bool success = LanceDBHelper::add_vectors(table, keys, vectors, metadata_list,
                                               function_names, class_names, dimension, error);

    lancedb_table_free(table);
    lancedb_connection_free(conn);

    if (!success) {
        return make_error(500, "InternalServerException", error);
    }

    return make_success();
}

// GetVectors: Retrieves vectors by their keys
ApiResponse GetVectors(const json& request) {
    if (!request.contains("keys") || !request["keys"].is_array()) {
        return make_error(400, "ValidationException", "keys array is required");
    }
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName") && !request.contains("indexArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or indexArn is required");
    }

    std::string bucket_name;
    std::string index_name = request["indexName"].get<std::string>();

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        std::string arn = request["indexArn"].get<std::string>();
        size_t bucket_pos = arn.find("vector-bucket/");
        size_t index_pos = arn.find("/index/");
        if (bucket_pos != std::string::npos && index_pos != std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14, index_pos - bucket_pos - 14);
        }
    }

    // Read index configuration
    std::string config_path = utils::get_metadata_path(bucket_name) + "/" +
                             index_name + INDEX_CONFIG_SUFFIX;
    std::string config_str = utils::read_file(config_path);
    if (config_str.empty()) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found");
    }

    int dimension;
    try {
        json config = json::parse(config_str);
        dimension = config["dimension"].get<int>();
    } catch (...) {
        return make_error(500, "InternalServerException",
            "Failed to read index configuration");
    }

    // Get keys
    std::vector<std::string> keys;
    for (const auto& key : request["keys"]) {
        keys.push_back(key.get<std::string>());
    }

    // Connect to LanceDB
    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    LanceDBConnection* conn = LanceDBHelper::connect(db_path);
    if (!conn) {
        return make_error(500, "InternalServerException",
            "Failed to connect to index database");
    }

    LanceDBTable* table = LanceDBHelper::open_table(conn, "vectors");
    if (!table) {
        lancedb_connection_free(conn);
        return make_error(500, "InternalServerException",
            "Failed to open vectors table");
    }

    std::string error;
    auto results = LanceDBHelper::get_vectors_by_keys(table, keys, dimension, error);

    lancedb_table_free(table);
    lancedb_connection_free(conn);

    if (!error.empty() && results.empty()) {
        return make_error(500, "InternalServerException", error);
    }

    json vectors_response = json::array();
    for (const auto& result : results) {
        json vec;
        vec["key"] = result.key;
        vec["data"] = {{"float32", result.data}};

        if (!result.metadata.empty()) {
            try {
                vec["metadata"] = json::parse(result.metadata);
            } catch (...) {
                vec["metadata"] = result.metadata;
            }
        }

        vectors_response.push_back(vec);
    }

    return make_success({{"vectors", vectors_response}});
}

// DeleteVectors: Deletes vectors by their keys
ApiResponse DeleteVectors(const json& request) {
    if (!request.contains("keys") || !request["keys"].is_array()) {
        return make_error(400, "ValidationException", "keys array is required");
    }
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName") && !request.contains("indexArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or indexArn is required");
    }

    std::string bucket_name;
    std::string index_name = request["indexName"].get<std::string>();

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        std::string arn = request["indexArn"].get<std::string>();
        size_t bucket_pos = arn.find("vector-bucket/");
        size_t index_pos = arn.find("/index/");
        if (bucket_pos != std::string::npos && index_pos != std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14, index_pos - bucket_pos - 14);
        }
    }

    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    if (!utils::directory_exists(db_path)) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found");
    }

    LanceDBConnection* conn = LanceDBHelper::connect(db_path);
    if (!conn) {
        return make_error(500, "InternalServerException",
            "Failed to connect to index database");
    }

    LanceDBTable* table = LanceDBHelper::open_table(conn, "vectors");
    if (!table) {
        lancedb_connection_free(conn);
        return make_error(500, "InternalServerException",
            "Failed to open vectors table");
    }

    // Build predicate for deletion
    std::string predicate;
    for (size_t i = 0; i < request["keys"].size(); i++) {
        if (i > 0) predicate += " OR ";
        predicate += "key = \"" + request["keys"][i].get<std::string>() + "\"";
    }

    char* err_msg = nullptr;
    LanceDBError result = lancedb_table_delete(table, predicate.c_str(), &err_msg);

    lancedb_table_free(table);
    lancedb_connection_free(conn);

    if (result != LANCEDB_SUCCESS) {
        std::string error = err_msg ? err_msg : lancedb_error_to_message(result);
        if (err_msg) lancedb_free_string(err_msg);
        return make_error(500, "InternalServerException", error);
    }

    return make_success();
}

// ListVectors: Lists vectors in an index (with pagination)
ApiResponse ListVectors(const json& request) {
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName") && !request.contains("indexArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or indexArn is required");
    }

    std::string bucket_name;
    std::string index_name = request["indexName"].get<std::string>();

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        std::string arn = request["indexArn"].get<std::string>();
        size_t bucket_pos = arn.find("vector-bucket/");
        size_t index_pos = arn.find("/index/");
        if (bucket_pos != std::string::npos && index_pos != std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14, index_pos - bucket_pos - 14);
        }
    }

    int max_results = request.value("maxResults", 100);

    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    if (!utils::directory_exists(db_path)) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found");
    }

    LanceDBConnection* conn = LanceDBHelper::connect(db_path);
    if (!conn) {
        return make_error(500, "InternalServerException",
            "Failed to connect to index database");
    }

    LanceDBTable* table = LanceDBHelper::open_table(conn, "vectors");
    if (!table) {
        lancedb_connection_free(conn);
        return make_error(500, "InternalServerException",
            "Failed to open vectors table");
    }

    // Use query to list keys only
    LanceDBQuery* query = lancedb_query_new(table);
    if (!query) {
        lancedb_table_free(table);
        lancedb_connection_free(conn);
        return make_error(500, "InternalServerException",
            "Failed to create query");
    }

    const char* columns[] = {"key"};
    lancedb_query_select(query, columns, 1, nullptr);
    lancedb_query_limit(query, max_results, nullptr);

    LanceDBQueryResult* query_result = lancedb_query_execute(query);
    if (!query_result) {
        lancedb_table_free(table);
        lancedb_connection_free(conn);
        return make_error(500, "InternalServerException",
            "Failed to execute query");
    }

    struct ArrowArray** c_arrays = nullptr;
    struct ArrowSchema* c_schema = nullptr;
    size_t count = 0;

    json keys_array = json::array();

    if (lancedb_query_result_to_arrow(
            query_result,
            reinterpret_cast<FFI_ArrowArray***>(&c_arrays),
            reinterpret_cast<FFI_ArrowSchema**>(&c_schema),
            &count, nullptr) == LANCEDB_SUCCESS && count > 0) {

        auto schema_result = arrow::ImportSchema(c_schema);
        if (schema_result.ok()) {
            auto schema = *schema_result;
            auto batch_result = arrow::ImportRecordBatch(
                reinterpret_cast<struct ArrowArray*>(*c_arrays), schema);

            if (batch_result.ok()) {
                auto batch = *batch_result;
                int key_idx = schema->GetFieldIndex("key");

                if (key_idx >= 0) {
                    auto key_array = std::static_pointer_cast<arrow::StringArray>(
                        batch->column(key_idx));

                    for (int64_t row = 0; row < batch->num_rows(); row++) {
                        if (!key_array->IsNull(row)) {
                            keys_array.push_back(key_array->GetString(row));
                        }
                    }
                }
            }
        }
    }

    if (c_arrays) lancedb_free_arrow_arrays(reinterpret_cast<FFI_ArrowArray**>(c_arrays), count);
    if (c_schema) lancedb_free_arrow_schema(reinterpret_cast<FFI_ArrowSchema*>(c_schema));

    lancedb_table_free(table);
    lancedb_connection_free(conn);

    return make_success({{"vectorKeys", keys_array}});
}

// QueryVectors: Performs vector similarity search
ApiResponse QueryVectors(const json& request) {
    if (!request.contains("queryVector")) {
        return make_error(400, "ValidationException", "queryVector is required");
    }
    if (!request.contains("topK")) {
        return make_error(400, "ValidationException", "topK is required");
    }
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName") && !request.contains("indexArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or indexArn is required");
    }

    std::string bucket_name;
    std::string index_name = request["indexName"].get<std::string>();

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        std::string arn = request["indexArn"].get<std::string>();
        size_t bucket_pos = arn.find("vector-bucket/");
        size_t index_pos = arn.find("/index/");
        if (bucket_pos != std::string::npos && index_pos != std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14, index_pos - bucket_pos - 14);
        }
    }

    int top_k = request["topK"].get<int>();
    bool return_distance = request.value("returnDistance", false);
    bool return_metadata = request.value("returnMetadata", false);

    // Parse query vector
    std::vector<float> query_vector;
    if (request["queryVector"].contains("float32")) {
        for (const auto& val : request["queryVector"]["float32"]) {
            query_vector.push_back(val.get<float>());
        }
    } else if (request["queryVector"].is_array()) {
        for (const auto& val : request["queryVector"]) {
            query_vector.push_back(val.get<float>());
        }
    }

    // Read index configuration
    std::string config_path = utils::get_metadata_path(bucket_name) + "/" +
                             index_name + INDEX_CONFIG_SUFFIX;
    std::string config_str = utils::read_file(config_path);
    if (config_str.empty()) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found");
    }

    std::string distance_metric;
    int dimension;
    try {
        json config = json::parse(config_str);
        distance_metric = config.value("distanceMetric", "euclidean");
        dimension = config["dimension"].get<int>();
    } catch (...) {
        return make_error(500, "InternalServerException",
            "Failed to read index configuration");
    }

    if (static_cast<int>(query_vector.size()) != dimension) {
        return make_error(400, "ValidationException",
            "Query vector dimension mismatch. Expected " + std::to_string(dimension) +
            ", got " + std::to_string(query_vector.size()));
    }

    // Map distance metric to LanceDB type
    LanceDBDistanceType dist_type = LANCEDB_DISTANCE_L2;
    if (distance_metric == "cosine") {
        dist_type = LANCEDB_DISTANCE_COSINE;
    }

    // Parse filter if provided
    // Limitation: S3 Vector uses a specific filter syntax, we translate to SQL-like syntax
    std::string filter;
    if (request.contains("filter")) {
        // Simple translation - assumes filter is already in compatible format
        // Real implementation would need proper filter parsing
        filter = request["filter"].dump();
        // Remove quotes around the filter if it's a string
        if (filter.front() == '"' && filter.back() == '"') {
            filter = filter.substr(1, filter.length() - 2);
        }
    }

    // Connect to LanceDB
    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    LanceDBConnection* conn = LanceDBHelper::connect(db_path);
    if (!conn) {
        return make_error(500, "InternalServerException",
            "Failed to connect to index database");
    }

    LanceDBTable* table = LanceDBHelper::open_table(conn, "vectors");
    if (!table) {
        lancedb_connection_free(conn);
        return make_error(500, "InternalServerException",
            "Failed to open vectors table");
    }

    std::string error;
    auto results = LanceDBHelper::query_vectors(
        table, query_vector, top_k, filter,
        return_distance, return_metadata, dist_type, error);

    lancedb_table_free(table);
    lancedb_connection_free(conn);

    if (!error.empty() && results.empty()) {
        return make_error(500, "InternalServerException", error);
    }

    json vectors_response = json::array();
    for (const auto& result : results) {
        json vec;
        vec["key"] = result.key;

        if (return_distance) {
            vec["distance"] = result.distance;
        }

        if (return_metadata && !result.metadata.empty()) {
            try {
                vec["metadata"] = json::parse(result.metadata);
            } catch (...) {
                vec["metadata"] = result.metadata;
            }
        }

        vectors_response.push_back(vec);
    }

    return make_success({
        {"distanceMetric", distance_metric},
        {"vectors", vectors_response}
    });
}

// PutVectorBucketPolicy: Sets bucket policy
ApiResponse PutVectorBucketPolicy(const json& request) {
    if (!request.contains("vectorBucketName")) {
        return make_error(400, "ValidationException", "vectorBucketName is required");
    }
    if (!request.contains("policy")) {
        return make_error(400, "ValidationException", "policy is required");
    }

    std::string bucket_name = request["vectorBucketName"].get<std::string>();
    std::string bucket_path = utils::get_bucket_path(bucket_name);

    if (!utils::directory_exists(bucket_path)) {
        return make_error(404, "NotFoundException",
            "Vector bucket '" + bucket_name + "' not found");
    }

    // Note: Policy is stored but not enforced - this is a simulation limitation
    std::string policy_path = utils::get_metadata_path(bucket_name) + "/" + POLICY_FILE_NAME;
    std::string policy_str = request["policy"].dump(2);

    if (!utils::write_file(policy_path, policy_str)) {
        return make_error(500, "InternalServerException",
            "Failed to save bucket policy");
    }

    return make_success();
}

// GetVectorBucketPolicy: Gets bucket policy
ApiResponse GetVectorBucketPolicy(const json& request) {
    if (!request.contains("vectorBucketName")) {
        return make_error(400, "ValidationException", "vectorBucketName is required");
    }

    std::string bucket_name = request["vectorBucketName"].get<std::string>();
    std::string bucket_path = utils::get_bucket_path(bucket_name);

    if (!utils::directory_exists(bucket_path)) {
        return make_error(404, "NotFoundException",
            "Vector bucket '" + bucket_name + "' not found");
    }

    std::string policy_path = utils::get_metadata_path(bucket_name) + "/" + POLICY_FILE_NAME;
    std::string policy_str = utils::read_file(policy_path);

    if (policy_str.empty()) {
        return make_error(404, "NotFoundException",
            "No policy found for bucket '" + bucket_name + "'");
    }

    try {
        json policy = json::parse(policy_str);
        return make_success({{"policy", policy}});
    } catch (...) {
        return make_error(500, "InternalServerException",
            "Failed to parse bucket policy");
    }
}

// DeleteVectorBucketPolicy: Deletes bucket policy
ApiResponse DeleteVectorBucketPolicy(const json& request) {
    if (!request.contains("vectorBucketName")) {
        return make_error(400, "ValidationException", "vectorBucketName is required");
    }

    std::string bucket_name = request["vectorBucketName"].get<std::string>();
    std::string bucket_path = utils::get_bucket_path(bucket_name);

    if (!utils::directory_exists(bucket_path)) {
        return make_error(404, "NotFoundException",
            "Vector bucket '" + bucket_name + "' not found");
    }

    std::string policy_path = utils::get_metadata_path(bucket_name) + "/" + POLICY_FILE_NAME;

    if (utils::file_exists(policy_path)) {
        utils::delete_file(policy_path);
    }

    return make_success();
}

// TagResource: Adds tags to a resource
ApiResponse TagResource(const json& request) {
    if (!request.contains("resourceArn")) {
        return make_error(400, "ValidationException", "resourceArn is required");
    }
    if (!request.contains("tags")) {
        return make_error(400, "ValidationException", "tags is required");
    }

    std::string arn = request["resourceArn"].get<std::string>();

    // Parse ARN to get bucket name
    std::string bucket_name;
    size_t bucket_pos = arn.find("vector-bucket/");
    if (bucket_pos != std::string::npos) {
        size_t end_pos = arn.find("/", bucket_pos + 14);
        if (end_pos == std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14);
        } else {
            bucket_name = arn.substr(bucket_pos + 14, end_pos - bucket_pos - 14);
        }
    }

    if (bucket_name.empty()) {
        return make_error(400, "ValidationException", "Invalid resource ARN");
    }

    std::string bucket_path = utils::get_bucket_path(bucket_name);
    if (!utils::directory_exists(bucket_path)) {
        return make_error(404, "NotFoundException",
            "Resource not found");
    }

    std::string tags_path = utils::get_metadata_path(bucket_name) + "/" + TAGS_FILE_NAME;

    // Read existing tags
    json existing_tags;
    std::string existing_str = utils::read_file(tags_path);
    if (!existing_str.empty()) {
        try {
            existing_tags = json::parse(existing_str);
        } catch (...) {}
    }

    // Merge new tags
    for (auto& [key, value] : request["tags"].items()) {
        existing_tags[key] = value;
    }

    if (!utils::write_file(tags_path, existing_tags.dump(2))) {
        return make_error(500, "InternalServerException",
            "Failed to save tags");
    }

    return make_success();
}

// UntagResource: Removes tags from a resource
ApiResponse UntagResource(const json& request) {
    if (!request.contains("resourceArn")) {
        return make_error(400, "ValidationException", "resourceArn is required");
    }
    if (!request.contains("tagKeys") || !request["tagKeys"].is_array()) {
        return make_error(400, "ValidationException", "tagKeys array is required");
    }

    std::string arn = request["resourceArn"].get<std::string>();

    std::string bucket_name;
    size_t bucket_pos = arn.find("vector-bucket/");
    if (bucket_pos != std::string::npos) {
        size_t end_pos = arn.find("/", bucket_pos + 14);
        if (end_pos == std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14);
        } else {
            bucket_name = arn.substr(bucket_pos + 14, end_pos - bucket_pos - 14);
        }
    }

    if (bucket_name.empty()) {
        return make_error(400, "ValidationException", "Invalid resource ARN");
    }

    std::string bucket_path = utils::get_bucket_path(bucket_name);
    if (!utils::directory_exists(bucket_path)) {
        return make_error(404, "NotFoundException",
            "Resource not found");
    }

    std::string tags_path = utils::get_metadata_path(bucket_name) + "/" + TAGS_FILE_NAME;

    std::string existing_str = utils::read_file(tags_path);
    if (existing_str.empty()) {
        return make_success();
    }

    try {
        json tags = json::parse(existing_str);

        for (const auto& key : request["tagKeys"]) {
            std::string key_str = key.get<std::string>();
            tags.erase(key_str);
        }

        utils::write_file(tags_path, tags.dump(2));
    } catch (...) {
        return make_error(500, "InternalServerException",
            "Failed to update tags");
    }

    return make_success();
}

// ListTagsForResource: Lists tags for a resource
ApiResponse ListTagsForResource(const json& request) {
    if (!request.contains("resourceArn")) {
        return make_error(400, "ValidationException", "resourceArn is required");
    }

    std::string arn = request["resourceArn"].get<std::string>();

    std::string bucket_name;
    size_t bucket_pos = arn.find("vector-bucket/");
    if (bucket_pos != std::string::npos) {
        size_t end_pos = arn.find("/", bucket_pos + 14);
        if (end_pos == std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14);
        } else {
            bucket_name = arn.substr(bucket_pos + 14, end_pos - bucket_pos - 14);
        }
    }

    if (bucket_name.empty()) {
        return make_error(400, "ValidationException", "Invalid resource ARN");
    }

    std::string bucket_path = utils::get_bucket_path(bucket_name);
    if (!utils::directory_exists(bucket_path)) {
        return make_error(404, "NotFoundException",
            "Resource not found");
    }

    std::string tags_path = utils::get_metadata_path(bucket_name) + "/" + TAGS_FILE_NAME;
    std::string tags_str = utils::read_file(tags_path);

    json tags = json::object();
    if (!tags_str.empty()) {
        try {
            tags = json::parse(tags_str);
        } catch (...) {}
    }

    return make_success({{"tags", tags}});
}

// CreateVectorIndex: Creates a vector (ANN) index on the data column
ApiResponse CreateVectorIndex(const json& request) {
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName") && !request.contains("indexArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or indexArn is required");
    }

    std::string bucket_name;
    std::string index_name = request["indexName"].get<std::string>();

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        std::string arn = request["indexArn"].get<std::string>();
        size_t bucket_pos = arn.find("vector-bucket/");
        size_t index_pos = arn.find("/index/");
        if (bucket_pos != std::string::npos && index_pos != std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14, index_pos - bucket_pos - 14);
        }
    }

    // Get index type (default: IVF_PQ)
    std::string index_type_str = request.value("indexType", "IVF_PQ");
    LanceDBIndexType index_type;
    if (index_type_str == "IVF_FLAT") {
        index_type = LANCEDB_INDEX_IVF_FLAT;
    } else if (index_type_str == "IVF_PQ") {
        index_type = LANCEDB_INDEX_IVF_PQ;
    } else if (index_type_str == "IVF_HNSW_PQ") {
        index_type = LANCEDB_INDEX_IVF_HNSW_PQ;
    } else if (index_type_str == "IVF_HNSW_SQ") {
        index_type = LANCEDB_INDEX_IVF_HNSW_SQ;
    } else {
        return make_error(400, "ValidationException",
            "indexType must be one of: IVF_FLAT, IVF_PQ, IVF_HNSW_PQ, IVF_HNSW_SQ");
    }

    // Read index configuration to get distance metric
    std::string config_path = utils::get_metadata_path(bucket_name) + "/" +
                             index_name + INDEX_CONFIG_SUFFIX;
    std::string config_str = utils::read_file(config_path);
    if (config_str.empty()) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found");
    }

    std::string distance_metric;
    try {
        json config = json::parse(config_str);
        distance_metric = config.value("distanceMetric", "euclidean");
    } catch (...) {
        return make_error(500, "InternalServerException",
            "Failed to read index configuration");
    }

    // Connect to LanceDB
    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    LanceDBConnection* conn = LanceDBHelper::connect(db_path);
    if (!conn) {
        return make_error(500, "InternalServerException",
            "Failed to connect to index database");
    }

    LanceDBTable* table = LanceDBHelper::open_table(conn, "vectors");
    if (!table) {
        lancedb_connection_free(conn);
        return make_error(500, "InternalServerException",
            "Failed to open vectors table");
    }

    // Configure vector index
    LanceDBVectorIndexConfig vec_config;
    vec_config.num_partitions = request.value("numPartitions", -1);
    vec_config.num_sub_vectors = request.value("numSubVectors", -1);
    vec_config.max_iterations = request.value("maxIterations", -1);
    vec_config.sample_rate = request.value("sampleRate", 0.0f);
    vec_config.distance_type = (distance_metric == "cosine") ?
                               LANCEDB_DISTANCE_COSINE : LANCEDB_DISTANCE_L2;
    vec_config.accelerator = nullptr;
    if (request.contains("accelerator")) {
        static std::string accel = request["accelerator"].get<std::string>();
        vec_config.accelerator = accel.c_str();
    }
    vec_config.replace = request.value("replace", 0);

    // Create the vector index on "data" column
    const char* columns[] = {"data"};
    char* err_msg = nullptr;
    LanceDBError result = lancedb_table_create_vector_index(
        table, columns, 1, index_type, &vec_config, &err_msg);

    lancedb_table_free(table);
    lancedb_connection_free(conn);

    if (result != LANCEDB_SUCCESS) {
        std::string error = err_msg ? err_msg : lancedb_error_to_message(result);
        if (err_msg) lancedb_free_string(err_msg);
        return make_error(500, "InternalServerException", error);
    }

    return make_success({
        {"message", "Vector index created successfully"},
        {"indexType", index_type_str},
        {"column", "data"}
    });
}

// CreateScalarIndex: Creates a scalar index on a column (for metadata filtering)
ApiResponse CreateScalarIndex(const json& request) {
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName") && !request.contains("indexArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or indexArn is required");
    }
    if (!request.contains("column")) {
        return make_error(400, "ValidationException", "column is required");
    }

    std::string bucket_name;
    std::string index_name = request["indexName"].get<std::string>();
    std::string column_name = request["column"].get<std::string>();

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        std::string arn = request["indexArn"].get<std::string>();
        size_t bucket_pos = arn.find("vector-bucket/");
        size_t index_pos = arn.find("/index/");
        if (bucket_pos != std::string::npos && index_pos != std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14, index_pos - bucket_pos - 14);
        }
    }

    // Get scalar index type (default: BTREE)
    std::string scalar_type_str = request.value("scalarIndexType", "BTREE");
    LanceDBIndexType scalar_type;
    if (scalar_type_str == "BTREE") {
        scalar_type = LANCEDB_INDEX_BTREE;
    } else if (scalar_type_str == "BITMAP") {
        scalar_type = LANCEDB_INDEX_BITMAP;
    } else if (scalar_type_str == "LABELLIST") {
        scalar_type = LANCEDB_INDEX_LABELLIST;
    } else {
        return make_error(400, "ValidationException",
            "scalarIndexType must be one of: BTREE, BITMAP, LABELLIST");
    }

    // Check index exists
    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    if (!utils::directory_exists(db_path)) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found");
    }

    // Connect to LanceDB
    LanceDBConnection* conn = LanceDBHelper::connect(db_path);
    if (!conn) {
        return make_error(500, "InternalServerException",
            "Failed to connect to index database");
    }

    LanceDBTable* table = LanceDBHelper::open_table(conn, "vectors");
    if (!table) {
        lancedb_connection_free(conn);
        return make_error(500, "InternalServerException",
            "Failed to open vectors table");
    }

    // Configure scalar index
    LanceDBScalarIndexConfig scalar_config;
    scalar_config.replace = request.value("replace", 0);
    scalar_config.force_update_statistics = request.value("forceUpdateStatistics", 0);

    // Create the scalar index
    const char* columns[] = {column_name.c_str()};
    char* err_msg = nullptr;
    LanceDBError result = lancedb_table_create_scalar_index(
        table, columns, 1, scalar_type, &scalar_config, &err_msg);

    lancedb_table_free(table);
    lancedb_connection_free(conn);

    if (result != LANCEDB_SUCCESS) {
        std::string error = err_msg ? err_msg : lancedb_error_to_message(result);
        if (err_msg) lancedb_free_string(err_msg);
        return make_error(500, "InternalServerException", error);
    }

    return make_success({
        {"message", "Scalar index created successfully"},
        {"scalarIndexType", scalar_type_str},
        {"column", column_name}
    });
}

// OptimizeIndex: Compacts files and/or rebuilds indices
ApiResponse OptimizeIndex(const json& request) {
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName") && !request.contains("indexArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or indexArn is required");
    }

    std::string bucket_name;
    std::string index_name = request["indexName"].get<std::string>();

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        std::string arn = request["indexArn"].get<std::string>();
        size_t bucket_pos = arn.find("vector-bucket/");
        size_t index_pos = arn.find("/index/");
        if (bucket_pos != std::string::npos && index_pos != std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14, index_pos - bucket_pos - 14);
        }
    }

    // Get optimization type (default: ALL)
    std::string optimize_type_str = request.value("optimizeType", "ALL");
    LanceDBOptimizeType optimize_type;
    if (optimize_type_str == "ALL") {
        optimize_type = LANCEDB_OPTIMIZE_ALL;
    } else if (optimize_type_str == "COMPACT") {
        optimize_type = LANCEDB_OPTIMIZE_COMPACT;
    } else if (optimize_type_str == "PRUNE") {
        optimize_type = LANCEDB_OPTIMIZE_PRUNE;
    } else if (optimize_type_str == "INDEX") {
        optimize_type = LANCEDB_OPTIMIZE_INDEX;
    } else {
        return make_error(400, "ValidationException",
            "optimizeType must be one of: ALL, COMPACT, PRUNE, INDEX");
    }

    // Check index exists
    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    if (!utils::directory_exists(db_path)) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found");
    }

    // Connect to LanceDB
    LanceDBConnection* conn = LanceDBHelper::connect(db_path);
    if (!conn) {
        return make_error(500, "InternalServerException",
            "Failed to connect to index database");
    }

    LanceDBTable* table = LanceDBHelper::open_table(conn, "vectors");
    if (!table) {
        lancedb_connection_free(conn);
        return make_error(500, "InternalServerException",
            "Failed to open vectors table");
    }

    // Run optimization
    char* err_msg = nullptr;
    LanceDBError result = lancedb_table_optimize(table, optimize_type, &err_msg);

    // Get stats after optimization
    int64_t row_count = lancedb_table_count_rows(table);

    lancedb_table_free(table);
    lancedb_connection_free(conn);

    if (result != LANCEDB_SUCCESS) {
        std::string error = err_msg ? err_msg : lancedb_error_to_message(result);
        if (err_msg) lancedb_free_string(err_msg);
        return make_error(500, "InternalServerException", error);
    }

    return make_success({
        {"message", "Optimization completed successfully"},
        {"optimizeType", optimize_type_str},
        {"vectorCount", row_count}
    });
}

// ListIndices: Lists all LanceDB indices on the table (vector and scalar)
ApiResponse ListLanceIndices(const json& request) {
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName") && !request.contains("indexArn")) {
        return make_error(400, "ValidationException",
            "Either vectorBucketName or indexArn is required");
    }

    std::string bucket_name;
    std::string index_name = request["indexName"].get<std::string>();

    if (request.contains("vectorBucketName")) {
        bucket_name = request["vectorBucketName"].get<std::string>();
    } else {
        std::string arn = request["indexArn"].get<std::string>();
        size_t bucket_pos = arn.find("vector-bucket/");
        size_t index_pos = arn.find("/index/");
        if (bucket_pos != std::string::npos && index_pos != std::string::npos) {
            bucket_name = arn.substr(bucket_pos + 14, index_pos - bucket_pos - 14);
        }
    }

    // Check index exists
    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    if (!utils::directory_exists(db_path)) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found");
    }

    // Connect to LanceDB
    LanceDBConnection* conn = LanceDBHelper::connect(db_path);
    if (!conn) {
        return make_error(500, "InternalServerException",
            "Failed to connect to index database");
    }

    LanceDBTable* table = LanceDBHelper::open_table(conn, "vectors");
    if (!table) {
        lancedb_connection_free(conn);
        return make_error(500, "InternalServerException",
            "Failed to open vectors table");
    }

    // List indices
    char** indices = nullptr;
    size_t count = 0;
    char* err_msg = nullptr;
    LanceDBError result = lancedb_table_list_indices(table, &indices, &count, &err_msg);

    if (result != LANCEDB_SUCCESS) {
        lancedb_table_free(table);
        lancedb_connection_free(conn);
        std::string error = err_msg ? err_msg : lancedb_error_to_message(result);
        if (err_msg) lancedb_free_string(err_msg);
        return make_error(500, "InternalServerException", error);
    }

    // Parse index information
    json indices_array = json::array();
    for (size_t i = 0; i < count; i++) {
        if (indices[i]) {
            // Each index info is returned as a JSON string
            try {
                json index_info = json::parse(indices[i]);
                indices_array.push_back(index_info);
            } catch (...) {
                // If not JSON, just add as string
                indices_array.push_back(indices[i]);
            }
        }
    }

    // Free indices
    if (indices) {
        lancedb_free_index_list(indices, count);
    }

    lancedb_table_free(table);
    lancedb_connection_free(conn);

    return make_success({
        {"indexName", index_name},
        {"vectorBucketName", bucket_name},
        {"lanceIndices", indices_array},
        {"count", count}
    });
}

// ============================================================================
// Command Line Interface
// ============================================================================

void print_help() {
    std::cout << R"(
S3 Vector API Simulation using LanceDB

USAGE:
    s3vector_simulation <command> [options]

COMMANDS:
    CreateVectorBucket      Create a new vector bucket
    DeleteVectorBucket      Delete a vector bucket
    GetVectorBucket         Get information about a vector bucket
    ListVectorBuckets       List all vector buckets

    CreateIndex             Create a vector index in a bucket
    DeleteIndex             Delete a vector index
    GetIndex                Get information about a vector index
    ListIndexes             List all indices in a bucket

    PutVectors              Add vectors to an index
    GetVectors              Get vectors by their keys
    DeleteVectors           Delete vectors by their keys
    ListVectors             List vector keys in an index
    QueryVectors            Perform vector similarity search

    PutVectorBucketPolicy   Set bucket policy
    GetVectorBucketPolicy   Get bucket policy
    DeleteVectorBucketPolicy Delete bucket policy

    TagResource             Add tags to a resource
    UntagResource           Remove tags from a resource
    ListTagsForResource     List tags for a resource

    CreateVectorIndex       Create ANN index on vector data column
    CreateScalarIndex       Create scalar index on a column (BTREE/BITMAP)
    OptimizeIndex           Compact files and/or rebuild indices
    ListLanceIndices        List all LanceDB indices on a table

    --help, -h              Show this help message

INPUT FORMAT:
    Each command accepts a JSON object as input via stdin or as a command-line argument.

EXAMPLES:

    # Create a vector bucket
    echo '{"vectorBucketName": "my-bucket"}' | ./s3vector_simulation CreateVectorBucket

    # Create an index
    ./s3vector_simulation CreateIndex '{"vectorBucketName": "my-bucket", "indexName": "my-index", "dimension": 128, "distanceMetric": "cosine"}'

    # Add vectors
    ./s3vector_simulation PutVectors '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index",
        "vectors": [
            {"key": "v1", "data": {"float32": [0.1, 0.2, ...]}, "metadata": {"label": "test"}}
        ]
    }'

    # Query vectors
    ./s3vector_simulation QueryVectors '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index",
        "queryVector": {"float32": [0.1, 0.2, ...]},
        "topK": 5,
        "returnDistance": true
    }'

    # Create ANN vector index (IVF_PQ)
    ./s3vector_simulation CreateVectorIndex '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index",
        "indexType": "IVF_PQ",
        "numPartitions": 16,
        "numSubVectors": 64
    }'

    # Create scalar index on key column
    ./s3vector_simulation CreateScalarIndex '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index",
        "column": "key",
        "scalarIndexType": "BTREE"
    }'

    # Optimize (compact files + rebuild indices)
    ./s3vector_simulation OptimizeIndex '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index",
        "optimizeType": "ALL"
    }'

    # List all LanceDB indices
    ./s3vector_simulation ListLanceIndices '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index"
    }'

STORAGE:
    All data is stored under /tmp/s3vectors/
    - Vector buckets: /tmp/s3vectors/<bucket-name>/
    - Indices: /tmp/s3vectors/<bucket-name>/<index-name>_lancedb/
    - Metadata: /tmp/s3vectors/<bucket-name>/.s3v_metadata/

LIMITATIONS:
    - No actual encryption (encryption config is stored but not applied)
    - No IAM/policy enforcement (policies are stored but not enforced)
    - Tags and policies are stored as JSON files
    - No true ARN generation (uses simulated path-based identifiers)
    - Concurrent write behavior depends on LanceDB's internal handling

CONCURRENT WRITES:
    This simulation can be used to test LanceDB's concurrent write behavior.
    Run multiple instances with PutVectors to the same index to observe:
    - How LanceDB handles concurrent writes
    - Whether data is properly serialized
    - Any conflict resolution mechanisms

)";
}

int main(int argc, char* argv[]) {
    // Ensure root directory exists
    if (!utils::directory_exists(S3_VECTORS_ROOT)) {
        utils::create_directories(S3_VECTORS_ROOT);
    }

    if (argc < 2) {
        print_help();
        return 1;
    }

    std::string command = argv[1];

    if (command == "--help" || command == "-h") {
        print_help();
        return 0;
    }

    // Get JSON input
    std::string json_input;
    if (argc > 2) {
        json_input = argv[2];
    } else {
        // Read from stdin
        std::getline(std::cin, json_input);
        if (json_input.empty()) {
            json_input = "{}";
        }
    }

    json request;
    try {
        request = json::parse(json_input);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON input: " << e.what() << std::endl;
        return 1;
    }

    // Route to appropriate handler
    ApiResponse response;

    if (command == "CreateVectorBucket") {
        response = CreateVectorBucket(request);
    } else if (command == "DeleteVectorBucket") {
        response = DeleteVectorBucket(request);
    } else if (command == "GetVectorBucket") {
        response = GetVectorBucket(request);
    } else if (command == "ListVectorBuckets") {
        response = ListVectorBuckets(request);
    } else if (command == "CreateIndex") {
        response = CreateIndex(request);
    } else if (command == "DeleteIndex") {
        response = DeleteIndex(request);
    } else if (command == "GetIndex") {
        response = GetIndex(request);
    } else if (command == "ListIndexes") {
        response = ListIndexes(request);
    } else if (command == "PutVectors") {
        response = PutVectors(request);
    } else if (command == "GetVectors") {
        response = GetVectors(request);
    } else if (command == "DeleteVectors") {
        response = DeleteVectors(request);
    } else if (command == "ListVectors") {
        response = ListVectors(request);
    } else if (command == "QueryVectors") {
        response = QueryVectors(request);
    } else if (command == "PutVectorBucketPolicy") {
        response = PutVectorBucketPolicy(request);
    } else if (command == "GetVectorBucketPolicy") {
        response = GetVectorBucketPolicy(request);
    } else if (command == "DeleteVectorBucketPolicy") {
        response = DeleteVectorBucketPolicy(request);
    } else if (command == "TagResource") {
        response = TagResource(request);
    } else if (command == "UntagResource") {
        response = UntagResource(request);
    } else if (command == "ListTagsForResource") {
        response = ListTagsForResource(request);
    } else if (command == "CreateVectorIndex") {
        response = CreateVectorIndex(request);
    } else if (command == "CreateScalarIndex") {
        response = CreateScalarIndex(request);
    } else if (command == "OptimizeIndex") {
        response = OptimizeIndex(request);
    } else if (command == "ListLanceIndices") {
        response = ListLanceIndices(request);
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        print_help();
        return 1;
    }

    // Output response
    std::cout << response.to_string() << std::endl;

    return response.is_success() ? 0 : 1;
}
