/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright The LanceDB Authors
 *
 * S3 Vector Concurrent Service with Background Indexing 
 * Note : This is a simplified example focusing on core logic. its limited single node only.
 *
 * This application provides a concurrent vector service that supports:
 * - Concurrent put-vector, remove-vector, and search-vector operations
 * - Background index building/rebuilding without blocking operations
 * - Smart index rebuild triggers checked on each put/remove operation
 * - LanceDB's ability to use old index + brute force on unindexed data
 *
 * Architecture:
 * - Operations (put/remove) update state counters and check rebuild thresholds
 * - If rebuild is needed, a background thread is spawned (one per table)
 * - Search operations continue using existing index + brute force on new data
 * - No polling - rebuild decisions made at operation time
 *
 * Index Rebuild Triggers (checked on each put/remove):
 * - Insertion threshold: when unindexed vector count exceeds threshold
 * - Deletion ratio: when deleted vectors exceed percentage threshold
 * - Manual trigger via API
 *
 * Logging:
 * - All operations logged with timestamp, operation type, table name, vector ID
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
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <iomanip>
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
const std::string INDEX_CONFIG_SUFFIX = "_index_config.json";
const std::string INDEX_STATE_SUFFIX = "_index_state.json";
const std::string LOG_FILE_NAME = "operations.log"; //conatain timestamped, operation, on what table and other details

// Default index rebuild thresholds
const size_t DEFAULT_UNINDEXED_THRESHOLD = 1000;    // Rebuild when this many new vectors, this may be overridden on runtime(user request) 
const double DEFAULT_DELETION_RATIO = 0.20;          // Rebuild when 20% deleted

// ============================================================================
// Logging
// ============================================================================

class Logger {
public:
    enum Level { DEBUG, INFO, WARN, ERROR };

private:
    std::mutex mutex_;
    std::ofstream log_file_;
    Level min_level_ = INFO;
    bool console_output_ = true;

    std::string level_to_string(Level level) {
        switch (level) {
            case DEBUG: return "DEBUG";
            case INFO:  return "INFO";
            case WARN:  return "WARN";
            case ERROR: return "ERROR";
            default:    return "UNKNOWN";
        }
    }

    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << ms.count() << "Z";
        return ss.str();
    }

public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void init(const std::string& log_path, Level level = INFO, bool console = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        log_file_.open(log_path, std::ios::app);
        min_level_ = level;
        console_output_ = console;
    }

    void log(Level level, const std::string& operation, const std::string& table_name,
             const std::string& vector_id, const std::string& message) {
        if (level < min_level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        std::stringstream ss;
        ss << get_timestamp() << " [" << level_to_string(level) << "] "
           << "op=" << operation;
        if (!table_name.empty()) ss << " table=" << table_name;
        if (!vector_id.empty()) ss << " vector_id=" << vector_id;
        ss << " " << message;

        std::string log_line = ss.str();

        if (console_output_) {
            std::cout << log_line << std::endl;
        }

        if (log_file_.is_open()) {
            log_file_ << log_line << std::endl;
            log_file_.flush();
        }
    }

    void log(Level level, const std::string& operation, const std::string& table_name,
             const std::string& message) {
        log(level, operation, table_name, "", message);
    }

    void log(Level level, const std::string& message) {
        log(level, "", "", "", message);
    }
};

#define LOG_DEBUG(op, table, msg) Logger::instance().log(Logger::DEBUG, op, table, msg)
#define LOG_INFO(op, table, msg) Logger::instance().log(Logger::INFO, op, table, msg)
#define LOG_WARN(op, table, msg) Logger::instance().log(Logger::WARN, op, table, msg)
#define LOG_ERROR(op, table, msg) Logger::instance().log(Logger::ERROR, op, table, msg)
#define LOG_OP(op, table, vid, msg) Logger::instance().log(Logger::INFO, op, table, vid, msg)

// ============================================================================
// Index Configuration
// ============================================================================

struct IndexConfig {
    std::string index_type = "IVF_PQ";  // IVF_FLAT, IVF_PQ, IVF_HNSW_PQ, IVF_HNSW_SQ
    std::string distance_metric = "euclidean";  // euclidean, cosine
    int num_partitions = -1;      // -1 = auto
    int num_sub_vectors = -1;     // -1 = auto
    int max_iterations = -1;      // -1 = default
    float sample_rate = 0.0f;     // 0.0 = default
    std::string accelerator = ""; // "cuda", "mps", or empty for CPU

    // Query parameters (used at search time)
    int nprobes = -1;             // -1 = default
    int refine_factor = -1;       // -1 = default
    int ef = -1;                  // -1 = default (for HNSW)

    json to_json() const {
        return {
            {"indexType", index_type},
            {"distanceMetric", distance_metric},
            {"numPartitions", num_partitions},
            {"numSubVectors", num_sub_vectors},
            {"maxIterations", max_iterations},
            {"sampleRate", sample_rate},
            {"accelerator", accelerator},
            {"nprobes", nprobes},
            {"refineFactor", refine_factor},
            {"ef", ef}
        };
    }

    static IndexConfig from_json(const json& j) {
        IndexConfig config;
        if (j.contains("indexType")) config.index_type = j["indexType"];
        if (j.contains("distanceMetric")) config.distance_metric = j["distanceMetric"];
        if (j.contains("numPartitions")) config.num_partitions = j["numPartitions"];
        if (j.contains("numSubVectors")) config.num_sub_vectors = j["numSubVectors"];
        if (j.contains("maxIterations")) config.max_iterations = j["maxIterations"];
        if (j.contains("sampleRate")) config.sample_rate = j["sampleRate"];
        if (j.contains("accelerator")) config.accelerator = j["accelerator"];
        if (j.contains("nprobes")) config.nprobes = j["nprobes"];
        if (j.contains("refineFactor")) config.refine_factor = j["refineFactor"];
        if (j.contains("ef")) config.ef = j["ef"];
        return config;
    }

    LanceDBIndexType to_lancedb_type() const {
        if (index_type == "IVF_FLAT") return LANCEDB_INDEX_IVF_FLAT;
        if (index_type == "IVF_PQ") return LANCEDB_INDEX_IVF_PQ;
        if (index_type == "IVF_HNSW_PQ") return LANCEDB_INDEX_IVF_HNSW_PQ;
        if (index_type == "IVF_HNSW_SQ") return LANCEDB_INDEX_IVF_HNSW_SQ;
        return LANCEDB_INDEX_IVF_PQ;  // default
    }

    LanceDBDistanceType to_lancedb_distance() const {
        if (distance_metric == "cosine") return LANCEDB_DISTANCE_COSINE;
        if (distance_metric == "dot") return LANCEDB_DISTANCE_DOT;
        return LANCEDB_DISTANCE_L2;  // default (euclidean)
    }
};

// ============================================================================
// Table Index State - Tracks changes since last index build
// ============================================================================

class TableIndexState {
public:
    std::string bucket_name;
    std::string index_name;

    // Index configuration
    IndexConfig config;
    int dimension = 0;

    // State tracking (atomic for thread safety)
    std::atomic<bool> index_build_in_progress{false};
    std::atomic<size_t> insertions_since_build{0};
    std::atomic<size_t> deletions_since_build{0};
    std::atomic<size_t> rows_at_last_build{0};
    std::atomic<unsigned long long> version_at_last_build{0};

    // Thresholds for auto-rebuild
    size_t unindexed_threshold = DEFAULT_UNINDEXED_THRESHOLD;
    double deletion_ratio_threshold = DEFAULT_DELETION_RATIO;

    // Mutex for config updates and persistence
    mutable std::mutex mutex;

    bool needs_rebuild() const {
        // Don't trigger if already building
        if (index_build_in_progress.load()) return false;

        size_t insertions = insertions_since_build.load();
        size_t deletions = deletions_since_build.load();
        size_t rows = rows_at_last_build.load();

        // Check if we have enough new insertions
        if (insertions >= unindexed_threshold) {
            return true;
        }

        // Check if deletion ratio is too high
        if (rows > 0) {
            double deletion_ratio = static_cast<double>(deletions) / static_cast<double>(rows);
            if (deletion_ratio >= deletion_ratio_threshold) {
                return true;
            }
        }

        // Check if no index exists but we have enough data
        if (version_at_last_build.load() == 0 && insertions >= unindexed_threshold) {
            return true;
        }

        return false;
    }

    void record_insertions(size_t count) {
        insertions_since_build.fetch_add(count);
    }

    void record_deletions(size_t count) {
        deletions_since_build.fetch_add(count);
    }

    void on_index_build_complete(size_t row_count, unsigned long long version) {
        insertions_since_build.store(0);
        deletions_since_build.store(0);
        rows_at_last_build.store(row_count);
        version_at_last_build.store(version);
        index_build_in_progress.store(false);
    }

    json to_json() const {
        std::lock_guard<std::mutex> lock(mutex);
        return {
            {"bucketName", bucket_name},
            {"indexName", index_name},
            {"dimension", dimension},
            {"config", config.to_json()},
            {"insertionsSinceBuild", insertions_since_build.load()},
            {"deletionsSinceBuild", deletions_since_build.load()},
            {"rowsAtLastBuild", rows_at_last_build.load()},
            {"versionAtLastBuild", version_at_last_build.load()},
            {"indexBuildInProgress", index_build_in_progress.load()},
            {"unindexedThreshold", unindexed_threshold},
            {"deletionRatioThreshold", deletion_ratio_threshold}
        };
    }

    void load_from_json(const json& j) {
        std::lock_guard<std::mutex> lock(mutex);
        if (j.contains("bucketName")) bucket_name = j["bucketName"];
        if (j.contains("indexName")) index_name = j["indexName"];
        if (j.contains("dimension")) dimension = j["dimension"];
        if (j.contains("config")) config = IndexConfig::from_json(j["config"]);
        if (j.contains("insertionsSinceBuild")) insertions_since_build = j["insertionsSinceBuild"].get<size_t>();
        if (j.contains("deletionsSinceBuild")) deletions_since_build = j["deletionsSinceBuild"].get<size_t>();
        if (j.contains("rowsAtLastBuild")) rows_at_last_build = j["rowsAtLastBuild"].get<size_t>();
        if (j.contains("versionAtLastBuild")) version_at_last_build = j["versionAtLastBuild"].get<unsigned long long>();
        if (j.contains("unindexedThreshold")) unindexed_threshold = j["unindexedThreshold"];
        if (j.contains("deletionRatioThreshold")) deletion_ratio_threshold = j["deletionRatioThreshold"];
    }
};

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

std::string get_bucket_path(const std::string& bucket_name) {
    return S3_VECTORS_ROOT + "/" + bucket_name;
}

std::string get_metadata_path(const std::string& bucket_name) {
    return get_bucket_path(bucket_name) + "/" + METADATA_DIR_NAME;
}

std::string get_index_db_path(const std::string& bucket_name, const std::string& index_name) {
    return get_bucket_path(bucket_name) + "/" + index_name + "_lancedb";
}

std::string get_index_state_path(const std::string& bucket_name, const std::string& index_name) {
    return get_metadata_path(bucket_name) + "/" + index_name + INDEX_STATE_SUFFIX;
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
	//TODO (below)the table field names and thier type should dynmaic based on the user input, for now we are hardcoding. 
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

            std::string fn = (i < function_names.size()) ? function_names[i] : "";
            if (!function_name_builder.Append(fn).ok()) {
                error = "Failed to append function_name";
                return false;
            }

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

	//TODO the arrow::RecordBatch::Make should be dynamic based on the user input, for now we are hardcoding the field names and their type.
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
        const IndexConfig& config,
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

        // Apply index-specific query parameters
        if (config.nprobes > 0) {
            lancedb_vector_query_nprobes(query, config.nprobes, nullptr);
        }
        if (config.refine_factor > 0) {
            lancedb_vector_query_refine_factor(query, config.refine_factor, nullptr);
        }
        if (config.ef > 0 && (config.index_type == "IVF_HNSW_PQ" || config.index_type == "IVF_HNSW_SQ")) {
            lancedb_vector_query_ef(query, config.ef, nullptr);
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
};

// ============================================================================
// Table State Manager - Manages per-table index state
// ============================================================================

class TableStateManager {
private:
    std::map<std::string, std::shared_ptr<TableIndexState>> states_;
    mutable std::mutex mutex_;

    std::string make_key(const std::string& bucket, const std::string& index) const {
        return bucket + "/" + index;
    }

public:
    static TableStateManager& instance() {
        static TableStateManager manager;
        return manager;
    }

    std::shared_ptr<TableIndexState> get_or_create(const std::string& bucket,
                                                     const std::string& index) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string key = make_key(bucket, index);

        auto it = states_.find(key);
        if (it != states_.end()) {
            return it->second;
        }

        // Try to load from disk
        std::string state_path = utils::get_index_state_path(bucket, index);
        std::string state_str = utils::read_file(state_path);

        auto state = std::make_shared<TableIndexState>();
        state->bucket_name = bucket;
        state->index_name = index;

        if (!state_str.empty()) {
            try {
                json j = json::parse(state_str);
                state->load_from_json(j);
            } catch (...) {
                // Ignore parse errors, use default state
            }
        }

        states_[key] = state;
        return state;
    }

    void save_state(const std::shared_ptr<TableIndexState>& state) {
        std::string state_path = utils::get_index_state_path(state->bucket_name, state->index_name);
        utils::write_file(state_path, state->to_json().dump(2));
    }

    void remove(const std::string& bucket, const std::string& index) {
        std::lock_guard<std::mutex> lock(mutex_);
        states_.erase(make_key(bucket, index));
    }
};

// ============================================================================
// Background Index Builder - Runs index builds in background threads
// ============================================================================

class BackgroundIndexBuilder {
public:
    static BackgroundIndexBuilder& instance() {
        static BackgroundIndexBuilder builder;
        return builder;
    }

    // Trigger rebuild for a table - spawns a detached background thread
    // Returns true if rebuild was triggered, false if already in progress
    bool trigger_rebuild_if_needed(std::shared_ptr<TableIndexState> state) {
        // Check if rebuild is needed
        if (!state->needs_rebuild()) {
            return false;
        }

        // Try to acquire the build lock (atomic compare-and-swap)
        bool expected = false;
        if (!state->index_build_in_progress.compare_exchange_strong(expected, true)) {
            // Another thread is already building
            return false;
        }

        // Spawn detached background thread for the actual build
        std::thread([this, state]() {
            this->run_build(state);
        }).detach();

        return true;
    }

    // Force trigger rebuild (for manual trigger)
    bool force_trigger_rebuild(std::shared_ptr<TableIndexState> state) {
        bool expected = false;
        if (!state->index_build_in_progress.compare_exchange_strong(expected, true)) {
            return false;  // Already in progress
        }

        std::thread([this, state]() {
            this->run_build(state);
        }).detach();

        return true;
    }

private:
    void run_build(std::shared_ptr<TableIndexState> state) {
        LOG_INFO("BUILD_INDEX", state->index_name,
                 "Starting background index build (type=" + state->config.index_type +
                 ", insertions=" + std::to_string(state->insertions_since_build.load()) +
                 ", deletions=" + std::to_string(state->deletions_since_build.load()) + ")");

        std::string db_path = utils::get_index_db_path(state->bucket_name, state->index_name);
        LanceDBConnection* conn = LanceDBHelper::connect(db_path);
        if (!conn) {
            LOG_ERROR("BUILD_INDEX", state->index_name, "Failed to connect to database");
            state->index_build_in_progress.store(false);
            return;
        }

        LanceDBTable* table = LanceDBHelper::open_table(conn, "vectors");
        if (!table) {
            lancedb_connection_free(conn);
            LOG_ERROR("BUILD_INDEX", state->index_name, "Failed to open table");
            state->index_build_in_progress.store(false);
            return;
        }

        // Get current row count before building
        unsigned long long row_count = lancedb_table_count_rows(table);

        // Configure vector index
        LanceDBVectorIndexConfig vec_config;
        vec_config.num_partitions = state->config.num_partitions;
        vec_config.num_sub_vectors = state->config.num_sub_vectors;
        vec_config.max_iterations = state->config.max_iterations;
        vec_config.sample_rate = state->config.sample_rate;
        vec_config.distance_type = state->config.to_lancedb_distance();
        vec_config.accelerator = state->config.accelerator.empty() ?
                                  nullptr : state->config.accelerator.c_str();
        vec_config.replace = 1;  // Always replace existing index

        // Create the vector index
        const char* columns[] = {"data"};
        char* err_msg = nullptr;
        LanceDBError result = lancedb_table_create_vector_index(
            table, columns, 1, state->config.to_lancedb_type(), &vec_config, &err_msg);

        if (result != LANCEDB_SUCCESS) {
            std::string error = err_msg ? err_msg : lancedb_error_to_message(result);
            if (err_msg) lancedb_free_string(err_msg);
            LOG_ERROR("BUILD_INDEX", state->index_name, "Index build failed: " + error);
            lancedb_table_free(table);
            lancedb_connection_free(conn);
            state->index_build_in_progress.store(false);
            return;
        }

        // Re-open table to get updated version
        lancedb_table_free(table);
        table = LanceDBHelper::open_table(conn, "vectors");
        unsigned long long version = table ? lancedb_table_version(table) : 0;
        if (table) lancedb_table_free(table);
        lancedb_connection_free(conn);

        // Update state - this resets counters
        state->on_index_build_complete(row_count, version);
        TableStateManager::instance().save_state(state);

        LOG_INFO("BUILD_INDEX", state->index_name,
                 "Index build complete (rows=" + std::to_string(row_count) +
                 ", version=" + std::to_string(version) + ")");
    }
};

// ============================================================================
// S3 Vector API Implementations
// ============================================================================

// CreateVectorBucket
ApiResponse CreateVectorBucket(const json& request) {
    if (!request.contains("vectorBucketName")) {
        return make_error(400, "ValidationException", "vectorBucketName is required");
    }

    std::string bucket_name = request["vectorBucketName"].get<std::string>();
    std::string bucket_path = utils::get_bucket_path(bucket_name);

    if (utils::directory_exists(bucket_path)) {
        return make_error(409, "ConflictException",
            "Vector bucket '" + bucket_name + "' already exists");
    }

    if (!utils::create_directories(bucket_path)) {
        return make_error(500, "InternalServerException",
            "Failed to create vector bucket directory");
    }

    std::string metadata_path = utils::get_metadata_path(bucket_name);
    if (!utils::create_directory(metadata_path)) {
        return make_error(500, "InternalServerException",
            "Failed to create metadata directory");
    }

    LOG_INFO("CREATE_BUCKET", bucket_name, "Bucket created");

    return make_success({{"vectorBucketName", bucket_name}});
}

// CreateIndex with index configuration
ApiResponse CreateIndex(const json& request) {
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("dimension")) {
        return make_error(400, "ValidationException", "dimension is required");
    }
    if (!request.contains("vectorBucketName")) {
        return make_error(400, "ValidationException", "vectorBucketName is required");
    }

    std::string bucket_name = request["vectorBucketName"].get<std::string>();
    std::string index_name = request["indexName"].get<std::string>();
    int dimension = request["dimension"].get<int>();

    // Parse index configuration
    IndexConfig config;
    if (request.contains("indexType")) config.index_type = request["indexType"];
    if (request.contains("distanceMetric")) config.distance_metric = request["distanceMetric"];
    if (request.contains("numPartitions")) config.num_partitions = request["numPartitions"];
    if (request.contains("numSubVectors")) config.num_sub_vectors = request["numSubVectors"];
    if (request.contains("maxIterations")) config.max_iterations = request["maxIterations"];
    if (request.contains("sampleRate")) config.sample_rate = request["sampleRate"];
    if (request.contains("accelerator")) config.accelerator = request["accelerator"];
    if (request.contains("nprobes")) config.nprobes = request["nprobes"];
    if (request.contains("refineFactor")) config.refine_factor = request["refineFactor"];
    if (request.contains("ef")) config.ef = request["ef"];

    // Default distance metric
    if (!request.contains("distanceMetric")) {
        config.distance_metric = "euclidean";
    }

    std::string bucket_path = utils::get_bucket_path(bucket_name);
    if (!utils::directory_exists(bucket_path)) {
        return make_error(404, "NotFoundException",
            "Vector bucket '" + bucket_name + "' not found");
    }

    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    if (utils::directory_exists(db_path)) {
        return make_error(409, "ConflictException",
            "Index '" + index_name + "' already exists");
    }

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
        return make_error(500, "InternalServerException", error_str);
    }

    lancedb_table_free(table);
    lancedb_connection_free(conn);

    // Create and save table state
    auto state = TableStateManager::instance().get_or_create(bucket_name, index_name);
    state->dimension = dimension;
    state->config = config;

    // Parse rebuild thresholds if provided
    if (request.contains("unindexedThreshold")) {
        state->unindexed_threshold = request["unindexedThreshold"];
    }
    if (request.contains("deletionRatioThreshold")) {
        state->deletion_ratio_threshold = request["deletionRatioThreshold"];
    }

    TableStateManager::instance().save_state(state);

    LOG_INFO("CREATE_INDEX", index_name,
             "Index created (dimension=" + std::to_string(dimension) +
             ", type=" + config.index_type + ")");

    return make_success({
        {"indexName", index_name},
        {"vectorBucketName", bucket_name},
        {"dimension", dimension},
        {"config", config.to_json()}
    });
}

// PutVectors - Add vectors, update state, check if rebuild needed
ApiResponse PutVectors(const json& request) {
    if (!request.contains("vectors") || !request["vectors"].is_array()) {
        return make_error(400, "ValidationException", "vectors array is required");
    }
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName")) {
        return make_error(400, "ValidationException", "vectorBucketName is required");
    }

    std::string bucket_name = request["vectorBucketName"].get<std::string>();
    std::string index_name = request["indexName"].get<std::string>();

    // Get table state
    auto state = TableStateManager::instance().get_or_create(bucket_name, index_name);
    int dimension = state->dimension;

    if (dimension == 0) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found or dimension not configured");
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

        std::string function_name = "";
        std::string class_name = "";

        if (vec.contains("function_name") && vec["function_name"].is_string()) {
            function_name = vec["function_name"].get<std::string>();
        }
        if (vec.contains("class_name") && vec["class_name"].is_string()) {
            class_name = vec["class_name"].get<std::string>();
        }

        if (vec.contains("metadata") && vec["metadata"].is_object()) {
            const auto& meta = vec["metadata"];
            if (function_name.empty() && meta.contains("function_name") && meta["function_name"].is_string()) {
                function_name = meta["function_name"].get<std::string>();
            }
            if (class_name.empty() && meta.contains("class_name") && meta["class_name"].is_string()) {
                class_name = meta["class_name"].get<std::string>();
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

    // Update state: record insertions
    state->record_insertions(keys.size());
    TableStateManager::instance().save_state(state);

    // Log each vector insertion
    for (const auto& key : keys) {
        LOG_OP("PUT_VECTOR", index_name, key, "Vector inserted");
    }

    // Check if rebuild is needed and trigger if so
    bool rebuild_triggered = BackgroundIndexBuilder::instance().trigger_rebuild_if_needed(state);
    if (rebuild_triggered) {
        LOG_INFO("PUT_VECTOR", index_name, "Index rebuild triggered (threshold reached)");
    }

    return make_success({
        {"inserted", keys.size()},
        {"rebuildTriggered", rebuild_triggered},
        {"indexState", {
            {"insertionsSinceBuild", state->insertions_since_build.load()},
            {"deletionsSinceBuild", state->deletions_since_build.load()},
            {"indexBuildInProgress", state->index_build_in_progress.load()}
        }}
    });
}

// DeleteVectors - Delete vectors, update state, check if rebuild needed
ApiResponse DeleteVectors(const json& request) {
    if (!request.contains("keys") || !request["keys"].is_array()) {
        return make_error(400, "ValidationException", "keys array is required");
    }
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName")) {
        return make_error(400, "ValidationException", "vectorBucketName is required");
    }

    std::string bucket_name = request["vectorBucketName"].get<std::string>();
    std::string index_name = request["indexName"].get<std::string>();

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
    std::vector<std::string> keys;
    for (size_t i = 0; i < request["keys"].size(); i++) {
        if (i > 0) predicate += " OR ";
        std::string key = request["keys"][i].get<std::string>();
        predicate += "key = \"" + key + "\"";
        keys.push_back(key);
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

    // Update state: record deletions
    auto state = TableStateManager::instance().get_or_create(bucket_name, index_name);
    state->record_deletions(keys.size());
    TableStateManager::instance().save_state(state);

    // Log each vector deletion
    for (const auto& key : keys) {
        LOG_OP("DELETE_VECTOR", index_name, key, "Vector deleted");
    }

    // Check if rebuild is needed and trigger if so
    bool rebuild_triggered = BackgroundIndexBuilder::instance().trigger_rebuild_if_needed(state);
    if (rebuild_triggered) {
        LOG_INFO("DELETE_VECTOR", index_name, "Index rebuild triggered (deletion ratio threshold reached)");
    }

    return make_success({
        {"deleted", keys.size()},
        {"rebuildTriggered", rebuild_triggered},
        {"indexState", {
            {"insertionsSinceBuild", state->insertions_since_build.load()},
            {"deletionsSinceBuild", state->deletions_since_build.load()},
            {"indexBuildInProgress", state->index_build_in_progress.load()}
        }}
    });
}

// QueryVectors - Search using index configuration
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
    if (!request.contains("vectorBucketName")) {
        return make_error(400, "ValidationException", "vectorBucketName is required");
    }

    std::string bucket_name = request["vectorBucketName"].get<std::string>();
    std::string index_name = request["indexName"].get<std::string>();

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

    // Get table state and configuration
    auto state = TableStateManager::instance().get_or_create(bucket_name, index_name);
    IndexConfig config = state->config;

    // Override config with request-level query parameters
    if (request.contains("nprobes")) config.nprobes = request["nprobes"];
    if (request.contains("refineFactor")) config.refine_factor = request["refineFactor"];
    if (request.contains("ef")) config.ef = request["ef"];

    int dimension = state->dimension;
    if (static_cast<int>(query_vector.size()) != dimension && dimension > 0) {
        return make_error(400, "ValidationException",
            "Query vector dimension mismatch. Expected " + std::to_string(dimension) +
            ", got " + std::to_string(query_vector.size()));
    }

    // Parse filter if provided
    std::string filter;
    if (request.contains("filter")) {
        filter = request["filter"].dump();
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
        return_distance, return_metadata, config.to_lancedb_distance(),
        config, error);

    lancedb_table_free(table);
    lancedb_connection_free(conn);

    if (!error.empty() && results.empty()) {
        return make_error(500, "InternalServerException", error);
    }

    LOG_INFO("SEARCH_VECTOR", index_name,
             "Query executed (topK=" + std::to_string(top_k) +
             ", results=" + std::to_string(results.size()) + ")");

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
        {"distanceMetric", config.distance_metric},
        {"indexType", config.index_type},
        {"vectors", vectors_response},
        {"indexState", {
            {"insertionsSinceBuild", state->insertions_since_build.load()},
            {"deletionsSinceBuild", state->deletions_since_build.load()},
            {"indexBuildInProgress", state->index_build_in_progress.load()}
        }}
    });
}

// TriggerRebuild - Manually trigger index rebuild
ApiResponse TriggerRebuild(const json& request) {
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName")) {
        return make_error(400, "ValidationException", "vectorBucketName is required");
    }

    std::string bucket_name = request["vectorBucketName"].get<std::string>();
    std::string index_name = request["indexName"].get<std::string>();

    auto state = TableStateManager::instance().get_or_create(bucket_name, index_name);

    // Update config if provided
    if (request.contains("indexType")) state->config.index_type = request["indexType"];
    if (request.contains("numPartitions")) state->config.num_partitions = request["numPartitions"];
    if (request.contains("numSubVectors")) state->config.num_sub_vectors = request["numSubVectors"];
    if (request.contains("accelerator")) state->config.accelerator = request["accelerator"];

    TableStateManager::instance().save_state(state);

    bool triggered = BackgroundIndexBuilder::instance().force_trigger_rebuild(state);

    if (triggered) {
        LOG_INFO("TRIGGER_REBUILD", index_name, "Manual rebuild triggered");
        return make_success({
            {"message", "Index rebuild triggered"},
            {"indexState", state->to_json()}
        });
    } else {
        return make_success({
            {"message", "Index build already in progress"},
            {"indexState", state->to_json()}
        });
    }
}

// GetIndexState - Get current index state
ApiResponse GetIndexState(const json& request) {
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName")) {
        return make_error(400, "ValidationException", "vectorBucketName is required");
    }

    std::string bucket_name = request["vectorBucketName"].get<std::string>();
    std::string index_name = request["indexName"].get<std::string>();

    std::string db_path = utils::get_index_db_path(bucket_name, index_name);
    if (!utils::directory_exists(db_path)) {
        return make_error(404, "NotFoundException",
            "Index '" + index_name + "' not found");
    }

    auto state = TableStateManager::instance().get_or_create(bucket_name, index_name);

    // Get current table stats
    LanceDBConnection* conn = LanceDBHelper::connect(db_path);
    if (conn) {
        LanceDBTable* table = LanceDBHelper::open_table(conn, "vectors");
        if (table) {
            unsigned long long current_rows = lancedb_table_count_rows(table);
            unsigned long long current_version = lancedb_table_version(table);

            json response = state->to_json();
            response["currentRows"] = current_rows;
            response["currentVersion"] = current_version;
            response["needsRebuild"] = state->needs_rebuild();

            lancedb_table_free(table);
            lancedb_connection_free(conn);

            return make_success(response);
        }
        lancedb_connection_free(conn);
    }

    return make_success(state->to_json());
}

// UpdateIndexConfig - Update index configuration
ApiResponse UpdateIndexConfig(const json& request) {
    if (!request.contains("indexName")) {
        return make_error(400, "ValidationException", "indexName is required");
    }
    if (!request.contains("vectorBucketName")) {
        return make_error(400, "ValidationException", "vectorBucketName is required");
    }

    std::string bucket_name = request["vectorBucketName"].get<std::string>();
    std::string index_name = request["indexName"].get<std::string>();

    auto state = TableStateManager::instance().get_or_create(bucket_name, index_name);

    // Update config fields
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (request.contains("indexType")) state->config.index_type = request["indexType"];
        if (request.contains("numPartitions")) state->config.num_partitions = request["numPartitions"];
        if (request.contains("numSubVectors")) state->config.num_sub_vectors = request["numSubVectors"];
        if (request.contains("maxIterations")) state->config.max_iterations = request["maxIterations"];
        if (request.contains("sampleRate")) state->config.sample_rate = request["sampleRate"];
        if (request.contains("accelerator")) state->config.accelerator = request["accelerator"];
        if (request.contains("nprobes")) state->config.nprobes = request["nprobes"];
        if (request.contains("refineFactor")) state->config.refine_factor = request["refineFactor"];
        if (request.contains("ef")) state->config.ef = request["ef"];

        // Update thresholds
        if (request.contains("unindexedThreshold")) {
            state->unindexed_threshold = request["unindexedThreshold"];
        }
        if (request.contains("deletionRatioThreshold")) {
            state->deletion_ratio_threshold = request["deletionRatioThreshold"];
        }
    }

    TableStateManager::instance().save_state(state);

    LOG_INFO("UPDATE_CONFIG", index_name, "Index configuration updated");

    return make_success({
        {"message", "Configuration updated"},
        {"config", state->config.to_json()},
        {"thresholds", {
            {"unindexedThreshold", state->unindexed_threshold},
            {"deletionRatioThreshold", state->deletion_ratio_threshold}
        }}
    });
}

// ============================================================================
// Command Line Interface
// ============================================================================

void print_help() {
    std::cout << R"(
S3 Vector Concurrent Service with Background Indexing

USAGE:
    s3vector_concurrent_service <command> <json>

COMMANDS:
    CreateVectorBucket      Create a new vector bucket
    CreateIndex             Create an index with configuration
    PutVectors              Add vectors (updates state, may trigger rebuild)
    DeleteVectors           Delete vectors (updates state, may trigger rebuild)
    QueryVectors            Search vectors using index
    TriggerRebuild          Manually trigger index rebuild
    GetIndexState           Get current index state and stats
    UpdateIndexConfig       Update index configuration and thresholds
    --help, -h              Show this help message

DESIGN:
    - On each PutVectors/DeleteVectors:
      1. Execute the operation
      2. Update state counters (insertions/deletions)
      3. Check if rebuild threshold is reached
      4. If yes, spawn background thread for index rebuild
      5. Return immediately (non-blocking)

    - During rebuild:
      - Queries continue using existing index + brute force on new data
      - Put/Delete operations continue normally
      - Only one rebuild per table at a time

EXAMPLES:

    # Create a bucket
    ./s3vector_concurrent_service CreateVectorBucket '{"vectorBucketName": "my-bucket"}'

    # Create index with IVF_HNSW_PQ configuration
    ./s3vector_concurrent_service CreateIndex '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index",
        "dimension": 768,
        "indexType": "IVF_HNSW_PQ",
        "distanceMetric": "cosine",
        "numPartitions": 256,
        "unindexedThreshold": 500,
        "deletionRatioThreshold": 0.15
    }'

    # Put vectors (may trigger rebuild if threshold reached)
    ./s3vector_concurrent_service PutVectors '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index",
        "vectors": [
            {"key": "v1", "data": [0.1, 0.2, ...]}
        ]
    }'

    # Delete vectors (may trigger rebuild if deletion ratio reached)
    ./s3vector_concurrent_service DeleteVectors '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index",
        "keys": ["v1", "v2"]
    }'

    # Query with custom nprobes
    ./s3vector_concurrent_service QueryVectors '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index",
        "queryVector": [0.1, 0.2, ...],
        "topK": 10,
        "nprobes": 20,
        "returnDistance": true
    }'

    # Trigger manual rebuild
    ./s3vector_concurrent_service TriggerRebuild '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index"
    }'

    # Get index state
    ./s3vector_concurrent_service GetIndexState '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index"
    }'

    # Update thresholds
    ./s3vector_concurrent_service UpdateIndexConfig '{
        "vectorBucketName": "my-bucket",
        "indexName": "my-index",
        "unindexedThreshold": 2000,
        "deletionRatioThreshold": 0.25
    }'

INDEX TYPES:
    IVF_FLAT      - Inverted file index (no compression)
    IVF_PQ        - IVF with product quantization (default)
    IVF_HNSW_PQ   - IVF with HNSW graph and PQ
    IVF_HNSW_SQ   - IVF with HNSW graph and scalar quantization

QUERY PARAMETERS:
    nprobes       - Number of IVF partitions to search (higher = more accurate)
    refineFactor  - Number of candidates to refine (higher = more accurate)
    ef            - HNSW ef parameter (for HNSW indices only)

REBUILD THRESHOLDS:
    unindexedThreshold     - Trigger rebuild when this many new vectors (default: 1000)
    deletionRatioThreshold - Trigger rebuild when deleted/total ratio exceeds (default: 0.20)

LOGGING:
    All operations are logged with timestamps to:
    - Console (stdout)
    - /tmp/s3vectors/operations.log

STORAGE:
    All data is stored under /tmp/s3vectors/
    - Buckets: /tmp/s3vectors/<bucket-name>/
    - Indices: /tmp/s3vectors/<bucket-name>/<index-name>_lancedb/
    - State:   /tmp/s3vectors/<bucket-name>/.s3v_metadata/<index-name>_index_state.json

)";
}

int main(int argc, char* argv[]) {
    // Ensure root directory exists
    if (!utils::directory_exists(S3_VECTORS_ROOT)) {
        utils::create_directories(S3_VECTORS_ROOT);
    }

    // Initialize logger
    std::string log_path = S3_VECTORS_ROOT + "/" + LOG_FILE_NAME;
    Logger::instance().init(log_path, Logger::INFO, true);

    if (argc < 2) {
        print_help();
        return 1;
    }

    std::string command = argv[1];

    if (command == "--help" || command == "-h") {
        print_help();
        return 0;
    }

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <command> <json>\n";
        std::cerr << "Use --help for more information.\n";
        return 1;
    }

    std::string json_input = argv[2];

    json request;
    try {
        request = json::parse(json_input);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON input: " << e.what() << std::endl;
        return 1;
    }

    ApiResponse response;

    if (command == "CreateVectorBucket") {
        response = CreateVectorBucket(request);
    } else if (command == "CreateIndex") {
        response = CreateIndex(request);
    } else if (command == "PutVectors") {
        response = PutVectors(request);
    } else if (command == "DeleteVectors") {
        response = DeleteVectors(request);
    } else if (command == "QueryVectors") {
        response = QueryVectors(request);
    } else if (command == "TriggerRebuild") {
        response = TriggerRebuild(request);
    } else if (command == "GetIndexState") {
        response = GetIndexState(request);
    } else if (command == "UpdateIndexConfig") {
        response = UpdateIndexConfig(request);
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        print_help();
        return 1;
    }

    std::cout << response.to_string() << std::endl;

    // Give background thread a moment to start if triggered
    // (for visual feedback in logs)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    return response.is_success() ? 0 : 1;
}
