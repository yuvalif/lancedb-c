/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright The LanceDB Authors
 *
 * Code Search Example - Demonstrates vector similarity search with metadata filtering
 *
 * This program:
 * 1. Loads embedding vectors from a NumPy (.npy) file
 * 2. Loads code chunk metadata from JSON files
 * 3. Creates a LanceDB table with vector and scalar indices
 * 4. Performs similarity search with optional metadata filters
 * 5. Converts text queries to embeddings using AWS Bedrock Titan
 *
 * note: before running this application, it needs to create the embeddings and the metadata json files using a container (https://hub.docker.com/repository/docker/galsl/semantic-search/general)
 *
 * Environment variables for AWS Bedrock:
 *   AWS_ACCESS_KEY_ID     - AWS access key
 *   AWS_SECRET_ACCESS_KEY - AWS secret key
 *   AWS_REGION            - AWS region (default: us-east-1)
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <curl/curl.h>
#include <openssl/hmac.h>
#include <openssl/sha.h>

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <nlohmann/json.hpp>

#include "lancedb.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

// =============================================================================
// CONSTANTS
// =============================================================================

constexpr size_t EMBEDDING_DIM = 1024;
const std::string DEFAULT_AWS_REGION = "us-east-1";
const std::string BEDROCK_MODEL_ID = "amazon.titan-embed-text-v2:0";

// =============================================================================
// AWS BEDROCK EMBEDDING CLIENT
// =============================================================================

// CURL write callback
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    static_cast<std::string*>(userp)->append(static_cast<char*>(contents), total_size);
    return total_size;
}

//purpose: this class is used to interact with AWS Bedrock service to get text embeddings using the Titan model
class BedrockEmbeddingClient {
public:
    BedrockEmbeddingClient() {
        // Read credentials from environment variables
        const char* access_key = std::getenv("AWS_ACCESS_KEY_ID");
        const char* secret_key = std::getenv("AWS_SECRET_ACCESS_KEY");
        const char* region = std::getenv("AWS_REGION");

        if (access_key) access_key_ = access_key;
        if (secret_key) secret_key_ = secret_key;
        region_ = region ? region : DEFAULT_AWS_REGION;

        // Initialize CURL globally (once per process)
        static bool curl_initialized = false;
        if (!curl_initialized) {
            curl_global_init(CURL_GLOBAL_DEFAULT);
            curl_initialized = true;
        }
    }

    bool is_configured() const {
        return !access_key_.empty() && !secret_key_.empty();
    }

    std::string get_region() const { return region_; }

    // Convert text to embedding vector using AWS Bedrock Titan
    std::vector<float> get_embedding(const std::string& text) {
        std::vector<float> embedding;

        if (!is_configured()) {
            std::cerr << "Error: AWS credentials not configured" << std::endl;
            std::cerr << "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables" << std::endl;
            return embedding;
        }

        // Prepare request
        std::string host = "bedrock-runtime." + region_ + ".amazonaws.com";
        std::string endpoint = "https://" + host + "/model/" + BEDROCK_MODEL_ID + "/invoke";

        // Build request body
        json request_body;
        request_body["inputText"] = text;
        std::string body = request_body.dump();

        // Get current time for signing
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm* gmt = std::gmtime(&now_time);

        char date_stamp[9];
        char amz_date[17];
        std::strftime(date_stamp, sizeof(date_stamp), "%Y%m%d", gmt);
        std::strftime(amz_date, sizeof(amz_date), "%Y%m%dT%H%M%SZ", gmt);

        // Create canonical request and sign
        std::string signed_headers = "content-type;host;x-amz-date";
        std::string content_type = "application/json";

        // Calculate payload hash
        std::string payload_hash = sha256_hex(body);

        // Canonical request
        std::string canonical_uri = "/model/" + url_encode(BEDROCK_MODEL_ID) + "/invoke";
        std::string canonical_querystring = "";
        std::string canonical_headers =
            "content-type:" + content_type + "\n" +
            "host:" + host + "\n" +
            "x-amz-date:" + amz_date + "\n";

        std::string canonical_request =
            "POST\n" +
            canonical_uri + "\n" +
            canonical_querystring + "\n" +
            canonical_headers + "\n" +
            signed_headers + "\n" +
            payload_hash;

        // String to sign
        std::string algorithm = "AWS4-HMAC-SHA256";
        std::string credential_scope = std::string(date_stamp) + "/" + region_ + "/bedrock/aws4_request";
        std::string string_to_sign =
            algorithm + "\n" +
            amz_date + "\n" +
            credential_scope + "\n" +
            sha256_hex(canonical_request);

        // Calculate signature
        std::string signing_key = get_signature_key(secret_key_, date_stamp, region_, "bedrock");
        std::string signature = hmac_sha256_hex(signing_key, string_to_sign);

        // Authorization header
        std::string authorization_header =
            algorithm + " " +
            "Credential=" + access_key_ + "/" + credential_scope + ", " +
            "SignedHeaders=" + signed_headers + ", " +
            "Signature=" + signature;

        // Make HTTP request with CURL
        CURL* curl = curl_easy_init();
        if (!curl) {
            std::cerr << "Failed to initialize CURL" << std::endl;
            return embedding;
        }

        std::string response_body;
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, ("Content-Type: " + content_type).c_str());
        headers = curl_slist_append(headers, ("X-Amz-Date: " + std::string(amz_date)).c_str());
        headers = curl_slist_append(headers, ("Authorization: " + authorization_header).c_str());

        curl_easy_setopt(curl, CURLOPT_URL, endpoint.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

        CURLcode res = curl_easy_perform(curl);

        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
            return embedding;
        }

        if (http_code != 200) {
            std::cerr << "Bedrock API error (HTTP " << http_code << "): " << response_body << std::endl;
            return embedding;
        }

        // Parse response
        try {
            json response = json::parse(response_body);
            if (response.contains("embedding") && response["embedding"].is_array()) {
                for (const auto& val : response["embedding"]) {
                    embedding.push_back(val.get<float>());
                }
            } else {
                std::cerr << "Unexpected response format: " << response_body << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse Bedrock response: " << e.what() << std::endl;
        }

        return embedding;
    }

private:
    std::string access_key_;
    std::string secret_key_;
    std::string region_;

    // URL encode a string (AWS requires uppercase hex)
    static std::string url_encode(const std::string& value) {
        std::ostringstream escaped;
        escaped.fill('0');
        escaped << std::hex << std::uppercase;

        for (char c : value) {
            if (isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.' || c == '~') {
                escaped << c;
            } else {
                escaped << '%' << std::setw(2) << static_cast<int>(static_cast<unsigned char>(c));
            }
        }
        return escaped.str();
    }

    // SHA256 hash as hex string
    static std::string sha256_hex(const std::string& data) {
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(data.c_str()), data.size(), hash);

        std::ostringstream ss;
        ss << std::hex << std::setfill('0');
        for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
            ss << std::setw(2) << static_cast<int>(hash[i]);
        }
        return ss.str();
    }

    // HMAC-SHA256
    static std::string hmac_sha256(const std::string& key, const std::string& data) {
        unsigned char hash[EVP_MAX_MD_SIZE];
        unsigned int hash_len;

        HMAC(EVP_sha256(),
             key.c_str(), static_cast<int>(key.size()),
             reinterpret_cast<const unsigned char*>(data.c_str()), data.size(),
             hash, &hash_len);

        return std::string(reinterpret_cast<char*>(hash), hash_len);
    }

    // HMAC-SHA256 as hex string
    static std::string hmac_sha256_hex(const std::string& key, const std::string& data) {
        std::string hash = hmac_sha256(key, data);

        std::ostringstream ss;
        ss << std::hex << std::setfill('0');
        for (unsigned char c : hash) {
            ss << std::setw(2) << static_cast<int>(c);
        }
        return ss.str();
    }

    // Generate AWS Signature Key
    static std::string get_signature_key(const std::string& key,
                                         const std::string& date_stamp,
                                         const std::string& region,
                                         const std::string& service) {
        std::string k_date = hmac_sha256("AWS4" + key, date_stamp);
        std::string k_region = hmac_sha256(k_date, region);
        std::string k_service = hmac_sha256(k_region, service);
        std::string k_signing = hmac_sha256(k_service, "aws4_request");
        return k_signing;
    }
};

// =============================================================================
// DATA STRUCTURES
// =============================================================================

struct CodeChunk {
	//this structure holds the code chunk metadata loaded from the json files
    std::string code;
    std::string file_path;
    int32_t line_start = 0;
    int32_t line_end = 0;
    std::string type;
    std::string function_name;
    std::string class_name;
    std::string return_type;
    bool is_template = false;
    std::string access;
    std::string ns;  // namespace (reserved keyword)
    std::string parameters;
    int embedding_index = 0;
};

struct SearchResult {
    std::string code;
    std::string file_path;
    int32_t line_start = 0;
    int32_t line_end = 0;
    std::string type;
    std::string class_name;
    std::string function_name;
    float distance = 0.0f;
};

// =============================================================================
// NPY FILE PARSER
// =============================================================================

//purpose : this class is used to read numpy .npy files containing float32 vectors, the npy file is a product of creating the embeddings using AWS Bedrock Titan model on the code chunks
//please review the above notes for more details
class NpyReader {
public:
    static std::vector<std::vector<float>> load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open NPY file: " + path);
        }

        // Read and validate magic number
        char magic[6];
        file.read(magic, 6);
        if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
            throw std::runtime_error("Invalid NPY file format");
        }

        // Read version
        uint8_t major_version, minor_version;
        file.read(reinterpret_cast<char*>(&major_version), 1);
        file.read(reinterpret_cast<char*>(&minor_version), 1);

        // Read header length
        uint32_t header_len = 0;
        if (major_version == 1) {
            uint16_t len16;
            file.read(reinterpret_cast<char*>(&len16), 2);
            header_len = len16;
        } else {
            file.read(reinterpret_cast<char*>(&header_len), 4);
        }

        // Read header (Python dict string)
        std::string header(header_len, '\0');
        file.read(&header[0], header_len);

        // Parse shape from header (e.g., "'shape': (693, 1024)") i.e. rows = 693, dimensions = 1024
        auto [rows, cols] = parse_shape(header);

        std::cout << "Loading NPY: " << rows << " vectors x " << cols << " dimensions" << std::endl;

        // Read data (it is not scalable for huge files, but works for moderate sizes)
        std::vector<std::vector<float>> vectors(rows, std::vector<float>(cols));//is it scaled to read row float32
        for (size_t i = 0; i < rows; i++) {
            file.read(reinterpret_cast<char*>(vectors[i].data()), cols * sizeof(float));
        }

        if (!file) {
            throw std::runtime_error("Error reading NPY data");
        }

        return vectors;
    }

private:
    static std::pair<size_t, size_t> parse_shape(const std::string& header) {
        // Find 'shape': (rows, cols)
        auto shape_pos = header.find("'shape'");
        if (shape_pos == std::string::npos) {
            shape_pos = header.find("\"shape\"");
        }
        if (shape_pos == std::string::npos) {
            throw std::runtime_error("Could not find shape in NPY header");
        }

        auto paren_start = header.find('(', shape_pos);
        auto paren_end = header.find(')', paren_start);
        if (paren_start == std::string::npos || paren_end == std::string::npos) {
            throw std::runtime_error("Could not parse shape from NPY header");
        }

        std::string shape_str = header.substr(paren_start + 1, paren_end - paren_start - 1);

        // Parse "rows, cols"
        auto comma_pos = shape_str.find(',');
        if (comma_pos == std::string::npos) {
            throw std::runtime_error("Invalid shape format in NPY header");
        }

        size_t rows = std::stoull(shape_str.substr(0, comma_pos));
        size_t cols = std::stoull(shape_str.substr(comma_pos + 1));

        return {rows, cols};
    }
};

// =============================================================================
// JSON CHUNK LOADER
// =============================================================================

class ChunkLoader {
public:
    static std::vector<CodeChunk> load(const std::string& data_dir) {
        std::vector<CodeChunk> chunks;

	//it scans the data directory recursively for json files and loads them into CodeChunk structures
        for (const auto& entry : fs::recursive_directory_iterator(data_dir)) {
            if (entry.path().extension() == ".json") {
                try {
                    std::ifstream file(entry.path());
                    if (!file.is_open()) continue;

                    json j = json::parse(file);
		    //json into CodeChunk struct
                    CodeChunk chunk = parse_chunk(j);
                    chunks.push_back(std::move(chunk));
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to parse " << entry.path()
                              << ": " << e.what() << std::endl;
                }
            }
        }

        // Sort by embedding_index to match vectors.npy order
        std::sort(chunks.begin(), chunks.end(),
                  [](const CodeChunk& a, const CodeChunk& b) {
                      return a.embedding_index < b.embedding_index;
                  });

        std::cout << "Loaded " << chunks.size() << " code chunks" << std::endl;
        return chunks;
    }

private:
    static CodeChunk parse_chunk(const json& j) {
        CodeChunk chunk;

        chunk.code = j.value("code", "");
        chunk.embedding_index = j.value("embedding_index", 0);

	//JSON -> CodeChunk metadata fields
        if (j.contains("metadata") && j["metadata"].is_object()) {
            const auto& meta = j["metadata"];
            chunk.file_path = meta.value("file_path", "");
            chunk.line_start = meta.value("line_start", 0);
            chunk.line_end = meta.value("line_end", 0);
            chunk.type = get_string_or_empty(meta, "type");
            chunk.function_name = get_string_or_empty(meta, "function_name");
            chunk.class_name = get_string_or_empty(meta, "class_name");
            chunk.return_type = get_string_or_empty(meta, "return_type");
            chunk.is_template = meta.value("is_template", false);
            chunk.access = get_string_or_empty(meta, "access");
            chunk.ns = get_string_or_empty(meta, "namespace");
            chunk.parameters = get_string_or_empty(meta, "parameters");
        }

        return chunk;
    }

    static std::string get_string_or_empty(const json& j, const std::string& key) {
        if (j.contains(key) && j[key].is_string()) {
            return j[key].get<std::string>();
        }
        return "";
    }
};

// =============================================================================
// CODE SEARCH INDEX
// =============================================================================

class CodeSearchIndex {
public:
    explicit CodeSearchIndex(const std::string& db_path)
        : db_path_(db_path) {}

    ~CodeSearchIndex() {
        cleanup();
    }

    // Prevent copying
    CodeSearchIndex(const CodeSearchIndex&) = delete;
    CodeSearchIndex& operator=(const CodeSearchIndex&) = delete;

    // Create index from vectors and metadata (how this routine can handle huge data sets?)
    bool create_index(const std::vector<std::vector<float>>& vectors,
                      const std::vector<CodeChunk>& chunks) {

        if (vectors.size() != chunks.size()) {
            std::cerr << "Error: Vector count (" << vectors.size()
                      << ") != chunk count (" << chunks.size() << ")" << std::endl;
            return false;
        }

        // 1. Connect to database
        std::cout << "Connecting to database: " << db_path_ << std::endl;
        auto* builder = lancedb_connect(db_path_.c_str());
        if (!builder) {
            std::cerr << "Failed to create connection builder" << std::endl;
            return false;
        }

        db_ = lancedb_connect_builder_execute(builder);
        if (!db_) {
            lancedb_connect_builder_free(builder);
            std::cerr << "Failed to connect to database" << std::endl;
            return false;
        }

        // 2. Create Arrow schema
        std::cout << "Creating table schema..." << std::endl;
        auto schema = create_schema();//the schema contain the embedding vector and the metadata fields
        struct ArrowSchema c_schema;
        if (auto status = arrow::ExportSchema(*schema, &c_schema); !status.ok()) {
            std::cerr << "Failed to export schema: " << status.ToString() << std::endl;
            return false;
        }

        // 3. Build record batch with all data
        std::cout << "Building record batch with " << chunks.size() << " rows..." << std::endl;
        auto record_batch = build_record_batch(schema, vectors, chunks);
        if (!record_batch) {
            std::cerr << "Failed to build record batch" << std::endl;
            if (c_schema.release) c_schema.release(&c_schema);
            return false;
        }

        struct ArrowArray c_array;
        if (auto status = arrow::ExportRecordBatch(*record_batch, &c_array, &c_schema); !status.ok()) {
            std::cerr << "Failed to export record batch: " << status.ToString() << std::endl;
            if (c_schema.release) c_schema.release(&c_schema);
            return false;
        }

        // 4. Create table with initial data
        std::cout << "Creating LanceDB table 'code_chunks'..." << std::endl;
        auto* reader = lancedb_record_batch_reader_from_arrow(
            reinterpret_cast<FFI_ArrowArray*>(&c_array),
            reinterpret_cast<FFI_ArrowSchema*>(&c_schema));

        if (!reader) {
            std::cerr << "Failed to create record batch reader" << std::endl;
            if (c_array.release) c_array.release(&c_array);
            if (c_schema.release) c_schema.release(&c_schema);
            return false;
        }

        char* error_message = nullptr;
        LanceDBError result = lancedb_table_create(
            db_, "code_chunks",
            reinterpret_cast<FFI_ArrowSchema*>(&c_schema),
            reader, &table_, &error_message);

        if (result != LANCEDB_SUCCESS) {
            std::cerr << "Failed to create table: " << lancedb_error_to_message(result);
            if (error_message) {
                std::cerr << " - " << error_message;
                lancedb_free_string(error_message);
            }
            std::cerr << std::endl;
            return false;
        }

        std::cout << "Table created with " << lancedb_table_count_rows(table_) << " rows" << std::endl;

        // 5. Create vector index
        if (!create_vector_index()) {
            std::cerr << "Warning: Failed to create vector index" << std::endl;
        }

        // 6. Create scalar indices for filtering
        if (!create_scalar_indices()) {
            std::cerr << "Warning: Failed to create some scalar indices" << std::endl;
        }

        if (c_schema.release) c_schema.release(&c_schema);
        return true;
    }

    // Open existing index
    bool open_index() {
        auto* builder = lancedb_connect(db_path_.c_str());
        if (!builder) return false;

        db_ = lancedb_connect_builder_execute(builder);
        if (!db_) {
            lancedb_connect_builder_free(builder);
            return false;
        }

        table_ = lancedb_connection_open_table(db_, "code_chunks");
        if (!table_) {
            std::cerr << "Failed to open table 'code_chunks'" << std::endl;
            return false;
        }

        std::cout << "Opened existing index with "
                  << lancedb_table_count_rows(table_) << " rows" << std::endl;
        return true;
    }

    // Search with optional filter
    std::vector<SearchResult> search(const std::vector<float>& query_vector,
                                     size_t top_k,
                                     const std::string& filter = "") {
        std::vector<SearchResult> results;

        if (!table_) {
            std::cerr << "Error: Table not opened" << std::endl;
            return results;
        }

        if (query_vector.size() != EMBEDDING_DIM) {
            std::cerr << "Error: Query vector dimension mismatch. Expected "
                      << EMBEDDING_DIM << ", got " << query_vector.size() << std::endl;
            return results;
        }

        // 1. Create vector query
        auto* query = lancedb_vector_query_new(table_, query_vector.data(), EMBEDDING_DIM);
        if (!query) {
            std::cerr << "Failed to create vector query" << std::endl;
            return results;
        }

        // 2. Configure query
        char* error_message = nullptr;

        if (lancedb_vector_query_limit(query, top_k, &error_message) != LANCEDB_SUCCESS) {
            std::cerr << "Failed to set query limit" << std::endl;
            lancedb_vector_query_free(query);
            return results;
        }

        if (lancedb_vector_query_column(query, "embedding", &error_message) != LANCEDB_SUCCESS) {
            std::cerr << "Failed to set vector column" << std::endl;
            lancedb_vector_query_free(query);
            return results;
        }

        if (lancedb_vector_query_distance_type(query, LANCEDB_DISTANCE_COSINE, &error_message) != LANCEDB_SUCCESS) {
            std::cerr << "Failed to set distance type" << std::endl;
            lancedb_vector_query_free(query);
            return results;
        }

        // 3. Set columns to return (excluding the large embedding vector)
        const char* select_cols[] = {
            "code", "file_path", "line_start", "line_end",
            "type", "class_name", "function_name"
        };
        if (lancedb_vector_query_select(query, select_cols, 7, &error_message) != LANCEDB_SUCCESS) {
            std::cerr << "Failed to set select columns" << std::endl;
            lancedb_vector_query_free(query);
            return results;
        }

        // 4. Apply filter if provided
        if (!filter.empty()) {
            if (lancedb_vector_query_where_filter(query, filter.c_str(), &error_message) != LANCEDB_SUCCESS) {
                std::cerr << "Failed to set filter: " << filter;
                if (error_message) {
                    std::cerr << " - " << error_message;
                    lancedb_free_string(error_message);
                }
                std::cerr << std::endl;
                lancedb_vector_query_free(query);
                return results;
            }
        }

        // 5. Execute query
        auto* query_result = lancedb_vector_query_execute(query);
        if (!query_result) {
            std::cerr << "Failed to execute query" << std::endl;
            return results;
        }

        // 6. Convert to Arrow
        struct ArrowArray** c_arrays = nullptr;
        struct ArrowSchema* c_schema = nullptr;
        size_t count = 0;

        LanceDBError res = lancedb_query_result_to_arrow(
            query_result,
            reinterpret_cast<FFI_ArrowArray***>(&c_arrays),
            reinterpret_cast<FFI_ArrowSchema**>(&c_schema),
            &count, &error_message);

        if (res != LANCEDB_SUCCESS) {
            std::cerr << "Failed to convert query result to Arrow";
            if (error_message) {
                std::cerr << ": " << error_message;
                lancedb_free_string(error_message);
            }
            std::cerr << std::endl;
            return results;
        }

        // 7. Parse results
        results = parse_arrow_results(c_arrays, c_schema, count);

        // 8. Cleanup
        if (c_arrays && count > 0) {
            lancedb_free_arrow_arrays(reinterpret_cast<FFI_ArrowArray**>(c_arrays), count);
        }
        if (c_schema) {
            lancedb_free_arrow_schema(reinterpret_cast<FFI_ArrowSchema*>(c_schema));
        }

        return results;
    }

    // Get table statistics
    void print_stats() const {
        if (!table_) {
            std::cout << "Table not opened" << std::endl;
            return;
        }

        std::cout << "\n=== Index Statistics ===" << std::endl;
        std::cout << "Row count: " << lancedb_table_count_rows(table_) << std::endl;
        std::cout << "Version: " << lancedb_table_version(table_) << std::endl;

        // List indices
        char** indices = nullptr;
        size_t indices_count = 0;
        if (lancedb_table_list_indices(table_, &indices, &indices_count, nullptr) == LANCEDB_SUCCESS) {
            std::cout << "Indices (" << indices_count << "):" << std::endl;
            for (size_t i = 0; i < indices_count; i++) {
                std::cout << "  - " << indices[i] << std::endl;
            }
            lancedb_free_index_list(indices, indices_count);
        }
        std::cout << "========================\n" << std::endl;
    }

private:
    std::string db_path_;
    LanceDBConnection* db_ = nullptr;
    LanceDBTable* table_ = nullptr;

    void cleanup() {
        if (table_) {
            lancedb_table_free(table_);
            table_ = nullptr;
        }
        if (db_) {
            lancedb_connection_free(db_);
            db_ = nullptr;
        }
    }

    std::shared_ptr<arrow::Schema> create_schema() {
        return arrow::schema({
            arrow::field("embedding", arrow::fixed_size_list(arrow::float32(), EMBEDDING_DIM)),
            arrow::field("code", arrow::utf8()),
            arrow::field("file_path", arrow::utf8()),
            arrow::field("line_start", arrow::int32()),
            arrow::field("line_end", arrow::int32()),
            arrow::field("type", arrow::utf8()),
            arrow::field("function_name", arrow::utf8()),
            arrow::field("class_name", arrow::utf8()),
            arrow::field("return_type", arrow::utf8()),
            arrow::field("is_template", arrow::boolean()),
            arrow::field("access", arrow::utf8()),
            arrow::field("namespace", arrow::utf8()),
            arrow::field("parameters", arrow::utf8())
        });
    }

    std::shared_ptr<arrow::RecordBatch> build_record_batch(
            const std::shared_ptr<arrow::Schema>& schema,
            const std::vector<std::vector<float>>& vectors,
            const std::vector<CodeChunk>& chunks) {

        // Builders for each column
        arrow::FixedSizeListBuilder embedding_builder(
            arrow::default_memory_pool(),
            std::make_unique<arrow::FloatBuilder>(),
            EMBEDDING_DIM);

        arrow::StringBuilder code_builder;
        arrow::StringBuilder file_path_builder;
        arrow::Int32Builder line_start_builder;
        arrow::Int32Builder line_end_builder;
        arrow::StringBuilder type_builder;
        arrow::StringBuilder function_name_builder;
        arrow::StringBuilder class_name_builder;
        arrow::StringBuilder return_type_builder;
        arrow::BooleanBuilder is_template_builder;
        arrow::StringBuilder access_builder;//the schema contain the embedding vector and the metadata fields
        arrow::StringBuilder namespace_builder;
        arrow::StringBuilder parameters_builder;

        // Populate builders
        for (const auto& chunk : chunks) {
            // Embedding vector
            auto* float_builder = static_cast<arrow::FloatBuilder*>(
                embedding_builder.value_builder());

            const auto& vec = vectors[chunk.embedding_index];
            for (size_t j = 0; j < EMBEDDING_DIM && j < vec.size(); j++) {
                if (!float_builder->Append(vec[j]).ok()) {
                    std::cerr << "Failed to append float value" << std::endl;
                    return nullptr;
                }
            }
            if (!embedding_builder.Append().ok()) {
                std::cerr << "Failed to append embedding list" << std::endl;
                return nullptr;
            }

            // Metadata fields
            code_builder.Append(chunk.code).ok();
            file_path_builder.Append(chunk.file_path).ok();
            line_start_builder.Append(chunk.line_start).ok();
            line_end_builder.Append(chunk.line_end).ok();
            type_builder.Append(chunk.type).ok();
            function_name_builder.Append(chunk.function_name).ok();
            class_name_builder.Append(chunk.class_name).ok();
            return_type_builder.Append(chunk.return_type).ok();
            is_template_builder.Append(chunk.is_template).ok();
            access_builder.Append(chunk.access).ok();
            namespace_builder.Append(chunk.ns).ok();
            parameters_builder.Append(chunk.parameters).ok();
        }

        // Finish all arrays
        std::shared_ptr<arrow::Array> embedding_array;
        std::shared_ptr<arrow::Array> code_array;
        std::shared_ptr<arrow::Array> file_path_array;
        std::shared_ptr<arrow::Array> line_start_array;
        std::shared_ptr<arrow::Array> line_end_array;
        std::shared_ptr<arrow::Array> type_array;
        std::shared_ptr<arrow::Array> function_name_array;
        std::shared_ptr<arrow::Array> class_name_array;
        std::shared_ptr<arrow::Array> return_type_array;
        std::shared_ptr<arrow::Array> is_template_array;
        std::shared_ptr<arrow::Array> access_array;
        std::shared_ptr<arrow::Array> namespace_array;
        std::shared_ptr<arrow::Array> parameters_array;

        if (!embedding_builder.Finish(&embedding_array).ok() ||
            !code_builder.Finish(&code_array).ok() ||
            !file_path_builder.Finish(&file_path_array).ok() ||
            !line_start_builder.Finish(&line_start_array).ok() ||
            !line_end_builder.Finish(&line_end_array).ok() ||
            !type_builder.Finish(&type_array).ok() ||
            !function_name_builder.Finish(&function_name_array).ok() ||
            !class_name_builder.Finish(&class_name_array).ok() ||
            !return_type_builder.Finish(&return_type_array).ok() ||
            !is_template_builder.Finish(&is_template_array).ok() ||
            !access_builder.Finish(&access_array).ok() ||
            !namespace_builder.Finish(&namespace_array).ok() ||
            !parameters_builder.Finish(&parameters_array).ok()) {
            std::cerr << "Failed to finish building arrays" << std::endl;
            return nullptr;
        }

	//is it scaled to return a record batch with all the data?
        return arrow::RecordBatch::Make(
            schema,
            static_cast<int64_t>(chunks.size()),
            {
                embedding_array,
                code_array,
                file_path_array,
                line_start_array,
                line_end_array,
                type_array,
                function_name_array,
                class_name_array,
                return_type_array,
                is_template_array,
                access_array,
                namespace_array,
                parameters_array
            });
    }

    bool create_vector_index() {
        std::cout << "Creating vector index on 'embedding' column..." << std::endl;

        const char* vec_cols[] = {"embedding"};
        LanceDBVectorIndexConfig config = {
            .num_partitions = -1,        // auto
            .num_sub_vectors = -1,       // auto
            .max_iterations = -1,        // default
            .sample_rate = 0.0f,         // default
            .distance_type = LANCEDB_DISTANCE_COSINE,
            .accelerator = nullptr,      // CPU
            .replace = 1                 // replace if exists
        };

        char* error_message = nullptr;
        LanceDBError result = lancedb_table_create_vector_index(
            table_, vec_cols, 1, LANCEDB_INDEX_IVF_FLAT, &config, &error_message);

        if (result != LANCEDB_SUCCESS) {
            std::cerr << "Failed to create vector index: " << lancedb_error_to_message(result);
            if (error_message) {
                std::cerr << " - " << error_message;
                lancedb_free_string(error_message);
            }
            std::cerr << std::endl;
            return false;
        }

        std::cout << "Vector index created successfully" << std::endl;
        return true;
    }

    bool create_scalar_indices() {
        std::cout << "Creating scalar indices for filtering..." << std::endl;

        LanceDBScalarIndexConfig config = {
            .replace = 1,
            .force_update_statistics = 0
        };

        struct IndexDef {
            const char* column;
            LanceDBIndexType type;
        };

	// Define indices to create(is it limited to no more than 4 indices?)
	// how it is possible to create more indices if needed? or change the existing ones?
        std::vector<IndexDef> indices = {
            {"type", LANCEDB_INDEX_BITMAP},
            {"class_name", LANCEDB_INDEX_BITMAP},
            {"function_name", LANCEDB_INDEX_BITMAP},
            {"file_path", LANCEDB_INDEX_BTREE}
        };

        bool all_success = true;
        for (const auto& idx : indices) {
            const char* cols[] = {idx.column};
            char* error_message = nullptr;

	    //to measure create time of each index
            LanceDBError result = lancedb_table_create_scalar_index(
                table_, cols, 1, idx.type, &config, &error_message);

            if (result != LANCEDB_SUCCESS) {
                std::cerr << "  Warning: Failed to create index on '" << idx.column << "'";
                if (error_message) {
                    std::cerr << ": " << error_message;
                    lancedb_free_string(error_message);
                }
                std::cerr << std::endl;
                all_success = false;
            } else {
                std::cout << "  Created index on '" << idx.column << "'" << std::endl;
            }
        }

        return all_success;
    }

    std::vector<SearchResult> parse_arrow_results(
            struct ArrowArray** c_arrays,
            struct ArrowSchema* c_schema,
            size_t count) {

        std::vector<SearchResult> results;

        if (!c_arrays || !c_schema || count == 0) {
            return results;
        }

        auto schema_result = arrow::ImportSchema(c_schema);
        if (!schema_result.ok()) {
            std::cerr << "Failed to import schema: " << schema_result.status() << std::endl;
            return results;
        }
        auto schema = *schema_result;

        for (size_t i = 0; i < count; i++) {
            auto batch_result = arrow::ImportRecordBatch(
                reinterpret_cast<struct ArrowArray*>(c_arrays[i]), schema);

            if (!batch_result.ok()) {
                std::cerr << "Failed to import record batch: " << batch_result.status() << std::endl;
                continue;
            }
            auto batch = *batch_result;

            // Get column arrays
            auto code_col = std::dynamic_pointer_cast<arrow::StringArray>(
                batch->GetColumnByName("code"));
            auto file_path_col = std::dynamic_pointer_cast<arrow::StringArray>(
                batch->GetColumnByName("file_path"));
            auto line_start_col = std::dynamic_pointer_cast<arrow::Int32Array>(
                batch->GetColumnByName("line_start"));
            auto line_end_col = std::dynamic_pointer_cast<arrow::Int32Array>(
                batch->GetColumnByName("line_end"));
            auto type_col = std::dynamic_pointer_cast<arrow::StringArray>(
                batch->GetColumnByName("type"));
            auto class_name_col = std::dynamic_pointer_cast<arrow::StringArray>(
                batch->GetColumnByName("class_name"));
            auto function_name_col = std::dynamic_pointer_cast<arrow::StringArray>(
                batch->GetColumnByName("function_name"));
            auto distance_col = std::dynamic_pointer_cast<arrow::FloatArray>(
                batch->GetColumnByName("_distance"));

            for (int64_t row = 0; row < batch->num_rows(); row++) {
                SearchResult r;

                if (code_col && !code_col->IsNull(row)) {
                    r.code = code_col->GetString(row);
                }
                if (file_path_col && !file_path_col->IsNull(row)) {
                    r.file_path = file_path_col->GetString(row);
                }
                if (line_start_col && !line_start_col->IsNull(row)) {
                    r.line_start = line_start_col->Value(row);
                }
                if (line_end_col && !line_end_col->IsNull(row)) {
                    r.line_end = line_end_col->Value(row);
                }
                if (type_col && !type_col->IsNull(row)) {
                    r.type = type_col->GetString(row);
                }
                if (class_name_col && !class_name_col->IsNull(row)) {
                    r.class_name = class_name_col->GetString(row);
                }
                if (function_name_col && !function_name_col->IsNull(row)) {
                    r.function_name = function_name_col->GetString(row);
                }
                if (distance_col && !distance_col->IsNull(row)) {
                    r.distance = distance_col->Value(row);
                }

                results.push_back(std::move(r));
            }
        }

        return results;
    }
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

void print_results(const std::vector<SearchResult>& results, bool show_code = true) {
    if (results.empty()) {
        std::cout << "  No results found" << std::endl;
        return;
    }

    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];

        // Print separator
        std::cout << "─────────────────────────────────────────────────────────────────" << std::endl;

        // Print header
        std::cout << "  [" << (i + 1) << "] " << r.file_path
                  << ":" << r.line_start << "-" << r.line_end << std::endl;

        // Print metadata
        std::cout << "      Type: " << (r.type.empty() ? "(none)" : r.type);
        if (!r.class_name.empty()) {
            std::cout << " | Class: " << r.class_name;
        }
        if (!r.function_name.empty()) {
            std::cout << " | Function: " << r.function_name;
        }
        std::cout << std::endl;
        std::cout << "      Distance: " << std::fixed << std::setprecision(6)
                  << r.distance << std::endl;

        // Print code
        if (show_code && !r.code.empty()) {
            std::cout << std::endl << "      Code:" << std::endl;
            std::cout << "      ┌─────────────────────────────────────────────────────────" << std::endl;

            // Print code with line prefix
            std::istringstream code_stream(r.code);
            std::string line;
            while (std::getline(code_stream, line)) {
                std::cout << "      │ " << line << std::endl;
            }

            std::cout << "      └─────────────────────────────────────────────────────────" << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "─────────────────────────────────────────────────────────────────" << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <command> [options]\n\n"
              << "Commands:\n"
              << "  create <index_dir> <db_path>   Create index from vectors.npy and JSON files\n"
              << "  search <db_path> [options]     Search the index\n"
              << "  stats <db_path>                Show index statistics\n"
              << "\nSearch options:\n"
              << "  --query <text>                 Text query (converted to embedding via AWS Bedrock)\n"
              << "  --top-k <n>                    Number of results (default: 5)\n"
              << "  --filter <expr>                SQL WHERE clause filter\n"
              << "  --no-code                      Hide code in results (code is shown by default)\n"
              << "\nEnvironment variables for AWS Bedrock:\n"
              << "  AWS_ACCESS_KEY_ID              AWS access key\n"
              << "  AWS_SECRET_ACCESS_KEY          AWS secret key\n"
              << "  AWS_REGION                     AWS region (default: us-east-1)\n"
              << "\nFilter examples:\n"
              << "  --filter \"type = 'class'\"\n"
              << "  --filter \"class_name = 'AesEncryptor'\"\n"
              << "  --filter \"file_path LIKE '%encryption%'\"\n"
              << "  --filter \"type = 'function' AND class_name != ''\"\n"
              << "\nSearch examples:\n"
              << "  " << program_name << " search ./db --query \"encryption function\"\n"
              << "  " << program_name << " search ./db --query \"parse CSV\" --filter \"type = 'function'\"\n"
              << std::endl;
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "create") {
        if (argc < 4) {
            std::cerr << "Error: Missing arguments for 'create' command" << std::endl;
            print_usage(argv[0]);
            return 1;
        }

        std::string index_dir = argv[2];
        std::string db_path = argv[3];

        // Check if index directory exists
        if (!fs::exists(index_dir)) {
            std::cerr << "Error: Index directory does not exist: " << index_dir << std::endl;
            return 1;
        }

        // Load vectors
        std::string vectors_path = index_dir + "/vectors.npy";
        if (!fs::exists(vectors_path)) {
            std::cerr << "Error: vectors.npy not found in " << index_dir << std::endl;
            return 1;
        }

        std::vector<std::vector<float>> vectors;
        try {
            vectors = NpyReader::load(vectors_path);
        } catch (const std::exception& e) {
            std::cerr << "Error loading vectors: " << e.what() << std::endl;
            return 1;
        }

        // Load chunks
        std::string data_dir = index_dir + "/data";
        if (!fs::exists(data_dir)) {
            std::cerr << "Error: data directory not found in " << index_dir << std::endl;
            return 1;
        }

        std::vector<CodeChunk> chunks = ChunkLoader::load(data_dir);
        if (chunks.empty()) {
            std::cerr << "Error: No code chunks found" << std::endl;
            return 1;
        }

        // Remove existing database if present
        if (fs::exists(db_path)) {
            std::cout << "Removing existing database: " << db_path << std::endl;
            fs::remove_all(db_path);
        }

        // Create index
        CodeSearchIndex index(db_path);
        if (!index.create_index(vectors, chunks)) {
            std::cerr << "Failed to create index" << std::endl;
            return 1;
        }

        index.print_stats();
        std::cout << "Index created successfully at: " << db_path << std::endl;

    } else if (command == "search") {
        if (argc < 3) {
            std::cerr << "Error: Missing database path for 'search' command" << std::endl;
            print_usage(argv[0]);
            return 1;
        }

        std::string db_path = argv[2];
        size_t top_k = 5;
        std::string filter;
        std::string query_text;
        bool show_code = true;  // Show code by default

        // Parse options
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--top-k" && i + 1 < argc) {
                top_k = std::stoul(argv[++i]);
            } else if (arg == "--filter" && i + 1 < argc) {
                filter = argv[++i];
            } else if (arg == "--query" && i + 1 < argc) {
                query_text = argv[++i];
            } else if (arg == "--no-code") {
                show_code = false;
            }
        }

        // Check if query text is provided
        if (query_text.empty()) {
            std::cerr << "Error: --query <text> is required for search" << std::endl;
            print_usage(argv[0]);
            return 1;
        }

        // Initialize Bedrock client and get embedding
        BedrockEmbeddingClient bedrock;
        if (!bedrock.is_configured()) {
            std::cerr << "Error: AWS credentials not configured" << std::endl;
            std::cerr << "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables" << std::endl;
            return 1;
        }

        std::cout << "Converting query to embedding using AWS Bedrock (region: "
                  << bedrock.get_region() << ")..." << std::endl;
        std::cout << "Query: \"" << query_text << "\"" << std::endl;

        std::vector<float> query_vector = bedrock.get_embedding(query_text);
        if (query_vector.empty()) {
            std::cerr << "Failed to get embedding from AWS Bedrock" << std::endl;
            return 1;
        }

        if (query_vector.size() != EMBEDDING_DIM) {
            std::cerr << "Warning: Embedding dimension mismatch. Expected "
                      << EMBEDDING_DIM << ", got " << query_vector.size() << std::endl;
        }

        std::cout << "Embedding received (" << query_vector.size() << " dimensions)" << std::endl;

        // Open index
        CodeSearchIndex index(db_path);
        if (!index.open_index()) {
            std::cerr << "Failed to open index at: " << db_path << std::endl;
            return 1;
        }

        std::cout << "\n=== Search Results (top " << top_k << ") ===" << std::endl;
        std::cout << "Query: \"" << query_text << "\"" << std::endl;
        if (!filter.empty()) {
            std::cout << "Filter: " << filter << std::endl;
        }
        std::cout << std::endl;

        auto results = index.search(query_vector, top_k, filter);
        print_results(results, show_code);

    } else if (command == "stats") {
        if (argc < 3) {
            std::cerr << "Error: Missing database path for 'stats' command" << std::endl;
            print_usage(argv[0]);
            return 1;
        }

        std::string db_path = argv[2];

        CodeSearchIndex index(db_path);
        if (!index.open_index()) {
            std::cerr << "Failed to open index at: " << db_path << std::endl;
            return 1;
        }

        index.print_stats();

    } else if (command == "--help" || command == "-h") {
        print_usage(argv[0]);
        return 0;

    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
