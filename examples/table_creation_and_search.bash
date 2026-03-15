#!/bin/bash

#########################
s3vector_simulation()
{
/home/gsalomon/work/lancedb/yuvalif_c_binding/lancedb-c/build/s3vector_simulation "$@"	
}

#########################
create_vector_bucket() {
	s3vector_simulation CreateVectorBucket "{\"vectorBucketName\": \"$BUCKET_NAME\"}"
}

#########################
create_index() {
    	s3vector_simulation CreateIndex "{\"vectorBucketName\": \"$BUCKET_NAME\", 
	\"indexName\": \"$INDEX_NAME\", 
	\"dimension\": $DIMENSION, 
	\"distanceMetric\": \"$DISTANCE_METRIC\"
}"

}

#########################
create_batch_of_vectors() {
# the input is the npy that contains the vectors, and the metadata directory that contains the metadata files
# the JSON files are in batches of 10 vectors(according to batch-size).

if [ "$#" -ne 3 ]; then
    echo "Usage: create_batch_of_vectors <npy_file> <metadata_dir> <output_prefix>"
    return 1
fi

npy_file="$1"
metadata_dir="$2"
output_prefix="$3"
python3 ../examples/npy_to_s3vector.py --npy-file ${npy_file}  --metadata-dir ${metadata_dir}  --output ${output_prefix} --batch-size 10

}

###########################
create_put_vectors_request() {

#according to aws spec its possible to upload upto 500 vectors in a single request (not less than 1)

if [ "$#" -ne 1 ]; then
    echo "Usage: create_put_vectors_request <batch_file>"
    return 1
fi

local batch_file="$1"
echo $(
python3 -c "
import json
import sys

batch = json.load(open('$batch_file'))
request = {
    'vectorBucketName': '$BUCKET_NAME',
    'indexName': '$INDEX_NAME',
    'vectors': batch['vectors']
}
print(json.dumps(request))
" )

}

###########################
s3vector_putvectors() {
## the output of this function should be pipe into aws cli of put-vectors
if [ "$#" -ne 1 ]; then
    echo "Usage: s3vector_putvectors <batch_file>"
    return 1
fi
#create_put_vectors_request $1 > /tmp/put_vectors_request.json ## debug line
create_put_vectors_request $1 | s3vector_simulation PutVectors 
}

###########################
get_index(){
	s3vector_simulation GetIndex "{\"vectorBucketName\": \"$BUCKET_NAME\", \"indexName\": \"$INDEX_NAME\"}"
}

#########################
create_vector_ann_index() {
    # Creates ANN vector index on the 'data' column for fast similarity search
    # Optional parameters: indexType (IVF_FLAT, IVF_PQ, IVF_HNSW_PQ, IVF_HNSW_SQ)
    local index_type="${1:-IVF_FLAT}"
    echo "Creating vector index (type: $index_type)..."
    s3vector_simulation CreateVectorIndex "{
        \"vectorBucketName\": \"$BUCKET_NAME\",
        \"indexName\": \"$INDEX_NAME\",
        \"indexType\": \"$index_type\"
    }"
}

#########################
create_scalar_indices() {
    # Creates scalar indices on function_name and class_name columns for filtering
    echo "Creating scalar index on function_name column..."
    s3vector_simulation CreateScalarIndex "{
        \"vectorBucketName\": \"$BUCKET_NAME\",
        \"indexName\": \"$INDEX_NAME\",
        \"column\": \"function_name\",
        \"scalarIndexType\": \"BITMAP\"
    }"

    echo "Creating scalar index on class_name column..."
    s3vector_simulation CreateScalarIndex "{
        \"vectorBucketName\": \"$BUCKET_NAME\",
        \"indexName\": \"$INDEX_NAME\",
        \"column\": \"class_name\",
        \"scalarIndexType\": \"BITMAP\"
    }"
}

#########################
optimize_index() {
    # Compacts fragments and rebuilds indices
    local optimize_type="${1:-ALL}"
    echo "Optimizing index (type: $optimize_type)..."
    s3vector_simulation OptimizeIndex "{
        \"vectorBucketName\": \"$BUCKET_NAME\",
        \"indexName\": \"$INDEX_NAME\",
        \"optimizeType\": \"$optimize_type\"
    }"
}

#########################
list_lance_indices() {
    # Lists all LanceDB indices on the table
    s3vector_simulation ListLanceIndices "{
        \"vectorBucketName\": \"$BUCKET_NAME\",
        \"indexName\": \"$INDEX_NAME\"
    }"
}

#########################
generate_query_embedding() {
    # Generates embedding for a query string using AWS Bedrock Titan
    # Returns: JSON object with 'embedding' array and 'elapsed_ms' timing
    local query_text="$1"

    # Call Bedrock to generate embedding with timing
    python3 -c "
import boto3
import json
import time

start_time = time.time()

client = boto3.client('bedrock-runtime', region_name='us-west-2')
response = client.invoke_model(
    modelId='amazon.titan-embed-text-v2:0',
    body=json.dumps({'inputText': '''$query_text'''}),
    contentType='application/json',
    accept='application/json'
)
result = json.loads(response['body'].read())

elapsed_ms = (time.time() - start_time) * 1000

# Return both embedding and timing
output = {
    'embedding': result['embedding'],
    'elapsed_ms': elapsed_ms
}
print(json.dumps(output))
"
}

#########################
format_search_results() {
    # Formats search results in a readable manner
    # Input: JSON from QueryVectors via stdin
    python3 -c "
import json
import sys

data = json.load(sys.stdin)

if 'error' in data:
    print(f\"Error: {data['error']}\")
    sys.exit(1)

vectors = data.get('vectors', [])
distance_metric = data.get('distanceMetric', 'unknown')

print(f\"\n{'='*80}\")
print(f\"SEARCH RESULTS ({len(vectors)} matches, metric: {distance_metric})\")
print(f\"{'='*80}\")

for i, vec in enumerate(vectors, 1):
    key = vec.get('key', 'unknown')
    distance = vec.get('distance', 0)
    metadata = vec.get('metadata', {})

    # Handle nested metadata structure
    if isinstance(metadata, dict) and 'metadata' in metadata:
        inner_meta = metadata.get('metadata', {})
        code = metadata.get('code', '')
    else:
        inner_meta = metadata
        code = metadata.get('code', '')

    print(f\"\n{'─'*80}\")
    print(f\"Rank {i} | Key: {key} | Distance: {distance:.6f}\")
    print(f\"{'─'*80}\")

    # Display metadata fields
    file_path = inner_meta.get('file_path', 'unknown')
    line_start = inner_meta.get('line_start', '?')
    line_end = inner_meta.get('line_end', '?')
    func_name = inner_meta.get('function_name', '')
    class_name = inner_meta.get('class_name', '')
    return_type = inner_meta.get('return_type', '')
    chunk_type = inner_meta.get('type', '')

    print(f\"File: {file_path}:{line_start}-{line_end}\")

    if class_name and func_name:
        print(f\"Method: {class_name}::{func_name}\")
    elif func_name:
        print(f\"Function: {func_name}\")
    elif class_name:
        print(f\"Class: {class_name}\")

    if return_type:
        print(f\"Return Type: {return_type}\")
    if chunk_type:
        print(f\"Type: {chunk_type}\")

    # Display code in readable format
    if code:
        print(f\"\nCode:\")
        print(f\"{'─'*40}\")
        lines = code.split('\\n')
        # Show first 20 lines, truncate if longer
        for j, line in enumerate(lines[:20]):
            print(f\"  {line}\")
        if len(lines) > 20:
            print(f\"  ... ({len(lines) - 20} more lines)\")

print(f\"\n{'='*80}\")
"
}

#########################
query_vectors() {
    # Query vectors using human-language query
    # Args: query_text, top_k (optional), filter (optional)
    local query_text="$1"
    local top_k="${2:-10}"
    local filter="${3:-}"

    echo "" >&2
    echo "════════════════════════════════════════════════════════════════════════════════" >&2
    echo "VECTOR SEARCH" >&2
    echo "════════════════════════════════════════════════════════════════════════════════" >&2
    echo "Query: '$query_text'" >&2
    echo "" >&2

    # Generate embedding with timing
    echo "Generating query embedding via AWS Bedrock..." >&2
    local embedding_result=$(generate_query_embedding "$query_text")

    if [ -z "$embedding_result" ]; then
        echo "Error: Failed to generate embedding" >&2
        return 1
    fi

    # Extract embedding and timing
    local embedding=$(echo "$embedding_result" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['embedding']))")
    local embedding_time=$(echo "$embedding_result" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"{d['elapsed_ms']:.2f}\")")

    echo "✓ Embedding generated in ${embedding_time} ms" >&2

    # Build query request
    local request
    if [ -n "$filter" ]; then
        echo "Filter: $filter" >&2
        request=$(python3 -c "
import json
embedding = $embedding
request = {
    'vectorBucketName': '$BUCKET_NAME',
    'indexName': '$INDEX_NAME',
    'queryVector': {'float32': embedding},
    'topK': $top_k,
    'returnDistance': True,
    'returnMetadata': True,
    'filter': '''$filter'''
}
print(json.dumps(request))
")
    else
        request=$(python3 -c "
import json
embedding = $embedding
request = {
    'vectorBucketName': '$BUCKET_NAME',
    'indexName': '$INDEX_NAME',
    'queryVector': {'float32': embedding},
    'topK': $top_k,
    'returnDistance': True,
    'returnMetadata': True
}
print(json.dumps(request))
")
    fi

    # Execute search with timing
    echo "" >&2
    echo "Executing vector search (top_k=$top_k)..." >&2

    local search_start=$(python3 -c "import time; print(time.time())")
    local search_result=$(echo "$request" | s3vector_simulation QueryVectors)
    local search_end=$(python3 -c "import time; print(time.time())")

    local search_time=$(python3 -c "print(f'{($search_end - $search_start) * 1000:.2f}')")
    echo "✓ Search completed in ${search_time} ms" >&2

    # Display timing summary
    echo "" >&2
    echo "────────────────────────────────────────────────────────────────────────────────" >&2
    echo "TIMING SUMMARY" >&2
    echo "────────────────────────────────────────────────────────────────────────────────" >&2
    echo "  Embedding generation: ${embedding_time} ms" >&2
    echo "  Vector search:        ${search_time} ms" >&2
    local total_time=$(python3 -c "print(f'{$embedding_time + $search_time:.2f}')")
    echo "  Total:                ${total_time} ms" >&2

    # Format and display results
    echo "$search_result" | format_search_results
}


#########################
create_table_of_vectors()
{

# the vectors and the metadata directory are generated by other scripts that scan the code repository, chunk it, embed it and generate the metadata files.
create_batch_of_vectors ${1} ${2} ${3}

create_vector_bucket
create_index

## looping on the batch files and uploading them one by one
for batch_file in ${3}_batch_*.json; do
    echo "Uploading vectors from batch file: $batch_file"
    s3vector_putvectors $batch_file
done

## After all vectors are loaded, create indices for fast search
echo "Creating vector ANN index..."
create_vector_ann_index "IVF_FLAT"

echo "Creating scalar indices on function_name and class_name..."
create_scalar_indices

echo "Optimizing index (compaction)..."
optimize_index "ALL"

echo "Listing all indices..."
list_lance_indices

}

##################################################
# the s3vector_simulation is the  application that simulates the s3vector service locally.
DIMENSION=1024
DISTANCE_METRIC="cosine"


#check if BUCKET_NAME and INDEX_NAME are set
if [ -z "$BUCKET_NAME" ] || [ -z "$INDEX_NAME" ]; then
    echo "Error: BUCKET_NAME and INDEX_NAME environment variables must be set."
    return 1
fi

#check if ${1} is file and exists and ${2} is a directory and exists
if [ ! -f "${1}" ]; then
    echo "Error: ${1} is not a valid npy file."
    return 1
fi

if [ ! -d "${2}" ]; then
    echo "Error: ${2} is not a valid metadata directory."
    return 1
fi

# check if ${3} is set
if [ -z "${3}" ]; then
	echo "Error: output prefix ${3} is not set(argument #3)."
    return 1
fi

create_table_of_vectors ${1} ${2} ${3}
## compaction and optimization for the index fragments
