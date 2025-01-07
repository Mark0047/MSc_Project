import os
import json
import requests
import pandas as pd
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
import hashlib

# Constants and Configuration
CKAN_PACKAGE_LIST_URL = "https://ckan.publishing.service.gov.uk/api/action/package_list"
CKAN_PACKAGE_SHOW_URL = "https://ckan.publishing.service.gov.uk/api/action/package_show?id="

# Directories for file handling
DATA_FOLDER = "./Project/gov_data_json"
TEMP_DOWNLOAD_DIR = "./Project/temp_files"
SNAPSHOT_DIR = "./Project/snapshot"
CACHE_DIR = "./Project/cache"  # For caching metadata and embeddings

os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Embedding model details
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

if torch.cuda.is_available():
    embedding_model = embedding_model.to('cuda')

vector_dimension = 1024
faiss_index = faiss.IndexFlatL2(vector_dimension)

# Metadata and Document Tracking
document_metadata = []
processed_files = set()
metadata_embeddings = None  # FAISS index for metadata
metadata_index_path = os.path.join(SNAPSHOT_DIR, "metadata_faiss_index.bin")
metadata_data_path = os.path.join(SNAPSHOT_DIR, "metadata_data.json")
processed_files_tracker = os.path.join(SNAPSHOT_DIR, "processed_files.json")
metadata_cache_dir = os.path.join(CACHE_DIR, "metadata")
metadata_embedding_cache_dir = os.path.join(CACHE_DIR, "metadata_embeddings")

os.makedirs(metadata_cache_dir, exist_ok=True)
os.makedirs(metadata_embedding_cache_dir, exist_ok=True)

CHUNK_SIZE = 512  # For text chunking
TOP_N = 10        # Number of top relevant documents to fetch per query

# Initialize Metadata FAISS Index
def initialize_metadata_faiss_index():
    global metadata_embeddings
    if os.path.exists(metadata_index_path) and os.path.exists(metadata_data_path):
        print("Loading metadata FAISS index and data...")
        metadata_embeddings = faiss.read_index(metadata_index_path)
        with open(metadata_data_path, "r") as f:
            global document_metadata
            document_metadata = json.load(f)
    else:
        print("Initializing empty metadata FAISS index...")
        metadata_embeddings = faiss.IndexFlatL2(vector_dimension)

# Embedding Function
def embed_text(text):
    """
    Generate embeddings for a given text using the embedding model.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE)
    if torch.cuda.is_available():
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Caching Utilities
def get_cache_path(identifier, cache_type="metadata"):
    """
    Generate a cache file path based on a unique identifier.
    """
    hash_id = hashlib.md5(identifier.encode()).hexdigest()
    if cache_type == "metadata":
        return os.path.join(metadata_cache_dir, f"{hash_id}.json")
    elif cache_type == "metadata_embedding":
        return os.path.join(metadata_embedding_cache_dir, f"{hash_id}.npy")
    return None

def cache_metadata(package_id, metadata):
    """
    Cache the fetched metadata to avoid redundant API calls.
    """
    cache_path = get_cache_path(package_id, "metadata")
    with open(cache_path, "w") as f:
        json.dump(metadata, f)

def load_cached_metadata(package_id):
    """
    Load cached metadata if available.
    """
    cache_path = get_cache_path(package_id, "metadata")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return None

def cache_metadata_embedding(package_id, embedding):
    """
    Cache the metadata embedding.
    """
    cache_path = get_cache_path(package_id, "metadata_embedding")
    np.save(cache_path, embedding)

def load_cached_metadata_embedding(package_id):
    """
    Load cached metadata embedding if available.
    """
    cache_path = get_cache_path(package_id, "metadata_embedding")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return None

# Fetching Metadata
def fetch_all_package_ids():
    """
    Fetch all package IDs from the CKAN API.
    """
    try:
        response = requests.get(CKAN_PACKAGE_LIST_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            return data.get('result', [])
        else:
            print("Failed to fetch package list.")
            return []
    except Exception as e:
        print(f"Error fetching package list: {e}")
        return []

def fetch_package_metadata(package_id):
    """
    Fetch metadata for a specific package from the CKAN API.
    Utilizes caching to minimize API calls.
    """
    cached_metadata = load_cached_metadata(package_id)
    if cached_metadata:
        return cached_metadata

    try:
        response = requests.get(f"{CKAN_PACKAGE_SHOW_URL}{package_id}", timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            metadata = data.get('result', {})
            cache_metadata(package_id, metadata)
            return metadata
        else:
            print(f"Failed to fetch metadata for package {package_id}.")
            return None
    except Exception as e:
        print(f"Error fetching metadata for package {package_id}: {e}")
        return None

def process_and_embed_metadata(package_id):
    """
    Process and embed metadata for a given package ID.
    Returns the embedding if successful, else None.
    """
    metadata = fetch_package_metadata(package_id)
    if not metadata:
        return None

    # Convert relevant metadata fields to a single text string
    relevant_fields = [
        metadata.get('title', ''),
        metadata.get('notes', ''),
        metadata.get('description', ''),
        metadata.get('tags', [])
    ]
    metadata_text = " ".join([str(field) for field in relevant_fields if field])

    if not metadata_text.strip():
        return None

    # Check if embedding is cached
    cached_embedding = load_cached_metadata_embedding(package_id)
    if cached_embedding is not None:
        embedding = cached_embedding
    else:
        embedding = embed_text(metadata_text)
        cache_metadata_embedding(package_id, embedding)

    # Add to FAISS index
    metadata_embeddings.add(embedding)

    # Update document_metadata
    document_metadata.append({
        'package_id': package_id,
        'metadata_text': metadata_text,
        'relevant_fields': relevant_fields
    })

    return embedding

# Initialize Metadata FAISS Index
initialize_metadata_faiss_index()

# Load Processed Files Tracker
def load_processed_files():
    """
    Load the set of processed files from the tracker.
    """
    global processed_files
    if os.path.exists(processed_files_tracker):
        with open(processed_files_tracker, "r") as f:
            processed_files = set(json.load(f))
    else:
        processed_files = set()

def save_processed_files():
    """
    Save the set of processed files to the tracker.
    """
    with open(processed_files_tracker, "w") as f:
        json.dump(list(processed_files), f)

# Save and Load Metadata Snapshot
def save_metadata_snapshot():
    """
    Save the metadata FAISS index and data to disk.
    """
    faiss.write_index(metadata_embeddings, metadata_index_path)
    with open(metadata_data_path, "w") as f:
        json.dump(document_metadata, f)
    print("Metadata snapshot saved successfully.")

def load_metadata_snapshot():
    """
    Load the metadata FAISS index and data from a snapshot.
    """
    if os.path.exists(metadata_index_path) and os.path.exists(metadata_data_path):
        initialize_metadata_faiss_index()
        print("Metadata snapshot loaded successfully.")
        return True
    return False

# Document Handling Functions
def download_file(url, file_name):
    """
    Downloads a file from a URL and saves it locally.
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        file_path = os.path.join(TEMP_DOWNLOAD_DIR, file_name)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path
    except Exception as e:
        print(f"Error downloading file from {url}: {e}")
        return None

def read_csv_file(file_path):
    """
    Reads a CSV file and returns its content as text.
    """
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return ""

def process_text_chunks(text):
    """
    Split a large text into chunks for efficient embedding.
    """
    words = text.split()
    for i in range(0, len(words), CHUNK_SIZE):
        yield " ".join(words[i:i + CHUNK_SIZE])

def download_and_process_files(documents):
    """
    Downloads and processes the specified documents, updating the FAISS index accordingly.
    """
    new_embeddings = []
    new_metadata = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_doc = {
            executor.submit(process_single_document, doc): doc for doc in documents
        }
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            result = future.result()
            if result:
                embedding, metadata_entry = result
                new_embeddings.append(embedding)
                new_metadata.append(metadata_entry)

    if new_embeddings:
        faiss_index.add(np.vstack(new_embeddings))
        document_metadata.extend(new_metadata)
        save_snapshot()
        save_processed_files()
        print(f"Processed and added {len(new_embeddings)} document embeddings to FAISS index.")

def process_single_document(doc):
    """
    Downloads and processes a single document.
    Returns the embedding and metadata entry if successful.
    """
    file_name = doc.get('file_name')
    url = doc.get('url')
    if not file_name or not url:
        return None

    if file_name in processed_files:
        return None  # Skip already processed files

    file_path = download_file(url, file_name)
    if not file_path:
        print(f"Failed to download {url}")
        return None  # Skip if download failed

    file_content = read_csv_file(file_path)
    if not file_content.strip():
        print(f"No content in {file_path}")
        return None

    # Process chunks of text
    embeddings = []
    metadata_entries = []
    for chunk in process_text_chunks(file_content):
        embedding = embed_text(chunk)
        embeddings.append(embedding)
        metadata_entries.append({
            'file_name': file_name,
            'text': chunk
        })

    # Mark as processed
    processed_files.add(file_name)

    # Return combined embeddings and metadata entries
    return np.vstack(embeddings), metadata_entries

# Save and Load Snapshot
def save_snapshot():
    """
    Save both metadata and document FAISS indexes and metadata to disk.
    """
    save_metadata_snapshot()
    faiss.write_index(faiss_index, os.path.join(SNAPSHOT_DIR, "faiss_index.bin"))
    with open(os.path.join(SNAPSHOT_DIR, "document_metadata.json"), "w") as f:
        json.dump(document_metadata, f)
    with open(PROCESSED_FILES_TRACKER, "w") as f:
        json.dump(list(processed_files), f)
    print("All snapshots saved successfully.")

def load_snapshot():
    """
    Load both metadata and document FAISS indexes and metadata from snapshots.
    """
    metadata_loaded = load_metadata_snapshot()
    if metadata_loaded:
        index_path = os.path.join(SNAPSHOT_DIR, "faiss_index.bin")
        metadata_path = os.path.join(SNAPSHOT_DIR, "document_metadata.json")
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            faiss_index.read(index_path)
            with open(metadata_path, "r") as f:
                global document_metadata
                document_metadata = json.load(f)
            print("Document FAISS index and metadata loaded successfully.")
        else:
            print("Document FAISS index or metadata not found.")
        load_processed_files()
        return True
    return False

# Fetch Top N Relevant Metadata Packages
def fetch_top_n_metadata_packages(query, n=TOP_N):
    """
    Given a query, fetch the top N relevant packages based on metadata embeddings.
    """
    query_embedding = embed_text(query)
    D, I = metadata_embeddings.search(query_embedding, n)
    top_indices = I[0]
    top_packages = [document_metadata[idx]['package_id'] for idx in top_indices if idx != -1]
    return top_packages

# Fetch Document Resources from Package Metadata
def extract_document_resources(package_metadata):
    """
    Extract document resources (URLs and file names) from package metadata.
    """
    resources = package_metadata.get('resources', [])
    documents = []
    for resource in resources:
        url = resource.get('url')
        resource_id = resource.get('id') or resource.get('name')  # Fallback if 'id' not present
        if url and resource_id:
            file_name = f"{package_metadata.get('id')}_{resource_id}.csv"
            documents.append({
                'url': url,
                'file_name': file_name
            })
    return documents

# Handle User Query
def handle_query(query):
    """
    Process a user query by fetching relevant documents and preparing them for the LLM.
    """
    if not metadata_embeddings or metadata_embeddings.ntotal == 0:
        print("Metadata FAISS index is empty. Building index...")
        build_metadata_index()

    print("Fetching top relevant packages based on metadata...")
    top_package_ids = fetch_top_n_metadata_packages(query, TOP_N)
    if not top_package_ids:
        print("No relevant packages found for the query.")
        return

    # Fetch and process documents from the top packages
    documents_to_process = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_pkg = {
            executor.submit(fetch_and_extract_documents, pkg_id): pkg_id for pkg_id in top_package_ids
        }
        for future in as_completed(future_to_pkg):
            pkg_id = future_to_pkg[future]
            pkg_documents = future.result()
            if pkg_documents:
                documents_to_process.extend(pkg_documents)

    if not documents_to_process:
        print("No documents found to process for the query.")
        return

    print(f"Downloading and processing {len(documents_to_process)} documents...")
    download_and_process_files(documents_to_process)

    # At this point, you can integrate with your LLM to generate responses
    # For example:
    # relevant_texts = [doc['text'] for doc in document_metadata if doc['file_name'] in relevant_files]
    # llm_response = generate_llm_response(relevant_texts, query)
    # print(llm_response)

    print(f"Processed top {TOP_N} packages and their documents for the query.")

def fetch_and_extract_documents(package_id):
    """
    Fetch package metadata and extract document resources.
    """
    metadata = fetch_package_metadata(package_id)
    if not metadata:
        return None
    return extract_document_resources(metadata)

# Build Metadata FAISS Index (If Not Already Built)
def build_metadata_index():
    """
    Fetch all package IDs and build the metadata FAISS index.
    """
    package_ids = fetch_all_package_ids()
    print(f"Fetched {len(package_ids)} package IDs. Building metadata embeddings...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_and_embed_metadata, pkg_id): pkg_id for pkg_id in package_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing metadata"):
            pkg_id = futures[future]
            result = future.result()
            if not result:
                print(f"Skipping package {pkg_id} due to failed metadata processing.")

    save_metadata_snapshot()
    print("Metadata FAISS index built successfully.")

# Main Function
def main():
    """
    Main function to initialize the system and handle user queries.
    """
    if not load_snapshot():
        print("No existing snapshot found. Building metadata index...")
        build_metadata_index()
        save_snapshot()

    print("System is ready to accept queries.")

    while True:
        query = input("Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() in ['exit', 'quit']:
            print("Exiting the system. Goodbye!")
            break
        if not query:
            print("Empty query. Please enter a valid query.")
            continue
        handle_query(query)

if __name__ == "__main__":
    load_processed_files()
    main()
