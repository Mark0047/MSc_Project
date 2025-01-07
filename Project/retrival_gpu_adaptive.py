import os
import json
import requests
import pandas as pd
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch

# Directories for file handling
DATA_FOLDER = "./Project/gov_data_json"
TEMP_DOWNLOAD_DIR = "./Project/temp_files"
SNAPSHOT_DIR = "./Project/snapshot"

os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Embedding model details
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

if torch.cuda.is_available():
    embedding_model = embedding_model.to('cuda')

vector_dimension = 1024
faiss_index = faiss.IndexFlatL2(vector_dimension)
document_metadata = []

CHUNK_SIZE = 512  # For text chunking
TOP_N = 10        # Number of top relevant documents to fetch per query

# File to track processed documents
PROCESSED_FILES_TRACKER = os.path.join(SNAPSHOT_DIR, "processed_files.json")
if os.path.exists(PROCESSED_FILES_TRACKER):
    with open(PROCESSED_FILES_TRACKER, "r") as f:
        processed_files = set(json.load(f))
else:
    processed_files = set()

# Step 1: Load Metadata and Initialize FAISS Index for Metadata
def load_metadata():
    """
    Loads metadata from all JSON files and builds an initial FAISS index based on metadata.
    """
    global document_metadata
    print("Loading metadata...")
    for package_id in tqdm(fetch_local_package_ids(), desc="Loading metadata"):
        package_details = fetch_local_package_details(package_id)
        if not package_details:
            continue

        for resource in package_details.get('resources', []):
            if 'url' in resource and 'id' in resource:
                metadata_text = json.dumps(resource)  # Convert metadata to string
                embedding = embed_text(metadata_text)
                faiss_index.add(embedding)
                document_metadata.append({
                    'package_id': package_id,
                    'resource_id': resource['id'],
                    'url': resource['url'],
                    'file_name': f"{package_id}_{resource['id']}.csv",
                    'text': metadata_text
                })

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

# Step 2: Fetch Top N Relevant Documents Based on Metadata
def fetch_top_n_documents(query, n=TOP_N):
    """
    Given a query, fetch the top N relevant documents based on metadata embeddings.
    """
    query_embedding = embed_text(query)
    D, I = faiss_index.search(query_embedding, n)
    top_indices = I[0]
    top_documents = [document_metadata[idx] for idx in top_indices if idx != -1]
    return top_documents

# Step 3: Download and Process Specific Files
def download_and_process_files(documents):
    """
    Downloads and processes the specified documents, updating the FAISS index accordingly.
    """
    new_embeddings = []
    new_metadata = []
    for doc in tqdm(documents, desc="Downloading and processing files"):
        file_name = doc['file_name']
        if file_name in processed_files:
            continue  # Skip already processed files

        file_path = download_file(doc['url'], file_name)
        if not file_path:
            print(f"Failed to download {doc['url']}")
            continue  # Skip if download failed

        file_content = read_csv_file(file_path)
        if not file_content.strip():
            print(f"No content in {file_path}")
            continue

        # Process chunks of text
        for chunk in process_text_chunks(file_content):
            embedding = embed_text(chunk)
            new_embeddings.append(embedding)
            new_metadata.append({
                'file_name': file_name,
                'text': chunk
            })

        processed_files.add(file_name)  # Mark as processed

    if new_embeddings:
        faiss_index.add(np.vstack(new_embeddings))
        document_metadata.extend(new_metadata)
        save_snapshot()

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

# Step 4: Download a Single File with Error Handling
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

# Utility functions
def fetch_local_package_ids():
    """
    Load all JSON file names in the local data folder as package IDs.
    """
    return [f[:-5] for f in os.listdir(DATA_FOLDER) if f.endswith(".json")]

def fetch_local_package_details(package_id):
    """
    Load the details of a package from a local JSON file.
    """
    package_file = os.path.join(DATA_FOLDER, f"{package_id}.json")
    try:
        with open(package_file, "r") as f:
            return json.load(f).get('result', [])
    except Exception as e:
        print(f"Error loading package {package_id}: {e}")
    return None

def save_snapshot():
    """
    Save the FAISS index, document metadata, and processed files tracker to disk.
    """
    faiss.write_index(faiss_index, os.path.join(SNAPSHOT_DIR, "faiss_index.bin"))
    with open(os.path.join(SNAPSHOT_DIR, "document_metadata.json"), "w") as f:
        json.dump(document_metadata, f)
    with open(PROCESSED_FILES_TRACKER, "w") as f:
        json.dump(list(processed_files), f)
    print("Snapshot saved successfully.")

def load_snapshot():
    """
    Load the FAISS index, document metadata, and processed files tracker from a snapshot.
    """
    index_path = os.path.join(SNAPSHOT_DIR, "faiss_index.bin")
    metadata_path = os.path.join(SNAPSHOT_DIR, "document_metadata.json")
    tracker_path = PROCESSED_FILES_TRACKER
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        faiss_index.read(index_path)
        with open(metadata_path, "r") as f:
            global document_metadata
            document_metadata = json.load(f)
        if os.path.exists(tracker_path):
            with open(tracker_path, "r") as f:
                global processed_files
                processed_files = set(json.load(f))
        print("Snapshot loaded successfully.")
        return True
    return False

# Step 5: Handling Queries
def handle_query(query):
    """
    Process a user query by fetching relevant documents and returning responses.
    """
    top_docs = fetch_top_n_documents(query)
    if not top_docs:
        print("No relevant documents found in metadata.")
        return

    # Attempt to download and process top documents
    download_and_process_files(top_docs)

    # After processing, you can perform your LLM operations here
    # For example, retrieve relevant texts from document_metadata and pass to LLM
    # This part depends on your specific LLM integration

    print(f"Processed top {TOP_N} documents for the query.")

# Step 6: Main Function
def main():
    """
    Main function to initialize the system and handle user queries.
    """
    if not load_snapshot():
        # If no snapshot exists, load metadata and build initial index
        load_metadata()
        save_snapshot()

    print("System is ready for queries!")

    # Example loop to handle user queries
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            break
        handle_query(query)

if __name__ == "__main__":
    main()
