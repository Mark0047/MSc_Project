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


# Step 1: Downloading all required files
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


def download_all_files():
    """
    Downloads all files referenced in the packages to a local directory.
    """
    package_ids = fetch_local_package_ids()
    for package_id in tqdm(package_ids, desc="Downloading all files"):
        package_details = fetch_local_package_details(package_id)
        if not package_details:
            continue

        # Download resources in the package
        for resource in package_details.get('resources', []):
            if 'url' in resource and 'id' in resource:
                file_name = f"{package_id}_{resource['id']}.csv"
                download_file(resource['url'], file_name)


# Step 2: Process the downloaded files
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


def embed_text(text):
    """
    Generate embeddings for a given text using the best model.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE)
    if torch.cuda.is_available():
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


def process_document_embeddings():
    """
    Processes all downloaded files to build embeddings and the FAISS index.
    """
    for file_name in tqdm(os.listdir(TEMP_DOWNLOAD_DIR), desc="Processing documents"):
        file_path = os.path.join(TEMP_DOWNLOAD_DIR, file_name)
        if not os.path.isfile(file_path):
            continue

        # Extract text from file
        file_content = read_csv_file(file_path)
        if not file_content.strip():
            continue

        # Process chunks of text
        for chunk in process_text_chunks(file_content):
            embedding = embed_text(chunk)
            faiss_index.add(embedding)
            document_metadata.append({
                'file_name': file_name,
                'text': chunk
            })


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
    Save the FAISS index and document metadata to disk.
    """
    faiss.write_index(faiss_index, os.path.join(SNAPSHOT_DIR, "faiss_index.bin"))
    with open(os.path.join(SNAPSHOT_DIR, "document_metadata.json"), "w") as f:
        json.dump(document_metadata, f)
    print("Snapshot saved successfully.")


def load_snapshot():
    """
    Load the FAISS index and document metadata from a snapshot.
    """
    index_path = os.path.join(SNAPSHOT_DIR, "faiss_index.bin")
    metadata_path = os.path.join(SNAPSHOT_DIR, "document_metadata.json")
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        faiss_index.read(index_path)
        with open(metadata_path, "r") as f:
            global document_metadata
            document_metadata = json.load(f)
        print("Snapshot loaded successfully.")
        return True
    return False


def main():
    """
    Main function to handle downloading, embedding, and answering queries.
    """
    if not load_snapshot():
        print("Downloading all files...")
        download_all_files()

        print("Processing document embeddings...")
        process_document_embeddings()

        print("Saving snapshot...")
        save_snapshot()

    print("System is ready for queries!")


if __name__ == "__main__":
    main()
