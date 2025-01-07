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




DATA_FOLDER = "./Project/gov_data_json"       
SNAPSHOT_DIR = "./Project/snapshot"          
CACHE_DIR = "./Project/cache"                


os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

if torch.cuda.is_available():
    embedding_model = embedding_model.to('cuda')


EMBEDDING_DIMENSION = 1024
METADATA_FAISS_INDEX_PATH = os.path.join(SNAPSHOT_DIR, "metadata_faiss.index")
METADATA_SNAPSHOT_PATH = os.path.join(SNAPSHOT_DIR, "metadata_snapshot.json")


METADATA_CACHE_DIR = os.path.join(CACHE_DIR, "metadata_embeddings")
os.makedirs(METADATA_CACHE_DIR, exist_ok=True)


CHUNK_SIZE = 512  
THREAD_POOL_WORKERS = 8  



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

def get_cache_path(package_id):
    """
    Generate a cache file path based on a unique package ID.
    """
    hash_id = hashlib.md5(package_id.encode()).hexdigest()
    return os.path.join(METADATA_CACHE_DIR, f"{hash_id}.npy")

def cache_metadata_embedding(package_id, embedding):
    """
    Cache the metadata embedding.
    """
    cache_path = get_cache_path(package_id)
    if cache_path:
        np.save(cache_path, embedding)

def load_cached_metadata_embedding(package_id):
    """
    Load cached metadata embedding if available.
    """
    cache_path = get_cache_path(package_id)
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return None

def read_metadata_file(file_path):
    """
    Reads a metadata JSON file and extracts relevant text.
    """
    try:
        with open(file_path, "r") as f:
            metadata = json.load(f)
        result = metadata.get('result', {})
        
        
        relevant_fields = [
            result.get('title', ''),
            result.get('notes', ''),
            result.get('description', ''),
            " ".join([tag.get('name', '') for tag in result.get('tags', [])])
        ]
        metadata_text = " ".join([str(field) for field in relevant_fields if field])
        
        if not metadata_text.strip():
            return None  
        
        package_id = result.get('id')
        if not package_id:
            return None  
        
        return package_id, metadata_text
    except Exception as e:
        print(f"Error reading metadata file {file_path}: {e}")
        return None

def process_metadata_file(file_name):
    """
    Process a single metadata JSON file and return its embedding and relevant info.
    """
    file_path = os.path.join(DATA_FOLDER, file_name)
    data = read_metadata_file(file_path)
    if not data:
        return None
    package_id, metadata_text = data
    
    
    cached_embedding = load_cached_metadata_embedding(package_id)
    if cached_embedding is not None:
        embedding = cached_embedding
    else:
        embedding = embed_text(metadata_text)
        cache_metadata_embedding(package_id, embedding)
    
    
    metadata_entry = {
        'package_id': package_id,
        'file_name': file_name,
        'metadata_text': metadata_text
    }
    
    return embedding, metadata_entry

def build_metadata_faiss_index():
    """
    Build the metadata FAISS index from local metadata files.
    """
    metadata_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".json")]
    print(f"Found {len(metadata_files)} metadata files. Processing...")
    
    
    metadata_faiss = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    metadata_snapshot = []
    
    with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
        futures = {executor.submit(process_metadata_file, file_name): file_name for file_name in metadata_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building metadata FAISS index"):
            result = future.result()
            if result:
                embedding, metadata_entry = result
                metadata_faiss.add(embedding)
                metadata_snapshot.append(metadata_entry)
    
    print("Metadata FAISS index built successfully.")
    
    
    faiss.write_index(metadata_faiss, METADATA_FAISS_INDEX_PATH)
    with open(METADATA_SNAPSHOT_PATH, "w") as f:
        json.dump(metadata_snapshot, f)
    print(f"Metadata FAISS index and snapshot saved to {SNAPSHOT_DIR}.")

def update_metadata_faiss_index():
    """
    Update the metadata FAISS index with new or modified metadata files.
    """
    
    if os.path.exists(METADATA_FAISS_INDEX_PATH) and os.path.exists(METADATA_SNAPSHOT_PATH):
        metadata_faiss = faiss.read_index(METADATA_FAISS_INDEX_PATH)
        with open(METADATA_SNAPSHOT_PATH, "r") as f:
            metadata_snapshot = json.load(f)
    else:
        print("Metadata FAISS index or snapshot not found. Building from scratch.")
        build_metadata_faiss_index()
        return
    
    existing_package_ids = set(entry['package_id'] for entry in metadata_snapshot)
    
    metadata_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".json")]
    new_metadata_files = [f for f in metadata_files if f[:-5] not in existing_package_ids]
    
    if not new_metadata_files:
        print("No new metadata files to update.")
        return
    
    print(f"Found {len(new_metadata_files)} new metadata files. Updating FAISS index...")
    
    with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
        futures = {executor.submit(process_metadata_file, file_name): file_name for file_name in new_metadata_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Updating metadata FAISS index"):
            result = future.result()
            if result:
                embedding, metadata_entry = result
                metadata_faiss.add(embedding)
                metadata_snapshot.append(metadata_entry)
    
    
    faiss.write_index(metadata_faiss, METADATA_FAISS_INDEX_PATH)
    with open(METADATA_SNAPSHOT_PATH, "w") as f:
        json.dump(metadata_snapshot, f)
    print(f"Metadata FAISS index and snapshot updated with new entries.")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess metadata to build/update FAISS index.")
    parser.add_argument('--update', action='store_true', help='Update existing FAISS index with new metadata files.')
    args = parser.parse_args()

    if args.update:
        update_metadata_faiss_index()
    else:
        build_metadata_faiss_index()
