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




TEMP_DOWNLOAD_DIR = "./Project/temp_files"   
SNAPSHOT_DIR = "./Project/snapshot"          
CACHE_DIR = "./Project/cache"                


os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

if torch.cuda.is_available():
    embedding_model = embedding_model.to('cuda')


EMBEDDING_DIMENSION = 1024
METADATA_FAISS_INDEX_PATH = os.path.join(SNAPSHOT_DIR, "metadata_faiss.index")
DOCUMENT_FAISS_INDEX_PATH = os.path.join(SNAPSHOT_DIR, "document_faiss.index")


METADATA_SNAPSHOT_PATH = os.path.join(SNAPSHOT_DIR, "metadata_snapshot.json")
DOCUMENT_SNAPSHOT_PATH = os.path.join(SNAPSHOT_DIR, "document_snapshot.json")
PROCESSED_FILES_TRACKER = os.path.join(SNAPSHOT_DIR, "processed_files.json")


METADATA_CACHE_DIR = os.path.join(CACHE_DIR, "metadata_embeddings")
os.makedirs(METADATA_CACHE_DIR, exist_ok=True)


CHUNK_SIZE = 512  
TOP_K_METADATA = 10  
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

def save_json(data, path):
    """
    Save data to a JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f)

def load_json(path):
    """
    Load data from a JSON file.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None



def initialize_faiss_index(index_path, dimension):
    """
    Initialize or load a FAISS index.
    """
    if os.path.exists(index_path):
        print(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
    else:
        print("Creating a new FAISS index...")
        index = faiss.IndexFlatL2(dimension)
    return index

def save_faiss_index(index, index_path):
    """
    Save a FAISS index to disk.
    """
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}.")



def load_metadata_snapshot():
    """
    Load the metadata snapshot.
    """
    return load_json(METADATA_SNAPSHOT_PATH)

def load_processed_files():
    """
    Load the set of processed files.
    """
    data = load_json(PROCESSED_FILES_TRACKER)
    if data:
        return set(data)
    return set()

def load_metadata_faiss_index():
    """
    Load the metadata FAISS index.
    """
    if os.path.exists(METADATA_FAISS_INDEX_PATH):
        return faiss.read_index(METADATA_FAISS_INDEX_PATH)
    else:
        print("Metadata FAISS index not found. Please run the preprocessing script first.")
        exit(1)

def load_document_faiss_index():
    """
    Load the document FAISS index.
    """
    if os.path.exists(DOCUMENT_FAISS_INDEX_PATH):
        return faiss.read_index(DOCUMENT_FAISS_INDEX_PATH)
    else:
        print("Document FAISS index not found. Initializing a new one.")
        return faiss.IndexFlatL2(EMBEDDING_DIMENSION)



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

def process_document(file_path, file_name):
    """
    Process a single document: read, chunk, embed.
    Returns embeddings and metadata entries.
    """
    file_content = read_csv_file(file_path)
    if not file_content.strip():
        print(f"No content in {file_path}.")
        return None, None
    
    embeddings = []
    metadata_entries = []
    for chunk in process_text_chunks(file_content):
        embedding = embed_text(chunk)
        embeddings.append(embedding)
        metadata_entries.append({
            'file_name': file_name,
            'text': chunk
        })
    
    return np.vstack(embeddings), metadata_entries

def download_and_process_documents(documents, document_faiss, document_metadata, processed_files):
    """
    Downloads and processes a list of documents.
    Updates the FAISS index and document metadata.
    """
    new_embeddings = []
    new_metadata = []

    with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
        futures = {}
        for doc in documents:
            file_name = doc.get('file_name')
            url = doc.get('url')
            if not file_name or not url:
                continue
            if file_name in processed_files:
                continue  
            futures[executor.submit(download_file, url, file_name)] = doc

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading files"):
            doc = futures[future]
            file_name = doc.get('file_name')
            url = doc.get('url')
            file_path = future.result()
            if not file_path:
                print(f"Failed to download {url}.")
                continue  

            embeddings, metadata_entries = process_document(file_path, file_name)
            if embeddings is not None and metadata_entries is not None:
                document_faiss.add(embeddings)
                document_metadata.extend(metadata_entries)
                new_embeddings.append(embeddings)
                new_metadata.extend(metadata_entries)
                processed_files.add(file_name)
    
    return new_embeddings, new_metadata



def fetch_top_k_metadata(query_embedding, metadata_faiss, top_k):
    """
    Fetch the top K metadata entries based on the query embedding.
    """
    distances, indices = metadata_faiss.search(query_embedding, top_k)
    top_indices = indices[0]
    return top_indices

def handle_query(query, metadata_faiss, document_faiss, metadata_snapshot, document_metadata, processed_files):
    """
    Handle a user query by retrieving relevant documents and processing them.
    """
    print("Embedding the query...")
    query_emb = embed_text(query)

    print("Searching for top relevant metadata entries...")
    top_indices = fetch_top_k_metadata(query_emb, metadata_faiss, TOP_K_METADATA)
    
    
    relevant_metadata = [metadata_snapshot[idx] for idx in top_indices if idx < len(metadata_snapshot)]
    
    
    documents_to_fetch = []
    for meta in relevant_metadata:
        
        
        
        
        
        
        
        
        
        url = meta.get('url', '')  
        file_name = meta.get('file_name', '')
        if url and file_name and file_name not in processed_files:
            documents_to_fetch.append({'url': url, 'file_name': file_name})
    
    
    required_documents = TOP_K_METADATA - len(documents_to_fetch)
    if required_documents > 0:
        
        
        additional_top_k = TOP_K_METADATA * 2
        additional_indices = fetch_top_k_metadata(query_emb, metadata_faiss, additional_top_k)
        for idx in additional_indices[0]:
            if len(documents_to_fetch) >= TOP_K_METADATA:
                break
            meta = metadata_snapshot[idx]
            url = meta.get('url', '')  
            file_name = meta.get('file_name', '')
            if url and file_name and file_name not in processed_files and {'url': url, 'file_name': file_name} not in documents_to_fetch:
                documents_to_fetch.append({'url': url, 'file_name': file_name})
    
    if not documents_to_fetch:
        print("No new documents to fetch for this query.")
        return
    
    print(f"Fetching and processing {len(documents_to_fetch)} documents...")
    new_embeddings, new_metadata = download_and_process_documents(documents_to_fetch, document_faiss, document_metadata, processed_files)
    
    if new_embeddings:
        
        faiss.write_index(document_faiss, DOCUMENT_FAISS_INDEX_PATH)
        save_json(document_metadata, DOCUMENT_SNAPSHOT_PATH)
        save_json(list(processed_files), PROCESSED_FILES_TRACKER)
        print(f"Successfully processed {len(new_embeddings)} documents.")
    else:
        print("No new embeddings were added.")



def main():
    """
    Main function to initialize the system and handle user queries.
    """
    
    metadata_faiss = load_metadata_faiss_index()
    metadata_snapshot = load_metadata_snapshot()
    
    if metadata_snapshot is None:
        print("Metadata snapshot not found. Please run the preprocessing script first.")
        exit(1)
    
    
    document_faiss = load_document_faiss_index()
    document_metadata = load_json(DOCUMENT_SNAPSHOT_PATH) or []
    
    
    processed_files = load_processed_files()
    
    print("System is ready to accept queries.")
    
    while True:
        query = input("Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() in ['exit', 'quit']:
            print("Exiting the system. Goodbye!")
            break
        if not query:
            print("Empty query. Please enter a valid query.")
            continue
        handle_query(query, metadata_faiss, document_faiss, metadata_snapshot, document_metadata, processed_files)
        



if __name__ == "__main__":
    main()
