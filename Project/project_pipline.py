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
import torch




import struct 
print(struct.calcsize("P") * 8)

# Ensure CUDA is available
# assert torch.cuda.is_available(), "CUDA is not available. Check your CUDA installation."


# ---------------------------------------------------------------------
# SET YOUR GLOBALS (paths, embedding model, etc.) 
# ---------------------------------------------------------------------
DATA_FOLDER = "./gov_data_json"
TEMP_DOWNLOAD_DIR = "./temp_files"
SNAPSHOT_DIR = "./snapshot"
CACHE_DIR = "./cache"

os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

if torch.cuda.is_available():
    print("Running on GPU")
    embedding_model = embedding_model.to("cuda")
else:
    print('Check CUDA')

EMBEDDING_DIMENSION = 1024
metadata_faiss_index_path = os.path.join(SNAPSHOT_DIR, "metadata_faiss.index")
document_faiss_index_path = os.path.join(SNAPSHOT_DIR, "document_faiss.index")

metadata_snapshot_path = os.path.join(SNAPSHOT_DIR, "metadata_snapshot.json")
document_snapshot_path = os.path.join(SNAPSHOT_DIR, "document_snapshot.json")
processed_files_tracker = os.path.join(SNAPSHOT_DIR, "processed_files.json")

metadata_cache_dir = os.path.join(CACHE_DIR, "metadata_embeddings")
os.makedirs(metadata_cache_dir, exist_ok=True)

CHUNK_SIZE = 512
TOP_K_METADATA = 10
THREAD_POOL_WORKERS = 5




def embed_text(text):
    """
    Generate embeddings for a given text using the embedding model.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE)
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    # We'll average the sequence dimension
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


def get_cache_path(identifier, cache_type="metadata"):
    """
    Generate a cache file path based on a unique identifier.
    """
    hash_id = hashlib.md5(identifier.encode()).hexdigest()
    if cache_type == "metadata":
        return os.path.join(metadata_cache_dir, f"{hash_id}.npy")
    return None

def cache_metadata_embedding(package_id, embedding):
    """
    Cache the metadata embedding.
    """
    cache_path = get_cache_path(package_id, "metadata")
    if cache_path:
        np.save(cache_path, embedding)

def load_cached_metadata_embedding(package_id):
    """
    Load cached metadata embedding if available.
    """
    cache_path = get_cache_path(package_id, "metadata")
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    return None

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        return df.to_string(index=False)
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return ""

def process_text_chunks(text):
    words = text.split()
    for i in range(0, len(words), CHUNK_SIZE):
        yield " ".join(words[i:i + CHUNK_SIZE])

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# ... plus your caching logic, etc.
def initialize_faiss_index(index_path, dimension):
    if os.path.exists(index_path):
        print(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
    else:
        print("Creating a new FAISS index...")
        index = faiss.IndexFlatL2(dimension)
    return index

def save_faiss_index(index, index_path):
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}.")




def load_metadata_files():
    """
    Load all local metadata JSON files.
    """
    metadata_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".json")]
    return metadata_files

def process_metadata_file(file_name):
    """
    Process a single metadata JSON file and return its embedding and relevant info.
    """
    file_path = os.path.join(DATA_FOLDER, file_name)
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
        
        cached_embedding = load_cached_metadata_embedding(package_id)
        if cached_embedding is not None:
            embedding = cached_embedding
        else:
            embedding = embed_text(metadata_text)
            cache_metadata_embedding(package_id, embedding)
        
        
        metadata_entry = {
            'package_id': package_id,
            'file_name': file_name,
            'url': result.get('url', ''),  
            'relevant_fields': relevant_fields
        }
        
        return embedding, metadata_entry
    except Exception as e:
        print(f"Error processing metadata file {file_name}: {e}")
        return None

def build_metadata_faiss_index(metadata_faiss, document_metadata, max_files=3000):
    """
    Build the metadata FAISS index from local metadata files, processing only up to `max_files`.
    """
    # Load all metadata files
    metadata_files = load_metadata_files()
    
    # Limit to `max_files` unless user provides a specific number
    if max_files and max_files > 0:
        metadata_files = metadata_files[:max_files]
    
    print(f"Found {len(metadata_files)} metadata files. Processing up to {max_files}...")


    with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
        futures = {executor.submit(process_metadata_file, file_name): file_name for file_name in metadata_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building metadata FAISS index"):
            result = future.result()
            if result:
                embedding, metadata_entry = result
                metadata_faiss.add(embedding)
                document_metadata.append(metadata_entry)
    
    print("Metadata FAISS index built successfully.")



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
    Read CSV content, split into chunks, and embed each chunk.
    Returns (embeddings, metadata_entries).
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
            "file_name": file_name,
            "text": chunk
        })

    return np.vstack(embeddings), metadata_entries

def download_and_process_documents(documents, faiss_index, document_metadata, processed_files):
    """
    Downloads and processes a list of documents (CSV). Updates FAISS + metadata.
    Returns (new_embeddings, new_metadata) or ([], []) if none.
    """
    new_embeddings = []
    new_metadata = []

    with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
        futures = {}
        for doc in documents:
            file_name = doc.get("file_name")
            url = doc.get("url")
            if not file_name or not url:
                continue
            if file_name in processed_files:
                continue
            # Download file asynchronously
            futures[executor.submit(download_file, url, file_name)] = doc

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading files"):
            doc = futures[future]
            file_name = doc.get("file_name")
            url = doc.get("url")
            file_path = future.result()
            if not file_path:
                print(f"Failed to download {url}.")
                continue

            # Process the file -> chunk + embed
            embeddings, metadata_entries = process_document(file_path, file_name)
            if embeddings is not None and metadata_entries is not None:
                faiss_index.add(embeddings)            # Add to FAISS
                document_metadata.extend(metadata_entries)  # Keep track
                new_embeddings.append(embeddings)
                new_metadata.extend(metadata_entries)
                processed_files.add(file_name)

    return new_embeddings, new_metadata




def fetch_top_k_metadata(query_embedding, metadata_faiss, top_k):
    distances, indices = metadata_faiss.search(query_embedding, top_k)
    return indices

def fetch_top_k_documents(query_embedding, document_faiss, k):
    distances, indices = document_faiss.search(query_embedding, k)
    return distances, indices

def handle_query(
    query: str,
    metadata_faiss,
    document_faiss,
    document_metadata,
    processed_files,
    top_k_metadata=10,
    top_k_docs=3
):
    """
    1) Embed query
    2) Retrieve top-K metadata from metadata_faiss
    3) Download & embed any missing documents
    4) Finally, retrieve top-K doc chunks from document_faiss
    5) Return list of chunk texts
    """
    # 1. Embed query
    query_emb = embed_text(query)

    # 2. Metadata search
    top_indices = fetch_top_k_metadata(query_emb, metadata_faiss, top_k_metadata)
    
    metadata_snapshot = load_json(metadata_snapshot_path)
    if metadata_snapshot is None:
        print("Metadata snapshot not found. Ensure metadata FAISS index is built.")
        return []

    # Identify relevant metadata
    relevant_metadata = [
        metadata_snapshot[idx]
        for idx in top_indices[0]
        if idx < len(metadata_snapshot)
    ]
    
    # print('relevant_metadata--------------------',relevant_metadata)

    # Build list of documents to fetch if not processed
    documents_to_fetch = []
    for meta in relevant_metadata:
        url = meta.get("url")
        file_name = meta.get("file_name")
        if url and file_name and file_name not in processed_files:
            documents_to_fetch.append({"url": url, "file_name": file_name})

    # Possibly expand if we want more than `top_k_metadata`
    # (Your existing approach might do an additional pass if you didn't get enough docs)
    if len(documents_to_fetch) < top_k_metadata:
        top_k_metadata - len(documents_to_fetch)
        # e.g., fetch more from metadata_faiss again, etc.
        # omitted for brevity, but you can replicate your logic

    # 3. Download & embed new documents
    if documents_to_fetch:
        new_embeddings, new_metadata = download_and_process_documents(
            documents_to_fetch,
            document_faiss,
            document_metadata,
            processed_files
        )
        if new_embeddings:
            # Save the updated doc faiss and doc metadata
            save_faiss_index(document_faiss, document_faiss_index_path)
            save_json(document_metadata, document_snapshot_path)
            save_json(list(processed_files), processed_files_tracker)

    # 4. Now that document_faiss might have new data, let's do a final search 
    #    to get top document chunks relevant to the user query.
    distances, indices = document_faiss.search(query_emb, top_k_docs)

    top_chunks = []
    if len(indices) > 0:
        for idx_list, dist_list in zip(indices, distances):
            for i, dist in zip(idx_list, dist_list):
                if i > 0 and i < len(document_metadata):
                    # print('document_metadata[i]---------------', document_metadata)
                    text_chunk = document_metadata[i]["text"]
                    top_chunks.append((text_chunk, dist))

    # sort by ascending distance
    top_chunks.sort(key=lambda x: x[1])

    # Return just the chunk texts (or keep the scores if you want)
    return [chunk for chunk, _ in top_chunks]



def save_snapshot(metadata_faiss, document_faiss, metadata_snapshot, document_metadata, processed_files):
    """
    Save FAISS indexes and metadata to disk.
    """
    save_faiss_index(metadata_faiss, metadata_faiss_index_path)
    save_faiss_index(document_faiss, document_faiss_index_path)
    save_json(metadata_snapshot, metadata_snapshot_path)
    save_json(document_metadata, document_snapshot_path)
    save_json(list(processed_files), processed_files_tracker)
    print("All snapshots saved successfully.")

def load_snapshot(metadata_faiss, document_faiss, metadata_snapshot, document_metadata, processed_files):
    """
    Load FAISS indexes and metadata from disk.
    """
    if os.path.exists(metadata_faiss_index_path) and os.path.exists(metadata_snapshot_path):
        metadata_faiss = faiss.read_index(metadata_faiss_index_path)
        metadata_snapshot = load_json(metadata_snapshot_path)
    else:
        metadata_faiss = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        metadata_snapshot = []
    
    if os.path.exists(document_faiss_index_path) and os.path.exists(document_snapshot_path):
        document_faiss = faiss.read_index(document_faiss_index_path)
        document_metadata = load_json(document_snapshot_path)
    else:
        document_faiss = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        document_metadata = []
    
    if os.path.exists(processed_files_tracker):
        processed_files = set(load_json(processed_files_tracker))
    else:
        processed_files = set()
    
    return metadata_faiss, document_faiss, metadata_snapshot, document_metadata, processed_files


# main.py (or wherever your main logic is):
from llm_interface import MultiLLM, ChatGPTLLM, HuggingFaceLLM, GeminiLLM

def rag_query_all_llms(
    query: str,
    metadata_faiss,
    document_faiss,
    document_metadata,
    processed_files,
    multi_llm: MultiLLM,      # <-- Instead of a single LLMInterface, we pass in the MultiLLM
    top_k_metadata=10,
    top_k_docs=3
):
    """
    1) Use handle_query(...) to retrieve top-k doc chunks relevant to the user query
    2) Pass the same query + context to ALL LLMs in multi_llm
    3) Return dict of {model_name: answer}
    """
    top_chunks = handle_query(
        query=query,
        metadata_faiss=metadata_faiss,
        document_faiss=document_faiss,
        document_metadata=document_metadata,
        processed_files=processed_files,
        top_k_metadata=top_k_metadata,
        top_k_docs=top_k_docs
    )

    # Combine into one big context
    context = "\n\n".join(top_chunks)
    
    # Ask all LLMs
    answers = multi_llm.get_all_responses(query, context)
    return answers


def main():
    # 0) Initialize or Load existing snapshots
    metadata_faiss = initialize_faiss_index(metadata_faiss_index_path, EMBEDDING_DIMENSION)
    document_faiss = initialize_faiss_index(document_faiss_index_path, EMBEDDING_DIMENSION)

    metadata_snapshot = load_json(metadata_snapshot_path)
    document_metadata = load_json(document_snapshot_path) or []
    if not metadata_snapshot:
        metadata_snapshot = []
    processed_files = set(load_json(processed_files_tracker) or [])

    # If no metadata yet, build the metadata index
    if len(metadata_snapshot) == 0:
        print("Building metadata FAISS index from local JSON files...")
        build_metadata_faiss_index(metadata_faiss, metadata_snapshot)
        save_faiss_index(metadata_faiss, metadata_faiss_index_path)
        save_json(metadata_snapshot, metadata_snapshot_path)
        print("Done building metadata index.")
    else:
        print("Metadata index loaded with", len(metadata_snapshot), "items.")

    if not document_metadata:
        print("No document metadata found, starting empty.")
    else:
        print("Loaded", len(document_metadata), "document metadata entries.")

    if not processed_files:
        print("No processed files found, starting fresh.")
    else:
        print("Already processed", len(processed_files), "files.")

    print("System ready for queries.")
    
    # (2) Create an instance of MultiLLM, adding whatever LLMs you want:
    multi_llm = MultiLLM([
        HuggingFaceLLM(
            model_name="google/flan-t5-large", 
            task="text2text-generation",
            device=0  # Use GPU if available; set to -1 for CPU
        ),
        # HuggingFaceLLM(
        #     model_name="tiiuae/falcon-7b-instruct",
        #     task="text2text-generation",
        #     device=0
        # ),
        # # HuggingFaceLLM(
        # #     model_name="meta-llama/Llama-2-7b-chat-hf",
        # #     task="text-generation",  # Using text-generation for chat model
        # #     device=0
        # # ),
        # HuggingFaceLLM(
        #     model_name="TinyPixel/Llama-2-7B-bf16-sharded",
        #     task="text-generation",  # Using text-generation for chat model
        #     device=0
        # ),
        # # Add more HuggingFaceLLM instances here if you want to include additional models.
        # HuggingFaceLLM(
        #     model_name="google/flan-t5-xxl",  # or "tiiuae/falcon-7b-instruct"
        #     task="text2text-generation",
        #     device=0  # or 0 if you have GPU
        # ),
        # HuggingFaceLLM(
        #     model_name="EleutherAI/gpt-j-6b",  
        #     task="text2text-generation",
        #     device=0  # or 0 if you have GPU
        # ),
        GeminiLLM(
            api_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash",
            api_key= os.environ.get("GEMINI_API_KEY")
            #  export GEMINI_API_KEY=''
        )
    ])

    print("System ready for queries about UK Open Government Data.")

    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        
        if not query:
            print("Empty query. Try again.")
            continue

        # (3) RAG flow for multiple LLMs
        all_answers = rag_query_all_llms(
            query=query,
            metadata_faiss=metadata_faiss,
            document_faiss=document_faiss,
            document_metadata=document_metadata,
            processed_files=processed_files,
            multi_llm=multi_llm,
            top_k_metadata=10,
            top_k_docs=3
        )

        # (4) Print out the answers from each model
        print("\n=== ANSWERS FROM ALL LLMs ===")
        for llm_name, answer in all_answers.items():
            print(f"[{llm_name}]: {answer}\n")
        print("============================================\n")
   

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an optional argument for max_files
    parser.add_argument(
        "--max_files", 
        type=int, 
        default=3000, 
        help="Maximum number of metadata files to process"
    )
    args = parser.parse_args()
    main()