import os
import json
import pandas as pd
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch

# Load the best embedding model dynamically
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"  # Example of a state-of-the-art embedding model
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

if torch.cuda.is_available():
    embedding_model = embedding_model.to('cuda')

vector_dimension = 1024  # Update based on the embedding model dimensions
faiss_index = faiss.IndexFlatL2(vector_dimension)
document_metadata = []

SNAPSHOT_DIR = "snapshot"
INDEX_FILE = os.path.join(SNAPSHOT_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(SNAPSHOT_DIR, "document_metadata.json")
DATA_FOLDER = "gov_data_json"
CHUNK_SIZE = 512  # Optimize for token limits in LLMs

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

def load_snapshot():
    """
    Load the FAISS index and document metadata from a snapshot.
    """
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        faiss_index.read(INDEX_FILE)
        with open(METADATA_FILE, "r") as f:
            global document_metadata
            document_metadata = json.load(f)
        print("Snapshot loaded successfully.")
        return True
    return False

def save_snapshot():
    """
    Save the FAISS index and document metadata to disk.
    """
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    faiss.write_index(faiss_index, INDEX_FILE)
    with open(METADATA_FILE, "w") as f:
        json.dump(document_metadata, f)
    print("Snapshot saved successfully.")

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

def process_text_chunks(text):
    """
    Split a large text into chunks for efficient embedding.
    """
    words = text.split()
    for i in range(0, len(words), CHUNK_SIZE):
        yield " ".join(words[i:i + CHUNK_SIZE])

def process_package(package_id):
    """
    Process a single package: load metadata, resources, and compute embeddings.
    """
    try:
        details = fetch_local_package_details(package_id)
        full_text = details.get('title', '') + " " + details.get('notes', '')

        for resource in details.get('resources', []):
            if resource.get('format', '').lower() == 'csv':
                csv_url = resource.get('url', '')
                try:
                    df = pd.read_csv(csv_url)
                    full_text += "\n" + df.to_string(index=False)
                except Exception as e:
                    print(f"Error reading CSV from {csv_url}: {e}")

        # Process text chunks
        for chunk in process_text_chunks(full_text):
            embedding = embed_text(chunk)
            faiss_index.add(embedding)
            document_metadata.append({
                'id': package_id,
                'title': details.get('title', ''),
                'description': details.get('notes', ''),
                'text': chunk
            })
    except Exception as e:
        print(f"Error processing package {package_id}: {e}")

def process_document_embeddings():
    """
    Process all packages in parallel and build the FAISS index and metadata.
    """
    package_ids = fetch_local_package_ids()
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(process_package, package_ids), total=len(package_ids)))

def find_relevant_documents(question, top_k=5):
    """
    Search for the top-k relevant documents based on the user's question.
    """
    question_embedding = embed_text(question)
    _, indices = faiss_index.search(question_embedding, top_k)
    return [document_metadata[i] for i in indices[0]]

def answer_question_with_rag(question, relevant_docs):
    """
    Placeholder for integrating a Retrieval-Augmented Generation (RAG) pipeline.
    """
    context = " ".join([doc['text'] for doc in relevant_docs])
    print(f"LLM Placeholder:\nQuestion: {question}\nContext: {context[:500]}...")

def main():
    """
    Main function to build embeddings or load a snapshot and answer questions.
    """
    if not load_snapshot():
        print("Building document embeddings and indexing...")
        process_document_embeddings()
        save_snapshot()

    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        print("Retrieving relevant documents...")
        relevant_docs = find_relevant_documents(question)

        print("\nTop Relevant Documents:")
        for doc in relevant_docs:
            print(f"ID: {doc['id']}\nTitle: {doc['title']}\nDescription: {doc['description']}\n")

        answer_question_with_rag(question, relevant_docs)

if __name__ == "__main__":
    main()
