import os
import json
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

vector_dimension = 384  
faiss_index = faiss.IndexFlatL2(vector_dimension)
document_metadata = []

SNAPSHOT_DIR = "snapshot"
INDEX_FILE = os.path.join(SNAPSHOT_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(SNAPSHOT_DIR, "document_metadata.json")
DATA_FOLDER = "gov_data_json"


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
            # print(json.load(f).get('title', 'No title found'))
            return json.load(f).get('result', [])
    except Exception as e:
        print(f"Unexpected error for package ID {package_id}: {e}")
    return None


def extract_csv_content(resource_url):
    """
    Extract content from a CSV file, falling back gracefully if unavailable.
    """
    try:
        df = pd.read_csv(resource_url)
        return df.to_string(index=False)
    except Exception as e:
        print(f"Error reading CSV from {resource_url}: {e}")
        return ""


def process_package(package_id):
    """
    Process a single package: load metadata, resources, and compute embeddings.
    """
    try:
        details = fetch_local_package_details(package_id)
        doc_text = details.get('title', '') + " " + details.get('notes', '')


        for resource in details.get('resources', []):
            
            if resource.get('format', '').lower() == 'csv':
                print('found csv')
                csv_content = extract_csv_content(resource['url'])
                doc_text += " " + csv_content

        if doc_text.strip():
            embedding = embedding_model.encode(doc_text)
            faiss_index.add(np.array([embedding]).astype('float32'))
            document_metadata.append({
                'id': package_id,
                'title': details.get('title', ''),
                'description': details.get('notes', ''),
                'text': doc_text
            })
    except Exception as e:
        print(f"Error processing package {package_id}: {e}")


def process_document_embeddings():
    """
    Process all packages in parallel and build the FAISS index and metadata.
    """
    package_ids = fetch_local_package_ids()
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_package, package_ids)


def find_relevant_documents(question, top_k=5):
    """
    Search for the top-k relevant documents based on the user's question.
    """
    question_embedding = embedding_model.encode(question)
    _, indices = faiss_index.search(np.array([question_embedding]).astype('float32'), top_k)
    return [document_metadata[i] for i in indices[0]]


def main():
    """
    Main function to build embeddings or load a snapshot and answer questions.
    """
    try:
        if not load_snapshot():
            print("Building document embeddings and indexing...")
            process_document_embeddings()
            print('done with processing')
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
    except Exception as e:
        print('exception in main ', e)



if __name__ == "__main__":
    main()
