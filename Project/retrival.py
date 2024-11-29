import requests
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from io import StringIO


embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')  


CKAN_PACKAGE_LIST_URL = "https://ckan.publishing.service.gov.uk/api/action/package_list"
CKAN_PACKAGE_SHOW_URL = "https://ckan.publishing.service.gov.uk/api/action/package_show?id="


vector_dimension = 384  
faiss_index = faiss.IndexFlatL2(vector_dimension)
document_metadata = []

def fetch_all_package_ids():
    
    response = requests.get(CKAN_PACKAGE_LIST_URL)
    if response.status_code == 200:
        return response.json().get('result', [])
    else:
        raise Exception(f"Error fetching package list: {response.status_code}")

def fetch_package_details(package_id):
    
    print(package_id)
    response = requests.get(CKAN_PACKAGE_SHOW_URL + package_id)
    print(response)
    if response.status_code == 200:
        return response.json().get('result', {})
    else:
        raise Exception(f"Error fetching package details for {package_id}: {response.status_code}")

def extract_csv_content(resource_url):
    
    try:
        response = requests.get(resource_url)
        if response.status_code == 200:
            csv_data = pd.read_csv(StringIO(response.text))
            return csv_data.to_string(index=False)  
        else:
            return ""
    except Exception as e:
        print(f"Error fetching CSV from {resource_url}: {e}")
        return ""

def process_document_embeddings():
    
    package_ids = fetch_all_package_ids()
    i = 0
    for package_id in package_ids:
        details = fetch_package_details(package_id)
        
        
        doc_text = details.get('title', '') + " " + details.get('notes', '')
        for resource in details.get('resources', []):
            if resource.get('format', '').lower() == 'csv':
                csv_content = extract_csv_content(resource['url'])
                doc_text += " " + csv_content
        
        
        if not doc_text.strip():
            continue

        
        embedding = embedding_model.encode(doc_text)
        faiss_index.add(np.array([embedding]).astype('float32'))
        document_metadata.append({
            'id': package_id,
            'title': details.get('title', ''),
            'description': details.get('notes', ''),
            'text': doc_text
        })
        i += 1
        if i > 10:
            break


def find_relevant_documents(question, top_k=5):
    
    question_embedding = embedding_model.encode(question)
    _, indices = faiss_index.search(np.array([question_embedding]).astype('float32'), top_k)
    return [document_metadata[i] for i in indices[0]]

def main():
    
    print("Building document embeddings and indexing...")
    process_document_embeddings()
    
    
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        
        print("Retrieving relevant documents...")
        relevant_docs = find_relevant_documents(question)
        
        
        print("\nTop Relevant Documents:")
        for doc in relevant_docs:
            print(f"ID: {doc['id']}\nTitle: {doc['title']}\nDescription: {doc['description']}\n")

if __name__ == "__main__":
    main()
