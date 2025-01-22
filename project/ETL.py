import numpy as np
from clearml import Task
from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
import json
from apify_client import ApifyClient
# from git import repo
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List,Dict
from sentence_transformers import CrossEncoder
import gradio as gr
from clearml import PipelineDecorator, PipelineController

import clearml
from clearml import Task
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from typing import List
import uuid
import re
import unicodedata
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import requests
import openai
import subprocess


# Run ETL 
def run_pipeline():
    """
    Run the ETL pipeline to search and store GitHub repositories and YouTube videos.
    """
    GITHUB_TOKEN = "your_token"
    GITHUB_HEADERS = {
        "Authorization": f"token {GITHUB_TOKEN}"
    }

    APIFY_TOKEN = 'your_token'  
    apify_client = ApifyClient(APIFY_TOKEN)

    # Queries related to ROS2 and its subdomains
    queries = [
        "ros2",
        "ros2 middleware",
        "ros2 navigation",
        "ros2 motion planning",
        "ros2 simulation"
    ]
    
    # MongoDB setup
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ros2_database_tmp"]
    gt_name = "github_data_new"
    github_collection = db[gt_name]
    med_name = 'medium_data'
    medium_collection = db[med_name]
    li_name = 'linkedin_data'
    linkedin_collection = db[li_name]

    max_results=5
    for query in queries:
        url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page={max_results}"  
        response = requests.get(url, headers=GITHUB_HEADERS)  
        
        if response.status_code == 200:  
            repos = response.json()["items"]  
            for repo in repos:  
                repo_data = {  
                    "repo_name": repo["name"],  
                    "repo_url": repo["html_url"],  
                    "description": repo.get("description", ""),  
                    "stars": repo["stargazers_count"],  
                    "files": []  # Initialize a list to store file information  
                }  
                
                # Fetch file contents from the repository  
                contents_url = f"https://api.github.com/repos/{repo['full_name']}/contents"  
                contents_response = requests.get(contents_url, headers=GITHUB_HEADERS)  
                
                if contents_response.status_code == 200:  
                    files = contents_response.json()  
                    for file in files:  
                        if file['type'] == 'file':  # Only process files, not directories  
                            file_data = {  
                                "file_path": file["path"],  
                                "file_type": file["name"].split('.')[-1] if '.' in file["name"] else 'unknown',  
                                "file_content": ""  # Placeholder for file content  
                            }  
                            
                            # Ensure there is a valid download_url  
                            if 'download_url' in file and file["download_url"]:  
                                file_content_response = requests.get(file["download_url"], headers=GITHUB_HEADERS)  
                                if file_content_response.status_code == 200:  
                                    file_data["file_content"] = file_content_response.text  
                                else:  
                                    print(f"Error fetching content for {file['path']}: {file_content_response.status_code}")  
                            else:  
                                print(f"No download URL found for {file['path']}. Skipping...")  
                            
                            repo_data["files"].append(file_data)  
                
                github_collection.insert_one(repo_data)  
                print(f"GitHub repo '{repo['name']}' added to database with {len(repo_data['files'])} files.")  
                
        else:  
            print(f"GitHub API Error: {response.status_code}, {response.text}")  

    url = "https://medium2.p.rapidapi.com/search/articles"

    querystring = {"query":"ros2"}

    headers = {
        "x-rapidapi-key": "your_key",
        "x-rapidapi-host": "medium2.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    content_response = []
    NUM_ARTICLES = 100
    for i in range(NUM_ARTICLES):

        url = f"https://medium2.p.rapidapi.com/article/{response.json()['articles'][i]}/content"

        headers = {
        "x-rapidapi-key": "your_key",
        "x-rapidapi-host": "medium2.p.rapidapi.com"
        }

        content_response.append(requests.get(url, headers=headers))
        print(url)
    for item in content_response:
        medium_collection.insert_one(item.json())

    def sanitize_document(doc):
        """
        Ensure document is MongoDB-compatible and JSON serializable.
        """
        for key, value in list(doc.items()):
            if isinstance(value, (tuple, set)):
                doc[key] = list(value)
            elif value is None:
                doc[key] = None
            elif isinstance(value, dict):
                doc[key] = sanitize_document(value)
            elif isinstance(value, str):
                doc[key] = value[:50000]
        return doc
    
    url = "https://fresh-linkedin-profile-data.p.rapidapi.com/search-posts"

    payload = {
        "search_keywords": "ros2",
        "sort_by": "Top match",
        "page": 5
    }
    headers = {
        "x-rapidapi-key": "your_key",
        "x-rapidapi-host": "fresh-linkedin-profile-data.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        try:
            for item in response.json()['data']:
                sanitized_item = sanitize_document(item)
                linkedin_collection.insert_one(sanitized_item)
                print(f"Inserted document: {url}")
        except Exception as e:
            print(f"Error processing or inserting document: {e}")
    else:
        print(f"API request failed with status code {response.status_code}")

    return gt_name,med_name,li_name


# Featurization
# def process_and_store_documents(github_collection,medium_collection,linkedin_collection):
def process_and_store_documents(gt_name="github_data_new",med_name="medium_data",li_name="linkedin_data"):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ros2_database"]
    github_collection = db[gt_name]
    medium_collection = db[med_name]
    linkedin_collection = db[li_name]

    
    qdrant_client = QdrantClient(url="http://localhost:6333")
    github_qdrant = "github"
    medium_qdrant = "medium"
    linkedin_qdrant = "linkedin"

    # Initialize embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # @PipelineDecorator.component(cache=False)
    def create_feature_store_collections():
        """
        Create Qdrant collections:
        - One for cleaned documents (uses a dummy vector configuration).
        - One for chunked and embedded documents (uses real vector configuration).
        """
        # Vector configuration for embedded documents
        embedded_vector_params = VectorParams(
            size=embedding_model.get_sentence_embedding_dimension(),
            distance=Distance.COSINE,
        )
        
        feature_store_collections = [
            (github_qdrant, embedded_vector_params),
            (medium_qdrant, embedded_vector_params),
            (linkedin_qdrant, embedded_vector_params),
        ]
        
        for collection_name, vector_params in feature_store_collections:
            if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vector_params,
                )

    def clean_document(doc: str) -> str:
        """
        Clean a single document by removing non-ASCII characters, 
        normalizing text, and replacing URLs with placeholders.
        
        Args:
            doc (str): The document to be cleaned.
        
        Returns:
            str: The cleaned and normalized document.
        """
        # Normalize Unicode characters to ASCII
        doc = unicodedata.normalize("NFKD", doc).encode("ascii", "ignore").decode("utf-8")
        
        # Replace URLs with a placeholder
        doc = re.sub(r"http[s]?://\S+|www\.\S+", "[URL]", doc)
        
        # Remove special characters except for placeholders and preserve semantic spacing
        doc = re.sub(r"[^\w\s\[\]]", "", doc)
        
        # Normalize multiple spaces to a single space
        doc = re.sub(r"\s+", " ", doc).strip()
        
        return doc

    def chunk_document(doc: str, chunk_size: int = 1500, chunk_overlap: int = 100, embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2") -> List[str]:
        """
        Chunk a document into smaller pieces, first using a character-based splitter 
        and then refining using token-based splitting based on an embedding model.
        
        Args:
            doc (str): The document to be chunked.
            chunk_size (int): Initial character-based chunk size for splitting.
            chunk_overlap (int): Overlap size for token-based splitting.
            embedding_model_name (str): Name of the SentenceTransformers model for token-based splitting.
            
        Returns:
            List[str]: A list of chunked document pieces.
        """
        # Initialize the character-based splitter
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],  # Flexible separators for splitting
            chunk_size=chunk_size,
            chunk_overlap=0  # No overlap in this stage
        )
        # First split by characters
        text_split_by_characters = character_splitter.split_text(doc)

        # Load the embedding model to get max input length
        embedding_model = SentenceTransformer(embedding_model_name)
        max_tokens = embedding_model.get_max_seq_length()

        # Initialize token-based splitter
        token_splitter = RecursiveCharacterTextSplitter(
            separators=[" "],  # Word-based splitting for token alignment
            chunk_size=max_tokens,
            chunk_overlap=chunk_overlap,
        )
        # Refine chunks using token-based splitting
        chunks_by_tokens = []
        for section in text_split_by_characters:
            chunks_by_tokens.extend(token_splitter.split_text(section))

        return chunks_by_tokens

    def embed_chunks(chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of document chunks.
        """
        return embedding_model.encode(chunks).tolist()

    # Storing in Qdrant
    def store_in_qdrant(collection_name, embeddings: List[List[float]], metadata: List[dict]):
        """
        Store embeddings and metadata in Qdrant.
        """
        zipped_data = zip(embeddings, metadata)
        # Now create the list of PointStruct instances
        # pointss = []
        for embedding, meta in zipped_data:
            qdrant_client.upsert(collection_name=collection_name, points = [
                                PointStruct(id=str(uuid.uuid4()),
                                            vector=embedding, 
                                            payload=meta,)])
            
    """  
    Clean, chunk, embed, and store documents and their file contents from MongoDB.  
    Two snapshots are stored:  
    1. Cleaned data for fine-tuning LLMs, including file contents.  
    2. Chunked and embedded data for RAG.  
    """  
    create_feature_store_collections()
    # Fetch documents from MongoDB  
    documents_cursor = github_collection.find({}, {"_id": 1, "repo_name": 1, "repo_url": 1, "description": 1, "stars": 1, "files": 1})  
    documents = [  
        {  
            "id": str(doc["_id"]),  
            "repo_name": doc["repo_name"],  
            "repo_url": doc["repo_url"],  
            "description": doc["description"],  
            "stars": doc["stars"],  
            "files": doc.get("files", [])  # Array of file documents  
        }  
        for doc in documents_cursor  
    ]  

    if not documents:  
        return  

    # Process documents  
    for doc in documents:  
        doc_id = doc["id"]  
        repo_name = doc["repo_name"]  
        repo_url = doc["repo_url"]  
        description = doc["description"]  
        stars = doc["stars"]  

        # Step 2: Prepare to store each file content  
        cleaned_file_contents = []  # To hold cleaned file contents  

        # Process each file document  
        for file_doc in doc["files"]:  
            file_path = file_doc.get("file_path")  
            file_type = file_doc.get("file_type")  
            file_content = file_doc.get("file_content")  

            if file_content:  
                # Clean file content  
                cleaned_file_content = clean_document(file_content)  

                # Add cleaned file content to the list for storage  
                cleaned_file_contents.append({  
                    "file_path": file_path,  
                    "file_type": file_type,  
                    "content": cleaned_file_content,  
                })  

                # Optionally, also chunk and embed the file content (if needed)  
                file_chunks = chunk_document(cleaned_file_content)  
                file_embeddings = embed_chunks(file_chunks)  

                # Prepare metadata for file chunks  
                file_metadata = [  
                    {  
                        "doc_id": doc_id,  
                        "repo_name": repo_name,  
                        "repo_url": repo_url,  
                        "stars": stars,  
                        "chunk_text": chunk,  
                        "file_path": file_path,  
                        "file_type": file_type  
                    }  
                    for chunk in file_chunks  
                ]  

                # Store chunked and embedded file contents in Qdrant  
                store_in_qdrant(github_qdrant,file_embeddings, file_metadata)  
    """
    Clean, chunk, embed, and store Medium documents in Qdrant.
    """
    # Fetch documents from MongoDB
    documents_cursor = medium_collection.find({}, {"_id": 1, "id": 1, "content": 1})
    documents = [
        {
            "id": str(doc["_id"]),
            "medium_id": doc["id"],
            "content": doc["content"]
        }
        for doc in documents_cursor
    ]

    if not documents:
        return

    for doc in documents:
        doc_id = doc["id"]
        medium_id = doc["medium_id"]
        content = doc["content"]

        # Step 1: Clean the content
        cleaned_content = clean_document(content)

        # Step 2: Chunk the cleaned content
        chunks = chunk_document(cleaned_content)

        # Step 3: Embed the chunks
        embeddings = embed_chunks(chunks)

        # Step 4: Prepare metadata for each chunk
        metadata = [
            {
                "doc_id": doc_id,
                "medium_id": medium_id,
                "chunk_text": chunk
            }
            for chunk in chunks
        ]

        # Step 5: Store in Qdrant
        store_in_qdrant(medium_qdrant, embeddings, metadata)

    """
    Clean, chunk, embed, and store LinkedIn documents in Qdrant.
    """
    # Fetch documents from MongoDB
    documents_cursor = linkedin_collection.find({}, {"_id": 1, "urn": 1, "text": 1})
    documents = [
        {
            "id": str(doc["_id"]),
            "urn": doc["urn"],
            "text": doc["text"]
        }
        for doc in documents_cursor
    ]

    if not documents:
        return

    for doc in documents:
        doc_id = doc["id"]
        urn = doc["urn"]
        content = doc["text"]


        # Step 1: Clean the content
        cleaned_content = clean_document(content)

        # Step 2: Chunk the cleaned content
        chunks = chunk_document(cleaned_content)

        # Step 3: Embed the chunks
        embeddings = embed_chunks(chunks)

        # Step 4: Prepare metadata for each chunk
        metadata = [
            {
                "doc_id": doc_id,
                "urn": urn,
                "chunk_text": chunk
            }
            for chunk in chunks
        ]

        # Step 5: Store in Qdrant
        store_in_qdrant(linkedin_qdrant, embeddings, metadata)
    print('feature store created')
    return github_qdrant,medium_qdrant,linkedin_qdrant


if __name__ == '__main__':

    # create the pipeline controller
    pipe = PipelineController(
        project='RAG',
        name='Pipeline',
        version='1.1',
        add_pipeline_tags=False,
    )

    # set the default execution queue to be used (per step we can override the execution)
    pipe.set_default_execution_queue('default')

    pipe.add_function_step(
        name='scraping',
        function=run_pipeline,
        function_return=['gt_name', 'med_name', 'li_name']
    )
    
    pipe.add_function_step(
        name='featurization',
        function=process_and_store_documents,
        function_kwargs=dict(gt_name='${step_one.gt_name}',med_name='${step_one.med_name}',li_name='${step_one.li_name}'),
        function_return=['github_qdrant','medium_qdrant','linkedin_qdrant']
    )

    pipe.start_locally(run_pipeline_steps_locally=True)
