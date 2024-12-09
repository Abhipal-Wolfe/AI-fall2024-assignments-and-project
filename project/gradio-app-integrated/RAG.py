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
    

# RAG
#Query expansion
def generate_query_expansions_with_chatopenai(original_query, num_variations=5, model_name = "gpt-3.5-turbo"):
    """
    Generate query expansions using ChatOpenAI.

    Args:
        original_query (str): The original query.
        num_variations (int): Number of query variations to generate.
        model_name (str): The name of the OpenAI model to use (default is gpt-3.5-turbo).

    Returns:
        list: A list containing the original query and its expanded variations.
    """
    # openai_key = "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"
    # # Initialize the ChatOpenAI model
    # chat_model = ChatOpenAI(model=model_name, temperature=0, api_key=openai_key)
    client = openai.OpenAI(api_key="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")
    seperator = '|'
    # Define the prompt
    prompt = (
        f""""You are an AI language model assistant. Your task is to generate {num_variations}
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by '{seperator}'.
    Original question: {original_query}"""
    )

    
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful expert in Robot Operating System - 2."},
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ]
    )
    response = completion.choices[0].message

    # Extract the text from the response
    generated_text = response.content.strip()

    # Split the variations into a list (assuming each variation is on a new line)
    generated_variations = [variation.strip() for variation in generated_text.split(seperator) if variation.strip()]

    # Combine the original query with the generated variations
    all_queries = [original_query] + generated_variations
    print(all_queries)
    return all_queries

#retrieving top hits
def retrieve_from_all_collections(queries,github_qdrant="github",medium_qdrant="medium",linkedin_qdrant="linkedin"): 
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    qdrant = os.getenv("QDRANT_HOST")
    # qdrant = "localhost"
    qdrant_client = QdrantClient(url=f"http://{qdrant}:6333")


    def retrieve_from_qdrant_as_list(
        expanded_queries: List[str],
        qdrant_client,
        embedding_model,
        collection_name,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant documents from Qdrant and return a single list of results.

        Args:
            expanded_queries (List[str]): A list of expanded queries.
            top_k (int): Number of top results to retrieve for each query.

        Returns:
            List[Dict]: A single list of results across all queries.
        """

        # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # # Initialize Qdrant client
        # qdrant_client = QdrantClient(url="http://localhost:6333")  # Ensure it's properly configured

        # Prepare a list to store all results
        all_results = []

        # Retrieve results for each query
        for query in expanded_queries:
            # Generate the query vector using your embedding model
            query_vector = embedding_model.encode([query]).tolist()[0]

            # Perform search
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
            )

            # Add results to the list
            all_results.extend([
                {
                    "id": result.id,
                    "source": collection_name,
                    "score": result.score,
                    "payload": result.payload,
                }
                for result in results
            ])

        return all_results
    

    results_combined = []
    results = retrieve_from_qdrant_as_list(queries,qdrant_client,embedding_model,github_qdrant,5)
    results_combined.extend(results)
    results = retrieve_from_qdrant_as_list(queries,qdrant_client,embedding_model,medium_qdrant,5)
    results_combined.extend(results)
    results = retrieve_from_qdrant_as_list(queries,qdrant_client,embedding_model,linkedin_qdrant,5)
    results_combined.extend(results)
    print(results_combined)
    return results_combined

#reranking
def rerank(query, results, keep_top_k=5):
    """
    Rerank the retrieved results based on relevance to the query and ensure uniqueness.
    Stores and displays the source of each result.
    
    Args:
        query (str): The user query.
        results (List[Dict]): A list of results from Qdrant retrieval, each containing 'source'.
        keep_top_k (int): The number of top results to retain after reranking.
    
    Returns:
        List[Dict]: The top reranked unique results with their sources.
    """

    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Prepare query-document pairs for the model
    query_doc_tuples = [(query, result['payload']['chunk_text']) for result in results]

    # Generate relevance scores using the CrossEncoder model
    scores = model.predict(query_doc_tuples)

    # Combine scores with results
    scored_results = list(zip(scores, results))

    # Sort results by scores in descending order
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # Select the top-k results based on scores while ensuring uniqueness
    seen_ids = set()
    reranked_results = []
    for score, result in scored_results:
        if result['id'] not in seen_ids:
            seen_ids.add(result['id'])
            reranked_results.append({
                "id": result["id"],
                "chunk_text": result['payload']['chunk_text'],
                "source": result.get('source', 'Unknown'),  # Include the source
                "score": score  # Store the score for debugging or further use
            })
        if len(reranked_results) >= keep_top_k:
            break
    print(reranked_results)
    return reranked_results

# Passing to LLM
def call_LLM(query, reranked_results):
    """
    Call the LLM with a user query and reranked context from the RAG system.

    Args:
        query (str): The user's question.
        reranked_results (list): A list of reranked context chunks with payload information.
        subdomain (str): The specific subdomain (e.g., 'Nav2', 'MoveIt2', 'Gazebo') to specialize the prompt. Default is None for general queries.

    Returns:
        str: The generated response from the LLM.
    """
    subdomains = ['nav2','movit2','ros2','gazebo']
    code = False

    for sb in subdomains:
        if sb in str(query).lower():
            subdomain = sb
        else:
            subdomain = None
    if 'code' in str(query).lower():
        code = True 

    # #HF token
    # hugging_face_token = "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"

    # # Define the model name
    # # repo_name = "Abhipal/finetuned_RAG-llama" 
    # # #model_name = "meta-llama/Llama-3.1-8B" 
    # model_name = "distilgpt2"
    

    # ## Load the tokenizer and model
    # # tokenizer = AutoTokenizer.from_pretrained(repo_name,use_auth_token=hugging_face_token)
    # # model = AutoModelForCausalLM.from_pretrained(repo_name,use_auth_token=hugging_face_token)

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)

    # # Extract context from reranked results
    context = [result['chunk_text'] for result in reranked_results]

    # # Combine the context into a single string
    context_str = ". ".join(context)

    # Construct the prompt based on the subdomain
    if subdomain is None and code is None:
        prompt = f"""
        You are an expert in ROS2. Answer the user's question by providing detailed, accurate, and actionable information using the context provided.
        User query: {query}
        Context: {context_str}
        """
    elif subdomain is None and code:
        prompt = f"""You are an expert in ROS2. Write code based on the user's question and provided context.
                    Question: {query}
                    Context: {context_str}"""
    elif subdomain and code is None:
        prompt = f"""
        You are an expert in ROS2's {subdomain} subdomain. Answer the following question by using the provided context and your expertise in {subdomain}.
        User query: {query}
        Context: {context_str}
        """
    else:
        prompt = f"""You are an expert in ROS2. Write code based on the user's question and provided context.
                    Question: {query}
                    Context: {context_str}"""

    # # Encode the prompt
    # input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # # Generate a response
    # output = model.generate(
    #     input_ids,
    #     max_length=5000,  # Adjust max_length based on expected output size
    #     num_return_sequences=1,
    #     no_repeat_ngram_size=2,
    #     top_k=50,
    #     top_p=0.9,
    # )

    # # Decode the response
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # USING OLLAMA 

    ollama_url = os.getenv("OLLAMA_URL")
    model = "unsloth_model_8"

    print(prompt)
    def clean_prompt_for_api(raw_prompt):
    
        # Remove placeholders like `[URL]`, `[[Build Status]]`, etc.
        cleaned_prompt = re.sub(r"\[\[.*?\]\]|\[.*?\]", "", raw_prompt)
        
        # Remove excessive whitespace
        cleaned_prompt = re.sub(r"\s+", " ", cleaned_prompt).strip()
        
        # Escape special characters for JSON compatibility
        cleaned_prompt = cleaned_prompt.replace("\n", "\\n")
        
        return cleaned_prompt

    prompt = clean_prompt_for_api(prompt)

    # curl_command = [
    #     "curl", "-X", "POST", f"{ollama_url}/api/chat",
    #     "-H", "Content-Type: application/json",
    #     "-d", '''{
    #         "model": "{model}",
    #         "messages": [
    #             {"role": "user", "content": "{prompt}"}
    #         ],
    #         "stream": false
    #     }'''.format(model=model, prompt=prompt)
    # ]

    # result = subprocess.run(curl_command, capture_output=True, text=True)

    # print("Response:", result.stdout)


    url = f"{ollama_url}/api/chat"  # Adjust host/port if needed
    payload = {
        "model": model,
        "messages": [
            {"role":"user", "content":prompt}
        ],
        "stream": False
    }
    print("Sending prompt to ollama...")  # This will print to stdout
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        generated_response = response.json()['message']['content']  # Parse JSON response
        print(response.json())
        
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return None

    generated_response = generated_response[len(prompt):].strip()
    print(generated_response)
    Task.current_task().upload_artifact('llm_response',generated_response)
    return generated_response



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

    pipe.add_parameter(
        name='query',
        default = ""
    )

    pipe.add_function_step(
        name='query_expansion',
        # parents=['step_one'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=generate_query_expansions_with_chatopenai,
        function_kwargs=dict(original_query='${pipeline.query}'),
        function_return=['all_queries'],
    )

    pipe.add_function_step(
        name='retrieve_collections',
        # parents=['step_two'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=retrieve_from_all_collections, 
        function_kwargs=dict(queries='${query_expansion.all_queries}'),
        function_return=['results_combined'],
    )

    pipe.add_function_step(
        name='reranker',
        # parents=['step_two'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=rerank,
        function_kwargs=dict(query='${pipeline.query}',results='${retrieve_collections.results_combined}'),
        function_return=['reranked_results'],
    )

    task = Task.current_task()
    
    pipe.add_function_step(
        name='inference',
        # parents=['step_two'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=call_LLM,
        function_kwargs=dict(query='${pipeline.query}', reranked_results='${reranker.reranked_results}'),
        function_return=['llm_response'],
        monitor_artifacts=['llm_response'],
    )
    


    print('pipeline completed, launching gradio')

    # Gradio interface

    def gradio_entrance(query):
    # Get similar chunks from Qdrant
    
        pipe.add_parameter(
            name='query',
            default = f"{query}"
        )
        pipe.start_locally(run_pipeline_steps_locally=True)
        
        outputs = task.artifacts['llm_response'].get()
        
        
        
        return outputs
        

    # # Define Gradio app interface
    # iface = gr.Interface(
    #     fn=gradio_interface,   # Function to run when user submits the input
    #     inputs="text",         # Input type (Text box for query)
    #     outputs="text",        # Output type (Text output with similar chunks)
    #     live=False              # Optionally set live=True to update as the user types
    # )

    # # Launch the Gradio app
    # iface.launch(server_name="0.0.0.0",server_port=7860)


    with gr.Blocks() as app:
        gr.Markdown("# Code Assistance with ROS2")

        with gr.Row():
            question_type = gr.Dropdown(
                label="Select Question Type",
                choices=["Navigation to Specific Pose, include replanning aspects in your answer", "Provide Code for Navigation to a specific pose, include replanning aspects in your answer"],
                value="Navigation to Specific Pose, include replanning aspects in your answer",
            )

        output = gr.Textbox(label="Generated Response")
        generate_button = gr.Button("Generate")
        generate_button.click(gradio_entrance, inputs=question_type, outputs=output)

        # Launch the app
        app.launch(server_name="localhost", server_port=7860)