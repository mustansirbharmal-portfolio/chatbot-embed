import openai  # Ensure you have installed the `openai` package
from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS
import pandas as pd
from openai import AzureOpenAI
from datetime import datetime
import tiktoken
import re
import secrets
import pinecone  # Pinecone for vector database
from pinecone import Pinecone
import requests
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(16))

# Global variables for lazy loading
embedding_model = None
index = None
azure_client = None
EMBEDDING_DIMENSION = 1024  # dimension for multilingual-e5-large model

def initialize_embedding_model():
    """Lazy loading of the embedding model"""
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
    return embedding_model

def initialize_pinecone():
    """Lazy loading of Pinecone index"""
    global index
    if index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME environment variable is not set")
        
        try:
            index = pc.Index(index_name)
            print(f"Successfully connected to existing index '{index_name}'")
        except Exception as e:
            print(f"Creating new index '{index_name}' with dimension {EMBEDDING_DIMENSION}")
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIMENSION,
                metric='cosine',
                spec={"serverless": {"cloud": "aws", "region": "us-west-2"}}
            )
            time.sleep(10)  # Wait for index to be ready
            index = pc.Index(index_name)
        print(f"Index '{index_name}' is ready.")
    return index

def initialize_azure():
    """Lazy loading of Azure client"""
    global azure_client
    if azure_client is None and os.getenv("AZURE_API_KEY"):
        azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT")
        )
    return azure_client

@app.route('/')
def home():
   session.clear()
   return render_template('index.html')

@app.route('/embed')
def embed():
    """Serve the embed code page"""
    server_url = os.getenv('SERVER_URL', 'http://localhost:5000')
    return render_template('embed.html', server_url=server_url)

@app.route('/ask', methods=['POST'])
def ask_route():
    # Initialize services only when needed
    initialize_pinecone()
    initialize_embedding_model()
    initialize_azure()
    
    data = request.get_json()
    user_query = data.get('query')
    response_message = ask(user_query, token_budget=4096 - 100, print_message=False)
    return jsonify({"response": response_message})

def clean_text(text):
    cleaned_text = re.sub(r'<.*?>', '', text)
    cleaned_text = re.sub(r'[\t\r\n]+', '', cleaned_text)
    return cleaned_text

def generate_embeddings(text, model="text-embedding-3-large-model"):
    """Generate embeddings using either Azure OpenAI or SentenceTransformer"""
    azure = initialize_azure()
    if azure:
        try:
            return azure.embeddings.create(input=[text], model=model).data[0].embedding
        except Exception as e:
            print(f"Azure embedding failed: {e}")
    
    # Fallback to SentenceTransformer
    model = initialize_embedding_model()
    embeddings = model.encode([text])[0]
    return embeddings.tolist()

def strings_ranked_by_relatedness(query: str, top_n: int = 100):
    """Search Pinecone for similar embeddings based on the query"""
    query_embedding = generate_embeddings(query)
    pinecone_index = initialize_pinecone()
    
    vector_id = str(hash(query))
    metadata = {"text": query}
    
    pinecone_index.upsert(vectors=[(vector_id, query_embedding, metadata)])
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_n,
        include_metadata=True
    )

    strings = [item["metadata"]["text"] for item in results["matches"]]
    relatednesses = [item["score"] for item in results["matches"]]
    return strings, relatednesses

def num_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(query: str, model: str, token_budget: int) -> str:
    strings, _ = strings_ranked_by_relatedness(query)
    introduction = 'You are a customer assistant that answers questions or give information about text entered by the user from the given data. The Characters before the fisrt space are the Campaign Ids.'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nConcat:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question

def ask(query: str, model: str = "gpt-4", token_budget: int = 4096 - 100, print_message: bool = False) -> str:
    message = query_message(query, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You are a customer assistant that answers based on the questions that are ask."},
        {"role": "user", "content": message},
    ]

    azure = initialize_azure()
    if azure:
        try:
            response = azure.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0
            )
            response_message = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Azure chat completion failed: {e}")
            response_message = None
    
    # If Azure fails or is not configured, use OpenRouter
    if not azure or not response_message:
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
                "HTTP-Referer": "https://your-site.com",
                "X-Title": "QA Bot"
            }
            
            data = {
                "model": "mistralai/mistral-7b-instruct",
                "messages": messages,
                "temperature": 0
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            response_message = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"OpenRouter chat completion failed: {e}")
            return ["I apologize, but I'm having trouble processing your request at the moment. Please try again later."]

    return response_message.split('\n\n')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
