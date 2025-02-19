from config import *
from document_processor import DocumentProcessor
from embedding_handler import EmbeddingHandler
from mongodb_handler import MongoDBHandler
from typing import List
import torch
from pathlib import Path
import logging
from openai import OpenAI
import os
import json
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {str(e)}")
    raise

def should_process_documents() -> bool:
    """Check if documents need to be processed based on timestamp file"""
    timestamp_file = Path("last_processing.json")
    
    if not timestamp_file.exists():
        logger.info("No previous processing timestamp found. Will process documents.")
        return True
        
    try:
        with open(timestamp_file, 'r') as f:
            data = json.load(f)
            last_processed = datetime.fromisoformat(data['timestamp'])
            processed_files = set(data['processed_files'])
            
        # Check if it's been less than 24 hours
        if datetime.now() - last_processed < timedelta(hours=24):
            # Get current files in the directory
            current_files = {str(f.name) for f in DOCUMENTS_PATH.glob("*") 
                           if f.suffix.lower() in SUPPORTED_EXTENSIONS}
            
            # If no files have changed, skip processing
            if current_files == processed_files:
                logger.info("Documents were processed recently and no files have changed. Skipping processing.")
                return False
                
        logger.info("Documents need to be reprocessed due to time elapsed or file changes.")
        return True
            
    except Exception as e:
        logger.error(f"Error reading timestamp file: {str(e)}")
        return True

def save_processing_timestamp():
    """Save the current timestamp and processed files list"""
    timestamp_file = Path("last_processing.json")
    
    # Get list of processed files
    processed_files = [f.name for f in DOCUMENTS_PATH.glob("*") 
                      if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'processed_files': processed_files
    }
    
    with open(timestamp_file, 'w') as f:
        json.dump(data, f)
    
    logger.info(f"Saved processing timestamp and processed {len(processed_files)} files")

def process_and_store_documents():
    """Process documents and store in MongoDB if needed"""
    if not should_process_documents():
        return
        
    try:
        # Initialize handlers
        doc_processor = DocumentProcessor(DOCUMENTS_PATH)
        embedding_handler = EmbeddingHandler(MODEL_NAME)
        mongodb_handler = MongoDBHandler(MONGODB_URI, DB_NAME, COLLECTION_NAME)

        # Process documents
        documents = doc_processor.process_documents()
        
        # Create embeddings and prepare for storage
        for doc in documents:
            doc["embedding"] = embedding_handler.create_embedding(doc["text"])
        
        # Store in MongoDB
        mongodb_handler.insert_documents(documents)
        
        # Save timestamp after successful processing
        save_processing_timestamp()
        
        logger.info("Successfully processed and stored documents")
    except Exception as e:
        logger.error(f"Error in process_and_store_documents: {str(e)}")
        raise

def get_llm_response(prompt: str) -> str:
    """Get response from OpenAI's LLM"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in MongoDB. Provide clear, concise, and practical answers based on the context provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        return f"Error: Unable to get response from LLM - {str(e)}"

def query_documents(query: str, results_file: Path = None):
    """
    Query documents and get response from LLM.
    
    Args:
        query (str): The user's question
        results_file (Path, optional): Path to save search results
        
    Returns:
        dict: Contains prompt and LLM response
    """
    try:
        # Initialize handlers
        embedding_handler = EmbeddingHandler(MODEL_NAME)
        mongodb_handler = MongoDBHandler(MONGODB_URI, DB_NAME, COLLECTION_NAME)

        # Create query embedding
        query_embedding = embedding_handler.create_embedding(query)

        # Search similar documents
        similar_docs = mongodb_handler.search_similar_documents(query_embedding)

        # Save results to file if path is provided
        if results_file:
            mongodb_handler.save_search_results(similar_docs, results_file)
            logger.info(f"Search results saved to: {results_file}")

        # Prepare context for LLM
        context = " ".join([doc["text"] for doc in similar_docs])
        
        # Create prompt for LLM
        prompt = f"""As a MongoDB consulting engineer, based on the following context, please provide a detailed and practical answer to this - Question: {query}.
        
Context: {context}"""
        
        # Get response from LLM
        llm_response = get_llm_response(prompt)
        
        return {
            "prompt": prompt,
            "llm_response": llm_response
        }
    except Exception as e:
        logger.error(f"Error in query_documents: {str(e)}")
        raise
