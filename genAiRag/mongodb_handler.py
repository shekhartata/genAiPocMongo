from pymongo import MongoClient
from typing import List, Dict
import numpy as np
from pathlib import Path

class MongoDBHandler:
    def __init__(self, uri: str, db_name: str, collection_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_documents(self, documents: List[Dict]):
        # Create index on filename and chunk_index if it doesn't exist
        self.collection.create_index([("filename", 1), ("chunk_index", 1)])
        self.collection.insert_many(documents)

    def search_similar_documents(self, query_embedding: List[float], limit: int = 5):
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index_RAG",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 200,
                    "limit": 20
                }
            },
            {
                "$project": {
                    "text": 1,
                    "filename": 1,
                    "chunk_index": 1,
                    "total_chunks": 1,
                    "directory": 1,
                    "score": {"$meta": "searchScore"}
                }
            },
            {
                "$sort": {
                    "score": -1
                }
            },
            {
                "$limit": limit
            }
        ]
        
        return list(self.collection.aggregate(pipeline))

    def save_search_results(self, results: List[Dict], output_file: Path):
        """Save chunked results to a file with metadata"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in results:
                f.write(f"=== From {doc['filename']} (Chunk {doc['chunk_index'] + 1}/{doc['total_chunks']}) ===\n")
                f.write(f"Directory: {doc['directory']}\n")
                f.write(doc['text'])
                f.write("\n\n" + "="*80 + "\n\n") 