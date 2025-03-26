import csv
from pathlib import Path
from typing import List, Dict
import os
import logging
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download required NLTK data with error handling and SSL workaround"""
    try:
        # SSL workaround for NLTK downloads
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Download required NLTK data
        for resource in ['punkt_tab', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Downloading {resource} for NLTK...")
                nltk.download(resource, quiet=True)
                logger.info(f"Successfully downloaded {resource}")

    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        raise

# Download NLTK data when module is imported
download_nltk_data()

class DocumentProcessor:
    def __init__(self, documents_path: Path, chunk_size: int = 1000, chunk_overlap: int = 200, min_chunk_size: int = 100):
        self.documents_path = documents_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        if not self.documents_path.exists():
            raise ValueError(f"Documents path does not exist: {self.documents_path}")
        logger.info(f"Initialized DocumentProcessor with path: {self.documents_path}")

    def extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from a .txt file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"Successfully extracted text from TXT file: {file_path.name}")
            return text
        except Exception as e:
            logger.error(f"Error processing TXT file {file_path.name}: {str(e)}")
            return ""

    def extract_text_from_csv(self, file_path: Path) -> str:
        """Extract text from a .csv file"""
        try:
            text = ""
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                # Convert CSV data to a text format
                for row in csv_reader:
                    text += " | ".join(row) + "\n"
            logger.info(f"Successfully extracted text from CSV file: {file_path.name}")
            return text
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path.name}: {str(e)}")
            return ""

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters but keep sentence endings
        text = re.sub(r'[^\w\s.!?]', '', text)
        return text

    def get_semantic_similarity(self, chunk1: str, chunk2: str) -> float:
        """Calculate semantic similarity between two text chunks"""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([chunk1, chunk2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0

    def create_semantic_chunks(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks"""
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])

        for i in range(1, len(sentences)):
            current_sentence = sentences[i]
            sentence_size = len(current_sentence)

            # Check if adding the next sentence would exceed chunk size
            if current_size + sentence_size > self.chunk_size:
                # Create a chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)

                # Start new chunk, considering semantic similarity
                if chunks:
                    last_chunk = chunks[-1]
                    similarity = self.get_semantic_similarity(last_chunk, current_sentence)
                    
                    # If highly similar to previous chunk, add to overlap
                    if similarity > 0.3:  # Threshold can be adjusted
                        current_chunk = current_chunk[-2:] if len(current_chunk) > 1 else current_chunk[-1:]
                    else:
                        current_chunk = []
                else:
                    current_chunk = []

                current_chunk.append(current_sentence)
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(current_sentence)
                current_size += sentence_size

        # Add the last chunk if it exists
        if current_chunk and len(' '.join(current_chunk)) >= self.min_chunk_size:
            chunks.append(' '.join(current_chunk))

        logger.info(f"Created {len(chunks)} semantic chunks from text")
        return chunks

    def process_documents(self) -> List[Dict]:
        documents = []
        processed_count = 0
        
        logger.info(f"Starting document processing in: {self.documents_path}")
        
        for file_path in self.documents_path.rglob("*"):
            file_ext = file_path.suffix.lower()
            
            if file_ext not in ['.txt', '.csv']:
                continue

            logger.info(f"Processing file: {file_path.name} from directory: {file_path.parent}")
            
            # Extract text based on file type
            if file_ext == '.txt':
                text = self.extract_text_from_txt(file_path)
            elif file_ext == '.csv':
                text = self.extract_text_from_csv(file_path)
            else:
                continue

            if text.strip():
                # Create semantic chunks from the text
                chunks = self.create_semantic_chunks(text)
                relative_path = file_path.relative_to(self.documents_path)
                
                # Create a document entry for each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    documents.append({
                        "filename": file_path.name,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "text": chunk,
                        "file_path": str(file_path),
                        "relative_path": str(relative_path),
                        "directory": str(file_path.parent),
                        "file_type": file_ext[1:]  # Store file type without dot
                    })
                
                processed_count += 1
                logger.info(f"Successfully processed {file_path.name} into {len(chunks)} semantic chunks")

        logger.info(f"Completed processing {processed_count} text/CSV documents into {len(documents)} semantic chunks recursively")
        return documents 