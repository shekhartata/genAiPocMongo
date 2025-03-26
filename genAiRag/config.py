from pathlib import Path
import os

# MongoDB Atlas Configuration
MONGODB_URI = "mongodb+srv://user-test-sync:shekhartesting@cluster0.eyv8o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "testing"
COLLECTION_NAME = "genAiPoc"

# Document settings
DOCUMENTS_PATH = Path(os.path.expanduser("~/Desktop/Consulting reports"))  # Points to Desktop folder
SUPPORTED_EXTENSIONS = {".txt", ".csv"}

# Embedding Model Configuration
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768  # depends on the model you choose 

# OpenAI Configuration
OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4o-mini"  # or "gpt-4" if you have access
MAX_TOKENS = 15000

# Llama Configuration
#LLAMA_MODEL_PATH = "llama-2-7b-chat.Q4_K_M.gguf"  # You'll need to download this
#LLAMA_N_CTX = 20529  # Context window size
#LLAMA_N_THREADS = 8  # Number of CPU threads to use 