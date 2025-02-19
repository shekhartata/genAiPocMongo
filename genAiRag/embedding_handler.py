from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
import numpy as np

class EmbeddingHandler:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def create_embedding(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                              max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings[0].cpu().numpy().tolist() 