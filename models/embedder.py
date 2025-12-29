from sentence_transformers import SentenceTransformer
import numpy as np

default_model = SentenceTransformer('BAAI/bge-m3')

class Embedder:
    def __init__(self, model=default_model):
        self.model = model
    
    def encode(self, text):
        return self.model.encode(text,normalize_embeddings=True)
        
    def similarity(self, text1, text2):
        v1 = self.encode(text1)
        v2 = self.encode(text2)
        return np.dot(v1,v2)


if __name__ == "__main__":
    embedder = Embedder()

    sim = embedder.similarity("Hello world", "नमस्ते विश्व")

    print(sim)
