import numpy as np
from .embedder import Embedder

def similarity_matrix(text1, text2):
    tokens1 = text1.strip().split(" ")
    tokens2 = text2.strip().split(" ")

    embedder = Embedder()

    tokens1 = embedder.encode(tokens1)
    tokens2 = embedder.encode(tokens2)

    return tokens1 @ tokens2.T

if __name__ == "__main__":
    print(similarity_matrix("Hello world", "Hello world"))

    