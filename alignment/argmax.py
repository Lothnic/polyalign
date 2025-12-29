import numpy as np
from ..models.similarity import similarity_matrix

def argmax_align(S):
    aligns = []

    row_max = S.argmax(axis=1)
    col_max = S.argmax(axis=0)

    for i, j in enumerate(row_max):
        if col_max[j] == i:
            aligns.append((i, j))

    return aligns

if __name__ == "__main__":
    S = similarity_matrix("Hello World","Hello World")
    print(argmax_align(S))  