import numpy as np
from .argmax import argmax_align

def itermax_align(S, max_iter=2, alpha=0.9):
    A = np.zeros_like(S)

    S_work = S.copy()

    for _ in range(max_iter):
        
        new_aligns = argmax_align(S_work)
        
        for i,j in new_aligns:
            A[i,j] = 1

        M = np.ones_like(S)

        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                row_aligned = A[i,:].sum()>0
                col_aligned = A[:,j].sum()>0
                
                if row_aligned and col_aligned:
                    M[i,j] = 0
                elif row_aligned or col_aligned:
                    M[i,j] = alpha
                else:
                    M[i,j] = 1

        S_work = S * M

    aligns = [(i,j) for i,j in zip(*np.where(A>0))]
    
    return aligns
