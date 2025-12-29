from alignment.argmax import argmax_align
from models.similarity import similarity_matrix

S = similarity_matrix("Hello World","Hello World")
print(argmax_align(S))  
