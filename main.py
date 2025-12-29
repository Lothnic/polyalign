from alignment.argmax import argmax_align
from models.similarity import similarity_matrix

src_sentence = "the dog runs fast"
tgt_sentence = "कुत्ता तेज़ दौड़ता है"

src_tokens = src_sentence.split(" ")
tgt_tokens = tgt_sentence.split(" ")

S = similarity_matrix(src_sentence, tgt_sentence)
aligns = argmax_align(S)

for i, j in aligns:
    print(src_tokens[i], "→", tgt_tokens[j])
