from alignment.argmax import argmax_align
from models.similarity import similarity_matrix

src_sentence = "मैंने बड़ी देर तक प्रतीक्षा की।"
tgt_sentence = "हाऊ मुलुक पौखि थाको।"

src_tokens = src_sentence.split(" ")
tgt_tokens = tgt_sentence.split(" ")

S = similarity_matrix(src_sentence, tgt_sentence)
aligns = argmax_align(S)

for i, j in aligns:
    print(src_tokens[i], "→", tgt_tokens[j])
