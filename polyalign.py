from alignment.argmax import argmax_align
from alignment.itermax import itermax_align
from models.similarity import similarity_matrix

src_sentence = "मैंने बड़ी देर तक प्रतीक्षा की।"
tgt_sentence = "हाऊ मुलुक पौखि थाको।" # kinnauri

src_tokens = src_sentence.split(" ")
tgt_tokens = tgt_sentence.split(" ")

S = similarity_matrix(src_sentence, tgt_sentence)
align = argmax_align(S)
align = itermax_align(S, max_iter=2, alpha=0.9)

for i, j in align:
    print(src_tokens[i], "→", tgt_tokens[j])
