import numpy as np
from sentence_transformers import SentenceTransformer
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 5]

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b, a)

def sentence_coverage_ratio(answer: str, chunks: list[str], threshold=0.6):
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0

    supported = 0
    sent_embeddings = model.encode(sentences)

    for sent_emb in sent_embeddings:
        max_sim = 0.0

        for chunk in chunks:
            chunk_emb = model.encode([chunk])
            sims = cosine_similarity(sent_emb, chunk_emb)
            max_sim = max(max_sim, float(np.max(sims)))

        if max_sim >= threshold:
            supported += 1

    return supported / len(sentences)
