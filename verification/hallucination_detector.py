import numpy as np
from sentence_transformers import SentenceTransformer

# Use same embedding model everywhere
model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_hallucination_score(answer: str, chunks: list[str]) -> float:
    """
    Computes hallucination score by comparing the answer
    against EACH retrieved chunk and returning the MAX similarity.
    """

    answer_emb = model.encode([answer])

    scores = []
    for chunk in chunks:
        chunk_emb = model.encode([chunk])
        score = np.dot(answer_emb, chunk_emb.T)[0][0]
        scores.append(score)

    return max(scores) if scores else 0.0
