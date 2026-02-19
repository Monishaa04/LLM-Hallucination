import faiss
import pickle
from sentence_transformers import SentenceTransformer

from generate_answer import generate_answer
from verification.hallucination_detector import compute_hallucination_score
from verification.sentence_coverage import sentence_coverage_ratio
from verification.concept_overlap import concept_overlap_score
from verification.final_trust import final_trust_score
from verification.trust_score import interpret_score

# ✅ GLOBAL MODEL INITIALIZATION
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and documents
index = faiss.read_index("embeddings/ml_index.faiss")
documents = pickle.load(open("embeddings/docs.pkl", "rb"))

def retrieve_chunks_with_confidence(question, k=5):
    q_emb = embed_model.encode([question])
    distances, indices = index.search(q_emb, k)

    chunks = [documents[i] for i in indices[0]]
    retrieval_confidence = float(1 / (1 + distances[0][0]))

    return chunks, retrieval_confidence

def answer_question(question):
    chunks, retrieval_conf = retrieve_chunks_with_confidence(question)

    context = " ".join(chunks)
    answer = generate_answer(context, question)

    similarity = compute_hallucination_score(answer, chunks)
    coverage = sentence_coverage_ratio(answer, chunks)
    overlap = concept_overlap_score(answer, chunks)

    trust_value = final_trust_score(
        similarity, coverage, overlap, retrieval_conf
    )

    trust_level = interpret_score(trust_value)

    return {
        "question": question,
        "answer": answer,
        "semantic_similarity": round(similarity, 3),
        "sentence_coverage": round(coverage, 3),
        "concept_overlap": round(overlap, 3),
        "retrieval_confidence": round(retrieval_conf, 3),
        "final_trust_score": round(trust_value, 3),
        "trust_level": trust_level
    }
