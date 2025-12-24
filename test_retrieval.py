import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and metadata
index = faiss.read_index("embeddings/ml_index.faiss")
documents = pickle.load(open("embeddings/docs.pkl", "rb"))
doc_names = pickle.load(open("embeddings/doc_names.pkl", "rb"))

def retrieve(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    print("\n🔎 Question:", query)
    print("📌 Most relevant ML resources:\n")

    for i in indices[0]:
        print("➡", doc_names[i])

# Test queries
retrieve("What is overfitting in machine learning?")
retrieve("Explain supervised learning")
retrieve("What are the types of machine learning?")
retrieve("What is bias variance tradeoff?")
