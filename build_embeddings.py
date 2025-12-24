import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Path to ML text files
DATA_PATH = "D:\Projects\LLM\Data\ml_texts"

# Load embedding model (pretrained)
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
doc_names = []

# Read all .txt files
for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)
            doc_names.append(file)

# Convert text → embeddings
embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, "embeddings/ml_index.faiss")
pickle.dump(documents, open("embeddings/docs.pkl", "wb"))
pickle.dump(doc_names, open("embeddings/doc_names.pkl", "wb"))

print("Embeddings created successfully!")
print(f"Total documents indexed: {len(documents)}")
