import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Config
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200

ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT, "Data", "ml_texts")
OUT_DIR = os.path.join(ROOT, "embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

# Load embedding model (pretrained)
model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if len(text) <= size:
        yield text
        return

    start = 0
    step = size - overlap
    while start < len(text):
        end = start + size
        yield text[start:end]
        if end >= len(text):
            break
        start += step


chunks = []
metadata = []

# Read and chunk all .txt files
for filename in os.listdir(DATA_PATH):
    if not filename.lower().endswith(".txt"):
        continue

    path = os.path.join(DATA_PATH, filename)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        continue

    for i, chunk in enumerate(chunk_text(text)):
        chunks.append(chunk)
        metadata.append({
            "source_file": filename,
            "chunk_index": i,
            "start_char": i * (CHUNK_SIZE - CHUNK_OVERLAP),
            "length": len(chunk)
        })

if not chunks:
    print("No text chunks found in:", DATA_PATH)
    raise SystemExit(1)

print(f"Encoding {len(chunks)} chunks with model '{model.__class__.__name__}'...")
embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

# Ensure float32
embeddings = np.array(embeddings, dtype=np.float32)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, os.path.join(OUT_DIR, "ml_index.faiss"))
pickle.dump(chunks, open(os.path.join(OUT_DIR, "docs.pkl"), "wb"))
pickle.dump(metadata, open(os.path.join(OUT_DIR, "metadata.pkl"), "wb"))

print("Embeddings created successfully!")
print(f"Total chunks indexed: {len(chunks)}")
