import os
import json
from collections import defaultdict
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- Step 1: Load Quran Windowed Chunks ---
windowed_chunks_path = "data/quran_windowed_chunks.json"
if not os.path.exists(windowed_chunks_path):
    raise FileNotFoundError(f"Windowed chunks file not found: {windowed_chunks_path}")

with open(windowed_chunks_path, "r", encoding="utf-8") as f:
    quran_chunks = json.load(f)

# --- Step 2: Setup Chroma-compatible Embedding Function ---
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# --- Step 3: Initialize Chroma Client & Collection ---
client = chromadb.PersistentClient(path="quran_db")
collection = client.get_or_create_collection(
    name="quran_windowed",
    embedding_function=embedding_function
)

# --- Step 4: Ensure Unique IDs ---
id_counts = defaultdict(int)
for chunk in quran_chunks:
    base_id = chunk["id"]
    id_counts[base_id] += 1
    if id_counts[base_id] > 1:
        chunk["id"] = f"{base_id}_{id_counts[base_id]}"

# Helper: stringify list to comma-separated string

def stringify_list(lst):
    return ",".join(lst) if lst else None

# --- Step 5: Batch Insert with Clean Metadata ---
batch_size = 100
for i in range(0, len(quran_chunks), batch_size):
    batch = quran_chunks[i : i + batch_size]

    documents = []
    ids = []
    metadatas = []

    for chunk in batch:
        documents.append(chunk["text"])
        ids.append(chunk["id"])

        meta = {}
        if chunk.get("surah"):
            meta["surah"] = chunk["surah"]
        if chunk.get("start_id"):
            meta["start_id"] = chunk["start_id"]
        if chunk.get("end_id"):
            meta["end_id"] = chunk["end_id"]
        if chunk.get("center_id"):
            meta["center_id"] = chunk["center_id"]

        tags_str = stringify_list(chunk.get("tags", []))
        if tags_str is not None:
            meta["tags"] = tags_str

        neighbors_str = stringify_list(chunk.get("neighbors", []))
        if neighbors_str is not None:
            meta["neighbors"] = neighbors_str

        metadatas.append(meta)

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

print(f"✅ Stored {len(quran_chunks)} windowed chunks in vector DB: '{collection.name}'")
