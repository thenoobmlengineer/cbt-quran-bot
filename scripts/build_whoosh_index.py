# scripts/build_whoosh_index.py
import os
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
import json

# 1. Define schema
schema = Schema(
    verse_id=ID(stored=True, unique=True),
    text=TEXT(analyzer=StemmingAnalyzer(), stored=False)
)

# 2. Create/open index
os.makedirs("whoosh_index", exist_ok=True)
if not index.exists_in("whoosh_index"):
    ix = index.create_in("whoosh_index", schema)
else:
    ix = index.open_dir("whoosh_index")

# 3. Index all chunks
with ix.writer() as writer, open("data/quran_windowed_chunks.json","r",encoding="utf-8") as f:
    chunks = json.load(f)
    for chunk in chunks:
        writer.update_document(
            verse_id=chunk["id"],
            text=chunk["text"]
        )
print("✅ Whoosh index built.")
