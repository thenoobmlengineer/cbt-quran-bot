#!/usr/bin/env python3
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

# ─── 0) Resolve project paths ─────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = os.path.join(BASE_DIR, "data")
DB_DIR      = os.path.join(BASE_DIR, "quran_db")
INPUT_JSON  = os.path.join(DATA_DIR, "quran_chunks.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "quran_chunks_tagged.json")

# ─── 1) Load env & init OpenAI ────────────────────────────────────────────────
load_dotenv(os.path.join(BASE_DIR, ".env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── 2) Tag taxonomy ─────────────────────────────────────────────────────────
TAGS = [
    "hope","anxiety","gratitude","comfort","trust","forgiveness","patience",
    "strength","mercy","guidance","remembrance","submission","self-worth",
    "repentance","resilience","peace","humility","perseverance","contentment",
    "compassion","fear","grief","self-control","sincerity","reliance",
    "optimism","community"
]

# ─── 3) Load verses ───────────────────────────────────────────────────────────
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    verses = json.load(f)

# ─── 4) LLM‐based tagging function ─────────────────────────────────────────────
def tag_verse(text: str) -> list[str]:
    prompt = f"""
You are an Islamic counselor. From this list of topics, choose all that apply to this verse (comma-separated, lowercase), and nothing else:

{', '.join(TAGS)}

Verse:
\"\"\"{text}\"\"\"

Tags:"""

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user", "content": prompt}],
        temperature=0
    )
    return [t.strip() for t in resp.choices[0].message.content.split(",") if t.strip()]

# ─── 5) Tag ALL verses ────────────────────────────────────────────────────────
SAVE_INTERVAL = 500

print(f"📝 Tagging all {len(verses)} verses…")
for i, verse in enumerate(verses, start=1):
    try:
        verse["tags"] = tag_verse(verse["text"])
        print(f"{i:4d}/{len(verses)} tagged: {verse['id']} → {verse['tags']}")
    except Exception as e:
        print(f"⚠️ Error tagging {verse['id']}: {e}")
        verse["tags"] = []
    time.sleep(1)  # avoid rate limits

    # intermediate save
    if i % SAVE_INTERVAL == 0:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
            json.dump(verses, out, ensure_ascii=False, indent=2)
        print(f"💾 Progress saved at {i} verses → {OUTPUT_JSON}")

# ─── 6) Final save tagged JSON ────────────────────────────────────────────────
with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
    json.dump(verses, out, ensure_ascii=False, indent=2)
print("✅ All verses tagged and saved to", OUTPUT_JSON)

# ─── 7) Update ChromaDB metadata (tags as CSV) ────────────────────────────────
print("🔄 Updating ChromaDB metadata for all verses…")
db = chromadb.PersistentClient(path=DB_DIR)
col = db.get_or_create_collection("quran")
for verse in verses:
    tags_csv = ",".join(verse.get("tags", []))
    col.update(
        ids=[verse["id"]],
        metadatas=[{"surah": verse["surah"], "tags": tags_csv}]
    )
print("✅ ChromaDB metadata updated for all verses.")
