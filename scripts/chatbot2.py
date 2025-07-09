
#!/usr/bin/env python3
import os
import re
import json
import time
import requests
import numpy as np
from random import choice
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

# ─── 0) Paths & env ───────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = os.path.join(BASE_DIR, "data")
DB_DIR      = os.path.join(BASE_DIR, "quran_db")
TAGGED_JSON = os.path.join(DATA_DIR, "quran_chunks_tagged.json")
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HADITH_API_KEY = os.getenv("HADITH_API_KEY")

# ─── 1) LLM & memory ───────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.5,
    openai_api_key=OPENAI_API_KEY
)
memory = ConversationBufferMemory(
    memory_key="history",
    input_key="question",
    return_messages=False
)

# ─── 2) ChromaDB setup ────────────────────────────────────────────────────────
emb_fn = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client_db  = chromadb.PersistentClient(path=DB_DIR)
collection = client_db.get_or_create_collection(
    name="quran", embedding_function=emb_fn
)

# ─── 3) Load verse→tags map ────────────────────────────────────────────────────
with open(TAGGED_JSON, "r", encoding="utf-8") as f:
    tagged = json.load(f)
verse_tags = {}
for v in tagged:
    t = v.get("tags", [])
    tags = t if isinstance(t, list) else [s.strip() for s in t.split(",") if s.strip()]
    verse_tags[v["id"]] = set(tags)

# ─── 4) Helper: detect user emotion tag ─────────────────────────────────────────
EMOTION_PATTERNS = {
    "anxiety":    [r"\banxiou?s?\b", r"\bworry(ing)?\b", r"\bpanic\b"],
    "hope":       [r"\bhope\b", r"\boptimis(m|tic)\b"],
    # add more patterns as needed...
}

def detect_tag(text: str) -> list[str]:
    low = text.lower()
    for tag, pats in EMOTION_PATTERNS.items():
        if any(re.search(p, low) for p in pats):
            return [tag]
    return []

# ─── 5) Soften input ────────────────────────────────────────────────────────────
def soften_input(text: str) -> str:
    reps = {
        "hopeless": "emotionally drained",
        # add more replacements...
    }
    for w, r in reps.items():
        text = re.sub(w, r, text, flags=re.IGNORECASE)
    return text

# ─── 6) Retrieve Quran verses ───────────────────────────────────────────────────
def retrieve_ayahs(query: str, tags: list[str], n_results: int = 3):
    res = collection.query(query_texts=[query], n_results=n_results*2)
    docs, metas = res["documents"][0], res["metadatas"][0]
    pairs = list(zip(docs, metas))
    if tags:
        pairs = [
            (d, m) for d, m in pairs
            if any(t in verse_tags.get(m["verse_id"],{}) for t in tags)
        ]
    return pairs[:n_results]

# ─── 7) Retrieve Hadiths ────────────────────────────────────────────────────────
# ─── 7) Retrieve Hadiths ───────────────────────────────────────────────────────
HADITH_URL = "https://hadithapi.com/api/hadiths"

def retrieve_hadiths(query: str, n_results: int = 1) -> list[dict]:
    params = {
        "apiKey":        os.getenv("HADITH_API_KEY"),
        "hadithEnglish": query,
        "status":        "Sahih",
        "paginate":      n_results
    }
    resp = requests.get(HADITH_URL, params=params, timeout=10)
    data = resp.json()

    # Normalize into a Python list
    if isinstance(data, list):
        hadiths = data
    elif isinstance(data, dict):
        # Try common keys
        if isinstance(data.get("data"), list):
            hadiths = data["data"]
        elif isinstance(data.get("hadiths"), list):
            hadiths = data["hadiths"]
        else:
            hadiths = []
    else:
        hadiths = []

    # Return only the first n_results items
    return hadiths[:n_results]


# ─── 8) Core prompt ─────────────────────────────────────────────────────────────
base_prompt = """You are ImanTherapist, a CBT therapist. Use Holy texts only when they
directly address the user’s concern, otherwise remain CBT-focused.
Provide replies in 2–8 lines.
Always:
1. Reflect the user’s feelings in their own words.
2. When relevant, share a concise Qur’an verse or Hadith.
3. Ask a brief CBT-style question.
4. Suggest one small actionable step.
"""

# ─── 9) Chat loop ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("CBT + Quran & Hadith Chatbot (type 'exit' to quit)")
    while True:
        user = input("You: ").strip()
        if user.lower() in ("exit", "quit"):
            print("Bot: Goodbye. May Allah grant you ease.")
            break

        safe = soften_input(user)
        tags = detect_tag(safe)

        ayahs = retrieve_ayahs(safe, tags)
        hadiths = retrieve_hadiths(tags[0] if tags else safe)

        q_block = ""
        if ayahs:
            q_block = f"Relevant Quran Verse:\\n{ayahs[0][0]}\\n\\n"
        h_block = ""
        if hadiths:
            ht = hadiths[0]
            txt = ht.get("hadithEnglish","").strip()
            h_block = f"Relevant Hadith:\\n“{txt}”\\n\\n"

        history = memory.load_memory_variables({})["history"]
        system_msg = SystemMessage(content=base_prompt)
        human_msg  = HumanMessage(content=base_prompt +
                                 f"Conversation so far:\\n{history}\\n\\n" +
                                 q_block + h_block +
                                 f"User's message:\\n{safe}\\n\nYour reply:")
        reply = llm.invoke([system_msg, human_msg]).content

        # Print up to 8 lines
        lines = reply.strip().split("\\n")
        for line in lines[:8]:
            print("Bot:", line)
        print()
        time.sleep(0.5)
