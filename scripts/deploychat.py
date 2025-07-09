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
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage

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
client_db = chromadb.PersistentClient(path=DB_DIR)
collection = client_db.get_or_create_collection(
    name="quran",
    embedding_function=emb_fn
)

# ─── 3) Load verse→tags map ────────────────────────────────────────────────────
with open(TAGGED_JSON, "r", encoding="utf-8") as f:
    tagged = json.load(f)
verse_tags = {}
for v in tagged:
    t = v.get("tags", [])
    if isinstance(t, list):
        tags = t
    else:
        tags = [s.strip() for s in t.split(",") if s.strip()]
    verse_tags[v["id"]] = set(tags)

# ─── 4) Tag detection helpers ──────────────────────────────────────────────────
EMOTION_PATTERNS = {
    "anxiety":    [r"\banxiou?s?\b", r"\bworry(ing)?\b", r"\bpanic\b"],
    "hope":       [r"\bhope\b", r"\boptimis(m|tic)\b"],
    # ... other patterns ...
}

def detect_tag_regex(text: str) -> str | None:
    low = text.lower()
    for tag, patterns in EMOTION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, low):
                return tag
    return None

local_model = SentenceTransformer("all-MiniLM-L6-v2")
tag_list = list(EMOTION_PATTERNS.keys())
tag_embs = local_model.encode(tag_list, convert_to_numpy=True)

def detect_tag_fuzzy(text: str, threshold: float = 0.4) -> str | None:
    vec = local_model.encode([text], convert_to_numpy=True)
    sims = cosine_similarity(vec, tag_embs)[0]
    idx, score = int(np.argmax(sims)), float(np.max(sims))
    return tag_list[idx] if score >= threshold else None

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def detect_tag_openai(text: str) -> list[str]:
    prompt = f"""
Pick all relevant topics (comma-separated, lowercase) from this list:
{', '.join(tag_list)}

User message: "{text}"

Tags:"""
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    return [t.strip() for t in resp.choices[0].message.content.split(",") if t.strip()]

def detect_tag(text: str) -> list[str]:
    tag = detect_tag_regex(text) or detect_tag_fuzzy(text)
    return [tag] if tag else detect_tag_openai(text)

# ─── 5) Softening input ─────────────────────────────────────────────────────────
def soften_input(text: str) -> str:
    reps = {
        "hopeless": "emotionally drained",
        # ... others ...
    }
    for w, r in reps.items():
        text = re.sub(w, r, text, flags=re.IGNORECASE)
    return text

# ─── 6) Retrieve Quran verses ─────────────────────────────────────────────────
def retrieve_ayahs(query: str, tags: list[str] | None = None, n_results: int = 5) -> str:
    res = collection.query(query_texts=[query], n_results=n_results * 2)
    docs, metas = res["documents"][0], res["metadatas"][0]
    pairs = list(zip(docs, metas))
    if tags:
        pairs = [(d, m) for d, m in pairs if any(t in verse_tags.get(m["verse_id"], {}) for t in tags)]
    pairs = pairs[:n_results]
    return "\n".join(f"{m['verse_id']} ({m['surah']}): {d}" for d, m in pairs) if pairs else ""

# ─── 7) Retrieve Hadiths (fixed) ───────────────────────────────────────────────
HADITH_URL = "https://hadithapi.com/api/hadiths"

def retrieve_hadiths(query: str, n_results: int = 1) -> str:
    params = {
        "apiKey":        os.getenv("HADITH_API_KEY"),
        "hadithEnglish": query,
        "status":        "Sahih",
        "paginate":      n_results
    }
    resp = requests.get(HADITH_URL, params=params, timeout=10)
    data = resp.json()

    hadiths = []
    if isinstance(data, dict):
        for key in ("data", "hadiths"):
            if isinstance(data.get(key), list):
                hadiths = data[key]
                break
    elif isinstance(data, list):
        hadiths = data

    out = []
    for h in hadiths[:n_results]:
        text = h.get("hadithEnglish", "").strip()
        b = h.get("book")
        book = b.get("nameEnglish") if isinstance(b, dict) else (b or "")
        num  = h.get("hadithNumber", "")
        ref  = f"{book} (Hadith {num})" if book and num else ""
        if text:
            out.append(f"“{text}”{' — '+ref if ref else ''}")
    return "\n".join(out)

# ─── 8) Prompt with conditional context ─────────────────────────────────────────
prompt = PromptTemplate(
    input_variables=["history", "q_block", "h_block", "question"],
    template="""
You are “ImanTherapist,” an AI trained in CBT and Islam. Always:
1. Reflect the user’s feelings in their own words.
2. Use Quran or Ḥadīth **only** if they directly address the user’s concern.
3. Ask a concise CBT-style question.
4. Suggest one gentle action.

Conversation so far:
{history}

{q_block}{h_block}User's message:
{question}

Your reply (2–6 sentences):
"""
)
main_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

followup_prompt = PromptTemplate(
    input_variables=["user_input","bot_response"],
    template="""
Based on the user’s last message and the bot’s reply, ask one brief, open-ended CBT-style question:

User said: {user_input}
Bot replied: {bot_response}

Question:
"""
)
followup_chain = LLMChain(llm=llm, prompt=followup_prompt)

closers = [
    "How does that feel for you?",
    "Does that help you reflect further?",
    "What stands out to you now?",
    "In what way might this help you grow?",
    "How do you plan to apply this insight?",
]

FALLBACK_SIG = "unable to provide the help that you need"

def is_greeting(text: str) -> bool:
    return bool(re.search(r"\b(hi|hello|salam|assalamu alaikum|hey)\b", text.lower()))

# ─── 9) Chat loop ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("CBT + Quran & Hadith Chatbot (type 'exit' to quit)\n")
    # proactive greeting
    print("Bot: Hi, how are you doing today?")
    while True:
        user = input("You: ").strip()
        if user.lower() in ("exit", "quit"):
            print("Bot: Goodbye. May Allah grant you ease.")
            break

    

        safe = soften_input(user)
        tags = detect_tag(safe)
        ctx  = retrieve_ayahs(safe, tags=tags, n_results=3)
        hquery = tags[0] if tags else safe
        hctx   = retrieve_hadiths(hquery, n_results=1)

        q_block = f"Relevant Quran Verses:\n{ctx}\n\n" if ctx else ""
        h_block = f"Relevant Hadith:\n{hctx}\n\n" if hctx else ""

        reply = main_chain.predict(
            history=memory.load_memory_variables({})['history'],
            q_block=q_block,
            h_block=h_block,
            question=safe
        )

        if FALLBACK_SIG in reply.lower():
            hist = memory.load_memory_variables({})["history"]
            override_sys = "You are a compassionate CBT therapist. Use scripture only when relevant."
            override_usr = (
                f"Conversation so far:\n{hist}\n\n"
                f"{q_block}{h_block}User's message:\n{safe}\n\n"
                "Your reply (2–6 sentences):"
            )
            msgs = [SystemMessage(content=override_sys), HumanMessage(content=override_usr)]
            reply = llm.invoke(msgs).content

        # single Bot prefix for reply
        print(f"Bot: {reply}")
        time.sleep(0.5)

        # follow-up without extra "Bot:" prefix
        print(choice(closers), followup_chain.predict(user_input=user, bot_response=reply), "\n")
