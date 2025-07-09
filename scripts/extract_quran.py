import os
import json
import pdfplumber
import re

# --- Step 1: Setup Paths ---
# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Path to your Quran PDF (inside data folder)
pdf_path = "data/quran.pdf"
output_path = "data/quran_chunks.json"

# --- Step 2: Extract Text from PDF ---
print("📖 Extracting text from:", pdf_path)

with pdfplumber.open(pdf_path) as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages)

# --- Step 3: Parse Surahs and Ayahs ---
surahs = []
current_surah = None
lines = text.splitlines()

for line in lines:
    line = line.strip()

    # Match Surah titles like: "1. THE OPENING"
    if re.match(r"^\d+\.\s+[A-Z\s]+$", line):
        num, title = line.split(".", 1)
        current_surah = {
            "surah_number": int(num.strip()),
            "surah_title": title.strip(),
            "ayahs": []
        }
        surahs.append(current_surah)

    # Match Ayahs like: "1. In the name of Allah..."
    elif re.match(r"^\d+\.\s", line) and current_surah:
        ayah_num, ayah_text = line.split(".", 1)
        current_surah["ayahs"].append({
            "ayah_number": int(ayah_num.strip()),
            "text": ayah_text.strip()
        })

    # Multiline continuation (join with previous ayah)
    elif current_surah and current_surah["ayahs"]:
        current_surah["ayahs"][-1]["text"] += " " + line.strip()

# --- Step 4: Flatten for Embedding ---
quran_chunks = []
for surah in surahs:
    for ayah in surah['ayahs']:
        quran_chunks.append({
            "id": f"{surah['surah_number']}:{ayah['ayah_number']}",
            "surah": surah['surah_title'],
            "text": ayah['text']
        })

# --- Step 5: Save to JSON ---
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(quran_chunks, f, indent=2)

print(f"✅ Extraction complete. {len(quran_chunks)} ayahs saved to: {output_path}")
