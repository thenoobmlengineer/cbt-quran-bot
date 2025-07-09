# revamp_chunking.py
import json
from collections import defaultdict

# --- Step 1: Load your one-ayah chunks ---
with open("data/quran_chunks.json", "r", encoding="utf-8") as f:
    verses = json.load(f)

# --- Step 2: Group by surah and sort by ayah number ---
surah_map = defaultdict(list)
for v in verses:
    # Assumes your id is like "2:19" or "2:19_2" (we split off any suffix)
    surah_num, ayah_part = v["id"].split(":")
    ayah_num = int(ayah_part.split("_")[0])
    surah_map[int(surah_num)].append((ayah_num, v))

for s in surah_map:
    surah_map[s].sort(key=lambda x: x[0])  # sort by ayah_num

# --- Step 3: Build sliding windows ---
window_size = 5
half = window_size // 2

windowed_chunks = []
for surah_num, ayah_list in surah_map.items():
    n = len(ayah_list)
    for idx, (ayah_num, verse) in enumerate(ayah_list):
        start = max(0, idx - half)
        end   = min(n, idx + half + 1)
        window = [v for _, v in ayah_list[start:end]]

        # aggregate text
        text = " ".join([w["text"] for w in window])

        # aggregate tags if present
        tags = set()
        for w in window:
            tags.update(w.get("tags", []))

        # build a new chunk ID based on range
        start_id = window[0]["id"]
        end_id   = window[-1]["id"]
        chunk_id = f"{surah_num}:{start_id.split(':')[1].split('_')[0]}-{end_id.split(':')[1].split('_')[0]}"

        # record direct neighbors of the center ayah
        neighbors = []
        if idx > 0:
            neighbors.append(ayah_list[idx-1][1]["id"])
        if idx < n-1:
            neighbors.append(ayah_list[idx+1][1]["id"])

        windowed_chunks.append({
            "id":        chunk_id,
            "surah":     verse["surah"],
            "start_id":  start_id,
            "end_id":    end_id,
            "center_id": verse["id"],
            "text":      text,
            "tags":      list(tags),
            "neighbors": neighbors
        })

# --- Step 4: Save the new chunks ---
out_path = "data/quran_windowed_chunks.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(windowed_chunks, f, indent=2, ensure_ascii=False)

print(f"✅ Generated {len(windowed_chunks)} windowed chunks and saved to {out_path}")
