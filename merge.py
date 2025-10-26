import json
from pathlib import Path

# Directories
input_dir = Path("input_data")          # original reddit data
classified_dir = Path("lang_split_out") # classifier outputs
output_dir = Path("merged_output_split") # final simplified output
output_dir.mkdir(exist_ok=True)

# Step 1: Load all classified results into a dictionary {id: language}
classification_map = {}

print("🔍 Loading classification results...")
for file in classified_dir.glob("*.jsonl"):
    if "temp" in file.name or "completed_ids" in file.name:
        continue
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                cid = rec.get("id")
                lang = rec.get("language")
                if cid and lang:
                    classification_map[cid] = lang
            except Exception:
                continue
print(f"✅ Loaded {len(classification_map):,} classifications.")

# Step 2: Prepare output files by language
languages = ["English", "Swahili", "Sheng", "English and Swahili"]
writers = {lang: open(output_dir / f"{lang.lower().replace(' ', '_')}.jsonl", "w", encoding="utf-8") for lang in languages}
unknown_writer = open(output_dir / "unknown.jsonl", "w", encoding="utf-8")

# Step 3: Read original Reddit comments and split by language
print("🔄 Splitting original data by language...")
count_total, count_matched = 0, 0

for infile in sorted(input_dir.glob("*.jsonl")):
    with open(infile, "r", encoding="utf-8") as fin:
        for line in fin:
            count_total += 1
            try:
                rec = json.loads(line)
                cid = rec.get("id")
                body = rec.get("body") or rec.get("text") or ""
                if not cid or not body.strip():
                    continue

                lang = classification_map.get(cid)
                if lang in writers:
                    writers[lang].write(json.dumps({"id": cid, "body": body}, ensure_ascii=False) + "\n")
                    count_matched += 1
                else:
                    unknown_writer.write(json.dumps({"id": cid, "body": body}, ensure_ascii=False) + "\n")
            except Exception:
                continue

# Step 4: Close all writers
for f in writers.values():
    f.close()
unknown_writer.close()

print(f"✅ Split complete! Matched {count_matched:,}/{count_total:,} comments.")
print(f"📁 Output saved in: {output_dir.resolve()}")
