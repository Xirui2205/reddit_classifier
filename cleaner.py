import json

infile  = "kenya.jsonl"          # your current file
outfile = "kenya_clean_utf8.jsonl"    # new cleaned file

with open(infile, "r", encoding="utf-8", errors="ignore") as fin, \
     open(outfile, "w", encoding="utf-8") as fout:
    for line in fin:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        body = (rec.get("body") or "").strip()
        cid  = rec.get("id") or ""

        # skip deleted/empty comments
        if not body or body.lower() in ("[deleted]", "[removed]"):
            continue

        # keep minimal fields
        cleaned = {"id": cid, "body": body}
        fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

print("✅ Done. Saved cleaned file:", outfile)