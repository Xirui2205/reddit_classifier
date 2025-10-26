import json, time, random, os, openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------- CONFIG ----------------
INPUT_PATH = "kenya_clean_utf8.json"          # JSONL or JSON
OUTPUT_DIR = Path("lang_split_out")
TEMP_FILE  = OUTPUT_DIR / "temp_results.jsonl"
BATCH_SIZE = 25
MAX_WORKERS = 8
MODEL = "deepseek-chat"
RETRIES = 3

# ---------------- API KEY (explicit) ----------------
openai.api_key = "sk-33a98b12f11f4300857e5ca93bf90e24"
openai.api_base = "https://api.deepseek.com"

# ---------------- PROMPT ----------------
SYSTEM_PROMPT = (
    "You are an expert linguist trained to classify Kenyan Reddit comments by language type. "
    "Each object contains {id, body}. For every comment, identify the dominant language category "
    "using one of the following EXACT labels (return only one label per id):\n\n"
    "1. 'Swahili' — The comment is written mainly or entirely in standard Swahili. "
    "It may contain a few English loanwords, but the grammar, structure, and function words are Swahili.\n\n"
    "2. 'English and Swahili' — The comment mixes full English and full Swahili sentences or clauses. "
    "Both languages appear clearly and intentionally. Example: 'Niko sawa today, just finished my work.'\n\n"
    "3. 'Sheng' — The comment is written in Kenyan urban slang that blends English and Swahili "
    "at the word or phrase level with heavy code-switching, slang expressions, or phonetic distortions. "
    "Sheng is informal and youth-oriented (e.g., 'Aje bro, niko area base manze!'). "
    "If a sentence feels slangy or phonetically altered Swahili-English mix, label it 'Sheng'.\n\n"
    "4. 'English' — The comment is purely in English, with no Swahili or Sheng words.\n\n"
    "If you are uncertain between 'Sheng' and 'English and Swahili', "
    "prefer 'Sheng' when the style is informal or slangy.\n\n"
    "Return output as a pure JSON list with no explanations, for example:\n"
    "[{\"id\": \"abc123\", \"language\": \"Sheng\"}, {\"id\": \"def456\", \"language\": \"Swahili\"}]\n\n"
    "CRITICAL RULES:\n"
    "- Do NOT include markdown, commentary, or explanations.\n"
    "- Use exactly one of the four labels above.\n"
    "- Output must be valid JSON only."
)

# ---------------- AUTO-TUNER ----------------
class AutoTuner:
    """Dynamically adjusts concurrency and batch size based on API stability."""
    def __init__(self, max_workers=8, batch_size=25):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.success_streak = 0
        self.fail_streak = 0
        self.last_adjust = time.time()

    def adjust(self, success: bool):
        now = time.time()
        if success:
            self.success_streak += 1
            self.fail_streak = 0
        else:
            self.fail_streak += 1
            self.success_streak = 0

        # every 2 minutes check and adapt
        if now - self.last_adjust > 120:
            self.last_adjust = now

            if self.fail_streak >= 3:
                if self.max_workers > 8:
                    self.max_workers = max(8, int(self.max_workers * 0.8))
                if self.batch_size > 30:
                    self.batch_size = max(30, int(self.batch_size * 0.8))
                print(f"⚠️ AutoTuner: reducing load → workers={self.max_workers}, batch={self.batch_size}")
                self.fail_streak = 0

            elif self.success_streak >= 10:
                if self.max_workers < 32:
                    self.max_workers = min(32, int(self.max_workers * 1.2))
                if self.batch_size < 100:
                    self.batch_size = min(100, int(self.batch_size * 1.1))
                print(f"🚀 AutoTuner: increasing load → workers={self.max_workers}, batch={self.batch_size}")
                self.success_streak = 0

    def get_params(self):
        return self.max_workers, self.batch_size


# ---------------- HELPERS ----------------
def deepseek_batch(batch):
    """Send a batch to DeepSeek and parse response"""
    content = json.dumps([{"id": b["id"], "text": b["text"][:800]} for b in batch], ensure_ascii=False)
    for attempt in range(RETRIES):
        try:
            r = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ],
                temperature=0,
                max_tokens=1500,
            )
            txt = r["choices"][0]["message"]["content"].strip()
            return json.loads(txt)
        except Exception as e:
            print(f"⚠️ DeepSeek error (attempt {attempt+1}/{RETRIES}): {e}")
            time.sleep(2 + attempt)
    return []

def safe_write(path, recs):
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------- MAIN ----------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Loading stream from {INPUT_PATH} ...")

    tuner = AutoTuner(max_workers=MAX_WORKERS, batch_size=BATCH_SIZE)
    max_workers, batch_size = tuner.get_params()
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = []
    current_batch = []
    total = 0
    synthetic_counter = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = rec.get("text") or rec.get("body") or ""
            if not text.strip():
                continue

            cid = rec.get("id")
            if not cid:
                synthetic_counter += 1
                cid = f"auto_{synthetic_counter}"

            current_batch.append({"id": cid, "text": text})

            if len(current_batch) >= batch_size:
                futures.append(executor.submit(deepseek_batch, current_batch))
                current_batch = []

            # collect completed batches
            if len(futures) >= max_workers * 3:
                for fut in as_completed(futures):
                    result = fut.result()
                    if result:
                        safe_write(TEMP_FILE, result)
                        total += len(result)
                        tuner.adjust(True)
                    else:
                        tuner.adjust(False)
                futures.clear()
                max_workers, batch_size = tuner.get_params()
                executor.shutdown(wait=True)
                executor = ThreadPoolExecutor(max_workers=max_workers)
                print(f"Processed ≈ {total} comments so far ...")

    # remaining
    if current_batch:
        futures.append(executor.submit(deepseek_batch, current_batch))

    for fut in as_completed(futures):
        result = fut.result()
        if result:
            safe_write(TEMP_FILE, result)
            total += len(result)
            tuner.adjust(True)
        else:
            tuner.adjust(False)

    executor.shutdown(wait=True)
    print(f"\n✅ Classification finished. {total} labeled comments → {TEMP_FILE}")

    # --------------- SPLITTING ---------------
    counts = {"Swahili": 0, "English and Swahili": 0, "Sheng": 0, "English": 0, "Unknown": 0}
    out_files = {
        "Swahili": open(OUTPUT_DIR / "swahili.jsonl", "w", encoding="utf-8"),
        "English and Swahili": open(OUTPUT_DIR / "mixed.jsonl", "w", encoding="utf-8"),
        "Sheng": open(OUTPUT_DIR / "sheng.jsonl", "w", encoding="utf-8"),
        "English": open(OUTPUT_DIR / "english.jsonl", "w", encoding="utf-8"),
    }

    with open(TEMP_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                lang = rec.get("language") or rec.get("Language") or rec.get("label") or "Unknown"
                counts[lang] = counts.get(lang, 0) + 1
                if lang in out_files:
                    out_files[lang].write(line)
            except Exception:
                continue

    for f in out_files.values():
        f.close()

    print("\n📊 Summary:")
    for k, v in counts.items():
        print(f"{k:<22}: {v}")
    print("\n📁 Split files in:", OUTPUT_DIR.absolute())


if __name__ == "__main__":
    main()
