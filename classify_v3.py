import json, time, random, os, logging, hashlib, gzip, openai
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Thread

# ---------------- CONFIG ----------------
INPUT_PATH = "kenya_clean_utf8.json"          # JSONL or JSON
OUTPUT_DIR = Path("lang_split_out")
TEMP_PREFIX = OUTPUT_DIR / "temp_results_part"
CHECKPOINT_FILE = OUTPUT_DIR / "completed_ids.txt"
LOG_FILE = OUTPUT_DIR / "classifier.log"

MODEL = "deepseek-chat"
MAX_WORKERS = 20
BATCH_SIZE = 40
RETRIES = 3
MAX_LINES_PER_PART = 100000
LONG_TEXT_LIMIT = 500       # words
SLEEP_BETWEEN_BATCHES = 0.2

# ---------------- API KEY ----------------
openai.api_key = "sk-33a98b12f11f4300857e5ca93bf90e24"
openai.api_base = "https://api.deepseek.com"

# ---------------- LOGGING ----------------
OUTPUT_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

# ---------------- SYSTEM PROMPT ----------------
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

# ---------------- GLOBALS ----------------
cache = {}
done_ids = set()
if CHECKPOINT_FILE.exists():
    done_ids = set(open(CHECKPOINT_FILE).read().split())
    logging.info(f"Resuming with {len(done_ids):,} comments already processed.")

writer_q = Queue()
part_index = 1
lines_in_part = 0

# ---------------- WRITER THREAD ----------------
def writer_thread():
    global part_index, lines_in_part
    current_path = f"{TEMP_PREFIX}_{part_index:03d}.jsonl"
    while True:
        recs = writer_q.get()
        if recs is None:
            break
        with open(current_path, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                lines_in_part += 1
                with open(CHECKPOINT_FILE, "a") as ckp:
                    ckp.write(r["id"] + "\n")
        if lines_in_part >= MAX_LINES_PER_PART:
            lines_in_part = 0
            part_index += 1
            current_path = f"{TEMP_PREFIX}_{part_index:03d}.jsonl"
    logging.info("Writer thread finished.")

Thread(target=writer_thread, daemon=True).start()

# ---------------- UTILITIES ----------------
def safe_text_check(text):
    if not text.strip():
        return False
    if '"' not in text and "'" not in text:
        return False
    return True

def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# ---------------- DEEPSEEK BATCH ----------------
def deepseek_batch(batch):
    """Send batch to DeepSeek with retry, rate-limit handling, and parser recovery."""
    time.sleep(random.uniform(0.05, 0.3))
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
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                start, end = txt.find("["), txt.rfind("]")
                if start != -1 and end != -1:
                    try:
                        return json.loads(txt[start:end + 1])
                    except Exception:
                        pass
                raise
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                logging.warning("Rate limit hit; sleeping 10s")
                time.sleep(10)
            else:
                logging.warning(f"DeepSeek error (attempt {attempt+1}/{RETRIES}): {msg}")
                time.sleep(2 + attempt)
    return []

# ---------------- MAIN ----------------
def main():
    start_time = time.time()
    total = 0
    batch, futures = [], []
    synthetic_counter = 0
    total_lines = sum(1 for _ in open(INPUT_PATH, "r", encoding="utf-8"))
    logging.info(f"Starting classification on {total_lines:,} comments.")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor, open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = rec.get("id") or f"auto_{synthetic_counter+1}"
            if cid in done_ids:
                continue
            synthetic_counter += 1

            text = rec.get("text") or rec.get("body") or ""
            if not safe_text_check(text):
                continue
            if len(text.split()) > LONG_TEXT_LIMIT:
                text = " ".join(text.split()[:LONG_TEXT_LIMIT])

            h = hash_text(text)
            if h in cache:
                lang = cache[h]
                writer_q.put([{"id": cid, "language": lang}])
                done_ids.add(cid)
                total += 1
                continue

            batch.append({"id": cid, "text": text})
            if len(batch) >= BATCH_SIZE:
                futures.append(executor.submit(deepseek_batch, batch))
                batch = []

            if len(futures) >= MAX_WORKERS * 2:
                for fut in as_completed(futures):
                    result = fut.result()
                    if result:
                        writer_q.put(result)
                        for r in result:
                            cache[hash_text(r.get("text", ""))] = r.get("language", "")
                        total += len(result)
                futures.clear()
                elapsed = time.time() - start_time
                rate = total / elapsed if elapsed > 0 else 0
                eta = (total_lines - total) / rate / 3600 if rate > 0 else 0
                logging.info(f"{total:,}/{total_lines:,} • {rate:.1f}/s • ETA {eta:.2f}h")

    if batch:
        result = deepseek_batch(batch)
        if result:
            writer_q.put(result)
            total += len(result)

    writer_q.put(None)
    elapsed = time.time() - start_time
    logging.info(f"✅ Done {total:,} comments in {elapsed/3600:.2f}h ({total/elapsed:.1f}/s)")

    # --------------- FINAL SPLITTING ---------------
    counts = {"Swahili": 0, "English and Swahili": 0, "Sheng": 0, "English": 0, "Unknown": 0}
    out_files = {
        "Swahili": open(OUTPUT_DIR / "swahili.jsonl", "w", encoding="utf-8"),
        "English and Swahili": open(OUTPUT_DIR / "mixed.jsonl", "w", encoding="utf-8"),
        "Sheng": open(OUTPUT_DIR / "sheng.jsonl", "w", encoding="utf-8"),
        "English": open(OUTPUT_DIR / "english.jsonl", "w", encoding="utf-8"),
    }

    # merge all temp parts first
    temp_files = sorted(OUTPUT_DIR.glob("temp_results_part_*.jsonl"))
    for part in temp_files:
        with open(part, "r", encoding="utf-8") as f:
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

    logging.info("\n📊 Summary:")
    for k, v in counts.items():
        logging.info(f"{k:<22}: {v}")
    logging.info("\n📁 Split files in: %s", OUTPUT_DIR.absolute())

    os.system(f"wc -l {INPUT_PATH}")
    os.system(f"wc -l {OUTPUT_DIR}/*jsonl || true")

if __name__ == "__main__":
    main()
