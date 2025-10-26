import json
import time
import random
import os
import logging
import hashlib
import gzip
import openai
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Thread

# ---------------- CONFIG ----------------
INPUT_PATH = "kenya_clean_utf8.json"  # default JSONL or JSON input
OUTPUT_DIR = Path("lang_split_out")
TEMP_PREFIX = OUTPUT_DIR / "temp_results_part"
CHECKPOINT_FILE = OUTPUT_DIR / "completed_ids.txt"
LOG_FILE = OUTPUT_DIR / "classifier.log"

MODEL = "deepseek-chat"
MAX_WORKERS = 18
BATCH_SIZE = 40
RETRIES = 3
MAX_LINES_PER_PART = 100000
LONG_TEXT_LIMIT = 500
SLEEP_BETWEEN_BATCHES = 1.0

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
    "Each object contains {id, text}. For every comment, identify the dominant language category "
    "using one of the following EXACT labels (return only one label per id):\n\n"
    "1. 'Swahili' — The comment is written mainly or entirely in standard Swahili.\n"
    "2. 'English and Swahili' — The comment mixes full English and full Swahili sentences or clauses.\n"
    "3. 'Sheng' — Kenyan urban slang blending English and Swahili, with slang expressions or phonetic distortions.\n"
    "4. 'English' — Purely English, no Swahili or Sheng words.\n\n"
    "If uncertain between 'Sheng' and 'English and Swahili', prefer 'Sheng' when informal or slangy.\n"
    "Return JSON list only: "
    "[{\"id\": \"abc123\", \"language\": \"Sheng\"}, {\"id\": \"def456\", \"language\": \"Swahili\"}]\n\n"
    "CRITICAL RULES:\n"
    "- No markdown, explanations, or commentary.\n"
    "- Use exactly one label.\n"
    "- Output must be valid JSON only."
)

# ---------------- AUTO-TUNER ----------------
class AutoTuner:
    """Safe dynamic tuner: slow ramp-up, immediate back-off."""
    def __init__(self, max_workers=MAX_WORKERS, batch_size=BATCH_SIZE):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.success_streak = 0
        self.fail_streak = 0
        self.last_adjust = time.time()

    def adjust(self, success: bool):
        if success:
            self.success_streak += 1
            self.fail_streak = 0
        else:
            self.fail_streak += 1
            self.success_streak = 0

    # --- BACKOFF: 3 consecutive fails → reduce load
        if self.fail_streak >= 3:
            old_w, old_b = self.max_workers, self.batch_size
            self.max_workers = max(1, int(self.max_workers * 0.75))
            self.batch_size = max(5, int(self.batch_size * 0.8))
            logging.warning(
                f"⚠️ AutoTuner: reducing load → workers {old_w}->{self.max_workers}, batch {old_b}->{self.batch_size}"
            )
            self.fail_streak = 0
            self.last_adjust = time.time()
            return

    # --- INCREASE: 20 clean batches and 2 min since last change
        now = time.time()
        if self.success_streak >= 20 and now - self.last_adjust > 120:
            old_w, old_b = self.max_workers, self.batch_size
            self.max_workers = min(self.max_workers * 1.1, self.max_ceiling_workers)
            self.batch_size  = min(self.batch_size * 1.05, self.max_ceiling_batch)
            logging.info(
                f"🚀 AutoTuner: increasing load → workers {old_w}->{int(self.max_workers)}, batch {old_b}->{int(self.batch_size)}"
            )
            self.success_streak = 0
            self.last_adjust = now

    # --- LONG-TERM ADAPTIVE CEILING ---
    # every 30 minutes of stable runtime, raise the allowed ceiling by 10%
        if success and (now - getattr(self, 'last_ceiling_boost', 0) > 1800):
            self.max_ceiling_workers = min(32, int(self.max_workers * 1.2))
            self.max_ceiling_batch   = min(80, int(self.batch_size * 1.2))
            self.last_ceiling_boost  = now
            logging.info(
                f"🧠 Adaptive ceiling raised → workers ≤ {self.max_ceiling_workers}, batch ≤ {self.max_ceiling_batch}"
            )

    def get_params(self):
        return self.max_workers, self.batch_size


# ---------------- GLOBALS ----------------
cache = {}
done_ids = set()
if CHECKPOINT_FILE.exists():
    done_ids = set(open(CHECKPOINT_FILE).read().split())
    logging.info(f"Resuming with {len(done_ids):,} comments already processed.")

    # --- SAFETY FIX: avoid false "everything done" on resume ---
    try:
        total_lines = sum(1 for _ in open(INPUT_PATH, "r", encoding="utf-8"))
        if len(done_ids) >= total_lines - 1000:
            logging.warning("⚠️ Checkpoint nearly complete or corrupted — restarting full run.")
            done_ids.clear()
    except Exception as e:
        logging.warning(f"Could not verify total lines: {e}")
        
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
    """Send batch to DeepSeek with retry, parser recovery, and adaptive rate-limit handling."""
    time.sleep(random.uniform(0.05, 0.3))
    content = json.dumps(
        [{"id": b["id"], "text": b["text"][:800]} for b in batch],
        ensure_ascii=False
    )
    last_err_msg = ""
    for attempt in range(RETRIES):
        try:
            r = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                max_tokens=1500,
            )
            txt = r["choices"][0]["message"]["content"].strip()
            # try direct JSON
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                # try to salvage bracketed JSON
                start, end = txt.find("["), txt.rfind("]")
                if start != -1 and end != -1:
                    try:
                        return json.loads(txt[start:end + 1])
                    except Exception:
                        pass
                # fall through to except to retry
                raise
        except Exception as e:
            msg = str(e)
            last_err_msg = msg

            # True rate-limit only (avoid "prematurely")
            if "429" in msg or "rate limit" in msg.lower():
                wait = 62 + random.uniform(0, 5)
                logging.warning(f"🚫 Rate limit hit (429); pausing {wait:.1f}s to reset window")
                time.sleep(wait)
                continue  # retry same batch, next attempt

            # Truncated / early-closed response → retry smaller batch immediately
            if "Response ended prematurely" in msg:
                logging.warning("Partial response → retrying smaller batch")
                time.sleep(5)
                # recursive retry on a halved batch
                return deepseek_batch(batch[: max(10, len(batch) // 2)])

            # Generic transient error backoff
            logging.warning(f"DeepSeek error (attempt {attempt+1}/{RETRIES}): {msg}")
            time.sleep(2 + attempt)
            continue

    # After RETRIES exhausted
    logging.warning(f"DeepSeek error: retries exhausted. Last error: {last_err_msg}")
    return []

# ---------------- MAIN ----------------
def main():
    start_time = time.time()
    total = 0
    batch, futures = [], []
    synthetic_counter = 0
    tuner = AutoTuner(MAX_WORKERS, BATCH_SIZE)
    max_workers, batch_size = tuner.get_params()
    total_lines = sum(1 for _ in open(INPUT_PATH, "r", encoding="utf-8"))
    logging.info(f"Starting classification on {total_lines:,} comments.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor, open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = rec.get("id") or f"auto_{synthetic_counter + 1}"
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
                writer_q.put([{"id": cid, "text": text, "language": lang}])
                done_ids.add(cid)
                total += 1
                continue

            batch.append({"id": cid, "text": text})
            if len(batch) >= batch_size:
                futures.append(executor.submit(deepseek_batch, batch))
                batch = []

            if len(futures) >= max_workers * 2:
                for fut in as_completed(futures):
                    result = fut.result()
                    if result:
                        for r in result:
                            cid = r["id"]
                            orig = next((b["text"] for b in batch if b["id"] == cid), "")
                            r["text"] = orig
                        writer_q.put(result)
                        for r in result:
                            cache[hash_text(r["text"])] = r.get("language", "")
                        total += len(result)
                        tuner.adjust(True)
                    else:
                        tuner.adjust(False)

                futures.clear()
                max_workers, batch_size = tuner.get_params()
                elapsed = time.time() - start_time
                rate = total / elapsed if elapsed > 0 else 0
                eta = (total_lines - total) / rate / 3600 if rate > 0 else 0
                logging.info(
                    f"✅ Progress: {total:,}/{total_lines:,} processed "
                    f"({rate:.1f}/s) | ETA ≈ {eta:.2f}h | "
                    f"Current batch={batch_size}, workers={max_workers}"
                )

    if batch:
        result = deepseek_batch(batch)
        if result:
            for r in result:
                cid = r["id"]
                orig = next((b["text"] for b in batch if b["id"] == cid), "")
                r["text"] = orig
            writer_q.put(result)
            total += len(result)

    writer_q.put(None)
    elapsed = time.time() - start_time
    logging.info(f"✅ Done {total:,} comments in {elapsed/3600:.2f}h ({total/elapsed:.1f}/s)")

    # --- FINAL SPLITTING ---
    counts = {"Swahili": 0, "English and Swahili": 0, "Sheng": 0, "English": 0, "Unknown": 0}
    out_files = {
        "Swahili": open(OUTPUT_DIR / "swahili.jsonl", "w", encoding="utf-8"),
        "English and Swahili": open(OUTPUT_DIR / "mixed.jsonl", "w", encoding="utf-8"),
        "Sheng": open(OUTPUT_DIR / "sheng.jsonl", "w", encoding="utf-8"),
        "English": open(OUTPUT_DIR / "english.jsonl", "w", encoding="utf-8"),
    }

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

    logging.info("📊 Summary:")
    for k, v in counts.items():
        logging.info(f"{k:<22}: {v}")
    logging.info(f"📁 Split files in: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_PATH, help="Path to JSONL input file")
    args = parser.parse_args()

    if args.input and args.input != INPUT_PATH:
        INPUT_PATH = args.input
        logging.info(f"Overriding input file: {INPUT_PATH}")

    main()
