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

# ---- Optional accurate token counting (fallbacks to heuristic) ----
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def estimate_tokens(txt: str) -> int:
        try:
            return len(_enc.encode(txt))
        except Exception:
            # small fallback
            return max(1, int(len(txt) / 4))
except Exception:
    def estimate_tokens(txt: str) -> int:
        # heuristics: ~4 chars per token baseline for Latin scripts
        # bias a bit higher to be safe
        wc = len(txt.split())
        return max(wc, int(len(txt) * 0.32))

# ---------------- CONFIG ----------------
INPUT_PATH = "kenya_clean_utf8.jsonl"  # default JSONL or JSON input
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

# ---------------- AUTO-TUNER V4 ----------------
class AutoTuner:
    """
    Efficiency & latency aware tuner with adaptive ceilings and token-budgeted batching.

    Inputs to adjust():
      - success: bool
      - rate:    comments/sec (throughput)
      - token_rate: tokens/sec (optional, 0 if unknown)
      - avg_latency: average seconds per batch recently (EMA-friendly)

    It will:
      * Immediately back off on repeated failures.
      * Use EMA of rate & latency; if efficiency (rate/worker) drops >10%, back off.
      * Slowly ramp up when stable.
      * Raise soft ceilings every 30 minutes of stability.
      * Apply model-aware concurrency cap = min(ceiling, floor(60 / max(0.5, avg_latency))).
      * Adjust batch size toward a target token budget.
    """

    def __init__(self, max_workers=MAX_WORKERS, batch_size=BATCH_SIZE):
        self.max_workers = max(1, int(max_workers))
        self.batch_size = max(1, int(batch_size))

        # Soft ceilings start at config; can grow slowly
        self.ceiling_workers = self.max_workers
        self.ceiling_batch   = self.batch_size

        # Smoothing
        self.ema_alpha = 0.2
        self.ema_rate = 0.0
        self.ema_token_rate = 0.0
        self.ema_latency = 0.0

        self.prev_ema_rate = 0.0
        self.prev_eff = 0.0
        self.success_streak = 0
        self.fail_streak = 0
        self.last_adjust = time.time()
        self.last_ceiling_raise = time.time()

        # Token budget for a batch (tune as needed)
        self.target_batch_tokens = 3800

    def _ema(self, prev, x):
        return x if prev <= 0 else (self.ema_alpha * x + (1 - self.ema_alpha) * prev)

    def adjust(self, success: bool, rate: float = 0.0, total: int = 0, token_rate: float = 0.0, avg_latency: float = 0.0):
        now = time.time()

        # Update EMAs
        if rate > 0:
            self.ema_rate = self._ema(self.ema_rate, rate)
        if token_rate > 0:
            self.ema_token_rate = self._ema(self.ema_token_rate, token_rate)
        if avg_latency > 0:
            self.ema_latency = self._ema(self.ema_latency, avg_latency)

        # Success/failure streaks
        if success:
            self.success_streak += 1
            self.fail_streak = 0
        else:
            self.fail_streak += 1
            self.success_streak = 0

        # Immediate backoff on repeated failures
        if self.fail_streak >= 3:
            old_w, old_b = self.max_workers, self.batch_size
            self.max_workers = max(1, int(self.max_workers * 0.75))
            self.batch_size  = max(5, int(self.batch_size * 0.8))
            logging.warning(f"⚠️ AutoTuner: reducing load → workers {old_w}->{self.max_workers}, batch {old_b}->{self.batch_size}")
            self.fail_streak = 0
            self.last_adjust = now
            return

        # Efficiency check every ~2 minutes
        if now - self.last_adjust > 120 and self.ema_rate > 0:
            efficiency = self.ema_rate / max(self.max_workers, 1)
            if self.prev_eff > 0:
                delta_eff = (efficiency - self.prev_eff) / self.prev_eff
                if delta_eff < -0.10:
                    # efficiency per worker dropped >10% → back off gently
                    old_w, old_b = self.max_workers, self.batch_size
                    self.max_workers = max(1, int(self.max_workers * 0.9))
                    self.batch_size  = max(5, int(self.batch_size * 0.95))
                    logging.warning(f"📉 Efficiency↓ {delta_eff*100:.1f}% → workers {old_w}->{self.max_workers}, batch {old_b}->{self.batch_size}")
                    self.last_adjust = now
                    self.prev_eff = efficiency
                    return
            self.prev_eff = efficiency

        # Model-aware concurrency cap by latency
        if self.ema_latency > 0:
            cap_by_latency = max(1, int(min(MAX_WORKERS, 180.0 / max(0.5, self.ema_latency))))
            if self.ceiling_workers > cap_by_latency:
                self.ceiling_workers = max(1, cap_by_latency)
                logging.info(f"⏱️ Latency cap applied → workers ≤ {self.ceiling_workers}")

        # Gradual scale up on stability
        if self.success_streak >= 20 and now - self.last_adjust > 120:
            old_w, old_b = self.max_workers, self.batch_size
            new_w = min(int(self.max_workers * 1.10), self.ceiling_workers)
            new_b = min(int(self.batch_size * 1.05), self.ceiling_batch)

            if new_w > self.max_workers or new_b > self.batch_size:
                self.max_workers, self.batch_size = max(1, new_w), max(5, new_b)
                logging.info(f"🚀 AutoTuner: increasing load → workers {old_w}->{self.max_workers}, batch {old_b}->{self.batch_size}")
                self.last_adjust = now
            self.success_streak = 0

        # Adaptive ceiling raise every 30 minutes with no recent fails
        if now - self.last_ceiling_raise > 1800 and self.fail_streak == 0:
            old_cw, old_cb = self.ceiling_workers, self.ceiling_batch
            self.ceiling_workers = min(int(self.ceiling_workers * 1.10) + 1, 64)
            self.ceiling_batch   = min(int(self.ceiling_batch * 1.10) + 1, 120)
            logging.info(f"🧠 Adaptive ceiling raised → workers ≤ {self.ceiling_workers}, batch ≤ {self.ceiling_batch}")
            self.last_ceiling_raise = now

        # Token-budgeted batch sizing (only if we have token rate or rough latency)
        if self.ema_token_rate > 0 and self.ema_rate > 0:
            # average tokens per comment ≈ token_rate / rate
            avg_tpc = max(1.0, self.ema_token_rate / self.ema_rate)
            target_b = int(self.target_batch_tokens / avg_tpc)
            target_b = min(max(5, target_b), self.ceiling_batch)
            # Move slowly toward target
            if target_b > self.batch_size:
                self.batch_size = min(self.batch_size + max(1, int(0.1 * self.batch_size)), target_b)
            elif target_b < self.batch_size:
                self.batch_size = max(target_b, int(self.batch_size * 0.9))

    def get_params(self):
        return int(self.max_workers), int(self.batch_size)


# ---------------- GLOBALS ----------------
cache = {}
done_ids = set()

def validate_checkpoint():
    """Safely handle resume logic — detect stale or incomplete checkpoints."""
    if not CHECKPOINT_FILE.exists():
        logging.info("No checkpoint file found — starting fresh.")
        return set()

    try:
        done = set(open(CHECKPOINT_FILE, "r", encoding="utf-8").read().split())
        total_lines = sum(1 for _ in open(INPUT_PATH, "r", encoding="utf-8"))
        pct = (len(done) / total_lines) * 100 if total_lines else 0

        logging.info(f"Resuming with {len(done):,}/{total_lines:,} ({pct:.1f}%) comments already processed.")

        # ✅ If checkpoint seems too large or dataset changed → force partial restart
        if len(done) >= total_lines:
            logging.warning("⚠️ Checkpoint indicates all lines processed — forcing full recheck of unclassified records.")
            return set()  # treat as fresh start, but keep old outputs

        # ✅ If checkpoint covers >90% but not all, verify outputs actually exist
        if pct > 90:
            logging.info("Checkpoint covers >90%, verifying unprocessed records.")
        return done

    except Exception as e:
        logging.warning(f"⚠️ Checkpoint validation failed ({e}) — starting from scratch.")
        return set()


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
    """Send batch to DeepSeek with retry, parser recovery, and adaptive rate-limit handling.
       Returns: list_of_dicts  (each includes id, text, language)
       - No truncation of text
       - Retries same batch once before halving
       - Records latency and token usage for tuner telemetry
    """
    time.sleep(random.uniform(0.05, 0.3))

    # --- token stats for telemetry ---
    content_texts = [b["text"] for b in batch]
    approx_tokens = sum(estimate_tokens(t) for t in content_texts)

    # --- no truncation, send full text ---
    payload = [{"id": b["id"], "text": b["text"]} for b in batch]
    content = json.dumps(payload, ensure_ascii=False)

    last_err_msg = ""
    for attempt in range(RETRIES):
        start = time.time()
        try:
            r = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                max_tokens=3000,
            )
            txt = r["choices"][0]["message"]["content"].strip()
            try:
                out = json.loads(txt)
            except json.JSONDecodeError:
                s, e = txt.find("["), txt.rfind("]")
                if s != -1 and e != -1:
                    out = json.loads(txt[s:e + 1])
                else:
                    raise

            # attach meta fields for tuner
            latency = time.time() - start
            for rec in out:
                rec["_latency"] = latency
                rec["_tokens"] = approx_tokens
                rec["_batch"] = len(batch)
            return out

        except Exception as e:
            msg = str(e)
            last_err_msg = msg
        # --- handle content-filter rejections ---
        if "Content Exists Risk" in msg or "invalid_request_error" in msg:
            logging.warning("⚠️ DeepSeek content filter triggered — isolating offending texts.")
        # process each comment individually to find safe ones
            safe_results = []
            for b in batch:
                try:
                    single_payload = json.dumps([{"id": b["id"], "text": b["text"]}], ensure_ascii=False)
                    r = openai.ChatCompletion.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": single_payload},
                        ],
                        temperature=0,
                        max_tokens=3000,
                    )
                    txt = r["choices"][0]["message"]["content"].strip()
                    safe_results.extend(json.loads(txt))
                except Exception as inner_e:
                    logging.warning(f"🧩 Skipping one comment due to filter: {inner_e}")
                    continue
            if safe_results:
                return safe_results
            # if everything failed, return empty so the caller just moves on
            return []

            # --- handle rate limits ---
            if "429" in msg or "rate limit" in msg.lower():
                wait = 62 + random.uniform(0, 5)
                logging.warning(f"🚫 Rate limit hit (429); pausing {wait:.1f}s to reset window")
                time.sleep(wait)
                continue

            # --- handle partial response ---
            if "Response ended prematurely" in msg:
                logging.warning("Partial response → retrying same batch once")
                time.sleep(5)
                try:
                    # retry full batch once
                    return deepseek_batch(batch)
                except Exception as e2:
                    logging.warning(f"Second partial → shrinking batch and retrying ({e2})")
                    time.sleep(5)
                    smaller_batch = batch[: max(10, len(batch) // 2)]
                    return deepseek_batch(smaller_batch)

            # --- generic transient error backoff ---
            logging.warning(f"DeepSeek error (attempt {attempt+1}/{RETRIES}): {msg}")
            time.sleep(2 + attempt)
            continue

    # --- if retries exhausted ---
    logging.warning(f"DeepSeek error: retries exhausted. Last error: {last_err_msg}")
    return []

# ---------------- MAIN ----------------
def validate_checkpoint():
    """
    Verify the checkpoint and return the set of processed IDs.
    If checkpoint looks stale, partially corrupted, or covers everything, auto-trim.
    """
    if not CHECKPOINT_FILE.exists():
        logging.info("No checkpoint file found — starting fresh.")
        return set()

    try:
        done = set(open(CHECKPOINT_FILE, "r", encoding="utf-8").read().split())
        total_lines = sum(1 for _ in open(INPUT_PATH, "r", encoding="utf-8"))
        pct = (len(done) / total_lines) * 100 if total_lines else 0
        logging.info(f"Resuming with {len(done):,}/{total_lines:,} ({pct:.1f}%) already processed.")
        return done
    except Exception as e:
        logging.warning(f"⚠️ Checkpoint validation failed ({e}) — starting from scratch.")
        return set()


def get_unprocessed_lines(input_path, done_ids):
    """
    Generator that yields only lines whose IDs are NOT in done_ids.
    Logs how many remain.
    """
    unprocessed = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                cid = rec.get("id")
                if not cid or cid not in done_ids:
                    unprocessed.append(line)
            except Exception:
                continue

    if len(unprocessed) == 0:
        logging.warning("⚠️ No unprocessed lines found — everything in checkpoint already processed.")
    else:
        logging.info(f"✅ Found {len(unprocessed):,} unprocessed lines to classify.")
    return unprocessed


def main():
    start_time = time.time()
    total = 0
    batch, futures = [], []
    synthetic_counter = 0

    # ✅ Load and validate checkpoint
    done_ids = validate_checkpoint()
    unprocessed = get_unprocessed_lines(INPUT_PATH, done_ids)

    if not unprocessed:
        logging.info("All records already processed — exiting cleanly.")
        return

    tuner = AutoTuner(MAX_WORKERS, BATCH_SIZE)
    max_workers, batch_size = tuner.get_params()

    total_lines = sum(1 for _ in open(INPUT_PATH, "r", encoding="utf-8"))
    logging.info(f"Starting classification on {len(unprocessed):,} remaining comments out of {total_lines:,} total.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for line in unprocessed:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = rec.get("id") or f"auto_{synthetic_counter + 1}"
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
            
            # --- flush batch early if token count gets too high (safety guard) ---
            if sum(estimate_tokens(b["text"]) for b in batch) > 7000:
                futures.append(executor.submit(deepseek_batch, batch))
                batch = []
            # --- normal batch-size flush ---
            if len(batch) >= batch_size:
                futures.append(executor.submit(deepseek_batch, batch))
                batch = []

            # ---- drain futures when queue is full ----
            if len(futures) >= max_workers * 2:
                meta_batches = 0
                sum_latency = 0.0
                sum_tokens = 0
                sum_items  = 0

                for fut in as_completed(futures):
                    result = fut.result()
                    if result:
                        lat = result[0].get("_latency", 0.0)
                        toks = result[0].get("_tokens", 0)
                        bsz  = result[0].get("_batch", len(result))
                        meta_batches += 1
                        sum_latency += float(lat)
                        sum_tokens  += int(toks)
                        sum_items   += int(bsz)

                        for r in result:
                            cid = r["id"]
                            orig = next((b["text"] for b in batch if b["id"] == cid), "")
                            if "text" not in r or not r["text"]:
                                r["text"] = orig
                            r.pop("_latency", None)
                            r.pop("_tokens", None)
                            r.pop("_batch", None)

                        writer_q.put(result)
                        for r in result:
                            cache[hash_text(r["text"])] = r.get("language", "")
                        total += len(result)
                        success = True
                    else:
                        success = False

                    # --- tune dynamically ---
                    elapsed = time.time() - start_time
                    rate = total / elapsed if elapsed > 0 else 0.0
                    avg_latency = (sum_latency / meta_batches) if meta_batches else 0.0
                    token_rate = (sum_tokens / elapsed) if elapsed > 0 else 0.0
                    tuner.adjust(success, rate=rate, total=total, token_rate=token_rate, avg_latency=avg_latency)

                futures.clear()
                max_workers, batch_size = tuner.get_params()

                # Progress line
                elapsed = time.time() - start_time
                rate = total / elapsed if elapsed > 0 else 0.0
                avg_latency = (sum_latency / meta_batches) if meta_batches else 0.0
                token_rate = (sum_tokens / elapsed) if elapsed > 0 else 0.0
                eta = (total_lines - total) / rate / 3600 if rate > 0 else 0.0
                logging.info(
                    f"✅ Progress: {total:,}/{total_lines:,} processed "
                    f"({rate:.1f}/s, {int(token_rate)} tok/s, avg_lat {avg_latency:.2f}s) "
                    f"| ETA ≈ {eta:.2f}h | batch={batch_size}, workers={max_workers}"
                )
                # --- short-window token/s tracker ---
                if 'last_tokens' not in locals():
                    last_tokens, last_elapsed = 0, 0.0
                window_tokens = sum_tokens - last_tokens
                window_time = elapsed - last_elapsed
                if window_time > 0:
                    recent_tok_s = window_tokens / window_time
                    logging.info(f"⚡ Recent token/s (last window): {recent_tok_s:.0f}")
                last_tokens, last_elapsed = sum_tokens, elapsed

    # ---- tail batch ----
    if batch:
        result = deepseek_batch(batch)
        if result:
            for r in result:
                cid = r["id"]
                orig = next((b["text"] for b in batch if b["id"] == cid), "")
                if "text" not in r or not r["text"]:
                    r["text"] = orig
                r.pop("_latency", None)
                r.pop("_tokens", None)
                r.pop("_batch", None)
            writer_q.put(result)
            total += len(result)

    writer_q.put(None)
    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0.0
    logging.info(f"✅ Done {total:,} new comments in {elapsed/3600:.2f}h ({rate:.1f}/s)")

    # ---- split to final files ----
    counts = {"Swahili": 0, "English and Swahili": 0, "Sheng": 0, "English": 0, "Unknown": 0}
    out_files = {
        "Swahili": open(OUTPUT_DIR / "swahili.jsonl", "w", encoding="utf-8"),
        "English and Swahili": open(OUTPUT_DIR / "mixed.jsonl", "w", encoding="utf-8"),
        "Sheng":   open(OUTPUT_DIR / "sheng.jsonl",   "w", encoding="utf-8"),
        "English": open(OUTPUT_DIR / "english.jsonl", "w", encoding="utf-8"),
    }

    temp_files = sorted(OUTPUT_DIR.glob("temp_results_part_*.jsonl"))
    for part in temp_files:
        with open(part, "r", encoding="utf-8") as fpart:
            for line in fpart:
                try:
                    rec = json.loads(line)
                    lang = rec.get("language") or rec.get("Language") or rec.get("label") or "Unknown"
                    counts[lang] = counts.get(lang, 0) + 1
                    if lang in out_files:
                        out_files[lang].write(line)
                except Exception:
                    continue

    for fobj in out_files.values():
        fobj.close()

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


