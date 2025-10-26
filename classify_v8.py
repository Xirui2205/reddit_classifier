import json
import time
import random
import logging
import hashlib
import gzip
import itertools
import openai
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Thread

# ---------------- SKIP COUNTERS ----------------
from collections import Counter
skip_stats = Counter()

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
SKIPPED_FILE = OUTPUT_DIR / "skipped.jsonl"

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

def iter_input_lines(path: Path):
    """Yield non-empty lines from the input file, handling gzip transparently."""
    path = Path(path)
    if not path.exists():
        logging.error(f"Input file not found: {path}")
        return

    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield line
    except OSError as exc:
        logging.error(f"Failed to read input file {path}: {exc}")


def validate_checkpoint():
    """Safely handle resume logic — detect stale or incomplete checkpoints."""
    if not CHECKPOINT_FILE.exists():
        logging.info("No checkpoint file found — starting fresh.")
        return set()

    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as fh:
            done = {line.strip() for line in fh if line.strip()}
        total_lines = sum(1 for _ in iter_input_lines(INPUT_PATH))
        pct = (len(done) / total_lines) * 100 if total_lines else 0

        logging.info(f"Resuming with {len(done):,}/{total_lines:,} ({pct:.1f}%) comments already processed.")

        # ✅ If checkpoint seems too large or dataset changed → force partial restart
        if total_lines and len(done) >= total_lines:
            logging.warning("⚠️ Checkpoint indicates all lines processed — forcing full recheck of unclassified records.")
            return set()  # treat as fresh start, but keep old outputs

        # ✅ If checkpoint covers >90% but not all, verify outputs actually exist
        if pct > 90:
            logging.info("Checkpoint covers >90%, verifying unprocessed records.")
        return done

    except Exception as e:
        logging.warning(f"⚠️ Checkpoint validation failed ({e}) — starting from scratch.")
        return set()


def get_unprocessed_lines(path: Path, completed_ids):
    """Stream lines that still need processing, skipping ones already in the checkpoint."""
    for line in iter_input_lines(path):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skip_stats["malformed_json"] += 1
            continue

        cid = rec.get("id") or rec.get("comment_id") or rec.get("post_id") or ""
        cid = str(cid).strip()
        if cid and cid in completed_ids:
            continue

        yield line


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
                with open(CHECKPOINT_FILE, "a", encoding="utf-8") as ckp:
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
    """Send a batch to DeepSeek safely.
       Handles:
         • Full comment text (no truncation)
         • Rate limits and retries
         • Partial responses (retry once, then halve)
         • Content filter rejections (skips flagged comments)
         • Returns empty list gracefully if nothing parsed
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
                max_tokens=1500,
            )
            txt = r["choices"][0]["message"]["content"].strip()

            # --- try to parse model output ---
            try:
                out = json.loads(txt)
            except json.JSONDecodeError:
                s, e = txt.find("["), txt.rfind("]")
                if s != -1 and e != -1:
                    out = json.loads(txt[s:e + 1])
                else:
                    raise

            # --- attach meta info for tuner ---
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
                            max_tokens=1500,
                        )
                        txt_single = r["choices"][0]["message"]["content"].strip()
                        try:
                            parsed = json.loads(txt_single)
                            safe_results.extend(parsed)
                        except Exception as inner_parse_err:
                            logging.warning(f"🧩 Skip one comment: parse error {inner_parse_err}")
                            continue
                    except Exception as inner_e:
                        logging.warning(f"🧩 Skipping one comment due to filter: {inner_e}")
                        continue

                if safe_results:
                    return safe_results

                logging.warning("🧩 All comments in batch skipped due to content filter.")
                return []

            # --- rate limit handling ---
            if "429" in msg or "rate limit" in msg.lower():
                wait = 62 + random.uniform(0, 5)
                logging.warning(f"🚫 Rate limit hit (429); pausing {wait:.1f}s to reset window")
                time.sleep(wait)
                continue

            # --- handle partial responses ---
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

            # --- generic transient errors ---
            logging.warning(f"DeepSeek error (attempt {attempt+1}/{RETRIES}): {msg}")
            time.sleep(2 + attempt)
            continue

    # --- if all retries exhausted ---
    logging.warning(f"DeepSeek error: retries exhausted. Last error: {last_err_msg}")
    return []


def main():
    start_time = time.time()
    total = 0
    cumulative_tokens = 0
    batch, futures = [], []
    synthetic_counter = 0

    # ✅ Load and validate checkpoint
    done_ids = validate_checkpoint()
    tuner = AutoTuner(MAX_WORKERS, BATCH_SIZE)
    max_workers, batch_size = tuner.get_params()
    total_lines = sum(1 for _ in iter_input_lines(INPUT_PATH))

    line_iter = get_unprocessed_lines(INPUT_PATH, done_ids)
    try:
        first_line = next(line_iter)
    except StopIteration:
        logging.info("All records already processed — exiting cleanly.")
        return
    unprocessed = itertools.chain([first_line], line_iter)

    remaining_estimate = max(total_lines - len(done_ids), 0)
    logging.info(
        f"Starting classification on ≈{remaining_estimate:,} remaining comments out of {total_lines:,} total."
    )

    from collections import deque
    token_history = deque()

    pending_batches = {}

    def mark_skipped(records, reason):
        if not records:
            return 0
        reason_key = f"skip_{reason}"
        skip_stats[reason_key] += len(records)
        logging.warning(
            f"🧩 Skipping {len(records)} comment(s) due to {reason}."
        )
        with open(CHECKPOINT_FILE, "a", encoding="utf-8") as ckp, open(
            SKIPPED_FILE, "a", encoding="utf-8"
        ) as sf:
            for item in records:
                cid = str(item.get("id", "")).strip() or str(item.get("comment_id", "")).strip()
                text = item.get("text", "")
                if cid:
                    done_ids.add(cid)
                    ckp.write(cid + "\n")
                sf.write(
                    json.dumps(
                        {
                            "id": cid or item.get("id", ""),
                            "text": text,
                            "reason": reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return len(records)

    def drain_completed_futures():
        nonlocal futures, pending_batches, total, max_workers, batch_size, token_history, cumulative_tokens
        if not futures:
            return

        meta_batches = 0
        sum_latency = 0.0
        for fut in as_completed(list(futures)):
            try:
                result = fut.result()
            except Exception as e:
                logging.warning(f"⚠️ Worker future raised exception: {e}")
                result = []

            original_batch = pending_batches.pop(fut, [])

            # --- handle empty/failed batch ---
            if not result:
                skip_stats["empty_batch"] += 1
                logging.warning("⚠️ Empty batch returned — retrying in smaller chunks.")
                failed_records = []
                try:
                    for i in range(0, len(original_batch), 10):
                        sub = original_batch[i:i+10]
                        sub_result = deepseek_batch(sub)
                        if sub_result:
                            sub_map = {item["id"]: item["text"] for item in sub}
                            for r in sub_result:
                                cid = r.get("id")
                                if "text" not in r or not r["text"]:
                                    r["text"] = sub_map.get(cid, "")
                                r.pop("_latency", None)
                                r.pop("_tokens", None)
                                r.pop("_batch", None)
                            writer_q.put(sub_result)
                            for r in sub_result:
                                text_val = r.get("text", "")
                                if text_val:
                                    cache[hash_text(text_val)] = r.get("language", "")
                                rid = str(r.get("id", "")).strip()
                                if rid:
                                    done_ids.add(rid)
                            total += len(sub_result)
                        else:
                            failed_records.extend(sub)
                    continue
                except Exception as retry_err:
                    logging.warning(f"⚠️ Retry of empty batch failed: {retry_err}")
                    failed_records = list(original_batch)
                if failed_records:
                    total += mark_skipped(failed_records, "deepseek_failure")
                continue

            # --- normal successful batch handling ---
            lat = result[0].get("_latency", 0.0)
            toks = result[0].get("_tokens", 0)
            meta_batches += 1
            sum_latency += float(lat)
            cumulative_tokens += int(toks)

            id_to_text = {item["id"]: item["text"] for item in original_batch}

            for r in result:
                cid = r["id"]
                orig = id_to_text.get(cid, "")
                if "text" not in r or not r["text"]:
                    r["text"] = orig
                r.pop("_latency", None)
                r.pop("_tokens", None)
                r.pop("_batch", None)

            writer_q.put(result)
            for r in result:
                cache[hash_text(r["text"])] = r.get("language", "")
                rid = str(r.get("id", "")).strip()
                if rid:
                    done_ids.add(rid)
            total += len(result)

            # --- tune dynamically ---
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0.0
            avg_latency = (sum_latency / meta_batches) if meta_batches else 0.0
            token_rate = (cumulative_tokens / elapsed) if elapsed > 0 else 0.0
            tuner.adjust(True, rate=rate, total=total,
                         token_rate=token_rate, avg_latency=avg_latency)

        futures.clear()
        max_workers, batch_size = tuner.get_params()

        # --- progress logging ---
        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed > 0 else 0.0
        avg_latency = (sum_latency / meta_batches) if meta_batches else 0.0
        token_rate = (cumulative_tokens / elapsed) if elapsed > 0 else 0.0
        eta = (total_lines - total) / rate / 3600 if rate > 0 else 0.0
        logging.info(
            f"✅ Progress: {total:,}/{total_lines:,} processed "
            f"({rate:.1f}/s, {int(token_rate)} tok/s, avg_lat {avg_latency:.2f}s) "
            f"| ETA ≈ {eta:.2f}h | batch={batch_size}, workers={max_workers}"
        )

        # --- 30-min moving-average token/s ---
        now_ts = time.time()
        token_history.append((now_ts, cumulative_tokens))
        while token_history and now_ts - token_history[0][0] > 1800:
            token_history.popleft()
        if len(token_history) > 1:
            old_time, old_tokens = token_history[0]
            window_tokens = cumulative_tokens - old_tokens
            window_time = now_ts - old_time
            if window_time > 0:
                moving_avg_tok_s = window_tokens / window_time
                logging.info(f"⚡ 30-min moving-avg token/s: {moving_avg_tok_s:,.0f}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for line in unprocessed:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = rec.get("id") or f"auto_{synthetic_counter + 1}"
            synthetic_counter += 1
            text = rec.get("body") or rec.get("text") or ""
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

            # --- flush batch early if token count too high ---
            if sum(estimate_tokens(b["text"]) for b in batch) > 7000:
                payload = [dict(item) for item in batch]
                fut = executor.submit(deepseek_batch, payload)
                futures.append(fut)
                pending_batches[fut] = payload
                batch = []

            # --- normal batch-size flush ---
            if len(batch) >= batch_size:
                payload = [dict(item) for item in batch]
                fut = executor.submit(deepseek_batch, payload)
                futures.append(fut)
                pending_batches[fut] = payload
                batch = []

            # ---- drain futures when queue is full ----
            if len(futures) >= max_workers * 2:
                drain_completed_futures()

        # Drain any remaining futures after input exhaustion
        drain_completed_futures()

    # ---- tail batch ----
    if batch:
        result = deepseek_batch(batch)
        if result:
            id_to_text = {item["id"]: item["text"] for item in batch}
            for r in result:
                cid = r["id"]
                orig = id_to_text.get(cid, "")
                if "text" not in r or not r["text"]:
                    r["text"] = orig
                r.pop("_latency", None)
                r.pop("_tokens", None)
                r.pop("_batch", None)
            writer_q.put(result)
            for r in result:
                rid = str(r.get("id", "")).strip()
                if rid:
                    done_ids.add(rid)
            total += len(result)
        else:
            total += mark_skipped(batch, "deepseek_failure")

    writer_q.put(None)
    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0.0
    logging.info(f"✅ Done {total:,} new comments in {elapsed/3600:.2f}h ({rate:.1f}/s)")

    # ---- split to final files ----
    counts = {"Swahili": 0, "English and Swahili": 0,
              "Sheng": 0, "English": 0, "Unknown": 0}
    out_files = {
        "Swahili": open(OUTPUT_DIR / "swahili.jsonl", "w", encoding="utf-8"),
        "English and Swahili": open(OUTPUT_DIR / "mixed.jsonl", "w", encoding="utf-8"),
        "Sheng": open(OUTPUT_DIR / "sheng.jsonl", "w", encoding="utf-8"),
        "English": open(OUTPUT_DIR / "english.jsonl", "w", encoding="utf-8"),
    }

    temp_files = sorted(OUTPUT_DIR.glob("temp_results_part_*.jsonl"))
    for part in temp_files:
        with open(part, "r", encoding="utf-8") as fpart:
            for line in fpart:
                try:
                    rec = json.loads(line)
                    lang = rec.get("language") or rec.get("Language") or rec.get("label") or "Unknown"
                    if lang in out_files:
                        counts[lang] = counts.get(lang, 0) + 1
                        out_files[lang].write(line)
                    else:
                        skip_stats["unexpected_label"] += 1
                except Exception:
                    continue

    for fobj in out_files.values():
        fobj.close()

    logging.info("📊 Summary:")
    for k, v in counts.items():
        logging.info(f"{k:<22}: {v}")
    logging.info(f"📁 Split files in: {OUTPUT_DIR.absolute()}")

    if skip_stats:
        logging.info("🧩 Skipped content summary:")
        for reason, count in skip_stats.items():
            logging.info(f"   {reason:<18}: {count:,}")
    else:
        logging.info("🧩 No comments were skipped due to filters or parse errors.")


if __name__ == "__main__":
    main()
