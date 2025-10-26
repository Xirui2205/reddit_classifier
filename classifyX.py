#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import re
import json
import time
import gzip
import math
import queue
import atexit
import random
import sqlite3
import logging
import hashlib
import argparse
import threading
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import Counter, deque, OrderedDict

try:
    import fasttext  # type: ignore
except Exception:
    fasttext = None

# ---------------- CONFIG ----------------
INPUT_PATH = "kenya_clean_utf8.jsonl"   # .jsonl (default) or JSON array; .gz also supported
OUTPUT_DIR = Path("lang_split_out")
TEMP_PREFIX = OUTPUT_DIR / "temp_results_part"
CHECKPOINT_FILE = OUTPUT_DIR / "completed_ids.txt"
LOG_FILE = OUTPUT_DIR / "classifier.log"

MODEL = "deepseek-chat"
MAX_WORKERS = 12                 # conservative but fast
PREP_WORKERS = max(2, (os.cpu_count() or 8))  # parallel pre-processing
PREP_QUEUE_MAX = PREP_WORKERS * 6
TARGET_BATCH_TOKENS = 2500       # sweet spot for stability/accuracy
BATCH_HARD_LIMIT = 15            # cap items in one call
RETRIES = 3
LONG_TEXT_LIMIT_TOKS = 600       # truncate ultra-long comments for stability
SLEEP_BETWEEN_BATCHES = 0.3
CHECKPOINT_FLUSH_N = 200         # buffer checkpoint writes
CACHE_DB = OUTPUT_DIR / "cache.sqlite3"

# --- Drift monitoring knobs ---
RECHECK_PROB_CACHE = 0.02        # fraction of cache hits to revalidate
RECHECK_PROB_HEURISTIC = 0.08    # fraction of heuristic hits to revalidate (base, adaptive)
RECHECK_BATCH_SIZE = 10          # batch size for background rechecks
RECHECK_MAX_WORKERS = 2          # low concurrency to avoid extra load

# --- Memory safety ---
ORIGINAL_MAP_LIMIT = 20_000      # cap stored originals for alignment

# --- Heuristic guardrails ---
HEURISTIC_DIRECT_CONFIDENCE = 0.82
HEURISTIC_DIRECT_MAX_RATE = 0.25
HEURISTIC_DIRECT_RECHECK_GATE = 0.32
HEURISTIC_HINT_MIN_CONFIDENCE = 0.45
HEURISTIC_FEEDBACK_FLUSH = 40

# ---------------- API KEY/BASE ----------------
import openai
openai.api_key = "sk-33a98b12f11f4300857e5ca93bf90e24"
openai.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").strip()

# ---------------- LOGGING ----------------
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

# ---------------- TOKENIZER ----------------
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def estimate_tokens(txt: str) -> int:
        try:
            return len(_enc.encode(txt))
        except Exception:
            return max(1, int(len(txt) / 4))
except Exception:
    def estimate_tokens(txt: str) -> int:
        # fallback heuristic
        wc = len((txt or "").split())
        return max(wc, int(len(txt) * 0.32))

# ---------------- SKIP COUNTERS ----------------
skip_stats = Counter()
heuristic_usage = Counter()
heuristic_confidence_accum = 0.0

# ---------------- SQLITE PERSISTENT CACHE ----------------
def init_cache_db(path: Path):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            h TEXT PRIMARY KEY,
            label TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

CACHE_CONN = init_cache_db(CACHE_DB)
CACHE_CUR = CACHE_CONN.cursor()
def cache_get(h: str):
    row = CACHE_CUR.execute("SELECT label FROM cache WHERE h=?", (h,)).fetchone()
    return row[0] if row else None

def cache_set_many(pairs):
    # pairs: list[(hash, label)]
    CACHE_CUR.executemany("INSERT OR REPLACE INTO cache(h,label) VALUES(?,?)", pairs)
    CACHE_CONN.commit()

def cache_delete(h: str):
    CACHE_CUR.execute("DELETE FROM cache WHERE h=?", (h,))
    CACHE_CONN.commit()

@atexit.register
def _close_cache():
    try:
        CACHE_CONN.commit()
        CACHE_CONN.close()
    except Exception:
        pass

# ---------------- WRITER (buffered) ----------------
writer_q = queue.Queue()
part_index = 1
lines_in_part = 0
ckpt_buffer = []
ckpt_lock = False

def _flush_checkpoint_buffer():
    global ckpt_buffer
    if not ckpt_buffer:
        return
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as ckp:
        ckp.write("\n".join(ckpt_buffer) + "\n")
    ckpt_buffer = []

def writer_thread():
    global part_index, lines_in_part, ckpt_buffer
    current_path = f"{TEMP_PREFIX}_{part_index:03d}.jsonl"
    while True:
        recs = writer_q.get()
        if recs is None:
            break
        with open(current_path, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                lines_in_part += 1
                ckpt_buffer.append(r["id"])
                if len(ckpt_buffer) >= CHECKPOINT_FLUSH_N:
                    _flush_checkpoint_buffer()

        if lines_in_part >= 100_000:
            lines_in_part = 0
            part_index += 1
            current_path = f"{TEMP_PREFIX}_{part_index:03d}.jsonl"
    _flush_checkpoint_buffer()
    logging.info("Writer thread finished.")

import threading
threading.Thread(target=writer_thread, daemon=True).start()

# ---------------- UTILITIES ----------------
def safe_text_check(text: str) -> bool:
    t = (text or "").strip()
    return len(t) >= 2

def md5_hash(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()

def _open_any(path: str):
    return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8") if str(path).endswith(".gz") else open(path, "r", encoding="utf-8")

def validate_checkpoint():
    if not CHECKPOINT_FILE.exists():
        logging.info("No checkpoint file found — starting fresh.")
        return set()
    try:
        done = set(open(CHECKPOINT_FILE, "r", encoding="utf-8").read().split())
        total_lines = sum(1 for _ in open(INPUT_PATH, "r", encoding="utf-8")) if os.path.exists(INPUT_PATH) else 0
        pct = (len(done) / total_lines) * 100 if total_lines else 0
        logging.info(f"Resuming with {len(done):,}/{total_lines:,} ({pct:.1f}%) comments already processed.")
        return done
    except Exception as e:
        logging.warning(f"Checkpoint read failed ({e}) — starting fresh.")
        return set()

def get_unprocessed_lines(path: str, already_done_ids: set):
    if not os.path.exists(path):
        logging.error(f"Input file not found: {path}")
        return []

    # sniff
    with _open_any(path) as f:
        head = f.read(4096)
    is_array = head.lstrip().startswith("[")
    del head

    def _yield_jsonl():
        with _open_any(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception as e:
                    skip_stats["input_parse_error"] += 1
                    logging.warning(f"Skipping malformed JSON line: {e}")
                    continue
                cid = rec.get("id") or rec.get("_id") or rec.get("cid")
                if cid and cid in already_done_ids:
                    continue
                yield rec

    def _yield_array():
        with _open_any(path) as f:
            try:
                arr = json.load(f)
            except Exception as e:
                logging.error(f"Failed to parse JSON array: {e}")
                skip_stats["input_parse_error"] += 1
                return
        for rec in arr:
            if not isinstance(rec, dict):
                continue
            cid = rec.get("id") or rec.get("_id") or rec.get("cid")
            if cid and cid in already_done_ids:
                continue
            yield rec

    return _yield_array() if is_array else _yield_jsonl()

# ---------------- HEURISTIC PRE-CLASSIFIER ----------------
# Fast pass to skip obvious cases (saves cost; keeps DeepSeek for ambiguous/Sheng)
FASTTEXT_MODEL_PATH = Path(os.getenv("FASTTEXT_MODEL_PATH", "lid.176.ftz"))
FASTTEXT_LABEL_MAP = {
    "__label__en": "English",
    "__label__en_us": "English",
    "__label__en_uk": "English",
    "__label__sw": "Swahili",
    "__label__sw_ke": "Swahili",
}
FASTTEXT_DIRECT_THRESHOLD = 0.82
FASTTEXT_HINT_THRESHOLD = 0.45
FASTTEXT_MIN_PROBABILITY = 0.30
FASTTEXT_MIN_CHAR_COUNT = 12

SHENG_REGEX = re.compile(r"manz+e|msee|nduthi|mbogi|mtaa|mresh|maze|brathe|msoo|msupa|demi|op|wuod", re.IGNORECASE)
SWAHILI_SUFFIX_REGEX = re.compile(r"\b(ni|wa|me|sha)\b", re.IGNORECASE)
ENGLISH_CUES_REGEX = re.compile(r"(th|sh|ing)", re.IGNORECASE)
DOUBLE_VOWEL_REGEX = re.compile(r"([aeiou])\1", re.IGNORECASE)


@dataclass
class HeuristicDecision:
    label: str
    confidence: float
    scores: dict
    features: dict
    reason: str


@dataclass
class PreparedItem:
    id: str
    text: str
    token_count: int
    decision: Optional[HeuristicDecision]


class HeuristicMonitor:
    def __init__(self, base_recheck_prob: float):
        self.base_recheck_prob = base_recheck_prob
        self.recheck_prob = base_recheck_prob
        self.window = deque(maxlen=20)

    def record_batch(self, matches: int, mismatches: int):
        total = matches + mismatches
        if total <= 0:
            return
        rate = mismatches / total
        self.window.append(rate)
        if len(self.window) >= 5:
            avg_rate = sum(self.window) / len(self.window)
            if avg_rate > 0.15:
                self.recheck_prob = min(0.5, self.recheck_prob + 0.05)
            elif avg_rate < 0.05:
                self.recheck_prob = max(self.base_recheck_prob, self.recheck_prob - 0.02)

    def should_recheck(self) -> bool:
        return random.random() < self.recheck_prob


heuristic_monitor = HeuristicMonitor(RECHECK_PROB_HEURISTIC)

_fasttext_model = None
_fasttext_lock = threading.Lock()


def _load_fasttext_model():
    global _fasttext_model
    if fasttext is None:
        return None
    with _fasttext_lock:
        if _fasttext_model is None:
            if not FASTTEXT_MODEL_PATH.exists():
                logging.warning(
                    f"FastText model not found at {FASTTEXT_MODEL_PATH}; disabling heuristic guidance."
                )
                _fasttext_model = False
                return None
            try:
                _fasttext_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
            except Exception as exc:
                logging.warning(f"FastText load failed: {exc}")
                _fasttext_model = False
                return None
        if _fasttext_model is False:
            return None
    return _fasttext_model


def heuristic_decision(text: str) -> Optional[HeuristicDecision]:
    clean = (text or "").strip()
    if len(clean) < FASTTEXT_MIN_CHAR_COUNT:
        return None
    model = _load_fasttext_model()
    if model is None:
        return None

    try:
        labels, probs = model.predict(clean.replace("\n", " "), k=3)
    except Exception as exc:
        logging.warning(f"FastText prediction error: {exc}")
        return None

    scores = {}
    reasons = []

    for label, prob in zip(labels, probs):
        mapped = FASTTEXT_LABEL_MAP.get(label)
        if not mapped:
            continue
        scores[mapped] = max(scores.get(mapped, 0.0), float(prob))

    lower = clean.lower()
    sheng_signal = bool(SHENG_REGEX.search(lower))
    double_vowel_hits = len(DOUBLE_VOWEL_REGEX.findall(lower))
    sw_suffix_hits = len(SWAHILI_SUFFIX_REGEX.findall(lower))
    en_cues = len(ENGLISH_CUES_REGEX.findall(lower))

    if sheng_signal:
        scores["Sheng"] = max(scores.get("Sheng", 0.0), min(0.9, scores.get("Sheng", 0.0) + 0.25))
        reasons.append("sheng_regex")
    elif double_vowel_hits >= 2:
        scores["Sheng"] = max(scores.get("Sheng", 0.0), 0.55)
        reasons.append("double_vowel_hint")

    if sw_suffix_hits >= 2:
        scores["Swahili"] = max(scores.get("Swahili", 0.0), 0.55)
        reasons.append("swahili_suffix_hint")

    if en_cues >= 3 and scores.get("English", 0.0) < 0.5:
        scores["English"] = max(scores.get("English", 0.0), 0.5)
        reasons.append("english_cues")

    if scores.get("Swahili") and scores.get("English"):
        mix = min(scores["Swahili"], scores["English"]) + 0.05
        scores["English and Swahili"] = max(scores.get("English and Swahili", 0.0), mix)

    if not scores:
        return None

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = max(0.0, top_score - second_score)

    confidence = min(1.0, top_score + 0.3 * margin)

    if top_score < FASTTEXT_MIN_PROBABILITY or confidence < FASTTEXT_HINT_THRESHOLD:
        return None

    features = {
        "fasttext_top": top_score,
        "fasttext_second": second_score,
        "sheng_regex": int(sheng_signal),
        "double_vowel_hits": double_vowel_hits,
        "sw_suffix_hits": sw_suffix_hits,
        "english_cues": en_cues,
    }

    reason_str = ",".join(reasons) if reasons else "fasttext"

    return HeuristicDecision(top_label, confidence, scores, features, reason_str)


def heuristic_label(text: str):
    decision = heuristic_decision(text)
    return decision.label if decision else None


def prepare_item_worker(cid: str, text: str) -> PreparedItem:
    decision = heuristic_decision(text)
    token_count = estimate_tokens(text)
    return PreparedItem(id=cid, text=text, token_count=token_count, decision=decision)
# ---------------- PROMPTS (JSON-LINES format) ----------------
SYSTEM_PROMPT = (
    "You are an expert Kenyan linguist who classifies Reddit comments by language variety.\n"
    "Label definitions:\n"
    "- 'Swahili': standard Swahili vocabulary or grammar with minimal English mixing.\n"
    "- 'English and Swahili': clear code-mixing of both languages without heavy Sheng slang.\n"
    "- 'Sheng': Nairobi urban slang, youthful tone, or heavy Swahili-English blending with slangy spellings.\n"
    "- 'English': plain English (allowing occasional Swahili interjections that do not change the dominant language).\n"
    "Instructions:\n"
    "- Each input line is a JSON object with fields {id, text}; some also include an optional \"heuristic_hint\" {label, confidence, reason}.\n"
    "- Treat \"heuristic_hint\" only as soft context and always return your own judgment.\n"
    "- Respond with ONE JSON line {\"id\":...,\"language\":...}.\n"
    "- Preserve order, emit no explanations, markdown, or blank lines.\n"
    "- Always choose exactly one label. If text is slangy but intelligible, prefer 'Sheng'.\n"
    "- Treat obvious metadata, URLs, or emojis as neutral context and classify by the surrounding language.\n"
    "- When English and Swahili are balanced but formal, choose 'English and Swahili'; when slang dominates, choose 'Sheng'.\n"
)

def make_user_block(batch):
    # newline-separated JSON objects, each line an item
    # also truncate ultra-long by token count for stability
    lines = []
    for b in batch:
        txt = b["text"]
        if estimate_tokens(txt) > LONG_TEXT_LIMIT_TOKS:
            # conservative truncation at token/word boundary
            words = txt.split()
            words = words[: min(len(words), 2500)]
            txt = " ".join(words)
        payload = {"id": b["id"], "text": txt}
        hint = b.get("heuristic_hint") if isinstance(b, dict) else None
        if isinstance(hint, dict) and hint.get("label"):
            payload["heuristic_hint"] = hint
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)

# ---------------- DEEPSEEK CALL ----------------
def _parse_model_json_response(txt: str):
    """Best-effort parser that tolerates loose JSON lines/arrays."""
    results = []
    invalid_chunks = 0
    if not txt:
        return results, invalid_chunks

    def _consume_obj(obj):
        nonlocal invalid_chunks
        if isinstance(obj, dict):
            cid = obj.get("id")
            lang = obj.get("language")
            if cid and lang:
                results.append({"id": cid, "language": lang})
            else:
                invalid_chunks += 1
        elif isinstance(obj, list):
            for entry in obj:
                _consume_obj(entry)
        else:
            invalid_chunks += 1

    stripped = txt.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        _consume_obj(parsed)
        if results:
            return results, invalid_chunks

    # fallback: parse line by line / embedded json substrings
    for raw in stripped.splitlines():
        part = raw.strip()
        if not part:
            continue
        # try to extract balanced braces if the line has noise around JSON
        if not part.startswith("{"):
            start = part.find("{")
            end = part.rfind("}")
            if start != -1 and end != -1 and end > start:
                part = part[start: end + 1]
        if "[" in part and "]" in part and part.count("{") > 1:
            try:
                parsed = json.loads(part)
                _consume_obj(parsed)
                continue
            except Exception:
                invalid_chunks += 1
                continue
        try:
            parsed = json.loads(part)
        except Exception:
            invalid_chunks += 1
            continue
        _consume_obj(parsed)

    return results, invalid_chunks


def deepseek_batch(batch):
    """
    Stable micro-batch caller with:
      - strict token budgeting
      - JSON-lines prompt format
      - content filter fallback (per-item)
      - retry & 429 handling
    Returns list[ {id, language} ] (no text field)
    """
    time.sleep(random.uniform(0.02, 0.15))

    content = make_user_block(batch)
    last_err = ""

    for attempt in range(1, RETRIES + 1):
        start = time.time()
        try:
            r = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                max_tokens=1200,  # safe for <=15 outputs
            )
            txt = (r["choices"][0]["message"]["content"] or "").strip()
            out, invalid_chunks = _parse_model_json_response(txt)
            if invalid_chunks:
                logging.warning(f"Model output had {invalid_chunks} unparsable chunk(s); continuing with {len(out)} record(s).")
            # attach meta for tuning (not used here but kept)
            latency = time.time() - start
            for o in out:
                o["_latency"] = latency
            return out

        except Exception as e:
            msg = str(e)
            last_err = msg

            # content filter path → fall back to per-item
            if "Content Exists Risk" in msg or "invalid_request_error" in msg:
                logging.warning("Content filter hit — falling back to per-item calls.")
                results = []
                for b in batch:
                    try:
                        single = json.dumps({"id": b["id"], "text": b["text"]}, ensure_ascii=False)
                        r2 = openai.ChatCompletion.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": single},
                            ],
                            temperature=0,
                            max_tokens=200,
                        )
                        line = (r2["choices"][0]["message"]["content"] or "").strip()
                        parsed, invalid_chunks = _parse_model_json_response(line)
                        if invalid_chunks:
                            logging.warning(
                                f"Per-item response for {b['id']} had {invalid_chunks} unparsable chunk(s)."
                            )
                        for obj in parsed:
                            results.append(obj)
                    except Exception as ie:
                        logging.warning(f"Per-item skip due to filter/parse: {ie}")
                        continue
                return results

            # rate limit
            if "429" in msg or "rate limit" in msg.lower():
                wait = 62 + random.uniform(0, 4)
                logging.warning(f"429 rate limit — sleeping {wait:.1f}s")
                time.sleep(wait)
                continue

            logging.warning(f"DeepSeek error attempt {attempt}/{RETRIES}: {msg}")
            time.sleep(1.5 * attempt)

    logging.warning(f"DeepSeek batch failed after retries. Last error: {last_err}")
    return []

# ---------------- BATCHER ----------------
class TokenBudgetBatcher:
    def __init__(self, target_tokens=TARGET_BATCH_TOKENS, hard_cap=BATCH_HARD_LIMIT):
        self.target_tokens = target_tokens
        self.hard_cap = hard_cap
        self.current = []
        self.current_tok = 0

    def add(self, item, token_hint=None):
        t = token_hint if token_hint is not None else estimate_tokens(item["text"])
        if (self.current and (self.current_tok + t > self.target_tokens or len(self.current) >= self.hard_cap)):
            full = self.current
            self.current = [item]
            self.current_tok = token_hint if token_hint is not None else estimate_tokens(item["text"])
            return full
        else:
            self.current.append(item)
            self.current_tok += t
            return None

    def flush(self):
        if self.current:
            full = self.current
            self.current, self.current_tok = [], 0
            return full
        return None

# ---------------- MAIN ----------------
def split_to_final_files():
    counts = {"Swahili": 0, "English and Swahili": 0, "Sheng": 0, "English": 0, "Unknown": 0}
    outs = {
        "Swahili": open(OUTPUT_DIR / "swahili.jsonl", "w", encoding="utf-8"),
        "English and Swahili": open(OUTPUT_DIR / "mixed.jsonl", "w", encoding="utf-8"),
        "Sheng": open(OUTPUT_DIR / "sheng.jsonl", "w", encoding="utf-8"),
        "English": open(OUTPUT_DIR / "english.jsonl", "w", encoding="utf-8"),
    }

    try:
        for part in sorted(OUTPUT_DIR.glob("temp_results_part_*.jsonl")):
            with open(part, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    lang = rec.get("language") or rec.get("Language") or rec.get("label") or "Unknown"
                    if lang in outs:
                        outs[lang].write(line)
                        counts[lang] = counts.get(lang, 0) + 1
                    else:
                        counts["Unknown"] += 1
    finally:
        for fp in outs.values():
            fp.close()

    logging.info("📊 Final split summary:")
    for k, v in counts.items():
        logging.info(f"   {k:<22}: {v:,}")
    logging.info(f"📁 Split files in: {OUTPUT_DIR.absolute()}")

def main():
    global heuristic_confidence_accum
    start_time = time.time()
    done_ids = validate_checkpoint()
    stream = get_unprocessed_lines(INPUT_PATH, done_ids)
    if not stream:
        logging.info("Nothing to process — exiting.")
        return

    # count total for ETA
    try:
        total_lines = sum(1 for _ in open(INPUT_PATH, "r", encoding="utf-8"))
    except Exception:
        total_lines = 0

    logging.info("🚀 Starting classifier...")
    total = 0
    batcher = TokenBudgetBatcher()
    futures = []
    pending_batches = {}
    tok_hist = deque()
    cache_pairs = []  # for bulk sqlite insert
    recheck_buffer = []
    recheck_futures = []
    recheck_stats = Counter()
    recheck_seq = 0
    heuristic_pending = {}
    heuristic_feedback_counts = Counter()
    heuristic_feedback_conf_total = 0.0
    heuristic_feedback_conf_n = 0

    def flush_heuristic_feedback(force: bool = False):
        nonlocal heuristic_feedback_counts, heuristic_feedback_conf_total, heuristic_feedback_conf_n
        total_fb = heuristic_feedback_counts.get("match", 0) + heuristic_feedback_counts.get("mismatch", 0)
        if not force and total_fb < HEURISTIC_FEEDBACK_FLUSH:
            return
        if total_fb <= 0:
            return
        avg_conf = (
            heuristic_feedback_conf_total / heuristic_feedback_conf_n
            if heuristic_feedback_conf_n
            else None
        )
        matches = int(heuristic_feedback_counts.get("match", 0))
        mismatches = int(heuristic_feedback_counts.get("mismatch", 0))
        heuristic_monitor.record_batch(matches, mismatches)
        if avg_conf is not None:
            logging.info(f"🔍 Heuristic sample avg_conf={avg_conf:.2f}")
        heuristic_feedback_counts.clear()
        heuristic_feedback_conf_total = 0.0
        heuristic_feedback_conf_n = 0

    def submit_batch(executor, batch):
        if not batch:
            return
        fut = executor.submit(deepseek_batch, batch)
        futures.append(fut)
        # store a shallow copy so retries know which ids were expected
        pending_batches[fut] = [dict(item) for item in batch]

    def write_results(results, original_by_id, expected_lookup=None):
        nonlocal total, cache_pairs, heuristic_pending
        nonlocal heuristic_feedback_counts, heuristic_feedback_conf_total, heuristic_feedback_conf_n
        out = []
        processed_ids = set()
        for r in results:
            cid = r.get("id")
            lang = r.get("language")
            if not cid or not lang:
                continue
            text = original_by_id.pop(cid, None)
            if text is None and expected_lookup:
                text = expected_lookup.get(cid)
            if text is None:
                text = ""
            processed_ids.add(cid)
            out.append({"id": cid, "text": text, "language": lang})

            # persistent cache
            if text:
                cache_pairs.append((md5_hash(text), lang))

            pending_diag = heuristic_pending.pop(cid, None)
            if pending_diag:
                expected_label = pending_diag.get("expected")
                conf_val = pending_diag.get("confidence")
                if isinstance(conf_val, (int, float)):
                    heuristic_feedback_conf_total += float(conf_val)
                    heuristic_feedback_conf_n += 1
                if expected_label:
                    if lang == expected_label:
                        heuristic_feedback_counts["match"] += 1
                        heuristic_usage["hint_match"] += 1
                    else:
                        heuristic_feedback_counts["mismatch"] += 1
                        heuristic_usage["hint_mismatch"] += 1
                        msg = (
                            f"Heuristic hint disagreed with DeepSeek for {cid}: "
                            f"expected {expected_label}, got {lang}"
                        )
                        if isinstance(conf_val, (int, float)):
                            msg += f" (conf={float(conf_val):.2f})"
                        reason = pending_diag.get("reason")
                        if reason:
                            msg += f" reason={reason}"
                        logging.info(msg)
                flush_heuristic_feedback()

        if out:
            writer_q.put(out)
            total += len(out)

        if len(cache_pairs) >= 2000:
            cache_set_many(cache_pairs)
            cache_pairs = []

        return processed_ids

    def handle_missing_items(missing_items):
        """Retry items that failed to parse or were dropped."""
        for item in missing_items:
            cid = item.get("id")
            if not cid:
                continue
            text = original_map.get(cid) or item.get("text") or ""
            if not text:
                skip_stats["missing_text_for_retry"] += 1
                continue
            payload = {"id": cid, "text": text}
            pending_hint = heuristic_pending.get(cid)
            if pending_hint and pending_hint.get("expected"):
                payload["heuristic_hint"] = {
                    "label": pending_hint.get("expected"),
                    "confidence": round(float(pending_hint.get("confidence", 0.0)), 3),
                    "reason": pending_hint.get("reason"),
                }
            recovered = False
            for attempt in range(1, RETRIES + 1):
                single_result = deepseek_batch([payload])
                if single_result:
                    processed = write_results(single_result, original_map, {cid: text})
                    if cid in processed:
                        recovered = True
                        break
                sleep_for = min(5.0, 1.5 * attempt)
                time.sleep(sleep_for)
            if not recovered:
                skip_stats["unrecoverable_model_output"] += 1
                original_map.pop(cid, None)
                heuristic_pending.pop(cid, None)
                logging.error(
                    f"Failed to recover classification for {cid} after {RETRIES} retries; skipping."
                )

    # Maintain a sliding dict of originals for batch alignment
    original_map = OrderedDict()

    def dispatch_recheck(executor):
        nonlocal recheck_buffer
        if not recheck_buffer:
            return
        payload = [{"id": item["recheck_id"], "text": item["text"]} for item in recheck_buffer]
        meta = {item["recheck_id"]: item for item in recheck_buffer}
        future = executor.submit(deepseek_batch, payload)
        recheck_futures.append((future, meta))
        recheck_buffer = []

    def schedule_recheck(executor, item_id, text, expected, h, source, diagnostics=None):
        nonlocal recheck_seq
        recheck_seq += 1
        re_id = f"recheck::{item_id}::{recheck_seq}"
        recheck_buffer.append(
            {
                "recheck_id": re_id,
                "original_id": item_id,
                "text": text,
                "expected": expected,
                "hash": h,
                "source": source,
                "diagnostics": diagnostics or {},
            }
        )
        if len(recheck_buffer) >= RECHECK_BATCH_SIZE:
            dispatch_recheck(executor)

    def process_recheck_futures(done_only=True):
        nonlocal recheck_futures, cache_pairs
        remaining = []
        for fut, meta in recheck_futures:
            if done_only and not fut.done():
                remaining.append((fut, meta))
                continue
            if not fut.done():
                remaining.append((fut, meta))
                continue
            try:
                results = fut.result()
            except Exception as e:
                logging.warning(f"Recheck batch failed: {e}")
                continue
            seen = set()
            batch_counts = Counter()
            confidence_total = 0.0
            confidence_n = 0
            for obj in results or []:
                rid = obj.get("id")
                lang = obj.get("language")
                if not rid or rid not in meta:
                    continue
                item = meta[rid]
                seen.add(rid)
                expected = item.get("expected")
                if not lang:
                    continue
                diag = item.get("diagnostics") or {}
                bucket = diag.get("bucket")
                if bucket:
                    recheck_stats[f"{item['source']}_{bucket}_seen"] += 1
                conf_val = diag.get("confidence")
                if item["source"] == "heuristic" and isinstance(conf_val, (int, float)):
                    confidence_total += float(conf_val)
                    confidence_n += 1
                if lang != expected:
                    recheck_stats["mismatch"] += 1
                    recheck_stats[f"{item['source']}_mismatch"] += 1
                    batch_counts[f"{item['source']}_mismatch"] += 1
                    if bucket:
                        recheck_stats[f"{item['source']}_{bucket}_mismatch"] += 1
                    details = []
                    reason = diag.get("reason")
                    if reason:
                        details.append(f"reason={reason}")
                    if isinstance(conf_val, (int, float)):
                        details.append(f"conf={float(conf_val):.2f}")
                    scores_excerpt = diag.get("scores") or {}
                    if scores_excerpt:
                        top_scores = sorted(scores_excerpt.items(), key=lambda kv: kv[1], reverse=True)[:3]
                        details.append(
                            "scores=" + ",".join(f"{k}:{v:.2f}" for k, v in top_scores)
                        )
                    feat_excerpt = diag.get("features") or {}
                    if feat_excerpt:
                        top_feats = ",".join(
                            f"{k}:{feat_excerpt[k]:.2f}" for k in sorted(feat_excerpt)[:3]
                        )
                        details.append("features=" + top_feats)
                    msg = (
                        f"Recheck drift detected for {item['original_id']} ({item['source']}): "
                        f"expected {expected}, got {lang}."
                    )
                    if details:
                        msg += " [" + "; ".join(details) + "]"
                    logging.warning(msg)
                    h = item.get("hash")
                    if h:
                        cache_delete(h)
                        for idx, (hh, lbl) in enumerate(cache_pairs):
                            if hh == h:
                                cache_pairs[idx] = (hh, lang)
                                break
                        else:
                            cache_set_many([(h, lang)])
                else:
                    recheck_stats["match"] += 1
                    recheck_stats[f"{item['source']}_match"] += 1
                    batch_counts[f"{item['source']}_match"] += 1
                    if bucket:
                        recheck_stats[f"{item['source']}_{bucket}_match"] += 1
            missing = set(meta.keys()) - seen
            for rid in missing:
                item = meta[rid]
                diag = item.get("diagnostics") or {}
                bucket = diag.get("bucket")
                logging.warning(
                    f"Recheck returned no result for {item['original_id']} ({item['source']})."
                )
                recheck_stats["missing"] += 1
                if bucket:
                    recheck_stats[f"{item['source']}_{bucket}_missing"] += 1
            h_match = batch_counts.get("heuristic_match", 0)
            h_miss = batch_counts.get("heuristic_mismatch", 0)
            if h_match or h_miss:
                avg_conf = (confidence_total / confidence_n) if confidence_n else None
                heuristic_monitor.record_batch(h_match, h_miss)
                if avg_conf is not None:
                    logging.info(f"🔁 Recheck sample avg_conf={avg_conf:.2f}")
        recheck_futures = remaining

    prep_futures = deque()

    with ProcessPoolExecutor(max_workers=PREP_WORKERS) as prep_ex, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex, ThreadPoolExecutor(max_workers=RECHECK_MAX_WORKERS) as rex:
        def process_prepared(prepared: PreparedItem, meta: dict):
            nonlocal cache_pairs, total
            global heuristic_confidence_accum
            cid = prepared.id
            text = prepared.text
            h = meta["hash"]
            decision = prepared.decision
            item = {"id": cid, "text": text}
            diag = None
            took_direct = False

            if decision:
                guess = decision.label
                heuristic_usage["total"] += 1
                heuristic_usage[guess] += 1
                heuristic_confidence_accum += decision.confidence
                if decision.confidence >= 0.75:
                    conf_bucket = "high"
                elif decision.confidence >= 0.55:
                    conf_bucket = "mid"
                else:
                    conf_bucket = "low"
                heuristic_usage[f"confidence_{conf_bucket}"] += 1

                diag_features = {
                    k: round(float(decision.features.get(k, 0.0)), 3)
                    for k in (
                        "fasttext_top",
                        "fasttext_second",
                        "sheng_regex",
                        "double_vowel_hits",
                        "sw_suffix_hits",
                        "english_cues",
                    )
                }
                diag = {
                    "confidence": round(float(decision.confidence), 4),
                    "bucket": conf_bucket,
                    "reason": decision.reason,
                    "scores": {k: round(float(v), 3) for k, v in decision.scores.items()},
                    "features": diag_features,
                }

                direct_eligible = decision.confidence >= HEURISTIC_DIRECT_CONFIDENCE
                if direct_eligible and heuristic_monitor.recheck_prob <= HEURISTIC_DIRECT_RECHECK_GATE:
                    processed_so_far = max(1, total + 1)
                    projected_ratio = (
                        (heuristic_usage.get("direct_total", 0) + 1) / processed_so_far
                    )
                    if total < 20:
                        projected_ratio = 0.0
                    if projected_ratio <= HEURISTIC_DIRECT_MAX_RATE:
                        heuristic_usage["direct_total"] += 1
                        heuristic_usage[f"direct_{guess}"] += 1
                        writer_q.put([{"id": cid, "text": text, "language": guess}])
                        cache_pairs.append((h, guess))
                        total += 1
                        if len(cache_pairs) >= 2000:
                            cache_set_many(cache_pairs)
                            cache_pairs = []
                        trigger_recheck = heuristic_monitor.should_recheck() or decision.confidence < 0.9
                        if trigger_recheck:
                            schedule_recheck(rex, cid, text, guess, h, "heuristic", diag)
                        took_direct = True
                    else:
                        heuristic_usage["direct_limited"] += 1
                elif direct_eligible:
                    heuristic_usage["direct_blocked"] += 1

                if took_direct:
                    return

                if decision.confidence >= HEURISTIC_HINT_MIN_CONFIDENCE:
                    heuristic_usage["hint_total"] += 1
                    heuristic_pending[cid] = {
                        "expected": guess,
                        "confidence": float(decision.confidence),
                        "reason": decision.reason,
                        "scores": {k: float(v) for k, v in decision.scores.items()},
                        "features": diag_features,
                        "bucket": conf_bucket,
                    }
                    item["heuristic_hint"] = {
                        "label": guess,
                        "confidence": round(float(decision.confidence), 3),
                        "reason": decision.reason,
                    }
                else:
                    heuristic_usage["hint_suppressed"] += 1

            original_map[cid] = text
            original_map.move_to_end(cid)
            while len(original_map) > ORIGINAL_MAP_LIMIT:
                original_map.popitem(last=False)
            ready = batcher.add(item, token_hint=prepared.token_count)
            if ready:
                submit_batch(ex, ready)
                time.sleep(SLEEP_BETWEEN_BATCHES)

        def drain_prepared(force: bool = False):
            while prep_futures:
                fut, meta = prep_futures[0]
                if not (force or fut.done()):
                    break
                prep_futures.popleft()
                cid = meta["id"]
                text = meta["text"]
                try:
                    prepared = fut.result()
                except Exception as e:
                    logging.warning(f"Prep worker failure for {cid}: {e}")
                    prepared = PreparedItem(
                        id=cid,
                        text=text,
                        token_count=estimate_tokens(text),
                        decision=None,
                    )
                process_prepared(prepared, meta)

        for rec in stream:
            cid = rec.get("id") or rec.get("_id") or rec.get("cid")
            text = rec.get("body") or rec.get("text") or ""
            if not cid or not safe_text_check(text):
                continue

            h = md5_hash(text)
            cached = cache_get(h)
            if cached:
                writer_q.put([{"id": cid, "text": text, "language": cached}])
                total += 1
                if random.random() < RECHECK_PROB_CACHE:
                    schedule_recheck(rex, cid, text, cached, h, "cache")
                continue

            fut = prep_ex.submit(
                prepare_item_worker,
                cid,
                text,
            )
            prep_futures.append((fut, {"id": cid, "text": text, "hash": h}))
            drain_prepared()
            if len(prep_futures) >= PREP_QUEUE_MAX:
                drain_prepared(force=True)

            if len(futures) >= MAX_WORKERS * 2:
                drain_prepared(force=True)
                sum_latency = 0.0
                meta_batches = 0
                for fut in as_completed(list(futures)):
                    futures.remove(fut)
                    expected_batch = pending_batches.pop(fut, [])
                    expected_lookup = {
                        itm.get("id"): itm.get("text", "") for itm in expected_batch if itm.get("id")
                    }
                    try:
                        result = fut.result()
                    except Exception as e:
                        logging.warning(f"Worker exception: {e}")
                        result = []
                    processed_ids = set()
                    if result:
                        processed_ids = write_results(result, original_map, expected_lookup)
                        lat = result[0].get("_latency", 0.0)
                        sum_latency += float(lat)
                        meta_batches += 1
                    missing_items = [item for item in expected_batch if item.get("id") not in processed_ids]
                    if missing_items:
                        skip_stats["missing_from_batch"] += len(missing_items)
                        logging.warning(
                            f"Model response missing {len(missing_items)} item(s); retrying individually."
                        )
                        handle_missing_items(missing_items)
                pending_batches.clear()
                process_recheck_futures()

                elapsed = time.time() - start_time
                rate = total / elapsed if elapsed > 0 else 0.0
                avg_lat = (sum_latency / meta_batches) if meta_batches else 0.0
                logging.info(
                    f"✅ Progress: {total:,}/{total_lines:,} processed "
                    f"({rate:.1f}/s, avg_lat {avg_lat:.2f}s) | batch≈{BATCH_HARD_LIMIT}, workers={MAX_WORKERS}"
                )

        drain_prepared(force=True)

        tail = batcher.flush()
        if tail:
            submit_batch(ex, tail)

        for fut in as_completed(list(futures)):
            futures.remove(fut)
            expected_batch = pending_batches.pop(fut, [])
            expected_lookup = {
                itm.get("id"): itm.get("text", "") for itm in expected_batch if itm.get("id")
            }
            try:
                result = fut.result()
            except Exception as e:
                logging.warning(f"Worker exception (tail): {e}")
                result = []
            processed_ids = set()
            if result:
                processed_ids = write_results(result, original_map, expected_lookup)
            missing_items = [item for item in expected_batch if item.get("id") not in processed_ids]
            if missing_items:
                skip_stats["missing_from_batch"] += len(missing_items)
                logging.warning(
                    f"Tail drain missing {len(missing_items)} item(s); retrying individually."
                )
                handle_missing_items(missing_items)
        pending_batches.clear()

        dispatch_recheck(rex)
        process_recheck_futures(done_only=False)

    # process any remaining recheck futures after executors close
    if recheck_futures:
        process_recheck_futures(done_only=False)

    # final cache flush
    if cache_pairs:
        cache_set_many(cache_pairs)

    writer_q.put(None)
    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0.0
    logging.info(f"🏁 Done {total:,} new comments in {elapsed/3600:.2f}h ({rate:.1f}/s)")

    # split temps to final channel files
    split_to_final_files()

    flush_heuristic_feedback(force=True)
    if heuristic_pending:
        logging.warning(
            f"Clearing {len(heuristic_pending)} heuristic hint(s) without model confirmation."
        )
        heuristic_pending.clear()

    if skip_stats:
        logging.info(
            "⚠️ Skip stats: " + ", ".join(f"{k}={v}" for k, v in sorted(skip_stats.items()))
        )

    if recheck_stats:
        logging.info(
            "🔍 Recheck summary: "
            + ", ".join(f"{k}={v}" for k, v in sorted(recheck_stats.items()))
        )

    if heuristic_usage:
        logging.info(
            "🤖 Heuristic usage: "
            + ", ".join(f"{k}={v}" for k, v in sorted(heuristic_usage.items()))
        )
        total_h = heuristic_usage.get("total", 0)
        if total_h:
            avg_conf = heuristic_confidence_accum / total_h
            logging.info(f"📐 Heuristic avg confidence: {avg_conf:.2f}")
        logging.info(
            f"⚙️ Heuristic monitor recheck_prob={heuristic_monitor.recheck_prob:.2f}"
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
