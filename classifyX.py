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
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, deque, OrderedDict

# ---------------- CONFIG ----------------
INPUT_PATH = "kenya_clean_utf8.jsonl"   # .jsonl (default) or JSON array; .gz also supported
OUTPUT_DIR = Path("lang_split_out")
TEMP_PREFIX = OUTPUT_DIR / "temp_results_part"
CHECKPOINT_FILE = OUTPUT_DIR / "completed_ids.txt"
LOG_FILE = OUTPUT_DIR / "classifier.log"

MODEL = "deepseek-chat"
MAX_WORKERS = 12                 # conservative but fast
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
HEURISTIC_DIRECT_CONFIDENCE = 0.86
HEURISTIC_DIRECT_MAX_RATE = 0.25
HEURISTIC_DIRECT_RECHECK_GATE = 0.32
HEURISTIC_HINT_MIN_CONFIDENCE = 0.25
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
SWAHILI_COMMON = set(
    """
    na ya kwa ni sana bado bila kama ndio haya wewe sisi wapi leo jana kesho hivyo yake yetu wao wao
    mimi wewe yeye huku pale ndani nje sasa pia kutu kitu vitu wakati wakati mwingine kwa sababu shukrani
    watu mungu nchi wakati kila baada yake kama vile kutoka kwamba kupata kuenda maana mbele karibu
    """.split()
)

ENGLISH_COMMON = set(
    """
    the and you your from have this that what when where will they them then here there with about much people
    which because would could should know just like into only over very even never still those being after before
    """.split()
)

SHENG_HINTS = set(
    """
    ndo msee buda wasee niko aje nikoarea nashida manze manzee manzee dem mresh sherehe mtaa kejani nduthi
    mbogi form ng'ara mafta gari jobo wacha bro brathe sheng mtoi tao keja madem mabeste omosh
    """.split()
)

SHENG_STRONG_RE = re.compile(r"manz+e|msee|nduthi|mbogi|mtaa|mresh")
SWAHILI_SUFFIX_RE = re.compile(r"(ni|wa|me|sha|ku|tu|ke)$")
SWAHILI_PREFIX_RE = re.compile(r"^(ku|kwa|wa|ni|si|hu|tu|m|u|a)")
SWAHILI_TENSE_RE = re.compile(r"^(ni|si|hu|wa|tu|m|u|a)?(na|me|ta|li|ku|ki|nge|sha)")
SWAHILI_SECONDARY_RE = re.compile(r"\b(ni|wa|me|sha|tuko|mimi|sisi|wetu)\b")
SWAHILI_CHAR_RE = re.compile(r"(ng'|ny|mb|nd|mw|ch|sh)")
ENGLISH_CUE_RE = re.compile(r"(th|sh|ing|tion|ough|ight|ment|ness|ever|est|ous)")
ENGLISH_SUFFIX_RE = re.compile(r"(ing|ers?|ed|tion|ments?|ness|ity|ous|able|ively|less)$")
DOUBLE_VOWEL_RE = re.compile(r"([aeiou])\1")
SHENG_EXTRA_RE = re.compile(r"(maze|brathe|msoo|msupa|demi|kuflip|op|wuod)")

BASE_HEURISTIC_THRESHOLDS = {
    "sheng_ratio": 0.20,
    "swahili_ratio": 0.24,
    "english_ratio": 0.55,
    "mixed_low": 0.16,
    "mixed_gate": 0.30,
    "english_ascii": 0.78,
}


@dataclass
class HeuristicDecision:
    label: str
    confidence: float
    scores: dict
    features: dict
    reason: str


class AdaptiveHeuristicTuner:
    def __init__(self, base_thresholds, base_recheck_prob):
        self.base_thresholds = dict(base_thresholds)
        self.thresholds = dict(base_thresholds)
        self.recheck_prob = base_recheck_prob
        self.window = deque(maxlen=25)
        self.conf_window = deque(maxlen=25)
        self.min_samples = 15
        self.last_adjust = 0.0

    def record_batch(self, matches: int, mismatches: int, avg_confidence: Optional[float] = None):
        total = matches + mismatches
        if total <= 0:
            return
        self.window.append((mismatches, total))
        if avg_confidence is not None:
            self.conf_window.append(avg_confidence)
            logging.info(f"🔍 Heuristic sample avg_conf={avg_confidence:.2f}")
        rate = mismatches / total
        logging.info(
            f"🔁 Heuristic sample confusion: mismatch_rate={rate:.1%} ({mismatches}/{total})"
        )
        self._maybe_adjust()

    def should_recheck(self) -> bool:
        return random.random() < self.recheck_prob

    def _window_totals(self):
        mism = sum(m for m, _ in self.window)
        total = sum(t for _, t in self.window)
        return mism, total

    def _window_confidence(self):
        if not self.conf_window:
            return None
        return sum(self.conf_window) / len(self.conf_window)

    def _maybe_adjust(self):
        mism, total = self._window_totals()
        if total < self.min_samples:
            return
        rate = (mism / total) if total else 0.0
        now = time.time()
        if rate >= 0.15 and now - self.last_adjust > 60:
            self._tighten(rate)
        elif rate <= 0.05 and now - self.last_adjust > 120:
            self._loosen(rate)

    def _tighten(self, rate: float):
        conf = self._window_confidence() or 0.0
        self.thresholds["sheng_ratio"] = min(
            self.thresholds["sheng_ratio"] + 0.02, self.base_thresholds["sheng_ratio"] + 0.18
        )
        self.thresholds["swahili_ratio"] = min(
            self.thresholds["swahili_ratio"] + 0.02, self.base_thresholds["swahili_ratio"] + 0.18
        )
        self.thresholds["english_ratio"] = min(
            self.thresholds["english_ratio"] + 0.03, 0.92
        )
        self.thresholds["mixed_gate"] = min(
            self.thresholds["mixed_gate"] + 0.02, 0.52
        )
        prob_step = 0.02 if conf < 0.45 else 0.04
        self.recheck_prob = min(self.recheck_prob + prob_step, 0.5)
        self.last_adjust = time.time()
        logging.warning(
            "⚠️ Heuristic drift high (rate={:.1%}, avg_conf={:.2f}); tightening thresholds and increasing recheck sampling to {:.0%}.".format(
                rate, conf, self.recheck_prob
            )
        )

    def _loosen(self, rate: float):
        conf = self._window_confidence() or 0.0
        self.thresholds["sheng_ratio"] = max(
            self.base_thresholds["sheng_ratio"], self.thresholds["sheng_ratio"] - 0.01
        )
        self.thresholds["swahili_ratio"] = max(
            self.base_thresholds["swahili_ratio"], self.thresholds["swahili_ratio"] - 0.01
        )
        self.thresholds["english_ratio"] = max(
            self.base_thresholds["english_ratio"], self.thresholds["english_ratio"] - 0.01
        )
        self.thresholds["mixed_gate"] = max(
            self.base_thresholds["mixed_gate"], self.thresholds["mixed_gate"] - 0.01
        )
        decay = 0.01 if conf > 0.55 else 0.02
        self.recheck_prob = max(RECHECK_PROB_HEURISTIC, self.recheck_prob - decay)
        self.last_adjust = time.time()
        logging.info(
            "✅ Heuristic drift low (rate={:.1%}, avg_conf={:.2f}); relaxing thresholds and recheck sampling to {:.0%}.".format(
                rate, conf, self.recheck_prob
            )
        )


heuristic_tuner = AdaptiveHeuristicTuner(BASE_HEURISTIC_THRESHOLDS, RECHECK_PROB_HEURISTIC)


def _tokenize_for_heuristic(text: str):
    return re.findall(r"[a-zA-Z']+", text)


def heuristic_decision(text: str) -> Optional[HeuristicDecision]:
    t = (text or "").lower()
    if not t:
        return None

    letters = sum(ch.isalpha() for ch in t)
    if letters < 3:
        return None

    tokens = _tokenize_for_heuristic(t)
    if not tokens:
        return None

    total_tokens = len(tokens)
    if total_tokens < 2:
        return None

    unique_tokens = set(tokens)
    ascii_tokens = sum(1 for tok in tokens if tok.isascii())
    non_ascii_chars = sum(1 for ch in text if ord(ch) > 127)
    digit_chars = sum(1 for ch in text if ch.isdigit())

    sw_hits = sum(1 for w in unique_tokens if w in SWAHILI_COMMON)
    sh_hits = sum(1 for w in unique_tokens if w in SHENG_HINTS)
    en_hits = sum(1 for w in unique_tokens if w in ENGLISH_COMMON)

    sw_suffix_hits = sum(1 for tok in tokens if SWAHILI_SUFFIX_RE.search(tok))
    sw_prefix_hits = sum(1 for tok in tokens if SWAHILI_PREFIX_RE.match(tok))
    sw_tense_hits = sum(1 for tok in tokens if SWAHILI_TENSE_RE.match(tok))
    sw_char_hits = sum(1 for tok in tokens if SWAHILI_CHAR_RE.search(tok))
    sh_double_hits = sum(1 for tok in tokens if DOUBLE_VOWEL_RE.search(tok))
    sh_extra_hits = sum(1 for tok in tokens if SHENG_EXTRA_RE.search(tok))
    english_cues = sum(1 for tok in tokens if ENGLISH_CUE_RE.search(tok))
    english_suffix_hits = sum(1 for tok in tokens if ENGLISH_SUFFIX_RE.search(tok))

    strong_sheng = 1 if SHENG_STRONG_RE.search(t) else 0
    sw_secondary = 1 if SWAHILI_SECONDARY_RE.search(t) else 0

    sw_ratio = (
        sw_hits
        + 0.6 * sw_suffix_hits
        + 0.5 * sw_prefix_hits
        + 0.4 * sw_tense_hits
        + 0.3 * sw_char_hits
    ) / total_tokens
    if sw_secondary:
        sw_ratio += 0.04

    sh_ratio = (
        1.2 * sh_hits
        + 0.8 * sh_double_hits
        + 0.6 * sh_extra_hits
        + (1.8 if strong_sheng else 0.0)
    ) / total_tokens

    eng_ratio = (
        1.0 * en_hits
        + 0.7 * english_suffix_hits
        + 0.5 * english_cues
        + 0.35 * ascii_tokens
    ) / total_tokens

    ascii_ratio = ascii_tokens / total_tokens
    mixed_signal = min(sw_ratio, eng_ratio)

    features = {
        "total_tokens": total_tokens,
        "sw_ratio": sw_ratio,
        "sw_affix_ratio": (sw_suffix_hits + sw_prefix_hits + sw_tense_hits) / total_tokens,
        "sw_char_ratio": sw_char_hits / total_tokens,
        "sheng_ratio": sh_ratio,
        "sheng_double_ratio": sh_double_hits / total_tokens,
        "sheng_regex": strong_sheng,
        "english_ratio": eng_ratio,
        "english_suffix_ratio": english_suffix_hits / total_tokens,
        "english_char_ratio": english_cues / total_tokens,
        "ascii_ratio": ascii_ratio,
        "non_ascii_ratio": non_ascii_chars / max(1, len(text)),
        "digit_ratio": digit_chars / max(1, len(text)),
    }

    thresholds = heuristic_tuner.thresholds

    scores = {
        "Sheng": sh_ratio,
        "Swahili": sw_ratio,
        "English": eng_ratio,
        "English and Swahili": mixed_signal,
    }

    # Determine label with layered checks
    reason_parts = []
    label = None

    if strong_sheng and sh_ratio >= thresholds["sheng_ratio"] * 0.5:
        label = "Sheng"
        reason_parts.append("strong_sheng_regex")

    if label is None and sh_ratio >= thresholds["sheng_ratio"]:
        label = "Sheng"
        reason_parts.append(f"sheng_ratio={sh_ratio:.2f}")

    if label is None and sw_ratio >= thresholds["swahili_ratio"] and sh_ratio < thresholds["sheng_ratio"] * 0.85:
        label = "Swahili"
        reason_parts.append(f"sw_ratio={sw_ratio:.2f}")

    if label is None and eng_ratio >= thresholds["english_ratio"] and sw_ratio < thresholds["mixed_gate"]:
        label = "English"
        reason_parts.append(f"eng_ratio={eng_ratio:.2f}")

    if label is None and sw_ratio >= thresholds["mixed_low"] and eng_ratio >= thresholds["mixed_low"]:
        label = "English and Swahili"
        reason_parts.append(f"mixed sw={sw_ratio:.2f} en={eng_ratio:.2f}")

    if label is None and ascii_ratio >= thresholds["english_ascii"] and sw_hits == 0 and sh_hits == 0:
        label = "English"
        reason_parts.append("ascii_dominant")

    if label is None and sh_hits >= 1 and sh_ratio >= thresholds["sheng_ratio"] * 0.75:
        label = "Sheng"
        reason_parts.append("sheng_single_hit")

    # compute confidence & fallback heuristics when uncertain or noisy
    ordered_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_score = ordered_scores[0][1] if ordered_scores else 0.0
    second_score = ordered_scores[1][1] if len(ordered_scores) > 1 else 0.0
    margin = max(0.0, best_score - second_score)
    confidence = max(0.0, min(1.0, best_score + 0.5 * margin))

    if label is None and confidence >= 0.58:
        label = ordered_scores[0][0]
        reason_parts.append("auto_high_score")

    if label is not None and confidence < 0.18:
        # too shaky to trust
        label = None

    if label is not None and features["non_ascii_ratio"] > 0.12:
        label = None

    if label is not None and features["digit_ratio"] > 0.18:
        label = None

    degrade_gate = heuristic_tuner.recheck_prob >= 0.3
    if label is not None and degrade_gate and confidence < 0.45:
        label = None

    if label is None:
        return None

    reason = ",".join(reason_parts) if reason_parts else "signal"
    return HeuristicDecision(label=label, confidence=confidence, scores=scores, features=features, reason=reason)


def heuristic_label(text: str):
    decision = heuristic_decision(text)
    return decision.label if decision else None

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

    def add(self, item):
        t = estimate_tokens(item["text"])
        if (self.current and (self.current_tok + t > self.target_tokens or len(self.current) >= self.hard_cap)):
            full = self.current
            self.current = [item]
            self.current_tok = estimate_tokens(item["text"])
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
        heuristic_tuner.record_batch(
            int(heuristic_feedback_counts.get("match", 0)),
            int(heuristic_feedback_counts.get("mismatch", 0)),
            avg_conf,
        )
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
                heuristic_tuner.record_batch(h_match, h_miss, avg_conf)
        recheck_futures = remaining

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex, ThreadPoolExecutor(max_workers=RECHECK_MAX_WORKERS) as rex:
        for rec in stream:
            cid = rec.get("id") or rec.get("_id") or rec.get("cid")
            text = rec.get("body") or rec.get("text") or ""
            if not cid or not safe_text_check(text):
                continue

            # de-dup by content via sqlite cache
            h = md5_hash(text)
            cached = cache_get(h)
            if cached:
                writer_q.put([{"id": cid, "text": text, "language": cached}])
                total += 1
                if random.random() < RECHECK_PROB_CACHE:
                    schedule_recheck(rex, cid, text, cached, h, "cache")
                continue

            item = {"id": cid, "text": text}

            # heuristic suggestion (prefer DeepSeek for most cases)
            decision = heuristic_decision(text)
            if decision:
                guess = decision.label
                heuristic_usage["total"] += 1
                heuristic_usage[guess] += 1
                heuristic_confidence_accum += decision.confidence
                if decision.confidence >= 0.65:
                    conf_bucket = "high"
                elif decision.confidence >= 0.42:
                    conf_bucket = "mid"
                else:
                    conf_bucket = "low"
                heuristic_usage[f"confidence_{conf_bucket}"] += 1

                feature_keys = (
                    "sw_ratio",
                    "sheng_ratio",
                    "english_ratio",
                    "ascii_ratio",
                    "sw_affix_ratio",
                    "english_suffix_ratio",
                )
                diag_features = {
                    k: round(float(decision.features.get(k, 0.0)), 3) for k in feature_keys
                }
                diag = {
                    "confidence": round(float(decision.confidence), 4),
                    "bucket": conf_bucket,
                    "reason": decision.reason,
                    "scores": {k: round(float(v), 3) for k, v in decision.scores.items()},
                    "features": diag_features,
                }

                direct_eligible = decision.confidence >= HEURISTIC_DIRECT_CONFIDENCE
                took_direct = False
                if direct_eligible and heuristic_tuner.recheck_prob <= HEURISTIC_DIRECT_RECHECK_GATE:
                    processed_so_far = max(1, total + 1)
                    projected_ratio = (
                        (heuristic_usage.get("direct_total", 0) + 1)
                        / processed_so_far
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
                        trigger_recheck = heuristic_tuner.should_recheck() or decision.confidence < 0.9
                        if trigger_recheck:
                            schedule_recheck(rex, cid, text, guess, h, "heuristic", diag)
                        took_direct = True
                        continue
                    else:
                        heuristic_usage["direct_limited"] += 1
                elif direct_eligible:
                    heuristic_usage["direct_blocked"] += 1

                if not took_direct:
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
            ready = batcher.add(item)
            if ready:
                submit_batch(ex, ready)
                time.sleep(SLEEP_BETWEEN_BATCHES)

            # drain when futures pile up
            if len(futures) >= MAX_WORKERS * 2:
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

        # tail batch
        tail = batcher.flush()
        if tail:
            submit_batch(ex, tail)

        # drain all
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
            "⚙️ Final heuristic thresholds: "
            + ", ".join(
                f"{k}={v:.2f}" for k, v in sorted(heuristic_tuner.thresholds.items())
            )
            + f", recheck_prob={heuristic_tuner.recheck_prob:.2f}"
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
