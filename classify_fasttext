#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Language classifier for Kenyan text (English / Swahili / Sheng mix)
Hybrid approach:
 - FastText per-word language scoring (fast, CPU)
 - Heuristic slang detection for Sheng
 - DeepSeek model for ambiguous or mixed content
"""

import os, re, json, time, hashlib, logging, gzip
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import fasttext
import openai
from tqdm import tqdm

# ============================================================
# ---------------- CONFIGURATION -----------------------------
# ============================================================

INPUT_PATH = "kenya_clean_utf8.jsonl"
OUTPUT_PATH = "classified_out.jsonl"
MODEL_PATH = "lid.176.ftz"              # downloaded FastText model
OPENAI_MODEL = "deepseek-chat"
MAX_WORKERS = 8                         # use all CPU cores
BATCH_SIZE = 10                         # DeepSeek micro-batch
LONG_TEXT_LIMIT = 2500                  # words cap for stability
RETRIES = 3

openai.api_key = os.getenv("DEEPSEEK_API_KEY", "sk-xxxxxxxxxxxxxxxx")
openai.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").strip()

# ============================================================
# ---------------- LOGGING -----------------------------------
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("classifier")

# ============================================================
# ---------------- LOAD FASTTEXT -----------------------------
# ============================================================

logger.info("Loading FastText language model...")
FT = fasttext.load_model(MODEL_PATH)
logger.info("FastText model loaded.")

# ============================================================
# ---------------- WORDLISTS (Sheng & Swahili) ---------------
# ============================================================

SWAHILI_COMMON = set("""
na ya kwa ni sana bado bila kama ndio haya wewe sisi wapi leo jana kesho hivyo yake yetu wao wao
mimi wewe yeye huku pale ndani nje sasa pia kitu wakati mwingine sababu shukrani nimeona nimefanya
hakuna jambo mtoto kazi nyumba rafiki salamu asante shule mungu
""".split())

SHENG_HINTS = set("""
manze manzee buda wasee msee nikoaje nikoarea nduthi mtaa kejani mresh dem brathe bro
mbogi form ngara jobo bazuu nduthi kude kudea naje ule supu chuo ploti umsee madem mabeste
""".split())

# ============================================================
# ---------------- UTILITIES ---------------------------------
# ============================================================

def md5_hash(txt: str) -> str:
    return hashlib.md5(txt.encode("utf-8")).hexdigest()

def open_any(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

# ============================================================
# ---------------- FASTTEXT + HEURISTIC SCORE ----------------
# ============================================================

def fasttext_detect(word: str):
    """Return (lang, prob) for one token."""
    try:
        preds = FT.predict(word)
        lang = preds[0][0].replace("__label__", "")
        prob = preds[1][0]
        return lang, prob
    except Exception:
        return "unk", 0.0


def heuristic_score(text: str):
    """
    Token-level language ratios using FastText + Sheng lexicon.
    Returns a dict: {en_ratio, sw_ratio, sh_ratio, tokens:[(word,label)]}
    """
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    total = len(tokens) or 1
    en = sw = sh = 0
    tagged = []

    for w in tokens:
        if w in SHENG_HINTS:
            sh += 1
            tagged.append((w, "Sheng"))
            continue
        if w in SWAHILI_COMMON:
            sw += 1
            tagged.append((w, "Swahili"))
            continue

        lang, prob = fasttext_detect(w)
        if lang == "en" and prob > 0.7:
            en += 1
            tagged.append((w, "English"))
        elif lang == "sw" and prob > 0.7:
            sw += 1
            tagged.append((w, "Swahili"))
        else:
            tagged.append((w, "Unknown"))

    return {
        "en_ratio": en / total,
        "sw_ratio": sw / total,
        "sh_ratio": sh / total,
        "tokens": tagged
    }


def heuristic_label(text: str):
    """Decide if heuristic alone is confident enough to classify."""
    h = heuristic_score(text)
    en, sw, sh = h["en_ratio"], h["sw_ratio"], h["sh_ratio"]

    # strong Sheng signal
    if sh > 0.2 and sh > sw and sh > en:
        return "Sheng"
    # strong Swahili
    if sw > 0.7 and sw > sh:
        return "Swahili"
    # strong English
    if en > 0.8 and sw < 0.1:
        return "English"
    # moderate mix
    if 0.2 < sw < 0.6 and en > 0.2:
        return "English and Swahili"
    return None, h


# ============================================================
# ---------------- DEEPSEEK PROMPT ---------------------------
# ============================================================

SYSTEM_PROMPT = (
    "You are a linguist specializing in East African languages. "
    "Classify each JSON object into one of: Swahili | English and Swahili | Sheng | English. "
    "Each object contains the text and heuristic ratios. "
    "Return one JSON line per input, exactly: {\"id\": \"...\", \"language\": \"...\"}. "
    "No markdown, no commentary."
)

def make_user_block(batch):
    lines = []
    for b in batch:
        heur = heuristic_score(b["text"])
        hint = f"Heuristic ratios: Sw={heur['sw_ratio']:.2f}, Sh={heur['sh_ratio']:.2f}, En={heur['en_ratio']:.2f}"
        meta = {
            "id": b["id"],
            "text": b["text"][:1500],  # safe length
            "heuristic_hint": hint
        }
        lines.append(json.dumps(meta, ensure_ascii=False))
    return "\n".join(lines)


def parse_model_output(txt: str):
    """Robust JSON line parser."""
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "id" in obj and "language" in obj:
                out.append(obj)
        except Exception:
            # attempt to extract {...}
            m = re.search(r"\{.*?\}", line)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    if "id" in obj and "language" in obj:
                        out.append(obj)
                except Exception:
                    continue
    return out


def deepseek_batch(batch):
    """Call DeepSeek model for classification."""
    time.sleep(0.1)
    user_content = make_user_block(batch)
    for attempt in range(RETRIES):
        try:
            r = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=1000,
            )
            txt = r["choices"][0]["message"]["content"]
            return parse_model_output(txt)
        except Exception as e:
            msg = str(e)
            if "rate limit" in msg.lower():
                wait = 60
                logger.warning(f"Rate limit, sleeping {wait}s...")
                time.sleep(wait)
                continue
            logger.warning(f"DeepSeek error: {msg}")
            time.sleep(2)
    return []


# ============================================================
# ---------------- MAIN PIPELINE -----------------------------
# ============================================================

def main():
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Input not found: {INPUT_PATH}")
        return

    fout = open(OUTPUT_PATH, "w", encoding="utf-8")
    total, from_cache = 0, 0
    start = time.time()

    with open_any(INPUT_PATH) as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        buffer, futures = [], []

        for line in tqdm(f, desc="Reading"):
            try:
                rec = json.loads(line)
                cid = rec.get("id") or rec.get("_id") or rec.get("cid") or md5_hash(line)
                txt = rec.get("text") or rec.get("body") or ""
                if not txt.strip():
                    continue
                label = heuristic_label(txt)
                if isinstance(label, tuple):  # returned (None, heur)
                    label, heur = label
                if label:
                    out = {"id": cid, "language": label}
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    total += 1
                    continue
                buffer.append({"id": cid, "text": txt})
                if len(buffer) >= BATCH_SIZE:
                    futures.append(ex.submit(deepseek_batch, buffer[:]))
                    buffer.clear()

                # process finished futures as we go
                done = [f for f in futures if f.done()]
                for d in done:
                    futures.remove(d)
                    results = d.result()
                    for r in results:
                        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                        total += 1

            except Exception as e:
                logger.warning(f"Parse error: {e}")
                continue

        # flush tail
        if buffer:
            results = deepseek_batch(buffer)
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1

    fout.close()
    elapsed = time.time() - start
    logger.info(f"✅ Done {total:,} lines in {elapsed/60:.1f} min ({total/elapsed:.1f}/s)")


if __name__ == "__main__":
    main()
