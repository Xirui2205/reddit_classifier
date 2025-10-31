#!/usr/bin/env python3
"""English vs. non-English detector for Reddit/TikTok style corpora.

This script analyses an input JSON Lines (or plain text) file and classifies
sentences based on their ratio of English tokens.  The script mirrors the
behaviour of the original `detect_en_nonen.py`, but is adapted to run directly
from this repository.

Usage
-----
python detect_en_nonen.py --input data.jsonl --output phase1.jsonl

The input can either contain raw strings (one sentence per line) or JSON
objects with a textual field such as ``body`` or ``text``.  The output is
another JSON Lines file where every sentence receives one of the following
labels:

``short`` (<3 tokens), ``English`` (>=95% English), ``code_mix_light``
(80–95% English), ``light_swahili`` (50–80% English), or ``heavy_swahili``
(<50% English).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

try:  # pragma: no cover - external dependency check
    from wordfreq import zipf_frequency
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'wordfreq' package is required. Install it with 'pip install wordfreq'."
    ) from exc

# ---------------- CONFIG ----------------
TOKEN_RE = re.compile(r"[A-Za-z']+")
MIN_TOKENS = 3
ENGLISH_THRESHOLD = 3.0  # zipf_frequency >= 3 ≈ common English word
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


# ---------------- DATA STRUCTURES ----------------
@dataclass(frozen=True)
class SentenceResult:
    """Classification result for a single sentence."""

    line_id: int
    sent_id: int
    label: str
    total_tokens: int
    en_tokens: int
    en_ratio: float
    text: str

    def to_json(self) -> str:
        """Serialise the result to a JSON string."""
        payload = {
            "line_id": self.line_id,
            "sent_id": self.sent_id,
            "label": self.label,
            "total_tokens": self.total_tokens,
            "en_tokens": self.en_tokens,
            "en_ratio": round(self.en_ratio, 3),
            "text": self.text,
        }
        return json.dumps(payload, ensure_ascii=False)


# ---------------- HELPERS ----------------
def is_english_word(word: str) -> bool:
    """Return ``True`` if ``word`` resembles a common English word."""
    if not word or not word.isascii():
        return False
    freq = zipf_frequency(word.lower(), "en")
    return freq >= ENGLISH_THRESHOLD


def label_from_ratio(en_ratio: float, total_tokens: int) -> str:
    """Return the label that corresponds to an English token ratio."""
    if total_tokens < MIN_TOKENS:
        return "short"
    if en_ratio >= 0.95:
        return "English"
    if en_ratio >= 0.80:
        return "code_mix_light"
    if en_ratio >= 0.50:
        return "light_swahili"
    return "heavy_swahili"


def extract_text(line: str) -> Optional[str]:
    """Extract a text payload from a JSON object or plain string."""
    stripped = line.strip()
    if not stripped:
        return None

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

    if isinstance(obj, dict):
        for key in ("body", "text", "content", "comment", "message"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
    elif isinstance(obj, str) and obj.strip():
        return obj

    return None


def process_line(payload: Tuple[int, str]) -> List[SentenceResult]:
    """Process a single line of input and classify its sentences."""
    line_id, text = payload
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    results: List[SentenceResult] = []

    for sent_id, sentence in enumerate(sentences, 1):
        tokens = TOKEN_RE.findall(sentence)
        total = len(tokens)
        if total == 0:
            continue
        en_tokens = sum(1 for token in tokens if is_english_word(token))
        en_ratio = en_tokens / total
        label = label_from_ratio(en_ratio, total)
        results.append(
            SentenceResult(
                line_id=line_id,
                sent_id=sent_id,
                label=label,
                total_tokens=total,
                en_tokens=en_tokens,
                en_ratio=en_ratio,
                text=sentence,
            )
        )

    return results


def iter_jobs(path: Path) -> Iterator[Tuple[int, str]]:
    """Yield (line_id, text) pairs for the multiprocessing pool."""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_id, raw_line in enumerate(handle, 1):
            text = extract_text(raw_line)
            if text:
                yield line_id, text


# ---------------- MAIN ----------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the input JSONL/text file")
    parser.add_argument("--output", required=True, help="Path to write the JSONL results")
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count() or 1,
        help="Number of worker processes (defaults to available CPUs)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of lines distributed to each worker batch",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    jobs = iter_jobs(input_path)

    # Remove any pre-existing output to avoid mixing runs.
    if output_path.exists():
        output_path.unlink()

    with Pool(processes=args.workers) as pool:
        try:
            with output_path.open("a", encoding="utf-8") as handle:
                for batch in pool.imap_unordered(process_line, jobs, chunksize=args.chunk_size):
                    for result in batch:
                        handle.write(result.to_json() + "\n")
        except KeyboardInterrupt:  # pragma: no cover - interactive convenience
            pool.terminate()
            raise

    print(f"✅ Done. Output saved to {output_path}")


if __name__ == "__main__":
    main()
