import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any

import openai

DEFAULT_INPUT_PATH = "kenya_clean_utf8.jsonl"
DEFAULT_OUTPUT_DIR = Path("prompt_tests")
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_BATCH_SIZE = 20
DEFAULT_RETRIES = 3
DEFAULT_SLEEP = 0.5
DEFAULT_TEMPERATURE = 0.0
DEFAULT_API_KEY = "sk-33a98b12f11f4300857e5ca93bf90e24"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample 1,000 random Reddit comments and classify them using DeepSeek."
    )
    parser.add_argument("--input", dest="input_path", default=DEFAULT_INPUT_PATH,
                        help="Path to the source JSONL file with Reddit comments (default: %(default)s)")
    parser.add_argument("--output-dir", dest="output_dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory where results will be written (default: %(default)s)")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="Number of random comments to classify (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Number of comments to send per API call (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Chat model to use (default: %(default)s)")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help="Number of retry attempts per batch (default: %(default)s)")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                        help="Seconds to wait between successful batches (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="Sampling temperature for the model (default: %(default)s)")
    parser.add_argument("--api-key", dest="api_key", default=None,
                        help="Override API key for DeepSeek (default: env/constant)")
    return parser.parse_args()


def configure_api(args: argparse.Namespace) -> None:
    api_key = (
        args.api_key
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or DEFAULT_API_KEY
    )
    openai.api_key = api_key
    openai.api_base = "https://api.deepseek.com"


def read_comments(path: str) -> List[Dict[str, Any]]:
    comments: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            # The source files can expose the comment text via either a "text" or
            # "body" field. Accept both so we can run against the cleaner output
            # without rewriting the data on disk.
            if "id" not in obj:
                continue
            text_val = obj.get("text") or obj.get("body")
            if text_val is None:
                continue
            comments.append({"id": str(obj["id"]), "text": str(text_val)})
    return comments


def choose_sample(
    comments: List[Dict[str, Any]], size: int, rng: random.Random
) -> List[Dict[str, Any]]:
    if size >= len(comments):
        print(
            f"Requested sample of {size} comments but only {len(comments)} available; using all comments."
        )
        return list(comments) if not comments else rng.sample(comments, len(comments))
    return rng.sample(comments, size)


def chunked(data: List[Dict[str, Any]], chunk_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def classify_batch(batch: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    payload = json.dumps([
        {"id": item["id"], "text": item["text"]}
        for item in batch
    ], ensure_ascii=False)

    for attempt in range(1, args.retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                temperature=args.temperature,
                max_tokens=3000,
            )
            content = response["choices"][0]["message"]["content"].strip()
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                start, end = content.find("["), content.rfind("]")
                if start != -1 and end != -1:
                    parsed = json.loads(content[start : end + 1])
                else:
                    raise
            if not isinstance(parsed, list):
                raise ValueError("Model output was not a list of classifications")
            return parsed
        except Exception:
            if attempt == args.retries:
                raise
            time.sleep(1.5 * attempt)
    return []


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise SystemExit("Sample size must be a positive integer")

    rng = random.Random()
    if args.seed is not None:
        rng.seed(args.seed)

    # Avoid mutating global random state so repeated runs without --seed remain random.
    if args.seed is None:
        rng.seed()

    configure_api(args)

    comments = read_comments(args.input_path)
    if not comments:
        raise SystemExit(f"No comments found in {args.input_path}")

    sample = choose_sample(comments, args.sample_size, rng)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"sample_{len(sample)}_{timestamp}.jsonl"

    results: List[Dict[str, Any]] = []
    total = len(sample)
    for idx, batch in enumerate(chunked(sample, max(1, args.batch_size)), start=1):
        classified = classify_batch(batch, args)
        results.extend(classified)
        processed = min(len(results), total)
        print(f"Processed batch {idx}: {processed}/{total} comments classified")
        time.sleep(max(0.0, args.sleep))

    with open(output_path, "w", encoding="utf-8") as out_file:
        for item in results:
            out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    meta = {
        "input_path": args.input_path,
        "total_available": len(comments),
        "sampled": len(sample),
        "output_file": str(output_path),
        "model": args.model,
        "temperature": args.temperature,
        "generated_at": timestamp,
        "seed": args.seed,
    }
    meta_path = output_dir / f"sample_{len(sample)}_{timestamp}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} classifications to {output_path}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
