import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import openai

DEFAULT_INPUT_PATH = "kenya_clean_utf8.jsonl"
DEFAULT_OUTPUT_DIR = "prompt_tests"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_BATCH_SIZE = 20
DEFAULT_WORKERS = 4
DEFAULT_RETRIES = 3
DEFAULT_SLEEP = 0.5
DEFAULT_TEMPERATURE = 0.0
DEFAULT_API_KEY = "sk-33a98b12f11f4300857e5ca93bf90e24"

DEFAULT_SYSTEM_PROMPT = (
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

LANGUAGE_LABELS = ["English", "Swahili", "Sheng", "English and Swahili"]


@dataclass
class RunConfig:
    input_path: str
    output_dir: Path
    sample_size: int
    batch_size: int
    workers: int
    seed: Optional[int]
    model: str
    retries: int
    sleep: float
    temperature: float
    api_key: Optional[str]
    system_prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively sample Reddit comments and classify them using DeepSeek."
    )
    parser.add_argument("--input", dest="input_path", default=DEFAULT_INPUT_PATH,
                        help="Path to the source JSONL file with Reddit comments (default: %(default)s)")
    parser.add_argument("--output-dir", dest="output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory where results will be written (default: %(default)s)")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="Number of random comments to classify (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Number of comments to send per API call (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of concurrent workers to use (default: %(default)s)")
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


def prompt_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a positive integer.")


def prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def prompt_optional_int(prompt: str, default: Optional[int]) -> Optional[int]:
    shown = "random" if default is None else str(default)
    while True:
        raw = input(f"{prompt} [{shown}]: ").strip()
        if not raw:
            return default
        if raw.lower() in {"none", "random"}:
            return None
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer, 'none', or leave blank.")


def prompt_string(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return default if not raw else raw


def prompt_api_key(default: Optional[str]) -> Optional[str]:
    hint = default if default else "env/constant"
    raw = input(f"API key override [{hint}]: ").strip()
    return raw or default


def prompt_system_prompt(default_prompt: str) -> str:
    print("Enter classification system prompt. Press Enter on an empty line to accept the default.")
    preview_lines = default_prompt.splitlines()
    preview = "\n".join(preview_lines[: min(5, len(preview_lines))])
    print("Default prompt preview:\n" + preview)
    lines: List[str] = []
    while True:
        line = input()
        if line == "":
            if not lines:
                return default_prompt
            break
        lines.append(line)
    return "\n".join(lines)


def prompt_for_config(args: argparse.Namespace) -> RunConfig:
    print("Configure the prompt test run. Press Enter to keep the default shown in brackets.")
    input_path = prompt_string("Input JSONL path", args.input_path)
    output_dir = Path(prompt_string("Output directory", args.output_dir))
    sample_size = prompt_int("Sample size", args.sample_size)
    batch_size = prompt_int("Batch size", args.batch_size)
    workers = prompt_int("Number of workers", args.workers)
    seed = prompt_optional_int("Random seed", args.seed)
    model = prompt_string("Model name", args.model)
    retries = prompt_int("Retry attempts", args.retries)
    sleep = prompt_float("Sleep between batches (seconds)", args.sleep)
    temperature = prompt_float("Sampling temperature", args.temperature)
    api_key = prompt_api_key(args.api_key)
    system_prompt = prompt_system_prompt(DEFAULT_SYSTEM_PROMPT)
    return RunConfig(
        input_path=input_path,
        output_dir=output_dir,
        sample_size=sample_size,
        batch_size=batch_size,
        workers=max(1, workers),
        seed=seed,
        model=model,
        retries=max(1, retries),
        sleep=max(0.0, sleep),
        temperature=temperature,
        api_key=api_key,
        system_prompt=system_prompt,
    )


def configure_api(config: RunConfig) -> None:
    api_key = (
        config.api_key
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
            if not isinstance(obj, dict) or "id" not in obj:
                continue
            text_val = obj.get("text") or obj.get("body")
            if text_val is None:
                continue
            comments.append({"id": str(obj["id"]), "text": str(text_val)})
    return comments


def choose_sample(comments: List[Dict[str, Any]], size: int, rng: random.Random) -> List[Dict[str, Any]]:
    if size >= len(comments):
        print(
            f"Requested sample of {size} comments but only {len(comments)} available; using all comments."
        )
        return list(comments) if not comments else rng.sample(comments, len(comments))
    return rng.sample(comments, size)


def chunked(data: List[Dict[str, Any]], chunk_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def classify_batch(batch: List[Dict[str, Any]], config: RunConfig) -> List[Dict[str, Any]]:
    payload = json.dumps(
        [{"id": item["id"], "text": item["text"]} for item in batch],
        ensure_ascii=False,
    )
    for attempt in range(1, config.retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": payload},
                ],
                temperature=config.temperature,
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
            time.sleep(max(0.0, config.sleep))
            return parsed
        except Exception:
            if attempt == config.retries:
                raise
            time.sleep(1.5 * attempt)
    return []


def retry_batch_individually(batch: List[Dict[str, Any]], config: RunConfig) -> List[Dict[str, Any]]:
    recovered: List[Dict[str, Any]] = []
    for item in batch:
        try:
            recovered.extend(classify_batch([item], config))
        except Exception as err:
            print(f"Failed to classify comment {item['id']}: {err}")
    return recovered


def classify_all(sample: List[Dict[str, Any]], config: RunConfig) -> List[Dict[str, Any]]:
    batches = [list(chunk) for chunk in chunked(sample, max(1, config.batch_size))]
    total = len(sample)
    processed = 0
    results: List[Dict[str, Any]] = []
    if not batches:
        return results
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        future_map = {executor.submit(classify_batch, batch, config): batch for batch in batches}
        for future in as_completed(future_map):
            batch = future_map[future]
            try:
                batch_result = future.result()
            except Exception as exc:
                print(f"Batch failed with {exc!r}. Retrying sequentially...")
                batch_result = retry_batch_individually(batch, config)
            results.extend(batch_result)
            processed += len(batch)
            print(f"Progress: {processed}/{total} comments processed")
    return results


def slugify_language(language: str) -> str:
    return language.lower().replace(" ", "_")


def write_outputs(
    sample: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    config: RunConfig,
    run_dir: Path,
) -> Dict[str, int]:
    run_dir.mkdir(parents=True, exist_ok=True)
    combined_path = run_dir / "classifications.jsonl"
    text_lookup = {item["id"]: item["text"] for item in sample}
    writers = {}
    for label in LANGUAGE_LABELS:
        file_handle = open(run_dir / f"{slugify_language(label)}.csv", "w", encoding="utf-8", newline="")
        writer = csv.writer(file_handle)
        writer.writerow(["id", "body"])
        writers[label] = (file_handle, writer)

    unknown_handle = open(run_dir / "unknown.csv", "w", encoding="utf-8", newline="")
    unknown_writer = csv.writer(unknown_handle)
    unknown_writer.writerow(["id", "body"])
    counts = {label: 0 for label in LANGUAGE_LABELS}
    counts["unknown"] = 0
    seen_ids = set()

    with open(combined_path, "w", encoding="utf-8") as combined:
        for record in results:
            rid = str(record.get("id", "")).strip()
            lang = str(record.get("language", "")).strip()
            if not rid or rid in seen_ids:
                continue
            seen_ids.add(rid)
            combined.write(json.dumps({"id": rid, "language": lang}, ensure_ascii=False) + "\n")
            payload = [rid, text_lookup.get(rid, "")]
            if lang in writers:
                _, writer = writers[lang]
                writer.writerow(payload)
                counts[lang] += 1
            else:
                unknown_writer.writerow(payload)
                counts["unknown"] += 1

    missing = [item for item in sample if item["id"] not in seen_ids]
    for item in missing:
        payload = [item["id"], item["text"]]
        unknown_writer.writerow(payload)
        counts["unknown"] += 1

    for file_handle, _ in writers.values():
        file_handle.close()
    unknown_handle.close()

    return counts


def main() -> None:
    args = parse_args()
    config = prompt_for_config(args)

    rng = random.Random()
    if config.seed is not None:
        rng.seed(config.seed)
    else:
        rng.seed()

    configure_api(config)

    comments = read_comments(config.input_path)
    if not comments:
        raise SystemExit(f"No comments found in {config.input_path}")

    sample = choose_sample(comments, config.sample_size, rng)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = config.output_dir / f"sample_{len(sample)}_{timestamp}"

    print(f"Classifying {len(sample)} comments using {config.workers} worker(s)...")
    results = classify_all(sample, config)

    counts = write_outputs(sample, results, config, run_dir)

    metadata = {
        "input_path": config.input_path,
        "total_available": len(comments),
        "sampled": len(sample),
        "output_directory": str(run_dir),
        "model": config.model,
        "temperature": config.temperature,
        "seed": config.seed,
        "workers": config.workers,
        "batch_size": config.batch_size,
        "retries": config.retries,
        "sleep_seconds": config.sleep,
        "language_counts": counts,
        "system_prompt": config.system_prompt,
        "generated_at": timestamp,
    }
    meta_path = run_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=2)

    print(f"Run complete. Outputs saved in {run_dir}")
    for language, count in counts.items():
        print(f"  {language}: {count}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
