#!/usr/bin/env python3
"""
tokencount_predictor.py

Predicts the approximate number of ChatGPT tokens used by the content
of a .json file or all .json files in a given directory (recursively).

Usage:
    python tokencount_predictor.py --input_path data/SomeDirectory
    python tokencount_predictor.py --input_path data/example.json
"""

import os
import json
import argparse
from pathlib import Path

# Optional dependency for token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None


def count_tokens_in_text(text: str, encoding_name: str = "cl100k_base") -> int:
    """Estimate number of tokens in the given text."""
    if not text:
        return 0

    if tiktoken:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    else:
        # Fallback: rough approximation (1 token ≈ 4 characters)
        return len(text) // 4


def count_tokens_in_json_file(file_path: Path) -> int:
    """Reads a JSON file and estimates total tokens from all string values."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read {file_path}: {e}")
        return 0

    def traverse(obj):
        total = 0
        if isinstance(obj, dict):
            for v in obj.values():
                total += traverse(v)
        elif isinstance(obj, list):
            for v in obj:
                total += traverse(v)
        elif isinstance(obj, str):
            total += count_tokens_in_text(obj)
        else:
            total += count_tokens_in_text(str(obj))
        return total

    return traverse(data)


def main():
    parser = argparse.ArgumentParser(
        description="Predict total ChatGPT token count for a JSON file or directory of JSON files."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to a .json file or a directory containing .json files",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: '{input_path}' does not exist.")
        return

    total_tokens = 0
    json_files = []

    if input_path.is_dir():
        json_files = list(input_path.rglob("*.json"))
    elif input_path.is_file() and input_path.suffix.lower() == ".json":
        json_files = [input_path]
    else:
        print(f"Error: '{input_path}' must be a directory or a .json file.")
        return

    if not json_files:
        print("No .json files found.")
        return

    for file in json_files:
        tokens = count_tokens_in_json_file(file)
        total_tokens += tokens

    print(total_tokens)


if __name__ == "__main__":
    main()
