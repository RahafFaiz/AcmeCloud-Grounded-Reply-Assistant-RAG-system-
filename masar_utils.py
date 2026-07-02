"""
Masar Training, shared helper utilities.

Importable from notebooks like:

    import sys, pathlib
    sys.path.append(str(pathlib.Path.cwd().parents[1] / "shared"))  # weekX/notebooks -> repo/shared
    from masar_utils import get_client, MODEL, pretty, count_tokens

Everything here is intentionally tiny and dependency-light so it works
in the very first session before students have learned the rest.
"""
from __future__ import annotations

import os
import json
import textwrap
from typing import Any

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except Exception:  # dotenv optional
    pass

# Default models (overridable via .env)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MODEL_LARGE = os.getenv("OPENAI_MODEL_LARGE", "gpt-4o")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")


def get_client():
    """Return a configured OpenAI client, with a friendly error if the key is missing."""
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            "1) Copy .env.example to .env\n"
            "2) Put your key in OPENAI_API_KEY=...\n"
            "3) Restart the notebook kernel."
        )
    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAI(base_url=base_url) if base_url else OpenAI()


def pretty(obj: Any, title: str | None = None, width: int = 100) -> None:
    """Pretty-print dicts / JSON-able objects / strings for notebook output."""
    if title:
        print("=" * width)
        print(title)
        print("=" * width)
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    elif isinstance(obj, str):
        print("\n".join(textwrap.fill(line, width) for line in obj.splitlines()))
    else:
        print(obj)


def count_tokens(text: str, model: str = MODEL) -> int:
    """Count tokens with tiktoken (falls back to a rough estimate)."""
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate_cost(prompt_tokens: int, completion_tokens: int,
                  in_per_1m: float = 0.15, out_per_1m: float = 0.60) -> float:
    """Rough USD cost estimate (defaults are gpt-4o-mini list prices)."""
    return (prompt_tokens / 1_000_000) * in_per_1m + (completion_tokens / 1_000_000) * out_per_1m


if __name__ == "__main__":
    print("Default model:", MODEL)
    print("Token count of 'hello world':", count_tokens("hello world"))
    try:
        get_client()
        print("OpenAI client: OK")
    except RuntimeError as e:
        print("OpenAI client: NOT configured ->", e)
