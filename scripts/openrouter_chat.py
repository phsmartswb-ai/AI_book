"""Minimal terminal chat client for testing OpenRouter independently of RAG."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def main() -> int:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is missing. Copy .env.example to .env and add your key.")
        return 1

    model = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
    messages: list[dict[str, str]] = []
    print("OpenRouter chat. Type 'exit' to stop.")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt.lower() in {"exit", "quit", "bye"}:
            return 0
        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages},
                timeout=60,
            )
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            print(f"Request failed: {exc}")
            messages.pop()
            continue

        print(f"Assistant: {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    sys.exit(main())
