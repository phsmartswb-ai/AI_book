"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _ratio(name: str, default: float) -> float:
    value = float(os.getenv(name, default))
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime settings for retrieval and answer generation."""

    api_key: str | None
    model: str
    documents_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    request_timeout: int
    search_mode: str = "hybrid"
    embedding_model: str = "openai/text-embedding-3-small"
    semantic_weight: float = 0.7
    semantic_threshold: float = 0.2
    embedding_batch_size: int = 32
    embedding_cache: Path = PROJECT_ROOT / ".cache" / "embeddings.json"

    @classmethod
    def from_env(cls) -> "Settings":
        documents_dir = Path(
            os.getenv("RAG_DOCUMENTS_DIR", PROJECT_ROOT / "data" / "documents")
        ).expanduser()
        if not documents_dir.is_absolute():
            documents_dir = PROJECT_ROOT / documents_dir

        chunk_size = _positive_int("RAG_CHUNK_SIZE", 800)
        chunk_overlap = _positive_int("RAG_CHUNK_OVERLAP", 120)
        if chunk_overlap >= chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")

        search_mode = os.getenv("RAG_SEARCH_MODE", "hybrid").lower()
        if search_mode not in {"hybrid", "semantic", "keyword"}:
            raise ValueError("RAG_SEARCH_MODE must be hybrid, semantic, or keyword")

        cache_path = Path(
            os.getenv("RAG_EMBEDDING_CACHE", PROJECT_ROOT / ".cache" / "embeddings.json")
        ).expanduser()
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path

        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
            documents_dir=documents_dir.resolve(),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=_positive_int("RAG_TOP_K", 4),
            request_timeout=_positive_int("OPENROUTER_TIMEOUT", 60),
            search_mode=search_mode,
            embedding_model=os.getenv(
                "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
            ),
            semantic_weight=_ratio("RAG_SEMANTIC_WEIGHT", 0.7),
            semantic_threshold=_ratio("RAG_SEMANTIC_THRESHOLD", 0.2),
            embedding_batch_size=_positive_int("RAG_EMBEDDING_BATCH_SIZE", 32),
            embedding_cache=cache_path.resolve(),
        )
