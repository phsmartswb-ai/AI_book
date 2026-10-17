"""Core document loading, retrieval, and OpenRouter answer generation."""

from __future__ import annotations

import json
import hashlib
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import requests
from PyPDF2 import PdfReader

from .config import Settings


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
}


@dataclass(frozen=True)
class Document:
    path: Path
    content: str

    @property
    def kind(self) -> str:
        return self.path.suffix.lower().removeprefix(".")


@dataclass(frozen=True)
class Chunk:
    content: str
    source: Path
    index: int


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class EmbeddingError(RuntimeError):
    """Raised when semantic indexing cannot be completed."""


class RAGEngine:
    """Small, dependency-light RAG engine suitable for local document sets."""

    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    embeddings_endpoint = "https://openrouter.ai/api/v1/embeddings"

    def __init__(
        self,
        doc_paths: Iterable[str | Path] | None = None,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.session = session or requests.Session()
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self._chunk_embeddings: list[list[float]] | None = None
        self._embedding_lock = threading.Lock()
        self.last_search_mode = "keyword"
        self.semantic_error: str | None = None

        paths = list(doc_paths) if doc_paths is not None else self.discover_documents()
        self.add_documents(paths)

    @property
    def api_key(self) -> str | None:
        return self.settings.api_key

    @property
    def model(self) -> str:
        return self.settings.model

    def discover_documents(self) -> list[Path]:
        directory = self.settings.documents_dir
        if not directory.exists():
            return []
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def add_documents(self, paths: Iterable[str | Path]) -> int:
        """Load supported documents, de-duplicate paths, and rebuild the index."""
        known_paths = {document.path.resolve() for document in self.documents}
        added = 0
        errors: list[str] = []

        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            if path in known_paths:
                continue
            try:
                document = self._read_document(path)
            except (OSError, ValueError) as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            self.documents.append(document)
            known_paths.add(path)
            added += 1

        if added:
            self._rebuild_chunks()
        if errors:
            raise ValueError("Could not load: " + "; ".join(errors))
        return added

    def add_document(self, path: str | Path) -> bool:
        return bool(self.add_documents([path]))

    def remove_document(self, name: str) -> bool:
        """Remove a document from the active in-memory index without deleting its file."""
        remaining = [document for document in self.documents if document.path.name != name]
        if len(remaining) == len(self.documents):
            return False
        self.documents = remaining
        self._rebuild_chunks()
        return True

    def reload_documents(self) -> int:
        """Reload the configured documents directory and rebuild the active index."""
        self.documents = []
        self.chunks = []
        self._chunk_embeddings = None
        return self.add_documents(self.discover_documents())

    def _read_document(self, path: Path) -> Document:
        if not path.is_file():
            raise ValueError("file does not exist")
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError("only PDF and TXT files are supported")

        if suffix == ".pdf":
            try:
                content = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
            except Exception as exc:
                raise ValueError(f"invalid or unreadable PDF ({exc})") from exc
        else:
            content = path.read_text(encoding="utf-8")

        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        if not content:
            raise ValueError("no extractable text found")
        return Document(path=path, content=content)

    def _rebuild_chunks(self) -> None:
        self.chunks = []
        self._chunk_embeddings = None
        for document in self.documents:
            for index, text in enumerate(self._split_text(document.content)):
                self.chunks.append(Chunk(text, document.path, index))

    def _split_text(self, text: str) -> list[str]:
        size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap
        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
                if boundary > start + size // 2:
                    end = boundary + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
        return chunks

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            token for token in TOKEN_PATTERN.findall(text.lower())
            if len(token) > 1 and token not in STOP_WORDS
        ]

    def _keyword_scores(self, query_tokens: list[str]) -> list[float]:
        """Return normalized lexical relevance for every chunk."""
        if not query_tokens:
            return [0.0] * len(self.chunks)

        query_counts = Counter(query_tokens)
        query_phrase = " ".join(query_tokens)
        scores: list[float] = []
        for chunk in self.chunks:
            content_tokens = self._tokens(chunk.content)
            counts = Counter(content_tokens)
            source_tokens = set(self._tokens(chunk.source.stem))
            content_overlap = sum(
                min(counts[token], count) for token, count in query_counts.items()
            )
            source_overlap = len(set(query_tokens) & source_tokens)
            if not content_overlap and not source_overlap:
                scores.append(0.0)
                continue
            matched_tokens = set(query_tokens) & (set(content_tokens) | source_tokens)
            coverage = len(matched_tokens) / len(set(query_tokens))
            phrase_bonus = 1.5 if query_phrase in " ".join(content_tokens) else 0.0
            score = content_overlap + (source_overlap * 3) + (coverage * 2) + phrase_bonus
            scores.append(score)

        maximum = max(scores, default=0.0)
        return [score / maximum for score in scores] if maximum else scores

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        dot_product = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot_product / (left_norm * right_norm)

    def _cache_key(self, text: str) -> str:
        value = f"{self.settings.embedding_model}\0{text}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def _load_embedding_cache(self) -> dict[str, list[float]]:
        path = self.settings.embedding_cache
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("model") != self.settings.embedding_model:
                return {}
            vectors = payload.get("vectors", {})
            if isinstance(vectors, dict):
                return {
                    key: [float(value) for value in vector]
                    for key, vector in vectors.items()
                    if isinstance(key, str)
                    and isinstance(vector, list)
                    and vector
                    and all(isinstance(value, (int, float)) for value in vector)
                }
        except (OSError, ValueError, AttributeError):
            pass
        return {}

    def _save_embedding_cache(self, vectors: dict[str, list[float]]) -> None:
        path = self.settings.embedding_cache
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps({"model": self.settings.embedding_model, "vectors": vectors}),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError:
            # A read-only cache must not prevent search from working.
            pass

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise EmbeddingError("OPENROUTER_API_KEY is not configured")
        try:
            response = self.session.post(
                self.embeddings_endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.settings.embedding_model, "input": texts},
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
            items = sorted(response.json()["data"], key=lambda item: item["index"])
            embeddings = [item["embedding"] for item in items]
            if (
                len(embeddings) != len(texts)
                or not all(isinstance(vector, list) and vector for vector in embeddings)
                or not all(
                    isinstance(value, (int, float))
                    for vector in embeddings
                    for value in vector
                )
            ):
                raise EmbeddingError("embedding response did not match the request")
            return [[float(value) for value in vector] for vector in embeddings]
        except requests.RequestException as exc:
            raise EmbeddingError(self._response_error(exc)) from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EmbeddingError("OpenRouter returned invalid embeddings") from exc

    def _ensure_chunk_embeddings(self) -> list[list[float]]:
        if self._chunk_embeddings is not None:
            return self._chunk_embeddings

        with self._embedding_lock:
            if self._chunk_embeddings is not None:
                return self._chunk_embeddings

            cache = self._load_embedding_cache()
            keys = [self._cache_key(chunk.content) for chunk in self.chunks]
            missing = [
                (index, self.chunks[index].content)
                for index, key in enumerate(keys)
                if key not in cache
            ]
            batch_size = self.settings.embedding_batch_size
            for start in range(0, len(missing), batch_size):
                batch = missing[start:start + batch_size]
                vectors = self._embed_texts([text for _, text in batch])
                for (index, _), vector in zip(batch, vectors):
                    cache[keys[index]] = vector

            self._save_embedding_cache(cache)
            self._chunk_embeddings = [cache[key] for key in keys]
            return self._chunk_embeddings

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Rank chunks with hybrid semantic and lexical relevance."""
        query = query.strip()
        query_tokens = self._tokens(query)
        if not query or not self.chunks:
            return []

        keyword_scores = self._keyword_scores(query_tokens)
        mode = self.settings.search_mode
        combined_scores = keyword_scores
        self.last_search_mode = "keyword"
        self.semantic_error = None

        if mode != "keyword":
            try:
                chunk_embeddings = self._ensure_chunk_embeddings()
                query_embedding = self._embed_texts([query])[0]
                semantic_scores = [
                    max(0.0, self._cosine_similarity(query_embedding, embedding))
                    for embedding in chunk_embeddings
                ]
                if mode == "semantic":
                    combined_scores = semantic_scores
                    self.last_search_mode = "semantic"
                else:
                    weight = self.settings.semantic_weight
                    combined_scores = [
                        (weight * semantic) + ((1 - weight) * keyword)
                        for semantic, keyword in zip(semantic_scores, keyword_scores)
                    ]
                    self.last_search_mode = "hybrid"
            except EmbeddingError as exc:
                self.semantic_error = str(exc)

        limit = top_k or self.settings.top_k
        results = [
            SearchResult(chunk, score)
            for chunk, score in zip(self.chunks, combined_scores)
            if score >= self.settings.semantic_threshold
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    @staticmethod
    def contextualize_query(
        question: str,
        history: Sequence[dict[str, str]] | None = None,
        previous_questions: int = 2,
    ) -> str:
        """Add recent user turns to short follow-up queries for better retrieval."""
        prior = [
            str(message.get("content", "")).strip()
            for message in (history or [])
            if message.get("role") == "user" and str(message.get("content", "")).strip()
        ][-previous_questions:]
        if not prior:
            return question.strip()
        context = "\n".join(f"Previous question: {item}" for item in prior)
        return f"{context}\nCurrent question: {question.strip()}"

    def answer_question(
        self,
        question: str,
        verbose: bool = False,
        results: Sequence[SearchResult] | None = None,
        history: Sequence[dict[str, str]] | None = None,
    ) -> str:
        question = question.strip()
        if not question:
            return "Please enter a question."
        if not self.chunks:
            return "No documents are loaded. Add PDF or TXT files and try again."

        if results is None:
            search_query = self.contextualize_query(question, history)
            results = self.search(search_query)
        if not results:
            return "I couldn't find relevant information in the loaded documents."
        if not self.api_key:
            return "OPENROUTER_API_KEY is not configured. Add it to .env and restart the app."

        context = "\n\n".join(
            f"[Source: {result.chunk.source.name}, chunk {result.chunk.index + 1}]\n"
            f"{result.chunk.content}"
            for result in results
        )
        if verbose:
            for result in results:
                print(f"{result.chunk.source.name}: {result.score:.2f}")

        try:
            recent_history = [
                {"role": role, "content": content}
                for message in (history or [])[-6:]
                if (role := message.get("role")) in {"user", "assistant"}
                and isinstance((content := message.get("content")), str)
                and content.strip()
            ]
            response = self.session.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer only from the supplied context. If the context is insufficient, "
                                "say so. Cite sources using their file names."
                            ),
                        },
                        *recent_history,
                        {
                            "role": "user",
                            "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
                        },
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.2,
                },
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"].strip()
        except requests.Timeout:
            return "OpenRouter timed out. Please try again."
        except requests.RequestException as exc:
            detail = self._response_error(exc)
            return f"OpenRouter request failed: {detail}"
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return "OpenRouter returned an unexpected response. Please try again."

    @staticmethod
    def _response_error(exc: requests.RequestException) -> str:
        response = exc.response
        if response is not None:
            try:
                message = response.json().get("error", {}).get("message")
                if message:
                    return str(message)
            except (ValueError, AttributeError):
                pass
            return f"HTTP {response.status_code}"
        return str(exc)

    def save_qa_pairs(self, qa_pairs: Sequence[dict[str, str]], filename: str | Path) -> None:
        Path(filename).write_text(json.dumps(list(qa_pairs), indent=2), encoding="utf-8")


# Backwards-compatible name used by the original examples.
SimpleRAG = RAGEngine
