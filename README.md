# Document Intelligence & RAG Studio

A local-first Retrieval-Augmented Generation (RAG) application that searches PDF/TXT files with hybrid semantic retrieval and uses OpenRouter to produce grounded, cited answers.

## Features

- Hybrid retrieval combining OpenRouter embeddings with deterministic keyword ranking
- Three selectable search strategies: hybrid, semantic, and keyword-only
- Conversational document Q&A with an immediate processing indicator
- Retrieved source cards with file names, chunk numbers, text previews, and relevance scores
- Retrieval Explorer for inspecting search quality without generating an answer
- Knowledge Base manager for uploading, rescanning, and inspecting extracted document text
- Runtime controls for top-k, semantic weight, relevance threshold, and generation model
- Persistent embedding cache with automatic keyword fallback when semantic search is unavailable
- PDF and UTF-8 text support with duplicate detection and readable extraction errors
- Offline unit tests for document loading, retrieval, API failures, semantic search, and UI helpers

## Quick start

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add your OpenRouter key to `.env`, then run:

```bash
python app.py
```

Open <http://127.0.0.1:7860>. The original command `python rag_web_ui.py` remains available as a compatibility launcher.

## Using the interface

The web application is organized into four tabs:

1. **Q&A Assistant** — ask follow-up questions, review cited source chunks, tune search mode/top-k, and clear the conversation.
2. **Retrieval Explorer** — run retrieval directly and inspect ranked passages and scores without calling the generation model.
3. **Knowledge Base** — upload PDF/TXT files, rescan `data/documents/`, and inspect document metadata and extracted content.
4. **Settings & Diagnostics** — adjust retrieval parameters and the OpenRouter generation model for the current session.

Uploaded documents and settings changes remain in memory for the current process. Put permanent knowledge files in `data/documents/` and use `.env` for persistent configuration.

## Documents

Put `.pdf` and `.txt` knowledge files in `data/documents/`. Files uploaded in the UI are added to the in-memory index for the current app session.

The app extracts text locally. In semantic/hybrid mode, chunks and questions are sent to OpenRouter for embedding; generation sends only the retrieved chunks plus the question. Use keyword mode if document text must not be sent for embedding. Scanned PDFs need OCR before this app can read them.

## Configuration

Copy `.env.example` to `.env`. Available settings are:

| Variable | Default | Purpose |
|---|---:|---|
| `OPENROUTER_API_KEY` | required | OpenRouter credential |
| `OPENROUTER_MODEL` | `openrouter/auto` | Generation model |
| `OPENROUTER_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Semantic embedding model |
| `OPENROUTER_TIMEOUT` | `60` | Request timeout in seconds |
| `RAG_DOCUMENTS_DIR` | `data/documents` | Local knowledge directory |
| `RAG_CHUNK_SIZE` | `800` | Maximum chunk size in characters |
| `RAG_CHUNK_OVERLAP` | `120` | Context shared by adjacent chunks |
| `RAG_TOP_K` | `4` | Chunks supplied to the model |
| `RAG_SEARCH_MODE` | `hybrid` | `hybrid`, `semantic`, or `keyword` retrieval |
| `RAG_SEMANTIC_WEIGHT` | `0.7` | Semantic contribution to hybrid ranking |
| `RAG_SEMANTIC_THRESHOLD` | `0.2` | Minimum normalized retrieval score |
| `RAG_EMBEDDING_BATCH_SIZE` | `32` | Texts per embedding request |
| `RAG_EMBEDDING_CACHE` | `.cache/embeddings.json` | Persistent vector cache |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | UI bind address |
| `GRADIO_SERVER_PORT` | automatic | Require a specific UI port |

## Project layout

```text
.
├── app.py                    # Main web entry point
├── rag_app/
│   ├── config.py             # Environment-based settings
│   ├── engine.py             # Loading, chunking, hybrid search, generation
│   └── ui.py                 # Gradio studio, source cards, diagnostics
├── data/documents/           # Local PDF/TXT knowledge base
├── scripts/openrouter_chat.py # Standalone API test/chat
├── tests/                    # Offline unit tests
├── docs/                     # Additional project notes
├── simple_rag.py             # Compatibility CLI/API
└── rag_web_ui.py             # Compatibility web launcher
```

## Commands

Run the web app:

```bash
python app.py
```

Run a terminal document chat:

```bash
python simple_rag.py
```

Test OpenRouter without document retrieval:

```bash
python scripts/openrouter_chat.py
```

Run the offline test suite:

```bash
python -m unittest discover -v
```

## How answers are produced

1. PDF/TXT files are loaded and normalized.
2. Text is split into overlapping chunks.
3. OpenRouter embeddings measure semantic similarity between the question and chunks.
4. Hybrid ranking combines semantic similarity with deterministic keyword relevance.
5. The highest-ranked context is sent to OpenRouter.
6. The model is instructed to answer only from context and cite file names.

Document embeddings are created lazily on the first question and cached in `.cache/embeddings.json`. Later searches reuse the cached vectors and embed only the new query. If embeddings are unavailable, hybrid mode automatically falls back to keyword search.

## Troubleshooting

- **No documents loaded:** add PDF/TXT files to `data/documents/` and restart, or upload them in the UI.
- **No extractable text:** the PDF may be scanned; run OCR first.
- **Missing API key:** copy `.env.example` to `.env`, add the key, and restart.
- **Port already in use:** with no `GRADIO_SERVER_PORT` setting, Gradio automatically tries the next available port. If you configured one, change or remove it.
- **No relevant information:** ask with more specific terms that appear in the documents.
- **First question is slower:** the first semantic search indexes uncached chunks; later questions reuse the cache.

More background is available in [docs/PROJECT_DESCRIPTION.md](docs/PROJECT_DESCRIPTION.md) and [docs/RAG_GUIDE.md](docs/RAG_GUIDE.md).
