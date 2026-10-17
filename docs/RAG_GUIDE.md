# RAG implementation guide

## Request flow

```text
PDF/TXT files -> text extraction -> overlapping chunks -> cached embeddings
                                                       |
question -> query embedding + keywords -> hybrid ranking
                                                       |
                                          top matching chunks
                                                       |
                                      OpenRouter generation -> answer
```

## Use from Python

```python
from rag_app import RAGEngine

rag = RAGEngine()
print(rag.answer_question("What is the MVP?"))
```

Load a specific collection:

```python
rag = RAGEngine(["report.pdf", "notes.txt"])
```

Add a document to a running engine:

```python
rag.add_document("new-report.pdf")
```

Inspect retrieval (semantic and hybrid modes call the embeddings API):

```python
for result in rag.search("revenue projection", top_k=5):
    print(result.score, result.chunk.source.name, result.chunk.content[:100])
```

## Extension points

- Replace the in-memory cosine scan with FAISS or a vector store for large collections.
- Add another parser in `_read_document()` and include its suffix in `SUPPORTED_EXTENSIONS`.
- Pass a configured `requests.Session` to reuse connections or customize retries.
- Pass a `Settings` instance for explicit configuration in integrations and tests.

## Design constraints

- Extracted document text and the embedding cache remain local; embedding inputs and queries are sent to OpenRouter in semantic/hybrid mode.
- Network calls have a timeout and validated response handling.
- The API key is required only for generation, so loading and search work offline.
- Uploads live in memory until the process exits; the application does not silently copy user files.

## Search modes

- `hybrid` (default): combines semantic and keyword relevance.
- `semantic`: ranks only by embedding similarity.
- `keyword`: stays fully local and makes no embedding request.

Set the mode through `RAG_SEARCH_MODE` in `.env`. Hybrid and semantic modes cache document vectors in `.cache/embeddings.json`; query vectors are generated per search.
