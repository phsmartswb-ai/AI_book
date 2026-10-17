# Project description

Document Q&A is a lightweight local-document RAG application. It lets a user load PDF and TXT files, retrieve passages relevant to a question, and generate a grounded answer through OpenRouter.

## Goals

- Keep document parsing and retrieval local.
- Send only selected context to the generation provider.
- Make sources visible by instructing the model to cite file names.
- Remain easy to understand and run without a vector database or local ML runtime.
- Keep the retrieval layer replaceable if embeddings are needed later.

## Components

- `rag_app/config.py`: validates environment configuration.
- `rag_app/engine.py`: loads documents, chunks text, ranks passages, and calls OpenRouter.
- `rag_app/ui.py`: builds the Gradio interface and upload workflow.
- `app.py`: starts the web server.
- `tests/`: verifies core behavior without network access.

## Current retrieval approach

The default hybrid engine combines OpenRouter embedding cosine similarity with normalized lexical relevance. Semantic ranking finds related ideas with different wording, while the lexical contribution preserves exact terms and file-name matches. Vectors are cached locally, and keyword search remains available as an explicit mode and automatic fallback.

## Security and privacy

Credentials are read from `.env`, which is ignored by Git. Documents are parsed locally. Retrieved passages and the user's question are sent to OpenRouter over HTTPS, so the application should not be used with content that must never leave the machine.
