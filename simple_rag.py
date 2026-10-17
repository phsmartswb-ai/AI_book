"""Compatibility entry point for the original SimpleRAG API."""

from rag_app.engine import RAGEngine, SimpleRAG

__all__ = ["RAGEngine", "SimpleRAG"]


def main() -> None:
    rag = SimpleRAG()
    print(f"Loaded {len(rag.documents)} documents ({len(rag.chunks)} chunks).")
    print("Ask questions about the documents. Type 'exit' to stop.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            print(f"\nAssistant: {rag.answer_question(question, verbose=True)}")


if __name__ == "__main__":
    main()
