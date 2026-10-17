import unittest
from pathlib import Path
from unittest.mock import Mock

import gradio as gr

from rag_app.engine import Chunk, Document, SearchResult
from rag_app.ui import (
    LOADING_HTML,
    _format_search_explorer_results,
    _format_sources_html,
    _get_document_metadata,
    build_app,
    stream_answer,
)


class UIStreamTests(unittest.TestCase):
    def test_loading_state_is_emitted_before_answer(self) -> None:
        engine = Mock()
        engine.answer_question.return_value = "Final answer"
        engine.semantic_error = None

        updates = list(stream_answer(engine, "What is the MVP?"))

        self.assertEqual(updates[0], (LOADING_HTML, ""))
        self.assertEqual(updates[1], ("", "Final answer"))

    def test_blank_question_skips_engine(self) -> None:
        engine = Mock()

        updates = list(stream_answer(engine, "  "))

        self.assertEqual(updates, [("", "Please enter a question.")])
        engine.answer_question.assert_not_called()

    def test_format_sources_html_empty_and_populated(self) -> None:
        empty_html = _format_sources_html([])
        self.assertIn("No matching source passages", empty_html)

        chunk = Chunk(content="This is the MVP definition.", source=Path("MVP.pdf"), index=0)
        result = SearchResult(chunk=chunk, score=0.92)
        populated_html = _format_sources_html([result])
        self.assertIn("MVP.pdf", populated_html)
        self.assertIn("Chunk #1", populated_html)
        self.assertIn("92% Match", populated_html)
        self.assertIn("This is the MVP definition.", populated_html)

    def test_format_search_explorer_results(self) -> None:
        empty_query_html = _format_search_explorer_results([], "", "hybrid")
        self.assertIn("Enter a query above", empty_query_html)

        no_match_html = _format_search_explorer_results([], "unknown query", "hybrid")
        self.assertIn("No chunks found matching", no_match_html)

        chunk = Chunk(content="Financial projection details", source=Path("projections.pdf"), index=2)
        result = SearchResult(chunk=chunk, score=0.88)
        results_html = _format_search_explorer_results([result], "financial", "hybrid")
        self.assertIn("projections.pdf", results_html)
        self.assertIn("Chunk #3", results_html)
        self.assertIn("Score: 0.880", results_html)

    def test_get_document_metadata(self) -> None:
        engine = Mock()
        doc = Document(path=Path("/tmp/sample.txt"), content="Hello world, this is a sample test document.")
        chunk = Chunk(content="Hello world", source=Path("/tmp/sample.txt"), index=0)
        engine.documents = [doc]
        engine.chunks = [chunk]

        meta_html, content = _get_document_metadata("sample.txt", engine)
        self.assertIn("TXT", meta_html)
        self.assertIn("Chunks", meta_html)
        self.assertEqual(content, "Hello world, this is a sample test document.")

        missing_meta, _ = _get_document_metadata("nonexistent.pdf", engine)
        self.assertIn("not found", missing_meta)

    def test_build_app_returns_blocks(self) -> None:
        engine = Mock()
        engine.documents = []
        engine.chunks = []
        engine.api_key = "test-key"
        engine.model = "test-model"
        engine.settings = Mock()
        engine.settings.search_mode = "hybrid"
        engine.settings.top_k = 4
        engine.settings.semantic_weight = 0.7
        engine.settings.semantic_threshold = 0.2
        engine.settings.embedding_model = "test-embed"
        engine.settings.documents_dir = Path("/tmp/data/documents")
        engine.settings.embedding_cache = Path("/tmp/cache.json")

        app = build_app(engine)
        self.assertIsInstance(app, gr.Blocks)


if __name__ == "__main__":
    unittest.main()
