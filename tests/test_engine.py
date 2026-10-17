from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

from rag_app.config import Settings
from rag_app.engine import RAGEngine


def settings(
    directory: Path,
    api_key: str | None = "test-key",
    search_mode: str = "keyword",
) -> Settings:
    return Settings(
        api_key=api_key,
        model="test/model",
        documents_dir=directory,
        chunk_size=80,
        chunk_overlap=15,
        top_k=3,
        request_timeout=10,
        search_mode=search_mode,
        embedding_cache=directory / "embedding-cache.json",
    )


class RAGEngineTests(unittest.TestCase):
    @staticmethod
    def embedding_response(vectors: list[list[float]]) -> Mock:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [
                {"index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ]
        }
        return response

    def test_discovers_loads_and_searches_text_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "market.txt").write_text(
                "Pune was selected for its technology talent and growing market. " * 4,
                encoding="utf-8",
            )
            (directory / "ignored.md").write_text("Pune", encoding="utf-8")

            engine = RAGEngine(settings=settings(directory))

            self.assertEqual(len(engine.documents), 1)
            self.assertGreater(len(engine.chunks), 1)
            results = engine.search("Why was Pune selected?")
            self.assertTrue(results)
            self.assertEqual(results[0].chunk.source.name, "market.txt")

    def test_duplicate_document_is_not_added_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("A useful product roadmap.", encoding="utf-8")
            engine = RAGEngine([path], settings=settings(Path(temp_dir)))

            self.assertFalse(engine.add_document(path))
            self.assertEqual(len(engine.documents), 1)

    def test_remove_document_rebuilds_index_without_deleting_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("A useful product roadmap.", encoding="utf-8")
            engine = RAGEngine([path], settings=settings(Path(temp_dir)))

            self.assertTrue(engine.remove_document("notes.txt"))
            self.assertEqual(engine.documents, [])
            self.assertEqual(engine.chunks, [])
            self.assertTrue(path.exists())

    def test_search_includes_document_file_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "MVP-notes.txt"
            path.write_text("Launch scope and schedule.", encoding="utf-8")
            engine = RAGEngine([path], settings=settings(Path(temp_dir)))

            results = engine.search("MVP")

            self.assertTrue(results)
            self.assertEqual(results[0].chunk.source.name, "MVP-notes.txt")

    def test_missing_key_is_reported_only_when_generating(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("Revenue is projected to grow next year.", encoding="utf-8")
            engine = RAGEngine([path], settings=settings(Path(temp_dir), api_key=None))

            answer = engine.answer_question("What is projected about revenue?")

            self.assertIn("OPENROUTER_API_KEY", answer)

    def test_answer_uses_retrieved_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("The MVP launch is planned for June.", encoding="utf-8")
            session = Mock()
            response = Mock()
            response.json.return_value = {
                "choices": [{"message": {"content": "The MVP launches in June [notes.txt]."}}]
            }
            response.raise_for_status.return_value = None
            session.post.return_value = response
            engine = RAGEngine([path], settings=settings(Path(temp_dir)), session=session)

            answer = engine.answer_question("When is the MVP launch?")

            self.assertEqual(answer, "The MVP launches in June [notes.txt].")
            request_payload = session.post.call_args.kwargs["json"]
            self.assertIn("notes.txt", request_payload["messages"][1]["content"])

    def test_answer_reuses_results_and_includes_recent_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("Revenue reaches break-even in year two.", encoding="utf-8")
            session = Mock()
            response = Mock()
            response.json.return_value = {
                "choices": [{"message": {"content": "Break-even is in year two."}}]
            }
            response.raise_for_status.return_value = None
            session.post.return_value = response
            engine = RAGEngine([path], settings=settings(Path(temp_dir)), session=session)
            results = engine.search("break-even")
            engine.search = Mock(side_effect=AssertionError("search should not run twice"))
            history = [
                {"role": "user", "content": "Tell me about revenue."},
                {"role": "assistant", "content": "Revenue grows over time."},
            ]

            answer = engine.answer_question("When does it break even?", results=results, history=history)

            self.assertEqual(answer, "Break-even is in year two.")
            messages = session.post.call_args.kwargs["json"]["messages"]
            self.assertEqual(messages[1:3], history)

    def test_contextualize_query_adds_recent_user_questions(self) -> None:
        history = [
            {"role": "user", "content": "What are the projections?"},
            {"role": "assistant", "content": "They cover five years."},
        ]

        query = RAGEngine.contextualize_query("What about year two?", history)

        self.assertIn("What are the projections?", query)
        self.assertIn("What about year two?", query)

    def test_timeout_has_a_friendly_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("The project timeline starts in June.", encoding="utf-8")
            session = Mock()
            session.post.side_effect = requests.Timeout()
            engine = RAGEngine([path], settings=settings(Path(temp_dir)), session=session)

            self.assertIn("timed out", engine.answer_question("When does the timeline start?"))

    def test_semantic_search_finds_related_meaning_without_shared_words(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            first = directory / "transport.txt"
            second = directory / "cooking.txt"
            first.write_text("A vehicle carries passengers across the city.", encoding="utf-8")
            second.write_text("Bread is baked in a hot oven.", encoding="utf-8")
            session = Mock()
            session.post.side_effect = [
                self.embedding_response([[1.0, 0.0], [0.0, 1.0]]),
                self.embedding_response([[1.0, 0.0]]),
            ]
            engine = RAGEngine(
                [first, second],
                settings=settings(directory, search_mode="semantic"),
                session=session,
            )

            results = engine.search("automobile")

            self.assertEqual(results[0].chunk.source.name, "transport.txt")
            self.assertEqual(engine.last_search_mode, "semantic")

    def test_semantic_failure_falls_back_to_keyword_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = directory / "timeline.txt"
            path.write_text("The project begins in June.", encoding="utf-8")
            session = Mock()
            session.post.side_effect = requests.Timeout()
            engine = RAGEngine(
                [path],
                settings=settings(directory, search_mode="hybrid"),
                session=session,
            )

            results = engine.search("When does the project begin?")

            self.assertTrue(results)
            self.assertEqual(engine.last_search_mode, "keyword")
            self.assertIsNotNone(engine.semantic_error)


if __name__ == "__main__":
    unittest.main()
