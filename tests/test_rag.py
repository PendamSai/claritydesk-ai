import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    _format_context,
    find_passages,
    answer_question,
    build_vector_store,
)


class FakePDFPage:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


class FakePDFReader:
    def __init__(self, pages: list[FakePDFPage]) -> None:
        self.pages = pages


class FakeEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FormatContextTests(unittest.TestCase):
    def test_tuned_chunking_reduces_passage_count(self) -> None:
        pages = [
            Document(
                page_content=(
                    "Technical terms, requirements, dates, and references. " * 100
                )
            )
            for _ in range(10)
        ]
        previous_count = len(
            RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=150
            ).split_documents(pages)
        )
        tuned_count = len(
            RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            ).split_documents(pages)
        )

        self.assertLess(tuned_count, previous_count)

    def test_instant_index_skips_ollama_and_keeps_page_citations(self):
        with (
            patch("rag.PdfReader", return_value=FakePDFReader([FakePDFPage("Payment is due in thirty days."), FakePDFPage("Delivery is Friday.")])),
            patch("rag.OllamaEmbeddings") as embeddings,
            patch("rag.ChatOllama") as model,
        ):
            store = build_vector_store([("terms.pdf", b"pdf")], "unused", search_mode="keyword")
            answer, sources = find_passages(store, "payment")
            self.assertIn("thirty days", answer)
            self.assertEqual([(s.filename, s.page) for s in sources], [("terms.pdf", 1)])
            no_match, sources = find_passages(store, "warranty")
            self.assertEqual(sources, [])
            self.assertIn("No matching", no_match)
        embeddings.assert_not_called()
        model.assert_not_called()

    def test_streams_answer_tokens_and_preserves_sources(self) -> None:
        document = Document(page_content="Delivery is due Friday.", metadata={"source": "plan.pdf", "page": 0})
        from unittest.mock import Mock
        store = Mock()
        store.similarity_search.return_value = [document]
        tokens = []
        with patch("rag.ChatOllama") as model:
            model.return_value.stream.return_value = [
                SimpleNamespace(content="Friday"), SimpleNamespace(content=""),
                SimpleNamespace(content=". [S1]"),
            ]
            answer, sources = answer_question(store, "When?", "llama3.2", token_callback=tokens.append)
        self.assertEqual(tokens, ["Friday", ". [S1]"])
        self.assertEqual(answer, "Friday. [S1]")
        self.assertEqual(sources[0].page, 1)
        model.return_value.invoke.assert_not_called()

    def test_formats_source_labels_and_one_based_page_numbers(self) -> None:
        documents = [
            Document(
                page_content="The warranty lasts two years.",
                metadata={"source": "manual.pdf", "page": 2},
            ),
            Document(page_content="Keep the receipt.", metadata={"source": "guide.pdf"}),
        ]

        context, sources = _format_context(documents)

        self.assertIn("[S1] manual.pdf, page 3", context)
        self.assertIn("[S2] guide.pdf, page unknown", context)
        self.assertEqual(
            [
                (source.label, source.filename, source.page, source.excerpt)
                for source in sources
            ],
            [
                ("S1", "manual.pdf", 3, "The warranty lasts two years."),
                ("S2", "guide.pdf", None, "Keep the receipt."),
            ],
        )

    def test_builds_embeddings_in_bounded_batches(self) -> None:
        page = Document(page_content="one two three four five six seven eight")
        embeddings = FakeEmbeddings()
        with (
            patch(
                "rag.PdfReader",
                return_value=FakePDFReader(
                    [FakePDFPage(page.page_content)]
                ),
            ) as pdf_reader,
            patch("rag.OllamaEmbeddings", return_value=embeddings),
            patch("rag.EMBEDDING_BATCH_SIZE", 2),
        ):
            build_vector_store(
                [("terms.pdf", b"test PDF bytes")],
                "nomic-embed-text",
                chunk_size=5,
                chunk_overlap=0,
            )

        pdf_reader.assert_called_once()
        self.assertEqual(pdf_reader.call_args.args[0].read(), b"test PDF bytes")
        self.assertGreater(len(embeddings.batch_sizes), 1)
        self.assertLessEqual(max(embeddings.batch_sizes), 2)

    def test_rejects_scanned_pdf_without_calling_embedding_model(self) -> None:
        with (
            patch(
                "rag.PdfReader",
                return_value=FakePDFReader([FakePDFPage("")]),
            ),
            patch("rag.OllamaEmbeddings") as ollama_embeddings,
        ):
            with self.assertRaisesRegex(ValueError, "Scanned PDFs need OCR"):
                build_vector_store([("scan.pdf", b"pdf bytes")], "nomic-embed-text")

        ollama_embeddings.assert_not_called()

    def test_builds_index_and_returns_answer_sources(self) -> None:
        page = Document(
            page_content="The contract lasts 24 months.",
            metadata={"page": 0},
        )
        progress_events: list[tuple[float, str]] = []
        with (
            patch(
                "rag.PdfReader",
                return_value=FakePDFReader(
                    [FakePDFPage(page.page_content)]
                ),
            ),
            patch("rag.OllamaEmbeddings", return_value=FakeEmbeddings()),
            patch("rag.ChatOllama") as chat_model,
        ):
            chat_model.return_value.invoke.return_value = SimpleNamespace(
                content="The contract lasts 24 months. [S1]"
            )

            vector_store = build_vector_store(
                [("terms.pdf", b"test PDF bytes")],
                "nomic-embed-text",
                chunk_size=100,
                chunk_overlap=10,
                progress_callback=lambda progress, message: progress_events.append(
                    (progress, message)
                ),
            )
            answer, sources = answer_question(
                vector_store, "How long is the contract?", "llama3.2"
            )

        self.assertIn("24 months", answer)
        self.assertEqual(
            [(source.label, source.filename, source.page) for source in sources],
            [("S1", "terms.pdf", 1)],
        )
        progress_values = [progress for progress, _ in progress_events]
        self.assertEqual(progress_values[-1], 1.0)
        self.assertEqual(progress_values, sorted(progress_values))


if __name__ == "__main__":
    unittest.main()
