import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.chunker import chunk_documents, Chunk
from src.rag.embedder import Embedder
from src.rag.vectorstore import VectorStore

SAMPLE_DOCS = [
    {"source": "billing_faq", "text": "How do I get a refund? Refunds are processed within 5-7 business days. Contact support with your order ID."},
    {"source": "technical_faq", "text": "Why can't I log in? Clear your browser cache, try incognito mode, or reset your password via the login page."},
    {"source": "general_faq", "text": "Do you offer a student discount? Yes, we offer 50% off for verified students. Email support with your student ID."},
]

class TestChunker(unittest.TestCase):

    def test_returns_chunks(self):
        chunks, stats = chunk_documents(SAMPLE_DOCS)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

    def test_chunk_has_required_fields(self):
        chunks, _ = chunk_documents(SAMPLE_DOCS)
        chunk = chunks[0]
        self.assertIsInstance(chunk, Chunk)
        self.assertTrue(chunk.text)
        self.assertTrue(chunk.source)

    def test_stats_returned(self):
        _, stats = chunk_documents(SAMPLE_DOCS)
        self.assertIn("total_chunks", stats)
        self.assertIn("chunking_time_ms", stats)
        self.assertIn("chunk_size", stats)

    def test_empty_docs_handled(self):
        chunks, stats = chunk_documents([])
        self.assertEqual(len(chunks), 0)
        self.assertEqual(stats["total_chunks"], 0)

    def test_empty_text_skipped(self):
        docs = [{"source": "empty", "text": "   "}]
        chunks, _ = chunk_documents(docs)
        self.assertEqual(len(chunks), 0)

    def test_smaller_chunk_size_produces_more_chunks(self):
        _, stats_500 = chunk_documents(SAMPLE_DOCS, chunk_size=500)
        _, stats_100 = chunk_documents(SAMPLE_DOCS, chunk_size=100)
        self.assertGreaterEqual(stats_100["total_chunks"], stats_500["total_chunks"])

class TestEmbedder(unittest.TestCase):

    def setUp(self):
        self.embedder = Embedder()

    def test_embed_returns_list(self):
        embeddings, stats = self.embedder.embed(["hello world"])
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 1)

    def test_embedding_dimension(self):
        embeddings, stats = self.embedder.embed(["test text"])
        self.assertEqual(stats["embedding_dim"], 384)

    def test_embed_one_returns_list(self):
        result = self.embedder.embed_one("hello")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 384)

    def test_stats_returned(self):
        _, stats = self.embedder.embed(["a", "b"])
        self.assertIn("embedding_time_ms", stats)
        self.assertIn("num_texts", stats)
        self.assertEqual(stats["num_texts"], 2)

class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.vs = VectorStore(persist_dir="data/chroma_test")
        chunks, _ = chunk_documents(SAMPLE_DOCS)
        self.vs.add_documents(chunks)

    def test_chunks_stored(self):
        stats = self.vs.get_stats()
        self.assertGreater(stats["total_chunks"], 0)

    def test_query_returns_results(self):
        hits, stats = self.vs.query("how do I get a refund", n=2)
        self.assertIsInstance(hits, list)
        self.assertGreater(len(hits), 0)

    def test_query_result_has_fields(self):
        hits, _ = self.vs.query("login problem", n=1)
        self.assertIn("text", hits[0])
        self.assertIn("source", hits[0])
        self.assertIn("distance", hits[0])

    def test_query_stats_returned(self):
        _, stats = self.vs.query("refund", n=2)
        self.assertIn("query_time_ms", stats)
        self.assertIn("num_results", stats)

    def test_refund_query_hits_billing(self):
        hits, _ = self.vs.query("I need a refund", n=1)
        self.assertEqual(hits[0]["source"], "billing_faq")

    def test_login_query_hits_technical(self):
        hits, _ = self.vs.query("I cannot log in", n=1)
        self.assertEqual(hits[0]["source"], "technical_faq")

    def test_student_discount_hits_general(self):
        hits, _ = self.vs.query("student discount", n=1)
        self.assertEqual(hits[0]["source"], "general_faq")

if __name__ == "__main__":
    unittest.main()
