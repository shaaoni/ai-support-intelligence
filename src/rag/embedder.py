import logging
import time
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model ready")

    def embed(self, texts: list[str]) -> tuple[list, dict]:
        """Embed a list of texts. Returns embeddings + stats."""
        start = time.time()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        elapsed = round((time.time() - start) * 1000, 2)

        stats = {
            "num_texts": len(texts),
            "embedding_dim": embeddings.shape[1],
            "embedding_time_ms": elapsed,
            "avg_time_per_text_ms": round(elapsed / max(len(texts), 1), 2),
        }

        logger.info(f"Embedded {len(texts)} texts in {elapsed}ms")
        return embeddings.tolist(), stats

    def embed_one(self, text: str) -> list:
        """Embed a single text."""
        return self.model.encode([text])[0].tolist()
