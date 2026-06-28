import logging
import time
import chromadb
from pathlib import Path
from src.rag.embedder import Embedder
from src.rag.chunker import Chunk

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_dir: str = "data/chroma"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="support_docs",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = Embedder()
        logger.info(f"VectorStore ready — {self.collection.count()} docs in collection")

    def add_documents(self, chunks: list[Chunk]) -> dict:
        """Embed and store chunks in ChromaDB."""
        if not chunks:
            return {"added": 0}

        texts = [c.text for c in chunks]
        ids = [f"{c.source}__chunk{c.chunk_index}" for c in chunks]
        metadatas = [{"source": c.source, "chunk_index": c.chunk_index, "total_chunks": c.total_chunks} for c in chunks]

        embeddings, embed_stats = self.embedder.embed(texts)

        start = time.time()
        self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        upsert_time = round((time.time() - start) * 1000, 2)

        stats = {**embed_stats, "upsert_time_ms": upsert_time, "added": len(chunks)}
        logger.info(f"Added {len(chunks)} chunks to vector store")
        return stats

    def query(self, text: str, n: int = 5) -> tuple[list[dict], dict]:
        """Query vector store. Returns top n results + stats."""
        start = time.time()
        query_embedding = self.embedder.embed_one(text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, self.collection.count()),
        )
        elapsed = round((time.time() - start) * 1000, 2)

        hits = []
        for i, doc in enumerate(results["documents"][0]):
            hits.append({
                "text": doc,
                "source": results["metadatas"][0][i]["source"],
                "distance": round(results["distances"][0][i], 4),
            })

        stats = {"query_time_ms": elapsed, "num_results": len(hits)}
        return hits, stats

    def get_stats(self) -> dict:
        """Return collection stats."""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection.name,
        }
