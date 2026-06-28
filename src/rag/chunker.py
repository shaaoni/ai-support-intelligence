import logging
import time
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    total_chunks: int
    chunk_size: int

def chunk_documents(documents: list[dict], chunk_size: int = 500, overlap: int = 50) -> tuple[list[Chunk], dict]:
    """Split documents into chunks. Returns chunks + stats."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []
    start = time.time()

    for doc in documents:
        text = doc.get("text", "")
        source = doc.get("source", "unknown")
        if not text.strip():
            continue
        parts = splitter.split_text(text)
        for i, part in enumerate(parts):
            chunks.append(Chunk(
                text=part,
                source=source,
                chunk_index=i,
                total_chunks=len(parts),
                chunk_size=chunk_size,
            ))

    stats = {
        "total_chunks": len(chunks),
        "total_docs": len(documents),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunking_time_ms": round((time.time() - start) * 1000, 2),
    }

    logger.info(f"Chunked {len(documents)} docs → {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks, stats
