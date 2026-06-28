import sys
import time
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.chunker import chunk_documents
from src.rag.vectorstore import VectorStore

def load_markdown_docs() -> list[dict]:
    """Load real markdown docs from data/raw/docs/"""
    docs = []
    docs_path = Path("data/raw/docs")
    for md_file in docs_path.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        docs.append({"source": md_file.stem, "text": text})
        print(f"  Loaded doc: {md_file.name} ({len(text)} chars)")
    return docs

def load_kaggle_tweets(n: int = 200) -> list[dict]:
    """Load a sample of real Kaggle tweets as support examples."""
    csv_path = Path("data/raw/CSVs/twcs/twcs.csv")
    df = pd.read_csv(csv_path, usecols=["author_id", "inbound", "text"])
    # Keep only inbound customer messages (not company replies)
    inbound = df[df["inbound"] == True].dropna(subset=["text"])
    # Remove @mentions and short tweets
    inbound = inbound[inbound["text"].str.len() > 30]
    sample = inbound.head(n)
    docs = []
    for _, row in sample.iterrows():
        text = str(row["text"])
        docs.append({"source": f"kaggle_tweet_{len(docs)}", "text": text})
    print(f"  Loaded {len(docs)} Kaggle tweets")
    return docs

def compare_strategies(docs: list[dict]):
    """Compare two chunking strategies side by side."""
    print("\n── Chunking Strategy Comparison ──────────────────────────────")
    strategies = [
        {"chunk_size": 500, "overlap": 50},
        {"chunk_size": 200, "overlap": 30},
    ]
    for s in strategies:
        chunks, stats = chunk_documents(docs, chunk_size=s["chunk_size"], overlap=s["overlap"])
        print(f"\n  Strategy: size={s['chunk_size']}, overlap={s['overlap']}")
        print(f"    Chunks created:   {stats['total_chunks']}")
        print(f"    Chunking time:    {stats['chunking_time_ms']}ms")
        print(f"    Sample chunk:     {chunks[0].text[:100]}...")
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-strategies", action="store_true", help="Compare chunk sizes side by side")
    args = parser.parse_args()

    print("\n── Phase 3: Building Vector Store ───────────────────────────")

    # Load real data
    print("\n  Loading markdown docs...")
    docs = load_markdown_docs()

    print("\n  Loading Kaggle tweets...")
    tweet_docs = load_kaggle_tweets(n=200)
    all_docs = docs + tweet_docs

    print(f"\n  Total documents: {len(all_docs)} ({len(docs)} markdown + {len(tweet_docs)} tweets)")

    # Step 3 flag: compare strategies
    if args.compare_strategies:
        compare_strategies(all_docs)
        return

    # Default: build the vector store with size=500
    print(f"\n  → Chunking with size=500, overlap=50")
    chunks, chunk_stats = chunk_documents(all_docs, chunk_size=500, overlap=50)
    print(f"    Chunks created: {chunk_stats['total_chunks']}")

    vs = VectorStore()
    print(f"\n  → Embedding and storing chunks...")
    add_stats = vs.add_documents(chunks)
    store_stats = vs.get_stats()

    print(f"\n── Observability Stats ───────────────────────────────────────")
    print(f"  Docs ingested:      {len(all_docs)}")
    print(f"  Chunks stored:      {store_stats['total_chunks']}")
    print(f"  Embedding time:     {add_stats['embedding_time_ms']}ms")
    print(f"  Upsert time:        {add_stats['upsert_time_ms']}ms")
    print(f"  Embedding dim:      {add_stats['embedding_dim']}")
    print(f"  ✓ Vector store built successfully!\n")

if __name__ == "__main__":
    main()
