import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.vectorstore import VectorStore

QUESTIONS = [
    ("How do I get a refund?", "billing"),
    ("I can't log into my account", "technical"),
    ("How do I cancel my subscription?", "billing"),
    ("The app is running very slowly", "technical"),
    ("Do you have a discount for students?", "general"),
    ("What plans do you offer?", "product_wiki"),
    ("How do I export my data?", "technical"),
    ("Is there a mobile app?", "technical"),
    ("What integrations do you support?", "product_wiki"),
    ("My payment failed, what happens next?", "billing"),
]

def run_evaluation(vs: VectorStore, docs_only: bool = False):
    """Run all 10 questions and print top 3 chunks each."""
    print("\n── Retrieval Evaluation (10 queries) ────────────────────────")
    stats_list = []

    for question, expected_source in QUESTIONS:
        hits, stats = vs.query(question, n=3)

        # Filter to docs only if flag is set
        if docs_only:
            hits = [h for h in hits if not h["source"].startswith("kaggle")]

        top_source = hits[0]["source"] if hits else "none"
        correct = "✅" if expected_source in top_source else "⚠️"

        print(f"\n{correct} ❓ {question}")
        print(f"   Expected source: {expected_source} | Got: {top_source} | Time: {stats['query_time_ms']}ms")
        for i, hit in enumerate(hits[:3], 1):
            print(f"   [{i}] source={hit['source']}  distance={hit['distance']}")
            print(f"       {hit['text'][:120]}...")

        stats_list.append(stats["query_time_ms"])

    avg_latency = round(sum(stats_list) / len(stats_list), 2)
    print(f"\n── Summary ───────────────────────────────────────────────────")
    print(f"  Queries run:       {len(QUESTIONS)}")
    print(f"  Avg latency:       {avg_latency}ms")
    print(f"  ✓ Retrieval evaluation complete!\n")

def run_custom_query(vs: VectorStore, query: str):
    """Test a single custom query."""
    print(f"\n── Custom Query ──────────────────────────────────────────────")
    print(f"  Query: {query}")
    hits, stats = vs.query(query, n=3)
    print(f"  Retrieval time: {stats['query_time_ms']}ms\n")
    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] source={hit['source']}  distance={hit['distance']}")
        print(f"       {hit['text'][:200]}...")
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Test a custom query")
    parser.add_argument("--docs-only", action="store_true", help="Filter results to markdown docs only")
    args = parser.parse_args()

    vs = VectorStore()
    stats = vs.get_stats()
    print(f"\n  Vector store has {stats['total_chunks']} chunks")

    if args.query:
        run_custom_query(vs, args.query)
    else:
        run_evaluation(vs, docs_only=args.docs_only)

if __name__ == "__main__":
    main()
